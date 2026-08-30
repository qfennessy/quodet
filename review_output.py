"""Stable terminal and machine renderers for validated review batches."""

from __future__ import annotations

import json
import unicodedata
from typing import Protocol, Sequence

from review_lifecycle import batch_timing, short_batch_id


OUTPUT_SCHEMA_VERSION = "quodet-review-output-v1"
OUTPUT_MODES = ("human", "json")
DEFAULT_OUTPUT_MODE = "human"
MAX_HUMAN_TITLE_LENGTH = 160
MAX_HUMAN_DETAIL_LENGTH = 320


class ReviewedFileLike(Protocol):
    path: str
    sha256: str
    size: int


class ReviewFindingLike(Protocol):
    file: str
    line: int
    severity: str
    confidence: float
    title: str
    explanation: str
    suggested_fix: str


class FindingLifecycleLike(Protocol):
    status: str
    fingerprint: str
    file: str
    line: int
    previous_fingerprint: str | None
    reason: str | None


class ReviewBatchLike(Protocol):
    batch_id: str
    root: str
    created_at: float
    reviewed_files: Sequence[ReviewedFileLike]
    findings: Sequence[ReviewFindingLike]
    session_id: str | None
    session_generation: int | None
    feedback_round: int
    debounce_ms: float
    provider_ms: float
    first_observed_at: float
    batch_flushed_at: float
    provider_started_at: float
    provider_completed_at: float
    published_at: float
    redactions: RedactionSummaryLike
    lifecycle: Sequence[FindingLifecycleLike]
    stale_files: Sequence[str]


class RedactionNoticeLike(Protocol):
    file: str | None
    line: int | None
    category: str
    masked_identifier: str | None
    disposition: str


class RedactionSummaryLike(Protocol):
    total: int
    notices: Sequence[RedactionNoticeLike]
    omitted: int


def _reviewed_file_document(reviewed: ReviewedFileLike) -> dict[str, object]:
    return {
        "path": reviewed.path,
        "sha256": reviewed.sha256,
        "size": reviewed.size,
    }


def _finding_document(finding: ReviewFindingLike) -> dict[str, object]:
    return {
        "file": finding.file,
        "line": finding.line,
        "severity": finding.severity,
        "confidence": finding.confidence,
        "title": finding.title,
        "explanation": finding.explanation,
        "suggested_fix": finding.suggested_fix,
    }


def _redaction_document(summary: RedactionSummaryLike) -> dict[str, object]:
    return {
        "total": summary.total,
        "notices": [
            {
                "file": notice.file,
                "line": notice.line,
                "category": notice.category,
                "masked_identifier": notice.masked_identifier,
                "disposition": notice.disposition,
            }
            for notice in summary.notices
        ],
        "omitted": summary.omitted,
    }


def _lifecycle_document(event: FindingLifecycleLike) -> dict[str, object]:
    return {
        "status": event.status,
        "fingerprint": event.fingerprint,
        "file": event.file,
        "line": event.line,
        "previous_fingerprint": event.previous_fingerprint,
        "reason": event.reason,
    }


def review_output_document(batch: ReviewBatchLike) -> dict[str, object]:
    """Return the complete, explicitly versioned public review document.

    Keep this serializer explicit: new internal dataclass fields must not leak into
    the public contract accidentally. Additive presentation metadata, including
    redaction and finding-lifecycle records, can be added here deliberately.
    """
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "batch_id": batch.batch_id,
        "root": batch.root,
        "created_at": batch.created_at,
        "reviewed_files": [
            _reviewed_file_document(reviewed) for reviewed in batch.reviewed_files
        ],
        "findings": [_finding_document(finding) for finding in batch.findings],
        "session_id": batch.session_id,
        "session_generation": batch.session_generation,
        "feedback_round": batch.feedback_round,
        "timing": {
            "first_observed_at": batch.first_observed_at,
            "batch_flushed_at": batch.batch_flushed_at,
            "debounce_ms": batch.debounce_ms,
            "provider_started_at": batch.provider_started_at,
            "provider_completed_at": batch.provider_completed_at,
            "provider_ms": batch.provider_ms,
            "published_at": batch.published_at,
        },
        "redactions": _redaction_document(batch.redactions),
        "lifecycle": [_lifecycle_document(event) for event in batch.lifecycle],
        "stale_files": list(batch.stale_files),
    }


def render_json_review(batch: ReviewBatchLike) -> str:
    """Render one deterministic JSON Lines document for machine consumers."""
    return json.dumps(
        review_output_document(batch),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _terminal_text(value: str, *, maximum: int) -> str:
    # Provider text is untrusted. Remove terminal controls and collapse whitespace
    # before placing it in the human interface; JSON mode retains the validated text.
    printable = "".join(
        character
        if character.isprintable() and unicodedata.category(character) != "Cf"
        else " "
        for character in value
    )
    compact = " ".join(printable.split())
    if len(compact) <= maximum:
        return compact
    return compact[: maximum - 1].rstrip() + "…"


def _count(value: int, singular: str, plural: str | None = None) -> str:
    return f"{value} {singular if value == 1 else plural or singular + 's'}"


def render_redaction_summary(summary: RedactionSummaryLike) -> str:
    """Render only bounded, value-free redaction hints."""
    if not summary.total:
        return ""
    noun = "potential secret" if summary.total == 1 else "potential secrets"
    lines = [f"Redacted {summary.total} {noun} before provider upload:"]
    for notice in summary.notices:
        if notice.file is None:
            location = "review prompt"
        else:
            location = _terminal_text(notice.file, maximum=1_024)
        if notice.line is not None:
            location += f":{notice.line}"
        category = notice.category.replace("-", " ")
        identifier = f" {notice.masked_identifier}" if notice.masked_identifier else ""
        disposition = "sent sanitized" if notice.disposition == "sent" else "excluded"
        lines.append(f"  {location} {category}{identifier} ({disposition})")
    if summary.omitted:
        lines.append(f"  … and {summary.omitted} more")
    return "\n".join(lines)


def render_human_review(batch: ReviewBatchLike) -> str:
    """Render a compact deterministic summary for a person watching a terminal."""
    reviewed = _count(len(batch.reviewed_files), "file")
    timing = batch_timing(batch)
    batch_label = short_batch_id(batch.batch_id)
    stages = (
        f"[debounce {timing.debounce_ms:.1f}ms, provider "
        f"{timing.provider_ms:.1f}ms, publication {timing.publication_ms:.1f}ms]"
    )
    lifecycle_counts: dict[str, int] = {}
    for event in batch.lifecycle:
        lifecycle_counts[event.status] = lifecycle_counts.get(event.status, 0) + 1
    finding_count = len(batch.findings)
    if finding_count == 0:
        if batch.stale_files or lifecycle_counts.get("stale"):
            lines = [
                f"{batch_label} discarded after {timing.total_ms / 1_000:.2f}s: "
                f"source changed during review {stages}"
            ]
        else:
            omitted = lifecycle_counts.get("no_longer_reported", 0)
            if omitted:
                result = (
                    f"{_count(omitted, 'prior finding')} no longer reported "
                    "in the latest snapshot"
                )
            else:
                result = "no confident findings"
            lines = [
                f"{batch_label} reviewed {reviewed} in "
                f"{timing.total_ms / 1_000:.2f}s: {result} {stages}"
            ]
    else:
        lifecycle_summary = ", ".join(
            f"{count} {status.replace('_', ' ')}"
            for status, count in sorted(lifecycle_counts.items())
            if status != "no_longer_reported"
        )
        heading = (
            f"{batch_label} reviewed {reviewed} in {timing.total_ms / 1_000:.2f}s: "
            f"{_count(finding_count, 'likely defect')}"
        )
        if lifecycle_summary:
            heading += f" ({lifecycle_summary})"
        heading += f" {stages}"
        lines = [heading]
        for finding in batch.findings:
            path = _terminal_text(finding.file, maximum=1_024)
            severity = _terminal_text(finding.severity, maximum=32)
            title = _terminal_text(finding.title, maximum=MAX_HUMAN_TITLE_LENGTH)
            explanation = _terminal_text(
                finding.explanation, maximum=MAX_HUMAN_DETAIL_LENGTH
            )
            suggested_fix = _terminal_text(
                finding.suggested_fix, maximum=MAX_HUMAN_DETAIL_LENGTH
            )
            lines.extend(
                [
                    f"{path}:{finding.line} "
                    f"[{severity}, {finding.confidence:.2f}] {title}",
                    f"  {explanation}",
                    f"  Suggested action: {suggested_fix}",
                ]
            )
    redaction_summary = render_redaction_summary(batch.redactions)
    if redaction_summary:
        lines.extend(["", redaction_summary])
    return "\n".join(lines)


def render_review(batch: ReviewBatchLike, mode: str) -> str:
    if mode == "human":
        return render_human_review(batch)
    if mode == "json":
        return render_json_review(batch)
    raise ValueError(f"unsupported output mode: {mode}")
