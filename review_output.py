"""Stable terminal and machine renderers for validated review batches."""

from __future__ import annotations

import json
import unicodedata
from typing import Protocol, Sequence


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


def render_human_review(batch: ReviewBatchLike) -> str:
    """Render a compact deterministic summary for a person watching a terminal."""
    reviewed = _count(len(batch.reviewed_files), "file")
    finding_count = len(batch.findings)
    if finding_count == 0:
        return f"Quodet reviewed {reviewed}: no confident findings"

    heading = (
        f"Quodet reviewed {reviewed}: "
        f"{_count(finding_count, 'likely defect')}"
    )
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
                f"{path}:{finding.line} [{severity}, {finding.confidence:.2f}] {title}",
                f"  {explanation}",
                f"  Suggested action: {suggested_fix}",
            ]
        )
    return "\n".join(lines)


def render_review(batch: ReviewBatchLike, mode: str) -> str:
    if mode == "human":
        return render_human_review(batch)
    if mode == "json":
        return render_json_review(batch)
    raise ValueError(f"unsupported output mode: {mode}")
