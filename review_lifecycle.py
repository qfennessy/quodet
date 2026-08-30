"""Deterministic lifecycle and timing metadata for review batches."""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from feedback import ReviewBatch, ReviewFinding


LIFECYCLE_STATUSES = frozenset(
    {"new", "retained", "replaced", "no_longer_reported", "stale"}
)
STALE_REASONS = frozenset({"source_changed", "out_of_order"})


@dataclass(frozen=True)
class FindingLifecycle:
    """One source-free relationship between findings in adjacent snapshots."""

    status: str
    fingerprint: str
    file: str
    line: int
    previous_fingerprint: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class BatchTiming:
    debounce_ms: float
    provider_ms: float
    publication_ms: float
    agent_delivery_ms: float | None
    total_ms: float


@dataclass(frozen=True)
class _TrackedFinding:
    fingerprint: str
    line: int


@dataclass(frozen=True)
class _FileState:
    snapshot_at: float
    findings: tuple[_TrackedFinding, ...]


def short_batch_id(batch_id: str) -> str:
    """Return a safe, compact display identifier derived from a UUID."""
    compact = batch_id.replace("-", "").lower()
    if len(compact) < 8 or any(character not in "0123456789abcdef" for character in compact):
        raise ValueError("batch id must be a UUID-like hexadecimal value")
    return f"qdt-{compact[:8]}"


def finding_fingerprint(finding: ReviewFinding) -> str:
    """Hash normalized provider metadata, never source text or explanation prose."""
    normalized_title = re.sub(r"[^a-z0-9]+", " ", finding.title.casefold()).strip()
    encoded = f"quodet-finding-v1\0{finding.file}\0{normalized_title}".encode()
    return hashlib.sha256(encoded).hexdigest()


def stale_lifecycle(finding: ReviewFinding, *, reason: str) -> FindingLifecycle:
    if reason not in STALE_REASONS:
        raise ValueError("invalid stale reason")
    return FindingLifecycle(
        status="stale",
        fingerprint=finding_fingerprint(finding),
        file=finding.file,
        line=finding.line,
        reason=reason,
    )


def batch_timing(
    batch: ReviewBatch,
    *,
    delivered_at: float | None = None,
    now: float | None = None,
) -> BatchTiming:
    """Calculate available non-negative timing segments for presentation."""
    current = time.time() if now is None else now
    publication_end = batch.published_at or current
    publication_ms = max(
        0.0, (publication_end - batch.provider_completed_at) * 1_000
    )
    delivery_ms = (
        max(0.0, (delivered_at - publication_end) * 1_000)
        if delivered_at is not None
        else None
    )
    total_end = delivered_at if delivered_at is not None else publication_end
    return BatchTiming(
        debounce_ms=max(0.0, batch.debounce_ms),
        provider_ms=max(0.0, batch.provider_ms),
        publication_ms=publication_ms,
        agent_delivery_ms=delivery_ms,
        total_ms=max(0.0, (total_end - batch.first_observed_at) * 1_000),
    )


class FindingLifecycleTracker:
    """Relate ordered snapshots without treating model omission as resolution."""

    def __init__(self) -> None:
        self._files: dict[str, _FileState] = {}

    def classify(self, batch: ReviewBatch) -> ReviewBatch:
        events = list(batch.lifecycle)
        current_by_file: dict[str, list[ReviewFinding]] = {}
        for finding in batch.findings:
            current_by_file.setdefault(finding.file, []).append(finding)
        snapshot_at = batch.batch_flushed_at or batch.provider_started_at or batch.created_at

        for reviewed in sorted(batch.reviewed_files, key=lambda item: item.path):
            path = reviewed.path
            if path in batch.stale_files:
                continue
            current = sorted(
                (
                    _TrackedFinding(finding_fingerprint(finding), finding.line)
                    for finding in current_by_file.get(path, ())
                ),
                key=lambda item: (item.fingerprint, item.line),
            )
            previous_state = self._files.get(path)
            if previous_state is not None and snapshot_at < previous_state.snapshot_at:
                events.extend(
                    FindingLifecycle(
                        status="stale",
                        fingerprint=item.fingerprint,
                        file=path,
                        line=item.line,
                        reason="out_of_order",
                    )
                    for item in current
                )
                continue

            previous = list(previous_state.findings) if previous_state is not None else []
            unmatched_previous = previous.copy()
            unmatched_current: list[_TrackedFinding] = []
            for current_item in current:
                previous_index = next(
                    (
                        index
                        for index, previous_item in enumerate(unmatched_previous)
                        if previous_item.fingerprint == current_item.fingerprint
                    ),
                    None,
                )
                if previous_index is None:
                    unmatched_current.append(current_item)
                    continue
                previous_item = unmatched_previous.pop(previous_index)
                events.append(
                    FindingLifecycle(
                        status="retained",
                        fingerprint=current_item.fingerprint,
                        file=path,
                        line=current_item.line,
                        previous_fingerprint=previous_item.fingerprint,
                    )
                )
            replacement_count = min(len(unmatched_previous), len(unmatched_current))
            for previous_item, current_item in zip(
                unmatched_previous[:replacement_count],
                unmatched_current[:replacement_count],
            ):
                events.append(
                    FindingLifecycle(
                        status="replaced",
                        fingerprint=current_item.fingerprint,
                        file=path,
                        line=current_item.line,
                        previous_fingerprint=previous_item.fingerprint,
                    )
                )
            events.extend(
                FindingLifecycle(
                    status="no_longer_reported",
                    fingerprint=item.fingerprint,
                    file=path,
                    line=item.line,
                    previous_fingerprint=item.fingerprint,
                )
                for item in unmatched_previous[replacement_count:]
            )
            events.extend(
                FindingLifecycle(
                    status="new",
                    fingerprint=item.fingerprint,
                    file=path,
                    line=item.line,
                )
                for item in unmatched_current[replacement_count:]
            )
            self._files[path] = _FileState(snapshot_at, tuple(current))
        return replace(batch, lifecycle=tuple(events))
