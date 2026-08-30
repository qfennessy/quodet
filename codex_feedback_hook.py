#!/usr/bin/env python3
"""Consume Quodet feedback from Codex or Claude Code agent hooks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
import time
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Sequence

from feedback import (
    MAX_SPOOL_PAYLOAD_BYTES,
    UNTRUSTED_NOTICE,
    ReviewValidationError,
    ReviewedFile,
    _state_generation,
    fresh_spooled_payload,
    matching_review_in_flight,
    publish_flush_hint,
    request_flush_hint,
    read_bounded_beneath_root,
    read_session_state,
    retire_reviewed_flush_hints,
    session_route_lock,
    validate_spooled_payload,
    write_session_state,
)
from redaction import redact_path, redact_text
from review_lifecycle import short_batch_id


MAX_DELIVERY_FINDINGS = 10
MAX_DELIVERY_CHARS = 48_000
MAX_HOOK_INPUT_BYTES = 1_048_576
DEFAULT_STOP_GRACE_SECONDS = 2.0
STOP_POLL_SECONDS = 0.025
MAX_HINT_FILE_BYTES = 2_000_000


def _hint_reviewed_files(
    event_input: dict[str, object], *, root: Path
) -> tuple[ReviewedFile, ...]:
    """Snapshot only path/digest metadata for files named by a direct edit hook."""
    tool_input = event_input.get("tool_input")
    if not isinstance(tool_input, dict):
        return ()
    candidates: set[str] = set()
    file_path = tool_input.get("file_path")
    if isinstance(file_path, str):
        candidates.add(file_path)
    command = tool_input.get("command")
    if isinstance(command, str):
        candidates.update(
            match.group(1).strip()
            for match in re.finditer(
                r"^\*\*\* (?:Add|Update|Delete) File: (.+)$",
                command,
                flags=re.MULTILINE,
            )
        )
    reviewed: list[ReviewedFile] = []
    canonical_root = root.resolve()
    for candidate in sorted(candidates):
        path = Path(candidate).expanduser()
        if not path.is_absolute():
            path = canonical_root / path
        try:
            relative = path.resolve(strict=False).relative_to(canonical_root)
            if redact_path(relative).total:
                # This hint is retained before attachment sanitization runs.
                # Omit secret-bearing paths instead of persisting the filename.
                continue
            raw = read_bounded_beneath_root(
                canonical_root, relative, max_bytes=MAX_HINT_FILE_BYTES
            )
        except (OSError, ValueError):
            continue
        if len(raw) > MAX_HINT_FILE_BYTES:
            continue
        try:
            sanitized = redact_text(raw.decode("utf-8")).text.encode()
        except UnicodeDecodeError:
            continue
        reviewed.append(
            ReviewedFile(
                relative.as_posix(),
                hashlib.sha256(sanitized).hexdigest(),
                MAX_HINT_FILE_BYTES,
            )
        )
        if len(reviewed) == 100:
            break
    return tuple(reviewed)


AGENT_ACTION = (
    "Next: independently reproduce each finding against the current code. "
    "If valid, apply the smallest focused fix and add or update a regression "
    "test; if invalid, do not edit. Do not quote the full feedback unless the "
    "user asks."
)


def _record_delivery_metric(
    directory: Path,
    *,
    event: str,
    event_input: dict[str, object],
    payload: dict[str, object],
    response: dict[str, object],
    started_at: float,
) -> None:
    """Record bounded latency/protocol metadata without source or finding text."""
    metrics = _private_directory(directory.expanduser().resolve(), "metrics")
    response_bytes = json.dumps(response, sort_keys=True, separators=(",", ":")).encode()
    first_observed_at = float(payload["first_observed_at"])
    batch_flushed_at = float(payload["batch_flushed_at"])
    provider_started_at = float(payload["provider_started_at"])
    provider_completed_at = float(payload["provider_completed_at"])
    published_at = float(payload["published_at"])
    debounce_ms = float(payload["debounce_ms"])
    provider_ms = float(payload["provider_ms"])
    delivered_at = time.time()
    hook_wait_ms = max(0.0, (delivered_at - published_at) * 1_000)
    value = {
        "version": 1,
        "batch_id": payload["batch_id"],
        "root": payload["root"],
        "session_id": payload["session_id"],
        "recorded_at": time.time(),
        "event": event,
        "input_fields": sorted(event_input),
        "agent_session_sha256": hashlib.sha256(
            str(event_input.get("session_id", "")).encode()
        ).hexdigest(),
        "tool_name": event_input.get("tool_name"),
        "stop_hook_active": event_input.get("stop_hook_active"),
        "response_fields": sorted(response),
        "response_sha256": hashlib.sha256(response_bytes).hexdigest(),
        "response_bytes": len(response_bytes),
        "debounce_ms": debounce_ms,
        "detection_to_flush_ms": max(
            0.0, (batch_flushed_at - first_observed_at) * 1_000
        ),
        "flush_to_provider_ms": max(
            0.0, (provider_started_at - batch_flushed_at) * 1_000
        ),
        "provider_ms": provider_ms,
        "publication_ms": max(
            0.0, (published_at - provider_completed_at) * 1_000
        ),
        "hook_wait_ms": hook_wait_ms,
        "hook_execution_ms": max(0.0, (time.perf_counter() - started_at) * 1_000),
        "total_edit_to_feedback_ms": max(
            0.0, (delivered_at - first_observed_at) * 1_000
        ),
    }
    descriptor, temporary_name = tempfile.mkstemp(prefix=".metric-", dir=metrics)
    temporary = Path(temporary_name)
    destination = metrics / f"{time.time_ns()}-{uuid.uuid4()}.json"
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _private_directory(directory: Path, name: str) -> Path:
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory.chmod(0o700)
    metadata = directory.stat()
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
        raise PermissionError(f"spool directory must be owner-only: {directory}")
    path = directory / name
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.chmod(0o700)
    return path


def _load_payload(path: Path) -> dict[str, object] | None:
    try:
        if path.stat().st_size > MAX_SPOOL_PAYLOAD_BYTES:
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def verify_session_lease(
    directory: Path,
    *,
    root: Path,
    configured_session_id: str,
    codex_session_id: object,
) -> bool:
    """Atomically bind one configured route generation to one real session."""
    if not isinstance(codex_session_id, str) or not codex_session_id:
        return False
    directory = directory.expanduser().resolve()
    root = root.expanduser().resolve()
    with session_route_lock(
        directory,
        root=root,
        configured_session_id=configured_session_id,
        exclusive=True,
    ):
        state = read_session_state(
            directory, root=root, configured_session_id=configured_session_id
        )
        route_state, generation = _state_generation(
            state, root=root, configured_session_id=configured_session_id
        )
        if route_state == "bound":
            return state is not None and state.get("codex_session_id") == codex_session_id
        write_session_state(
            directory,
            root=root,
            configured_session_id=configured_session_id,
            value={
                "version": 1,
                "state": "bound",
                "generation": generation,
                "root": os.fspath(root),
                "configured_session_id": configured_session_id,
                "codex_session_id": codex_session_id,
            },
        )
        return True


def release_session_lease(
    directory: Path,
    *,
    root: Path,
    configured_session_id: str,
    agent_session_id: object,
) -> bool:
    """Close one exact session generation without stopping its active watcher."""
    if not isinstance(agent_session_id, str) or not agent_session_id:
        return False
    directory = directory.expanduser().resolve()
    root = root.expanduser().resolve()
    with session_route_lock(
        directory,
        root=root,
        configured_session_id=configured_session_id,
        exclusive=True,
    ):
        state = read_session_state(
            directory, root=root, configured_session_id=configured_session_id
        )
        route_state, generation = _state_generation(
            state, root=root, configured_session_id=configured_session_id
        )
        if route_state != "bound" or state is None:
            return False
        if state.get("codex_session_id") != agent_session_id:
            return False
        expected_root = os.fspath(root)
        write_session_state(
            directory,
            root=root,
            configured_session_id=configured_session_id,
            value={
                "version": 1,
                "state": "closed",
                "generation": generation + 1,
                "root": expected_root,
                "configured_session_id": configured_session_id,
            },
        )
        return True


def recover_abandoned_claims(directory: Path, *, claim_timeout: float) -> None:
    """Make crashed, unacknowledged claims available for a later retry."""
    now = time.time()
    claimed = _private_directory(directory, "claimed")
    pending = _private_directory(directory, "pending")
    for path in claimed.glob("*.json"):
        try:
            if now - path.stat().st_mtime >= claim_timeout:
                os.replace(path, pending / path.name)
        except FileNotFoundError:
            continue


def claim_feedback(
    directory: Path,
    *,
    root: Path,
    session_id: str,
    claim_timeout: float = 300,
    agent_session_id: str | None = None,
) -> Path | None:
    """Atomically claim feedback for one route and, when supplied, generation."""
    directory = directory.expanduser().resolve()
    root = root.expanduser().resolve()
    lock = (
        session_route_lock(
            directory,
            root=root,
            configured_session_id=session_id,
            exclusive=False,
        )
        if agent_session_id is not None
        else nullcontext()
    )
    with lock:
        expected_generation: int | None = None
        if agent_session_id is not None:
            state = read_session_state(
                directory, root=root, configured_session_id=session_id
            )
            route_state, expected_generation = _state_generation(
                state, root=root, configured_session_id=session_id
            )
            if (
                route_state != "bound"
                or state is None
                or state.get("codex_session_id") != agent_session_id
            ):
                return None
        recover_abandoned_claims(directory, claim_timeout=claim_timeout)
        pending = _private_directory(directory, "pending")
        claimed = _private_directory(directory, "claimed")
        expected_root = os.fspath(root)
        candidates: list[tuple[float, Path]] = []
        for path in pending.glob("*.json"):
            try:
                candidates.append((path.stat().st_mtime, path))
            except FileNotFoundError:
                continue
        for _, candidate in sorted(candidates):
            payload = _load_payload(candidate)
            if payload is None:
                continue
            if payload.get("root") != expected_root or payload.get("session_id") != session_id:
                continue
            if (
                expected_generation is not None
                and payload.get("session_generation", 0) != expected_generation
            ):
                continue
            destination = claimed / candidate.name
            try:
                # A claim timeout measures time since the claim, not time since the
                # batch was originally published. Touch before the atomic rename so
                # another consumer can never observe an old newly-claimed file.
                os.utime(candidate, None)
                os.replace(candidate, destination)
            except FileNotFoundError:
                continue
            return destination
        return None


def acknowledge(directory: Path, claim: Path) -> None:
    acknowledged = _private_directory(directory.expanduser().resolve(), "acknowledged")
    os.replace(claim, acknowledged / claim.name)


def reject(directory: Path, claim: Path) -> None:
    rejected = _private_directory(directory.expanduser().resolve(), "rejected")
    os.replace(claim, rejected / claim.name)


def requeue_feedback(
    directory: Path, claim: Path, payload: dict[str, object]
) -> None:
    """Replace a claim with its undelivered remainder and make it pending."""
    descriptor, temporary_name = tempfile.mkstemp(prefix=".requeue-", dir=claim.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, claim)
        pending = _private_directory(directory.expanduser().resolve(), "pending")
        os.replace(claim, pending / claim.name)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def render_feedback_chunk(
    payload: dict[str, object],
) -> tuple[str | None, list[dict[str, object]]]:
    """Render one bounded hook message and return every undelivered finding."""
    findings = payload.get("findings")
    if not isinstance(findings, list) or not findings:
        return None, []
    reviewed_files = payload.get("reviewed_files")
    reviewed_count = len(reviewed_files) if isinstance(reviewed_files, list) else 0
    finding_count = sum(isinstance(finding, dict) for finding in findings)
    finding_label = "defect" if finding_count == 1 else "defects"
    file_label = "file" if reviewed_count == 1 else "files"
    ready_line = (
        f"Quodet review ready: {finding_count} likely {finding_label} in "
        f"{reviewed_count} reviewed {file_label}."
    )
    published_at = payload.get("published_at")
    debounce_ms = payload.get("debounce_ms")
    provider_ms = payload.get("provider_ms")
    if all(isinstance(value, (int, float)) for value in (published_at, debounce_ms, provider_ms)):
        hook_wait_ms = max(0.0, (time.time() - float(published_at)) * 1_000)
        provider_completed_at = payload.get("provider_completed_at")
        publication_ms = (
            max(0.0, (float(published_at) - float(provider_completed_at)) * 1_000)
            if isinstance(provider_completed_at, (int, float))
            else 0.0
        )
        first_observed_at = payload.get("first_observed_at")
        total_ms = (
            max(0.0, (time.time() - float(first_observed_at)) * 1_000)
            if isinstance(first_observed_at, (int, float))
            else float(debounce_ms) + float(provider_ms) + hook_wait_ms
        )
        latency_line = (
            f"Batch {short_batch_id(str(payload['batch_id']))} latency: watcher debounce "
            f"{float(debounce_ms):.1f} ms; provider {float(provider_ms):.1f} ms; "
            f"publication {publication_ms:.1f} ms; "
            f"hook delivery wait {hook_wait_ms:.1f} ms; "
            f"total edit-to-feedback {total_ms:.1f} ms."
        )
        lines = [ready_line, UNTRUSTED_NOTICE, AGENT_ACTION, latency_line]
    else:
        lines = [ready_line, UNTRUSTED_NOTICE, AGENT_ACTION]
    lifecycle = payload.get("lifecycle")
    if isinstance(lifecycle, list):
        counts: dict[str, int] = {}
        for item in lifecycle:
            if isinstance(item, dict) and isinstance(item.get("status"), str):
                status = str(item["status"])
                counts[status] = counts.get(status, 0) + 1
        visible = {key: value for key, value in counts.items() if key != "new"}
        if visible:
            summary = ", ".join(
                f"{count} {status.replace('_', ' ')}"
                for status, count in sorted(visible.items())
            )
            lines.append(f"Finding lifecycle: {summary}.")
    delivered = 0
    for index, finding in enumerate(findings):
        if not isinstance(finding, dict):
            continue
        block = [
            "",
            f"- {finding.get('file')}:{finding.get('line')} "
            f"[{finding.get('severity')}, confidence {finding.get('confidence')}]",
            f"  {finding.get('title')}",
            f"  Evidence: {finding.get('explanation')}",
            f"  Suggested fix: {finding.get('suggested_fix')}",
        ]
        candidate = "\n".join([*lines, *block])
        if delivered == MAX_DELIVERY_FINDINGS or len(candidate) > MAX_DELIVERY_CHARS:
            remainder = [item for item in findings[index:] if isinstance(item, dict)]
            return "\n".join(lines), remainder
        lines.extend(block)
        delivered += 1
    return "\n".join(lines), []


def render_feedback(payload: dict[str, object]) -> str | None:
    """Render the next bounded feedback chunk."""
    return render_feedback_chunk(payload)[0]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", choices=("PostToolUse", "Stop"))
    parser.add_argument("--spool-dir", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--claim-timeout", type=float, default=300)
    parser.add_argument("--stop-grace", type=float, default=DEFAULT_STOP_GRACE_SECONDS)
    parser.add_argument("--flush-hint-ttl", type=float, default=30.0)
    return parser.parse_args(argv)


def _claim_with_stop_grace(
    directory: Path,
    *,
    root: Path,
    session_id: str,
    agent_session_id: str,
    claim_timeout: float,
    stop_grace: float,
) -> Path | None:
    """Wait only while this exact agent session owns a live provider review."""
    claim = claim_feedback(
        directory,
        root=root,
        session_id=session_id,
        claim_timeout=claim_timeout,
        agent_session_id=agent_session_id,
    )
    if claim is not None or stop_grace <= 0:
        return claim
    deadline = time.monotonic() + stop_grace
    active = matching_review_in_flight(
        directory,
        root=root,
        session_id=session_id,
        agent_session_id=agent_session_id,
    )
    while active:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        time.sleep(min(STOP_POLL_SECONDS, remaining))
        claim = claim_feedback(
            directory,
            root=root,
            session_id=session_id,
            claim_timeout=claim_timeout,
            agent_session_id=agent_session_id,
        )
        if claim is not None:
            return claim
        active = matching_review_in_flight(
            directory,
            root=root,
            session_id=session_id,
            agent_session_id=agent_session_id,
            validate_hint_files=False,
        )
    return None


def main(argv: Sequence[str] | None = None) -> int:
    hook_started_at = time.perf_counter()
    args = parse_args(argv)
    try:
        raw_input = sys.stdin.read(MAX_HOOK_INPUT_BYTES + 1)
        if len(raw_input.encode("utf-8")) > MAX_HOOK_INPUT_BYTES:
            event_input = {}
        else:
            event_input = json.loads(raw_input)
    except (UnicodeEncodeError, json.JSONDecodeError):
        event_input = {}
    if not isinstance(event_input, dict):
        event_input = {}
    event = args.event or event_input.get("hook_event_name")
    if event not in {"PostToolUse", "Stop"}:
        return 0
    if (
        not 0 <= args.stop_grace <= 10
        or not 0 < args.flush_hint_ttl <= 300
    ):
        if event == "Stop":
            print("{}")
        return 0
    # When Codex supplies identity/root fields, independently verify them. A
    # copied hook configuration must never consume another session's batch.
    input_cwd = event_input.get("cwd")
    if not isinstance(input_cwd, str) or Path(input_cwd).resolve() != args.root.resolve():
        if event == "Stop":
            print("{}")
        return 0
    agent_session_id = event_input.get("session_id")
    if not verify_session_lease(
        args.spool_dir,
        root=args.root,
        configured_session_id=args.session_id,
        codex_session_id=agent_session_id,
    ):
        if event == "Stop":
            print("{}")
        return 0
    if event == "Stop" and event_input.get("stop_hook_active") is True:
        print("{}")
        return 0
    assert isinstance(agent_session_id, str)

    if event == "PostToolUse":
        reviewed_files = _hint_reviewed_files(event_input, root=args.root)
        try:
            if reviewed_files:
                publish_flush_hint(
                    args.spool_dir,
                    root=args.root,
                    session_id=args.session_id,
                    agent_session_id=agent_session_id,
                    ttl_seconds=args.flush_hint_ttl,
                    reviewed_files=reviewed_files,
                )
        except (OSError, ValueError) as error:
            print(f"Could not publish watcher flush hint: {error}", file=sys.stderr)

    if event == "Stop":
        try:
            request_flush_hint(
                args.spool_dir,
                root=args.root,
                session_id=args.session_id,
                agent_session_id=agent_session_id,
                ttl_seconds=min(10.0, max(0.1, args.stop_grace or 0.1)),
            )
        except OSError as error:
            print(f"Could not request watcher flush: {error}", file=sys.stderr)
        except ValueError:
            # A producer may have exited between lease verification and Stop.
            # With no active watcher there is nothing to flush or wait for.
            pass
        claim = _claim_with_stop_grace(
            args.spool_dir,
            root=args.root,
            session_id=args.session_id,
            agent_session_id=agent_session_id,
            claim_timeout=args.claim_timeout,
            stop_grace=args.stop_grace,
        )
    else:
        claim = claim_feedback(
            args.spool_dir,
            root=args.root,
            session_id=args.session_id,
            claim_timeout=args.claim_timeout,
            agent_session_id=agent_session_id,
        )
    if claim is None:
        if event == "Stop":
            print("{}")
        return 0
    payload = _load_payload(claim)
    try:
        validated = validate_spooled_payload(
            payload or {}, root=args.root, session_id=args.session_id
        )
        validated = fresh_spooled_payload(validated, root=args.root)
    except ReviewValidationError as error:
        print(f"Rejected invalid feedback batch: {error}", file=sys.stderr)
        reject(args.spool_dir, claim)
        if event == "Stop":
            print("{}")
        return 0
    retire_reviewed_flush_hints(
        args.spool_dir,
        root=args.root,
        session_id=args.session_id,
        reviewed_files=tuple(
            ReviewedFile(
                str(item["path"]), str(item["sha256"]), int(item["size"])
            )
            for item in validated["reviewed_files"]  # type: ignore[union-attr]
        ),
    )
    message, remaining = render_feedback_chunk(validated)
    if message is None:
        acknowledge(args.spool_dir, claim)
        if event == "Stop":
            print("{}")
        return 0

    if event == "PostToolUse":
        response = {
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": message,
            }
        }
    else:
        response = {"decision": "block", "reason": message}
    print(json.dumps(response))
    sys.stdout.flush()
    try:
        _record_delivery_metric(
            args.spool_dir,
            event=event,
            event_input=event_input,
            payload=validated,
            response=response,
            started_at=hook_started_at,
        )
    except OSError as error:
        print(f"Could not record hook latency: {error}", file=sys.stderr)
    if remaining:
        remainder = validated.copy()
        remainder["findings"] = remaining
        # Lifecycle and stale-path summaries describe the whole provider batch,
        # not each delivery chunk. They were emitted with the first chunk.
        remainder["lifecycle"] = []
        remainder["stale_files"] = []
        requeue_feedback(args.spool_dir, claim, remainder)
    else:
        acknowledge(args.spool_dir, claim)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
