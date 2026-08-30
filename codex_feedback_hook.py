#!/usr/bin/env python3
"""Consume Quodet feedback from Codex or Claude Code agent hooks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
import time
from pathlib import Path
from typing import Sequence

from feedback import (
    MAX_PROVIDER_OUTPUT_BYTES,
    UNTRUSTED_NOTICE,
    ReviewValidationError,
    fresh_spooled_payload,
    validate_spooled_payload,
)


MAX_DELIVERY_FINDINGS = 10
MAX_DELIVERY_CHARS = 48_000


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
        if path.stat().st_size > MAX_PROVIDER_OUTPUT_BYTES:
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
    """Atomically bind one configured route to one real Codex session."""
    if not isinstance(codex_session_id, str) or not codex_session_id:
        return False
    leases = _private_directory(directory.expanduser().resolve(), "sessions")
    identity = f"{root.resolve()}\0{configured_session_id}".encode()
    lease = leases / f"{hashlib.sha256(identity).hexdigest()}.json"
    expected = {
        "root": os.fspath(root.resolve()),
        "configured_session_id": configured_session_id,
        "codex_session_id": codex_session_id,
    }
    descriptor, temporary_name = tempfile.mkstemp(prefix=".lease-", dir=leases)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(expected, stream, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, lease)
            return True
        except FileExistsError:
            return _load_payload(lease) == expected
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


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
    directory: Path, *, root: Path, session_id: str, claim_timeout: float = 300
) -> Path | None:
    """Atomically claim the oldest batch belonging to this exact root/session."""
    directory = directory.expanduser().resolve()
    recover_abandoned_claims(directory, claim_timeout=claim_timeout)
    pending = _private_directory(directory, "pending")
    claimed = _private_directory(directory, "claimed")
    expected_root = os.fspath(root.expanduser().resolve())
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
    lines = [UNTRUSTED_NOTICE]
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        event_input = json.load(sys.stdin)
    except json.JSONDecodeError:
        event_input = {}
    event = args.event or event_input.get("hook_event_name")
    if event not in {"PostToolUse", "Stop"}:
        return 0
    # When Codex supplies identity/root fields, independently verify them. A
    # copied hook configuration must never consume another session's batch.
    input_cwd = event_input.get("cwd")
    if not isinstance(input_cwd, str) or Path(input_cwd).resolve() != args.root.resolve():
        if event == "Stop":
            print("{}")
        return 0
    if not verify_session_lease(
        args.spool_dir,
        root=args.root,
        configured_session_id=args.session_id,
        codex_session_id=event_input.get("session_id"),
    ):
        if event == "Stop":
            print("{}")
        return 0
    if event == "Stop" and event_input.get("stop_hook_active") is True:
        print("{}")
        return 0

    claim = claim_feedback(
        args.spool_dir,
        root=args.root,
        session_id=args.session_id,
        claim_timeout=args.claim_timeout,
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
    if remaining:
        remainder = validated.copy()
        remainder["findings"] = remaining
        requeue_feedback(args.spool_dir, claim, remainder)
    else:
        acknowledge(args.spool_dir, claim)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
