"""Validated review results and durable feedback delivery.

Provider responses are untrusted.  This module is the boundary that turns them
into bounded values before they can be displayed or delivered to an agent.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import stat
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Protocol, Sequence

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows lacks flock.
    fcntl = None  # type: ignore[assignment]


MAX_FINDINGS = 50
MAX_REVIEWED_FILES = 100
MAX_PROVIDER_OUTPUT_BYTES = 256_000
MAX_SPOOL_PAYLOAD_BYTES = 1_048_576
MAX_REVIEWED_FILE_BYTES = 100_000_000
MAX_PATH_LENGTH = 1_024
MAX_TITLE_LENGTH = 300
MAX_EXPLANATION_LENGTH = 8_000
MAX_FIX_LENGTH = 2_000
DEFAULT_ROUND_RESET_SECONDS = 900
UNTRUSTED_NOTICE = (
    "Untrusted automated review suggestions follow. Independently verify every "
    "finding against the current code before editing."
)


class ReviewValidationError(ValueError):
    """The provider returned output that is unsafe or does not match the schema."""


@dataclass(frozen=True)
class ReviewedFile:
    path: str
    sha256: str
    size: int


@dataclass(frozen=True)
class ReviewFinding:
    file: str
    line: int
    severity: str
    confidence: float
    title: str
    explanation: str
    suggested_fix: str


@dataclass(frozen=True)
class ReviewBatch:
    batch_id: str
    root: str
    created_at: float
    reviewed_files: tuple[ReviewedFile, ...]
    findings: tuple[ReviewFinding, ...]
    session_id: str | None = None
    feedback_round: int = 1
    debounce_ms: float = 0.0
    provider_ms: float = 0.0
    first_observed_at: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class FeedbackSink(Protocol):
    def publish(self, batch: ReviewBatch) -> bool: ...


def read_bounded_beneath_root(
    root: Path, relative_path: Path, *, max_bytes: int
) -> bytes:
    """Read one regular file without following symlinks below a trusted root."""
    if relative_path.is_absolute() or ".." in relative_path.parts or max_bytes < 0:
        raise OSError(errno.EINVAL, "invalid bounded relative read")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    if not nofollow or not directory_flag or os.open not in os.supports_dir_fd:
        raise OSError(errno.ENOTSUP, "safe descriptor-relative opens are unavailable")
    common_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow
    descriptors: list[int] = []
    try:
        current = os.open(root, common_flags | directory_flag)
        descriptors.append(current)
        for component in relative_path.parts[:-1]:
            current = os.open(
                component,
                common_flags | directory_flag,
                dir_fd=current,
            )
            descriptors.append(current)
        file_descriptor = os.open(
            relative_path.name,
            common_flags | getattr(os, "O_NONBLOCK", 0),
            dir_fd=current,
        )
        descriptors.append(file_descriptor)
        if not stat.S_ISREG(os.fstat(file_descriptor).st_mode):
            raise OSError(errno.EINVAL, "source is not a regular file")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(file_descriptor, min(remaining, 128 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _sha256_inside_root(root: Path, relative_path: str, *, max_bytes: int) -> str:
    raw = read_bounded_beneath_root(
        root, Path(relative_path), max_bytes=max_bytes
    )
    if len(raw) > max_bytes:
        raise OSError(errno.EFBIG, "source exceeds reviewed size")
    return hashlib.sha256(raw).hexdigest()


def _bounded_string(value: object, field: str, maximum: int) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ReviewValidationError(f"{field} must be a non-empty string <= {maximum}")
    return value


def _normalize_finding_path(value: object, root: Path, reviewed: set[str]) -> str:
    raw = _bounded_string(value, "file", MAX_PATH_LENGTH).replace("\\", "/")
    candidate = PurePosixPath(raw)
    if candidate.is_absolute() or ".." in candidate.parts or raw.startswith("~"):
        raise ReviewValidationError("finding path must be relative and cannot traverse")
    normalized = candidate.as_posix()
    try:
        (root / normalized).resolve(strict=False).relative_to(root)
    except (OSError, ValueError) as error:
        raise ReviewValidationError("finding path escapes watched root") from error
    if normalized not in reviewed:
        raise ReviewValidationError("finding path was not part of the reviewed snapshot")
    return normalized


def parse_review_output(
    output: str,
    *,
    root: Path,
    reviewed_files: Sequence[ReviewedFile],
    session_id: str | None = None,
    feedback_round: int = 1,
    debounce_ms: float = 0.0,
    provider_ms: float = 0.0,
    first_observed_at: float | None = None,
) -> ReviewBatch:
    """Parse and strictly validate one provider response."""
    if len(output.encode("utf-8")) > MAX_PROVIDER_OUTPUT_BYTES:
        raise ReviewValidationError("provider output exceeds size limit")
    try:
        value = json.loads(output)
    except json.JSONDecodeError as error:
        raise ReviewValidationError("provider output is not valid JSON") from error
    if not isinstance(value, dict) or set(value) != {"findings"}:
        raise ReviewValidationError("provider output must contain only findings")
    raw_findings = value["findings"]
    if not isinstance(raw_findings, list) or len(raw_findings) > MAX_FINDINGS:
        raise ReviewValidationError(f"findings must be an array of at most {MAX_FINDINGS}")

    expected_fields = {
        "file", "line", "severity", "confidence", "title", "explanation", "suggested_fix"
    }
    reviewed = {item.path for item in reviewed_files}
    findings: list[ReviewFinding] = []
    for index, raw in enumerate(raw_findings):
        if not isinstance(raw, dict) or set(raw) != expected_fields:
            raise ReviewValidationError(f"finding {index} has unexpected or missing fields")
        line = raw["line"]
        confidence = raw["confidence"]
        severity = raw["severity"]
        if isinstance(line, bool) or not isinstance(line, int) or line < 1:
            raise ReviewValidationError("line must be a positive integer")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ReviewValidationError("confidence must be numeric")
        if not 0.95 <= float(confidence) <= 1.0:
            raise ReviewValidationError("confidence must be between 0.95 and 1.0")
        if severity not in {"critical", "high", "medium", "low"}:
            raise ReviewValidationError("severity is invalid")
        findings.append(
            ReviewFinding(
                file=_normalize_finding_path(raw["file"], root, reviewed),
                line=line,
                severity=severity,
                confidence=float(confidence),
                title=_bounded_string(raw["title"], "title", MAX_TITLE_LENGTH),
                explanation=_bounded_string(
                    raw["explanation"], "explanation", MAX_EXPLANATION_LENGTH
                ),
                suggested_fix=_bounded_string(
                    raw["suggested_fix"], "suggested_fix", MAX_FIX_LENGTH
                ),
            )
        )
    created_at = time.time()
    observed_at = first_observed_at
    if observed_at is None:
        observed_at = created_at - (debounce_ms + provider_ms) / 1_000
    return ReviewBatch(
        batch_id=str(uuid.uuid4()),
        root=os.fspath(root),
        created_at=created_at,
        reviewed_files=tuple(reviewed_files),
        findings=tuple(findings),
        session_id=session_id,
        feedback_round=feedback_round,
        debounce_ms=debounce_ms,
        provider_ms=provider_ms,
        first_observed_at=observed_at,
    )


def fresh_findings(batch: ReviewBatch) -> ReviewBatch:
    """Remove findings for files that no longer match the reviewed bytes."""
    root = Path(batch.root).resolve()
    fresh_paths: set[str] = set()
    for reviewed in batch.reviewed_files:
        try:
            if _sha256_inside_root(
                root, reviewed.path, max_bytes=reviewed.size
            ) == reviewed.sha256:
                fresh_paths.add(reviewed.path)
        except (OSError, ValueError):
            pass
    return replace(batch, findings=tuple(f for f in batch.findings if f.file in fresh_paths))


class ConsoleSink:
    """Preserve the historical JSON result printed in the terminal."""

    def publish(self, batch: ReviewBatch) -> bool:
        print(json.dumps({"findings": [asdict(item) for item in batch.findings]}, indent=2))
        return True


def _ensure_private_directory(path: Path) -> Path:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.chmod(0o700)
    metadata = path.stat()
    mode = stat.S_IMODE(metadata.st_mode)
    if metadata.st_uid != os.getuid() or mode != 0o700:
        raise PermissionError(f"spool directory must be owner-only: {path}")
    return path


def _global_root_lease_directory() -> Path:
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    base = Path(runtime) if runtime else Path(tempfile.gettempdir())
    return _ensure_private_directory(base / f"quodet-{os.getuid()}" / "active-roots")


def _atomic_private_json(path: Path, value: dict[str, object]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


class SpoolSink:
    """Publish batches atomically to an explicitly owned agent session."""

    def __init__(
        self,
        directory: Path,
        *,
        root: Path,
        session_id: str,
        max_feedback_rounds: int = 3,
        round_reset_seconds: float = DEFAULT_ROUND_RESET_SECONDS,
    ) -> None:
        if not session_id or len(session_id) > 200:
            raise ValueError("session_id must be a non-empty bounded string")
        self.root = root.resolve()
        self.directory = directory.expanduser().resolve()
        if self.directory == self.root or self.root in self.directory.parents:
            raise ValueError("spool directory must be outside the watched root")
        self.session_id = session_id
        if not 1 <= max_feedback_rounds <= 3:
            raise ValueError("max_feedback_rounds must be between one and three")
        self.max_feedback_rounds = max_feedback_rounds
        self.round_reset_seconds = round_reset_seconds
        _ensure_private_directory(self.directory)
        _ensure_private_directory(self.directory / "pending")
        _ensure_private_directory(self.directory / "claimed")
        _ensure_private_directory(self.directory / "acknowledged")
        _ensure_private_directory(self.directory / "dedupe")
        if fcntl is None:
            raise RuntimeError("spool delivery requires process-exclusive file locks")
        active_roots = _global_root_lease_directory()
        root_key = hashlib.sha256(os.fspath(self.root).encode()).hexdigest()
        lease = active_roots / f"{root_key}.lock"
        self._root_lease_stream = lease.open("a+", encoding="utf-8")
        lease.chmod(0o600)
        try:
            fcntl.flock(
                self._root_lease_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
            )
        except BlockingIOError as error:
            self._root_lease_stream.close()
            raise ValueError(
                "watched root is already leased to another feedback producer"
            ) from error
        self._root_lease_stream.seek(0)
        self._root_lease_stream.truncate()
        json.dump(
            {"root": os.fspath(self.root), "session_id": self.session_id},
            self._root_lease_stream,
            separators=(",", ":"),
        )
        self._root_lease_stream.flush()

    def close(self) -> None:
        stream = getattr(self, "_root_lease_stream", None)
        if stream is not None and not stream.closed:
            stream.close()

    def __del__(self) -> None:
        self.close()

    def _apply_round_policy(self, batch: ReviewBatch) -> ReviewBatch:
        """Bound repeated feedback per file while allowing unrelated changes."""
        policy_directory = _ensure_private_directory(self.directory / "policy")
        policy_path = policy_directory / "rounds.json"
        try:
            state = json.loads(policy_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            state = {}
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("invalid feedback round policy state") from error
        if not isinstance(state, dict):
            raise ValueError("invalid feedback round policy state")

        now = time.time()
        digests = {item.path: item.sha256 for item in batch.reviewed_files}
        allowed: list[ReviewFinding] = []
        allowed_rounds: list[int] = []
        findings_by_file: dict[str, list[ReviewFinding]] = {}
        for finding in batch.findings:
            findings_by_file.setdefault(finding.file, []).append(finding)
        for finding_file, file_findings in findings_by_file.items():
            # Provider prose can vary between identical reviews. Defect
            # identity is the cited source bytes and location, not wording.
            serialized_findings = sorted(
                f"{finding.file}:{finding.line}" for finding in file_findings
            )
            signature = hashlib.sha256(
                "\0".join(serialized_findings).encode()
            ).hexdigest()
            previous = state.get(finding_file)
            if (
                isinstance(previous, dict)
                and isinstance(previous.get("updated_at"), (int, float))
                and now - float(previous["updated_at"]) <= self.round_reset_seconds
            ):
                previous_digest = previous.get("sha256")
                previous_signature = previous.get("finding_signature")
                previous_round = previous.get("round")
                if isinstance(previous_round, int):
                    round_number = (
                        previous_round
                        if previous_digest == digests[finding_file]
                        and previous_signature == signature
                        else previous_round + 1
                    )
                else:
                    round_number = 1
            else:
                round_number = 1
            state[finding_file] = {
                "sha256": digests[finding_file],
                "finding_signature": signature,
                "round": round_number,
                "updated_at": now,
            }
            if round_number <= self.max_feedback_rounds:
                allowed.extend(file_findings)
                allowed_rounds.append(round_number)
        _atomic_private_json(policy_path, state)
        return replace(
            batch,
            findings=tuple(allowed),
            feedback_round=max(allowed_rounds, default=self.max_feedback_rounds + 1),
        )

    def _fingerprint(self, batch: ReviewBatch) -> str:
        cited_paths = {finding.file for finding in batch.findings}
        stable = {
            "root": batch.root,
            "session_id": batch.session_id,
            "reviewed_files": [
                asdict(item) for item in batch.reviewed_files if item.path in cited_paths
            ],
            "findings": sorted(
                {f"{item.file}:{item.line}" for item in batch.findings}
            ),
        }
        encoded = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def publish(self, batch: ReviewBatch) -> bool:
        if batch.session_id != self.session_id or Path(batch.root).resolve() != self.root:
            raise ValueError("batch ownership does not match this spool")
        batch = fresh_findings(batch)
        if not batch.findings:
            return False
        batch = self._apply_round_policy(batch)
        if not batch.findings:
            return False
        payload = batch.to_dict()
        payload["notice"] = UNTRUSTED_NOTICE
        encoded_payload = json.dumps(payload, separators=(",", ":")).encode()
        if len(encoded_payload) > MAX_SPOOL_PAYLOAD_BYTES:
            raise ReviewValidationError("spooled batch exceeds envelope size limit")
        fingerprint = self._fingerprint(batch)
        filename = f"{fingerprint}.json"
        states = ("pending", "claimed", "acknowledged")
        if any((self.directory / state / filename).exists() for state in states):
            return False

        # The durable full-payload dedupe record also repairs a crash between
        # recording the fingerprint and publishing it to pending/.
        record = self.directory / "dedupe" / filename
        if not record.exists():
            _atomic_private_json(record, payload)
        destination = self.directory / "pending" / filename
        try:
            os.link(record, destination)
        except FileExistsError:
            return False
        return True


class CompositeSink:
    def __init__(self, sinks: Sequence[FeedbackSink]) -> None:
        self.sinks = tuple(sinks)

    def publish(self, batch: ReviewBatch) -> bool:
        published = False
        for sink in self.sinks:
            published = sink.publish(batch) or published
        return published


def validate_spooled_payload(
    payload: dict[str, object], *, root: Path, session_id: str
) -> dict[str, object]:
    """Independently validate a durable record at the consumer boundary."""
    expected_fields = {
        "batch_id",
        "root",
        "created_at",
        "reviewed_files",
        "findings",
        "session_id",
        "feedback_round",
        "debounce_ms",
        "provider_ms",
        "first_observed_at",
        "notice",
    }
    if set(payload) != expected_fields:
        raise ReviewValidationError("spooled batch has unexpected or missing fields")
    if payload["root"] != os.fspath(root.resolve()) or payload["session_id"] != session_id:
        raise ReviewValidationError("spooled batch ownership does not match")
    try:
        uuid.UUID(str(payload["batch_id"]))
    except (ValueError, AttributeError) as error:
        raise ReviewValidationError("invalid batch id") from error
    if not isinstance(payload["created_at"], (int, float)):
        raise ReviewValidationError("invalid creation time")
    if (
        isinstance(payload["first_observed_at"], bool)
        or not isinstance(payload["first_observed_at"], (int, float))
        or not 0 < float(payload["first_observed_at"]) <= float(payload["created_at"])
    ):
        raise ReviewValidationError("invalid first observed time")
    feedback_round = payload["feedback_round"]
    if isinstance(feedback_round, bool) or not isinstance(feedback_round, int):
        raise ReviewValidationError("invalid feedback round")
    if not 1 <= feedback_round <= 3 or payload["notice"] != UNTRUSTED_NOTICE:
        raise ReviewValidationError("invalid feedback policy metadata")
    for field in ("debounce_ms", "provider_ms"):
        latency = payload[field]
        if (
            isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or not 0 <= float(latency) <= 86_400_000
        ):
            raise ReviewValidationError(f"invalid {field}")
    raw_reviewed = payload["reviewed_files"]
    if not isinstance(raw_reviewed, list) or len(raw_reviewed) > MAX_REVIEWED_FILES:
        raise ReviewValidationError("invalid reviewed file collection")
    reviewed: list[ReviewedFile] = []
    for item in raw_reviewed:
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "size"}:
            raise ReviewValidationError("invalid reviewed file")
        path = _normalize_finding_path(
            item["path"], root.resolve(), {str(item["path"]).replace("\\", "/")}
        )
        digest = item["sha256"]
        size = item["size"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ReviewValidationError("invalid reviewed file digest")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or not 0 <= size <= MAX_REVIEWED_FILE_BYTES
        ):
            raise ReviewValidationError("invalid reviewed file size")
        reviewed.append(ReviewedFile(path=path, sha256=digest, size=size))
    normalized = parse_review_output(
        json.dumps({"findings": payload["findings"]}),
        root=root.resolve(),
        reviewed_files=reviewed,
        session_id=session_id,
        feedback_round=feedback_round,
        debounce_ms=float(payload["debounce_ms"]),
        provider_ms=float(payload["provider_ms"]),
        first_observed_at=float(payload["first_observed_at"]),
    )
    result = payload.copy()
    result["findings"] = [asdict(item) for item in normalized.findings]
    result["reviewed_files"] = [asdict(item) for item in reviewed]
    return result


def fresh_spooled_payload(payload: dict[str, object], *, root: Path) -> dict[str, object]:
    """Recheck source digests again at the independent consumer boundary."""
    reviewed = {
        str(item["path"]): (str(item["sha256"]), int(item["size"]))
        for item in payload["reviewed_files"]  # type: ignore[union-attr]
    }
    fresh: set[str] = set()
    for relative_path, (digest, size) in reviewed.items():
        try:
            if _sha256_inside_root(
                root.resolve(), relative_path, max_bytes=size
            ) == digest:
                fresh.add(relative_path)
        except (OSError, ValueError):
            pass
    result = payload.copy()
    result["findings"] = [
        finding
        for finding in payload["findings"]  # type: ignore[union-attr]
        if finding["file"] in fresh
    ]
    return result
