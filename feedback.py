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
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Protocol, Sequence, TextIO

from review_output import DEFAULT_OUTPUT_MODE, OUTPUT_MODES, render_review
from redaction import RedactionSummary, redaction_summary_from_document, redact_text

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
DEFAULT_FLUSH_HINT_TTL_SECONDS = 30.0
DEFAULT_AGENT_EDIT_QUIET_SECONDS = 0.25
DEFAULT_AGENT_EDIT_MAX_AGE_SECONDS = 1.0
MAX_IN_FLIGHT_SECONDS = 3_600.0
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
    session_generation: int | None = None
    batch_flushed_at: float = 0.0
    provider_started_at: float = 0.0
    provider_completed_at: float = 0.0
    published_at: float = 0.0
    redactions: RedactionSummary = RedactionSummary()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class FeedbackSink(Protocol):
    def publish(self, batch: ReviewBatch) -> bool: ...


@dataclass(frozen=True)
class FlushHint:
    """One authenticated logical edit boundary from an agent hook."""

    agent_session_id: str
    created_at: float
    path: Path
    reviewed_files: tuple[ReviewedFile, ...]
    constituent_paths: tuple[Path, ...] = ()

    @property
    def paths(self) -> tuple[Path, ...]:
        """Return every durable hint represented by this logical edit group."""
        return self.constituent_paths or (self.path,)


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
    try:
        sanitized = redact_text(raw.decode("utf-8")).text.encode()
    except UnicodeDecodeError as error:
        raise OSError(errno.EILSEQ, "source is not UTF-8 text") from error
    return hashlib.sha256(sanitized).hexdigest()


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
    session_generation: int | None = None,
    batch_flushed_at: float | None = None,
    provider_started_at: float | None = None,
    provider_completed_at: float | None = None,
    redactions: RedactionSummary = RedactionSummary(),
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
    completed_at = provider_completed_at or created_at
    started_at = provider_started_at or completed_at - provider_ms / 1_000
    flushed_at = batch_flushed_at or started_at
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
        session_generation=(
            0 if session_id is not None and session_generation is None
            else session_generation
        ),
        batch_flushed_at=flushed_at,
        provider_started_at=started_at,
        provider_completed_at=completed_at,
        redactions=redactions,
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
    """Render validated results for a person or a machine on stdout."""

    def __init__(
        self,
        *,
        mode: str = DEFAULT_OUTPUT_MODE,
        stream: TextIO | None = None,
    ) -> None:
        if mode not in OUTPUT_MODES:
            raise ValueError(f"unsupported output mode: {mode}")
        self.mode = mode
        self.stream = stream

    def publish(self, batch: ReviewBatch) -> bool:
        import sys

        stream = self.stream if self.stream is not None else sys.stdout
        rendered = render_review(batch, self.mode)
        encoding = getattr(stream, "encoding", None)
        if isinstance(encoding, str):
            rendered = rendered.encode(
                encoding, errors="backslashreplace"
            ).decode(encoding)
        print(rendered, file=stream, flush=True)
        return True


def _ensure_private_directory(path: Path) -> Path:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.chmod(0o700)
    metadata = path.stat()
    mode = stat.S_IMODE(metadata.st_mode)
    if metadata.st_uid != os.getuid() or mode != 0o700:
        raise PermissionError(f"spool directory must be owner-only: {path}")
    return path


def _session_route_key(root: Path, configured_session_id: str) -> str:
    identity = f"{root.resolve()}\0{configured_session_id}".encode()
    return hashlib.sha256(identity).hexdigest()


def session_state_path(
    directory: Path, *, root: Path, configured_session_id: str
) -> Path:
    sessions = _ensure_private_directory(directory.expanduser().resolve() / "sessions")
    return sessions / f"{_session_route_key(root, configured_session_id)}.json"


@contextmanager
def session_route_lock(
    directory: Path,
    *,
    root: Path,
    configured_session_id: str,
    exclusive: bool,
):
    """Serialize route binding changes against generation-aware publication."""
    if fcntl is None:
        raise RuntimeError("session routing requires process-exclusive file locks")
    sessions = _ensure_private_directory(directory.expanduser().resolve() / "sessions")
    lock_path = sessions / f"{_session_route_key(root, configured_session_id)}.lock"
    stream = lock_path.open("a+", encoding="utf-8")
    lock_path.chmod(0o600)
    try:
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(stream.fileno(), operation)
        yield
    finally:
        stream.close()


def read_session_state(
    directory: Path, *, root: Path, configured_session_id: str
) -> dict[str, object] | None:
    """Read one state while the caller holds its session route lock."""
    path = session_state_path(
        directory, root=root, configured_session_id=configured_session_id
    )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid agent session state") from error
    if not isinstance(value, dict):
        raise ValueError("invalid agent session state")
    return value


def write_session_state(
    directory: Path,
    *,
    root: Path,
    configured_session_id: str,
    value: dict[str, object],
) -> None:
    """Replace one state while the caller holds its exclusive route lock."""
    _atomic_private_json(
        session_state_path(
            directory, root=root, configured_session_id=configured_session_id
        ),
        value,
    )


def _state_generation(
    state: dict[str, object] | None,
    *,
    root: Path,
    configured_session_id: str,
) -> tuple[str, int]:
    if state is None:
        return "unbound", 0
    expected_root = os.fspath(root.resolve())
    if (
        state.get("root") != expected_root
        or state.get("configured_session_id") != configured_session_id
    ):
        raise ValueError("agent session state ownership does not match route")
    # PR #7 state files did not carry an explicit state or generation.
    if "state" not in state and isinstance(state.get("codex_session_id"), str):
        return "bound", 0
    route_state = state.get("state")
    generation = state.get("generation")
    if (
        route_state not in {"bound", "closed"}
        or isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation < 0
    ):
        raise ValueError("invalid agent session generation")
    return str(route_state), generation


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


def _load_bounded_object(path: Path, *, maximum: int = 262_144) -> dict[str, object] | None:
    try:
        if path.stat().st_size > maximum:
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _session_lease_path(directory: Path, root: Path, session_id: str) -> Path:
    identity = f"{root.resolve()}\0{session_id}".encode()
    spool = directory.expanduser().resolve()
    return spool / "sessions" / f"{hashlib.sha256(identity).hexdigest()}.json"


def _producer_is_active(root: Path) -> bool:
    if fcntl is None:
        return False
    lease = _global_root_lease_directory() / (
        f"{hashlib.sha256(os.fspath(root.resolve()).encode()).hexdigest()}.lock"
    )
    if not lease.exists():
        return False
    stream = lease.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        return False
    finally:
        stream.close()


def _session_is_owned(
    directory: Path,
    *,
    root: Path,
    session_id: str,
    agent_session_id: str,
) -> bool:
    with session_route_lock(
        directory,
        root=root,
        configured_session_id=session_id,
        exclusive=False,
    ):
        state = read_session_state(
            directory, root=root, configured_session_id=session_id
        )
        try:
            route_state, _ = _state_generation(
                state, root=root, configured_session_id=session_id
            )
        except ValueError:
            return False
        return (
            route_state == "bound"
            and state is not None
            and state.get("codex_session_id") == agent_session_id
        )


def _reviewed_files_are_current(root: Path, reviewed_files: Sequence[ReviewedFile]) -> bool:
    if not reviewed_files:
        return False
    for item in reviewed_files:
        try:
            digest = _sha256_inside_root(
                root.resolve(), item.path, max_bytes=item.size
            )
        except OSError:
            return False
        if digest != item.sha256:
            return False
    return True


def _raw_hint_files_are_current(root: Path, value: object) -> bool:
    if not isinstance(value, list) or not 0 < len(value) <= MAX_REVIEWED_FILES:
        return False
    reviewed: list[ReviewedFile] = []
    for item in value:
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "sha256", "size"}
            or not isinstance(item["path"], str)
            or not isinstance(item["sha256"], str)
            or len(item["sha256"]) != 64
            or any(
                character not in "0123456789abcdef" for character in item["sha256"]
            )
            or isinstance(item["size"], bool)
            or not isinstance(item["size"], int)
            or not 0 <= item["size"] <= MAX_REVIEWED_FILE_BYTES
        ):
            return False
        try:
            path = _normalize_finding_path(
                item["path"], root.resolve(), {item["path"]}
            )
        except ReviewValidationError:
            return False
        reviewed.append(ReviewedFile(path, item["sha256"], item["size"]))
    return _reviewed_files_are_current(root, reviewed)


def publish_flush_hint(
    directory: Path,
    *,
    root: Path,
    session_id: str,
    agent_session_id: str,
    ttl_seconds: float = DEFAULT_FLUSH_HINT_TTL_SECONDS,
    reviewed_files: Sequence[ReviewedFile] = (),
) -> Path:
    """Publish a bounded, session-owned request to flush the current edit batch."""
    if (
        not 0 < ttl_seconds <= 300
        or not _producer_is_active(root)
        or not _session_is_owned(
            directory,
            root=root,
            session_id=session_id,
            agent_session_id=agent_session_id,
        )
    ):
        raise ValueError("flush hint does not match an active watcher and agent session")
    hints = _ensure_private_directory(directory.expanduser().resolve() / "flush-hints")
    if not reviewed_files or len(reviewed_files) > MAX_REVIEWED_FILES:
        raise ValueError("flush hint requires bounded reviewed file metadata")
    normalized_files: list[ReviewedFile] = []
    for item in reviewed_files:
        path = _normalize_finding_path(item.path, root.resolve(), {item.path})
        if (
            len(item.sha256) != 64
            or any(character not in "0123456789abcdef" for character in item.sha256)
            or not 0 <= item.size <= MAX_REVIEWED_FILE_BYTES
        ):
            raise ValueError("flush hint contains invalid reviewed file metadata")
        normalized_files.append(ReviewedFile(path, item.sha256, item.size))
    created_at = time.time()
    destination = hints / f"{time.time_ns()}-{uuid.uuid4()}.json"
    _atomic_private_json(
        destination,
        {
            "version": 1,
            "root": os.fspath(root.resolve()),
            "session_id": session_id,
            "agent_session_id": agent_session_id,
            "created_at": created_at,
            "expires_at": created_at + ttl_seconds,
            "reviewed_files": [asdict(item) for item in normalized_files],
        },
    )
    return destination


def request_flush_hint(
    directory: Path,
    *,
    root: Path,
    session_id: str,
    agent_session_id: str,
    ttl_seconds: float = 10.0,
) -> Path:
    """Ask the watcher to flush edits already hinted by this exact session."""
    if (
        not 0 < ttl_seconds <= 10
        or not _producer_is_active(root)
        or not _session_is_owned(
            directory,
            root=root,
            session_id=session_id,
            agent_session_id=agent_session_id,
        )
    ):
        raise ValueError(
            "flush request does not match an active watcher and agent session"
        )
    requests = _ensure_private_directory(
        directory.expanduser().resolve() / "flush-requests"
    )
    created_at = time.time()
    destination = requests / f"{time.time_ns()}-{uuid.uuid4()}.json"
    _atomic_private_json(
        destination,
        {
            "version": 1,
            "root": os.fspath(root.resolve()),
            "session_id": session_id,
            "agent_session_id": agent_session_id,
            "created_at": created_at,
            "expires_at": created_at + ttl_seconds,
        },
    )
    return destination


def consume_flush_hint(
    directory: Path,
    *,
    root: Path,
    session_id: str,
    changed_paths: Sequence[Path] | None = None,
    quiet_seconds: float = 0.0,
    max_age_seconds: float = DEFAULT_AGENT_EDIT_MAX_AGE_SECONDS,
    force_ready: bool = False,
) -> FlushHint | None:
    """Consume one quiet or max-aged logical edit group for this exact route."""
    if not 0 <= quiet_seconds <= 10 or not 0 < max_age_seconds <= 30:
        raise ValueError("invalid agent edit coalescing window")
    hints = _ensure_private_directory(directory.expanduser().resolve() / "flush-hints")
    requests = _ensure_private_directory(
        directory.expanduser().resolve() / "flush-requests"
    )
    expected_root = os.fspath(root.resolve())
    changed_relative: set[str] | None = None
    if changed_paths is not None:
        changed_relative = set()
        for changed_path in changed_paths:
            try:
                changed_relative.add(
                    changed_path.resolve(strict=False)
                    .relative_to(root.resolve())
                    .as_posix()
                )
            except ValueError:
                continue
    now = time.time()
    force_request_paths: list[Path] = []
    force_requested_at: float | None = None
    for request_path in requests.glob("*.json"):
        value = _load_bounded_object(request_path)
        created_at = value.get("created_at") if value is not None else None
        expires_at = value.get("expires_at") if value is not None else None
        valid = (
            value is not None
            and set(value) == {
                "version",
                "root",
                "session_id",
                "agent_session_id",
                "created_at",
                "expires_at",
            }
            and value.get("version") == 1
            and value.get("root") == expected_root
            and value.get("session_id") == session_id
            and isinstance(value.get("agent_session_id"), str)
            and isinstance(created_at, (int, float))
            and not isinstance(created_at, bool)
            and isinstance(expires_at, (int, float))
            and not isinstance(expires_at, bool)
            and float(created_at) <= now < float(expires_at)
            and 0 < float(expires_at) - float(created_at) <= 10
            and _session_is_owned(
                directory,
                root=root,
                session_id=session_id,
                agent_session_id=str(value.get("agent_session_id")),
            )
        )
        if not valid:
            request_path.unlink(missing_ok=True)
            continue
        force_request_paths.append(request_path)
        requested_at = float(created_at)
        force_requested_at = (
            requested_at
            if force_requested_at is None
            else max(force_requested_at, requested_at)
        )

    candidates: list[tuple[float, Path]] = []
    for path in hints.glob("*.json"):
        try:
            candidates.append((path.stat().st_mtime, path))
        except FileNotFoundError:
            continue
    valid_hints: list[FlushHint] = []
    for _, path in sorted(candidates):
        value = _load_bounded_object(path)
        valid_shape = value is not None and set(value) == {
            "version",
            "root",
            "session_id",
            "agent_session_id",
            "created_at",
            "expires_at",
            "reviewed_files",
        }
        if not valid_shape:
            path.unlink(missing_ok=True)
            continue
        assert value is not None
        agent_session_id = value["agent_session_id"]
        created_at = value["created_at"]
        expires_at = value["expires_at"]
        raw_reviewed_files = value["reviewed_files"]
        if (
            not isinstance(raw_reviewed_files, list)
            or not raw_reviewed_files
            or len(raw_reviewed_files) > MAX_REVIEWED_FILES
        ):
            path.unlink(missing_ok=True)
            continue
        reviewed_files: list[ReviewedFile] = []
        for item in raw_reviewed_files:
            if (
                not isinstance(item, dict)
                or set(item) != {"path", "sha256", "size"}
                or not isinstance(item["path"], str)
                or not isinstance(item["sha256"], str)
                or isinstance(item["size"], bool)
                or not isinstance(item["size"], int)
            ):
                reviewed_files = []
                break
            try:
                normalized_path = _normalize_finding_path(
                    item["path"], root.resolve(), {item["path"]}
                )
            except ReviewValidationError:
                reviewed_files = []
                break
            if (
                len(item["sha256"]) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in item["sha256"]
                )
                or not 0 <= item["size"] <= MAX_REVIEWED_FILE_BYTES
            ):
                reviewed_files = []
                break
            reviewed_files.append(
                ReviewedFile(normalized_path, item["sha256"], item["size"])
            )
        valid = (
            value["version"] == 1
            and value["root"] == expected_root
            and value["session_id"] == session_id
            and isinstance(agent_session_id, str)
            and bool(agent_session_id)
            and isinstance(created_at, (int, float))
            and not isinstance(created_at, bool)
            and isinstance(expires_at, (int, float))
            and not isinstance(expires_at, bool)
            and float(created_at) <= now < float(expires_at)
            and float(expires_at) - float(created_at) <= 300
            and len(reviewed_files) == len(raw_reviewed_files)
            and _session_is_owned(
                directory,
                root=root,
                session_id=session_id,
                agent_session_id=agent_session_id,
            )
        )
        if valid:
            valid_hints.append(
                FlushHint(
                    agent_session_id,
                    float(created_at),
                    path,
                    tuple(reviewed_files),
                )
            )
        else:
            path.unlink(missing_ok=True)

    groups: list[list[FlushHint]] = []
    for hint in sorted(valid_hints, key=lambda item: (item.created_at, item.path.name)):
        if not groups:
            groups.append([hint])
            continue
        group = groups[-1]
        known_paths = {
            reviewed.path for member in group for reviewed in member.reviewed_files
        }
        added_paths = {item.path for item in hint.reviewed_files} - known_paths
        if (
            hint.agent_session_id != group[0].agent_session_id
            or hint.created_at - group[-1].created_at > quiet_seconds
            or len(known_paths) + len(added_paths) > MAX_REVIEWED_FILES
        ):
            groups.append([hint])
        else:
            group.append(hint)

    has_current_hint = False
    for group in groups:
        latest_by_path: dict[str, ReviewedFile] = {}
        for member in group:
            for reviewed in member.reviewed_files:
                latest_by_path[reviewed.path] = reviewed
        reviewed_by_path = {
            path: reviewed
            for path, reviewed in latest_by_path.items()
            if _reviewed_files_are_current(root, (reviewed,))
        }
        active_members = [
            member
            for member in group
            if any(item.path in reviewed_by_path for item in member.reviewed_files)
        ]
        active_hint_paths = {member.path for member in active_members}
        inactive_members = [
            member for member in group if member.path not in active_hint_paths
        ]
        for member in inactive_members:
            member.path.unlink(missing_ok=True)
        if not active_members:
            continue
        has_current_hint = True
        if (
            changed_relative is not None
            and changed_relative.isdisjoint(reviewed_by_path)
        ):
            continue
        forced = force_ready or (
            force_requested_at is not None
            and active_members[-1].created_at <= force_requested_at
        )
        ready = (
            forced
            or now - active_members[-1].created_at >= quiet_seconds
            or now - active_members[0].created_at >= max_age_seconds
            or len(reviewed_by_path) == MAX_REVIEWED_FILES
        )
        if not ready:
            return None
        for request_path in force_request_paths:
            request_path.unlink(missing_ok=True)
        return FlushHint(
            active_members[0].agent_session_id,
            active_members[0].created_at,
            active_members[0].path,
            tuple(reviewed_by_path.values()),
            tuple(member.path for member in active_members),
        )
    if force_request_paths and not has_current_hint:
        for request_path in force_request_paths:
            request_path.unlink(missing_ok=True)
    return None


def matching_review_in_flight(
    directory: Path,
    *,
    root: Path,
    session_id: str,
    agent_session_id: str,
    validate_hint_files: bool = True,
) -> bool:
    """Return whether this real agent owns a live review, pruning stale markers."""
    in_flight = _ensure_private_directory(directory.expanduser().resolve() / "in-flight")
    expected_root = os.fspath(root.resolve())
    now = time.time()
    matched = False
    for path in in_flight.glob("*.json"):
        value = _load_bounded_object(path)
        if value is None or set(value) != {
            "version",
            "review_id",
            "root",
            "session_id",
            "agent_session_id",
            "started_at",
            "expires_at",
        }:
            path.unlink(missing_ok=True)
            continue
        expires_at = value["expires_at"]
        started_at = value["started_at"]
        live = (
            value["version"] == 1
            and isinstance(value["review_id"], str)
            and isinstance(started_at, (int, float))
            and not isinstance(started_at, bool)
            and isinstance(expires_at, (int, float))
            and not isinstance(expires_at, bool)
            and float(started_at) <= now < float(expires_at)
            and 0 < float(expires_at) - float(started_at) <= MAX_IN_FLIGHT_SECONDS
        )
        if not live:
            path.unlink(missing_ok=True)
            continue
        if (
            value["root"] == expected_root
            and value["session_id"] == session_id
            and value["agent_session_id"] == agent_session_id
            and _session_is_owned(
                directory,
                root=root,
                session_id=session_id,
                agent_session_id=agent_session_id,
            )
        ):
            matched = True
    if matched:
        return True
    hints = _ensure_private_directory(directory.expanduser().resolve() / "flush-hints")
    for path in hints.glob("*.json"):
        value = _load_bounded_object(path)
        if value is None or set(value) != {
            "version",
            "root",
            "session_id",
            "agent_session_id",
            "created_at",
            "expires_at",
            "reviewed_files",
        }:
            path.unlink(missing_ok=True)
            continue
        created_at = value.get("created_at")
        expires_at = value.get("expires_at")
        reviewed_files = value.get("reviewed_files")
        live = (
            value.get("version") == 1
            and value.get("root") == expected_root
            and value.get("session_id") == session_id
            and value.get("agent_session_id") == agent_session_id
            and isinstance(created_at, (int, float))
            and not isinstance(created_at, bool)
            and isinstance(expires_at, (int, float))
            and not isinstance(expires_at, bool)
            and float(created_at) <= now < float(expires_at)
            and float(expires_at) - float(created_at) <= 300
            and isinstance(reviewed_files, list)
            and 0 < len(reviewed_files) <= MAX_REVIEWED_FILES
            and (
                not validate_hint_files
                or _raw_hint_files_are_current(root, reviewed_files)
            )
            and _session_is_owned(
                directory,
                root=root,
                session_id=session_id,
                agent_session_id=agent_session_id,
            )
        )
        if live:
            return True
        path.unlink(missing_ok=True)
    return False


def retire_reviewed_flush_hints(
    directory: Path,
    *,
    root: Path,
    session_id: str,
    reviewed_files: Sequence[ReviewedFile],
) -> None:
    """Remove late hints whose post-edit digests were included in a review."""
    # The byte limit is a read bound, not source metadata. A direct hook and
    # watcher can use different safe bounds for the same sanitized content.
    reviewed = {(item.path, item.sha256) for item in reviewed_files}
    hints = directory.expanduser().resolve() / "flush-hints"
    for path in hints.glob("*.json"):
        value = _load_bounded_object(path)
        if value is None or value.get("root") != os.fspath(root.resolve()):
            continue
        if value.get("session_id") != session_id:
            continue
        raw_files = value.get("reviewed_files")
        if not isinstance(raw_files, list) or not raw_files:
            continue
        hinted: set[tuple[object, object]] = set()
        for item in raw_files:
            if not isinstance(item, dict):
                hinted.clear()
                break
            hinted.add((item.get("path"), item.get("sha256")))
        if hinted and hinted.issubset(reviewed):
            path.unlink(missing_ok=True)


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
        _ensure_private_directory(self.directory / "sessions")
        _ensure_private_directory(self.directory / "flush-hints")
        _ensure_private_directory(self.directory / "flush-requests")
        _ensure_private_directory(self.directory / "in-flight")
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

    def capture_session_generation(self) -> int | None:
        """Capture the route generation before provider work begins."""
        with session_route_lock(
            self.directory,
            root=self.root,
            configured_session_id=self.session_id,
            exclusive=False,
        ):
            state = read_session_state(
                self.directory,
                root=self.root,
                configured_session_id=self.session_id,
            )
            route_state, generation = _state_generation(
                state, root=self.root, configured_session_id=self.session_id
            )
            return None if route_state == "closed" else generation

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
                and previous.get("session_generation") == batch.session_generation
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
                "session_generation": batch.session_generation,
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
            "session_generation": batch.session_generation,
            "reviewed_files": [
                asdict(item) for item in batch.reviewed_files if item.path in cited_paths
            ],
            "findings": sorted(
                {f"{item.file}:{item.line}" for item in batch.findings}
            ),
        }
        encoded = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def consume_flush_hint(
        self,
        changed_paths: Sequence[Path] | None = None,
        *,
        quiet_seconds: float = 0.0,
        max_age_seconds: float = DEFAULT_AGENT_EDIT_MAX_AGE_SECONDS,
        force_ready: bool = False,
    ) -> FlushHint | None:
        return consume_flush_hint(
            self.directory,
            root=self.root,
            session_id=self.session_id,
            changed_paths=changed_paths,
            quiet_seconds=quiet_seconds,
            max_age_seconds=max_age_seconds,
            force_ready=force_ready,
        )

    def begin_review(
        self,
        *,
        agent_session_id: str | None,
        review_timeout: float,
        flush_hint: FlushHint | None = None,
    ) -> Path | None:
        """Publish an expiring marker before a provider review starts."""
        if agent_session_id is None or not _session_is_owned(
            self.directory,
            root=self.root,
            session_id=self.session_id,
            agent_session_id=agent_session_id,
        ):
            return None
        started_at = time.time()
        lifetime = min(MAX_IN_FLIGHT_SECONDS, max(5.0, review_timeout + 5.0))
        review_id = str(uuid.uuid4())
        marker = self.directory / "in-flight" / f"{review_id}.json"
        _atomic_private_json(
            marker,
            {
                "version": 1,
                "review_id": review_id,
                "root": os.fspath(self.root),
                "session_id": self.session_id,
                "agent_session_id": agent_session_id,
                "started_at": started_at,
                "expires_at": started_at + lifetime,
            },
        )
        if flush_hint is not None:
            for hint_path in flush_hint.paths:
                if hint_path.parent == self.directory / "flush-hints":
                    hint_path.unlink(missing_ok=True)
        return marker

    def finish_review(self, marker: Path | None) -> None:
        if marker is not None and marker.parent == self.directory / "in-flight":
            marker.unlink(missing_ok=True)

    def retire_reviewed_flush_hints(
        self, reviewed_files: Sequence[ReviewedFile]
    ) -> None:
        retire_reviewed_flush_hints(
            self.directory,
            root=self.root,
            session_id=self.session_id,
            reviewed_files=reviewed_files,
        )

    def publish(self, batch: ReviewBatch) -> bool:
        if batch.session_id != self.session_id or Path(batch.root).resolve() != self.root:
            raise ValueError("batch ownership does not match this spool")
        with session_route_lock(
            self.directory,
            root=self.root,
            configured_session_id=self.session_id,
            exclusive=False,
        ):
            state = read_session_state(
                self.directory,
                root=self.root,
                configured_session_id=self.session_id,
            )
            route_state, generation = _state_generation(
                state, root=self.root, configured_session_id=self.session_id
            )
            if route_state == "closed" or batch.session_generation != generation:
                return False
            self.retire_reviewed_flush_hints(batch.reviewed_files)
            batch = fresh_findings(batch)
            if not batch.findings:
                return False
            batch = self._apply_round_policy(batch)
            if not batch.findings:
                return False
            batch = replace(batch, published_at=time.time())
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
    base_fields = {
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
    generation_fields = {"session_generation"}
    lifecycle_fields = {
        "batch_flushed_at",
        "provider_started_at",
        "provider_completed_at",
        "published_at",
    }
    accepted_fields_without_redactions = (
        base_fields,
        base_fields | generation_fields,
        base_fields | lifecycle_fields,
        base_fields | generation_fields | lifecycle_fields,
    )
    accepted_fields = accepted_fields_without_redactions + tuple(
        fields | {"redactions"} for fields in accepted_fields_without_redactions
    )
    if set(payload) not in accepted_fields:
        raise ReviewValidationError("spooled batch has unexpected or missing fields")
    payload = payload.copy()
    payload.setdefault("session_generation", 0)
    if not lifecycle_fields.issubset(payload):
        legacy_created_at = payload["created_at"]
        legacy_provider_ms = payload["provider_ms"]
        if (
            isinstance(legacy_created_at, bool)
            or not isinstance(legacy_created_at, (int, float))
            or isinstance(legacy_provider_ms, bool)
            or not isinstance(legacy_provider_ms, (int, float))
        ):
            raise ReviewValidationError("invalid legacy review lifecycle timestamps")
        created_at = float(legacy_created_at)
        provider_ms = float(legacy_provider_ms)
        provider_started_at = created_at - provider_ms / 1_000
        payload.update(
            {
                "batch_flushed_at": provider_started_at,
                "provider_started_at": provider_started_at,
                "provider_completed_at": created_at,
                "published_at": created_at,
            }
        )
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
    session_generation = payload["session_generation"]
    if (
        isinstance(session_generation, bool)
        or not isinstance(session_generation, int)
        or session_generation < 0
    ):
        raise ReviewValidationError("invalid session generation")
    timestamps = [
        payload["first_observed_at"],
        payload["batch_flushed_at"],
        payload["provider_started_at"],
        payload["provider_completed_at"],
        payload["created_at"],
        payload["published_at"],
    ]
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in timestamps
    ) or any(
        float(left) > float(right)
        for left, right in zip(timestamps, timestamps[1:])
    ):
        raise ReviewValidationError("invalid review lifecycle timestamps")
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
        session_generation=session_generation,
        batch_flushed_at=float(payload["batch_flushed_at"]),
        provider_started_at=float(payload["provider_started_at"]),
        provider_completed_at=float(payload["provider_completed_at"]),
        redactions=_validated_redactions(payload.get("redactions")),
    )
    result = payload.copy()
    result["findings"] = [asdict(item) for item in normalized.findings]
    result["reviewed_files"] = [asdict(item) for item in reviewed]
    result["redactions"] = asdict(normalized.redactions)
    return result


def _validated_redactions(value: object) -> RedactionSummary:
    if value is None:
        return RedactionSummary()
    try:
        return redaction_summary_from_document(value)
    except ValueError as error:
        raise ReviewValidationError("invalid redaction metadata") from error


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
