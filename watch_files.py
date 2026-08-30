#!/usr/bin/env python3
"""Watch a directory and ask an LLM to review batches of changed files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from feedback import (
    CompositeSink,
    ConsoleSink,
    FeedbackSink,
    MAX_PROVIDER_OUTPUT_BYTES,
    MAX_REVIEWED_FILES,
    ReviewBatch,
    ReviewValidationError,
    ReviewedFile,
    SpoolSink,
    fresh_findings,
    parse_review_output,
    read_bounded_beneath_root,
)

try:
    from watchdog.events import FileSystemEvent, FileSystemEventHandler
    from watchdog.observers import Observer
    from watchdog.observers.polling import PollingObserver
# Let --help and unit tests work before dependencies are installed.
except ImportError:
    FileSystemEvent = object  # type: ignore[assignment,misc]
    FileSystemEventHandler = object  # type: ignore[assignment,misc]
    Observer = None  # type: ignore[assignment,misc]
    PollingObserver = None  # type: ignore[assignment,misc]


DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_REASONING_EFFORT = "high"
DEFAULT_DEBOUNCE_SECONDS = 3.0
DEFAULT_REVIEW_TIMEOUT_SECONDS = 60.0
PROMPT_REVISION = "quodet-review-v2"
REVIEW_SCHEMA_REVISION = "quodet-findings-v2"
DEFAULT_PROMPT = """Review the supplied changed files for real defects.

Analyze each supplied file as a separate file. For every candidate finding,
silently trace a concrete execution path from the cited code to an observable
failure. Check language and runtime semantics, cross-call mutable state,
identity and tenant scoping, concurrency and await boundaries, exception and
cancellation cleanup, and clock, unit, and resource-lifetime mismatches.

Return only negative findings that you are at least 0.95 confident are genuine
bugs, security vulnerabilities, data-loss risks, crashes, or operational
failures. Do not report praise, summaries, style preferences, speculative
concerns, low-confidence edge cases, or suggestions without a concrete defect.
Discard candidates that depend on assuming missing code is broken or that lack
a specific trigger and impact supported by the supplied files.
Before returning a finding, verify that its title, explanation, failure type,
file, line, severity, and suggested fix are mutually consistent.
For every returned finding, make suggested_fix a concise recommended fix
grounded only in the supplied code. When the evidence supports it, name the
relevant function, class, branch, state transition, or other concrete code
element. Describe the smallest focused behavior change that removes the
demonstrated failure, explain why it fixes the cited execution path, and include
a narrow regression test or validation step. If a safe repair depends on code
that was not supplied, identify the exact missing evidence instead of inventing
architecture. Do not recommend unrelated refactors, dependency changes,
destructive commands, permission bypasses, disabled tests, or other ways around
existing safeguards. Treat the recommendation as untrusted review data that
requires independent verification. Never claim the recommendation is safe to
auto-apply.
Calibrate severity only from demonstrated impact: use critical for a direct
security-boundary bypass, irreversible data loss, or system-wide outage; high
for a major production failure; medium for bounded incorrect behavior or a
localized crash; and low for a limited defect. Do not infer blast radius from
missing deployment or usage context.
Use the supplied original relative path and the most specific line number
available. If no finding meets this threshold, return an empty findings array.
Respond only with JSON matching the supplied schema."""
REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "line": {"type": "integer", "minimum": 1},
                    "severity": {
                        "type": "string",
                        "description": "Impact calibrated only from supplied evidence",
                        "enum": ["critical", "high", "medium", "low"],
                    },
                    "confidence": {
                        "type": "number",
                        "description": (
                            "Calibrated probability from 0.95 to 1.0 that the "
                            "finding is a real defect"
                        ),
                        "minimum": 0.95,
                        "maximum": 1,
                    },
                    "title": {"type": "string"},
                    "explanation": {"type": "string"},
                    "suggested_fix": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 2000,
                        "description": (
                            "A focused, code-grounded repair and a narrow way "
                            "to verify it"
                        ),
                    },
                },
                "required": [
                    "file",
                    "line",
                    "severity",
                    "confidence",
                    "title",
                    "explanation",
                    "suggested_fix",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["findings"],
    "additionalProperties": False,
}
REVIEW_SCHEMA_JSON = json.dumps(REVIEW_SCHEMA, separators=(",", ":"))
REDACTED = "[REDACTED]"
DEFAULT_IGNORED_PARTS = frozenset(
    {
        ".git",
        ".direnv",
        ".eggs",
        ".gradle",
        ".hg",
        ".mypy_cache",
        ".nox",
        ".npm",
        ".pnpm-store",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".tox",
        ".venv",
        "__pypackages__",
        "__pycache__",
        "bower_components",
        "build",
        "dist",
        "dist-packages",
        "jspm_packages",
        "node_modules",
        "site-packages",
        "venv",
    }
)
DEFAULT_IGNORED_PART_SEQUENCES = (
    (".yarn", "cache"),
    (".yarn", "unplugged"),
    ("vendor", "bundle"),
    ("vendor", "cache"),
)
PRIVATE_KEY_BLOCK_RE = re.compile(
    r"-----BEGIN (?P<label>[A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?)-----.*?"
    r"-----END (?P=label)-----",
    re.DOTALL,
)
SENSITIVE_ASSIGNMENT_RE = re.compile(
    r"(?i)(?P<prefix>[\"']?(?:"
    r"api[_-]?key|secret(?:[_-]?key)?|access[_-]?key|private[_-]?key|"
    r"client[_-]?secret|signing[_-]?key|encryption[_-]?key|"
    r"auth(?:entication)?[_-]?token|access[_-]?token|refresh[_-]?token|"
    r"token|password|passwd|credential(?:s)?|connection[_-]?string|database[_-]?url"
    r")[\"']?\s*(?:=|:)\s*)"
    r"(?P<value>\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|[^\s,;#}\]}]+)"
)
AUTHORIZATION_RE = re.compile(
    r"(?i)(?P<prefix>\bauthorization\b\s*(?::|=)\s*[\"']?"
    r"(?:bearer|basic)\s+)(?P<value>[^\s\"',;]+)"
)
URL_CREDENTIAL_RE = re.compile(
    r"(?i)(?P<prefix>\b[a-z][a-z0-9+.-]*://[^:/\s]+:)(?P<value>[^@\s/]+)(?=@)"
)
QUERY_SECRET_RE = re.compile(
    r"(?i)(?P<prefix>[?&](?:api[_-]?key|access[_-]?token|auth[_-]?token|"
    r"client[_-]?secret|password|secret)=)(?P<value>[^&#\s]+)"
)
KNOWN_SECRET_PATTERNS = (
    re.compile(r"(?<![A-Za-z0-9])AIza[0-9A-Za-z_-]{35}(?![A-Za-z0-9])"),
    re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])"),
    re.compile(r"(?<![A-Za-z0-9])gh[pousr]_[A-Za-z0-9]{20,}(?![A-Za-z0-9])"),
    re.compile(r"(?<![A-Za-z0-9])github_pat_[A-Za-z0-9_]{20,}(?![A-Za-z0-9])"),
    re.compile(r"(?<![A-Za-z0-9])sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}"),
    re.compile(r"(?<![A-Za-z0-9])xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(
        r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\."
        r"[A-Za-z0-9_-]{8,}(?![A-Za-z0-9_-])"
    ),
)
HIGH_ENTROPY_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_+/=-])[A-Za-z0-9_+/=-]{32,}(?![A-Za-z0-9_+/=-])"
)


@dataclass(frozen=True)
class Attachment:
    path: Path
    media_type: str


@dataclass(frozen=True)
class SourceSnapshot:
    path: Path
    relative_path: Path
    contents: str
    sha256: str
    size: int


@dataclass(frozen=True)
class BoundedProcessResult:
    returncode: int
    stdout: str
    stderr: str
    output_exceeded: bool = False


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively watch PATH and send debounced batches of changed files "
            "to an LLM through the llm CLI."
        )
    )
    parser.add_argument("path", type=Path, help="directory to watch")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="llm model or alias")
    parser.add_argument(
        "--reasoning-effort",
        choices=("auto", "low", "medium", "high"),
        default="auto",
        help=(
            "reasoning effort; auto uses high for the default Luna model and "
            "leaves custom models unchanged"
        ),
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT,
                        help="review prompt")
    parser.add_argument(
        "--debounce",
        type=positive_float,
        default=DEFAULT_DEBOUNCE_SECONDS,
        metavar="SECONDS",
        help="wait this long after the last change before reviewing (default: 3)",
    )
    parser.add_argument(
        "--max-bytes",
        type=positive_int,
        default=2_000_000,
        metavar="BYTES",
        help="skip individual files larger than this (default: 2000000)",
    )
    parser.add_argument(
        "--review-timeout",
        type=positive_float,
        default=DEFAULT_REVIEW_TIMEOUT_SECONDS,
        metavar="SECONDS",
        help="stop a stalled provider review after this long (default: 60)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="GLOB",
        help="ignore a relative-path glob; repeat for multiple patterns",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="save prompts and responses in llm's local history",
    )
    parser.add_argument(
        "--poll",
        action="store_true",
        help="poll for changes when native filesystem events are unavailable",
    )
    parser.add_argument(
        "--evaluation-events",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--spool-dir",
        type=Path,
        help=(
            "also publish validated feedback to this private directory outside "
            "the watched tree"
        ),
    )
    parser.add_argument(
        "--session-id",
        help="explicit coding-agent session owner required with --spool-dir",
    )
    return parser.parse_args(argv)


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def resolve_reasoning_effort(model: str, requested: str) -> str | None:
    if requested != "auto":
        return requested
    if model == DEFAULT_MODEL:
        return DEFAULT_REASONING_EFFORT
    return None


def relative_to_root(path: Path, root: Path) -> Path | None:
    try:
        return path.resolve(strict=False).relative_to(root)
    except (OSError, ValueError):
        return None


def _contains_part_sequence(parts: Sequence[str], sequence: Sequence[str]) -> bool:
    width = len(sequence)
    return any(
        tuple(parts[index: index + width]) == tuple(sequence)
        for index in range(len(parts))
    )


def _is_inside_python_virtual_environment(relative_path: Path, root: Path) -> bool:
    candidate = (root / relative_path).parent
    while candidate == root or root in candidate.parents:
        if (candidate / "pyvenv.cfg").is_file():
            return True
        if candidate == root:
            break
        candidate = candidate.parent
    return False


def is_excluded(
    relative_path: Path,
    patterns: Sequence[str],
    *,
    root: Path | None = None,
) -> bool:
    if any(part in DEFAULT_IGNORED_PARTS for part in relative_path.parts):
        return True
    if any(
        _contains_part_sequence(relative_path.parts, sequence)
        for sequence in DEFAULT_IGNORED_PART_SEQUENCES
    ):
        return True
    if root is not None and _is_inside_python_virtual_environment(relative_path, root):
        return True

    return any(
        relative_path.match(pattern)
        or any(Path(part).match(pattern) for part in relative_path.parts)
        for pattern in patterns
    )


def is_utf8_text(root: Path, relative_path: Path, *, max_bytes: int) -> bool:
    try:
        raw = read_bounded_beneath_root(root, relative_path, max_bytes=max_bytes)
        if len(raw) > max_bytes:
            return False
        contents = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    return "\x00" not in contents


def _redact_match_value(match: re.Match[str]) -> str:
    value = match.group("value")
    if value.startswith(('"', "'")) and value.endswith(value[0]):
        replacement = f"{value[0]}{REDACTED}{value[0]}"
    else:
        replacement = REDACTED
    return f"{match.group('prefix')}{replacement}"


def _entropy(value: str) -> float:
    frequencies = {character: value.count(
        character) for character in set(value)}
    return -sum(
        (count / len(value)) * math.log2(count / len(value))
        for count in frequencies.values()
    )


def _looks_like_high_entropy_secret(value: str) -> bool:
    character_classes = sum(
        bool(re.search(pattern, value))
        for pattern in (r"[a-z]", r"[A-Z]", r"[0-9]", r"[_+/=-]")
    )
    return character_classes >= 2 and _entropy(value) >= 3.5


def redact_sensitive_values(text: str) -> tuple[str, int]:
    """Redact likely credentials, returning sanitized text and replacement count."""
    redacted, count = PRIVATE_KEY_BLOCK_RE.subn("[REDACTED PRIVATE KEY]", text)

    for pattern in (
        SENSITIVE_ASSIGNMENT_RE,
        AUTHORIZATION_RE,
        URL_CREDENTIAL_RE,
        QUERY_SECRET_RE,
    ):
        redacted, replacements = pattern.subn(_redact_match_value, redacted)
        count += replacements

    for pattern in KNOWN_SECRET_PATTERNS:
        redacted, replacements = pattern.subn(REDACTED, redacted)
        count += replacements

    def redact_high_entropy(match: re.Match[str]) -> str:
        nonlocal count
        value = match.group(0)
        if not _looks_like_high_entropy_secret(value):
            return value
        count += 1
        return REDACTED

    return HIGH_ENTROPY_TOKEN_RE.sub(redact_high_entropy, redacted), count


def redact_sensitive_path(path: Path) -> tuple[str, int]:
    """Redact path components without treating separators as token characters."""
    sanitized_parts: list[str] = []
    total_redactions = 0
    for part in path.parts:
        sanitized, redaction_count = redact_sensitive_values(part)
        sanitized_parts.append(sanitized)
        total_redactions += redaction_count
    return "/".join(sanitized_parts), total_redactions


def collect_attachments(
    paths: Iterable[Path],
    *,
    root: Path,
    exclude_patterns: Sequence[str],
    max_bytes: int,
) -> list[Attachment]:
    attachments: list[Attachment] = []
    ordered_paths = sorted(set(paths))
    for index, path in enumerate(ordered_paths):
        relative_path = relative_to_root(path, root)
        if relative_path is None or is_excluded(
            relative_path,
            exclude_patterns,
            root=root,
        ):
            continue
        resolved_path = root / relative_path

        try:
            if not resolved_path.is_file():
                continue
            size = resolved_path.stat().st_size
        except OSError:
            continue

        if size > max_bytes:
            print(
                f"Skipping {relative_path}: {size} bytes exceeds --max-bytes {max_bytes}",
                file=sys.stderr,
            )
            continue

        if not is_utf8_text(root, relative_path, max_bytes=max_bytes):
            print(
                f"Skipping non-UTF-8 or unreadable file (cannot safely redact): "
                f"{relative_path}",
                file=sys.stderr,
            )
            continue
        attachments.append(Attachment(
            path=resolved_path, media_type="text/plain"))
        if len(attachments) == MAX_REVIEWED_FILES:
            if index + 1 < len(ordered_paths):
                print(
                    f"Direct review batch capped at {MAX_REVIEWED_FILES} files.",
                    file=sys.stderr,
                )
            break
    return attachments


def snapshot_attachments(
    attachments: Sequence[Attachment], *, root: Path, max_bytes: int
) -> list[SourceSnapshot]:
    """Read immutable review inputs and bind them to their exact source bytes."""
    snapshots: list[SourceSnapshot] = []
    for attachment in attachments:
        relative_path = attachment.path.relative_to(root)
        try:
            source_bytes = read_bounded_beneath_root(
                root, relative_path, max_bytes=max_bytes
            )
            if len(source_bytes) > max_bytes:
                print(
                    f"Skipping {relative_path}: exact snapshot exceeds "
                    f"--max-bytes {max_bytes}",
                    file=sys.stderr,
                )
                continue
            contents = source_bytes.decode("utf-8")
        except (OSError, UnicodeDecodeError) as error:
            print(f"Skipping {relative_path}: could not snapshot: {error}", file=sys.stderr)
            continue
        if "\x00" in contents:
            continue
        snapshots.append(
            SourceSnapshot(
                path=attachment.path,
                relative_path=relative_path,
                contents=contents,
                sha256=hashlib.sha256(source_bytes).hexdigest(),
                size=len(source_bytes),
            )
        )
    return snapshots


def sanitize_attachments(
    snapshots: Sequence[SourceSnapshot], *, destination: Path
) -> tuple[list[Attachment], int]:
    """Create provider-safe copies; a source file is never used as an attachment."""
    sanitized_attachments: list[Attachment] = []
    total_redactions = 0

    for index, snapshot in enumerate(snapshots, start=1):
        relative_path = snapshot.relative_path
        sanitized, redaction_count = redact_sensitive_values(snapshot.contents)
        sanitized_relative_path, path_redactions = redact_sensitive_path(relative_path)
        provider_contents = (
            f"Original relative path: {sanitized_relative_path}\n\n{sanitized}"
        )
        sanitized_path = destination / f"changed-file-{index:04d}.txt"
        try:
            sanitized_path.write_text(provider_contents, encoding="utf-8")
            sanitized_path.chmod(0o600)
        except OSError as error:
            print(
                f"Skipping {relative_path}: could not stage safely: {error}", file=sys.stderr)
            continue

        total_redactions += redaction_count + path_redactions
        sanitized_attachments.append(
            Attachment(path=sanitized_path, media_type="text/plain")
        )

    return sanitized_attachments, total_redactions


def build_llm_command(
    documents: Sequence[Attachment],
    *,
    model: str,
    prompt: str,
    log: bool,
    reasoning_effort: str | None,
) -> list[str]:
    command = [
        "llm",
        "prompt",
        "--model",
        model,
        "--no-stream",
        "--schema",
        REVIEW_SCHEMA_JSON,
    ]
    if reasoning_effort is not None:
        command.extend(["--option", "reasoning_effort", reasoning_effort])
    command.append("--log" if log else "--no-log")
    for document in documents:
        command.extend(["--fragment", os.fspath(document.path)])
    command.append(prompt)
    return command


def _subprocess_output_text(value: str | bytes | None) -> str | None:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def run_bounded_command(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout: float,
    output_limit: int,
) -> BoundedProcessResult:
    """Run a command without buffering unbounded provider output in memory."""
    with (
        tempfile.TemporaryFile(mode="w+b") as stdout_file,
        tempfile.TemporaryFile(mode="w+b") as stderr_file,
    ):
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=stdout_file,
            stderr=stderr_file,
        )
        deadline = time.monotonic() + timeout
        output_exceeded = False
        while process.poll() is None:
            if (
                os.fstat(stdout_file.fileno()).st_size > output_limit
                or os.fstat(stderr_file.fileno()).st_size > output_limit
            ):
                output_exceeded = True
                process.kill()
                process.wait()
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                process.kill()
                process.wait()
                stdout_file.seek(0)
                stderr_file.seek(0)
                stdout = stdout_file.read(output_limit + 1).decode(
                    "utf-8", errors="replace"
                )
                stderr = stderr_file.read(output_limit + 1).decode(
                    "utf-8", errors="replace"
                )
                raise subprocess.TimeoutExpired(
                    command,
                    timeout,
                    output=stdout,
                    stderr=stderr,
                )
            try:
                process.wait(timeout=min(0.05, remaining))
            except subprocess.TimeoutExpired:
                pass

        if (
            os.fstat(stdout_file.fileno()).st_size > output_limit
            or os.fstat(stderr_file.fileno()).st_size > output_limit
        ):
            output_exceeded = True
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read(output_limit + 1).decode("utf-8", errors="replace")
        stderr = stderr_file.read(output_limit + 1).decode("utf-8", errors="replace")
        return BoundedProcessResult(
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
            output_exceeded=output_exceeded,
        )


def review_files(
    paths: Iterable[Path],
    *,
    root: Path,
    exclude_patterns: Sequence[str],
    max_bytes: int,
    model: str,
    prompt: str,
    log: bool,
    review_timeout: float,
    reasoning_effort: str | None,
    evaluation_events: bool = False,
    sink: FeedbackSink | None = None,
    session_id: str | None = None,
    feedback_round: int = 1,
) -> ReviewBatch | None:
    attachments = collect_attachments(
        paths,
        root=root,
        exclude_patterns=exclude_patterns,
        max_bytes=max_bytes,
    )
    if not attachments:
        return None

    snapshots = snapshot_attachments(attachments, root=root, max_bytes=max_bytes)
    if not snapshots:
        return None

    labels = [str(snapshot.relative_path) for snapshot in snapshots]
    print(
        f"\nReviewing {len(labels)} changed file(s): {', '.join(labels)}", flush=True)

    with tempfile.TemporaryDirectory(prefix="quodet-sanitized-") as temporary_directory:
        sanitized_attachments, redaction_count = sanitize_attachments(
            snapshots,
            destination=Path(temporary_directory),
        )
        if not sanitized_attachments:
            return None
        sanitized_prompt, prompt_redactions = redact_sensitive_values(prompt)
        redaction_count += prompt_redactions
        if redaction_count:
            print(
                f"Redacted {redaction_count} potential secret(s) before provider upload.",
                file=sys.stderr,
            )

        command = build_llm_command(
            sanitized_attachments,
            model=model,
            prompt=sanitized_prompt,
            log=log,
            reasoning_effort=reasoning_effort,
        )

        try:
            result = run_bounded_command(
                command,
                cwd=root,
                timeout=review_timeout,
                output_limit=MAX_PROVIDER_OUTPUT_BYTES,
            )
        except subprocess.TimeoutExpired as error:
            if evaluation_events:
                print(json.dumps({"quodet_evaluation_event": {
                    "status": "timeout",
                    "returncode": None,
                    "raw_response": _subprocess_output_text(error.stdout),
                    "stderr": _subprocess_output_text(error.stderr),
                }}), flush=True)
            else:
                print(
                    f"llm review timed out after {review_timeout:g} seconds",
                    file=sys.stderr,
                )
            return None
        except OSError as error:
            if evaluation_events:
                print(json.dumps({"quodet_evaluation_event": {
                    "status": "provider-error",
                    "returncode": None,
                    "raw_response": None,
                    "stderr": str(error),
                }}), flush=True)
            else:
                print(f"Could not run llm: {error}", file=sys.stderr)
            return None

    if result.output_exceeded:
        diagnostic = (
            f"Rejected llm response: output exceeded "
            f"{MAX_PROVIDER_OUTPUT_BYTES} bytes"
        )
        if evaluation_events:
            print(json.dumps({"quodet_evaluation_event": {
                "status": "provider-error",
                "returncode": result.returncode,
                "raw_response": result.stdout,
                "stderr": diagnostic,
            }}), flush=True)
        else:
            print(diagnostic, file=sys.stderr)
        return None
    if evaluation_events:
        print(json.dumps({"quodet_evaluation_event": {
            "status": "success" if result.returncode == 0 else "provider-error",
            "returncode": result.returncode,
            "raw_response": result.stdout,
            "stderr": result.stderr,
        }}), flush=True)
    if result.returncode != 0:
        if not evaluation_events:
            diagnostic = result.stderr.strip()
            if diagnostic:
                print(diagnostic[:2_000], file=sys.stderr)
            print(f"llm exited with status {result.returncode}", file=sys.stderr)
        return None

    reviewed_files = tuple(
        ReviewedFile(
            path=snapshot.relative_path.as_posix(),
            sha256=snapshot.sha256,
            size=snapshot.size,
        )
        for snapshot in snapshots
    )
    try:
        batch = parse_review_output(
            result.stdout,
            root=root,
            reviewed_files=reviewed_files,
            session_id=session_id,
            feedback_round=feedback_round,
        )
    except ReviewValidationError as error:
        print(f"Rejected invalid llm response: {error}", file=sys.stderr)
        return None

    fresh_batch = fresh_findings(batch)
    if len(fresh_batch.findings) != len(batch.findings):
        print(
            "Discarded stale finding(s) because source changed during review.",
            file=sys.stderr,
        )
    (sink or ConsoleSink()).publish(fresh_batch)
    return fresh_batch


class ChangeHandler(FileSystemEventHandler):  # type: ignore[misc]
    """Translate relevant watchdog events into paths without doing blocking work."""

    def __init__(
        self,
        changes: queue.Queue[Path],
        *,
        root: Path,
        exclude_patterns: Sequence[str],
    ) -> None:
        super().__init__()
        self.changes = changes
        self.root = root
        self.exclude_patterns = exclude_patterns

    def on_created(self, event: FileSystemEvent) -> None:
        self._enqueue(event)

    def on_modified(self, event: FileSystemEvent) -> None:
        self._enqueue(event)

    def on_moved(self, event: FileSystemEvent) -> None:
        self._enqueue(event, moved=True)

    def _enqueue(self, event: FileSystemEvent, *, moved: bool = False) -> None:
        if getattr(event, "is_directory", False):
            return
        source = getattr(event, "dest_path", None) if moved else None
        source = source or getattr(event, "src_path", None)
        if not source:
            return

        relative_path = relative_to_root(Path(source), self.root)
        if relative_path is None or is_excluded(
            relative_path,
            self.exclude_patterns,
            root=self.root,
        ):
            return
        self.changes.put(self.root / relative_path)


def next_batch(changes: queue.Queue[Path], debounce: float) -> set[Path]:
    """Block for one change, then collect changes until the quiet period expires."""
    batch = {changes.get()}
    deadline = time.monotonic() + debounce

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return batch
        try:
            batch.add(changes.get(timeout=remaining))
            deadline = time.monotonic() + debounce
        except queue.Empty:
            return batch


def bounded_review_batches(paths: Iterable[Path]) -> Iterable[tuple[Path, ...]]:
    """Split a drained event set without dropping paths beyond the review cap."""
    ordered = sorted(set(paths))
    for index in range(0, len(ordered), MAX_REVIEWED_FILES):
        yield tuple(ordered[index : index + MAX_REVIEWED_FILES])


def validate_runtime(path: Path, model: str) -> Path:
    root = path.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Watch path is not a directory: {root}")
    if Observer is None:
        raise SystemExit(
            "The watchdog package is required. Install dependencies with `uv sync` "
            "or `python3 -m pip install watchdog`."
        )
    if shutil.which("llm") is None:
        raise SystemExit("The llm CLI is not installed or is not on PATH.")

    model_check = subprocess.run(
        ["llm", "models", "list"],
        check=False,
        capture_output=True,
        text=True,
    )
    if model_check.returncode != 0:
        raise SystemExit(
            "Could not list llm models; check the llm installation.")
    if model not in model_check.stdout:
        print(
            f"Warning: model {model!r} was not found by `llm models list`; "
            "the first review may fail.",
            file=sys.stderr,
        )
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = validate_runtime(args.path, args.model)
    if (args.spool_dir is None) != (args.session_id is None):
        raise SystemExit("--spool-dir and --session-id must be supplied together")
    sink: FeedbackSink = ConsoleSink()
    spool_sink: SpoolSink | None = None
    if args.spool_dir is not None:
        spool_sink = SpoolSink(
            args.spool_dir, root=root, session_id=args.session_id
        )
        sink = CompositeSink(
            [
                sink,
                spool_sink,
            ]
        )
    changes: queue.Queue[Path] = queue.Queue()
    observer_class = PollingObserver if args.poll else Observer
    observer = observer_class()
    observer.schedule(
        ChangeHandler(changes, root=root, exclude_patterns=args.exclude),
        os.fspath(root),
        recursive=True,
    )
    observer.start()

    print(f"Watching {root}")
    print(f"Model: {args.model}; debounce: {args.debounce:g}s")
    try:
        while True:
            event_batch = next_batch(changes, args.debounce)
            for review_batch in bounded_review_batches(event_batch):
                review_files(
                    review_batch,
                    root=root,
                    exclude_patterns=args.exclude,
                    max_bytes=args.max_bytes,
                    model=args.model,
                    prompt=args.prompt,
                    log=args.log,
                    review_timeout=args.review_timeout,
                    reasoning_effort=resolve_reasoning_effort(
                        args.model, args.reasoning_effort
                    ),
                    evaluation_events=args.evaluation_events,
                    sink=sink,
                    session_id=args.session_id,
                )
    except KeyboardInterrupt:
        print("\nStopping watcher.")
    finally:
        observer.stop()
        observer.join()
        if spool_sink is not None:
            spool_sink.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
