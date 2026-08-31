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
import threading
import uuid
from dataclasses import dataclass, field as dataclass_field, replace
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from feedback import (
    CompositeSink,
    ConsoleSink,
    DEFAULT_AGENT_EDIT_MAX_AGE_SECONDS,
    DEFAULT_AGENT_EDIT_QUIET_SECONDS,
    DEFAULT_AGENT_TURN_MAX_AGE_SECONDS,
    FeedbackSink,
    FlushHint,
    MAX_PROVIDER_OUTPUT_BYTES,
    MAX_REVIEWED_FILES,
    ReviewBatch,
    ReviewValidationError,
    ReviewedFile,
    SpoolSink,
    _sha256_inside_root,
    fresh_findings,
    parse_review_output,
    read_bounded_beneath_root,
)
from model_runner import (
    ModelDocument,
    ModelRunConfig,
    ModelRunRequest,
    load_model_run_config,
    model_run_config_sha256,
    run_model,
)
from recommendation_grounding import extract_test_symbols, is_test_file
from review_output import DEFAULT_OUTPUT_MODE, OUTPUT_MODES, render_redaction_summary
from redaction import (
    REDACTED,
    RedactionSummary,
    RedactionSummaryBuilder,
    redact_path,
    redact_text,
)
from review_lifecycle import FindingLifecycleTracker, short_batch_id

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
DEFAULT_REVIEW_TIMEOUT_SECONDS = 180.0
OUTPUT_MODE_ENVIRONMENT_VARIABLE = "QUODET_OUTPUT"
PROMPT_REVISION = "quodet-review-v3"
REVIEW_SCHEMA_REVISION = "quodet-findings-v3"
OPERATIONAL_EVENT_SCHEMA_REVISION = "quodet-operational-event-v1"
DEFAULT_PROMPT = """Review the supplied changed files for real defects.

Analyze each supplied file as a separate file. For every candidate finding,
silently trace a concrete execution path from the cited code to an observable
failure. Check language and runtime semantics, cross-call mutable state,
identity and tenant scoping, concurrency and await boundaries, exception and
cancellation cleanup, and clock, unit, and resource-lifetime mismatches.
Confirm that the trigger is reachable using the concrete implementations,
values, ordering constraints, and scheduler behavior visible in the supplied
files. Do not assume a hypothetical subclass, override, caller, deployment, or
contract violation unless the supplied code makes that case reachable. For
concurrency findings, trace which task can mutate state before and after every
await, cancellation, and cleanup step; discard schedules contradicted by the
shown delays or control flow.

Return only negative findings that you are at least 0.95 confident are genuine
bugs, security vulnerabilities, data-loss risks, crashes, or operational
failures. Do not report praise, summaries, style preferences, speculative
concerns, low-confidence edge cases, or suggestions without a concrete defect.
Discard candidates that depend on assuming missing code is broken or that lack
a specific trigger and impact supported by the supplied files.
Treat the numeric confidence as a self-reported threshold claim, not evidence:
return 0.95 or higher only after the reachable failure path survives the checks
above.
Before returning a finding, verify that its title, explanation, failure type,
file, line, severity, and suggested fix are mutually consistent.
For every returned finding, make suggested_fix a concise recommended fix
grounded only in the supplied code. When the evidence supports it, name the
relevant function, class, branch, state transition, or other concrete code
element. Describe the smallest focused behavior change that removes the
demonstrated failure, explain why it fixes the cited execution path, and include
a narrow regression test or validation step. If a safe repair depends on code
that was not supplied, identify the exact missing evidence instead of inventing
architecture. Never claim that a test, function, contract, safeguard, or other
supporting artifact exists unless it appears in the supplied files. In
particular, when no test file was supplied, recommend adding a test rather than
preserving, extending, or modifying an "existing" test. When a supplied test is
relevant, name its supplied relative path or a test symbol visible in it. Do not
recommend unrelated refactors, dependency changes,
destructive commands, permission bypasses, disabled tests, or other ways around
existing safeguards. Treat the recommendation as untrusted review data that
requires independent verification. Never claim the recommendation is safe to
auto-apply.
Calibrate severity only from demonstrated impact: use critical for a direct
security-boundary bypass, irreversible data loss, or system-wide outage; high
for a major production failure; medium for bounded incorrect behavior or a
localized crash; and low for a limited defect. Do not infer blast radius from
missing deployment or usage context.
For finding.file, copy the value after each supplied `Original relative path:`
label exactly and verbatim; never add, remove, normalize, or guess a directory
prefix, and never substitute a basename or path alias. Use the most specific
line number available. Return an empty findings array if no finding meets this
threshold.
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
                            "Raw model-reported confidence from 0.95 to 1.0; "
                            "not a calibrated probability"
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


def response_schema_json(provider_paths: Sequence[str]) -> str:
    """Bind finding.file to the exact sanitized labels visible to the provider."""
    labels = tuple(provider_paths)
    if not labels or len(labels) != len(set(labels)):
        raise ReviewValidationError("provider-visible paths must be unique")
    schema = json.loads(REVIEW_SCHEMA_JSON)
    schema["properties"]["findings"]["items"]["properties"]["file"][
        "enum"
    ] = list(labels)
    return json.dumps(schema, separators=(",", ":"))


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
CONFIG_SECRET_FIELD_RE = re.compile(
    r"(?i)(?:^|[_-])(?:"
    r"api[_-]?key|secret(?:[_-]?key)?|access[_-]?key|private[_-]?key|"
    r"client[_-]?secret|signing[_-]?key|encryption[_-]?key|"
    r"auth(?:entication)?[_-]?token|access[_-]?token|refresh[_-]?token|"
    r"token|password|passwd|credential(?:s)?|connection[_-]?string|database[_-]?url"
    r")(?:$|[_-])"
)
LOWERCASE_SHA256_RE = re.compile(r"[0-9a-f]{64}")
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
class SanitizedBatch:
    attachments: tuple[Attachment, ...]
    snapshots: tuple[SourceSnapshot, ...]
    redactions: RedactionSummary


@dataclass(frozen=True)
class BoundedProcessResult:
    returncode: int
    stdout: str
    stderr: str
    output_exceeded: bool = False


class ReviewSuperseded(RuntimeError):
    """An obsolete provider process was stopped before it could publish."""


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
        "--agent-edit-quiet",
        type=positive_float,
        default=DEFAULT_AGENT_EDIT_QUIET_SECONDS,
        metavar="SECONDS",
        help=(
            "coalesce rapid direct agent edits before review (default: 0.25)"
        ),
    )
    parser.add_argument(
        "--agent-edit-max-age",
        type=positive_float,
        default=DEFAULT_AGENT_EDIT_MAX_AGE_SECONDS,
        metavar="SECONDS",
        help=(
            "safety cap for unidentified direct agent edits (default: 1)"
        ),
    )
    parser.add_argument(
        "--agent-turn-max-age",
        type=positive_float,
        default=DEFAULT_AGENT_TURN_MAX_AGE_SECONDS,
        metavar="SECONDS",
        help=(
            "safety cap for one identified agent turn (default: 3)"
        ),
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
        help="stop a stalled provider review after this long (default: 180)",
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
        "--model-run-config",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--model-run-config-sha256",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--benchmark-plan",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--benchmark-plan-sha256",
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
    parser.add_argument(
        "--agent-config",
        type=Path,
        help="validated route.json generated by `quodet init`",
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--output",
        choices=OUTPUT_MODES,
        help="terminal result format; defaults to QUODET_OUTPUT or human",
    )
    output_group.add_argument(
        "--json",
        action="store_const",
        const="json",
        dest="output",
        help="alias for --output json",
    )
    args = parser.parse_args(argv)
    configured_output = args.output or os.environ.get(
        OUTPUT_MODE_ENVIRONMENT_VARIABLE, DEFAULT_OUTPUT_MODE
    )
    if configured_output not in OUTPUT_MODES:
        parser.error(
            f"{OUTPUT_MODE_ENVIRONMENT_VARIABLE} must be one of: "
            f"{', '.join(OUTPUT_MODES)}"
        )
    args.output = configured_output
    return args


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


def validate_model_run_config_privacy(config: ModelRunConfig) -> None:
    values: dict[str, object] = {
        "model": config.model,
        "model_artifact": config.model_artifact,
        "provider": config.provider,
        "runtime": config.runtime,
        "runtime_version": config.runtime_version,
        "quantization": config.quantization,
        "pricing_source": config.pricing.source,
    }
    unsafe: list[str] = []
    for name, value in values.items():
        _, redactions = redact_sensitive_values(str(value))
        if redactions:
            unsafe.append(name)

    for collection_name, collection in (
        ("model_option", config.model_options),
        ("hardware", config.hardware),
    ):
        for key, value in collection.items():
            name = f"{collection_name}_{key}"
            if CONFIG_SECRET_FIELD_RE.search(key):
                unsafe.append(name)
                continue
            if collection_name == "hardware" and key == "runtime_artifact_sha256":
                if not isinstance(value, str) or LOWERCASE_SHA256_RE.fullmatch(
                    value
                ) is None:
                    unsafe.append(name)
                continue
            if isinstance(value, str):
                _, redactions = redact_sensitive_values(value)
                if redactions:
                    unsafe.append(name)
    if unsafe:
        raise ValueError(
            "model run config contains potential secrets in "
            f"{sorted(unsafe)}; use provider-managed credentials instead"
        )


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


def _is_inside_cargo_target(relative_path: Path, root: Path) -> bool:
    """Return whether a path is beneath a Cargo package's build target."""
    for index, part in enumerate(relative_path.parts):
        if part != "target":
            continue
        package_root = root.joinpath(*relative_path.parts[:index])
        if (package_root / "Cargo.toml").is_file():
            return True
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
    if root is not None and _is_inside_cargo_target(relative_path, root):
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


def redact_sensitive_values(text: str) -> tuple[str, int]:
    """Redact likely credentials, returning sanitized text and replacement count."""
    redacted = redact_text(text)
    return redacted.text, redacted.total


def redact_provider_response(
    text: str,
    *,
    provider_paths: Sequence[str],
) -> tuple[str, int]:
    """Redact provider JSON while preserving validated provider-visible file labels.

    Generated evaluation labels can resemble high-entropy credentials once a
    directory and filename are joined.  The labels have already crossed the
    provider boundary after component-aware path redaction, so preserve only
    exact ``finding.file`` values from that allowlist.  Every other response
    field remains subject to the normal secret detectors.
    """
    try:
        document = json.loads(text)
    except (json.JSONDecodeError, RecursionError):
        return redact_sensitive_values(text)
    if not isinstance(document, dict) or not isinstance(
        document.get("findings"), list
    ):
        return redact_sensitive_values(text)

    allowed_paths = set(provider_paths)
    approved_paths: set[str] = set()
    for finding in document["findings"]:
        if not isinstance(finding, dict):
            continue
        path = finding.get("file")
        if isinstance(path, str) and path in allowed_paths:
            approved_paths.add(path)

    shielded = text
    restore_tokens: dict[str, str] = {}
    for path in sorted(approved_paths):
        representations = {
            json.dumps(path),
            json.dumps(path, ensure_ascii=False),
            json.dumps(path).replace("/", r"\/"),
            json.dumps(path, ensure_ascii=False).replace("/", r"\/"),
        }
        found_representation = False
        for representation in sorted(representations):
            if representation not in shielded:
                continue
            found_representation = True
            index = len(restore_tokens)
            placeholder = json.dumps(f"quodet-safe-path-{index}")
            while placeholder in shielded or placeholder in restore_tokens:
                index += 1
                placeholder = json.dumps(f"quodet-safe-path-{index}")
            shielded = shielded.replace(representation, placeholder)
            restore_tokens[placeholder] = representation
        if not found_representation:
            return redact_sensitive_values(text)

    sanitized, count = redact_sensitive_values(shielded)
    for placeholder, representation in restore_tokens.items():
        sanitized = sanitized.replace(placeholder, representation)
    return sanitized, count


def redact_sensitive_path(path: Path) -> tuple[str, int]:
    """Redact path components without treating separators as token characters."""
    redacted = redact_path(path)
    return redacted.text, redacted.total


def _safe_path_label(path: Path) -> str:
    return redact_path(path).text


def provider_path_mapping(
    snapshots: Sequence[SourceSnapshot],
) -> dict[str, str]:
    """Map exact sent labels to local paths without retaining sensitive paths."""
    mapping: dict[str, str] = {}
    for snapshot in snapshots:
        label, redactions = redact_sensitive_path(snapshot.relative_path)
        if redactions:
            raise ReviewValidationError(
                "provider-visible path should have been excluded before mapping"
            )
        original = snapshot.relative_path.as_posix()
        if label in mapping and mapping[label] != original:
            raise ReviewValidationError("provider-visible path labels collide")
        mapping[label] = original
    return mapping


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
                f"Skipping {_safe_path_label(relative_path)}: file exceeds --max-bytes",
                file=sys.stderr,
            )
            continue

        if not is_utf8_text(root, relative_path, max_bytes=max_bytes):
            print(
                f"Skipping non-UTF-8 or unreadable file (cannot safely redact): "
                f"{_safe_path_label(relative_path)}",
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
                    f"Skipping {_safe_path_label(relative_path)}: exact snapshot "
                    "exceeds --max-bytes",
                    file=sys.stderr,
                )
                continue
            contents = source_bytes.decode("utf-8")
        except (OSError, UnicodeDecodeError) as error:
            print(
                f"Skipping {_safe_path_label(relative_path)}: could not snapshot: "
                f"{type(error).__name__}",
                file=sys.stderr,
            )
            continue
        if "\x00" in contents:
            continue
        snapshots.append(
            SourceSnapshot(
                path=attachment.path,
                relative_path=relative_path,
                contents=contents,
                # The retained digest is computed only from provider-safe text.
                # The read bound is the configured cap, not the exact source size.
                sha256=hashlib.sha256(redact_text(contents).text.encode()).hexdigest(),
                size=max_bytes,
            )
        )
    return snapshots


def sanitize_attachments(
    snapshots: Sequence[SourceSnapshot], *, destination: Path
) -> SanitizedBatch:
    """Create provider-safe copies; a source file is never used as an attachment."""
    sanitized_attachments: list[Attachment] = []
    sent_snapshots: list[SourceSnapshot] = []
    summary = RedactionSummaryBuilder()

    for index, snapshot in enumerate(snapshots, start=1):
        relative_path = snapshot.relative_path
        redacted_contents = redact_text(snapshot.contents)
        redacted_path = redact_path(relative_path)
        sanitized_relative_path = redacted_path.text

        # A sensitive filename cannot be retained in a review batch for later
        # freshness checks. Exclude it and report only its sanitized display path.
        if redacted_path.total:
            summary.add(
                redacted_path,
                file=sanitized_relative_path,
                disposition="excluded",
                line_available=False,
            )
            summary.add(
                redacted_contents,
                file=sanitized_relative_path,
                disposition="excluded",
            )
            continue

        provider_contents = (
            f"Original relative path: {sanitized_relative_path}\n\n"
            f"{redacted_contents.text}"
        )
        sanitized_path = destination / f"changed-file-{index:04d}.txt"
        try:
            sanitized_path.write_text(provider_contents, encoding="utf-8")
            sanitized_path.chmod(0o600)
        except OSError as error:
            print(
                f"Skipping {sanitized_relative_path}: could not stage safely: "
                f"{type(error).__name__}",
                file=sys.stderr,
            )
            summary.add(
                redacted_contents,
                file=sanitized_relative_path,
                disposition="excluded",
            )
            continue

        summary.add(
            redacted_contents,
            file=sanitized_relative_path,
            disposition="sent",
        )
        sanitized_attachments.append(
            Attachment(path=sanitized_path, media_type="text/plain")
        )
        sent_snapshots.append(snapshot)

    return SanitizedBatch(
        attachments=tuple(sanitized_attachments),
        snapshots=tuple(sent_snapshots),
        redactions=summary.build(),
    )


def build_llm_command(
    documents: Sequence[Attachment],
    *,
    model: str,
    prompt: str,
    log: bool,
    reasoning_effort: str | None,
    schema_json: str = REVIEW_SCHEMA_JSON,
) -> list[str]:
    command = [
        "llm",
        "prompt",
        "--model",
        model,
        "--no-stream",
        "--schema",
        schema_json,
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


def _report_failed_review_redactions(summary: RedactionSummary) -> None:
    rendered = render_redaction_summary(summary)
    if rendered:
        print(rendered, file=sys.stderr)


def run_bounded_command(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout: float,
    output_limit: int,
    cancel_event: threading.Event | None = None,
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
            if cancel_event is not None and cancel_event.is_set():
                process.kill()
                process.wait()
                raise ReviewSuperseded
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
    debounce_ms: float = 0.0,
    first_observed_at: float | None = None,
    session_generation: int | None = None,
    batch_flushed_at: float | None = None,
    review_coordinator: SpoolSink | None = None,
    agent_session_id: str | None = None,
    model_run_config: ModelRunConfig | None = None,
    lifecycle_tracker: FindingLifecycleTracker | None = None,
    cancel_event: threading.Event | None = None,
    publication_guard: Callable[[Callable[[], None]], bool] | None = None,
    review_marker_managed_externally: bool = False,
    coalesced_events: int = 0,
    snapshot_observer: Callable[[Sequence[SourceSnapshot]], None] | None = None,
    output_mode: str = DEFAULT_OUTPUT_MODE,
) -> ReviewBatch | None:
    if cancel_event is not None and cancel_event.is_set():
        return None
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
    if snapshot_observer is not None:
        snapshot_observer(snapshots)

    batch_id = str(uuid.uuid4())
    with tempfile.TemporaryDirectory(prefix="quodet-sanitized-") as temporary_directory:
        sanitized_batch = sanitize_attachments(
            snapshots,
            destination=Path(temporary_directory),
        )
        summary = RedactionSummaryBuilder()
        summary.extend(sanitized_batch.redactions)
        if not sanitized_batch.attachments:
            redactions = summary.build()
            if not redactions.total:
                return None
            if session_generation is None and review_coordinator is not None:
                session_generation = review_coordinator.capture_session_generation()
            batch = parse_review_output(
                '{"findings":[]}',
                root=root,
                reviewed_files=(),
                session_id=session_id,
                feedback_round=feedback_round,
                debounce_ms=debounce_ms,
                provider_ms=0.0,
                first_observed_at=first_observed_at,
                session_generation=session_generation,
                batch_flushed_at=batch_flushed_at,
                redactions=redactions,
                batch_id=batch_id,
            )
            published: list[ReviewBatch] = []

            def publish_excluded_summary() -> None:
                published_batch = replace(batch, published_at=time.time())
                (sink or ConsoleSink()).publish(published_batch)
                published.append(published_batch)

            if publication_guard is not None:
                publication_guard(publish_excluded_summary)
            elif cancel_event is None or not cancel_event.is_set():
                publish_excluded_summary()
            if review_coordinator is not None:
                review_coordinator.retire_reviewed_flush_hints(
                    tuple(
                        ReviewedFile(
                            snapshot.relative_path.as_posix(),
                            snapshot.sha256,
                            snapshot.size,
                        )
                        for snapshot in snapshots
                    )
                )
            return published[0] if published else None

        provider_path_map = provider_path_mapping(sanitized_batch.snapshots)
        schema_json = response_schema_json(tuple(provider_path_map))

        labels = [
            _safe_path_label(snapshot.relative_path)
            for snapshot in sanitized_batch.snapshots
        ]
        print(
            f"\n{short_batch_id(batch_id)} reviewing {len(labels)} changed "
            f"file(s): {', '.join(labels)}"
            + (
                f" ({coalesced_events} rewrite events coalesced)"
                if coalesced_events
                else ""
            ),
            file=sys.stderr,
            flush=True,
        )

        sanitized_prompt = redact_text(prompt)
        summary.add(
            sanitized_prompt,
            file=None,
            disposition="sent",
        )
        redactions = summary.build()

        command = (
            build_llm_command(
                sanitized_batch.attachments,
                model=model,
                prompt=sanitized_prompt.text,
                log=log,
                reasoning_effort=reasoning_effort,
                schema_json=schema_json,
            )
            if model_run_config is None
            else None
        )
        if session_generation is None and review_coordinator is not None:
            session_generation = review_coordinator.capture_session_generation()
        marker = (
            review_coordinator.begin_review(
                agent_session_id=agent_session_id,
                review_timeout=review_timeout,
            )
            if review_coordinator is not None and not review_marker_managed_externally
            else None
        )
        try:
            return _execute_review_command(
                command,
                snapshots=sanitized_batch.snapshots,
                root=root,
                review_timeout=review_timeout,
                evaluation_events=evaluation_events,
                sink=sink,
                session_id=session_id,
                feedback_round=feedback_round,
                debounce_ms=debounce_ms,
                first_observed_at=first_observed_at,
                session_generation=session_generation,
                batch_flushed_at=batch_flushed_at,
                model_run_config=model_run_config,
                sanitized_attachments=sanitized_batch.attachments,
                sanitized_prompt=sanitized_prompt.text,
                redactions=redactions,
                schema_json=schema_json,
                provider_path_map=provider_path_map,
                log=log,
                batch_id=batch_id,
                lifecycle_tracker=lifecycle_tracker,
                cancel_event=cancel_event,
                publication_guard=publication_guard,
                output_mode=output_mode,
            )
        finally:
            if review_coordinator is not None:
                review_coordinator.retire_reviewed_flush_hints(
                    tuple(
                        ReviewedFile(
                            snapshot.relative_path.as_posix(),
                            snapshot.sha256,
                            snapshot.size,
                        )
                        for snapshot in snapshots
                    )
                )
                if not review_marker_managed_externally:
                    review_coordinator.finish_review(marker)


def _execute_review_command(
    command: Sequence[str] | None,
    *,
    snapshots: Sequence[SourceSnapshot],
    root: Path,
    review_timeout: float,
    evaluation_events: bool,
    sink: FeedbackSink | None,
    session_id: str | None,
    feedback_round: int,
    debounce_ms: float,
    first_observed_at: float | None,
    session_generation: int | None,
    batch_flushed_at: float | None,
    model_run_config: ModelRunConfig | None,
    sanitized_attachments: Sequence[Attachment],
    sanitized_prompt: str,
    redactions: RedactionSummary,
    schema_json: str,
    provider_path_map: dict[str, str],
    log: bool,
    batch_id: str,
    lifecycle_tracker: FindingLifecycleTracker | None,
    cancel_event: threading.Event | None,
    publication_guard: Callable[[Callable[[], None]], bool] | None,
    output_mode: str,
) -> ReviewBatch | None:
    provider_started_at = time.time()
    provider_started = time.monotonic()
    model_result = None
    runtime_attestation = None
    try:
        if cancel_event is not None and cancel_event.is_set():
            raise ReviewSuperseded
        if model_run_config is not None:
            from evals.agent_changes.live_eval import attest_runtime

            runtime_attestation = attest_runtime(model_run_config)
            model_result = run_model(
                model_run_config,
                ModelRunRequest(
                    documents=tuple(
                        ModelDocument(document.path, document.media_type)
                        for document in sanitized_attachments
                    ),
                    prompt=sanitized_prompt,
                    schema_json=schema_json,
                    cwd=root,
                    log=log,
                ),
            )
            result = BoundedProcessResult(
                returncode=(
                    model_result.returncode
                    if model_result.returncode is not None
                    else 1
                ),
                stdout=model_result.stdout,
                stderr=model_result.stderr,
                output_exceeded=model_result.status == "output-limit",
            )
            if cancel_event is not None and cancel_event.is_set():
                raise ReviewSuperseded
        else:
            assert command is not None
            result = run_bounded_command(
                command,
                cwd=root,
                timeout=review_timeout,
                output_limit=MAX_PROVIDER_OUTPUT_BYTES,
                cancel_event=cancel_event,
            )
    except ReviewSuperseded:
        _report_failed_review_redactions(redactions)
        if evaluation_events:
            print(json.dumps({"quodet_evaluation_event": {
                "status": "superseded",
                "returncode": None,
                "raw_response": None,
                "stderr": None,
            }}), flush=True)
        elif output_mode == "json":
            _print_operational_event(
                "superseded",
                batch_id=batch_id,
                elapsed_ms=max(
                    0.0,
                    (time.time() - (first_observed_at or time.time())) * 1_000,
                ),
            )
        else:
            elapsed = max(0.0, time.time() - (first_observed_at or time.time()))
            print(
                f"{short_batch_id(batch_id)} superseded after {elapsed:.2f}s; "
                "obsolete provider result discarded",
                file=sys.stderr,
            )
        return None
    except subprocess.TimeoutExpired as error:
        _report_failed_review_redactions(redactions)
        if evaluation_events:
            safe_stdout, _ = redact_sensitive_values(
                _subprocess_output_text(error.stdout) or ""
            )
            safe_stderr, _ = redact_sensitive_values(
                _subprocess_output_text(error.stderr) or ""
            )
            print(json.dumps({"quodet_evaluation_event": {
                "status": "timeout",
                "returncode": None,
                "raw_response": safe_stdout or None,
                "stderr": safe_stderr or None,
            }}), flush=True)
        elif output_mode == "json":
            _print_operational_event(
                "timeout",
                batch_id=batch_id,
                queue_ms=max(
                    0.0,
                    (
                        provider_started_at
                        - (batch_flushed_at or provider_started_at)
                    )
                    * 1_000,
                ),
                provider_ms=(time.monotonic() - provider_started) * 1_000,
            )
        else:
            print(
                f"llm review timed out after {review_timeout:g} seconds",
                file=sys.stderr,
            )
            _print_failed_review(
                batch_id,
                first_observed_at,
                "provider timed out",
                queue_ms=max(
                    0.0,
                    (
                        provider_started_at
                        - (batch_flushed_at or provider_started_at)
                    )
                    * 1_000,
                ),
                provider_ms=(time.monotonic() - provider_started) * 1_000,
            )
        return None

    except (OSError, ValueError) as error:
        _report_failed_review_redactions(redactions)

        if evaluation_events:
            safe_error, _ = redact_sensitive_values(str(error))
            print(json.dumps({"quodet_evaluation_event": {
                "status": "provider-error",
                "returncode": None,
                "raw_response": None,
                "stderr": safe_error,
                "model_attempted": False,
            }}), flush=True)
        else:
            print(f"Could not run llm: {error}", file=sys.stderr)
            _print_failed_review(batch_id, first_observed_at, "provider could not start")
        return None
    provider_completed_at = time.time()
    provider_ms = (time.monotonic() - provider_started) * 1_000
    safe_stdout, _ = redact_provider_response(
        result.stdout,
        provider_paths=tuple(provider_path_map),
    )
    safe_stderr, _ = redact_sensitive_values(result.stderr)
    result = BoundedProcessResult(
        returncode=result.returncode,
        stdout=safe_stdout,
        stderr=safe_stderr,
        output_exceeded=result.output_exceeded,
    )
    model_result_payload = None
    if model_result is not None:
        model_result_payload = model_result.to_dict()
        model_result_payload["stdout"] = safe_stdout
        model_result_payload["stderr"] = safe_stderr
        if model_result.status != "success":
            _report_failed_review_redactions(redactions)
            if evaluation_events:
                print(json.dumps({"quodet_evaluation_event": {
                    "status": model_result.status,
                    "returncode": model_result.returncode,
                    "raw_response": safe_stdout,
                    "stderr": safe_stderr,
                    "model_run_result": model_result_payload,
                    "runtime_attestation": runtime_attestation,
                }}), flush=True)
            else:
                print(safe_stderr, file=sys.stderr)
            return None

    if result.output_exceeded:
        _report_failed_review_redactions(redactions)
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
            _print_failed_review(batch_id, first_observed_at, "provider output rejected")
        return None
    if evaluation_events:
        print(json.dumps({"quodet_evaluation_event": {
            "status": "success" if result.returncode == 0 else "provider-error",
            "returncode": result.returncode,
            "raw_response": result.stdout,
            "stderr": result.stderr,
            "model_run_result": (
                model_result_payload
            ),
            "runtime_attestation": runtime_attestation,
        }}), flush=True)
    if result.returncode != 0:
        _report_failed_review_redactions(redactions)
        if not evaluation_events:
            diagnostic = result.stderr.strip()
            if diagnostic:
                print(diagnostic[:2_000], file=sys.stderr)
            print(f"llm exited with status {result.returncode}", file=sys.stderr)
            _print_failed_review(
                batch_id,
                first_observed_at,
                f"provider exited with status {result.returncode}",
            )
        return None

    reviewed_paths = set(provider_path_map.values())
    provider_label_by_path = {
        original: label for label, original in provider_path_map.items()
    }
    reviewed_files = tuple(
        ReviewedFile(
            path=snapshot.relative_path.as_posix(),
            sha256=snapshot.sha256,
            size=snapshot.size,
        )
        for snapshot in snapshots
        if snapshot.relative_path.as_posix() in reviewed_paths
    )
    supplied_test_symbols = extract_test_symbols(
        tuple(
            redact_sensitive_values(snapshot.contents)[0]
            for snapshot in snapshots
            if snapshot.relative_path.as_posix() in reviewed_paths
            and is_test_file(
                provider_label_by_path[snapshot.relative_path.as_posix()]
            )
        )
    )
    try:
        batch = parse_review_output(
            result.stdout,
            root=root,
            reviewed_files=reviewed_files,
            provider_path_map=provider_path_map,
            supplied_test_symbols=supplied_test_symbols,
            session_id=session_id,
            feedback_round=feedback_round,
            debounce_ms=debounce_ms,
            provider_ms=provider_ms,
            first_observed_at=first_observed_at,
            session_generation=session_generation,
            batch_flushed_at=batch_flushed_at,
            provider_started_at=provider_started_at,
            provider_completed_at=provider_completed_at,
            redactions=redactions,
            batch_id=batch_id,
        )
    except ReviewValidationError as error:
        _report_failed_review_redactions(redactions)
        print(
            f"Rejected invalid llm response: {error}. "
            "Review discarded; no console or agent feedback was published.",
            file=sys.stderr,
        )
        _print_failed_review(batch_id, first_observed_at, "provider response invalid")
        return None

    if cancel_event is not None and cancel_event.is_set():
        if output_mode == "json":
            _print_operational_event("superseded", batch_id=batch_id)
        else:
            print(
                f"{short_batch_id(batch_id)} superseded; obsolete provider result "
                "discarded",
                file=sys.stderr,
            )
        return None

    fresh_batch = fresh_findings(batch)
    if len(fresh_batch.findings) != len(batch.findings):
        print(
            "Discarded stale finding(s) because source changed during review.",
            file=sys.stderr,
        )
    published: list[ReviewBatch] = []

    def publish() -> None:
        classified = (
            lifecycle_tracker.classify(fresh_batch)
            if lifecycle_tracker is not None
            else fresh_batch
        )
        published_batch = replace(classified, published_at=time.time())
        (sink or ConsoleSink()).publish(published_batch)
        published.append(published_batch)

    if publication_guard is not None:
        if not publication_guard(publish):
            if output_mode == "json":
                _print_operational_event("superseded", batch_id=batch_id)
            else:
                print(
                    f"{short_batch_id(batch_id)} superseded; obsolete provider "
                    "result discarded",
                    file=sys.stderr,
                )
            return None
    else:
        publish()
    return published[0]


def _print_failed_review(
    batch_id: str,
    first_observed_at: float | None,
    reason: str,
    *,
    queue_ms: float | None = None,
    provider_ms: float | None = None,
) -> None:
    started_at = first_observed_at or time.time()
    elapsed = max(0.0, time.time() - started_at)
    stages = ""
    if queue_ms is not None or provider_ms is not None:
        stages = (
            f" [queue {queue_ms or 0.0:.1f}ms, "
            f"provider {provider_ms or 0.0:.1f}ms]"
        )
    print(
        f"{short_batch_id(batch_id)} failed after {elapsed:.2f}s: "
        f"{reason}{stages}",
        file=sys.stderr,
    )


def _print_operational_event(status: str, **details: object) -> None:
    """Write one source-free machine diagnostic without polluting JSON stdout."""
    print(
        json.dumps(
            {
                "schema_version": OPERATIONAL_EVENT_SCHEMA_REVISION,
                "status": status,
                **details,
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        file=sys.stderr,
    )


class LatestPathQueue:
    """A thread-safe FIFO with one bounded entry per canonical path."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._pending: dict[Path, int] = {}

    def put(self, path: Path) -> None:
        with self._condition:
            self._pending[path] = self._pending.get(path, 0) + 1
            self._condition.notify()

    def get_with_count(self, timeout: float | None = None) -> tuple[Path, int]:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while not self._pending:
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise queue.Empty
                self._condition.wait(remaining)
            path = next(iter(self._pending))
            return path, self._pending.pop(path)

    def get(self, timeout: float | None = None) -> Path:
        return self.get_with_count(timeout)[0]

    def get_nowait(self) -> Path:
        return self.get(timeout=0)

    def empty(self) -> bool:
        with self._condition:
            return not self._pending

    def qsize(self) -> int:
        with self._condition:
            return len(self._pending)


def _get_changed_path(
    changes: queue.Queue[Path] | LatestPathQueue,
    timeout: float | None = None,
) -> tuple[Path, int]:
    if isinstance(changes, LatestPathQueue):
        return changes.get_with_count(timeout)
    return changes.get(timeout=timeout), 1


class ChangeHandler(FileSystemEventHandler):  # type: ignore[misc]
    """Translate relevant watchdog events into paths without doing blocking work."""

    def __init__(
        self,
        changes: queue.Queue[Path] | LatestPathQueue,
        *,
        root: Path,
        exclude_patterns: Sequence[str],
        observed_at: dict[Path, float] | None = None,
        observed_at_lock: threading.Lock | None = None,
    ) -> None:
        super().__init__()
        self.changes = changes
        self.root = root
        self.exclude_patterns = exclude_patterns
        self.observed_at = observed_at
        self.observed_at_lock = observed_at_lock or threading.Lock()

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
        changed_path = self.root / relative_path
        if self.observed_at is not None:
            with self.observed_at_lock:
                self.observed_at.setdefault(changed_path, time.time())
        self.changes.put(changed_path)


def next_batch(
    changes: queue.Queue[Path] | LatestPathQueue, debounce: float
) -> set[Path]:
    """Block for one change, then collect changes until the quiet period expires."""
    return next_triggered_batch(changes, debounce).paths


@dataclass(frozen=True)
class TriggeredBatch:
    paths: set[Path]
    flush_hint: FlushHint | None
    suppressed_paths: set[Path]
    event_count: int = 0


@dataclass
class ScheduledReview:
    """One immutable-at-dispatch review plus its cancellation boundary."""

    path_digests: dict[Path, str | None]
    first_observed_at: float
    batch_flushed_at: float
    agent_session_id: str | None
    session_generation: int | None
    marker: Path | None
    priority_paths: set[Path] = dataclass_field(default_factory=set)
    coalesced_events: int = 0
    cancel_event: threading.Event = dataclass_field(default_factory=threading.Event)
    superseded: bool = False
    finishing: bool = False

    @property
    def paths(self) -> tuple[Path, ...]:
        return tuple(sorted(self.path_digests))


class CoalescingReviewScheduler:
    """Run one provider call at a time and retain only one latest follow-up."""

    def __init__(
        self,
        run_review: Callable[[ScheduledReview], None],
        *,
        review_timeout: float,
        spool_sink: SpoolSink | None = None,
        refresh_digests: (
            Callable[[Sequence[Path]], Mapping[Path, str | None]] | None
        ) = None,
        output_mode: str = DEFAULT_OUTPUT_MODE,
    ) -> None:
        self._run_review = run_review
        self._review_timeout = review_timeout
        self._spool_sink = spool_sink
        self._refresh_digests = refresh_digests
        self._output_mode = output_mode
        self._condition = threading.Condition()
        self._pending: ScheduledReview | None = None
        self._active: ScheduledReview | None = None
        self._stopping = False
        self._thread = threading.Thread(
            target=self._worker_main,
            name="quodet-provider-worker",
            daemon=True,
        )
        self._thread.start()

    def submit(
        self,
        triggered: TriggeredBatch,
        *,
        path_digests: Mapping[Path, str | None],
        first_observed_at: float,
        batch_flushed_at: float,
    ) -> None:
        incoming = {
            path: path_digests.get(path)
            for path in triggered.paths
        }
        if not incoming:
            return
        duplicate_events = max(0, triggered.event_count - len(incoming))
        superseded_now = False
        priority_paths: set[Path] = set()
        marker: Path | None = None
        stale_markers: list[Path] = []
        agent_session_id = (
            triggered.flush_hint.agent_session_id
            if triggered.flush_hint is not None
            else None
        )
        session_generation = (
            self._spool_sink.capture_session_generation()
            if self._spool_sink is not None
            else None
        )

        with self._condition:
            if self._stopping:
                return
            active = (
                self._active
                if self._active is not None and not self._active.finishing
                else None
            )
            active_route_changed = active is not None and (
                active.session_generation != session_generation
                or (
                    agent_session_id is not None
                    and active.agent_session_id is not None
                    and active.agent_session_id != agent_session_id
                )
            )
            if active is not None:
                priority_paths.update(set(incoming) - set(active.path_digests))
                unchanged_overlap = {
                    path
                    for path, digest in incoming.items()
                    if path in active.path_digests
                    and not active_route_changed
                    and digest is not None
                    and digest == active.path_digests[path]
                }
                changed_overlap = {
                    path
                    for path, digest in incoming.items()
                    if path in active.path_digests
                    and (
                        active_route_changed
                        or digest is None
                        or active.path_digests[path] is None
                        or digest != active.path_digests[path]
                    )
                }
                duplicate_events += len(unchanged_overlap)
                for path in unchanged_overlap:
                    incoming.pop(path)
                if changed_overlap:
                    if not active.superseded:
                        active.superseded = True
                        active.cancel_event.set()
                        superseded_now = True
                    for path, digest in active.path_digests.items():
                        incoming.setdefault(path, digest)
                    priority_paths.update(
                        set(active.path_digests) - changed_overlap
                    )
                    if agent_session_id is None and not active_route_changed:
                        agent_session_id = active.agent_session_id

            if not incoming:
                if self._spool_sink is not None and agent_session_id is not None:
                    marker = self._spool_sink.begin_review(
                        agent_session_id=agent_session_id,
                        review_timeout=self._review_timeout,
                        flush_hint=triggered.flush_hint,
                    )
                    if active is not None and active.marker is None:
                        active.marker = marker
                        active.agent_session_id = agent_session_id
                        if active.session_generation is None:
                            active.session_generation = session_generation
                    else:
                        self._spool_sink.finish_review(marker)
                return

            if self._pending is None:
                self._pending = ScheduledReview(
                    path_digests={},
                    first_observed_at=first_observed_at,
                    batch_flushed_at=batch_flushed_at,
                    agent_session_id=agent_session_id,
                    session_generation=session_generation,
                    marker=None,
                )
            pending = self._pending
            for path, digest in incoming.items():
                if path in pending.path_digests and pending.path_digests[path] == digest:
                    duplicate_events += 1
                pending.path_digests[path] = digest
            pending.first_observed_at = min(
                pending.first_observed_at, first_observed_at
            )
            pending.batch_flushed_at = max(
                pending.batch_flushed_at, batch_flushed_at
            )
            pending.coalesced_events += duplicate_events
            pending.priority_paths.update(priority_paths & set(incoming))
            route_changed = (
                pending.session_generation != session_generation
                or (
                    agent_session_id is not None
                    and pending.agent_session_id is not None
                    and pending.agent_session_id != agent_session_id
                )
            )
            if route_changed:
                if pending.marker is not None:
                    stale_markers.append(pending.marker)
                    pending.marker = None
                pending.agent_session_id = agent_session_id
                pending.session_generation = session_generation
            else:
                if agent_session_id is not None:
                    pending.agent_session_id = agent_session_id
                if session_generation is not None:
                    pending.session_generation = session_generation

            if self._spool_sink is not None and agent_session_id is not None:
                marker = self._spool_sink.begin_review(
                    agent_session_id=agent_session_id,
                    # A scheduled follow-up can wait behind one bounded active call.
                    review_timeout=(2 * self._review_timeout) + 5.0,
                    flush_hint=triggered.flush_hint,
                )
                if pending.marker is None:
                    pending.marker = marker
                    marker = None
            self._condition.notify()

        if marker is not None and self._spool_sink is not None:
            self._spool_sink.finish_review(marker)
        if self._spool_sink is not None:
            for stale_marker in stale_markers:
                self._spool_sink.finish_review(stale_marker)
        if superseded_now:
            if self._output_mode == "json":
                _print_operational_event(
                    "superseded",
                    reviewed_file_count=len(incoming),
                    coalesced_events=duplicate_events,
                    follow_up_scheduled=True,
                )
            else:
                print(
                    "Superseded active review; cancelled obsolete work and queued "
                    f"one latest follow-up ({len(incoming)} files, "
                    f"{duplicate_events} coalesced events).",
                    file=sys.stderr,
                )

    def publish_if_current(
        self, work: ScheduledReview, publish: Callable[[], None]
    ) -> bool:
        """Serialize the final supersession check with event submission."""
        with self._condition:
            if (
                self._stopping
                or self._active is not work
                or work.cancel_event.is_set()
            ):
                return False
            publish()
            return True

    def record_snapshots(
        self, work: ScheduledReview, snapshots: Sequence[SourceSnapshot]
    ) -> None:
        """Bind active generations to the exact bytes handed to the provider."""
        with self._condition:
            if self._active is not work or work.cancel_event.is_set():
                return
            for snapshot in snapshots:
                work.path_digests[snapshot.path] = snapshot.sha256

    def state(self) -> tuple[int, int, bool]:
        """Return bounded source-free state for diagnostics and tests."""
        with self._condition:
            return (
                len(self._active.path_digests) if self._active is not None else 0,
                len(self._pending.path_digests) if self._pending is not None else 0,
                bool(self._active and self._active.superseded),
            )

    def wait_idle(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._active is not None or self._pending is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def close(self, *, join_timeout: float = 1.0) -> bool:
        markers: list[Path] = []
        with self._condition:
            self._stopping = True
            if self._active is not None:
                self._active.cancel_event.set()
                if self._active.marker is not None:
                    markers.append(self._active.marker)
                    self._active.marker = None
            if self._pending is not None:
                self._pending.cancel_event.set()
                if self._pending.marker is not None:
                    markers.append(self._pending.marker)
                self._pending = None
            self._condition.notify_all()
        if self._spool_sink is not None:
            for marker in markers:
                self._spool_sink.finish_review(marker)
        self._thread.join(join_timeout)
        return not self._thread.is_alive()

    def _take_pending_locked(self) -> ScheduledReview:
        pending = self._pending
        assert pending is not None
        priority = pending.priority_paths & set(pending.path_digests)
        if priority and priority != set(pending.path_digests):
            # Keep the route-owned marker on the remainder rather than doing
            # filesystem I/O from the provider worker to mint a replacement.
            # The marker already has enough lifetime for an active call plus
            # one follow-up, and it continues to represent the aggregate
            # in-flight review while the fair-priority slice runs first.
            work = ScheduledReview(
                path_digests={
                    path: pending.path_digests.pop(path)
                    for path in sorted(priority)
                },
                first_observed_at=pending.first_observed_at,
                batch_flushed_at=pending.batch_flushed_at,
                agent_session_id=pending.agent_session_id,
                session_generation=pending.session_generation,
                marker=None,
            )
            pending.priority_paths.difference_update(priority)
            return work
        self._pending = None
        return pending

    def _worker_main(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return
                work = self._take_pending_locked()
                pending = self._pending
                if (
                    pending is not None
                    and pending.marker is not None
                    and self._spool_sink is not None
                    and pending.agent_session_id is not None
                ):
                    try:
                        replacement_marker = self._spool_sink.begin_review(
                            agent_session_id=pending.agent_session_id,
                            # The renewed marker starts after the previous
                            # active review and covers the fair-priority slice
                            # plus the one retained remainder.
                            review_timeout=(2 * self._review_timeout) + 5.0,
                        )
                    except Exception as error:
                        safe_error, _ = redact_sensitive_values(str(error))
                        print(
                            "Could not renew in-flight review marker; "
                            "provider worker will continue: "
                            f"{type(error).__name__}: {safe_error[:500]}",
                            file=sys.stderr,
                        )
                    else:
                        if replacement_marker is not None:
                            work.marker = pending.marker
                            pending.marker = replacement_marker
                self._active = work
            assert work is not None
            try:
                if self._refresh_digests is not None:
                    refreshed = self._refresh_digests(work.paths)
                    with self._condition:
                        if self._active is work and not work.cancel_event.is_set():
                            work.path_digests.update(refreshed)
                self._run_review(work)
            except Exception as error:
                safe_error, _ = redact_sensitive_values(str(error))
                print(
                    f"Review worker failed safely: {type(error).__name__}: "
                    f"{safe_error[:500]}",
                    file=sys.stderr,
                )
            finally:
                with self._condition:
                    work.finishing = True
                    marker = work.marker
                    work.marker = None
                if marker is not None and self._spool_sink is not None:
                    self._spool_sink.finish_review(marker)
                with self._condition:
                    if self._active is work:
                        self._active = None
                    self._condition.notify_all()


def consume_first_observed_at(
    triggered: TriggeredBatch,
    observed_at: dict[Path, float],
    *,
    fallback: float,
) -> float:
    """Consume timing for this batch without dropping retained-path observations."""
    for path in triggered.suppressed_paths - triggered.paths:
        observed_at.pop(path, None)
    return min(
        (observed_at.pop(path, fallback) for path in triggered.paths),
        default=fallback,
    )


class MaterializedPathSuppression:
    """Suppress delayed watchdog events only while their reviewed bytes match."""

    def __init__(self, root: Path, *, ttl_seconds: float) -> None:
        self.root = root.resolve()
        self.ttl_seconds = ttl_seconds
        self.entries: dict[Path, tuple[str, int, float]] = {}

    def record(self, hint: FlushHint) -> None:
        self.prune()
        expires_at = time.monotonic() + self.ttl_seconds
        for item in hint.reviewed_files:
            self.entries[self.root / item.path] = (
                item.sha256,
                item.size,
                expires_at,
            )

    def prune(self) -> None:
        now = time.monotonic()
        self.entries = {
            path: entry
            for path, entry in self.entries.items()
            if entry[2] > now
        }

    def matches(self, path: Path) -> bool:
        canonical = path.resolve(strict=False)
        entry = self.entries.get(canonical)
        if entry is None:
            return False
        digest, size, expires_at = entry
        if time.monotonic() >= expires_at:
            self.entries.pop(canonical, None)
            return False
        try:
            relative = canonical.relative_to(self.root)
            current_digest = _sha256_inside_root(
                self.root, relative.as_posix(), max_bytes=size
            )
        except (OSError, ValueError):
            self.entries.pop(canonical, None)
            return False
        if current_digest == digest:
            return True
        self.entries.pop(canonical, None)
        return False


def next_triggered_batch(
    changes: queue.Queue[Path] | LatestPathQueue,
    debounce: float,
    *,
    hint_source: SpoolSink | None = None,
    suppression: MaterializedPathSuppression | None = None,
    agent_edit_quiet: float = DEFAULT_AGENT_EDIT_QUIET_SECONDS,
    agent_edit_max_age: float = DEFAULT_AGENT_EDIT_MAX_AGE_SECONDS,
    agent_turn_max_age: float = DEFAULT_AGENT_TURN_MAX_AGE_SECONDS,
) -> TriggeredBatch:
    """Collect a filesystem batch or a bounded group of direct agent edits."""
    suppressed_paths: set[Path] = set()
    event_count = 0

    def materialize_hint(hint: FlushHint, batch: set[Path]) -> TriggeredBatch:
        hinted_paths = {
            hint_source.root / Path(item.path)  # type: ignore[union-attr]
            for item in hint.reviewed_files
        }
        deferred_paths = batch - hinted_paths
        for path in sorted(deferred_paths):
            changes.put(path)
        if suppression is not None:
            suppression.record(hint)
            suppressed_paths.update(batch & hinted_paths)
        return TriggeredBatch(
            hinted_paths,
            hint,
            suppressed_paths,
            max(event_count, len(hinted_paths)),
        )

    while True:
        if hint_source is not None:
            hint = hint_source.consume_flush_hint(
                quiet_seconds=agent_edit_quiet,
                max_age_seconds=agent_edit_max_age,
                turn_max_age_seconds=agent_turn_max_age,
            )
            if hint is not None:
                return materialize_hint(hint, set())
            try:
                first, observed_events = _get_changed_path(changes, timeout=0.025)
            except queue.Empty:
                continue
        else:
            first, observed_events = _get_changed_path(changes)
        if suppression is not None and suppression.matches(first):
            suppressed_paths.add(first)
            event_count += observed_events
            continue
        batch = {first}
        event_count += observed_events
        break
    deadline = time.monotonic() + debounce
    maximum_deadline = time.monotonic() + (2 * debounce)

    while True:
        hint = (
            hint_source.consume_flush_hint(
                tuple(batch),
                quiet_seconds=agent_edit_quiet,
                max_age_seconds=agent_edit_max_age,
                turn_max_age_seconds=agent_turn_max_age,
            )
            if hint_source is not None
            else None
        )
        if hint is not None:
            return materialize_hint(hint, batch)
        remaining = min(deadline, maximum_deadline) - time.monotonic()
        if remaining <= 0:
            return TriggeredBatch(batch, None, suppressed_paths, event_count)
        try:
            wait = min(remaining, 0.025) if hint_source is not None else remaining
            path, observed_events = _get_changed_path(changes, timeout=wait)
            event_count += observed_events
            if suppression is not None and suppression.matches(path):
                suppressed_paths.add(path)
            else:
                batch.add(path)
                deadline = time.monotonic() + debounce
        except queue.Empty:
            if hint_source is None:
                return TriggeredBatch(batch, None, suppressed_paths, event_count)


def bounded_review_batches(paths: Iterable[Path]) -> Iterable[tuple[Path, ...]]:
    """Split a drained event set without dropping paths beyond the review cap."""
    ordered = sorted(set(paths))
    for index in range(0, len(ordered), MAX_REVIEWED_FILES):
        yield tuple(ordered[index : index + MAX_REVIEWED_FILES])


def current_path_digests(
    paths: Iterable[Path], *, root: Path, max_bytes: int
) -> dict[Path, str | None]:
    """Read source-safe digests used only to collapse duplicate notifications."""
    digests: dict[Path, str | None] = {}
    for path in set(paths):
        canonical = path.resolve(strict=False)
        relative = relative_to_root(canonical, root)
        if relative is None:
            continue
        try:
            digests[canonical] = _sha256_inside_root(
                root, relative.as_posix(), max_bytes=max_bytes
            )
        except (OSError, ValueError):
            # A deletion or unreadable transition is still a real generation.
            # Never equate two unknown generations as identical.
            digests[canonical] = None
    return digests


def _listed_model_entries(output: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in output.splitlines():
        _, separator, details = line.partition(": ")
        if not separator:
            continue
        model_id, _, aliases = details.partition(" (aliases: ")
        entries[model_id.strip()] = line
        if aliases.endswith(")"):
            for alias in aliases[:-1].split(","):
                if alias.strip():
                    entries[alias.strip()] = line
    return entries


def _listed_model_ids(output: str) -> set[str]:
    return set(_listed_model_entries(output))


def validate_runtime(path: Path, model: str, *, strict_model: bool = False) -> Path:
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
    if model not in _listed_model_ids(model_check.stdout):
        if strict_model:
            raise SystemExit(
                f"Configured benchmark model {model!r} was not found exactly in "
                "`llm models list`; refusing to invoke another model."
            )
        print(
            f"Warning: model {model!r} was not found by `llm models list`; "
            "the first review may fail.",
            file=sys.stderr,
        )
    return root


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv and raw_argv[0] in {"init", "status", "cleanup"}:
        from agent_integration import main as agent_main

        return agent_main(raw_argv)
    args = parse_args(raw_argv)
    if (
        args.agent_edit_quiet > 10
        or args.agent_edit_max_age > 30
        or args.agent_turn_max_age > 30
    ):
        raise SystemExit(
            "--agent-edit-quiet must be at most 10 seconds and "
            "agent edit age limits at most 30 seconds"
        )
    benchmark_arguments = (
        args.model_run_config,
        args.model_run_config_sha256,
        args.benchmark_plan,
        args.benchmark_plan_sha256,
    )
    if any(value is not None for value in benchmark_arguments) and not all(
        value is not None for value in benchmark_arguments
    ):
        raise SystemExit(
            "benchmark execution requires --model-run-config, "
            "--model-run-config-sha256, --benchmark-plan, and "
            "--benchmark-plan-sha256 together"
        )
    if args.agent_edit_quiet > args.agent_edit_max_age:
        raise SystemExit("--agent-edit-quiet must not exceed --agent-edit-max-age")
    if args.agent_edit_max_age > args.agent_turn_max_age:
        raise SystemExit(
            "--agent-edit-max-age must not exceed --agent-turn-max-age"
        )
    if args.debounce < args.agent_edit_quiet:
        raise SystemExit("--debounce must not be shorter than --agent-edit-quiet")
    if args.debounce < args.agent_turn_max_age:
        raise SystemExit("--debounce must not be shorter than --agent-turn-max-age")
    model_run_config = (
        load_model_run_config(args.model_run_config)
        if args.model_run_config is not None
        else None
    )
    if model_run_config is not None:
        from evals.agent_changes import benchmark

        try:
            benchmark_plan = benchmark.load_plan(args.benchmark_plan)
            if benchmark.plan_sha256(benchmark_plan) != args.benchmark_plan_sha256:
                raise ValueError(
                    "benchmark plan changed after parent approval"
                )
            benchmark.validate_model_run_config_against_plan(
                benchmark_plan, model_run_config,
            )
        except (OSError, ValueError) as error:
            raise SystemExit(str(error)) from error
        if (
            model_run_config_sha256(model_run_config)
            != args.model_run_config_sha256
        ):
            raise SystemExit(
                "model run config changed after benchmark approval; refusing to run"
            )
        try:
            validate_model_run_config_privacy(model_run_config)
        except ValueError as error:
            raise SystemExit(str(error)) from error
        args.model = model_run_config.model
        args.review_timeout = model_run_config.timeout_seconds
    root = validate_runtime(
        args.path, args.model, strict_model=model_run_config is not None
    )
    if args.agent_config is not None:
        from agent_integration import load_route, validate_watcher_route

        try:
            route = load_route(args.agent_config)
            args.spool_dir, args.session_id = validate_watcher_route(
                route,
                root=root,
                spool_dir=args.spool_dir,
                session_id=args.session_id,
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise SystemExit(f"Invalid agent route: {error}") from error
    if (args.spool_dir is None) != (args.session_id is None):
        raise SystemExit("--spool-dir and --session-id must be supplied together")
    sink: FeedbackSink = ConsoleSink(mode=args.output)
    spool_sink: SpoolSink | None = None
    if args.spool_dir is not None:
        spool_sink = SpoolSink(
            args.spool_dir, root=root, session_id=args.session_id
        )
        sink = CompositeSink((sink, spool_sink))
    changes = LatestPathQueue()
    observed_at: dict[Path, float] = {}
    observed_at_lock = threading.Lock()
    observer_class = PollingObserver if args.poll else Observer
    observer = observer_class()
    observer.schedule(
        ChangeHandler(
            changes,
            root=root,
            exclude_patterns=args.exclude,
            observed_at=observed_at,
            observed_at_lock=observed_at_lock,
        ),
        os.fspath(root),
        recursive=True,
    )
    observer.start()

    print(f"Watching {root}", file=sys.stderr)
    print(f"Model: {args.model}; debounce: {args.debounce:g}s", file=sys.stderr)
    suppression = MaterializedPathSuppression(
        root, ttl_seconds=min(5.0, max(1.0, args.debounce))
    )
    lifecycle_tracker = FindingLifecycleTracker()

    scheduler: CoalescingReviewScheduler

    def run_scheduled_review(work: ScheduledReview) -> None:
        measured_debounce_ms = max(
            0.0, (work.batch_flushed_at - work.first_observed_at) * 1_000
        )
        for review_batch in bounded_review_batches(work.paths):
            if work.cancel_event.is_set():
                break
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
                debounce_ms=measured_debounce_ms,
                first_observed_at=work.first_observed_at,
                session_generation=work.session_generation,
                batch_flushed_at=work.batch_flushed_at,
                review_coordinator=spool_sink,
                agent_session_id=work.agent_session_id,
                model_run_config=model_run_config,
                lifecycle_tracker=lifecycle_tracker,
                cancel_event=work.cancel_event,
                publication_guard=lambda publish: scheduler.publish_if_current(
                    work, publish
                ),
                review_marker_managed_externally=True,
                coalesced_events=work.coalesced_events,
                snapshot_observer=lambda snapshots: scheduler.record_snapshots(
                    work, snapshots
                ),
                output_mode=args.output,
            )

    scheduler = CoalescingReviewScheduler(
        run_scheduled_review,
        review_timeout=args.review_timeout,
        spool_sink=spool_sink,
        refresh_digests=lambda paths: current_path_digests(
            paths, root=root, max_bytes=args.max_bytes
        ),
        output_mode=args.output,
    )
    try:
        while True:
            triggered = next_triggered_batch(
                changes,
                args.debounce,
                hint_source=spool_sink,
                suppression=suppression,
                agent_edit_quiet=args.agent_edit_quiet,
                agent_edit_max_age=args.agent_edit_max_age,
                agent_turn_max_age=args.agent_turn_max_age,
            )
            event_batch = triggered.paths
            batch_flushed_at = time.time()
            with observed_at_lock:
                first_observed_at = consume_first_observed_at(
                    triggered,
                    observed_at,
                    fallback=batch_flushed_at,
                )
            scheduler.submit(
                triggered,
                path_digests=current_path_digests(
                    event_batch, root=root, max_bytes=args.max_bytes
                ),
                first_observed_at=first_observed_at,
                batch_flushed_at=batch_flushed_at,
            )
    except KeyboardInterrupt:
        print("\nStopping watcher.", file=sys.stderr)
    finally:
        worker_stopped = scheduler.close()
        observer.stop()
        observer.join()
        if spool_sink is not None and worker_stopped:
            spool_sink.close()
        elif not worker_stopped:
            print(
                "Provider shutdown is still pending; its result is superseded "
                "and cannot be published.",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
