"""Privacy-safe secret redaction and user-facing redaction metadata.

This module deliberately keeps detected values out of its metadata objects.
Only the sanitized text, detector category, source line, and an identifier
derived exclusively from a nearby key name can leave the redaction boundary.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import PurePath, PurePosixPath


REDACTED = "[REDACTED]"
MAX_REDACTION_NOTICES = 20
VALID_DISPOSITIONS = frozenset({"sent", "excluded"})
VALID_CATEGORIES = frozenset(
    {
        "assignment-key",
        "authorization-header",
        "high-entropy-value",
        "private-key",
        "provider-token",
        "query-secret",
        "url-credential",
    }
)
MAX_REDACTIONS_PER_BATCH = 1_000_000
MAX_REDACTION_PATH_LENGTH = 1_024

PRIVATE_KEY_BLOCK_RE = re.compile(
    r"-----BEGIN (?P<label>[A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?)-----.*?"
    r"-----END (?P=label)-----",
    re.DOTALL,
)
SENSITIVE_ASSIGNMENT_RE = re.compile(
    r"(?i)(?P<prefix>[\"']?(?P<identifier>"
    r"(?:[a-z][a-z0-9]*[_-])*"
    r"(?:api[_-]?key|secret(?:[_-]?key)?|access[_-]?key|private[_-]?key|"
    r"client[_-]?secret|signing[_-]?key|encryption[_-]?key|"
    r"auth(?:entication)?[_-]?token|access[_-]?token|refresh[_-]?token|"
    r"token|password|passwd|credential(?:s)?|connection[_-]?string|database[_-]?url)"
    r")[\"']?\s*(?:=|:)\s*)"
    r"(?P<value>\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|[^\s,;#}\]}]+)"
)
AUTHORIZATION_RE = re.compile(
    r"(?i)(?P<prefix>\b(?P<identifier>authorization)\b\s*(?::|=)\s*[\"']?"
    r"(?:bearer|basic)\s+)(?P<value>[^\s\"',;]+)"
)
URL_CREDENTIAL_RE = re.compile(
    r"(?i)(?P<prefix>\b[a-z][a-z0-9+.-]*://[^:/\s]+:)(?P<value>[^@\s/]+)(?=@)"
)
QUERY_SECRET_RE = re.compile(
    r"(?i)(?P<prefix>[?&](?P<identifier>"
    r"api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|"
    r"password|secret)=)(?P<value>[^&#\s]+)"
)
KNOWN_SECRET_PATTERNS = (
    (
        "provider-token",
        re.compile(r"(?<![A-Za-z0-9])AIza[0-9A-Za-z_-]{35}(?![A-Za-z0-9])"),
    ),
    (
        "provider-token",
        re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])"),
    ),
    (
        "provider-token",
        re.compile(
            r"(?<![A-Za-z0-9])gh[pousr]_[A-Za-z0-9]{20,}(?![A-Za-z0-9])"
        ),
    ),
    (
        "provider-token",
        re.compile(
            r"(?<![A-Za-z0-9])github_pat_[A-Za-z0-9_]{20,}(?![A-Za-z0-9])"
        ),
    ),
    (
        "provider-token",
        re.compile(
            r"(?<![A-Za-z0-9])sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}"
            r"(?![A-Za-z0-9_-])"
        ),
    ),
    (
        "provider-token",
        re.compile(
            r"(?<![A-Za-z0-9])xox[baprs]-[A-Za-z0-9-]{10,}"
            r"(?![A-Za-z0-9-])"
        ),
    ),
    (
        "provider-token",
        re.compile(
            r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\."
            r"[A-Za-z0-9_-]{8,}(?![A-Za-z0-9_-])"
        ),
    ),
)
HIGH_ENTROPY_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_+/=-])[A-Za-z0-9_+/=-]{32,}(?![A-Za-z0-9_+/=-])"
)


@dataclass(frozen=True)
class DetectedRedaction:
    """One safe detector result with no captured value or value-derived data."""

    line: int
    category: str
    masked_identifier: str | None = None


@dataclass(frozen=True)
class RedactedText:
    """Sanitized text plus bounded, value-free detector results."""

    text: str
    total: int
    detections: tuple[DetectedRedaction, ...]


@dataclass(frozen=True)
class RedactionNotice:
    """A bounded redaction hint safe for terminal and structured output."""

    file: str | None
    line: int | None
    category: str
    masked_identifier: str | None
    disposition: str

    def __post_init__(self) -> None:
        if self.file is not None and not _is_safe_notice_path(self.file):
            raise ValueError("unsafe redaction notice path")
        if self.line is not None and (
            isinstance(self.line, bool) or not isinstance(self.line, int) or self.line < 1
        ):
            raise ValueError("invalid redaction notice line")
        if self.category not in VALID_CATEGORIES:
            raise ValueError("invalid redaction category")
        if self.disposition not in VALID_DISPOSITIONS:
            raise ValueError("invalid redaction disposition")
        if self.masked_identifier is not None and not _is_safe_mask(
            self.masked_identifier
        ):
            raise ValueError("invalid masked identifier")


@dataclass(frozen=True)
class RedactionSummary:
    """Bounded redaction metadata for one review batch."""

    total: int = 0
    notices: tuple[RedactionNotice, ...] = ()
    omitted: int = 0

    def __post_init__(self) -> None:
        if (
            isinstance(self.total, bool)
            or not isinstance(self.total, int)
            or not 0 <= self.total <= MAX_REDACTIONS_PER_BATCH
            or len(self.notices) > MAX_REDACTION_NOTICES
            or isinstance(self.omitted, bool)
            or not isinstance(self.omitted, int)
            or self.omitted < 0
            or self.omitted != self.total - len(self.notices)
        ):
            raise ValueError("invalid redaction summary bounds")


class RedactionSummaryBuilder:
    """Accumulate only bounded, privacy-safe metadata across attachments."""

    def __init__(self, *, limit: int = MAX_REDACTION_NOTICES) -> None:
        if not 0 <= limit <= MAX_REDACTION_NOTICES:
            raise ValueError("invalid redaction notice limit")
        self._limit = limit
        self._total = 0
        self._notices: list[RedactionNotice] = []

    def add(
        self,
        redacted: RedactedText,
        *,
        file: str | None,
        disposition: str,
        line_available: bool = True,
    ) -> None:
        if disposition not in VALID_DISPOSITIONS:
            raise ValueError("invalid redaction disposition")
        self._total = min(
            MAX_REDACTIONS_PER_BATCH,
            self._total + redacted.total,
        )
        available = self._limit - len(self._notices)
        if available <= 0:
            return
        # Display metadata is best-effort. If a caller supplies a path that is
        # not independently safe to retain, omit the path instead of crashing
        # the long-running watcher while preserving the value-free notice.
        notice_file = file if file is None or _is_safe_notice_path(file) else None
        for detected in redacted.detections[:available]:
            self._notices.append(
                RedactionNotice(
                    file=notice_file,
                    line=detected.line if line_available else None,
                    category=detected.category,
                    masked_identifier=detected.masked_identifier,
                    disposition=disposition,
                )
            )

    def build(self) -> RedactionSummary:
        notices = tuple(self._notices)
        return RedactionSummary(
            total=self._total,
            notices=notices,
            omitted=max(0, self._total - len(notices)),
        )

    def extend(self, summary: RedactionSummary) -> None:
        """Add an already-bounded summary without expanding omitted records."""
        self._total = min(
            MAX_REDACTIONS_PER_BATCH,
            self._total + summary.total,
        )
        available = self._limit - len(self._notices)
        if available > 0:
            self._notices.extend(summary.notices[:available])


def mask_identifier(identifier: str) -> str | None:
    """Mask a nearby key name without consulting any detected value."""
    normalized = unicodedata.normalize("NFKC", identifier)
    safe = "".join(character.upper() for character in normalized if character.isalnum())
    if not safe:
        return None
    if len(safe) == 1:
        return f"{safe}…"
    if len(safe) <= 4:
        return f"{safe[0]}…{safe[-1]}"
    if len(safe) <= 7:
        return f"{safe[:2]}…{safe[-2:]}"
    return f"{safe[:4]}…{safe[-3:]}"


def redact_path(path: PurePath) -> RedactedText:
    """Sanitize path components while retaining only safe, bounded metadata."""
    sanitized_parts: list[str] = []
    detections: list[DetectedRedaction] = []
    total = 0
    for part in path.parts:
        redacted = redact_text(part)
        sanitized_parts.append(redacted.text)
        total += redacted.total
        available = MAX_REDACTION_NOTICES - len(detections)
        if available > 0:
            detections.extend(redacted.detections[:available])
    return RedactedText(
        text="/".join(sanitized_parts),
        total=total,
        detections=tuple(detections),
    )


def _entropy(value: str) -> float:
    frequencies = {character: value.count(character) for character in set(value)}
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


def redact_text(text: str) -> RedactedText:
    """Return sanitized text and bounded metadata that never stores values."""
    detections: list[DetectedRedaction] = []
    total = 0

    def record(
        source: str,
        match: re.Match[str],
        *,
        category: str,
        identifier: str | None = None,
    ) -> None:
        nonlocal total
        total += 1
        if len(detections) >= MAX_REDACTION_NOTICES:
            return
        detections.append(
            DetectedRedaction(
                line=source.count("\n", 0, match.start()) + 1,
                category=category,
                masked_identifier=mask_identifier(identifier) if identifier else None,
            )
        )

    current = text

    def replace_private_key(match: re.Match[str]) -> str:
        record(current, match, category="private-key")
        # Preserve line positions without preserving any value characters.
        return "[REDACTED PRIVATE KEY]" + "\n" * match.group(0).count("\n")

    current = PRIVATE_KEY_BLOCK_RE.sub(replace_private_key, current)

    def apply_value_pattern(pattern: re.Pattern[str], category: str) -> None:
        nonlocal current
        source = current

        def replace(match: re.Match[str]) -> str:
            identifier = match.groupdict().get("identifier")
            record(source, match, category=category, identifier=identifier)
            value = match.group("value")
            if len(value) >= 2 and value[0] in {'"', "'"} and value[-1] == value[0]:
                newline_padding = "\n" * value.count("\n")
                replacement = (
                    f"{value[0]}{REDACTED}{newline_padding}{value[0]}"
                )
            else:
                replacement = REDACTED
            return f"{match.group('prefix')}{replacement}"

        current = pattern.sub(replace, source)

    apply_value_pattern(SENSITIVE_ASSIGNMENT_RE, "assignment-key")
    apply_value_pattern(AUTHORIZATION_RE, "authorization-header")
    apply_value_pattern(URL_CREDENTIAL_RE, "url-credential")
    apply_value_pattern(QUERY_SECRET_RE, "query-secret")

    for category, pattern in KNOWN_SECRET_PATTERNS:
        source = current

        def replace_known(
            match: re.Match[str],
            *,
            _source: str = source,
            _category: str = category,
        ) -> str:
            record(_source, match, category=_category)
            return REDACTED

        current = pattern.sub(replace_known, source)

    source = current

    def replace_high_entropy(match: re.Match[str]) -> str:
        value = match.group(0)
        if not _looks_like_high_entropy_secret(value):
            return value
        record(source, match, category="high-entropy-value")
        return REDACTED

    current = HIGH_ENTROPY_TOKEN_RE.sub(replace_high_entropy, source)
    return RedactedText(text=current, total=total, detections=tuple(detections))


def redaction_summary_from_document(value: object) -> RedactionSummary:
    """Strictly validate safe metadata crossing a retained-artifact boundary."""
    if not isinstance(value, dict) or set(value) != {"total", "notices", "omitted"}:
        raise ValueError("invalid redaction summary")
    total = value["total"]
    omitted = value["omitted"]
    raw_notices = value["notices"]
    if (
        isinstance(total, bool)
        or not isinstance(total, int)
        or not 0 <= total <= MAX_REDACTIONS_PER_BATCH
        or isinstance(omitted, bool)
        or not isinstance(omitted, int)
        or not isinstance(raw_notices, list)
        or len(raw_notices) > MAX_REDACTION_NOTICES
        or omitted != total - len(raw_notices)
    ):
        raise ValueError("invalid redaction summary bounds")

    notices: list[RedactionNotice] = []
    expected_fields = {
        "file",
        "line",
        "category",
        "masked_identifier",
        "disposition",
    }
    for raw in raw_notices:
        if not isinstance(raw, dict) or set(raw) != expected_fields:
            raise ValueError("invalid redaction notice")
        file = raw["file"]
        if file is not None and (
            not isinstance(file, str) or not _is_safe_notice_path(file)
        ):
            raise ValueError("unsafe redaction notice path")
        line = raw["line"]
        if line is not None and (
            isinstance(line, bool) or not isinstance(line, int) or line < 1
        ):
            raise ValueError("invalid redaction notice line")
        category = raw["category"]
        if category not in VALID_CATEGORIES:
            raise ValueError("invalid redaction category")
        disposition = raw["disposition"]
        if disposition not in VALID_DISPOSITIONS:
            raise ValueError("invalid redaction disposition")
        masked = raw["masked_identifier"]
        if masked is not None:
            if not isinstance(masked, str) or not _is_safe_mask(masked):
                raise ValueError("invalid masked identifier")
        notices.append(
            RedactionNotice(
                file=file,
                line=line,
                category=category,
                masked_identifier=masked,
                disposition=disposition,
            )
        )
    return RedactionSummary(total=total, notices=tuple(notices), omitted=omitted)


def _is_safe_mask(value: str) -> bool:
    if len(value) > 8 or value.count("…") != 1:
        return False
    left, right = value.split("…")
    if not left or len(left) > 4 or len(right) > 3:
        return False
    return value == value.upper() and all(
        character.isalnum() for character in left + right
    )


def _is_safe_notice_path(value: str) -> bool:
    path = PurePosixPath(value)
    normalized = path.as_posix()
    return not (
        not value
        or len(value) > MAX_REDACTION_PATH_LENGTH
        or value.startswith(("/", "~"))
        or "\\" in value
        or normalized != value
        or ".." in path.parts
        # Path sanitization is component-based so separators cannot join
        # otherwise benign build-artifact names into one base64-like token.
        or any(redact_text(part).total for part in path.parts)
    )
