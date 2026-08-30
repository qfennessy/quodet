"""Validate whether repair guidance invents tests outside a reviewed snapshot."""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Sequence


_TEST_PATH = re.compile(
    r"(?<![A-Za-z0-9_.-])((?:(?:[A-Za-z0-9_.-]+/)*__tests__/"
    r"[A-Za-z0-9_.-]+|(?:[A-Za-z0-9_.-]+/)*(?:test_[A-Za-z0-9_.-]+|"
    r"[A-Za-z0-9_.-]+_(?:test|spec)|[A-Za-z0-9_.-]+\.(?:spec|test)))\."
    r"(?:py|js|jsx|ts|tsx|rb|go|rs|java|kt))(?![A-Za-z0-9_.-])",
    re.IGNORECASE,
)
_EXISTING_TEST_CLAIM = re.compile(
    r"\b(?:existing|current|already[- ]present)\s+"
    r"(?:(?:[A-Za-z0-9_-]+)\s+){0,3}tests?\b|"
    r"\btests?\s+(?:already exists?|is present)\b",
    re.IGNORECASE,
)
_CLAUSE_BOUNDARY = re.compile(
    r"\.(?=\s+|$)|[;\n]+|,?\s+then\s+|"
    r"\s+and\s+(?=(?:add|create|write|introduce|"
    r"preserve|retain|extend|modify|update|keep|change|edit)\b)",
    re.IGNORECASE,
)
_TEST_WORD = re.compile(
    r"\btests?\b|(?<![A-Za-z0-9_])(?:test_[A-Za-z0-9_]+|"
    r"[A-Za-z0-9_]+_test)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_TEST_SYMBOL = re.compile(
    r"(?<![A-Za-z0-9_])(?:test_[A-Za-z0-9_]+|"
    r"[A-Za-z0-9_]+_(?:test|spec))(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_SPEC_SYMBOL = re.compile(
    r"(?<![A-Za-z0-9_])[A-Za-z0-9_]+_spec(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_MUTATION_VERB = re.compile(
    r"\b(?:preserve|retain|extend|modify|update|keep|change|edit)\b",
    re.IGNORECASE,
)
_CREATION_VERB = re.compile(
    r"\b(?:add|create|write|introduce)\b", re.IGNORECASE
)
_NEW_TEST_MODIFIERS = re.compile(
    r"\s*(?:(?:a|an)\s+)?"
    r"(?:(?:another|focused|integration|narrow|new|regression|unit)\s+)*$",
    re.IGNORECASE,
)
_PROPOSED_PATH_PREFIX = re.compile(
    r"\b(?:add|create|write|introduce)\s+"
    r"(?:(?:a|an)\s+)?(?:new\s+)?"
    r"(?:(?:(?:regression|unit|integration)\s+)?test(?:\s+file)?\s+"
    r"(?:(?:at|named|as)\s+)?)?$",
    re.IGNORECASE,
)


def is_test_file(path: str) -> bool:
    """Return whether a supplied relative path conventionally names a test."""
    candidate = PurePosixPath(path.replace("\\", "/"))
    name = candidate.name.lower()
    parts = {part.lower() for part in candidate.parts[:-1]}
    return (
        "test" in parts
        or "tests" in parts
        or "spec" in parts
        or "specs" in parts
        or "__tests__" in parts
        or name.startswith("test_")
        or any(
            marker in name
            for marker in ("_test.", "_spec.", ".test.", ".spec.")
        )
    )


def _clauses(value: str) -> list[str]:
    return [clause.strip() for clause in _CLAUSE_BOUNDARY.split(value) if clause.strip()]


def extract_test_symbols(sources: Sequence[str]) -> tuple[str, ...]:
    """Return test-style identifiers visibly present in supplied source text."""
    matches = {
        match.group(0)
        for source in sources
        for match in _TEST_SYMBOL.finditer(source)
    }
    return tuple(sorted(matches))


def _mutates_unsupplied_test(
    clause: str, *, supplied_test_symbols: set[str]
) -> bool:
    """Return whether the nearest intent verb targets a generic test mention."""
    visible_clause_symbols = {
        match.group(0)
        for match in _TEST_SYMBOL.finditer(clause)
        if match.group(0) in supplied_test_symbols
    }
    proposed_path_spans = [
        (match.start(), match.end())
        for match in _TEST_PATH.finditer(clause)
        if _path_is_proposed(clause, match.start())
    ]
    test_matches = list(_TEST_WORD.finditer(clause))
    visible_spec_symbol = any(
        _SPEC_SYMBOL.fullmatch(symbol) is not None
        for symbol in visible_clause_symbols
    )
    if test_matches or visible_spec_symbol:
        test_matches.extend(_SPEC_SYMBOL.finditer(clause))
        test_matches.sort(key=lambda match: match.start())
    for test_match in test_matches:
        if test_match.group(0) in supplied_test_symbols:
            continue
        if (
            test_match.group(0).lower() in {"test", "tests"}
            and visible_clause_symbols
        ):
            continue
        if any(
            start <= test_match.start() < end
            for start, end in proposed_path_spans
        ):
            continue
        preceding = clause[: test_match.start()]
        intents = [
            *((match.end(), "mutation") for match in _MUTATION_VERB.finditer(preceding)),
            *((match.end(), "creation") for match in _CREATION_VERB.finditer(preceding)),
        ]
        if not intents:
            continue
        intent_end, intent = max(intents)
        if intent == "mutation":
            return True
        if not (
            _NEW_TEST_MODIFIERS.fullmatch(preceding[intent_end:])
            or _PROPOSED_PATH_PREFIX.search(preceding)
        ):
            return True
    return False


def _path_is_proposed(clause: str, path_start: int) -> bool:
    """Recognize a missing path only when it is the object of creation intent."""
    return _PROPOSED_PATH_PREFIX.search(clause[:path_start]) is not None


def evaluate_recommendation(
    recommendation: str,
    *,
    supplied_files: Sequence[str],
    supplied_test_symbols: Sequence[str] = (),
) -> dict[str, object]:
    """Return machine-readable grounding failures, independent of bug detection."""
    normalized_files = {path.replace("\\", "/") for path in supplied_files}
    supplied_tests = sorted(path for path in normalized_files if is_test_file(path))
    visible_symbols = set(supplied_test_symbols)
    violations: list[dict[str, str]] = []
    clauses = _clauses(recommendation)
    path_references = [
        (
            clause_index,
            clause,
            path_match.group(1).replace("\\", "/"),
            _path_is_proposed(clause, path_match.start()),
        )
        for clause_index, clause in enumerate(clauses)
        for path_match in _TEST_PATH.finditer(clause)
    ]
    for clause_index, clause in enumerate(clauses):
        for path in supplied_tests:
            pattern = re.compile(
                rf"(?<![A-Za-z0-9_.-]){re.escape(path)}(?![A-Za-z0-9_.-])"
            )
            path_references.extend(
                (clause_index, clause, path, False)
                for _ in pattern.finditer(clause)
            )
    unsupported_existing_claim = any(
        _EXISTING_TEST_CLAIM.search(clause)
        and not any(
            reference_clause_index == clause_index and path in supplied_tests
            for reference_clause_index, _, path, _ in path_references
        )
        and not (
            {
                match.group(0)
                for match in _TEST_SYMBOL.finditer(clause)
            }
            & visible_symbols
        )
        for clause_index, clause in enumerate(clauses)
    )
    if unsupported_existing_claim:
        violations.append(
            {
                "code": "unsupported-existing-test-claim",
                "message": (
                    "recommendation claims a test exists without naming a "
                    "supplied test path"
                ),
            }
        )
    elif any(
        _mutates_unsupplied_test(clause, supplied_test_symbols=visible_symbols)
        and not any(
            reference_clause_index == clause_index and path in supplied_tests
            for reference_clause_index, _, path, _ in path_references
        )
        for clause_index, clause in enumerate(clauses)
    ):
        violations.append(
            {
                "code": "unsupported-test-mutation",
                "message": (
                    "recommendation changes a test without naming a supplied "
                    "test path"
                ),
            }
        )

    seen_paths: set[str] = set()
    for _, _, normalized, is_proposed in path_references:
        if normalized in seen_paths:
            continue
        seen_paths.add(normalized)
        if normalized not in normalized_files and not is_proposed:
            violations.append(
                {
                    "code": "unsupplied-test-path",
                    "message": f"recommendation names unsupplied test path {normalized}",
                }
            )

    return {
        "status": "grounded" if not violations else "failure",
        "violations": violations,
        "supplied_test_files": supplied_tests,
        "supplied_test_symbols": sorted(visible_symbols),
    }


def evaluate_findings(
    findings: Sequence[dict[str, object]],
    *,
    supplied_files: Sequence[str],
    supplied_test_symbols: Sequence[str] = (),
) -> dict[str, object]:
    """Score every provider recommendation while retaining per-finding evidence."""
    results = []
    for index, finding in enumerate(findings):
        recommendation = finding.get("suggested_fix")
        if not isinstance(recommendation, str):
            continue
        result = evaluate_recommendation(
            recommendation,
            supplied_files=supplied_files,
            supplied_test_symbols=supplied_test_symbols,
        )
        results.append(
            {
                "finding_index": index,
                "file": finding.get("file"),
                **result,
            }
        )
    grounded = sum(result["status"] == "grounded" for result in results)
    return {
        "evaluated": len(results),
        "grounded": grounded,
        "failures": len(results) - grounded,
        "results": results,
    }
