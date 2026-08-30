"""Validate whether repair guidance invents tests outside a reviewed snapshot."""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Sequence


_TEST_PATH = re.compile(
    r"(?<![A-Za-z0-9_.-])((?:[A-Za-z0-9_.-]+/)*(?:test_[A-Za-z0-9_.-]+|"
    r"[A-Za-z0-9_.-]+_test|[A-Za-z0-9_.-]+\.(?:spec|test))\."
    r"(?:py|js|jsx|ts|tsx|rb|go|rs|java|kt))(?![A-Za-z0-9_.-])",
    re.IGNORECASE,
)
_EXISTING_TEST_CLAIM = re.compile(
    r"\b(?:existing|current|already[- ]present|present)\b.{0,48}\btests?\b|"
    r"\btests?\b.{0,48}\b(?:already exists?|is present)\b",
    re.IGNORECASE,
)
_MUTATE_TEST_CLAIM = re.compile(
    r"\b(?:preserve|retain|extend|modify|update|keep)\b.{0,48}\btests?\b",
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
        or name.startswith("test_")
        or any(marker in name for marker in ("_test.", ".test.", ".spec."))
    )


def evaluate_recommendation(
    recommendation: str, *, supplied_files: Sequence[str]
) -> dict[str, object]:
    """Return machine-readable grounding failures, independent of bug detection."""
    normalized_files = {path.replace("\\", "/") for path in supplied_files}
    supplied_tests = sorted(path for path in normalized_files if is_test_file(path))
    violations: list[dict[str, str]] = []

    if not supplied_tests:
        if _EXISTING_TEST_CLAIM.search(recommendation):
            violations.append(
                {
                    "code": "unsupported-existing-test-claim",
                    "message": "recommendation claims a test exists but no test was supplied",
                }
            )
        elif _MUTATE_TEST_CLAIM.search(recommendation):
            violations.append(
                {
                    "code": "unsupported-test-mutation",
                    "message": "recommendation changes a test but no test was supplied",
                }
            )

    for referenced_path in sorted(set(_TEST_PATH.findall(recommendation))):
        normalized = referenced_path.replace("\\", "/")
        if normalized not in normalized_files:
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
    }


def evaluate_findings(
    findings: Sequence[dict[str, object]], *, supplied_files: Sequence[str]
) -> dict[str, object]:
    """Score every provider recommendation while retaining per-finding evidence."""
    results = []
    for index, finding in enumerate(findings):
        recommendation = finding.get("suggested_fix")
        if not isinstance(recommendation, str):
            continue
        result = evaluate_recommendation(
            recommendation, supplied_files=supplied_files
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
