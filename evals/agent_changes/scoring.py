"""Finding-level adjudication and metrics for Quodet live runs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


EVALUATION_SPLITS = (
    "calibration",
    "holdout",
    "temporal",
    "clean-control",
    "confirmation",
)
VERDICTS = frozenset({"true-positive", "false-positive"})


def _empty_counts() -> dict[str, int]:
    return {"tp": 0, "fp": 0, "fn": 0}


def load_adjudications(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("cases"), dict):
        raise ValueError("adjudication must contain a cases object")
    return value


def validate_adjudications(
    run: dict[str, Any], adjudications: dict[str, Any]
) -> None:
    cases = adjudications.get("cases", {})
    if adjudications.get("run_id") != run.get("run_id"):
        raise ValueError("adjudication run_id does not match the raw run")
    if adjudications.get("fixture_revision") != run["configuration"]["fixture"][
        "revision"
    ]:
        raise ValueError("adjudication fixture revision does not match the raw run")

    for outcome in run["cases"]:
        if outcome["status"] != "schema-valid":
            continue
        case_id = outcome["case_id"]
        findings = outcome["parsed_response"]["findings"]
        entries = cases.get(case_id, {}).get("findings", [])
        indexes = [entry.get("finding_index") for entry in entries]
        if sorted(indexes) != list(range(len(findings))):
            raise ValueError(
                f"{case_id} must adjudicate every finding index exactly once"
            )
        expected_ids = set(outcome["expected_finding_ids"])
        matched: set[str] = set()
        for entry in entries:
            verdict = entry.get("verdict")
            if verdict not in VERDICTS:
                raise ValueError(f"{case_id} has invalid verdict {verdict!r}")
            if not isinstance(entry.get("rationale"), str) or not entry["rationale"].strip():
                raise ValueError(f"{case_id} finding adjudication requires a rationale")
            expected_id = entry.get("expected_finding_id")
            if verdict == "true-positive":
                if expected_id not in expected_ids:
                    raise ValueError(
                        f"{case_id} true positive must name an expected finding"
                    )
                if expected_id in matched:
                    raise ValueError(
                        f"{case_id} expected finding {expected_id!r} matched twice"
                    )
                matched.add(expected_id)
            elif expected_id is not None:
                raise ValueError(
                    f"{case_id} false positive cannot name an expected finding"
                )


def _case_counts(
    outcome: dict[str, Any], case_adjudication: dict[str, Any] | None
) -> dict[str, int]:
    expected = set(outcome["expected_finding_ids"])
    if outcome["status"] != "schema-valid":
        return {"tp": 0, "fp": 0, "fn": len(expected)}

    entries = (case_adjudication or {}).get("findings", [])
    matched = {
        entry["expected_finding_id"]
        for entry in entries
        if entry["verdict"] == "true-positive"
    }
    return {
        "tp": len(matched),
        "fp": sum(entry["verdict"] == "false-positive" for entry in entries),
        "fn": len(expected - matched),
    }


def _add_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key in ("tp", "fp", "fn"):
        target[key] += source[key]


def score_run(
    run: dict[str, Any],
    adjudications: dict[str, Any] | None,
) -> dict[str, Any]:
    """Score a raw run; semantic matches must come from explicit adjudication."""
    schema_valid = sum(case["status"] == "schema-valid" for case in run["cases"])
    total_cases = len(run["cases"])
    base: dict[str, Any] = {
        "adjudication_status": "pending" if adjudications is None else "complete",
        "schema_valid": schema_valid,
        "schema_total": total_cases,
        "schema_valid_rate": schema_valid / total_cases if total_cases else 0.0,
    }
    if adjudications is None:
        base.update(
            {
                "tp": None,
                "fp": None,
                "fn": None,
                "by_split": {},
                "by_family": {},
                "clean_control_false_positive_rate_by_family": {},
            }
        )
        return base

    validate_adjudications(run, adjudications)
    totals = _empty_counts()
    by_split = {split: _empty_counts() for split in EVALUATION_SPLITS}
    by_family: dict[str, dict[str, int]] = defaultdict(_empty_counts)
    control_cases: dict[str, int] = defaultdict(int)
    control_cases_with_fp: dict[str, int] = defaultdict(int)

    adjudicated_cases = adjudications["cases"]
    for outcome in run["cases"]:
        counts = _case_counts(outcome, adjudicated_cases.get(outcome["case_id"]))
        _add_counts(totals, counts)
        _add_counts(by_split[outcome["evaluation_split"]], counts)
        for family in outcome["failure_families"]:
            _add_counts(by_family[family], counts)
            if outcome["evaluation_split"] == "clean-control":
                control_cases[family] += 1
                if counts["fp"]:
                    control_cases_with_fp[family] += 1

    base.update(totals)
    base["by_split"] = by_split
    base["by_family"] = dict(sorted(by_family.items()))
    base["clean_control_false_positive_rate_by_family"] = {
        family: control_cases_with_fp[family] / count
        for family, count in sorted(control_cases.items())
    }
    return base


def adjudication_template(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "format_version": 1,
        "run_id": run["run_id"],
        "fixture_revision": run["configuration"]["fixture"]["revision"],
        "instructions": (
            "Judge the explanation and demonstrated failure path, not only the "
            "filename. Mark each provider finding true-positive or false-positive."
        ),
        "cases": {
            outcome["case_id"]: {
                "expected_findings": outcome.get("expected_findings", []),
                "findings": [
                    {
                        "finding_index": index,
                        "verdict": "REPLACE_ME",
                        "expected_finding_id": None,
                        "rationale": "",
                    }
                    for index, _ in enumerate(
                        (outcome.get("parsed_response") or {}).get("findings", [])
                    )
                ]
            }
            for outcome in run["cases"]
            if outcome["status"] == "schema-valid"
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a Quodet raw live-run artifact")
    parser.add_argument("run", type=Path)
    parser.add_argument("--adjudication", type=Path)
    parser.add_argument("--write-template", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not args.adjudication and not args.write_template:
        parser.error("provide --adjudication or --write-template")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run = json.loads(args.run.read_text(encoding="utf-8"))
    if args.write_template:
        args.write_template.write_text(
            json.dumps(adjudication_template(run), indent=2) + "\n",
            encoding="utf-8",
        )
    if args.adjudication:
        adjudications = load_adjudications(args.adjudication)
        report = dict(run)
        report["metrics"] = score_run(run, adjudications)
        output = args.output or args.run.with_name(f"{args.run.stem}.scored.json")
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
