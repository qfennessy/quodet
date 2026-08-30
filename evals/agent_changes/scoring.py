"""Finding-level adjudication and metrics for Quodet live runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from evals.agent_changes import artifacts


EVALUATION_SPLITS = (
    "calibration",
    "holdout",
    "temporal",
    "clean-control",
    "confirmation",
    "challenge-development",
    "challenge-holdout",
)
VERDICTS = frozenset({"true-positive", "false-positive"})
FIX_QUALITY = {
    "not-actionable": 0.0,
    "partially-actionable": 0.5,
    "actionable": 1.0,
}


def adjudication_sha256(value: dict[str, Any]) -> str:
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


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
        case_id = outcome.get("sample_id", outcome["case_id"])
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
                if entry.get("fix_quality") not in FIX_QUALITY:
                    raise ValueError(
                        f"{case_id} true positive requires a valid fix_quality"
                    )
                if outcome.get("challenge_pair_id"):
                    evidence = entry.get("evidence")
                    if not isinstance(evidence, dict) or any(
                        not isinstance(evidence.get(field), str)
                        or not evidence[field].strip()
                        for field in ("trigger", "failure_path", "impact")
                    ):
                        raise ValueError(
                            f"{case_id} challenge true positive requires trigger, "
                            "failure_path, and impact evidence"
                        )
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
    grounding_evaluated = sum(
        case.get("diagnostics", {})
        .get("recommendation_grounding", {})
        .get("evaluated", 0)
        for case in run["cases"]
    )
    grounding_failures = sum(
        case.get("diagnostics", {})
        .get("recommendation_grounding", {})
        .get("failures", 0)
        for case in run["cases"]
    )
    base: dict[str, Any] = {
        "adjudication_status": "pending" if adjudications is None else "complete",
        "schema_valid": schema_valid,
        "schema_total": total_cases,
        "schema_valid_rate": schema_valid / total_cases if total_cases else 0.0,
        "attempted_cases": total_cases,
        "status_counts": {
            status: sum(case["status"] == status for case in run["cases"])
            for status in sorted({str(case["status"]) for case in run["cases"]})
        },
        "recommendation_grounding": {
            "evaluated": grounding_evaluated,
            "grounded": grounding_evaluated - grounding_failures,
            "failures": grounding_failures,
            "grounded_rate": (
                (grounding_evaluated - grounding_failures) / grounding_evaluated
                if grounding_evaluated
                else 0.0
            ),
        },
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
                "clean_control_false_positive_rate": None,
                "fix_quality_score": None,
                "fix_quality_adjudications": 0,
                "split_metrics": {},
            }
        )
        return base

    validate_adjudications(run, adjudications)
    totals = _empty_counts()
    by_split = {split: _empty_counts() for split in EVALUATION_SPLITS}
    by_family: dict[str, dict[str, int]] = defaultdict(_empty_counts)
    control_cases: dict[str, int] = defaultdict(int)
    control_cases_with_fp: dict[str, int] = defaultdict(int)
    total_control_cases = 0
    total_control_cases_with_fp = 0
    fix_quality_scores: list[float] = []

    adjudicated_cases = adjudications["cases"]
    for outcome in run["cases"]:
        counts = _case_counts(
            outcome, adjudicated_cases.get(outcome.get("sample_id", outcome["case_id"]))
        )
        _add_counts(totals, counts)
        _add_counts(by_split[outcome["evaluation_split"]], counts)
        if outcome["evaluation_split"] == "clean-control":
            total_control_cases += 1
            if counts["fp"]:
                total_control_cases_with_fp += 1
        for family in outcome["failure_families"]:
            _add_counts(by_family[family], counts)
            if outcome["evaluation_split"] == "clean-control":
                control_cases[family] += 1
                if counts["fp"]:
                    control_cases_with_fp[family] += 1
        sample_id = outcome.get("sample_id", outcome["case_id"])
        for entry in adjudicated_cases.get(sample_id, {}).get("findings", []):
            if entry.get("verdict") == "true-positive":
                fix_quality_scores.append(FIX_QUALITY[entry["fix_quality"]])

    base.update(totals)
    base["by_split"] = by_split
    base["by_family"] = dict(sorted(by_family.items()))
    base["clean_control_false_positive_rate_by_family"] = {
        family: control_cases_with_fp[family] / count
        for family, count in sorted(control_cases.items())
    }
    base["clean_control_false_positive_rate"] = (
        total_control_cases_with_fp / total_control_cases
        if total_control_cases else 0.0
    )
    base["fix_quality_score"] = (
        sum(fix_quality_scores) / len(fix_quality_scores)
        if fix_quality_scores else None
    )
    base["fix_quality_adjudications"] = len(fix_quality_scores)
    split_metrics: dict[str, dict[str, Any]] = {}
    for split in EVALUATION_SPLITS:
        split_outcomes = [
            outcome for outcome in run["cases"]
            if outcome["evaluation_split"] == split
        ]
        counts = by_split[split]
        split_tp = counts["tp"]
        split_fp = counts["fp"]
        split_fn = counts["fn"]
        split_fix_scores = [
            FIX_QUALITY[entry["fix_quality"]]
            for outcome in split_outcomes
            for entry in adjudicated_cases.get(
                outcome.get("sample_id", outcome["case_id"]), {}
            ).get(
                "findings", []
            )
            if entry.get("verdict") == "true-positive"
        ]
        split_schema_valid = sum(
            outcome["status"] == "schema-valid" for outcome in split_outcomes
        )
        split_metrics[split] = {
            **counts,
            "attempted_cases": len(split_outcomes),
            "schema_valid_rate": (
                split_schema_valid / len(split_outcomes) if split_outcomes else None
            ),
            "finding_precision": (
                split_tp / (split_tp + split_fp) if split_tp + split_fp else 1.0
            ),
            "finding_recall": (
                split_tp / (split_tp + split_fn) if split_tp + split_fn else 1.0
            ),
            "fix_quality_score": (
                sum(split_fix_scores) / len(split_fix_scores)
                if split_fix_scores else None
            ),
        }
    base["split_metrics"] = split_metrics
    challenge_pairs: dict[str, dict[str, Any]] = {}
    for outcome in run["cases"]:
        pair_id = outcome.get("challenge_pair_id")
        if not pair_id:
            continue
        variant = outcome["challenge_variant"]
        aggregate = challenge_pairs.setdefault(pair_id, {}).setdefault(
            variant,
            {
                "tp": 0, "fp": 0, "fn": 0,
                "attempts": 0, "samples": 0, "valid_samples": 0,
                "invalid_samples": 0, "samples_with_fp": 0,
            },
        )
        aggregate["attempts"] += 1
        if outcome["status"] != "schema-valid":
            aggregate["invalid_samples"] += 1
            continue
        counts = _case_counts(
            outcome, adjudicated_cases.get(outcome.get("sample_id", outcome["case_id"]))
        )
        _add_counts(aggregate, counts)
        aggregate["samples"] += 1
        aggregate["valid_samples"] += 1
        aggregate["samples_with_fp"] += counts["fp"] > 0
    if challenge_pairs:
        defects = [pair["buggy"] for pair in challenge_pairs.values() if "buggy" in pair]
        clean = [pair["clean"] for pair in challenge_pairs.values() if "clean" in pair]
        defect_denominator = sum(item["tp"] + item["fn"] for item in defects)
        clean_denominator = sum(item["valid_samples"] for item in clean)
        base["challenge_pairs"] = challenge_pairs
        base["challenge_valid_attempts"] = sum(
            item["valid_samples"]
            for pair in challenge_pairs.values()
            for item in pair.values()
        )
        base["challenge_invalid_attempts"] = sum(
            item["invalid_samples"]
            for pair in challenge_pairs.values()
            for item in pair.values()
        )
        base["challenge_defect_recall"] = (
            sum(item["tp"] for item in defects)
            / defect_denominator
            if defect_denominator else None
        )
        base["challenge_clean_twin_false_positive_rate"] = (
            sum(item["samples_with_fp"] for item in clean)
            / clean_denominator
            if clean_denominator else None
        )
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
            outcome.get("sample_id", outcome["case_id"]): {
                "expected_findings": outcome.get("expected_findings", []),
                "findings": [
                    {
                        "finding_index": index,
                        "verdict": "REPLACE_ME",
                        "expected_finding_id": None,
                        "fix_quality": "REPLACE_ME",
                        "rationale": "",
                        **({"evidence": {
                            "trigger": "", "failure_path": "", "impact": "",
                        }} if outcome.get("challenge_pair_id") else {}),
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
        artifacts.write_private_json(args.write_template, adjudication_template(run))
    if args.adjudication:
        adjudications = load_adjudications(args.adjudication)
        report = dict(run)
        report["metrics"] = score_run(run, adjudications)
        report["adjudication"] = adjudications
        report["adjudication_sha256"] = adjudication_sha256(adjudications)
        output = args.output or args.run.with_name(f"{args.run.stem}.scored.json")
        artifacts.write_private_json(output, report)
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
