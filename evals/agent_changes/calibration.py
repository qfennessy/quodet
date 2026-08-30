"""Leakage-safe confidence calibration for adjudicated Quodet runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from evals.agent_changes import artifacts, scoring


FORMAT_VERSION = 1
METHOD = "fixed-bin-empirical-v1"
DEFAULT_BUCKET_EDGES = (0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0)
FIT_SPLIT = "calibration"
DECLARED_NON_FIT_SPLITS = tuple(
    split for split in scoring.EVALUATION_SPLITS if split != FIT_SPLIT
) + ("challenge-development", "challenge-holdout")


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _raw_confidence(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or value > 1
    ):
        raise ValueError(f"malformed raw confidence: {value!r}")
    return float(value)


def _validate_edges(edges: Sequence[float]) -> tuple[float, ...]:
    normalized = tuple(float(edge) for edge in edges)
    if (
        len(normalized) < 2
        or normalized[0] != 0.0
        or normalized[-1] != 1.0
        or any(not math.isfinite(edge) for edge in normalized)
        or any(left >= right for left, right in zip(normalized, normalized[1:]))
    ):
        raise ValueError("bucket edges must increase strictly from 0.0 to 1.0")
    return normalized


def calibration_identity(configuration: Mapping[str, Any]) -> dict[str, Any]:
    """Return only fields whose drift invalidates a calibration artifact."""
    prompt = _require_mapping(configuration.get("prompt"), "configuration.prompt")
    schema = _require_mapping(configuration.get("schema"), "configuration.schema")
    fixture = _require_mapping(configuration.get("fixture"), "configuration.fixture")
    benchmark = configuration.get("benchmark")
    model_config: Mapping[str, Any] = {}
    candidate_id = None
    if isinstance(benchmark, Mapping):
        candidate_id = benchmark.get("candidate_id")
        candidate_config = benchmark.get("model_run_config")
        if isinstance(candidate_config, Mapping):
            model_config = candidate_config
    hardware = model_config.get("hardware")
    if not isinstance(hardware, Mapping):
        hardware = {}
    return {
        "model": configuration.get("model"),
        "model_revision": model_config.get("model_revision"),
        "model_artifact": model_config.get("model_artifact"),
        "provider_model_revision": hardware.get("provider_model_revision"),
        "runtime_model_id": hardware.get("runtime_model_id"),
        "runtime_artifact_sha256": hardware.get("runtime_artifact_sha256"),
        "candidate_id": candidate_id,
        "provider": model_config.get("provider"),
        "runtime": model_config.get("runtime"),
        "runtime_version": model_config.get("runtime_version"),
        "quantization": model_config.get("quantization"),
        "model_options": {
            "evaluation": configuration.get("model_options"),
            "runtime": model_config.get("model_options"),
        },
        "prompt": {
            "revision": prompt.get("revision"),
            "sha256": prompt.get("sha256"),
        },
        "schema": {
            "revision": schema.get("revision"),
            "sha256": schema.get("sha256"),
        },
        "fixture_revision": fixture.get("revision"),
    }


def _identity_missing(identity: Mapping[str, Any]) -> list[str]:
    required_paths = (
        ("model",), ("model_revision",), ("prompt", "revision"),
        ("prompt", "sha256"), ("schema", "revision"),
        ("schema", "sha256"), ("fixture_revision",),
    )
    missing: list[str] = []
    for path in required_paths:
        value: Any = identity
        for part in path:
            value = value.get(part) if isinstance(value, Mapping) else None
        if value is None or value == "":
            missing.append(".".join(path))
    return missing


def _validate_scored_run(run: Mapping[str, Any]) -> Mapping[str, Any]:
    adjudication = _require_mapping(run.get("adjudication"), "run.adjudication")
    expected_digest = scoring.adjudication_sha256(dict(adjudication))
    if run.get("adjudication_sha256") != expected_digest:
        raise ValueError("scored run adjudication hash does not match")
    scoring.validate_adjudications(dict(run), dict(adjudication))
    metrics = run.get("metrics")
    if metrics is not None and metrics != scoring.score_run(dict(run), dict(adjudication)):
        raise ValueError("scored run metrics do not match retained adjudication")
    return adjudication


def _finding_rows(
    run: Mapping[str, Any], adjudication: Mapping[str, Any],
    *, splits: set[str] | None = None,
) -> list[dict[str, Any]]:
    adjudicated_cases = _require_mapping(adjudication.get("cases"), "adjudication.cases")
    rows: list[dict[str, Any]] = []
    for outcome in run.get("cases", []):
        split = outcome.get("evaluation_split")
        if splits is not None and split not in splits:
            continue
        if outcome.get("status") != "schema-valid":
            continue
        parsed = _require_mapping(outcome.get("parsed_response"), "parsed_response")
        findings = parsed.get("findings")
        if not isinstance(findings, list):
            raise ValueError("schema-valid response requires a findings array")
        case_adjudication = adjudicated_cases.get(outcome.get("case_id"), {})
        entries = _require_mapping(case_adjudication, "case adjudication").get(
            "findings", []
        )
        by_index = {entry["finding_index"]: entry for entry in entries}
        for index, finding in enumerate(findings):
            entry = by_index[index]
            rows.append({
                "case_id": outcome.get("case_id"),
                "evaluation_split": split,
                "finding_index": index,
                "raw_confidence": _raw_confidence(finding.get("confidence")),
                "verdict": entry["verdict"],
                "correct": entry["verdict"] == "true-positive",
                "expected_finding_id": entry.get("expected_finding_id"),
            })
    return rows


def _bucket_index(value: float, edges: Sequence[float]) -> int:
    for index, upper in enumerate(edges[1:]):
        if value < upper or (upper == 1.0 and value <= upper):
            return index
    raise AssertionError("validated confidence did not fit a bucket")


def _bucket_report(
    rows: Sequence[Mapping[str, Any]], edges: Sequence[float],
    *, minimum_samples: int,
) -> list[dict[str, Any]]:
    counts = [0] * (len(edges) - 1)
    correct = [0] * (len(edges) - 1)
    for row in rows:
        index = _bucket_index(float(row["raw_confidence"]), edges)
        counts[index] += 1
        correct[index] += int(bool(row["correct"]))
    return [
        {
            "lower": lower,
            "upper": upper,
            "upper_inclusive": upper == 1.0,
            "count": count,
            "true_positives": successes,
            "false_positives": count - successes,
            "empirical_correctness": successes / count if count else None,
            "calibrated_score": (
                successes / count if count >= minimum_samples else None
            ),
            "status": "calibrated" if count >= minimum_samples else "sparse",
        }
        for lower, upper, count, successes in zip(
            edges, edges[1:], counts, correct
        )
    ]


def _evaluation_bucket_report(
    rows: Sequence[Mapping[str, Any]], edges: Sequence[float],
) -> list[dict[str, Any]]:
    """Report observed reliability without fitting a new score on eval labels."""
    report = _bucket_report(rows, edges, minimum_samples=1)
    return [
        {
            key: value for key, value in bucket.items()
            if key not in {"calibrated_score", "status"}
        }
        for bucket in report
    ]


def _select_threshold(
    rows: Sequence[Mapping[str, Any]], buckets: Sequence[Mapping[str, Any]],
    *, minimum_precision: float, minimum_samples: int,
    minimum_raw_confidence: float, edges: Sequence[float],
) -> dict[str, Any] | None:
    candidates = sorted({
        float(row["raw_confidence"]) for row in rows
        if float(row["raw_confidence"]) >= minimum_raw_confidence
    })
    qualifying: list[dict[str, Any]] = []
    for threshold in candidates:
        selected = [row for row in rows if row["raw_confidence"] >= threshold]
        successes = sum(bool(row["correct"]) for row in selected)
        precision = successes / len(selected) if selected else 0.0
        active_bucket_indexes = {
            _bucket_index(float(row["raw_confidence"]), edges) for row in selected
        }
        all_scores_available = all(
            buckets[index]["calibrated_score"] is not None
            for index in active_bucket_indexes
        )
        if (
            len(selected) >= minimum_samples
            and precision >= minimum_precision
            and all_scores_available
        ):
            qualifying.append({
                "raw_confidence": threshold,
                "sample_count": len(selected),
                "true_positives": successes,
                "false_positives": len(selected) - successes,
                "empirical_precision": precision,
            })
    return qualifying[0] if qualifying else None


def fit_calibration(
    run: Mapping[str, Any], *, minimum_precision: float,
    minimum_publication_samples: int, minimum_bucket_samples: int,
    minimum_raw_confidence: float = 0.0,
    bucket_edges: Sequence[float] = DEFAULT_BUCKET_EDGES,
) -> dict[str, Any]:
    """Fit only calibration labels and freeze the publication operating point."""
    if not 0 <= minimum_precision <= 1:
        raise ValueError("minimum precision must be between 0 and 1")
    if not 0 <= minimum_raw_confidence <= 1:
        raise ValueError("minimum raw confidence must be between 0 and 1")
    if minimum_publication_samples < 1 or minimum_bucket_samples < 1:
        raise ValueError("minimum sample counts must be positive")
    exposed_splits = {
        outcome.get("evaluation_split") for outcome in run.get("cases", [])
    }
    forbidden_splits = exposed_splits - {FIT_SPLIT}
    if forbidden_splits:
        raise ValueError(
            "calibration fit accepts calibration-only runs; refusing exposed "
            f"splits {sorted(str(split) for split in forbidden_splits)}"
        )
    edges = _validate_edges(bucket_edges)
    adjudication = _validate_scored_run(run)
    identity = calibration_identity(
        _require_mapping(run.get("configuration"), "run.configuration")
    )
    missing_identity = _identity_missing(identity)
    rows = _finding_rows(run, adjudication, splits={FIT_SPLIT})
    calibration_case_ids = [
        outcome.get("case_id") for outcome in run.get("cases", [])
        if outcome.get("evaluation_split") == FIT_SPLIT
    ]
    adjudicated_cases = _require_mapping(
        adjudication.get("cases"), "adjudication.cases"
    )
    fit_adjudication = {
        case_id: adjudicated_cases.get(case_id, {})
        for case_id in calibration_case_ids
    }
    buckets = _bucket_report(rows, edges, minimum_samples=minimum_bucket_samples)
    threshold = _select_threshold(
        rows, buckets, minimum_precision=minimum_precision,
        minimum_samples=minimum_publication_samples,
        minimum_raw_confidence=minimum_raw_confidence, edges=edges,
    )
    reasons: list[str] = []
    if missing_identity:
        reasons.append("missing version identity: " + ", ".join(missing_identity))
    if threshold is None:
        reasons.append("no threshold meets the frozen precision and sample targets")
    status = "calibrated" if not reasons else "uncalibrated"
    target = {
        "minimum_empirical_precision": minimum_precision,
        "minimum_publication_samples": minimum_publication_samples,
        "minimum_bucket_samples": minimum_bucket_samples,
        "minimum_raw_confidence_floor": minimum_raw_confidence,
        "selection_rule": (
            "lowest observed raw-confidence threshold meeting every target; "
            "fit on calibration findings only"
        ),
        "frozen_before_non_calibration_evaluation": True,
    }
    artifact = {
        "format_version": FORMAT_VERSION,
        "method": METHOD,
        "status": status,
        "status_reasons": reasons,
        "source": {
            "run_id": run.get("run_id"),
            "calibration_adjudication_sha256": _canonical_sha256(fit_adjudication),
        },
        "identity": identity,
        "identity_sha256": _canonical_sha256(identity),
        "fit_boundary": {
            "included_splits": [FIT_SPLIT],
            "excluded_splits": list(DECLARED_NON_FIT_SPLITS),
            "case_ids": calibration_case_ids,
            "finding_count": len(rows),
        },
        "target": target,
        "bucket_edges": list(edges),
        "reliability_buckets": buckets,
        "publication_threshold": threshold,
    }
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    if artifact.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported calibration artifact format")
    if artifact.get("method") != METHOD:
        raise ValueError("unsupported calibration method")
    identity = _require_mapping(artifact.get("identity"), "artifact.identity")
    if artifact.get("identity_sha256") != _canonical_sha256(identity):
        raise ValueError("calibration identity hash does not match")
    without_digest = dict(artifact)
    supplied_digest = without_digest.pop("artifact_sha256", None)
    if supplied_digest != _canonical_sha256(without_digest):
        raise ValueError("calibration artifact hash does not match")
    _validate_edges(artifact.get("bucket_edges", []))
    if artifact.get("status") not in {"calibrated", "uncalibrated"}:
        raise ValueError("invalid calibration status")


def _identity_differences(
    expected: Mapping[str, Any], actual: Mapping[str, Any], prefix: str = "",
) -> list[str]:
    differences: list[str] = []
    for key in sorted(set(expected) | set(actual)):
        path = f"{prefix}.{key}" if prefix else key
        left = expected.get(key)
        right = actual.get(key)
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            differences.extend(_identity_differences(left, right, path))
        elif left != right:
            differences.append(path)
    return differences


def _calibrated_score(
    raw: float, artifact: Mapping[str, Any],
) -> tuple[float | None, str]:
    edges = artifact["bucket_edges"]
    bucket = artifact["reliability_buckets"][_bucket_index(raw, edges)]
    score = bucket["calibrated_score"]
    return score, "calibrated" if score is not None else "uncalibrated"


def calibration_report(
    run: Mapping[str, Any], artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply a frozen artifact without refitting on evaluation answers."""
    validate_artifact(artifact)
    adjudication = _validate_scored_run(run)
    actual_identity = calibration_identity(
        _require_mapping(run.get("configuration"), "run.configuration")
    )
    identity_differences = _identity_differences(artifact["identity"], actual_identity)
    if identity_differences:
        status = "invalidated"
        status_reasons = ["changed calibration identity: " + ", ".join(identity_differences)]
    elif artifact["status"] != "calibrated":
        status = "uncalibrated"
        status_reasons = list(artifact.get("status_reasons", []))
    else:
        status = "calibrated"
        status_reasons = []

    rows = _finding_rows(run, adjudication)
    threshold_record = artifact.get("publication_threshold")
    threshold = (
        float(threshold_record["raw_confidence"])
        if status == "calibrated" and isinstance(threshold_record, Mapping)
        else None
    )
    decisions: list[dict[str, Any]] = []
    for row in rows:
        score, finding_status = _calibrated_score(row["raw_confidence"], artifact)
        if status == "invalidated":
            score = None
            finding_status = "invalidated"
        elif status != "calibrated":
            score = None
            finding_status = "uncalibrated"
        elif score is None:
            finding_status = "uncalibrated"
        published = bool(
            status == "calibrated"
            and score is not None
            and threshold is not None
            and row["raw_confidence"] >= threshold
        )
        decisions.append({
            **row,
            "calibrated_score": score,
            "calibration_status": finding_status,
            "published": published,
        })

    by_split: dict[str, dict[str, Any]] = {}
    outcomes_by_split: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for outcome in run.get("cases", []):
        outcomes_by_split[str(outcome.get("evaluation_split"))].append(outcome)
    for split in scoring.EVALUATION_SPLITS:
        split_rows = [row for row in decisions if row["evaluation_split"] == split]
        published_rows = [row for row in split_rows if row["published"]]
        expected = sum(
            len(outcome.get("expected_finding_ids", []))
            for outcome in outcomes_by_split[split]
        )
        published_tp = sum(row["correct"] for row in published_rows)
        published_fp = len(published_rows) - published_tp
        by_split[split] = {
            "case_count": len(outcomes_by_split[split]),
            "raw_finding_count": len(split_rows),
            "published_finding_count": len(published_rows),
            "published_true_positives": published_tp,
            "published_false_positives": published_fp,
            "expected_finding_count": expected,
            "published_precision": (
                published_tp / len(published_rows) if published_rows else 1.0
            ),
            "published_recall": published_tp / expected if expected else 1.0,
            "reliability_buckets": _evaluation_bucket_report(
                split_rows, artifact["bucket_edges"],
            ),
        }

    control_outcomes = outcomes_by_split["clean-control"]
    raw_control_cases_with_findings = {
        row["case_id"] for row in decisions
        if row["evaluation_split"] == "clean-control"
    }
    published_control_cases_with_findings = {
        row["case_id"] for row in decisions
        if row["evaluation_split"] == "clean-control" and row["published"]
    }
    control_count = len(control_outcomes)
    return {
        "format_version": FORMAT_VERSION,
        "status": status,
        "status_reasons": status_reasons,
        "run_id": run.get("run_id"),
        "calibration_artifact_sha256": artifact.get("artifact_sha256"),
        "raw_confidence_semantics": "untrusted model-reported score",
        "calibrated_score_semantics": (
            "empirical correctness within a fixed raw-confidence bucket; null "
            "means insufficient or invalidated evidence"
        ),
        "publication_threshold": threshold_record if status == "calibrated" else None,
        "findings": decisions,
        "by_split": by_split,
        "clean_controls": {
            "case_count": control_count,
            "raw_cases_with_false_positive": len(raw_control_cases_with_findings),
            "raw_false_positive_case_rate": (
                len(raw_control_cases_with_findings) / control_count
                if control_count else None
            ),
            "published_cases_with_false_positive": len(
                published_control_cases_with_findings
            ),
            "published_false_positive_case_rate": (
                len(published_control_cases_with_findings) / control_count
                if control_count else None
            ),
        },
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit or apply leakage-safe Quodet confidence calibration",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("fit", help="fit from a scored calibration run")
    fit.add_argument("run", type=Path)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--minimum-precision", type=float, default=0.90)
    fit.add_argument("--minimum-publication-samples", type=int, default=20)
    fit.add_argument("--minimum-bucket-samples", type=int, default=10)
    fit.add_argument("--minimum-raw-confidence", type=float, default=0.0)
    report = subparsers.add_parser(
        "report", help="apply a frozen artifact without refitting",
    )
    report.add_argument("run", type=Path)
    report.add_argument("--artifact", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run = _load_json(args.run)
    if args.command == "fit":
        result = fit_calibration(
            run,
            minimum_precision=args.minimum_precision,
            minimum_publication_samples=args.minimum_publication_samples,
            minimum_bucket_samples=args.minimum_bucket_samples,
            minimum_raw_confidence=args.minimum_raw_confidence,
        )
    else:
        result = calibration_report(run, _load_json(args.artifact))
    artifacts.write_private_json(args.output, result)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
