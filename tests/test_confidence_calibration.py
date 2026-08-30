from __future__ import annotations

import asyncio
import copy
import json
import tempfile
import unittest
from pathlib import Path

from evals.agent_changes import calibration, replay, scoring


def finding(confidence: object) -> dict[str, object]:
    return {
        "file": "service.py",
        "line": 1,
        "severity": "medium",
        "confidence": confidence,
        "title": "Finding",
        "explanation": "A concrete execution path reaches the failure.",
        "suggested_fix": "Change the branch and add a regression test.",
    }


def scored_run(
    cases: list[tuple[str, str, list[tuple[object, str]]]],
    *, model: str = "fixture-model",
    model_revision: str | None = "a" * 40,
    reasoning_effort: str = "high",
    prompt_sha256: str = "b" * 64,
    schema_sha256: str = "c" * 64,
    fixture_revision: int = 3,
) -> dict[str, object]:
    outcomes: list[dict[str, object]] = []
    adjudicated_cases: dict[str, object] = {}
    for case_id, split, provider_findings in cases:
        expected_ids = [
            f"{case_id}-expected"
            for _, verdict in provider_findings if verdict == "true-positive"
        ]
        outcomes.append({
            "case_id": case_id,
            "evaluation_split": split,
            "failure_families": ["retry/concurrency"],
            "expected_finding_ids": expected_ids,
            "status": "schema-valid",
            "parsed_response": {
                "findings": [finding(confidence) for confidence, _ in provider_findings]
            },
        })
        entries = []
        for index, (_, verdict) in enumerate(provider_findings):
            entry: dict[str, object] = {
                "finding_index": index,
                "verdict": verdict,
                "expected_finding_id": (
                    f"{case_id}-expected" if verdict == "true-positive" else None
                ),
                "rationale": "Deterministic fixture adjudication.",
            }
            if verdict == "true-positive":
                entry["fix_quality"] = "actionable"
            entries.append(entry)
        adjudicated_cases[case_id] = {"findings": entries}

    run: dict[str, object] = {
        "run_id": "fixture-run",
        "configuration": {
            "model": model,
            "model_options": {"reasoning_effort": reasoning_effort},
            "batching": {
                "debounce_seconds": 3.0,
                "inter_file_delay_seconds": 0.25,
            },
            "prompt": {"revision": "prompt-v1", "sha256": prompt_sha256},
            "schema": {"revision": "schema-v1", "sha256": schema_sha256},
            "fixture": {
                "revision": fixture_revision,
                "manifest_sha256": "d" * 64,
                "fixture_tree_sha256": "e" * 64,
                "provider_payload_sha256": "f" * 64,
            },
            "benchmark": {
                "candidate_id": "fixture-candidate",
                "model_run_config": {
                    "model": model,
                    "model_artifact": "fixture/model",
                    "model_revision": model_revision,
                    "provider": "fixture-provider",
                    "runtime": "fixture-runtime",
                    "runtime_version": "1.0",
                    "locality": "hosted",
                    "quantization": "fixture",
                    "model_options": {"temperature": 0},
                    "context_limit": 100_000,
                    "timeout_seconds": 60.0,
                    "max_output_bytes": 262_144,
                    "max_output_tokens": 4_096,
                    "max_output_tokens_option": "max_tokens",
                    "hardware": {"provider_model_revision": "provider-rev-1"},
                },
            },
        },
        "cases": outcomes,
    }
    adjudication = {
        "run_id": run["run_id"],
        "fixture_revision": fixture_revision,
        "cases": adjudicated_cases,
    }
    run["adjudication"] = adjudication
    run["adjudication_sha256"] = scoring.adjudication_sha256(adjudication)
    run["metrics"] = scoring.score_run(run, adjudication)
    return run


def fit(run: dict[str, object], **overrides: object) -> dict[str, object]:
    arguments: dict[str, object] = {
        "minimum_precision": 1.0,
        "minimum_publication_samples": 2,
        "minimum_bucket_samples": 2,
        "minimum_raw_confidence": 0.0,
        "bucket_edges": (0.0, 0.97, 1.0),
    }
    arguments.update(overrides)
    return calibration.fit_calibration(run, **arguments)


class ConfidenceCalibrationTests(unittest.TestCase):
    def test_cli_fits_and_applies_private_json_artifacts(self) -> None:
        calibration_run = scored_run([
            ("one", "calibration", [(0.99, "true-positive")]),
            ("two", "calibration", [(0.98, "true-positive")]),
        ])
        evaluation_run = scored_run([
            ("holdout", "holdout", [(0.99, "true-positive")]),
        ])
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            calibration_run_path = root / "calibration.scored.json"
            evaluation_run_path = root / "holdout.scored.json"
            artifact_path = root / "calibration.json"
            report_path = root / "report.json"
            calibration_run_path.write_text(json.dumps(calibration_run))
            evaluation_run_path.write_text(json.dumps(evaluation_run))

            self.assertEqual(calibration.main([
                "fit", str(calibration_run_path),
                "--minimum-precision", "1",
                "--minimum-publication-samples", "2",
                "--minimum-bucket-samples", "2",
                "--minimum-raw-confidence", "0.95",
                "--output", str(artifact_path),
            ]), 0)
            self.assertEqual(artifact_path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(calibration.main([
                "report", str(evaluation_run_path),
                "--artifact", str(artifact_path),
                "--output", str(report_path),
            ]), 0)
            report = json.loads(report_path.read_text())
            self.assertEqual(report["status"], "calibrated")
            self.assertTrue(report["findings"][0]["published"])

    def test_cancellation_false_positive_is_a_reachable_path_negative_control(self) -> None:
        case = replay.case_by_id(
            replay.load_manifest(), "26_clean_settled_async_rollback"
        )
        self.assertEqual(case["evaluation_split"], "clean-control")
        self.assertEqual(case["expected_findings"], [])
        source_root = replay.CASES_ROOT / case["id"]
        repository_namespace: dict[str, object] = {"__name__": "fixture_repository"}
        exec(
            (source_root / "repository.py").read_text(encoding="utf-8"),
            repository_namespace,
        )
        service_source = (source_root / "service.py").read_text(encoding="utf-8")
        service_source = service_source.replace("from repository import Repository\n", "")
        service_namespace: dict[str, object] = {
            "__name__": "fixture_service",
            "Repository": repository_namespace["Repository"],
        }
        exec(service_source, service_namespace)

        async def exercise() -> None:
            repository_type = repository_namespace["Repository"]
            create_pair = service_namespace["create_pair"]
            for cancel_after in (None, 0.0, 0.001, 0.02, 0.06):
                repository = repository_type()
                task = asyncio.create_task(create_pair(repository))
                if cancel_after is not None:
                    await asyncio.sleep(cancel_after)
                    task.cancel()
                try:
                    await task
                except (RuntimeError, asyncio.CancelledError):
                    pass
                await asyncio.sleep(0.06)
                self.assertEqual(
                    repository.records,
                    set(),
                    f"residual state at cancellation delay {cancel_after}",
                )

        asyncio.run(exercise())

    def test_fit_keeps_raw_confidence_distinct_and_reports_controls(self) -> None:
        calibration_run = scored_run([
            ("cal-one", "calibration", [(0.99, "true-positive")]),
            ("cal-two", "calibration", [(0.98, "true-positive")]),
            ("cal-low", "calibration", [(0.90, "false-positive")]),
        ])
        evaluation_run = scored_run([
            ("holdout", "holdout", [(0.99, "true-positive")]),
            ("clean", "clean-control", [(0.99, "false-positive")]),
        ])

        artifact = fit(calibration_run)
        self.assertEqual(artifact["status"], "calibrated")
        self.assertEqual(artifact["fit_boundary"]["included_splits"], ["calibration"])
        self.assertIn(
            "challenge-holdout", artifact["fit_boundary"]["excluded_splits"]
        )
        self.assertEqual(artifact["fit_boundary"]["finding_count"], 3)
        self.assertEqual(artifact["publication_threshold"]["raw_confidence"], 0.98)
        report = calibration.calibration_report(evaluation_run, artifact)

        self.assertEqual(report["status"], "calibrated")
        holdout_finding = next(
            row for row in report["findings"] if row["case_id"] == "holdout"
        )
        self.assertEqual(holdout_finding["raw_confidence"], 0.99)
        self.assertEqual(holdout_finding["calibrated_score"], 1.0)
        self.assertTrue(holdout_finding["published"])
        self.assertEqual(report["by_split"]["holdout"]["published_precision"], 1.0)
        self.assertNotIn(
            "calibrated_score",
            report["by_split"]["holdout"]["reliability_buckets"][-1],
        )
        self.assertEqual(report["clean_controls"]["raw_false_positive_case_rate"], 1.0)
        self.assertEqual(
            report["clean_controls"]["published_false_positive_case_rate"], 1.0
        )

    def test_fit_refuses_non_calibration_answers(self) -> None:
        run = scored_run([
            ("cal-one", "calibration", [(0.99, "true-positive")]),
            ("cal-two", "calibration", [(0.98, "true-positive")]),
            ("holdout", "holdout", [(0.99, "true-positive")]),
        ])

        with self.assertRaisesRegex(ValueError, "calibration-only"):
            fit(run)

    def test_threshold_is_lowest_value_meeting_frozen_target(self) -> None:
        run = scored_run([
            ("one", "calibration", [(0.90, "true-positive")]),
            ("two", "calibration", [(0.80, "true-positive")]),
            ("three", "calibration", [(0.70, "false-positive")]),
        ])
        artifact = fit(
            run, bucket_edges=(0.0, 0.8, 1.0), minimum_bucket_samples=1,
        )
        self.assertEqual(artifact["publication_threshold"], {
            "raw_confidence": 0.8,
            "sample_count": 2,
            "true_positives": 2,
            "false_positives": 0,
            "empirical_precision": 1.0,
        })

    def test_sparse_evidence_is_explicitly_uncalibrated(self) -> None:
        run = scored_run([
            ("one", "calibration", [(0.99, "true-positive")]),
        ])
        artifact = fit(run)
        self.assertEqual(artifact["status"], "uncalibrated")
        self.assertIsNone(artifact["publication_threshold"])
        self.assertIn("no threshold", artifact["status_reasons"][0])
        report = calibration.calibration_report(run, artifact)
        self.assertEqual(report["status"], "uncalibrated")
        self.assertFalse(report["findings"][0]["published"])
        self.assertIsNone(report["findings"][0]["calibrated_score"])

    def test_missing_exact_model_revision_is_uncalibrated(self) -> None:
        run = scored_run([
            ("one", "calibration", [(0.99, "true-positive")]),
            ("two", "calibration", [(0.98, "true-positive")]),
        ], model_revision=None)
        artifact = fit(run)
        self.assertEqual(artifact["status"], "uncalibrated")
        self.assertIn("model.revision", artifact["status_reasons"][0])
        report = calibration.calibration_report(run, artifact)
        self.assertTrue(
            all(row["calibrated_score"] is None for row in report["findings"])
        )

    def test_malformed_confidence_values_fail_closed(self) -> None:
        for malformed in (True, -0.1, 1.1, float("nan"), "0.99"):
            with self.subTest(value=malformed):
                run = scored_run([
                    ("one", "calibration", [(malformed, "true-positive")]),
                ])
                with self.assertRaisesRegex(ValueError, "malformed raw confidence"):
                    fit(run, minimum_publication_samples=1, minimum_bucket_samples=1)

    def test_identity_changes_invalidate_without_reusing_scores(self) -> None:
        base = scored_run([
            ("one", "calibration", [(0.99, "true-positive")]),
            ("two", "calibration", [(0.98, "true-positive")]),
        ])
        artifact = fit(base)

        def change_fixture_revision(run: dict[str, object]) -> None:
            run["configuration"]["fixture"]["revision"] = 4
            run["adjudication"]["fixture_revision"] = 4
            run["adjudication_sha256"] = scoring.adjudication_sha256(
                run["adjudication"]
            )
            run["metrics"] = scoring.score_run(run, run["adjudication"])

        mutations = {
            "model.requested_identifier": lambda run: run[
                "configuration"
            ].__setitem__("model", "other"),
            "model.revision": lambda run: run["configuration"]["benchmark"][
                "model_run_config"
            ].__setitem__("model_revision", "f" * 40),
            "options.evaluation.reasoning_effort": lambda run: run[
                "configuration"
            ]["model_options"].__setitem__("reasoning_effort", "medium"),
            "batching.debounce_seconds": lambda run: run["configuration"][
                "batching"
            ].__setitem__("debounce_seconds", 1.0),
            "batching.inter_file_delay_seconds": lambda run: run[
                "configuration"
            ]["batching"].__setitem__("inter_file_delay_seconds", 4.0),
            "execution.context_limit": lambda run: run["configuration"][
                "benchmark"
            ]["model_run_config"].__setitem__("context_limit", 90_000),
            "execution.timeout_seconds": lambda run: run["configuration"][
                "benchmark"
            ]["model_run_config"].__setitem__("timeout_seconds", 30.0),
            "execution.max_output_bytes": lambda run: run["configuration"][
                "benchmark"
            ]["model_run_config"].__setitem__("max_output_bytes", 131_072),
            "execution.max_output_tokens": lambda run: run["configuration"][
                "benchmark"
            ]["model_run_config"].__setitem__("max_output_tokens", 2_048),
            "execution.max_output_tokens_option": lambda run: run[
                "configuration"
            ]["benchmark"]["model_run_config"].__setitem__(
                "max_output_tokens_option", "max_output_tokens"
            ),
            "prompt.sha256": lambda run: run["configuration"]["prompt"].__setitem__(
                "sha256", "d" * 64
            ),
            "schema.sha256": lambda run: run["configuration"]["schema"].__setitem__(
                "sha256", "e" * 64
            ),
            "fixture.revision": change_fixture_revision,
            "fixture.manifest_sha256": lambda run: run["configuration"][
                "fixture"
            ].__setitem__("manifest_sha256", "a" * 64),
            "fixture.fixture_tree_sha256": lambda run: run["configuration"][
                "fixture"
            ].__setitem__("fixture_tree_sha256", "b" * 64),
            "fixture.provider_payload_sha256": lambda run: run["configuration"][
                "fixture"
            ].__setitem__("provider_payload_sha256", "c" * 64),
        }
        unchanged = calibration.calibration_report(copy.deepcopy(base), artifact)
        self.assertEqual(unchanged["status"], "calibrated")
        self.assertTrue(all(row["published"] for row in unchanged["findings"]))
        for expected_path, mutate in mutations.items():
            with self.subTest(path=expected_path):
                changed = copy.deepcopy(base)
                mutate(changed)
                report = calibration.calibration_report(changed, artifact)
                self.assertEqual(report["status"], "invalidated")
                self.assertIn(expected_path, report["status_reasons"][0])
                self.assertTrue(all(not row["published"] for row in report["findings"]))
                self.assertTrue(
                    all(row["calibrated_score"] is None for row in report["findings"])
                )

    def test_legacy_missing_and_malformed_identities_fail_closed(self) -> None:
        run = scored_run([
            ("one", "calibration", [(0.99, "true-positive")]),
            ("two", "calibration", [(0.98, "true-positive")]),
        ])
        artifact = fit(run)

        def resign(value: dict[str, object]) -> None:
            value["identity_sha256"] = calibration._canonical_sha256(
                value["identity"]
            )
            unsigned = dict(value)
            unsigned.pop("artifact_sha256", None)
            value["artifact_sha256"] = calibration._canonical_sha256(unsigned)

        legacy = copy.deepcopy(artifact)
        legacy["identity"].pop("format_version")
        resign(legacy)
        with self.assertRaisesRegex(ValueError, "identity format"):
            calibration.calibration_report(run, legacy)

        missing = copy.deepcopy(artifact)
        missing["identity"]["fixture"].pop("manifest_sha256")
        resign(missing)
        with self.assertRaisesRegex(ValueError, "missing required fields"):
            calibration.calibration_report(run, missing)

        malformed = copy.deepcopy(artifact)
        malformed["identity"]["fixture"]["provider_payload_sha256"] = "invalid"
        resign(malformed)
        with self.assertRaisesRegex(ValueError, "not a SHA-256"):
            calibration.calibration_report(run, malformed)

        malformed_limit = copy.deepcopy(artifact)
        malformed_limit["identity"]["batching"]["debounce_seconds"] = "fast"
        resign(malformed_limit)
        with self.assertRaisesRegex(ValueError, "not a valid limit"):
            calibration.calibration_report(run, malformed_limit)

    def test_tampered_artifact_is_rejected(self) -> None:
        run = scored_run([
            ("one", "calibration", [(0.99, "true-positive")]),
            ("two", "calibration", [(0.98, "true-positive")]),
        ])
        artifact = fit(run)
        artifact["publication_threshold"]["raw_confidence"] = 0.0
        with self.assertRaisesRegex(ValueError, "artifact hash"):
            calibration.calibration_report(run, artifact)


if __name__ == "__main__":
    unittest.main()
