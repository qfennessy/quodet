from __future__ import annotations

import argparse
import hashlib
import io
import json
import queue
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import watch_files
from evals.agent_changes import live_eval, replay, scoring


CORPUS_FAMILIES = {
    "state/lifecycle",
    "external API contract",
    "privacy/authorization",
    "retry/concurrency",
    "UI/cache",
    "CI/tooling",
    "persistence/atomicity",
    "performance/cost",
}
PRIMARY_CALIBRATION_IDS = {
    "A01", "A04", "A06", "A09", "A11", "A14", "A17",
    "B02", "B04", "B06", "B09", "B12",
}


def provider_finding(file: str, explanation: str) -> dict[str, object]:
    return {
        "file": file,
        "line": 1,
        "severity": "medium",
        "confidence": 0.99,
        "title": "Finding",
        "explanation": explanation,
        "suggested_fix": "Change the branch and add a focused regression test.",
    }


def raw_run(cases: list[dict[str, object]]) -> dict[str, object]:
    return {
        "run_id": "run-1",
        "configuration": {"fixture": {"revision": 2}},
        "cases": cases,
    }


class AgentChangeReplayTests(unittest.TestCase):
    def test_manifest_has_complete_taxonomy_metadata_and_fixture_files(self) -> None:
        manifest = replay.load_manifest()
        self.assertEqual(manifest["version"], 2)
        covered_families: set[str] = set()

        for case in manifest["cases"]:
            with self.subTest(case=case["id"]):
                for field in (
                    "failure_families", "scope", "expected_evidence_depth",
                    "evaluation_split", "provenance",
                ):
                    self.assertIn(field, case)
                self.assertIn(case["evaluation_split"], scoring.EVALUATION_SPLITS)
                self.assertIn(case["scope"], {"narrow", "cross-file"})
                fixture_names = sorted(
                    path.name for path in (replay.CASES_ROOT / case["id"]).iterdir()
                )
                self.assertEqual(sorted(case["files"]), fixture_names)
                covered_families.update(case["failure_families"])

        self.assertEqual(covered_families, CORPUS_FAMILIES)

    def test_calibration_provenance_cannot_reference_sealed_ids(self) -> None:
        manifest = replay.load_manifest()
        for case in manifest["cases"]:
            ids = set(case["provenance"]["calibration_ids"])
            self.assertLessEqual(ids, PRIMARY_CALIBRATION_IDS)
            if case["evaluation_split"] == "holdout":
                self.assertEqual(ids, set())

    def test_has_matched_clean_controls_at_multiple_sizes_and_depths(self) -> None:
        controls = [
            case for case in replay.load_manifest()["cases"]
            if case["evaluation_split"] == "clean-control"
        ]
        self.assertGreaterEqual(len(controls), 4)
        self.assertGreaterEqual({len(case["files"]) for case in controls}, {1, 2})
        self.assertGreaterEqual({case["scope"] for case in controls}, {"narrow", "cross-file"})
        self.assertTrue(all(case["expected_findings"] == [] for case in controls))

    def test_replay_writes_related_files_to_one_case_directory(self) -> None:
        case = replay.case_by_id(replay.load_manifest(), "10_aggregate_cache_fingerprint")
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory)
            written = replay.replay_case(case, destination=destination, inter_file_delay=0)
            self.assertEqual([path.name for path in written], ["cache.py", "dashboard.py"])
            self.assertTrue(all(path.parent == destination.resolve() / case["id"] for path in written))
            first_contents = [path.read_bytes() for path in written]
            replayed = replay.replay_case(case, destination=destination, inter_file_delay=0)
            self.assertNotEqual(first_contents, [path.read_bytes() for path in replayed])

    def test_sealed_empty_splits_are_selectable_without_exposing_fixtures(self) -> None:
        manifest = replay.load_manifest()
        self.assertEqual(live_eval.select_cases(manifest, "temporal"), [])
        self.assertEqual(live_eval.select_cases(manifest, "confirmation"), [])

    def test_live_configuration_records_exact_model_options_prompt_schema_and_fixture(self) -> None:
        manifest = replay.load_manifest()
        cases = manifest["cases"][:2]
        configuration = live_eval.evaluation_configuration(
            model="test-model", reasoning_effort="high", prompt="frozen prompt",
            fixture_revision=manifest["version"], cases=cases,
        )
        self.assertEqual(configuration["model"], "test-model")
        self.assertEqual(configuration["model_options"], {"reasoning_effort": "high"})
        self.assertEqual(configuration["prompt"]["text"], "frozen prompt")
        self.assertEqual(
            configuration["prompt"]["sha256"],
            hashlib.sha256(b"frozen prompt").hexdigest(),
        )
        self.assertEqual(configuration["prompt"]["revision"], watch_files.PROMPT_REVISION)
        self.assertEqual(configuration["schema"]["value"], watch_files.REVIEW_SCHEMA)
        self.assertEqual(configuration["schema"]["revision"], watch_files.REVIEW_SCHEMA_REVISION)
        self.assertEqual(configuration["fixture"]["revision"], 2)
        self.assertEqual(configuration["fixture"]["case_ids"], [case["id"] for case in cases])

    def test_wait_for_outcome_retains_raw_schema_valid_response_and_latency(self) -> None:
        output: queue.Queue[str] = queue.Queue()
        response = {"findings": [provider_finding("service.py", "Concrete failure path.")]}
        raw_response = json.dumps(response) + "\n"
        output.put("Reviewing 1 changed file(s): service.py\n")
        output.put(json.dumps({"quodet_evaluation_event": {
            "status": "success", "returncode": 0,
            "raw_response": raw_response, "stderr": "",
        }}) + "\n")
        outcome = live_eval.wait_for_outcome(output, timeout=1)
        self.assertEqual(outcome.status, "schema-valid")
        self.assertEqual(outcome.raw_response, raw_response)
        self.assertEqual(outcome.parsed_response, response)
        self.assertGreaterEqual(outcome.latency_ms, 0)

    def test_live_schema_validation_matches_recorded_schema_and_strict_json(self) -> None:
        finding = provider_finding("", "")
        finding["title"] = ""
        finding["extra_provider_field"] = "allowed by the recorded schema"
        self.assertIsNone(live_eval.validate_response({"findings": [finding]}))

        output: queue.Queue[str] = queue.Queue()
        raw_response = json.dumps({"findings": [provider_finding("a.py", "bug")]})
        raw_response = raw_response.replace("0.99", "NaN")
        output.put(json.dumps({"quodet_evaluation_event": {
            "status": "success", "returncode": 0,
            "raw_response": raw_response, "stderr": "",
        }}) + "\n")
        outcome = live_eval.wait_for_outcome(output, timeout=1)
        self.assertEqual(outcome.status, "schema-error")
        self.assertIn("invalid JSON constant", outcome.error or "")

    def test_live_eval_persists_completed_cases_when_replay_fails(self) -> None:
        manifest = replay.load_manifest()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            args = argparse.Namespace(
                case="all",
                destination=root / "replay",
                results_directory=root / "results",
                model="test-model",
                reasoning_effort="auto",
                debounce=0.01,
                review_timeout=0.01,
                inter_file_delay=0,
                settle=0.01,
                log=False,
            )
            process = mock.Mock(stdout=io.StringIO(), poll=mock.Mock(return_value=0))
            provider = live_eval.ProviderOutcome(
                status="schema-valid",
                latency_ms=1,
                transcript="",
                raw_response='{\"findings\": []}',
                parsed_response={"findings": []},
                error=None,
            )
            with (
                mock.patch("evals.agent_changes.live_eval.parse_args", return_value=args),
                mock.patch("evals.agent_changes.live_eval.subprocess.Popen", return_value=process),
                mock.patch("evals.agent_changes.live_eval.threading.Thread"),
                mock.patch("evals.agent_changes.live_eval.wait_for_startup"),
                mock.patch("evals.agent_changes.live_eval.time.sleep"),
                mock.patch(
                    "evals.agent_changes.live_eval.replay.replay_case",
                    side_effect=[None, RuntimeError("fixture failed")],
                ),
                mock.patch(
                    "evals.agent_changes.live_eval.wait_for_outcome",
                    return_value=provider,
                ),
                self.assertRaisesRegex(RuntimeError, "fixture failed"),
            ):
                live_eval.main([])

            artifacts = list((root / "results").glob("*.raw.json"))
            self.assertEqual(len(artifacts), 1)
            artifact = json.loads(artifacts[0].read_text())
            self.assertEqual(len(artifact["cases"]), 1)
            self.assertEqual(artifact["cases"][0]["case_id"], manifest["cases"][0]["id"])

    def test_filename_match_is_diagnostic_not_true_positive(self) -> None:
        case = replay.case_by_id(replay.load_manifest(), "12_semantically_invalid_external_value")
        provider = live_eval.ProviderOutcome(
            status="schema-valid", latency_ms=12, transcript="", raw_response="{}",
            parsed_response={"findings": [provider_finding("provider.py", "Wrong diagnosis")]},
            error=None,
        )
        outcome = live_eval.case_outcome(case, provider)
        self.assertTrue(outcome["diagnostics"]["filename_match"])

        run = raw_run([outcome])
        adjudication = {
            "run_id": "run-1", "fixture_revision": 2,
            "cases": {case["id"]: {"findings": [{
                "finding_index": 0, "verdict": "false-positive",
                "expected_finding_id": None, "rationale": "Right file, wrong failure path",
            }]}},
        }
        metrics = scoring.score_run(run, adjudication)
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (0, 1, 1))

    def test_adjudicated_metrics_include_splits_families_schema_and_control_rate(self) -> None:
        defect = {
            "case_id": "defect", "evaluation_split": "holdout",
            "failure_families": ["external API contract"],
            "expected_finding_ids": ["expected-1"], "status": "schema-valid",
            "parsed_response": {"findings": [provider_finding("a.py", "Correct path")]},
        }
        clean = {
            "case_id": "clean", "evaluation_split": "clean-control",
            "failure_families": ["external API contract"],
            "expected_finding_ids": [], "status": "schema-valid",
            "parsed_response": {"findings": [provider_finding("b.py", "False alarm")]},
        }
        run = raw_run([defect, clean])
        adjudication = {
            "run_id": "run-1", "fixture_revision": 2,
            "cases": {
                "defect": {"findings": [{
                    "finding_index": 0, "verdict": "true-positive",
                    "expected_finding_id": "expected-1", "rationale": "Matches behavior",
                }]},
                "clean": {"findings": [{
                    "finding_index": 0, "verdict": "false-positive",
                    "expected_finding_id": None, "rationale": "Control is correct",
                }]},
            },
        }
        metrics = scoring.score_run(run, adjudication)
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (1, 1, 0))
        self.assertEqual(metrics["schema_valid_rate"], 1.0)
        self.assertEqual(metrics["by_split"]["holdout"]["tp"], 1)
        self.assertEqual(metrics["by_split"]["temporal"], {"tp": 0, "fp": 0, "fn": 0})
        self.assertEqual(
            metrics["clean_control_false_positive_rate_by_family"]["external API contract"],
            1.0,
        )

    def test_schema_failure_counts_expected_findings_as_misses(self) -> None:
        failed = {
            "case_id": "failed", "evaluation_split": "calibration",
            "failure_families": ["state/lifecycle"],
            "expected_finding_ids": ["one", "two"], "status": "schema-error",
            "parsed_response": None,
        }
        run = raw_run([failed])
        adjudication = {"run_id": "run-1", "fixture_revision": 2, "cases": {}}
        metrics = scoring.score_run(run, adjudication)
        self.assertEqual(metrics["schema_valid_rate"], 0.0)
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (0, 0, 2))

    def test_adjudication_template_does_not_guess_semantic_matches(self) -> None:
        case = {
            "case_id": "case", "evaluation_split": "holdout",
            "failure_families": ["state/lifecycle"], "expected_finding_ids": ["expected"],
            "status": "schema-valid",
            "parsed_response": {"findings": [provider_finding("a.py", "Maybe")]},
        }
        template = scoring.adjudication_template(raw_run([case]))
        entry = template["cases"]["case"]["findings"][0]
        self.assertEqual(entry["verdict"], "REPLACE_ME")
        self.assertIsNone(entry["expected_finding_id"])


if __name__ == "__main__":
    unittest.main()
