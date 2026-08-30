from __future__ import annotations

import argparse
import hashlib
import io
import json
import queue
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import watch_files
from evals.agent_changes import artifacts, benchmark, live_eval, replay, scoring
from model_runner import ModelRunConfig, Pricing, model_run_config_sha256


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
        "configuration": {"fixture": {"revision": 3}},
        "cases": cases,
    }


def plan_with_approved_qwen_artifact() -> dict[str, object]:
    plan = benchmark.load_plan()
    candidate = plan["candidates"]["qwen35-a3b-local"]
    candidate["runtime_artifact"] = {
        "status": "approved",
        "runtime": "llm-ollama",
        "runtime_version": "1.0",
        "model": "qwen-eval",
        "runtime_model_id": "qwen3.5:35b-a3b",
        "runtime_artifact_sha256": "a" * 64,
        "quantization": "bfloat16",
        "max_output_tokens_option": "max_tokens",
        "source_binding": {
            "model_artifact": candidate["model_artifact"],
            "model_revision": candidate["model_revision"],
            "conversion_tool": "fixture-converter",
            "conversion_tool_version": "1.0",
            "recipe_sha256": "b" * 64,
        },
    }
    return plan


def approved_qwen_config(plan: dict[str, object]) -> ModelRunConfig:
    return benchmark.prepare_run_config(
        plan,
        candidate_id="qwen35-a3b-local",
        model="qwen-eval",
        provider="local",
        runtime="llm-ollama",
        runtime_version="1.0",
        quantization="bfloat16",
        model_options={"temperature": 0},
        timeout_seconds=60,
        max_output_tokens=4096,
        max_output_tokens_option="max_tokens",
        max_output_bytes=262144,
        pricing=Pricing(None, None, "not-applicable", "not-applicable"),
        max_cost_usd=None,
        external_upload_consent=False,
        hardware={
            "device": "test",
            "amortized_hourly_cost_usd": 1.0,
            "model_load_ms": 1,
            "peak_memory_bytes": 1024,
            "runtime_model_id": "qwen3.5:35b-a3b",
            "runtime_artifact_sha256": "a" * 64,
        },
    )


def plan_with_approved_hosted_artifact(
    *,
    model: str,
    provider: str,
    runtime: str,
    runtime_version: str,
    quantization: str,
    provider_model_revision: str,
) -> dict[str, object]:
    plan = benchmark.load_plan()
    candidate = plan["candidates"]["deepseek-v4-flash-hosted"]
    candidate["runtime_artifact"] = {
        "status": "approved",
        "provider": provider,
        "runtime": runtime,
        "runtime_version": runtime_version,
        "model": model,
        "provider_model_revision": provider_model_revision,
        "quantization": quantization,
        "max_output_tokens_option": "max_tokens",
        "source_binding": {
            "model_artifact": candidate["model_artifact"],
            "model_revision": candidate["model_revision"],
            "evidence_url": "https://provider.example/model-version",
            "evidence_as_of": "2026-08-30",
        },
    }
    return plan


def approved_hosted_config(plan: dict[str, object]) -> ModelRunConfig:
    return benchmark.prepare_run_config(
        plan,
        candidate_id="deepseek-v4-flash-hosted",
        model="hosted-deepseek",
        provider="test-provider",
        runtime="test-runtime",
        runtime_version="1.0.0",
        quantization="fp8",
        model_options={"temperature": 0},
        timeout_seconds=60,
        max_output_tokens=4096,
        max_output_tokens_option="max_tokens",
        max_output_bytes=262144,
        pricing=Pricing(
            1.0, 1.0, "https://provider.example/pricing", "2026-08-30",
        ),
        max_cost_usd=100.0,
        external_upload_consent=True,
        hardware={
            "endpoint": "fixture",
            "provider_model_revision": "deepseek-v4-flash-test-revision",
        },
    )


class AgentChangeReplayTests(unittest.TestCase):
    def test_pre_inference_validation_accepts_only_complete_frozen_binding(self) -> None:
        plan = plan_with_approved_qwen_artifact()
        config = approved_qwen_config(plan)
        self.assertEqual(
            benchmark.validate_model_run_config_against_plan(plan, config),
            "qwen35-a3b-local",
        )

        mutations = (
            {"candidate_id": "devstral-small-2-local"},
            {"model": "other-alias"},
            {"model_artifact": "other/model"},
            {"model_revision": "b" * 40},
            {"runtime": "other-runtime"},
            {"runtime_version": "2.0"},
            {"context_limit": config.context_limit - 1},
            {"timeout_seconds": config.timeout_seconds + 1},
            {"max_output_tokens": config.max_output_tokens - 1},
            {"max_output_bytes": config.max_output_bytes - 1},
            {"max_output_tokens_option": "unbounded_output"},
            {"model_options": {"temperature": 0, "extra_option": True}},
            {"model_options": {"temperature": False}},
            {"quantization": "different"},
            {"provider": "hosted-provider"},
            {"external_upload_consent": True},
            {"hardware": {
                **config.hardware,
                "runtime_artifact_sha256": "c" * 64,
            }},
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                tampered = ModelRunConfig.from_dict({
                    **config.to_dict(),
                    **mutation,
                })
                with self.assertRaises(ValueError):
                    benchmark.validate_model_run_config_against_plan(plan, tampered)

    def test_pre_inference_validation_rejects_hosted_binding_changes(self) -> None:
        plan = plan_with_approved_hosted_artifact(
            model="hosted-deepseek",
            provider="test-provider",
            runtime="test-runtime",
            runtime_version="1.0.0",
            quantization="fp8",
            provider_model_revision="deepseek-v4-flash-test-revision",
        )
        config = approved_hosted_config(plan)
        self.assertEqual(
            benchmark.validate_model_run_config_against_plan(plan, config),
            "deepseek-v4-flash-hosted",
        )
        mutations = (
            {"model": "gemini-3.7-flash"},
            {"provider": "other-provider"},
            {"runtime_version": "2.0.0"},
            {"quantization": "different"},
            {"max_output_tokens_option": "unbounded_output"},
            {"model_options": {"temperature": 0, "extra_option": True}},
            {"hardware": {
                **config.hardware,
                "provider_model_revision": "different-revision",
            }},
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                tampered = ModelRunConfig.from_dict({
                    **config.to_dict(),
                    **mutation,
                })
                with self.assertRaises(ValueError):
                    benchmark.validate_model_run_config_against_plan(plan, tampered)

    def test_live_eval_rejects_unregistered_config_before_any_runtime_work(self) -> None:
        hosted_plan = plan_with_approved_hosted_artifact(
            model="hosted-deepseek",
            provider="test-provider",
            runtime="test-runtime",
            runtime_version="1.0.0",
            quantization="fp8",
            provider_model_revision="deepseek-v4-flash-test-revision",
        )
        configs = (
            approved_qwen_config(plan_with_approved_qwen_artifact()),
            approved_hosted_config(hosted_plan),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for config in configs:
                with self.subTest(locality=config.locality):
                    config_path = root / f"{config.locality}-config.json"
                    artifacts.write_private_json(config_path, config.to_dict())
                    with (
                        mock.patch(
                            "evals.agent_changes.live_eval.benchmark_cost_preflight"
                        ) as cost_preflight,
                        mock.patch(
                            "evals.agent_changes.live_eval.attest_runtime"
                        ) as runtime_attestation,
                        mock.patch(
                            "evals.agent_changes.live_eval.watch_files."
                            "validate_model_run_config_privacy"
                        ) as privacy_validation,
                        mock.patch(
                            "evals.agent_changes.live_eval.subprocess.Popen"
                        ) as watcher,
                        self.assertRaisesRegex(
                            ValueError, "no approved runtime artifact"
                        ),
                    ):
                        live_eval.main([
                            "temporal",
                            "--model-run-config", str(config_path),
                            "--destination", str(root / "replay"),
                            "--results-directory", str(root / "results"),
                        ])
                    cost_preflight.assert_not_called()
                    runtime_attestation.assert_not_called()
                    privacy_validation.assert_not_called()
                    watcher.assert_not_called()

    def test_watcher_command_pins_the_validated_config_bytes(self) -> None:
        plan = plan_with_approved_qwen_artifact()
        config = approved_qwen_config(plan)
        args = argparse.Namespace(
            destination=Path("replay"),
            model=config.model,
            debounce=3,
            review_timeout=config.timeout_seconds,
            reasoning_effort="auto",
            log=False,
            model_run_config=Path("approved.json"),
            model_run_config_sha256=model_run_config_sha256(config),
            benchmark_plan=Path("approved-plan.json"),
            benchmark_plan_sha256=benchmark.plan_sha256(plan),
        )

        command = live_eval.watcher_command(args)

        digest_index = command.index("--model-run-config-sha256")
        self.assertEqual(
            command[digest_index + 1], model_run_config_sha256(config),
        )
        plan_digest_index = command.index("--benchmark-plan-sha256")
        self.assertEqual(
            command[plan_digest_index + 1], benchmark.plan_sha256(plan),
        )

    def test_private_artifact_write_does_not_repermission_existing_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            parent = Path(temporary_directory)
            parent.chmod(0o755)
            destination = parent / "artifact.json"
            artifacts.write_private_json(destination, {"safe": True})
            self.assertEqual(parent.stat().st_mode & 0o777, 0o755)
            self.assertEqual(destination.stat().st_mode & 0o777, 0o600)

    def test_manifest_has_complete_taxonomy_metadata_and_fixture_files(self) -> None:
        manifest = replay.load_manifest()
        self.assertEqual(manifest["version"], 3)
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
            self.assertEqual(first_contents, [path.read_bytes() for path in replayed])

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
            debounce_seconds=3.0, inter_file_delay_seconds=0.25,
        )
        self.assertEqual(configuration["model"], "test-model")
        self.assertEqual(configuration["model_options"], {"reasoning_effort": "high"})
        self.assertEqual(configuration["batching"], {
            "debounce_seconds": 3.0,
            "inter_file_delay_seconds": 0.25,
        })
        self.assertEqual(configuration["prompt"]["text"], "frozen prompt")
        self.assertEqual(
            configuration["prompt"]["sha256"],
            hashlib.sha256(b"frozen prompt").hexdigest(),
        )
        self.assertEqual(configuration["prompt"]["revision"], watch_files.PROMPT_REVISION)
        self.assertEqual(configuration["schema"]["value"], watch_files.REVIEW_SCHEMA)
        self.assertEqual(configuration["schema"]["revision"], watch_files.REVIEW_SCHEMA_REVISION)
        self.assertEqual(
            configuration["schema"]["finding_file_binding"],
            "per-batch-provider-visible-enum",
        )
        self.assertEqual(configuration["fixture"]["revision"], 3)
        self.assertEqual(configuration["fixture"]["case_ids"], [case["id"] for case in cases])

    def test_wait_for_outcome_retains_raw_schema_valid_response_and_latency(self) -> None:
        output: queue.Queue[str] = queue.Queue()
        response = {"findings": [provider_finding("service.py", "Concrete failure path.")]}
        raw_response = json.dumps(response) + "\n"
        runtime_attestation = {
            "runtime": {"name": "llm-ollama", "version": "1.0"},
            "model_registry_entry": "fixture",
        }
        output.put("Reviewing 1 changed file(s): service.py\n")
        output.put(json.dumps({"quodet_evaluation_event": {
            "status": "success", "returncode": 0,
            "raw_response": raw_response, "stderr": "",
            "runtime_attestation": runtime_attestation,
            "model_attempted": True,
        }}) + "\n")
        outcome = live_eval.wait_for_outcome(output, timeout=1)
        self.assertEqual(outcome.status, "schema-valid")
        self.assertEqual(outcome.raw_response, raw_response)
        self.assertEqual(outcome.parsed_response, response)
        self.assertEqual(outcome.runtime_attestation, runtime_attestation)
        self.assertIs(outcome.model_attempted, True)
        self.assertGreaterEqual(outcome.latency_ms, 0)

    def test_live_schema_rejects_file_outside_the_case_enum(self) -> None:
        output: queue.Queue[str] = queue.Queue()
        response = {"findings": [provider_finding("service.py", "Wrong prefix.")]}
        output.put(json.dumps({"quodet_evaluation_event": {
            "status": "success",
            "returncode": 0,
            "raw_response": json.dumps(response),
            "stderr": "",
        }}) + "\n")

        outcome = live_eval.wait_for_outcome(
            output,
            timeout=1,
            allowed_files=("03_cross_file_units/service.py",),
        )

        self.assertEqual(outcome.status, "schema-error")
        self.assertIn("per-case enum", outcome.error or "")

    def test_live_grounding_uses_case_prefixed_paths_and_visible_symbols(self) -> None:
        case = replay.case_by_id(
            replay.load_manifest(), "02_source_and_test_boundary"
        )
        source_path, test_path = live_eval.case_provider_paths(case)
        for suggested_fix in (
            f"Extend {test_path} with a second-page assertion.",
            "Extend test_first_page_starts_with_first_item with a second page.",
        ):
            finding = provider_finding(source_path, "Pagination skips an item.")
            finding["suggested_fix"] = suggested_fix
            provider = live_eval.ProviderOutcome(
                status="schema-valid",
                latency_ms=1,
                transcript="",
                raw_response=json.dumps({"findings": [finding]}),
                parsed_response={"findings": [finding]},
                error=None,
            )

            with self.subTest(suggested_fix=suggested_fix):
                outcome = live_eval.case_outcome(case, provider)
                grounding = outcome["diagnostics"]["recommendation_grounding"]
                self.assertEqual(outcome["status"], "schema-valid")
                self.assertEqual(grounding["failures"], 0)

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
            self.assertEqual(artifacts[0].stat().st_mode & 0o777, 0o600)
            self.assertEqual((root / "results").stat().st_mode & 0o777, 0o700)
            self.assertEqual(len(artifact["cases"]), 2)
            self.assertEqual(artifact["cases"][0]["case_id"], manifest["cases"][0]["id"])
            self.assertEqual(artifact["cases"][1]["status"], "harness-error")

    def test_filename_match_is_diagnostic_not_true_positive(self) -> None:
        case = replay.case_by_id(replay.load_manifest(), "12_semantically_invalid_external_value")
        provider = live_eval.ProviderOutcome(
            status="schema-valid", latency_ms=12, transcript="", raw_response="{}",
            parsed_response={"findings": [provider_finding(
                live_eval.case_provider_paths(case)[0], "Wrong diagnosis"
            )]},
            error=None,
        )
        outcome = live_eval.case_outcome(case, provider)
        self.assertTrue(outcome["diagnostics"]["filename_match"])

        run = raw_run([outcome])
        adjudication = {
            "run_id": "run-1", "fixture_revision": 3,
            "cases": {case["id"]: {"findings": [{
                "finding_index": 0, "verdict": "false-positive",
                "expected_finding_id": None, "rationale": "Right file, wrong failure path",
            }]}},
        }
        metrics = scoring.score_run(run, adjudication)
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (0, 1, 1))

    def test_recommendation_grounding_scores_separately_from_detection(self) -> None:
        case = replay.case_by_id(replay.load_manifest(), "01_obvious_runtime")
        finding = provider_finding(
            live_eval.case_provider_paths(case)[0],
            "The undefined name raises for non-empty input.",
        )
        finding["suggested_fix"] = (
            "Use values and preserve the existing non-empty regression test."
        )
        provider = live_eval.ProviderOutcome(
            status="schema-valid",
            latency_ms=12,
            transcript="",
            raw_response="{}",
            parsed_response={"findings": [finding]},
            error=None,
        )

        outcome = live_eval.case_outcome(case, provider)
        grounding = outcome["diagnostics"]["recommendation_grounding"]
        self.assertEqual(grounding["failures"], 1)
        self.assertEqual(
            grounding["results"][0]["violations"][0]["code"],
            "unsupported-existing-test-claim",
        )

        metrics = scoring.score_run(raw_run([outcome]), None)
        self.assertEqual(
            metrics["recommendation_grounding"],
            {"evaluated": 1, "grounded": 0, "failures": 1, "grounded_rate": 0.0},
        )
        self.assertIsNone(metrics["tp"])

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
            "run_id": "run-1", "fixture_revision": 3,
            "cases": {
                "defect": {"findings": [{
                    "finding_index": 0, "verdict": "true-positive",
                    "expected_finding_id": "expected-1", "fix_quality": "actionable",
                    "rationale": "Matches behavior",
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
        self.assertEqual(metrics["clean_control_false_positive_rate"], 1.0)
        self.assertEqual(metrics["fix_quality_score"], 1.0)

    def test_schema_failure_counts_expected_findings_as_misses(self) -> None:
        failed = {
            "case_id": "failed", "evaluation_split": "calibration",
            "failure_families": ["state/lifecycle"],
            "expected_finding_ids": ["one", "two"], "status": "schema-error",
            "parsed_response": None,
        }
        run = raw_run([failed])
        adjudication = {"run_id": "run-1", "fixture_revision": 3, "cases": {}}
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

    def test_frozen_benchmark_plan_pins_candidates_contract_and_decision_rule(self) -> None:
        plan = benchmark.load_plan()
        self.assertEqual(
            set(plan["candidate_order"]),
            {
                "qwen35-a3b-local",
                "deepseek-v4-flash-hosted",
                "devstral-small-2-local",
            },
        )
        self.assertTrue(plan["decision_rule"]["frozen_before_holdout"])
        for candidate in plan["candidates"].values():
            self.assertEqual(len(candidate["model_revision"]), 40)
            self.assertIn(candidate["required_locality"], {"local", "hosted"})
            if candidate["required_locality"] == "local":
                self.assertIsNone(candidate["runtime_artifact"]["runtime_model_id"])
                self.assertIsNone(
                    candidate["runtime_artifact"]["runtime_artifact_sha256"]
                )
            self.assertEqual(candidate["runtime_artifact"]["status"], "unregistered")

    def test_hosted_benchmark_config_requires_consent_and_cost_cap(self) -> None:
        plan = plan_with_approved_hosted_artifact(
            model="deepseek-v4-flash",
            provider="example-provider",
            runtime="llm-openai-compatible",
            runtime_version="1.2.3",
            quantization="fp8",
            provider_model_revision="deepseek-v4-flash-2026-08-01",
        )
        common = dict(
            plan=plan,
            candidate_id="deepseek-v4-flash-hosted",
            model="deepseek-v4-flash",
            provider="example-provider",
            runtime="llm-openai-compatible",
            runtime_version="1.2.3",
            quantization="fp8",
            model_options={"temperature": 0},
            timeout_seconds=60,
            max_output_tokens=4096,
            max_output_tokens_option="max_tokens",
            max_output_bytes=262144,
            pricing=Pricing(0.10, 0.20, "https://provider.example/pricing", "2026-08-30"),
            hardware={
                "region": "us-east",
                "provider_model_revision": "deepseek-v4-flash-2026-08-01",
            },
        )
        with self.assertRaisesRegex(ValueError, "external-upload consent"):
            benchmark.prepare_run_config(
                **common, max_cost_usd=1.0, external_upload_consent=False
            )
        with self.assertRaisesRegex(ValueError, "max-cost-usd"):
            benchmark.prepare_run_config(
                **common, max_cost_usd=None, external_upload_consent=True
            )
        invalid_provenance = dict(common)
        invalid_provenance["pricing"] = Pricing(
            0.10, 0.20, "not-applicable", "not-applicable"
        )
        with self.assertRaisesRegex(ValueError, "HTTPS pricing source"):
            benchmark.prepare_run_config(
                **invalid_provenance,
                max_cost_usd=1.0,
                external_upload_consent=True,
            )
        wrong_model = dict(common)
        wrong_model["model"] = "gemini-3.7-flash"
        with self.assertRaisesRegex(ValueError, "hosted execution identity"):
            benchmark.prepare_run_config(
                **wrong_model,
                max_cost_usd=1.0,
                external_upload_consent=True,
            )

    def test_local_runtime_attestation_rejects_hosted_plugin_and_alias(self) -> None:
        plan = plan_with_approved_qwen_artifact()
        common = dict(
            plan=plan, candidate_id="qwen35-a3b-local", model="qwen-eval",
            provider="local", runtime_version="1.0", quantization="bfloat16",
            model_options={"temperature": 0}, timeout_seconds=60,
            max_output_tokens=4096, max_output_tokens_option="max_tokens",
            max_output_bytes=262144,
            pricing=Pricing(None, None, "not-applicable", "not-applicable"),
            max_cost_usd=None, external_upload_consent=False,
            hardware={
                "device": "test", "amortized_hourly_cost_usd": 1.0,
                "model_load_ms": 1, "peak_memory_bytes": 1024,
                "runtime_model_id": "qwen3.5:35b-a3b",
                "runtime_artifact_sha256": "a" * 64,
            },
        )
        with self.assertRaisesRegex(ValueError, "require runtime llm-ollama"):
            benchmark.prepare_run_config(**common, runtime="llm-gemini")

        config = benchmark.prepare_run_config(**common, runtime="llm-ollama")
        command_results = [
            mock.Mock(returncode=0, stdout="llm, version 0.33\n"),
            mock.Mock(returncode=0, stdout=json.dumps([
                {"name": "llm-ollama", "version": "1.0"}
            ])),
            mock.Mock(
                returncode=0,
                stdout=(
                    "Ollama: devstral-small-2:24b "
                    "(aliases: qwen-eval)\n"
                ),
            ),
        ]
        with (
            mock.patch("evals.agent_changes.live_eval.subprocess.run", side_effect=command_results),
            self.assertRaisesRegex(ValueError, "does not resolve"),
        ):
            live_eval.attest_runtime(config)

    def test_local_runtime_attestation_binds_exact_ollama_blob(self) -> None:
        plan = plan_with_approved_qwen_artifact()
        expected_digest = "a" * 64
        config = benchmark.prepare_run_config(
            plan, candidate_id="qwen35-a3b-local", model="qwen-eval",
            provider="local", runtime="llm-ollama", runtime_version="1.0",
            quantization="bfloat16", model_options={"temperature": 0},
            timeout_seconds=60, max_output_tokens=4096,
            max_output_tokens_option="max_tokens", max_output_bytes=262144,
            pricing=Pricing(None, None, "not-applicable", "not-applicable"),
            max_cost_usd=None, external_upload_consent=False,
            hardware={
                "device": "test", "amortized_hourly_cost_usd": 1.0,
                "model_load_ms": 1, "peak_memory_bytes": 1024,
                "runtime_model_id": "qwen3.5:35b-a3b",
                "runtime_artifact_sha256": expected_digest,
            },
        )
        command_results = [
            mock.Mock(returncode=0, stdout="llm, version 0.33\n"),
            mock.Mock(returncode=0, stdout=json.dumps([
                {"name": "llm-ollama", "version": "1.0"}
            ])),
            mock.Mock(
                returncode=0,
                stdout="Ollama: qwen3.5:35b-a3b (aliases: qwen-eval)\n",
            ),
            mock.Mock(
                returncode=0,
                stdout=f"FROM /models/blobs/sha256-{expected_digest}\n",
            ),
        ]
        with mock.patch(
            "evals.agent_changes.live_eval.subprocess.run",
            side_effect=command_results,
        ):
            attestation = live_eval.attest_runtime(config)
        self.assertEqual(attestation["local_model"], {
            "runtime_model_id": "qwen3.5:35b-a3b",
            "runtime_artifact_sha256": expected_digest,
        })

        command_results[-1] = mock.Mock(
            returncode=0,
            stdout=f"FROM /models/blobs/sha256-{'b' * 64}\n",
        )
        with (
            mock.patch(
                "evals.agent_changes.live_eval.subprocess.run",
                side_effect=command_results,
            ),
            self.assertRaisesRegex(ValueError, "blob differs"),
        ):
            live_eval.attest_runtime(config)

    def test_runtime_attestation_commands_share_a_bounded_deadline(self) -> None:
        config = approved_qwen_config(plan_with_approved_qwen_artifact())
        timeout = subprocess.TimeoutExpired(["llm", "--version"], 10)
        with (
            mock.patch(
                "evals.agent_changes.live_eval.subprocess.run",
                side_effect=timeout,
            ) as command,
            self.assertRaisesRegex(ValueError, "runtime attestation timed out"),
        ):
            live_eval.attest_runtime(config)

        invoked_timeout = command.call_args.kwargs["timeout"]
        self.assertGreater(invoked_timeout, 0)
        self.assertLessEqual(
            invoked_timeout, live_eval.MAX_RUNTIME_ATTESTATION_SECONDS,
        )
        self.assertLessEqual(invoked_timeout, config.timeout_seconds)

    def test_benchmark_scorecard_retains_failed_attempts_and_no_auto_selection(self) -> None:
        plan = plan_with_approved_qwen_artifact()
        config = benchmark.prepare_run_config(
            plan,
            candidate_id="qwen35-a3b-local",
            model="qwen-eval",
            provider="local",
            runtime="llm-ollama",
            runtime_version="1.0",
            quantization="bfloat16",
            model_options={"temperature": 0},
            timeout_seconds=60,
            max_output_tokens=4096,
            max_output_tokens_option="max_tokens",
            max_output_bytes=262144,
            pricing=Pricing(None, None, "not-applicable", "not-applicable"),
            max_cost_usd=None,
            external_upload_consent=False,
            hardware={
                "device": "test", "amortized_hourly_cost_usd": 1.0,
                "model_load_ms": 1, "peak_memory_bytes": 1024,
                "runtime_model_id": "qwen3.5:35b-a3b",
                "runtime_artifact_sha256": "a" * 64,
            },
        )
        fixture_case = replay.load_manifest()["cases"][0]
        runtime_attestation = {
            "llm_cli_version": "fixture",
            "runtime": {"name": "llm-ollama", "version": "1.0"},
            "model_registry_entry": (
                "Ollama: qwen3.5:35b-a3b (aliases: qwen-eval)"
            ),
            "local_model": {
                "runtime_model_id": "qwen3.5:35b-a3b",
                "runtime_artifact_sha256": "a" * 64,
            },
        }
        cases = [
            {
                "case_id": fixture_case["id"],
                "evaluation_split": fixture_case["evaluation_split"],
                "failure_families": fixture_case["failure_families"],
                "expected_finding_ids": [
                    finding["id"] for finding in fixture_case["expected_findings"]
                ],
                "expected_findings": fixture_case["expected_findings"],
                "status": "timeout",
                "latency_ms": 30000, "parsed_response": None,
                "input_tokens": None, "output_tokens": None, "cost_usd": None,
                "model_attempted": True,
                "model_attempt_count": 1,
                "effective_model_config": config.to_dict(),
                "runtime_attestation": runtime_attestation,
            }
        ]
        run = raw_run(cases)
        run["configuration"] = live_eval.evaluation_configuration(
            model=config.model,
            reasoning_effort=None,
            prompt=watch_files.DEFAULT_PROMPT,
            fixture_revision=3,
            cases=[],
            debounce_seconds=3.0,
            inter_file_delay_seconds=0.25,
            model_run_config=config,
            benchmark_plan=plan,
            runtime_attestation=runtime_attestation,
        )
        adjudication = {"run_id": "run-1", "fixture_revision": 3, "cases": {}}
        run["metrics"] = scoring.score_run(run, adjudication)
        run["adjudication"] = adjudication
        run["adjudication_sha256"] = scoring.adjudication_sha256(adjudication)
        scorecard = benchmark.build_scorecard(plan, [run])
        summary = scorecard["candidates"]["qwen35-a3b-local"]
        self.assertEqual(summary["attempted_cases"], 1)
        self.assertEqual(summary["status_counts"], {"timeout": 1})
        self.assertEqual(summary["status"], "incomplete-run")
        self.assertIsNone(scorecard["decision"]["selection"])
        boolean_temperature = json.loads(json.dumps(run))
        boolean_temperature["configuration"]["benchmark"][
            "model_run_config"
        ]["model_options"]["temperature"] = False
        with self.assertRaisesRegex(ValueError, "model options differ"):
            benchmark.build_scorecard(plan, [boolean_temperature])
        missing_case_attestation = json.loads(json.dumps(run))
        del missing_case_attestation["cases"][0]["runtime_attestation"]
        with self.assertRaisesRegex(ValueError, "missing live runtime attestation"):
            benchmark.build_scorecard(plan, [missing_case_attestation])
        tampered_case_attestation = json.loads(json.dumps(run))
        tampered_case_attestation["cases"][0]["runtime_attestation"][
            "local_model"
        ]["runtime_artifact_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "local-model attestation differs"):
            benchmark.build_scorecard(plan, [tampered_case_attestation])
        cli_version_drift = json.loads(json.dumps(run))
        cli_version_drift["cases"][0]["runtime_attestation"][
            "llm_cli_version"
        ] = "different"
        with self.assertRaisesRegex(ValueError, "CLI version differs"):
            benchmark.build_scorecard(plan, [cli_version_drift])
        wrong_effective_config = json.loads(json.dumps(run))
        wrong_effective_config["cases"][0]["effective_model_config"][
            "runtime_version"
        ] = "different"
        with self.assertRaisesRegex(ValueError, "effective model config differs"):
            benchmark.build_scorecard(plan, [wrong_effective_config])
        boolean_attempt_count = json.loads(json.dumps(run))
        boolean_attempt_count["cases"][0]["model_attempt_count"] = True
        with self.assertRaisesRegex(ValueError, "single model attempt"):
            benchmark.build_scorecard(plan, [boolean_attempt_count])
        startup_only = json.loads(json.dumps(run))
        del startup_only["cases"][0]["model_attempted"]
        with self.assertRaisesRegex(ValueError, "missing model-attempt provenance"):
            benchmark.build_scorecard(plan, [startup_only])
        pre_inference_failure = json.loads(json.dumps(run))
        pre_inference_case = pre_inference_failure["cases"][0]
        pre_inference_case["status"] = "provider-error"
        pre_inference_case["model_attempted"] = False
        pre_inference_case["model_attempt_count"] = None
        pre_inference_case["runtime_attestation"] = None
        pre_inference_case["effective_model_config"] = None
        pre_inference_failure["metrics"] = scoring.score_run(
            pre_inference_failure, adjudication
        )
        benchmark.validate_run_against_plan(plan, pre_inference_failure)
        run["metrics"] = dict(run["metrics"])
        run["metrics"]["fn"] = 0
        with self.assertRaisesRegex(ValueError, "metrics do not match"):
            benchmark.build_scorecard(plan, [run])

    def test_hosted_suite_preflight_enforces_experiment_wide_cap(self) -> None:
        plan = plan_with_approved_hosted_artifact(
            model="hosted-deepseek",
            provider="test-provider",
            runtime="test-runtime",
            runtime_version="1.0.0",
            quantization="fp8",
            provider_model_revision="deepseek-v4-flash-test-revision",
        )
        config = benchmark.prepare_run_config(
            plan,
            candidate_id="deepseek-v4-flash-hosted",
            model="hosted-deepseek",
            provider="test-provider",
            runtime="test-runtime",
            runtime_version="1.0.0",
            quantization="fp8",
            model_options={"temperature": 0},
            timeout_seconds=60,
            max_output_tokens=4096,
            max_output_tokens_option="max_tokens",
            max_output_bytes=262144,
            pricing=Pricing(1.0, 1.0, "https://provider.example/pricing", "2026-08-30"),
            max_cost_usd=2.0,
            external_upload_consent=True,
            hardware={
                "endpoint": "fixture",
                "provider_model_revision": "deepseek-v4-flash-test-revision",
            },
        )
        cases = replay.load_manifest()["cases"]
        with self.assertRaisesRegex(ValueError, "experiment cap"):
            live_eval.benchmark_cost_preflight(config, cases)

    def test_candidate_decision_metrics_exclude_calibration(self) -> None:
        run = {
            "configuration": {"benchmark": {"model_run_config": {
                "hardware": {"device": "fixture"}, "max_cost_usd": None,
            }}},
            "cases": [
                {
                    "case_id": "cal", "evaluation_split": "calibration",
                    "status": "schema-valid", "latency_ms": 1,
                    "input_tokens": None, "output_tokens": None, "cost_usd": None,
                },
                {
                    "case_id": "hold", "evaluation_split": "holdout",
                    "status": "schema-valid", "latency_ms": 100,
                    "input_tokens": None, "output_tokens": None, "cost_usd": None,
                },
                {
                    "case_id": "clean", "evaluation_split": "clean-control",
                    "status": "schema-valid", "latency_ms": 200,
                    "input_tokens": None, "output_tokens": None, "cost_usd": None,
                },
            ],
            "metrics": {
                "adjudication_status": "complete",
                "tp": 100, "fp": 0, "fn": 0,
                "schema_valid_rate": 1.0,
                "fix_quality_score": 1.0,
                "clean_control_false_positive_rate": 0.25,
                "by_split": {}, "by_family": {},
                "split_metrics": {"holdout": {
                    "finding_precision": 0.6,
                    "finding_recall": 0.4,
                    "fix_quality_score": 0.5,
                }},
            },
        }
        summary = benchmark.summarize_run(
            run, required_case_ids={"cal", "hold", "clean"}
        )
        self.assertEqual(summary["finding_precision"], 0.6)
        self.assertEqual(summary["finding_recall"], 0.4)
        self.assertEqual(summary["fix_quality_score"], 0.5)
        self.assertEqual(summary["clean_control_false_positive_rate"], 0.25)
        self.assertEqual(summary["latency_ms"], {"p50": 150.0, "p95": 200.0})
        self.assertEqual(
            summary["aggregate_report_only"]["finding_precision"], 1.0
        )


if __name__ == "__main__":
    unittest.main()
