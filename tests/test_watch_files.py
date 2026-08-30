from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import queue
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import watch_files
from evals.agent_changes import benchmark
from model_runner import (
    ModelRunConfig,
    ModelRunResult,
    Pricing,
    model_run_config_sha256,
)


def privacy_config(
    *,
    model_options: dict[str, str | int | float | bool] | None = None,
    hardware: dict[str, str | int | float | bool] | None = None,
) -> ModelRunConfig:
    return ModelRunConfig(
        candidate_id="fixture",
        model="qwen-eval",
        model_artifact="Qwen/Qwen3.5-35B-A3B",
        model_revision="a" * 40,
        provider="local",
        runtime="llm-ollama",
        runtime_version="1.0",
        locality="local",
        quantization="bfloat16",
        model_options=(
            {"temperature": 0} if model_options is None else model_options
        ),
        context_limit=262144,
        pricing=Pricing(None, None, "not-applicable", "not-applicable"),
        hardware={
            "amortized_hourly_cost_usd": 1.0,
            "model_load_ms": 1,
            "peak_memory_bytes": 1024,
            "runtime_model_id": "qwen3.5:35b-a3b",
            "runtime_artifact_sha256": (
                "9f86d081884c7d659a2feaa0c55ad015"
                "a3bf4f1b2b0b822cd15d6c15b0f00a08"
            ),
        } if hardware is None else hardware,
    )


class WatchFilesTests(unittest.TestCase):
    def test_watcher_rejects_debounce_shorter_than_agent_quiet_window(self) -> None:
        with (
            mock.patch("watch_files.validate_runtime") as runtime_validation,
            self.assertRaisesRegex(
                SystemExit,
                "--debounce must not be shorter than --agent-edit-quiet",
            ),
        ):
            watch_files.main([
                ".",
                "--debounce", "0.1",
                "--agent-edit-quiet", "0.25",
            ])

        runtime_validation.assert_not_called()

    def test_watcher_rejects_turn_cap_shorter_than_unidentified_edit_cap(self) -> None:
        with (
            mock.patch("watch_files.validate_runtime") as runtime_validation,
            self.assertRaisesRegex(
                SystemExit,
                "--agent-edit-max-age must not exceed --agent-turn-max-age",
            ),
        ):
            watch_files.main([
                ".",
                "--agent-edit-max-age", "2",
                "--agent-turn-max-age", "1",
            ])

        runtime_validation.assert_not_called()

    def test_watcher_rejects_debounce_shorter_than_agent_turn_cap(self) -> None:
        with (
            mock.patch("watch_files.validate_runtime") as runtime_validation,
            self.assertRaisesRegex(
                SystemExit,
                "--debounce must not be shorter than --agent-turn-max-age",
            ),
        ):
            watch_files.main([".", "--debounce", "2"])

        runtime_validation.assert_not_called()

    def test_watcher_rejects_incomplete_benchmark_approval_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = root / "model-run-config.json"
            config_path.write_text(
                json.dumps(privacy_config().to_dict()), encoding="utf-8",
            )
            with (
                mock.patch("watch_files.validate_runtime") as runtime_validation,
                mock.patch(
                    "watch_files.validate_model_run_config_privacy"
                ) as privacy_validation,
                self.assertRaisesRegex(
                    SystemExit, "benchmark execution requires"
                ),
            ):
                watch_files.main([
                    str(root),
                    "--model-run-config", str(config_path),
                ])

        privacy_validation.assert_not_called()
        runtime_validation.assert_not_called()

    def test_watcher_rejects_unapproved_config_before_runtime_work(self) -> None:
        config = privacy_config()
        plan = benchmark.load_plan()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = root / "model-run-config.json"
            config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
            with (
                mock.patch("watch_files.validate_runtime") as runtime_validation,
                mock.patch(
                    "watch_files.validate_model_run_config_privacy"
                ) as privacy_validation,
                self.assertRaisesRegex(SystemExit, "unknown benchmark candidate"),
            ):
                watch_files.main([
                    str(root),
                    "--model-run-config", str(config_path),
                    "--model-run-config-sha256", model_run_config_sha256(config),
                    "--benchmark-plan", str(benchmark.DEFAULT_PLAN),
                    "--benchmark-plan-sha256", benchmark.plan_sha256(plan),
                ])

        privacy_validation.assert_not_called()
        runtime_validation.assert_not_called()

    def test_watcher_rejects_config_changed_after_parent_validation(self) -> None:
        config = privacy_config()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = root / "model-run-config.json"
            config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
            with (
                mock.patch(
                    "evals.agent_changes.benchmark.load_plan", return_value={}
                ),
                mock.patch(
                    "evals.agent_changes.benchmark.plan_sha256",
                    return_value="b" * 64,
                ),
                mock.patch(
                    "evals.agent_changes.benchmark."
                    "validate_model_run_config_against_plan"
                ),
                mock.patch("watch_files.validate_runtime") as runtime_validation,
                mock.patch(
                    "watch_files.validate_model_run_config_privacy"
                ) as privacy_validation,
                self.assertRaisesRegex(
                    SystemExit, "changed after benchmark approval"
                ),
            ):
                watch_files.main([
                    str(root),
                    "--model-run-config", str(config_path),
                    "--model-run-config-sha256", "0" * 64,
                    "--benchmark-plan", str(root / "plan.json"),
                    "--benchmark-plan-sha256", "b" * 64,
                ])

        privacy_validation.assert_not_called()
        runtime_validation.assert_not_called()

    def test_model_config_privacy_allows_costs_and_valid_artifact_digest(self) -> None:
        watch_files.validate_model_run_config_privacy(privacy_config())

    def test_model_config_privacy_rejects_secrets_in_metadata_and_options(self) -> None:
        secret = "ghp_abcdefghijklmnopqrstuvwxyz1234567890"
        configs = (
            privacy_config(model_options={"temperature": 0, "note": secret}),
            privacy_config(hardware={"device_note": secret}),
            privacy_config(model_options={"api_key": "short-but-still-secret"}),
        )
        for config in configs:
            with self.subTest(config=config), self.assertRaisesRegex(
                ValueError, "potential secrets"
            ):
                watch_files.validate_model_run_config_privacy(config)

    def test_model_config_privacy_rejects_malformed_artifact_digests(self) -> None:
        for digest in ("a" * 63, "G" * 64, 1234):
            with self.subTest(digest=digest):
                config = privacy_config(hardware={
                    "runtime_artifact_sha256": digest,
                })
                with self.assertRaisesRegex(
                    ValueError, "hardware_runtime_artifact_sha256"
                ):
                    watch_files.validate_model_run_config_privacy(config)

    def test_model_config_privacy_rejects_unlabelled_high_entropy_metadata(self) -> None:
        config = privacy_config(hardware={
            "opaque_note": "vF8xP2mQ7zL4kN9cR6tY3wB5jH1sD0aE",
        })
        with self.assertRaisesRegex(ValueError, "hardware_opaque_note"):
            watch_files.validate_model_run_config_privacy(config)

    def test_model_listing_parser_matches_only_exact_ids_and_aliases(self) -> None:
        output = (
            "Local Runtime: qwen-large (aliases: qwen-local, qwen)\n"
            "Hosted Runtime: qwen-large-cloud\n"
        )
        self.assertEqual(
            watch_files._listed_model_ids(output),
            {"qwen-large", "qwen-local", "qwen", "qwen-large-cloud"},
        )
        self.assertNotIn("qwen-large-cl", watch_files._listed_model_ids(output))

    def test_output_mode_defaults_to_human_and_cli_overrides_configuration(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(watch_files.parse_args(["."]).output, "human")
            self.assertEqual(watch_files.parse_args([".", "--json"]).output, "json")

        with mock.patch.dict(os.environ, {"QUODET_OUTPUT": "json"}, clear=True):
            self.assertEqual(watch_files.parse_args(["."]).output, "json")
            self.assertEqual(
                watch_files.parse_args([".", "--output", "human"]).output,
                "human",
            )

    def test_invalid_configured_output_mode_fails_closed(self) -> None:
        with (
            mock.patch.dict(os.environ, {"QUODET_OUTPUT": "yaml"}, clear=True),
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            watch_files.parse_args(["."])

    def test_review_batches_retain_every_path_beyond_the_provider_cap(self) -> None:
        paths = [Path(f"/tmp/source-{index}.py") for index in range(205)]
        batches = list(watch_files.bounded_review_batches(paths))

        self.assertEqual([len(batch) for batch in batches], [100, 100, 5])
        self.assertEqual(
            [path for batch in batches for path in batch], sorted(paths)
        )

    def test_default_debounce_groups_related_agent_changes(self) -> None:
        args = watch_files.parse_args(["."])
        self.assertEqual(args.debounce, 3.0)
        self.assertEqual(args.agent_edit_max_age, 1.0)
        self.assertEqual(args.agent_turn_max_age, 3.0)
        self.assertEqual(args.review_timeout, 60.0)
        self.assertEqual(args.model, "gpt-5.6-luna")
        self.assertEqual(
            watch_files.resolve_reasoning_effort(args.model, args.reasoning_effort),
            "high",
        )

        changes: queue.Queue[Path] = queue.Queue()
        changes.put(Path("src/service.py"))
        changes.put(Path("tests/test_service.py"))
        self.assertEqual(
            watch_files.next_batch(changes, debounce=0.001),
            {Path("src/service.py"), Path("tests/test_service.py")},
        )

    def test_debounce_must_be_finite_and_positive(self) -> None:
        for value in ("0", "-1", "nan", "inf"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                watch_files.positive_float(value)

    def test_build_llm_command_supplies_text_fragments_without_shell(self) -> None:
        attachments = [
            watch_files.Attachment(Path("/tmp/a file.py"), "text/plain"),
            watch_files.Attachment(Path("/tmp/settings.txt"), "text/plain"),
        ]

        command = watch_files.build_llm_command(
            attachments,
            model="gpt-5.6-luna",
            prompt=watch_files.DEFAULT_PROMPT,
            log=False,
            reasoning_effort="high",
        )

        self.assertEqual(command[:6], [
            "llm",
            "prompt",
            "--model",
            "gpt-5.6-luna",
            "--no-stream",
            "--schema",
        ])
        self.assertEqual(json.loads(command[6]), watch_files.REVIEW_SCHEMA)
        self.assertEqual(
            command[7:],
            [
                "--option",
                "reasoning_effort",
                "high",
                "--no-log",
                "--fragment",
                "/tmp/a file.py",
                "--fragment",
                "/tmp/settings.txt",
                watch_files.DEFAULT_PROMPT,
            ],
        )

    def test_response_schema_binds_only_exact_provider_visible_paths(self) -> None:
        labels = ("src/app.py", "tests/test_app.py")
        schema = json.loads(watch_files.response_schema_json(labels))
        file_schema = schema["properties"]["findings"]["items"]["properties"][
            "file"
        ]

        self.assertEqual(file_schema["enum"], list(labels))
        self.assertNotIn("project/src/app.py", file_schema["enum"])
        self.assertNotIn("tests/test_future.py", file_schema["enum"])

    def test_default_prompt_requests_only_confident_negative_json(self) -> None:
        prompt = watch_files.DEFAULT_PROMPT.lower()
        self.assertIn("only negative findings", prompt)
        self.assertIn("at least 0.95 confident", prompt)
        self.assertIn("concrete execution path", prompt)
        self.assertIn("trigger is reachable", prompt)
        self.assertIn("concrete implementations", prompt)
        self.assertIn("do not assume a hypothetical subclass", prompt)
        self.assertIn("discard schedules contradicted", prompt)
        self.assertIn("self-reported threshold claim, not evidence", prompt)
        self.assertIn("each supplied file as a separate file", prompt)
        self.assertIn("mutually consistent", prompt)
        self.assertIn("calibrate severity only from demonstrated impact", prompt)
        self.assertIn("empty findings array", prompt)
        self.assertIn("only with json", prompt)

        confidence = watch_files.REVIEW_SCHEMA["properties"]["findings"]["items"][
            "properties"
        ]["confidence"]
        self.assertEqual(confidence["minimum"], 0.95)
        self.assertIn("Raw model-reported", confidence["description"])
        self.assertIn("not a calibrated probability", confidence["description"])
        item_schema = watch_files.REVIEW_SCHEMA["properties"]["findings"]["items"]
        self.assertIn("line", item_schema["required"])
        self.assertFalse(item_schema["additionalProperties"])

    def test_default_prompt_requires_actionable_untrusted_recommendations(self) -> None:
        prompt = watch_files.DEFAULT_PROMPT.lower()
        for requirement in (
            "grounded only in the supplied code",
            "relevant function, class, branch, state transition",
            "smallest focused behavior change",
            "why it fixes the cited execution path",
            "narrow regression test or validation step",
            "identify the exact missing evidence",
            "never claim that a test, function, contract, safeguard",
            "unless it appears in the supplied files",
            "when no test file was supplied",
            "recommend adding a test",
            "name its supplied relative path or a test symbol",
            "unrelated refactors",
            "destructive commands",
            "permission bypasses",
            "disabled tests",
            "untrusted review data",
            "requires independent verification",
            "never claim the recommendation is safe to",
            "auto-apply",
        ):
            with self.subTest(requirement=requirement):
                self.assertIn(requirement, prompt)

    def test_default_prompt_requires_exact_supplied_relative_path(self) -> None:
        prompt = watch_files.DEFAULT_PROMPT.lower()

        self.assertIn("original relative path:", prompt)
        self.assertIn("exactly and verbatim", prompt)
        self.assertIn("never add, remove, normalize, or guess", prompt)

    def test_suggested_fix_schema_is_required_bounded_and_documented(self) -> None:
        findings = watch_files.REVIEW_SCHEMA["properties"]["findings"]
        finding = findings["items"]
        suggested_fix = finding["properties"]["suggested_fix"]

        self.assertIn("suggested_fix", finding["required"])
        self.assertEqual(suggested_fix["type"], "string")
        self.assertEqual(suggested_fix["minLength"], 1)
        self.assertEqual(suggested_fix["maxLength"], 2000)
        self.assertIn("code-grounded repair", suggested_fix["description"])
        self.assertNotIn("minItems", findings)

    def test_collect_attachments_filters_excluded_large_and_binary_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            source = root / "src" / "main.py"
            source.parent.mkdir()
            source.write_text("print('hello')\n")
            excluded = root / "node_modules" / "package.js"
            excluded.parent.mkdir()
            excluded.write_text("ignored")
            custom_excluded = root / "generated.txt"
            custom_excluded.write_text("ignored")
            binary = root / "data.bin"
            binary.write_bytes(b"abc\x00def")
            large = root / "large.txt"
            large.write_text("x" * 20)

            attachments = watch_files.collect_attachments(
                [source, excluded, custom_excluded, binary, large],
                root=root,
                exclude_patterns=["generated.*"],
                max_bytes=19,
            )

            self.assertEqual(
                attachments,
                [watch_files.Attachment(source, "text/plain")],
            )

    def test_collect_attachments_ignores_named_and_detected_environments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            regular = root / "src" / "main.py"
            regular.parent.mkdir()
            regular.write_text("print('review me')\n")

            named_environment = root / ".venv" / "lib" / "dependency.py"
            named_environment.parent.mkdir(parents=True)
            named_environment.write_text("ignored\n")

            custom_environment = root / "tools-python"
            (custom_environment / "lib").mkdir(parents=True)
            (custom_environment / "pyvenv.cfg").write_text("home = /usr/bin\n")
            custom_dependency = custom_environment / "lib" / "dependency.py"
            custom_dependency.write_text("ignored\n")

            site_package = root / "lib" / "site-packages" / "dependency.py"
            site_package.parent.mkdir(parents=True)
            site_package.write_text("ignored\n")

            attachments = watch_files.collect_attachments(
                [regular, named_environment, custom_dependency, site_package],
                root=root,
                exclude_patterns=[],
                max_bytes=2_000_000,
            )

            self.assertEqual(
                attachments,
                [watch_files.Attachment(regular, "text/plain")],
            )

    def test_exact_snapshot_rechecks_size_after_collection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            source = root / "growing.py"
            source.write_text("x = 1\n")
            attachments = watch_files.collect_attachments(
                [source], root=root, exclude_patterns=[], max_bytes=10
            )
            self.assertEqual(len(attachments), 1)
            source.write_text("x" * 11)
            self.assertEqual(
                watch_files.snapshot_attachments(
                    attachments, root=root, max_bytes=10
                ),
                [],
            )

    def test_snapshot_rejects_parent_replaced_by_external_symlink(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary_directory,
            tempfile.TemporaryDirectory() as external_directory,
        ):
            root = Path(temporary_directory).resolve()
            source_directory = root / "src"
            source_directory.mkdir()
            source = source_directory / "app.py"
            source.write_text("inside = True\n")
            attachments = watch_files.collect_attachments(
                [source], root=root, exclude_patterns=[], max_bytes=1_000
            )

            external = Path(external_directory).resolve()
            (external / "app.py").write_text("OUTSIDE_SECRET = 'never upload'\n")
            source_directory.rename(root / "original-src")
            source_directory.symlink_to(external, target_is_directory=True)

            self.assertEqual(
                watch_files.snapshot_attachments(
                    attachments, root=root, max_bytes=1_000
                ),
                [],
            )

    def test_redacts_sensitive_values(self) -> None:
        secrets = [
            "AIzaSyA12345678901234567890123456789012",
            "AKIAIOSFODNN7EXAMPLE",
            "ghp_abcdefghijklmnopqrstuvwxyz1234567890",
            "sk-proj-abcdefghijklmnopqrstuvwxyz123456",
            "xoxb-" + "1234567890-abcdefghijklmnop",
            "eyJabcdefghij.eyJklmnopqrst.uvwxyzABCDEFG",
            "correct-horse-battery-staple",
            "short-token-secret",
            "url-password",
            "bearer-token-value",
        ]
        source = "\n".join(
            [
                f'GEMINI_API_KEY="{secrets[0]}"',
                secrets[1],
                secrets[2],
                secrets[3],
                secrets[4],
                secrets[5],
                f"password: {secrets[6]}",
                f"GITHUB_TOKEN={secrets[7]}",
                f"postgres://user:{secrets[8]}@localhost/db",
                f"Authorization: Bearer {secrets[9]}",
                "-----BEGIN PRIVATE KEY-----",
                "private-material",
                "-----END PRIVATE KEY-----",
            ]
        )

        sanitized, count = watch_files.redact_sensitive_values(source)

        self.assertGreaterEqual(count, len(secrets) + 1)
        for secret in secrets:
            self.assertNotIn(secret, sanitized)
        self.assertNotIn("private-material", sanitized)

    def test_path_redaction_preserves_descriptive_paths_but_removes_keys(self) -> None:
        ordinary = Path("03_cross_file_units/token_service.py")
        secret = Path("src/ghp_abcdefghijklmnopqrstuvwxyz1234567890.py")

        ordinary_sanitized, ordinary_count = watch_files.redact_sensitive_path(ordinary)
        secret_sanitized, secret_count = watch_files.redact_sensitive_path(secret)

        self.assertEqual(ordinary_sanitized, ordinary.as_posix())
        self.assertEqual(ordinary_count, 0)
        self.assertNotIn("ghp_abcdefghijklmnopqrstuvwxyz1234567890", secret_sanitized)
        self.assertGreater(secret_count, 0)

    def test_provider_path_mapping_is_exact_and_rejects_sensitive_paths(self) -> None:
        secret = "ghp_abcdefghijklmnopqrstuvwxyz1234567890"

        def snapshot(relative: str) -> watch_files.SourceSnapshot:
            path = Path("/tmp") / relative
            return watch_files.SourceSnapshot(path, Path(relative), "", "0" * 64, 0)

        mapping = watch_files.provider_path_mapping([snapshot("src/app.py")])
        self.assertEqual(mapping, {"src/app.py": "src/app.py"})

        with self.assertRaisesRegex(
            watch_files.ReviewValidationError,
            "excluded before mapping",
        ) as raised:
            watch_files.provider_path_mapping([snapshot(f"src/{secret}.py")])
        self.assertNotIn(secret, str(raised.exception))

    def test_review_sends_only_sanitized_temporary_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            source = root / "source.env"
            secret = "aB3dE5gH7jK9mN2pQ4sT6vW8yZ1cF0iL"
            source.write_text(f"API_KEY={secret}\nname=safe\n")
            observed_path: Path | None = None
            prompt_secret = "sk-proj-abcdefghijklmnopqrstuvwxyz123456"
            terminal = io.StringIO()

            def fake_run(command: list[str], **_: object) -> object:
                nonlocal observed_path
                self.assertEqual(_["timeout"], 60)
                self.assertEqual(_["output_limit"], watch_files.MAX_PROVIDER_OUTPUT_BYTES)
                attachment_index = command.index("--fragment") + 1
                observed_path = Path(command[attachment_index])
                self.assertNotEqual(observed_path, source)
                self.assertEqual(observed_path.name, "changed-file-0001.txt")
                self.assertTrue(observed_path.is_file())
                uploaded = observed_path.read_text()
                self.assertNotIn(secret, uploaded)
                self.assertIn(watch_files.REDACTED, uploaded)
                schema = json.loads(command[command.index("--schema") + 1])
                file_enum = schema["properties"]["findings"]["items"][
                    "properties"
                ]["file"]["enum"]
                self.assertEqual(file_enum, ["source.env"])
                self.assertFalse(
                    any(
                        secret_value in argument
                        for argument in command
                        for secret_value in (secret, prompt_secret)
                    )
                )
                return type(
                    "Result",
                    (),
                    {
                        "returncode": 0,
                        "stdout": '{"findings": []}',
                        "stderr": "",
                        "output_exceeded": False,
                    },
                )()

            with mock.patch("watch_files.run_bounded_command", side_effect=fake_run):
                watch_files.review_files(
                    [source],
                    root=root,
                    exclude_patterns=[],
                    max_bytes=2_000_000,
                    model="gpt-5.6-luna",
                    prompt=f"tell me what is wrong; token={prompt_secret}",
                    log=False,
                    review_timeout=60,
                    reasoning_effort="high",
                    sink=watch_files.ConsoleSink(mode="json", stream=terminal),
                )

            self.assertIsNotNone(observed_path)
            self.assertFalse(observed_path.exists())
            rendered = terminal.getvalue()
            self.assertNotIn(secret, rendered)
            self.assertNotIn(prompt_secret, rendered)
            redactions = json.loads(rendered)["redactions"]
            self.assertEqual(redactions["total"], 2)
            self.assertEqual(
                {notice["disposition"] for notice in redactions["notices"]},
                {"sent"},
            )

    def test_sensitive_filename_is_excluded_without_retaining_its_value(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            filename_secret = "ghp_" + "a1B2" * 8
            source = root / f"{filename_secret}.env"
            source.write_text("name=safe\n", encoding="utf-8")
            sink = mock.Mock()
            sink.publish.return_value = True
            review_coordinator = mock.Mock()
            review_coordinator.capture_session_generation.return_value = 7
            batch_flushed_at = time.time()

            with mock.patch("watch_files.run_bounded_command") as provider:
                batch = watch_files.review_files(
                    [source],
                    root=root,
                    exclude_patterns=[],
                    max_bytes=2_000_000,
                    model="gpt-5.6-luna",
                    prompt="review",
                    log=False,
                    review_timeout=60,
                    reasoning_effort="high",
                    sink=sink,
                    session_id="synthetic-session",
                    batch_flushed_at=batch_flushed_at,
                    review_coordinator=review_coordinator,
                )

            provider.assert_not_called()
            self.assertIsNotNone(batch)
            assert batch is not None
            self.assertEqual(batch.reviewed_files, ())
            self.assertEqual(batch.redactions.total, 1)
            self.assertEqual(batch.redactions.notices[0].disposition, "excluded")
            self.assertEqual(batch.session_generation, 7)
            self.assertEqual(batch.batch_flushed_at, batch_flushed_at)
            self.assertGreaterEqual(batch.published_at, batch.created_at)
            sink.publish.assert_called_once_with(batch)
            review_coordinator.capture_session_generation.assert_called_once_with()
            review_coordinator.retire_reviewed_flush_hints.assert_called_once()
            retained = json.dumps(batch.to_dict())
            self.assertNotIn(filename_secret, retained)
            self.assertNotIn(filename_secret[:10], retained)

    def test_retained_freshness_digest_uses_only_sanitized_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            source = root / "app.py"
            first_secret = "ghp_" + "a1B2" * 8
            second_secret = "ghp_" + "c3D4" * 8
            source.write_text(f"API_KEY={first_secret}\n", encoding="utf-8")
            max_bytes = 2_000_000
            attachments = watch_files.collect_attachments(
                [source], root=root, exclude_patterns=[], max_bytes=max_bytes
            )

            snapshot = watch_files.snapshot_attachments(
                attachments, root=root, max_bytes=max_bytes
            )[0]

            raw_digest = hashlib.sha256(source.read_bytes()).hexdigest()
            self.assertNotEqual(snapshot.sha256, raw_digest)
            self.assertEqual(snapshot.size, max_bytes)
            source.write_text(f"API_KEY={second_secret}\n", encoding="utf-8")
            reviewed = (
                watch_files.ReviewedFile(
                    path="app.py", sha256=snapshot.sha256, size=snapshot.size
                ),
            )
            batch = watch_files.parse_review_output(
                json.dumps(
                    {
                        "findings": [
                            {
                                "file": "app.py",
                                "line": 1,
                                "severity": "medium",
                                "confidence": 0.99,
                                "title": "Synthetic finding",
                                "explanation": "Synthetic evidence.",
                                "suggested_fix": "Use a synthetic replacement and test it.",
                            }
                        ]
                    }
                ),
                root=root,
                reviewed_files=reviewed,
            )
            self.assertEqual(len(watch_files.fresh_findings(batch).findings), 1)

    def test_failed_review_still_reports_only_safe_redaction_hints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            source = root / "app.py"
            secret = "ghp_" + "a1B2" * 8
            source.write_text(f"OPENAI_API_KEY={secret}\n", encoding="utf-8")
            failure = type(
                "Result",
                (),
                {
                    "returncode": 2,
                    "stdout": "",
                    "stderr": "synthetic provider failure",
                    "output_exceeded": False,
                },
            )()
            errors = io.StringIO()

            with (
                mock.patch("watch_files.run_bounded_command", return_value=failure),
                contextlib.redirect_stderr(errors),
            ):
                result = watch_files.review_files(
                    [source],
                    root=root,
                    exclude_patterns=[],
                    max_bytes=2_000_000,
                    model="gpt-5.6-luna",
                    prompt="review",
                    log=False,
                    review_timeout=60,
                    reasoning_effort="high",
                )

            self.assertIsNone(result)
            diagnostic = errors.getvalue()
            self.assertIn("app.py:1 assignment key OPEN…KEY", diagnostic)
            self.assertNotIn(secret, diagnostic)
            self.assertNotIn(secret[:10], diagnostic)

    def test_review_grounds_a_visible_symbol_from_a_supplied_test(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            source = root / "tests" / "cache.py"
            source.parent.mkdir()
            source.write_text(
                "def test_unexpired_entry():\n    assert True\n",
                encoding="utf-8",
            )
            response = json.dumps(
                {
                    "findings": [
                        {
                            "file": "tests/cache.py",
                            "line": 1,
                            "severity": "medium",
                            "confidence": 0.99,
                            "title": "Missing expiry coverage",
                            "explanation": "The expiry branch is not exercised.",
                            "suggested_fix": (
                                "Extend test_unexpired_entry with an expiry assertion."
                            ),
                        }
                    ]
                }
            )
            result = type(
                "Result",
                (),
                {
                    "returncode": 0,
                    "stdout": response,
                    "stderr": "",
                    "output_exceeded": False,
                },
            )()

            with mock.patch(
                "watch_files.run_bounded_command", return_value=result
            ):
                batch = watch_files.review_files(
                    [source],
                    root=root,
                    exclude_patterns=[],
                    max_bytes=2_000_000,
                    model="gpt-5.6-luna",
                    prompt="review",
                    log=False,
                    review_timeout=60,
                    reasoning_effort="high",
                    sink=mock.Mock(),
                )

            self.assertIsNotNone(batch)
            assert batch is not None
            self.assertEqual(batch.findings[0].file, "tests/cache.py")

    def test_evaluation_event_preserves_provider_streams_and_exit_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            source = root / "source.py"
            source.write_text("value = 1\n")

            result = type("Result", (), {
                "returncode": 2,
                "stdout": '{"findings": []}\n',
                "stderr": "provider rejected request\n",
                "output_exceeded": False,
            })()
            output = io.StringIO()
            with (
                mock.patch(
                    "watch_files.run_bounded_command", return_value=result
                ) as run,
                contextlib.redirect_stdout(output),
            ):
                watch_files.review_files(
                    [source], root=root, exclude_patterns=[], max_bytes=2_000_000,
                    model="test-model", prompt="review", log=False,
                    review_timeout=60, reasoning_effort=None,
                    evaluation_events=True,
                )

            self.assertEqual(
                run.call_args.kwargs["output_limit"],
                watch_files.MAX_PROVIDER_OUTPUT_BYTES,
            )
            event = json.loads(output.getvalue().splitlines()[-1])[
                "quodet_evaluation_event"
            ]
            self.assertEqual(event["status"], "provider-error")
            self.assertEqual(event["returncode"], 2)
            self.assertEqual(event["raw_response"], '{"findings": []}\n')
            self.assertEqual(event["stderr"], "provider rejected request\n")

    def test_failed_model_run_redacts_provider_streams_before_event(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            source = root / "source.py"
            source.write_text("value = 1\n")
            secret = "ghp_abcdefghijklmnopqrstuvwxyz1234567890"
            failed = ModelRunResult(
                status="provider-error",
                returncode=2,
                stdout=f"request={secret}",
                stderr=f"rejected token {secret}",
                latency_ms=1,
                input_tokens=None,
                output_tokens=None,
                cost_usd=None,
                maximum_cost_usd=None,
                resource_usage={},
                effective_config={},
            )
            output = io.StringIO()
            attestation = {
                "runtime": {"name": "llm-ollama", "version": "1.0"},
                "model_registry_entry": "fixture",
            }
            with (
                mock.patch(
                    "evals.agent_changes.live_eval.attest_runtime",
                    return_value=attestation,
                ) as runtime_attestation,
                mock.patch("watch_files.run_model", return_value=failed),
                contextlib.redirect_stdout(output),
            ):
                watch_files.review_files(
                    [source], root=root, exclude_patterns=[], max_bytes=2_000_000,
                    model="test-model", prompt="review", log=False,
                    review_timeout=60, reasoning_effort=None,
                    evaluation_events=True,
                    model_run_config=mock.sentinel.config,
                )

            event = json.loads(output.getvalue().splitlines()[-1])[
                "quodet_evaluation_event"
            ]
            self.assertNotIn(secret, json.dumps(event))
            self.assertIn("[REDACTED]", event["raw_response"])
            self.assertIn("[REDACTED]", event["stderr"])
            self.assertIn("[REDACTED]", event["model_run_result"]["stdout"])
            self.assertEqual(event["runtime_attestation"], attestation)
            runtime_attestation.assert_called_once_with(mock.sentinel.config)

    def test_runtime_attestation_failure_blocks_each_benchmark_model_call(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            source = root / "source.py"
            source.write_text("value = 1\n", encoding="utf-8")
            output = io.StringIO()
            model_config = privacy_config()
            with (
                mock.patch(
                    "evals.agent_changes.live_eval.attest_runtime",
                    side_effect=ValueError("runtime blob changed"),
                ) as runtime_attestation,
                mock.patch("watch_files.run_model") as model_call,
                contextlib.redirect_stdout(output),
            ):
                result = watch_files.review_files(
                    [source], root=root, exclude_patterns=[], max_bytes=2_000_000,
                    model="test-model", prompt="review", log=False,
                    review_timeout=60, reasoning_effort=None,
                    evaluation_events=True,
                    model_run_config=model_config,
                )

            event = json.loads(output.getvalue().splitlines()[-1])[
                "quodet_evaluation_event"
            ]
            self.assertIsNone(result)
            self.assertEqual(event["status"], "provider-error")
            self.assertIs(event["model_attempted"], False)
            self.assertIn("runtime blob changed", event["stderr"])
            runtime_attestation.assert_called_once_with(model_config)
            model_call.assert_not_called()

    def test_json_sink_keeps_stdout_machine_clean(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            source = root / "source.py"
            source.write_text("value = 1\n")
            result = type(
                "Result",
                (),
                {
                    "returncode": 0,
                    "stdout": '{"findings": []}',
                    "stderr": "",
                    "output_exceeded": False,
                },
            )()
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                mock.patch("watch_files.run_bounded_command", return_value=result),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                batch = watch_files.review_files(
                    [source],
                    root=root,
                    exclude_patterns=[],
                    max_bytes=2_000_000,
                    model="test-model",
                    prompt="review",
                    log=False,
                    review_timeout=60,
                    reasoning_effort=None,
                    sink=watch_files.ConsoleSink(mode="json"),
                )

            document = json.loads(stdout.getvalue())
            self.assertEqual(document["schema_version"], "quodet-review-output-v1")
            self.assertEqual(document["findings"], [])
            self.assertGreaterEqual(
                document["timing"]["published_at"], document["created_at"]
            )
            self.assertIsNotNone(batch)
            assert batch is not None
            self.assertEqual(document["timing"]["published_at"], batch.published_at)
            self.assertIn("Reviewing 1 changed file", stderr.getvalue())
            self.assertNotIn("Reviewing", stdout.getvalue())

    def test_review_handles_malformed_nonzero_and_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            source = root / "app.py"
            source.write_text("print('hello')\n")
            common = dict(
                paths=[source],
                root=root,
                exclude_patterns=[],
                max_bytes=2_000_000,
                model="gpt-5.6-luna",
                prompt="review",
                log=False,
                review_timeout=60,
                reasoning_effort="high",
            )

            malformed = type(
                "Result",
                (),
                {
                    "returncode": 0,
                    "stdout": "not json",
                    "stderr": "",
                    "output_exceeded": False,
                },
            )()
            rejection = io.StringIO()
            with (
                mock.patch("watch_files.run_bounded_command", return_value=malformed),
                contextlib.redirect_stderr(rejection),
            ):
                self.assertIsNone(watch_files.review_files(**common))
            self.assertIn("Review discarded", rejection.getvalue())
            self.assertIn(
                "no console or agent feedback was published", rejection.getvalue()
            )

            failed = type(
                "Result",
                (),
                {
                    "returncode": 2,
                    "stdout": "",
                    "stderr": "provider error",
                    "output_exceeded": False,
                },
            )()
            with mock.patch("watch_files.run_bounded_command", return_value=failed):
                self.assertIsNone(watch_files.review_files(**common))

            with mock.patch(
                "watch_files.run_bounded_command",
                side_effect=subprocess.TimeoutExpired("llm", 60),
            ):
                self.assertIsNone(watch_files.review_files(**common))

    def test_review_returns_typed_batch_and_drops_finding_if_file_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            source = root / "app.py"
            source.write_text("value = 1\n")
            response = json.dumps(
                {
                    "findings": [
                        {
                            "file": "app.py",
                            "line": 1,
                            "severity": "medium",
                            "confidence": 0.99,
                            "title": "Wrong value",
                            "explanation": "This value violates the contract.",
                            "suggested_fix": "Use the required value.",
                        }
                    ]
                }
            )

            def change_during_review(*_: object, **__: object) -> object:
                source.write_text("value = 2\n")
                return type(
                    "Result",
                    (),
                    {
                        "returncode": 0,
                        "stdout": response,
                        "stderr": "",
                        "output_exceeded": False,
                    },
                )()

            sink = mock.Mock()
            with mock.patch(
                "watch_files.run_bounded_command", side_effect=change_during_review
            ):
                batch = watch_files.review_files(
                    [source],
                    root=root,
                    exclude_patterns=[],
                    max_bytes=2_000_000,
                    model="gpt-5.6-luna",
                    prompt="review",
                    log=False,
                    review_timeout=60,
                    reasoning_effort="high",
                    sink=sink,
                )

            self.assertIsInstance(batch, watch_files.ReviewBatch)
            self.assertEqual(batch.findings, ())
            sink.publish.assert_called_once_with(batch)

    def test_provider_output_is_bounded_before_memory_buffering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = watch_files.run_bounded_command(
                [sys.executable, "-c", "import sys; sys.stdout.write('x' * 100000)"],
                cwd=Path(temporary_directory),
                timeout=5,
                output_limit=1_024,
            )

        self.assertTrue(result.output_exceeded)
        self.assertLessEqual(len(result.stdout.encode("utf-8")), 1_025)

    def test_provider_timeout_preserves_bounded_partial_streams(self) -> None:
        command = [
            sys.executable,
            "-c",
            (
                "import sys, time; "
                "print('partial-out', flush=True); "
                "print('partial-err', file=sys.stderr, flush=True); "
                "time.sleep(10)"
            ),
        ]
        with tempfile.TemporaryDirectory() as temporary_directory, self.assertRaises(
            subprocess.TimeoutExpired
        ) as raised:
            watch_files.run_bounded_command(
                command,
                cwd=Path(temporary_directory),
                timeout=0.5,
                output_limit=1_024,
            )

        self.assertIn("partial-out", raised.exception.stdout)
        self.assertIn("partial-err", raised.exception.stderr)
        self.assertLessEqual(len(raised.exception.stdout.encode("utf-8")), 1_025)
        self.assertLessEqual(len(raised.exception.stderr.encode("utf-8")), 1_025)

    def test_change_handler_uses_destination_for_move(self) -> None:
        changes: queue.Queue[Path] = queue.Queue()
        observed_at: dict[Path, float] = {}
        root = Path("/tmp").resolve()
        handler = watch_files.ChangeHandler(
            changes,
            root=root,
            exclude_patterns=[],
            observed_at=observed_at,
        )
        event = type(
            "Event",
            (),
            {
                "is_directory": False,
                "src_path": "/tmp/old.py",
                "dest_path": "/tmp/new.py",
            },
        )()

        handler.on_moved(event)

        self.assertEqual(changes.get_nowait(), root / "new.py")
        self.assertIn(root / "new.py", observed_at)

    def test_change_handler_does_not_queue_dependency_events(self) -> None:
        changes: queue.Queue[Path] = queue.Queue()
        root = Path("/tmp/project").resolve()
        handler = watch_files.ChangeHandler(changes, root=root, exclude_patterns=[])
        event = type(
            "Event",
            (),
            {
                "is_directory": False,
                "src_path": "/tmp/project/node_modules/package/index.js",
            },
        )()

        handler.on_modified(event)

        self.assertTrue(changes.empty())

    def test_relative_to_root_rejects_outside_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            self.assertIsNone(watch_files.relative_to_root(Path("/tmp/outside"), root))


if __name__ == "__main__":
    unittest.main()
