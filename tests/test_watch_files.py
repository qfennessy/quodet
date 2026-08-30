from __future__ import annotations

import argparse
import contextlib
import io
import json
import queue
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import watch_files


class WatchFilesTests(unittest.TestCase):
    def test_default_debounce_groups_related_agent_changes(self) -> None:
        args = watch_files.parse_args(["."])
        self.assertEqual(args.debounce, 3.0)
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

    def test_default_prompt_requests_only_confident_negative_json(self) -> None:
        prompt = watch_files.DEFAULT_PROMPT.lower()
        self.assertIn("only negative findings", prompt)
        self.assertIn("at least 0.95 confident", prompt)
        self.assertIn("concrete execution path", prompt)
        self.assertIn("each supplied file as a separate file", prompt)
        self.assertIn("mutually consistent", prompt)
        self.assertIn("calibrate severity only from demonstrated impact", prompt)
        self.assertIn("empty findings array", prompt)
        self.assertIn("only with json", prompt)

        confidence = watch_files.REVIEW_SCHEMA["properties"]["findings"]["items"][
            "properties"
        ]["confidence"]
        self.assertEqual(confidence["minimum"], 0.95)
        self.assertNotIn("additionalProperties", watch_files.REVIEW_SCHEMA_JSON)

    def test_default_prompt_requires_actionable_untrusted_recommendations(self) -> None:
        prompt = watch_files.DEFAULT_PROMPT.lower()
        for requirement in (
            "grounded only in the supplied code",
            "relevant function, class, branch, state transition",
            "smallest focused behavior change",
            "why it fixes the cited execution path",
            "narrow regression test or validation step",
            "identify the exact missing evidence",
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

    def test_review_sends_only_sanitized_temporary_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            filename_secret = "ghp_abcdefghijklmnopqrstuvwxyz1234567890"
            source = root / f"{filename_secret}.env"
            secret = "aB3dE5gH7jK9mN2pQ4sT6vW8yZ1cF0iL"
            source.write_text(f"API_KEY={secret}\nname=safe\n")
            observed_path: Path | None = None
            prompt_secret = "sk-proj-abcdefghijklmnopqrstuvwxyz123456"

            def fake_run(command: list[str], **_: object) -> object:
                nonlocal observed_path
                self.assertEqual(_["timeout"], 60)
                attachment_index = command.index("--fragment") + 1
                observed_path = Path(command[attachment_index])
                self.assertNotEqual(observed_path, source)
                self.assertEqual(observed_path.name, "changed-file-0001.txt")
                self.assertTrue(observed_path.is_file())
                uploaded = observed_path.read_text()
                self.assertNotIn(secret, uploaded)
                self.assertNotIn(filename_secret, uploaded)
                self.assertIn(watch_files.REDACTED, uploaded)
                self.assertFalse(
                    any(
                        secret_value in argument
                        for argument in command
                        for secret_value in (filename_secret, prompt_secret)
                    )
                )
                return type("Result", (), {"returncode": 0})()

            with mock.patch("watch_files.subprocess.run", side_effect=fake_run):
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
                )

            self.assertIsNotNone(observed_path)
            self.assertFalse(observed_path.exists())

    def test_evaluation_event_preserves_provider_streams_and_exit_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            source = root / "source.py"
            source.write_text("value = 1\n")

            result = type("Result", (), {
                "returncode": 2,
                "stdout": '{"findings": []}\n',
                "stderr": "provider rejected request\n",
            })()
            output = io.StringIO()
            with (
                mock.patch("watch_files.subprocess.run", return_value=result) as run,
                contextlib.redirect_stdout(output),
            ):
                watch_files.review_files(
                    [source], root=root, exclude_patterns=[], max_bytes=2_000_000,
                    model="test-model", prompt="review", log=False,
                    review_timeout=60, reasoning_effort=None,
                    evaluation_events=True,
                )

            self.assertTrue(run.call_args.kwargs["capture_output"])
            self.assertTrue(run.call_args.kwargs["text"])
            event = json.loads(output.getvalue().splitlines()[-1])[
                "quodet_evaluation_event"
            ]
            self.assertEqual(event["status"], "provider-error")
            self.assertEqual(event["returncode"], 2)
            self.assertEqual(event["raw_response"], '{"findings": []}\n')
            self.assertEqual(event["stderr"], "provider rejected request\n")

    def test_change_handler_uses_destination_for_move(self) -> None:
        changes: queue.Queue[Path] = queue.Queue()
        root = Path("/tmp").resolve()
        handler = watch_files.ChangeHandler(changes, root=root, exclude_patterns=[])
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
