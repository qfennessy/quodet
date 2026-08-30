from __future__ import annotations

import hashlib
import io
import json
import os
import queue
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import agent_integration
import codex_feedback_hook
import watch_files
from feedback import (
    ReviewedFile,
    SpoolSink,
    parse_review_output,
    publish_flush_hint,
)


FIXTURES = Path(__file__).parent / "fixtures" / "agent_contracts"


def finding_output(count: int = 1) -> str:
    return json.dumps(
        {
            "findings": [
                {
                    "file": "src/app.py",
                    "line": index + 1,
                    "severity": "medium",
                    "confidence": 0.99,
                    "title": f"Finding {index + 1}",
                    "explanation": "A concrete execution path fails.",
                    "suggested_fix": "Change the branch and add a regression test.",
                }
                for index in range(count)
            ]
        }
    )


class AgentIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.root = self.base / "project"
        self.source = self.root / "src" / "app.py"
        self.source.parent.mkdir(parents=True)
        self.source.write_text("value = 1\n", encoding="utf-8")
        self.spool = self.base / "runtime" / "feedback"

    def _executable(self, name: str) -> str:
        path = self.base / name
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        path.chmod(0o700)
        return os.fspath(path)

    def _route(self, agent: str = "codex") -> tuple[agent_integration.RouteConfig, Path]:
        route_path, _ = agent_integration.initialize(
            agent,
            root=self.root,
            spool_dir=self.spool,
            session_id=f"{agent}-route",
            hook_command=self._executable(f"{agent}-hook"),
            agent_command=self._executable("agent-command"),
        )
        return agent_integration.load_route(route_path), route_path

    def _batch(self, route: agent_integration.RouteConfig, count: int = 1):
        raw = self.source.read_bytes()
        reviewed = (
            ReviewedFile(
                "src/app.py", hashlib.sha256(raw).hexdigest(), len(raw)
            ),
        )
        return parse_review_output(
            finding_output(count),
            root=self.root,
            reviewed_files=reviewed,
            session_id=route.session_id,
            debounce_ms=3_000,
            provider_ms=125,
        )

    def _publish(self, route: agent_integration.RouteConfig, count: int = 1) -> SpoolSink:
        batch = self._batch(route, count)
        sink = SpoolSink(self.spool, root=self.root, session_id=route.session_id)
        self.addCleanup(sink.close)
        self.assertTrue(sink.publish(batch))
        return sink

    def _fixture(self, agent: str, name: str) -> dict[str, object]:
        directory = "codex" if agent == "codex" else "claude-code"
        value = json.loads(
            (FIXTURES / directory / "docs-2026-08-30" / name).read_text()
        )
        encoded = json.dumps(value).replace("${ROOT}", os.fspath(self.root))
        return json.loads(encoded)

    def _invoke(
        self,
        route: agent_integration.RouteConfig,
        event: str,
        input_value: dict[str, object],
        extra_args: tuple[str, ...] = (),
    ) -> dict[str, object] | None:
        output = io.StringIO()
        with mock.patch("sys.stdin", io.StringIO(json.dumps(input_value))), mock.patch(
            "sys.stdout", output
        ):
            result = codex_feedback_hook.main(
                [
                    "--event", event,
                    "--spool-dir", route.spool_dir,
                    "--session-id", route.session_id,
                    "--root", route.root,
                    *extra_args,
                ]
            )
        self.assertEqual(result, 0)
        return json.loads(output.getvalue()) if output.getvalue() else None

    def test_init_creates_private_route_and_agent_settings_without_overwrite(self) -> None:
        for agent, settings_name in (
            ("codex", ".codex/hooks.json"),
            ("claude", ".claude/settings.json"),
        ):
            with self.subTest(agent=agent):
                self.spool = self.base / f"runtime-{agent}" / "feedback"
                route, route_path = self._route(agent)
                settings = self.root / settings_name
                self.assertTrue(settings.is_file())
                self.assertEqual(route.root, os.fspath(self.root))
                self.assertEqual(route.spool_dir, os.fspath(self.spool))
                self.assertEqual(route_path.stat().st_mode & 0o777, 0o600)
                self.assertEqual(self.spool.stat().st_mode & 0o777, 0o700)
                # An exact rerun is validation-only and remains successful.
                agent_integration.initialize(
                    agent,
                    root=self.root,
                    spool_dir=self.spool,
                    session_id=route.session_id,
                    hook_command=self._executable(f"{agent}-hook"),
                    agent_command=self._executable("agent-command"),
                )

    def test_init_refuses_existing_settings_and_route_mismatch(self) -> None:
        settings = self.root / ".codex" / "hooks.json"
        settings.parent.mkdir()
        settings.write_text('{"hooks":{"existing":[]}}\n', encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self._route("codex")
        self.assertEqual(json.loads(settings.read_text()), {"hooks": {"existing": []}})

        settings.unlink()
        route, route_path = self._route("codex")
        with self.assertRaisesRegex(ValueError, "root does not match"):
            agent_integration.validate_watcher_route(
                route,
                root=self.base / "other",
                spool_dir=None,
                session_id=None,
            )
        with self.assertRaisesRegex(ValueError, "spool-dir"):
            agent_integration.validate_watcher_route(
                route,
                root=self.root,
                spool_dir=self.base / "wrong-spool",
                session_id=route.session_id,
            )
        with self.assertRaisesRegex(ValueError, "session-id"):
            agent_integration.validate_watcher_route(
                route,
                root=self.root,
                spool_dir=self.spool,
                session_id="wrong-session",
            )
        self.assertEqual(agent_integration.load_route(route_path), route)
        with self.assertRaisesRegex(ValueError, "not an ancestor"):
            agent_integration.initialize(
                "codex",
                root=self.root,
                spool_dir=self.base,
                session_id="unsafe-ancestor",
                hook_command=self._executable("ancestor-hook"),
                agent_command=self._executable("ancestor-agent"),
            )

    def test_init_persists_configurable_bounded_stop_grace(self) -> None:
        route_path, settings_path = agent_integration.initialize(
            "codex",
            root=self.root,
            spool_dir=self.spool,
            session_id="codex-route",
            hook_command=self._executable("codex-hook"),
            agent_command=self._executable("agent-command"),
            stop_grace_seconds=0.75,
        )
        route = agent_integration.load_route(route_path)
        settings = json.loads(settings_path.read_text())
        stop_command = settings["hooks"]["Stop"][0]["hooks"][0]["command"]

        self.assertEqual(route.stop_grace_seconds, 0.75)
        self.assertIn("--stop-grace 0.75", stop_command)
        for invalid in (-1, 11, float("nan")):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError, "stop_grace_seconds"
            ):
                agent_integration.initialize(
                    "claude",
                    root=self.root,
                    spool_dir=self.base / f"spool-{invalid}",
                    session_id="claude-route",
                    hook_command=self._executable(f"claude-hook-{invalid}"),
                    agent_command=self._executable(f"agent-command-{invalid}"),
                    stop_grace_seconds=invalid,
                )

    def test_load_and_reinitialize_legacy_v1_route_without_migration(self) -> None:
        hook_command = self._executable("legacy-codex-hook")
        agent_command = self._executable("legacy-agent-command")
        route_path, settings_path = agent_integration.initialize(
            "codex",
            root=self.root,
            spool_dir=self.spool,
            session_id="legacy-route",
            hook_command=hook_command,
            agent_command=agent_command,
        )
        route = agent_integration.load_route(route_path)
        legacy_route = json.loads(route_path.read_text())
        legacy_route.pop("stop_grace_seconds")
        route_path.write_text(json.dumps(legacy_route))
        route_path.chmod(0o600)
        legacy_settings = agent_integration.hook_configuration(
            route,
            hook_command=hook_command,
            agent_command=agent_command,
            include_stop_grace=False,
        )
        settings_path.write_text(json.dumps(legacy_settings))
        settings_path.chmod(0o600)

        loaded = agent_integration.load_route(route_path)
        rerun = agent_integration.initialize(
            "codex",
            root=self.root,
            spool_dir=self.spool,
            session_id="legacy-route",
            hook_command=hook_command,
            agent_command=agent_command,
        )

        self.assertEqual(loaded.stop_grace_seconds, 2.0)
        self.assertEqual(rerun, (route_path, settings_path))

    def test_init_cannot_overwrite_concurrently_created_settings(self) -> None:
        settings = self.root / ".codex" / "hooks.json"
        original_create = agent_integration._create_private_json

        def create_with_competitor(path: Path, value: object) -> None:
            if path == settings:
                path.write_text('{"hooks":{"competitor":[]}}\n', encoding="utf-8")
                path.chmod(0o600)
            original_create(path, value)

        with mock.patch.object(
            agent_integration,
            "_create_private_json",
            side_effect=create_with_competitor,
        ), self.assertRaisesRegex(FileExistsError, "concurrently created settings"):
            self._route("codex")
        self.assertEqual(
            json.loads(settings.read_text()), {"hooks": {"competitor": []}}
        )

    def test_versioned_official_contract_replays_for_codex_and_claude(self) -> None:
        for agent in ("codex", "claude"):
            with self.subTest(agent=agent):
                self.spool = self.base / f"contract-{agent}" / "feedback"
                route, _ = self._route(agent)
                sink = self._publish(route, count=12)
                post = self._invoke(
                    route, "PostToolUse", self._fixture(agent, "post_tool_use.input.json")
                )
                self.assertEqual(
                    post["hookSpecificOutput"]["hookEventName"], "PostToolUse"  # type: ignore[index]
                )
                message = post["hookSpecificOutput"]["additionalContext"]  # type: ignore[index]
                self.assertIn("watcher debounce 3000.0 ms", message)
                self.assertIn("provider 125.0 ms", message)
                self.assertIn("hook delivery wait", message)
                self.assertEqual(len(list((self.spool / "pending").glob("*.json"))), 1)
                self.assertEqual(
                    agent_integration.route_status(route)["feedback"][  # type: ignore[index]
                        "flush-hints"
                    ],
                    0,
                )

                stop = self._invoke(
                    route, "Stop", self._fixture(agent, "stop.input.json")
                )
                self.assertEqual(stop["decision"], "block")  # type: ignore[index]
                self.assertEqual(len(list((self.spool / "pending").glob("*.json"))), 0)
                latency = agent_integration.route_status(route)["latency_ms"]
                self.assertEqual(latency["samples"], 2)  # type: ignore[index]
                for segment in (
                    "detection_to_flush_ms",
                    "flush_to_provider_ms",
                    "provider_ms",
                    "publication_ms",
                    "hook_wait_ms",
                    "hook_execution_ms",
                    "total_edit_to_feedback_ms",
                ):
                    self.assertIsNotNone(
                        latency[segment]["p95"],  # type: ignore[index]
                        segment,
                    )
                sink.close()
                agent_integration.cleanup_route(
                    route, agent_session_id="live-agent-session"
                )
                self.assertEqual(
                    len(list((self.spool / "metrics").glob("*.json"))), 2
                )

    def test_post_tool_hint_flushes_direct_edit_without_full_debounce(self) -> None:
        route, _ = self._route()
        self.assertTrue(
            codex_feedback_hook.verify_session_lease(
                self.spool,
                root=self.root,
                configured_session_id=route.session_id,
                codex_session_id="live-agent-session",
            )
        )
        sink = SpoolSink(self.spool, root=self.root, session_id=route.session_id)
        self.addCleanup(sink.close)
        publish_flush_hint(
            self.spool,
            root=self.root,
            session_id=route.session_id,
            agent_session_id="live-agent-session",
            reviewed_files=self._batch(route).reviewed_files,
        )
        changes: queue.Queue[Path] = queue.Queue()
        changes.put(self.source)

        started = time.monotonic()
        triggered = watch_files.next_triggered_batch(
            changes, 1.0, hint_source=sink
        )

        self.assertLess(time.monotonic() - started, 0.2)
        self.assertEqual(triggered.paths, {self.source})
        self.assertIsNotNone(triggered.flush_hint)
        self.assertEqual(
            triggered.flush_hint.agent_session_id,  # type: ignore[union-attr]
            "live-agent-session",
        )

    def test_duplicate_hint_event_preserves_original_observation_timing(self) -> None:
        route, _ = self._route()
        codex_feedback_hook.verify_session_lease(
            self.spool,
            root=self.root,
            configured_session_id=route.session_id,
            codex_session_id="live-agent-session",
        )
        sink = SpoolSink(self.spool, root=self.root, session_id=route.session_id)
        self.addCleanup(sink.close)
        publish_flush_hint(
            self.spool,
            root=self.root,
            session_id=route.session_id,
            agent_session_id="live-agent-session",
            reviewed_files=self._batch(route).reviewed_files,
        )
        changes: queue.Queue[Path] = queue.Queue()
        changes.put(self.source)
        changes.put(self.source)
        suppression = watch_files.MaterializedPathSuppression(
            self.root, ttl_seconds=1.0
        )

        triggered = watch_files.next_triggered_batch(
            changes, 1.0, hint_source=sink, suppression=suppression
        )

        self.assertEqual(triggered.paths, {self.source})
        self.assertEqual(triggered.suppressed_paths, {self.source})
        original_observation = 100.0
        observed_at = {self.source: original_observation}
        self.assertEqual(
            watch_files.consume_first_observed_at(
                triggered, observed_at, fallback=200.0
            ),
            original_observation,
        )
        self.assertEqual(observed_at, {})

    def test_suppression_only_timestamp_does_not_affect_next_batch(self) -> None:
        delayed_duplicate = self.root / "src" / "old.py"
        current = self.root / "src" / "current.py"
        unrelated = self.root / "src" / "unrelated.py"
        observed_at = {
            delayed_duplicate: 100.0,
            current: 150.0,
            unrelated: 75.0,
        }
        triggered = watch_files.TriggeredBatch(
            paths={current},
            flush_hint=None,
            suppressed_paths={delayed_duplicate},
        )

        first_observed_at = watch_files.consume_first_observed_at(
            triggered, observed_at, fallback=200.0
        )

        self.assertEqual(first_observed_at, 150.0)
        self.assertNotIn(delayed_duplicate, observed_at)
        self.assertNotIn(current, observed_at)
        self.assertEqual(observed_at, {unrelated: 75.0})

    def test_provider_review_refreshes_authenticated_in_flight_marker(self) -> None:
        route, _ = self._route()
        codex_feedback_hook.verify_session_lease(
            self.spool,
            root=self.root,
            configured_session_id=route.session_id,
            codex_session_id="live-agent-session",
        )
        sink = SpoolSink(self.spool, root=self.root, session_id=route.session_id)
        self.addCleanup(sink.close)
        observed_markers: list[dict[str, object]] = []

        def capture_marker(*args: object, **kwargs: object) -> None:
            marker_paths = list((self.spool / "in-flight").glob("*.json"))
            self.assertEqual(len(marker_paths), 1)
            observed_markers.append(json.loads(marker_paths[0].read_text()))

        with mock.patch(
            "watch_files._execute_review_command", side_effect=capture_marker
        ):
            watch_files.review_files(
                [self.source],
                root=self.root,
                exclude_patterns=[],
                max_bytes=1_000_000,
                model="test-model",
                prompt="review",
                log=False,
                review_timeout=1.0,
                reasoning_effort=None,
                review_coordinator=sink,
                agent_session_id="live-agent-session",
            )

        self.assertEqual(
            observed_markers[0]["agent_session_id"], "live-agent-session"
        )
        self.assertEqual(list((self.spool / "in-flight").glob("*.json")), [])

    def test_multi_file_hint_keeps_one_logical_edit_in_one_batch(self) -> None:
        route, _ = self._route()
        codex_feedback_hook.verify_session_lease(
            self.spool,
            root=self.root,
            configured_session_id=route.session_id,
            codex_session_id="live-agent-session",
        )
        second = self.root / "src" / "second.py"
        second.write_text("value = 2\n")
        first_metadata = self._batch(route).reviewed_files[0]
        second_raw = second.read_bytes()
        second_metadata = ReviewedFile(
            "src/second.py", hashlib.sha256(second_raw).hexdigest(), len(second_raw)
        )
        sink = SpoolSink(self.spool, root=self.root, session_id=route.session_id)
        self.addCleanup(sink.close)
        publish_flush_hint(
            self.spool,
            root=self.root,
            session_id=route.session_id,
            agent_session_id="live-agent-session",
            reviewed_files=(first_metadata, second_metadata),
        )
        changes: queue.Queue[Path] = queue.Queue()
        changes.put(self.source)
        suppression = watch_files.MaterializedPathSuppression(
            self.root, ttl_seconds=0.02
        )

        triggered = watch_files.next_triggered_batch(
            changes, 1.0, hint_source=sink, suppression=suppression
        )

        self.assertEqual(triggered.paths, {self.source, second})
        marker = sink.begin_review(
            agent_session_id="live-agent-session",
            review_timeout=1,
            flush_hint=triggered.flush_hint,
        )
        time.sleep(0.03)
        sink.finish_review(marker)
        suppression.record(triggered.flush_hint)  # type: ignore[arg-type]
        third = self.root / "src" / "third.py"
        third.write_text("value = 3\n")
        changes.put(second)
        changes.put(third)

        follow_up = watch_files.next_triggered_batch(
            changes, 0.01, hint_source=sink, suppression=suppression
        )

        self.assertEqual(follow_up.paths, {third})
        self.assertEqual(follow_up.suppressed_paths, {second})

    def test_direct_edit_hook_records_only_path_digest_metadata(self) -> None:
        reviewed = codex_feedback_hook._hint_reviewed_files(
            {
                "tool_name": "apply_patch",
                "tool_input": {
                    "command": "*** Begin Patch\n*** Update File: src/app.py\n"
                },
            },
            root=self.root,
        )

        self.assertEqual(len(reviewed), 1)
        self.assertEqual(reviewed[0].path, "src/app.py")
        self.assertEqual(reviewed[0].size, self.source.stat().st_size)
        self.assertEqual(len(reviewed[0].sha256), 64)

    def test_wrong_session_flush_hint_is_rejected(self) -> None:
        route, _ = self._route()
        self.assertTrue(
            codex_feedback_hook.verify_session_lease(
                self.spool,
                root=self.root,
                configured_session_id=route.session_id,
                codex_session_id="live-agent-session",
            )
        )
        with self.assertRaisesRegex(ValueError, "active watcher and agent session"):
            publish_flush_hint(
                self.spool,
                root=self.root,
                session_id=route.session_id,
                agent_session_id="other-agent-session",
                reviewed_files=self._batch(route).reviewed_files,
            )

    def test_malformed_hint_metadata_is_pruned_before_valid_hint(self) -> None:
        route, _ = self._route()
        codex_feedback_hook.verify_session_lease(
            self.spool,
            root=self.root,
            configured_session_id=route.session_id,
            codex_session_id="live-agent-session",
        )
        sink = SpoolSink(self.spool, root=self.root, session_id=route.session_id)
        self.addCleanup(sink.close)
        hints = self.spool / "flush-hints"
        now = time.time()
        malformed_paths: list[Path] = []
        for index, reviewed_files in enumerate((None, 7)):
            path = hints / f"000-malformed-{index}.json"
            path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "root": os.fspath(self.root),
                        "session_id": route.session_id,
                        "agent_session_id": "live-agent-session",
                        "created_at": now,
                        "expires_at": now + 60,
                        "reviewed_files": reviewed_files,
                    }
                )
            )
            os.utime(path, (now - 10, now - 10))
            malformed_paths.append(path)
        valid_path = publish_flush_hint(
            self.spool,
            root=self.root,
            session_id=route.session_id,
            agent_session_id="live-agent-session",
            reviewed_files=self._batch(route).reviewed_files,
        )

        hint = sink.consume_flush_hint()

        self.assertIsNotNone(hint)
        self.assertEqual(hint.path, valid_path)  # type: ignore[union-attr]
        self.assertTrue(all(not path.exists() for path in malformed_paths))

    def test_completed_review_retires_late_digest_matched_hint(self) -> None:
        route, _ = self._route()
        codex_feedback_hook.verify_session_lease(
            self.spool,
            root=self.root,
            configured_session_id=route.session_id,
            codex_session_id="live-agent-session",
        )
        sink = SpoolSink(self.spool, root=self.root, session_id=route.session_id)
        self.addCleanup(sink.close)
        batch = self._batch(route)
        publish_flush_hint(
            self.spool,
            root=self.root,
            session_id=route.session_id,
            agent_session_id="live-agent-session",
            reviewed_files=batch.reviewed_files,
        )

        sink.publish(batch)

        self.assertEqual(list((self.spool / "flush-hints").glob("*.json")), [])
        delivered = self._invoke(
            route,
            "Stop",
            self._fixture("codex", "stop.input.json"),
            ("--stop-grace", "0.2"),
        )
        started = time.monotonic()
        idle = self._invoke(
            route,
            "Stop",
            self._fixture("codex", "stop.input.json"),
            ("--stop-grace", "0.2"),
        )
        self.assertEqual(delivered["decision"], "block")  # type: ignore[index]
        self.assertEqual(idle, {})
        self.assertLess(time.monotonic() - started, 0.1)

    def test_digest_scoped_hint_does_not_flush_unrelated_batch(self) -> None:
        route, _ = self._route()
        codex_feedback_hook.verify_session_lease(
            self.spool,
            root=self.root,
            configured_session_id=route.session_id,
            codex_session_id="live-agent-session",
        )
        sink = SpoolSink(self.spool, root=self.root, session_id=route.session_id)
        self.addCleanup(sink.close)
        publish_flush_hint(
            self.spool,
            root=self.root,
            session_id=route.session_id,
            agent_session_id="live-agent-session",
            reviewed_files=self._batch(route).reviewed_files,
        )
        unrelated = self.root / "src" / "other.py"
        unrelated.write_text("value = 2\n")
        changes: queue.Queue[Path] = queue.Queue()
        changes.put(unrelated)

        triggered = watch_files.next_triggered_batch(
            changes, 0.01, hint_source=sink
        )

        self.assertIsNone(triggered.flush_hint)
        self.assertEqual(triggered.paths, {unrelated})
        self.assertEqual(
            len(list((self.spool / "flush-hints").glob("*.json"))), 1
        )

    def test_stale_same_path_hint_does_not_flush_new_edit(self) -> None:
        route, _ = self._route()
        codex_feedback_hook.verify_session_lease(
            self.spool,
            root=self.root,
            configured_session_id=route.session_id,
            codex_session_id="live-agent-session",
        )
        sink = SpoolSink(self.spool, root=self.root, session_id=route.session_id)
        self.addCleanup(sink.close)
        publish_flush_hint(
            self.spool,
            root=self.root,
            session_id=route.session_id,
            agent_session_id="live-agent-session",
            reviewed_files=self._batch(route).reviewed_files,
        )
        self.source.write_text("value = 2\n")
        changes: queue.Queue[Path] = queue.Queue()
        changes.put(self.source)

        started = time.monotonic()
        triggered = watch_files.next_triggered_batch(
            changes, 0.02, hint_source=sink
        )

        self.assertIsNone(triggered.flush_hint)
        self.assertGreaterEqual(time.monotonic() - started, 0.015)
        self.assertEqual(list((self.spool / "flush-hints").glob("*.json")), [])

    def test_stop_delivers_review_that_completes_during_grace(self) -> None:
        route, _ = self._route()
        self.assertTrue(
            codex_feedback_hook.verify_session_lease(
                self.spool,
                root=self.root,
                configured_session_id=route.session_id,
                codex_session_id="live-agent-session",
            )
        )
        sink = SpoolSink(self.spool, root=self.root, session_id=route.session_id)
        self.addCleanup(sink.close)
        marker = sink.begin_review(
            agent_session_id="live-agent-session", review_timeout=1
        )

        def complete() -> None:
            time.sleep(0.05)
            sink.publish(self._batch(route))
            sink.finish_review(marker)

        worker = threading.Thread(target=complete)
        worker.start()
        started = time.monotonic()
        response = self._invoke(
            route,
            "Stop",
            self._fixture("codex", "stop.input.json"),
            ("--stop-grace", "0.5"),
        )
        worker.join(timeout=1)

        self.assertEqual(response["decision"], "block")  # type: ignore[index]
        self.assertGreaterEqual(time.monotonic() - started, 0.04)
        self.assertFalse(worker.is_alive())

    def test_stop_timeout_keeps_late_review_for_next_boundary(self) -> None:
        route, _ = self._route()
        codex_feedback_hook.verify_session_lease(
            self.spool,
            root=self.root,
            configured_session_id=route.session_id,
            codex_session_id="live-agent-session",
        )
        sink = SpoolSink(self.spool, root=self.root, session_id=route.session_id)
        self.addCleanup(sink.close)
        marker = sink.begin_review(
            agent_session_id="live-agent-session", review_timeout=1
        )

        def complete_late() -> None:
            time.sleep(0.1)
            sink.publish(self._batch(route))
            sink.finish_review(marker)

        worker = threading.Thread(target=complete_late)
        worker.start()
        first = self._invoke(
            route,
            "Stop",
            self._fixture("codex", "stop.input.json"),
            ("--stop-grace", "0.02"),
        )
        worker.join(timeout=1)
        second = self._invoke(
            route,
            "Stop",
            self._fixture("codex", "stop.input.json"),
            ("--stop-grace", "0.02"),
        )

        self.assertEqual(first, {})
        self.assertEqual(second["decision"], "block")  # type: ignore[index]

    def test_stop_returns_when_provider_fails_and_marker_clears(self) -> None:
        route, _ = self._route()
        codex_feedback_hook.verify_session_lease(
            self.spool,
            root=self.root,
            configured_session_id=route.session_id,
            codex_session_id="live-agent-session",
        )
        sink = SpoolSink(self.spool, root=self.root, session_id=route.session_id)
        self.addCleanup(sink.close)
        marker = sink.begin_review(
            agent_session_id="live-agent-session", review_timeout=1
        )

        worker = threading.Thread(
            target=lambda: (time.sleep(0.05), sink.finish_review(marker))
        )
        worker.start()
        started = time.monotonic()
        response = self._invoke(
            route,
            "Stop",
            self._fixture("codex", "stop.input.json"),
            ("--stop-grace", "0.5"),
        )
        elapsed = time.monotonic() - started
        worker.join(timeout=1)

        self.assertEqual(response, {})
        self.assertLess(elapsed, 0.3)

    def test_stop_ignores_stale_marker_and_has_no_idle_grace_latency(self) -> None:
        route, _ = self._route()
        codex_feedback_hook.verify_session_lease(
            self.spool,
            root=self.root,
            configured_session_id=route.session_id,
            codex_session_id="live-agent-session",
        )
        sink = SpoolSink(self.spool, root=self.root, session_id=route.session_id)
        self.addCleanup(sink.close)
        marker = sink.begin_review(
            agent_session_id="live-agent-session", review_timeout=1
        )
        self.assertIsNotNone(marker)
        value = json.loads(marker.read_text())  # type: ignore[union-attr]
        value["started_at"] = time.time() - 20
        value["expires_at"] = time.time() - 10
        marker.write_text(json.dumps(value))  # type: ignore[union-attr]

        started = time.monotonic()
        response = self._invoke(
            route,
            "Stop",
            self._fixture("codex", "stop.input.json"),
            ("--stop-grace", "0.5"),
        )

        self.assertEqual(response, {})
        self.assertLess(time.monotonic() - started, 0.1)
        self.assertFalse(marker.exists())  # type: ignore[union-attr]

    def test_replays_fail_closed_for_recursive_stop_stale_root_and_session(self) -> None:
        for agent in ("codex", "claude"):
            with self.subTest(agent=agent):
                self.source.write_text("value = 1\n", encoding="utf-8")
                self.spool = self.base / f"fail-closed-{agent}" / "feedback"
                route, _ = self._route(agent)
                sink = self._publish(route)
                recursive = self._invoke(
                    route, "Stop", self._fixture(agent, "stop_recursive.input.json")
                )
                self.assertEqual(recursive, {})
                self.assertEqual(len(list((self.spool / "pending").glob("*.json"))), 1)

                wrong_root = self._fixture(agent, "stop.input.json")
                wrong_root["cwd"] = os.fspath(self.base / "other")
                self.assertEqual(self._invoke(route, "Stop", wrong_root), {})
                wrong_session = self._fixture(agent, "stop.input.json")
                wrong_session["session_id"] = "other-live-session"
                self.assertEqual(self._invoke(route, "Stop", wrong_session), {})

                self.source.write_text("value = 2\n", encoding="utf-8")
                self.assertEqual(
                    self._invoke(route, "Stop", self._fixture(agent, "stop.input.json")),
                    {},
                )
                self.assertEqual(len(list((self.spool / "acknowledged").glob("*.json"))), 1)
                sink.close()

    def test_replay_recovers_abandoned_claim(self) -> None:
        route, _ = self._route()
        sink = self._publish(route)
        claim = codex_feedback_hook.claim_feedback(
            self.spool, root=self.root, session_id=route.session_id
        )
        self.assertIsNotNone(claim)
        old = time.time() - 10
        os.utime(claim, (old, old))
        input_value = self._fixture("codex", "post_tool_use.input.json")
        output = io.StringIO()
        with mock.patch("sys.stdin", io.StringIO(json.dumps(input_value))), mock.patch(
            "sys.stdout", output
        ):
            codex_feedback_hook.main(
                [
                    "--event", "PostToolUse",
                    "--spool-dir", route.spool_dir,
                    "--session-id", route.session_id,
                    "--root", route.root,
                    "--claim-timeout", "1",
                ]
            )
        self.assertIn("additionalContext", output.getvalue())
        sink.close()

    def test_status_and_cleanup_are_session_scoped_and_preserve_pending_by_default(self) -> None:
        route, _ = self._route()
        sink = self._publish(route)
        codex_feedback_hook.verify_session_lease(
            self.spool,
            root=self.root,
            configured_session_id=route.session_id,
            codex_session_id="live-agent-session",
        )
        status = agent_integration.route_status(route)
        self.assertEqual(status["bound_agent_session"], "live-agent-session")
        self.assertEqual(status["feedback"]["pending"], 1)  # type: ignore[index]
        with self.assertRaises(RuntimeError):
            agent_integration.cleanup_route(
                route, agent_session_id="live-agent-session"
            )
        # Close the producer before identity and destructive-cleanup checks.
        sink.close()
        with self.assertRaises(PermissionError):
            agent_integration.cleanup_route(
                route, agent_session_id="different", discard_feedback=True
            )
        removed = agent_integration.cleanup_route(
            route,
            agent_session_id="live-agent-session",
            discard_feedback=True,
        )
        self.assertEqual(removed["pending"], 1)
        self.assertEqual(removed["session_lease"], 1)

    def test_platform_support_fails_closed(self) -> None:
        with mock.patch.object(agent_integration.os, "name", "nt"):
            with self.assertRaisesRegex(RuntimeError, "requires POSIX"):
                agent_integration.require_secure_platform()
        with mock.patch.object(agent_integration, "fcntl", None):
            with self.assertRaisesRegex(RuntimeError, "requires POSIX"):
                agent_integration.require_secure_platform()

    def test_cleanup_holds_root_lease_through_feedback_deletion(self) -> None:
        route, _ = self._route()
        sink = self._publish(route)
        sink.close()
        original_owned = agent_integration._payload_owned
        attempted_start = False

        def owned_while_starting(path: Path, checked_route: object) -> bool:
            nonlocal attempted_start
            if path.parent.name == "pending" and not attempted_start:
                attempted_start = True
                with self.assertRaisesRegex(ValueError, "already leased"):
                    SpoolSink(
                        self.spool,
                        root=self.root,
                        session_id=route.session_id,
                    )
            return original_owned(path, checked_route)  # type: ignore[arg-type]

        with mock.patch.object(
            agent_integration, "_payload_owned", side_effect=owned_while_starting
        ):
            agent_integration.cleanup_route(
                route, agent_session_id=None, discard_feedback=True
            )
        self.assertTrue(attempted_start)

    def test_separate_worktree_cleanup_cannot_touch_another_route(self) -> None:
        route_a, _ = self._route("codex")
        root_b = self.base / "other-worktree"
        source_b = root_b / "src" / "app.py"
        source_b.parent.mkdir(parents=True)
        source_b.write_text("value = 1\n", encoding="utf-8")
        spool_b = self.base / "runtime-b" / "feedback"
        route_b_path, _ = agent_integration.initialize(
            "claude",
            root=root_b,
            spool_dir=spool_b,
            session_id="claude-other-worktree",
            hook_command=self._executable("other-claude-hook"),
            agent_command=self._executable("other-agent-command"),
        )
        route_b = agent_integration.load_route(route_b_path)

        sink_a = self._publish(route_a)
        raw_b = source_b.read_bytes()
        batch_b = parse_review_output(
            finding_output(),
            root=root_b,
            reviewed_files=(
                ReviewedFile(
                    "src/app.py", hashlib.sha256(raw_b).hexdigest(), len(raw_b)
                ),
            ),
            session_id=route_b.session_id,
        )
        sink_b = SpoolSink(spool_b, root=root_b, session_id=route_b.session_id)
        self.addCleanup(sink_b.close)
        self.assertTrue(sink_b.publish(batch_b))
        codex_feedback_hook.verify_session_lease(
            self.spool,
            root=self.root,
            configured_session_id=route_a.session_id,
            codex_session_id="agent-a",
        )
        codex_feedback_hook.verify_session_lease(
            spool_b,
            root=root_b,
            configured_session_id=route_b.session_id,
            codex_session_id="agent-b",
        )

        sink_a.close()
        agent_integration.cleanup_route(
            route_a, agent_session_id="agent-a", discard_feedback=True
        )
        self.assertEqual(len(list((spool_b / "pending").glob("*.json"))), 1)
        self.assertEqual(
            agent_integration.route_status(route_b)["bound_agent_session"],
            "agent-b",
        )
        sink_b.close()

    def test_live_capture_metadata_is_versioned_and_privacy_bounded(self) -> None:
        captures = (
            FIXTURES / "codex" / "codex-cli-0.150.1-live-2026-08-30",
            FIXTURES / "claude-code" / "claude-code-2.1.251-live-2026-08-30",
        )
        for capture in captures:
            with self.subTest(capture=capture):
                metadata = json.loads((capture / "metadata.json").read_text())
                event = json.loads((capture / "post_tool_use.event.json").read_text())
                self.assertEqual(metadata["provenance"], "live-capture")
                self.assertTrue(metadata["agent_version"])
                self.assertEqual(metadata["provider_inference"], "synthetic; not exercised")
                self.assertEqual(event["root"], "${ROOT}")
                self.assertEqual(event["session_id"], "${CONFIGURED_SESSION}")
                self.assertNotIn("tool_input", event)
                self.assertNotIn("tool_response", event)
                self.assertEqual(len(event["response_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
