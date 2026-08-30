from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import agent_integration
import codex_feedback_hook
from feedback import ReviewedFile, SpoolSink, parse_review_output


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

    def _publish(self, route: agent_integration.RouteConfig, count: int = 1) -> SpoolSink:
        raw = self.source.read_bytes()
        reviewed = (
            ReviewedFile(
                "src/app.py", hashlib.sha256(raw).hexdigest(), len(raw)
            ),
        )
        batch = parse_review_output(
            finding_output(count),
            root=self.root,
            reviewed_files=reviewed,
            session_id=route.session_id,
            debounce_ms=3_000,
            provider_ms=125,
        )
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
                ]
            )
        self.assertEqual(result, 0)
        return json.loads(output.getvalue()) if output.getvalue() else None

    def _release_from_fixture(
        self,
        route: agent_integration.RouteConfig,
        input_value: dict[str, object],
    ) -> dict[str, object]:
        output = io.StringIO()
        with mock.patch("sys.stdin", io.StringIO(json.dumps(input_value))), mock.patch(
            "sys.stdout", output
        ):
            result = agent_integration.main(
                ["cleanup", "--config", os.fspath(agent_integration.route_path(self.spool)),
                 "--from-hook"]
            )
        self.assertEqual(result, 0)
        return json.loads(output.getvalue())

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

                stop = self._invoke(
                    route, "Stop", self._fixture(agent, "stop.input.json")
                )
                self.assertEqual(stop["decision"], "block")  # type: ignore[index]
                self.assertEqual(len(list((self.spool / "pending").glob("*.json"))), 0)
                latency = agent_integration.route_status(route)["latency_ms"]
                self.assertEqual(latency["samples"], 2)  # type: ignore[index]
                self.assertIsNotNone(latency["hook_execution_ms"]["p95"])  # type: ignore[index]
                sink.close()
                agent_integration.cleanup_route(
                    route, agent_session_id="live-agent-session"
                )
                self.assertEqual(
                    len(list((self.spool / "metrics").glob("*.json"))), 2
                )

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

    def test_session_end_releases_active_watcher_and_isolates_next_session(self) -> None:
        for agent in ("codex", "claude"):
            with self.subTest(agent=agent):
                self.spool = self.base / f"session-handoff-{agent}" / "feedback"
                self.source.write_text("value = 1\nother = 2\n", encoding="utf-8")
                route, _ = self._route(agent)
                sink = self._publish(route)
                self.assertTrue(
                    codex_feedback_hook.verify_session_lease(
                        self.spool,
                        root=self.root,
                        configured_session_id=route.session_id,
                        codex_session_id="live-agent-session",
                    )
                )
                self.assertEqual(sink.capture_session_generation(), 0)
                raw = self.source.read_bytes()
                late_batch = parse_review_output(
                    finding_output(2),
                    root=self.root,
                    reviewed_files=(
                        ReviewedFile(
                            "src/app.py", hashlib.sha256(raw).hexdigest(), len(raw)
                        ),
                    ),
                    session_id=route.session_id,
                    session_generation=0,
                )

                released = self._release_from_fixture(
                    route, self._fixture(agent, "session_end.input.json")
                )
                self.assertEqual(released, {"session_released": True})
                status = agent_integration.route_status(route)
                self.assertTrue(status["producer_active"])
                self.assertEqual(status["session_state"], "closed")
                self.assertEqual(status["session_generation"], 1)
                self.assertIsNone(status["bound_agent_session"])

                self.assertTrue(
                    codex_feedback_hook.verify_session_lease(
                        self.spool,
                        root=self.root,
                        configured_session_id=route.session_id,
                        codex_session_id="next-agent-session",
                    )
                )
                self.assertEqual(sink.capture_session_generation(), 1)
                self.assertFalse(sink.publish(late_batch))
                next_batch = parse_review_output(
                    finding_output(),
                    root=self.root,
                    reviewed_files=(
                        ReviewedFile(
                            "src/app.py", hashlib.sha256(raw).hexdigest(), len(raw)
                        ),
                    ),
                    session_id=route.session_id,
                    session_generation=1,
                )
                self.assertTrue(sink.publish(next_batch))

                next_input = self._fixture(agent, "post_tool_use.input.json")
                next_input["session_id"] = "next-agent-session"
                delivered = self._invoke(route, "PostToolUse", next_input)
                self.assertIn("Finding 1", json.dumps(delivered))
                pending = list((self.spool / "pending").glob("*.json"))
                self.assertEqual(len(pending), 1)
                old_payload = json.loads(pending[0].read_text(encoding="utf-8"))
                self.assertEqual(old_payload["session_generation"], 0)
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
