from __future__ import annotations

import hashlib
import json
import tempfile
import time
import unittest
from pathlib import Path

from feedback import (
    ReviewValidationError,
    ReviewedFile,
    SpoolSink,
    fresh_findings,
    parse_review_output,
    validate_spooled_payload,
)
from review_lifecycle import (
    FindingLifecycleTracker,
    batch_timing,
    short_batch_id,
)
from review_output import render_human_review


class ReviewLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.root = self.base / "project"
        self.source = self.root / "src" / "service.py"
        self.source.parent.mkdir(parents=True)
        self.source.write_text("value = 1\n", encoding="utf-8")
        self.base_time = time.time() - 10

    def batch(
        self,
        *,
        title: str | None = "Cleanup races with pending insert",
        line: int = 10,
        sequence: int = 1,
    ):
        raw = self.source.read_bytes()
        reviewed = (
            ReviewedFile(
                "src/service.py", hashlib.sha256(raw).hexdigest(), len(raw)
            ),
        )
        findings = []
        if title is not None:
            findings.append(
                {
                    "file": "src/service.py",
                    "line": line,
                    "severity": "medium",
                    "confidence": 0.99,
                    "title": title,
                    "explanation": "A concrete execution path leaves state behind.",
                    "suggested_fix": "Settle the task before cleanup and add a test.",
                }
            )
        flushed_at = self.base_time + sequence
        return parse_review_output(
            json.dumps({"findings": findings}),
            root=self.root,
            reviewed_files=reviewed,
            session_id="agent-a",
            debounce_ms=250,
            provider_ms=500,
            first_observed_at=flushed_at - 0.25,
            batch_flushed_at=flushed_at,
            provider_started_at=flushed_at,
            provider_completed_at=flushed_at + 0.5,
        )

    def test_repeat_finding_is_retained_when_only_line_changes(self) -> None:
        tracker = FindingLifecycleTracker()

        first = tracker.classify(self.batch(line=10, sequence=1))
        second = tracker.classify(self.batch(line=14, sequence=2))

        self.assertEqual(first.lifecycle[0].status, "new")
        self.assertEqual(second.lifecycle[0].status, "retained")
        self.assertEqual(second.lifecycle[0].line, 14)
        self.assertEqual(
            second.lifecycle[0].fingerprint,
            second.lifecycle[0].previous_fingerprint,
        )

    def test_replacement_and_model_omission_are_distinct(self) -> None:
        tracker = FindingLifecycleTracker()
        tracker.classify(self.batch(sequence=1))

        replaced = tracker.classify(
            self.batch(title="Cleanup drops the original exception", sequence=2)
        )
        omitted = tracker.classify(self.batch(title=None, sequence=3))

        self.assertEqual([item.status for item in replaced.lifecycle], ["replaced"])
        self.assertNotEqual(
            replaced.lifecycle[0].fingerprint,
            replaced.lifecycle[0].previous_fingerprint,
        )
        self.assertEqual(
            [item.status for item in omitted.lifecycle], ["no_longer_reported"]
        )

    def test_changed_source_and_out_of_order_completion_are_stale(self) -> None:
        tracker = FindingLifecycleTracker()
        latest = tracker.classify(self.batch(sequence=3))
        self.assertEqual(latest.lifecycle[0].status, "new")

        out_of_order = tracker.classify(
            self.batch(title="Older unrelated result", sequence=2)
        )
        self.assertEqual(out_of_order.lifecycle[0].status, "stale")
        self.assertEqual(out_of_order.lifecycle[0].reason, "out_of_order")

        source_changed = self.batch(sequence=4)
        self.source.write_text("value = 2\n", encoding="utf-8")
        stale = tracker.classify(fresh_findings(source_changed))
        self.assertEqual(stale.findings, ())
        self.assertEqual(stale.stale_files, ("src/service.py",))
        self.assertEqual(stale.lifecycle[0].status, "stale")
        self.assertEqual(stale.lifecycle[0].reason, "source_changed")

        # Neither stale result replaced the last current lifecycle state.
        retained = tracker.classify(self.batch(line=20, sequence=5))
        self.assertEqual(retained.lifecycle[0].status, "retained")

    def test_progress_is_one_line_and_never_claims_omission_is_resolved(self) -> None:
        tracker = FindingLifecycleTracker()
        tracker.classify(self.batch(sequence=1))
        omitted = tracker.classify(self.batch(title=None, sequence=2))

        output = render_human_review(omitted)

        self.assertEqual(len(output.splitlines()), 1)
        self.assertIn(short_batch_id(omitted.batch_id), output)
        self.assertIn("no longer reported in the latest snapshot", output)
        self.assertNotIn("resolved", output.lower())
        self.assertIn("debounce 250.0ms", output)
        self.assertIn("provider 500.0ms", output)

    def test_lifecycle_and_timing_survive_spool_json_validation(self) -> None:
        tracker = FindingLifecycleTracker()
        batch = tracker.classify(self.batch(sequence=1))
        spool = self.base / "runtime" / "feedback"
        sink = SpoolSink(spool, root=self.root, session_id="agent-a")
        self.addCleanup(sink.close)

        self.assertTrue(sink.publish(batch))
        payload = json.loads(next((spool / "pending").glob("*.json")).read_text())
        validated = validate_spooled_payload(
            payload, root=self.root, session_id="agent-a"
        )

        self.assertEqual(validated["lifecycle"][0]["status"], "new")
        self.assertEqual(validated["stale_files"], [])
        timing = batch_timing(batch, delivered_at=batch.created_at + 0.2)
        self.assertEqual(timing.debounce_ms, 250)
        self.assertEqual(timing.provider_ms, 500)
        self.assertGreaterEqual(timing.publication_ms, 0)
        self.assertGreaterEqual(timing.agent_delivery_ms, 0)
        self.assertGreater(timing.total_ms, 0)

        payload["lifecycle"][0]["status"] = "resolved"
        with self.assertRaises(ReviewValidationError):
            validate_spooled_payload(
                payload, root=self.root, session_id="agent-a"
            )


if __name__ == "__main__":
    unittest.main()
