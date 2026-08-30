from __future__ import annotations

from dataclasses import fields, replace
import json
import math
import unittest

from feedback import ReviewBatch, ReviewFinding, ReviewedFile
from redaction import RedactionNotice, RedactionSummary
from review_lifecycle import FindingLifecycle
from review_output import (
    OUTPUT_SCHEMA_VERSION,
    render_human_review,
    render_json_review,
    review_output_document,
)


def finding(
    *,
    file: str = "src/service.py",
    line: int = 10,
    title: str = "Failed cleanup races with a pending insert",
) -> ReviewFinding:
    return ReviewFinding(
        file=file,
        line=line,
        severity="medium",
        confidence=0.99,
        title=title,
        explanation="A delayed task can write after rollback completes.",
        suggested_fix="Cancel and await both tasks before deleting records.",
    )


def batch(*findings: ReviewFinding) -> ReviewBatch:
    return ReviewBatch(
        batch_id="3ea14b6a-36d4-4fb7-bc08-41bea4bc48fe",
        root="/workspace/project",
        created_at=1_788_120_000.0,
        reviewed_files=(
            ReviewedFile(path="src/service.py", sha256="a" * 64, size=120),
            ReviewedFile(path="src/repository.py", sha256="b" * 64, size=240),
        ),
        findings=findings,
        session_id="agent-a",
        session_generation=3,
        feedback_round=2,
        debounce_ms=3_000.0,
        provider_ms=425.5,
        first_observed_at=1_788_119_996.5,
        batch_flushed_at=1_788_119_999.5,
        provider_started_at=1_788_119_999.6,
        provider_completed_at=1_788_120_000.0,
        published_at=1_788_120_000.1,
    )


class ReviewOutputTests(unittest.TestCase):
    def test_zero_findings_collapse_to_one_human_line(self) -> None:
        rendered = render_human_review(batch())

        self.assertEqual(
            rendered,
            "qdt-3ea14b6a reviewed 2 files in 3.60s: no confident findings "
            "[debounce 3000.0ms, provider 425.5ms, publication 100.0ms]",
        )
        self.assertNotIn("{", rendered)

    def test_one_finding_has_compact_summary_evidence_and_action(self) -> None:
        rendered = render_human_review(batch(finding()))

        self.assertEqual(
            rendered,
            "\n".join(
                [
                    "qdt-3ea14b6a reviewed 2 files in 3.60s: 1 likely defect "
                    "[debounce 3000.0ms, provider 425.5ms, publication 100.0ms]",
                    "src/service.py:10 [medium, 0.99] Failed cleanup races with a pending insert",
                    "  A delayed task can write after rollback completes.",
                    "  Suggested action: Cancel and await both tasks before deleting records.",
                ]
            ),
        )

    def test_multiple_findings_render_in_validated_order(self) -> None:
        rendered = render_human_review(
            batch(
                finding(),
                finding(file="src/repository.py", line=21, title="Second defect"),
            )
        )

        self.assertEqual(rendered.count("Suggested action:"), 2)
        self.assertIn("2 likely defects", rendered.splitlines()[0])
        self.assertLess(
            rendered.index("src/service.py"), rendered.index("src/repository.py")
        )

    def test_human_output_neutralizes_terminal_controls_and_abbreviates(self) -> None:
        unsafe = finding(title="danger\x1b[31m\n" + "x" * 300)

        rendered = render_human_review(batch(unsafe))

        self.assertNotIn("\x1b", rendered)
        self.assertNotIn("\nxxxx", rendered)
        self.assertIn("…", rendered)

    def test_json_escapes_lone_surrogates_for_utf8_stdout(self) -> None:
        rendered = render_json_review(batch(finding(title="broken \ud800 value")))

        self.assertIn(r"\ud800", rendered)
        self.assertNotIn("\ud800", rendered)
        rendered.encode("utf-8")

    def test_json_is_versioned_complete_and_deterministic(self) -> None:
        review = batch(finding())
        first = render_json_review(review)
        second = render_json_review(review)
        document = json.loads(first)

        self.assertEqual(first, second)
        self.assertNotIn("\n", first)
        self.assertEqual(document, review_output_document(review))
        self.assertEqual(document["schema_version"], OUTPUT_SCHEMA_VERSION)
        self.assertEqual(
            document["findings"][0]["suggested_fix"],
            review.findings[0].suggested_fix,
        )
        self.assertEqual(document["reviewed_files"][1]["sha256"], "b" * 64)
        self.assertEqual(
            document["timing"],
            {
                "first_observed_at": 1_788_119_996.5,
                "batch_flushed_at": 1_788_119_999.5,
                "debounce_ms": 3_000.0,
                "provider_started_at": 1_788_119_999.6,
                "provider_completed_at": 1_788_120_000.0,
                "provider_ms": 425.5,
                "published_at": 1_788_120_000.1,
            },
        )
        self.assertEqual(
            set(document),
            {
                "schema_version",
                "batch_id",
                "root",
                "created_at",
                "reviewed_files",
                "findings",
                "session_id",
                "session_generation",
                "feedback_round",
                "timing",
                "redactions",
                "lifecycle",
                "stale_files",
            },
        )
        serialized_batch_fields = (
            set(document) - {"schema_version", "timing"}
        ) | set(document["timing"])
        self.assertEqual(
            serialized_batch_fields,
            {field.name for field in fields(ReviewBatch)},
        )
        self.assertEqual(
            document["redactions"], {"total": 0, "notices": [], "omitted": 0}
        )

        with self.assertRaises(ValueError):
            render_json_review(replace(review, provider_ms=math.nan))

    def test_human_and_json_outputs_expose_equivalent_safe_redaction_hints(self) -> None:
        summary = RedactionSummary(
            total=2,
            notices=(
                RedactionNotice(
                    file="src/app.py",
                    line=4,
                    category="assignment-key",
                    masked_identifier="OPEN…KEY",
                    disposition="sent",
                ),
                RedactionNotice(
                    file="[REDACTED].env",
                    line=None,
                    category="provider-token",
                    masked_identifier=None,
                    disposition="excluded",
                ),
            ),
            omitted=0,
        )
        review = replace(batch(), redactions=summary)

        human = render_human_review(review)
        document = json.loads(render_json_review(review))

        self.assertIn("src/app.py:4 assignment key OPEN…KEY (sent sanitized)", human)
        self.assertIn("[REDACTED].env provider token (excluded)", human)
        self.assertEqual(document["redactions"]["total"], 2)
        self.assertEqual(
            document["redactions"]["notices"][0],
            {
                "file": "src/app.py",
                "line": 4,
                "category": "assignment-key",
                "masked_identifier": "OPEN…KEY",
                "disposition": "sent",
            },
        )

    def test_lifecycle_is_explicit_in_json_and_compact_in_human_output(self) -> None:
        fingerprint = "a" * 64
        omitted = replace(
            batch(),
            lifecycle=(
                FindingLifecycle(
                    status="no_longer_reported",
                    fingerprint=fingerprint,
                    file="src/service.py",
                    line=10,
                    previous_fingerprint=fingerprint,
                ),
            ),
        )

        rendered = render_human_review(omitted)
        document = review_output_document(omitted)

        self.assertIn(
            "1 prior finding no longer reported in the latest snapshot",
            rendered,
        )
        self.assertNotIn("resolved", rendered.lower())
        self.assertEqual(document["lifecycle"][0]["status"], "no_longer_reported")
        self.assertEqual(document["stale_files"], [])

    def test_partial_staleness_keeps_fresh_omission_visible(self) -> None:
        review = replace(
            batch(),
            findings=(),
            lifecycle=(
                FindingLifecycle(
                    status="stale",
                    fingerprint="a" * 64,
                    file="src/service.py",
                    line=10,
                    reason="source_changed",
                ),
                FindingLifecycle(
                    status="no_longer_reported",
                    fingerprint="b" * 64,
                    file="src/repository.py",
                    line=20,
                    previous_fingerprint="b" * 64,
                ),
            ),
            stale_files=("src/service.py",),
        )

        rendered = render_human_review(review)

        self.assertIn("1 prior finding no longer reported", rendered)
        self.assertIn("1 stale reviewed file discarded", rendered)
        self.assertNotIn("discarded after", rendered)

    def test_mixed_findings_show_omissions_in_lifecycle_summary(self) -> None:
        review = replace(
            batch(finding()),
            lifecycle=(
                FindingLifecycle(
                    status="new",
                    fingerprint="a" * 64,
                    file="src/service.py",
                    line=10,
                ),
                FindingLifecycle(
                    status="no_longer_reported",
                    fingerprint="b" * 64,
                    file="src/repository.py",
                    line=20,
                    previous_fingerprint="b" * 64,
                ),
            ),
        )

        rendered = render_human_review(review)

        self.assertIn("1 prior finding no longer reported", rendered.splitlines()[0])

if __name__ == "__main__":
    unittest.main()
