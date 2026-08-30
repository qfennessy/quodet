from __future__ import annotations

import unittest
from dataclasses import asdict
from pathlib import Path, PureWindowsPath

from redaction import (
    MAX_REDACTION_NOTICES,
    MAX_REDACTIONS_PER_BATCH,
    RedactedText,
    RedactionSummaryBuilder,
    mask_identifier,
    redact_path,
    redact_text,
    redaction_summary_from_document,
)


SYNTHETIC_TOKEN_A = "ghp_" + "a1B2" * 8
SYNTHETIC_TOKEN_B = "ghp_" + "c3D4" * 8


class RedactionTests(unittest.TestCase):
    def test_assignment_metadata_uses_only_masked_key_name(self) -> None:
        first = redact_text(f'OPENAI_API_KEY="{SYNTHETIC_TOKEN_A}"')
        second = redact_text(f'OPENAI_API_KEY="{SYNTHETIC_TOKEN_B}"')

        self.assertEqual(first.text, 'OPENAI_API_KEY="[REDACTED]"')
        self.assertEqual(first.detections, second.detections)
        self.assertEqual(first.detections[0].category, "assignment-key")
        self.assertEqual(first.detections[0].masked_identifier, "OPEN…KEY")
        metadata = repr(first.detections)
        self.assertNotIn(SYNTHETIC_TOKEN_A, metadata)
        self.assertNotIn(SYNTHETIC_TOKEN_A[:10], metadata)
        self.assertNotIn(SYNTHETIC_TOKEN_A[-10:], metadata)

    def test_value_only_provider_token_has_no_fabricated_identifier(self) -> None:
        redacted = redact_text(f"before\n{SYNTHETIC_TOKEN_A}\nafter")

        self.assertEqual(redacted.total, 1)
        self.assertEqual(redacted.detections[0].line, 2)
        self.assertEqual(redacted.detections[0].category, "provider-token")
        self.assertIsNone(redacted.detections[0].masked_identifier)

        unknown = redact_text("aB3dE5gH7jK9mN2pQ4sT6vW8yZ1cF0iL")
        self.assertEqual(unknown.detections[0].category, "high-entropy-value")
        self.assertIsNone(unknown.detections[0].masked_identifier)

    def test_private_key_redaction_preserves_following_line_numbers(self) -> None:
        source = "\n".join(
            [
                "-----BEGIN PRIVATE KEY-----",
                "synthetic-material-only",
                "-----END PRIVATE KEY-----",
                f"password={SYNTHETIC_TOKEN_A}",
            ]
        )

        redacted = redact_text(source)

        self.assertEqual([item.line for item in redacted.detections], [1, 4])
        self.assertNotIn("synthetic-material-only", redacted.text)
        self.assertEqual(redacted.text.count("\n"), source.count("\n"))

    def test_multiline_quoted_assignment_preserves_following_line_numbers(self) -> None:
        source = 'PASSWORD="first\nsecond"\nAPI_KEY=synthetic-value\n'

        redacted = redact_text(source)

        self.assertEqual(redacted.text.count("\n"), source.count("\n"))
        self.assertEqual([item.line for item in redacted.detections], [1, 3])
        self.assertNotIn("first", redacted.text)
        self.assertNotIn("second", redacted.text)

    def test_mask_identifier_handles_short_and_unicode_key_names(self) -> None:
        self.assertEqual(mask_identifier("id"), "I…D")
        self.assertEqual(mask_identifier("clé_secrète"), "CLÉS…ÈTE")
        self.assertIsNone(mask_identifier("___"))

    def test_path_redaction_never_returns_the_original_token(self) -> None:
        path = Path("src") / f"{SYNTHETIC_TOKEN_A}.env"

        redacted = redact_path(path)

        self.assertEqual(redacted.total, 1)
        self.assertNotIn(SYNTHETIC_TOKEN_A, redacted.text)
        self.assertIn("[REDACTED]", redacted.text)

        windows = redact_path(PureWindowsPath("src\\nested\\app.py"))
        self.assertEqual(windows.text, "src/nested/app.py")

    def test_summary_is_bounded_and_records_sent_or_excluded(self) -> None:
        source = "\n".join(
            f"API_KEY={SYNTHETIC_TOKEN_A}{index:02d}" for index in range(30)
        )
        redacted = redact_text(source)
        builder = RedactionSummaryBuilder()
        builder.add(redacted, file="src/app.py", disposition="sent")
        builder.add(
            redact_text(f"password={SYNTHETIC_TOKEN_B}"),
            file="src/other.py",
            disposition="excluded",
        )

        summary = builder.build()

        self.assertEqual(len(summary.notices), MAX_REDACTION_NOTICES)
        self.assertEqual(summary.total, 31)
        self.assertEqual(summary.omitted, 11)
        self.assertEqual(summary.notices[0].disposition, "sent")

    def test_summary_count_saturates_instead_of_crashing_watcher(self) -> None:
        builder = RedactionSummaryBuilder()
        detected = RedactedText(
            text="[REDACTED]",
            total=MAX_REDACTIONS_PER_BATCH,
            detections=(),
        )

        builder.add(detected, file="generated.env", disposition="sent")
        builder.add(detected, file="generated.env", disposition="sent")
        summary = builder.build()

        self.assertEqual(summary.total, MAX_REDACTIONS_PER_BATCH)
        self.assertEqual(summary.omitted, MAX_REDACTIONS_PER_BATCH)

    def test_retained_document_validator_rejects_unbounded_or_unsafe_metadata(self) -> None:
        safe = {
            "total": 1,
            "notices": [
                {
                    "file": "src/app.py",
                    "line": 3,
                    "category": "assignment-key",
                    "masked_identifier": "OPEN…KEY",
                    "disposition": "sent",
                }
            ],
            "omitted": 0,
        }
        normalized = asdict(redaction_summary_from_document(safe))
        self.assertEqual(normalized["total"], safe["total"])
        self.assertEqual(list(normalized["notices"]), safe["notices"])
        self.assertEqual(normalized["omitted"], safe["omitted"])

        unsafe_values = [
            {**safe, "notices": [{**safe["notices"][0], "file": SYNTHETIC_TOKEN_A}]},
            {
                **safe,
                "notices": [
                    {**safe["notices"][0], "masked_identifier": SYNTHETIC_TOKEN_A}
                ],
            },
            {**safe, "omitted": 1},
            {**safe, "notices": [{**safe["notices"][0], "file": "src\\app.py"}]},
        ]
        for unsafe in unsafe_values:
            with self.subTest(unsafe=unsafe), self.assertRaises(ValueError):
                redaction_summary_from_document(unsafe)


if __name__ == "__main__":
    unittest.main()
