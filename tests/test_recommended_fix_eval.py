from __future__ import annotations

import json
import unittest
from pathlib import Path

import watch_files


EVAL_ROOT = Path(__file__).resolve().parents[1] / "evals" / "recommended_fixes"


class RecommendedFixEvaluationTests(unittest.TestCase):
    def test_frozen_fixture_demonstrates_actionable_recommendation_shape(self) -> None:
        fixture = json.loads((EVAL_ROOT / "fixture.json").read_text())
        source = (EVAL_ROOT / fixture["input_file"]).read_text()
        finding = fixture["expected_finding"]
        recommendation = finding["suggested_fix"]
        expectations = fixture["recommendation_expectations"]
        schema = watch_files.REVIEW_SCHEMA["properties"]["findings"]["items"]
        bounds = schema["properties"]["suggested_fix"]

        self.assertEqual(fixture["fixture_revision"], 1)
        self.assertIn("class Cache", source)
        self.assertLessEqual(set(schema["required"]), set(finding))
        self.assertLessEqual(set(finding), set(schema["properties"]))
        self.assertGreaterEqual(len(recommendation), bounds["minLength"])
        self.assertLessEqual(len(recommendation), bounds["maxLength"])
        for category, terms in expectations.items():
            for term in terms:
                with self.subTest(category=category, term=term):
                    self.assertIn(term, recommendation)


if __name__ == "__main__":
    unittest.main()
