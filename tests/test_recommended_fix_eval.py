from __future__ import annotations

import json
import unittest
from pathlib import Path

import watch_files
from evals.recommended_fixes import scoring


EVAL_ROOT = Path(__file__).resolve().parents[1] / "evals" / "recommended_fixes"


class RecommendedFixEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = json.loads((EVAL_ROOT / "fixture.json").read_text())

    def test_frozen_fixture_sources_and_recommendations_are_bounded(self) -> None:
        schema = watch_files.REVIEW_SCHEMA["properties"]["findings"]["items"]
        bounds = schema["properties"]["suggested_fix"]

        self.assertEqual(self.fixture["fixture_revision"], 2)
        for case in self.fixture["cases"]:
            with self.subTest(case=case["id"]):
                for path in case["supplied_files"]:
                    self.assertTrue((EVAL_ROOT / path).is_file())
                for candidate in case["candidates"]:
                    recommendation = candidate["suggested_fix"]
                    self.assertGreaterEqual(len(recommendation), bounds["minLength"])
                    self.assertLessEqual(len(recommendation), bounds["maxLength"])

    def test_absent_test_fixture_rejects_invented_existing_test(self) -> None:
        case = self._case("no-supplied-test")
        results = {
            candidate["id"]: scoring.evaluate_recommendation(
                candidate["suggested_fix"], supplied_files=case["supplied_files"]
            )
            for candidate in case["candidates"]
        }

        self.assertEqual(results["add-test"]["status"], "grounded")
        self.assertEqual(results["invent-existing-test"]["status"], "failure")
        self.assertEqual(
            results["invent-existing-test"]["violations"][0]["code"],
            "unsupported-existing-test-claim",
        )

    def test_supplied_test_fixture_can_extend_named_test(self) -> None:
        case = self._case("supplied-test")
        result = scoring.evaluate_recommendation(
            case["candidates"][0]["suggested_fix"],
            supplied_files=case["supplied_files"],
        )

        self.assertEqual(result["status"], "grounded")
        self.assertEqual(result["supplied_test_files"], ["test_expired_cache.py"])

    def test_unsupplied_test_path_is_a_separate_grounding_failure(self) -> None:
        result = scoring.evaluate_recommendation(
            "Extend tests/test_cache.py with an expired-entry assertion.",
            supplied_files=["expired_cache.py"],
        )

        self.assertEqual(result["status"], "failure")
        self.assertEqual(
            {violation["code"] for violation in result["violations"]},
            {"unsupported-test-mutation", "unsupplied-test-path"},
        )

    def test_production_change_followed_by_new_test_is_grounded(self) -> None:
        for recommendation in (
            "Modify Cache.get(), then add a regression test for expired entries.",
            "Update Cache.get() and add tests/test_cache.py for expired entries.",
        ):
            with self.subTest(recommendation=recommendation):
                result = scoring.evaluate_recommendation(
                    recommendation,
                    supplied_files=["expired_cache.py"],
                )
                self.assertEqual(result["status"], "grounded")

    def test_new_test_path_is_allowed_but_mutating_unsupplied_path_is_not(self) -> None:
        proposed = scoring.evaluate_recommendation(
            "Add tests/test_cache.py to cover expiry.",
            supplied_files=["expired_cache.py"],
        )
        mutation = scoring.evaluate_recommendation(
            "Add an expiry assertion to tests/test_cache.py.",
            supplied_files=["expired_cache.py"],
        )

        self.assertEqual(proposed["status"], "grounded")
        self.assertEqual(mutation["status"], "failure")
        self.assertEqual(
            mutation["violations"][0]["code"],
            "unsupplied-test-path",
        )

    def test_rspec_path_is_recognized_as_supplied_test_evidence(self) -> None:
        result = scoring.evaluate_recommendation(
            "Extend the test in spec/cache_spec.rb with an expired entry.",
            supplied_files=["cache.rb", "spec/cache_spec.rb"],
        )

        self.assertEqual(result["status"], "grounded")
        self.assertEqual(result["supplied_test_files"], ["spec/cache_spec.rb"])

    def test_unrelated_supplied_test_does_not_ground_generic_claim(self) -> None:
        for recommendation in (
            "Preserve the existing cache regression test.",
            "Extend the cache regression test.",
        ):
            with self.subTest(recommendation=recommendation):
                result = scoring.evaluate_recommendation(
                    recommendation,
                    supplied_files=["cache.py", "tests/test_auth.py"],
                )
                self.assertEqual(result["status"], "failure")

        named = scoring.evaluate_recommendation(
            "Extend tests/test_auth.py with another unauthorized request.",
            supplied_files=["cache.py", "tests/test_auth.py"],
        )
        self.assertEqual(named["status"], "grounded")

    def _case(self, case_id: str) -> dict[str, object]:
        return next(case for case in self.fixture["cases"] if case["id"] == case_id)


if __name__ == "__main__":
    unittest.main()
