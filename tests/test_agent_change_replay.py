from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evals.agent_changes import live_eval
from evals.agent_changes import replay


class AgentChangeReplayTests(unittest.TestCase):
    def test_manifest_declares_obvious_subtle_and_clean_cases(self) -> None:
        manifest = replay.load_manifest()
        difficulties = {case["difficulty"] for case in manifest["cases"]}
        self.assertIn("obvious", difficulties)
        self.assertIn("subtle", difficulties)
        self.assertIn("sophisticated", difficulties)
        self.assertIn("clean-control", difficulties)

        for case in manifest["cases"]:
            fixture_names = sorted(
                path.name for path in (replay.CASES_ROOT / case["id"]).iterdir()
            )
            self.assertEqual(sorted(case["files"]), fixture_names)

    def test_replay_writes_related_files_to_one_case_directory(self) -> None:
        manifest = replay.load_manifest()
        case = replay.case_by_id(manifest, "03_cross_file_units")

        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory)
            written = replay.replay_case(
                case,
                destination=destination,
                inter_file_delay=0,
            )

            self.assertEqual(
                [path.name for path in written],
                ["token_model.py", "token_service.py"],
            )
            self.assertTrue(all(path.is_file() for path in written))
            expected_parent = destination.resolve() / case["id"]
            self.assertTrue(all(path.parent == expected_parent for path in written))

            first_contents = [path.read_bytes() for path in written]
            replayed = replay.replay_case(
                case,
                destination=destination,
                inter_file_delay=0,
            )
            self.assertNotEqual(
                first_contents,
                [path.read_bytes() for path in replayed],
            )

    def test_live_eval_scores_expected_finding_files_and_clean_control(self) -> None:
        manifest = replay.load_manifest()
        defect_case = replay.case_by_id(manifest, "03_cross_file_units")
        clean_case = replay.case_by_id(manifest, "08_clean_related_change")

        defect_result = live_eval.score_response(
            defect_case,
            {"findings": [{"file": "03_cross_file_units/token_service.py"}]},
        )
        clean_result = live_eval.score_response(clean_case, {"findings": []})

        self.assertTrue(defect_result.passed)
        self.assertTrue(clean_result.passed)

    def test_live_eval_rejects_missing_or_extra_finding_files(self) -> None:
        case = replay.case_by_id(replay.load_manifest(), "03_cross_file_units")

        missing = live_eval.score_response(case, {"findings": []})
        extra = live_eval.score_response(
            case,
            {
                "findings": [
                    {"file": "token_service.py"},
                    {"file": "token_model.py"},
                ]
            },
        )

        self.assertFalse(missing.passed)
        self.assertFalse(extra.passed)


if __name__ == "__main__":
    unittest.main()
