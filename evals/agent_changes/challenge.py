"""Challenge-candidate fixture loading, replay metadata, and oracle execution."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


CHALLENGE_ROOT = Path(__file__).resolve().parent / "challenge"
INDEX_PATH = CHALLENGE_ROOT / "manifest.json"
SPLITS = ("challenge-development", "challenge-holdout")


def load_index(root: Path = CHALLENGE_ROOT) -> dict[str, Any]:
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))


def split_directory(split: str, root: Path = CHALLENGE_ROOT) -> Path:
    if split not in SPLITS:
        raise ValueError(f"Unknown challenge split: {split}")
    return root / split.removeprefix("challenge-")


def load_split(split: str, root: Path = CHALLENGE_ROOT) -> dict[str, Any]:
    directory = split_directory(split, root)
    return json.loads((directory / "manifest.json").read_text(encoding="utf-8"))


def model_cases(split: str, root: Path = CHALLENGE_ROOT) -> list[dict[str, Any]]:
    """Flatten pairs while ensuring a replay sees only one twin at a time."""
    directory = split_directory(split, root)
    manifest = load_split(split, root)
    cases: list[dict[str, Any]] = []
    for pair in manifest["pairs"]:
        for variant in ("buggy", "clean"):
            case_id = f"{pair['id']}__{variant}"
            finding = pair["expected_finding"] if variant == "buggy" else None
            cases.append({
                "id": case_id,
                "difficulty": "challenge-candidate" if variant == "buggy" else "clean-twin",
                "files": pair[variant]["files"],
                "failure_families": pair["failure_families"],
                "scope": pair["scope"],
                "expected_evidence_depth": pair["expected_evidence_depth"],
                "evaluation_split": split,
                "expected_findings": [finding] if finding else [],
                "challenge_pair_id": pair["id"],
                "challenge_variant": variant,
                "qualification": pair["qualification"],
                "_source_root": directory / "cases" / pair["id"] / variant,
            })
    return cases


def case_by_id(case_id: str, root: Path = CHALLENGE_ROOT) -> dict[str, Any]:
    for split in SPLITS:
        for case in model_cases(split, root):
            if case["id"] == case_id:
                return case
    raise ValueError(f"Unknown challenge case: {case_id}")


def run_oracle(pair: dict[str, Any], variant: str, *, split: str,
               root: Path = CHALLENGE_ROOT) -> subprocess.CompletedProcess[str]:
    directory = split_directory(split, root) / "cases" / pair["id"] / variant
    return subprocess.run(
        [sys.executable, "oracle_test.py"], cwd=directory, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


def unseal(pair_id: str, *, root: Path = CHALLENGE_ROOT) -> None:
    """Move an opened holdout pair to development before exposing its answer."""
    holdout = load_split("challenge-holdout", root)
    development = load_split("challenge-development", root)
    matches = [pair for pair in holdout["pairs"] if pair["id"] == pair_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown sealed pair: {pair_id}")
    pair = matches[0]
    source = split_directory("challenge-holdout", root) / "cases" / pair_id
    destination = split_directory("challenge-development", root) / "cases" / pair_id
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(source, destination)
    holdout["pairs"] = [item for item in holdout["pairs"] if item["id"] != pair_id]
    pair["qualification"]["seal_status"] = "opened-for-development"
    development["pairs"].append(pair)
    for split, value in (("challenge-holdout", holdout),
                         ("challenge-development", development)):
        path = split_directory(split, root) / "manifest.json"
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage deterministic challenge candidates")
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify", help="run all buggy and clean oracles")
    verify.add_argument("split", choices=(*SPLITS, "all"), default="all", nargs="?")
    opened = subparsers.add_parser("open-answer", help="move a sealed pair to development")
    opened.add_argument("pair_id")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "open-answer":
        unseal(args.pair_id)
        print(f"Moved {args.pair_id} from challenge-holdout to challenge-development")
        return 0
    splits = SPLITS if args.split == "all" else (args.split,)
    failed = False
    for split in splits:
        for pair in load_split(split)["pairs"]:
            buggy = run_oracle(pair, "buggy", split=split)
            clean = run_oracle(pair, "clean", split=split)
            valid = buggy.returncode != 0 and clean.returncode == 0
            failed |= not valid
            print(
                f"{pair['id']}: buggy={buggy.returncode} clean={clean.returncode} "
                f"{'PASS' if valid else 'FAIL'}"
            )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
