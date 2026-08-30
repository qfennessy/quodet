"""Replay related file saves to a directory watched by Quodet."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Sequence


EVAL_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = EVAL_ROOT / "manifest.json"
CASES_ROOT = EVAL_ROOT / "cases"
DEFAULT_DESTINATION = Path("prompt_eval_workspace/agent_replay")


def load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def fixture_tree_sha256(manifest: dict[str, Any] | None = None) -> str:
    """Hash case IDs, relative filenames, and exact provider fixture bytes."""
    digest = hashlib.sha256()
    for case in (manifest or load_manifest())["cases"]:
        for filename in case["files"]:
            relative = Path(case["id"]) / filename
            contents = (CASES_ROOT / relative).read_bytes()
            digest.update(relative.as_posix().encode())
            digest.update(b"\0")
            digest.update(len(contents).to_bytes(8, "big"))
            digest.update(contents)
    return digest.hexdigest()


def case_by_id(manifest: dict[str, Any], case_id: str) -> dict[str, Any]:
    for case in manifest["cases"]:
        if case["id"] == case_id:
            return case
    available = ", ".join(case["id"] for case in manifest["cases"])
    raise ValueError(f"Unknown case {case_id!r}. Available cases: {available}")


def replay_case(
    case: dict[str, Any],
    *,
    destination: Path,
    inter_file_delay: float,
) -> list[Path]:
    source_root = CASES_ROOT / case["id"]
    target_root = destination.resolve() / case["id"]
    target_root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for index, filename in enumerate(case["files"]):
        source = source_root / filename
        if not source.is_file():
            raise FileNotFoundError(f"Missing fixture: {source}")

        target = target_root / filename
        temporary = target.with_name(f".{target.name}.quodet-save")
        source_bytes = source.read_bytes()
        temporary.write_bytes(source_bytes)
        os.replace(temporary, target)
        written.append(target)
        if inter_file_delay and index + 1 < len(case["files"]):
            time.sleep(inter_file_delay)

    return written


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay coding-agent file saves into a Quodet watch directory."
    )
    parser.add_argument("case", help="case ID or 'all'")
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_DESTINATION,
        help=f"watched destination (default: {DEFAULT_DESTINATION})",
    )
    parser.add_argument(
        "--inter-file-delay",
        type=float,
        default=0.25,
        metavar="SECONDS",
        help="delay between related saves (default: 0.25)",
    )
    parser.add_argument(
        "--inter-case-delay",
        type=float,
        default=8.0,
        metavar="SECONDS",
        help="delay between cases when replaying all (default: 8)",
    )
    args = parser.parse_args(argv)
    if args.inter_file_delay < 0 or args.inter_case_delay < 0:
        parser.error("delays cannot be negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = load_manifest()
    cases = (
        manifest["cases"]
        if args.case == "all"
        else [case_by_id(manifest, args.case)]
    )

    for index, case in enumerate(cases):
        written = replay_case(
            case,
            destination=args.destination,
            inter_file_delay=args.inter_file_delay,
        )
        print(f"Replayed {case['id']}: {', '.join(path.name for path in written)}")
        expected = [finding["id"] for finding in case["expected_findings"]]
        print(f"Expected findings: {', '.join(expected) if expected else 'none'}")
        if args.inter_case_delay and index + 1 < len(cases):
            time.sleep(args.inter_case_delay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
