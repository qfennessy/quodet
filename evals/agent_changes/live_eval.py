"""Run Quodet and coding-agent change replays in one process tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import queue
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, TextIO

import watch_files
from evals.agent_changes import replay


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WATCHER_PATH = REPOSITORY_ROOT / "watch_files.py"


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    expected_files: tuple[str, ...]
    actual_files: tuple[str, ...]
    response: dict[str, Any] | None
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and self.actual_files == self.expected_files


def expected_files(case: dict[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(finding["file"] for finding in case["expected_findings"]))


def response_files(response: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        sorted(Path(finding["file"]).name for finding in response.get("findings", []))
    )


def score_response(case: dict[str, Any], response: dict[str, Any]) -> CaseResult:
    return CaseResult(
        case_id=case["id"],
        expected_files=expected_files(case),
        actual_files=response_files(response),
        response=response,
    )


def evaluation_provenance(
    *,
    model: str,
    prompt: str,
    fixture_revision: int,
    cases: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "model": model,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "fixture_revision": fixture_revision,
        "case_ids": [case["id"] for case in cases],
    }


def _read_lines(stream: TextIO, output: queue.Queue[str]) -> None:
    for line in stream:
        output.put(line)


def _next_line(output: queue.Queue[str], deadline: float) -> str:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError
    return output.get(timeout=remaining)


def wait_for_startup(output: queue.Queue[str], timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            line = _next_line(output, deadline)
        except (queue.Empty, TimeoutError) as error:
            raise TimeoutError("watcher did not start") from error
        print(line, end="", flush=True)
        if line.startswith("Watching "):
            return


def wait_for_response(
    output: queue.Queue[str], *, timeout: float
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    json_lines: list[str] = []
    review_started = False

    while True:
        try:
            line = _next_line(output, deadline)
        except (queue.Empty, TimeoutError) as error:
            phase = "provider response" if review_started else "filesystem event"
            raise TimeoutError(f"timed out waiting for {phase}") from error

        print(line, end="", flush=True)
        if line.startswith("Reviewing ") or line.startswith("\nReviewing "):
            review_started = True
        if "llm review timed out" in line or "llm exited with status" in line:
            raise RuntimeError(line.strip())

        stripped = line.strip()
        if not json_lines and stripped.startswith("{"):
            json_lines.append(line)
        elif json_lines:
            json_lines.append(line)
        else:
            continue

        try:
            parsed = json.loads("".join(json_lines))
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict) or not isinstance(parsed.get("findings"), list):
            raise RuntimeError("provider returned JSON without a findings array")
        return parsed


def watcher_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(WATCHER_PATH),
        str(args.destination),
        "--model",
        args.model,
        "--debounce",
        str(args.debounce),
        "--review-timeout",
        str(args.review_timeout),
        "--reasoning-effort",
        args.reasoning_effort,
        "--poll",
    ]
    if args.log:
        command.append("--log")
    return command


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run live Quodet evaluations in one process tree."
    )
    parser.add_argument("case", help="case ID or 'all'")
    parser.add_argument(
        "--destination",
        type=Path,
        default=replay.DEFAULT_DESTINATION,
        help=f"watched destination (default: {replay.DEFAULT_DESTINATION})",
    )
    parser.add_argument("--model", default=watch_files.DEFAULT_MODEL)
    parser.add_argument(
        "--reasoning-effort",
        choices=("auto", "low", "medium", "high"),
        default="auto",
    )
    parser.add_argument("--debounce", type=float, default=3.0)
    parser.add_argument("--review-timeout", type=float, default=60.0)
    parser.add_argument("--inter-file-delay", type=float, default=0.25)
    parser.add_argument(
        "--settle",
        type=float,
        default=1.5,
        help="allow the polling observer to take its initial snapshot",
    )
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args(argv)
    for name in ("debounce", "review_timeout", "settle"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be greater than zero")
    if args.inter_file_delay < 0:
        parser.error("--inter-file-delay cannot be negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = replay.load_manifest()
    cases = (
        manifest["cases"]
        if args.case == "all"
        else [replay.case_by_id(manifest, args.case)]
    )
    args.destination = args.destination.expanduser().resolve()
    args.destination.mkdir(parents=True, exist_ok=True)
    provenance = evaluation_provenance(
        model=args.model,
        prompt=watch_files.DEFAULT_PROMPT,
        fixture_revision=manifest["version"],
        cases=cases,
    )
    print(f"Evaluation provenance: {json.dumps(provenance, sort_keys=True)}")

    process = subprocess.Popen(
        watcher_command(args),
        cwd=REPOSITORY_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    output: queue.Queue[str] = queue.Queue()
    reader = threading.Thread(
        target=_read_lines,
        args=(process.stdout, output),
        daemon=True,
    )
    reader.start()
    results: list[CaseResult] = []

    try:
        wait_for_startup(output)
        time.sleep(args.settle)
        for case in cases:
            print(f"\nReplaying {case['id']} ({case['difficulty']})", flush=True)
            replay.replay_case(
                case,
                destination=args.destination,
                inter_file_delay=args.inter_file_delay,
            )
            try:
                response = wait_for_response(
                    output,
                    timeout=args.debounce + args.review_timeout + 15,
                )
                result = score_response(case, response)
            except (RuntimeError, TimeoutError) as error:
                result = CaseResult(
                    case_id=case["id"],
                    expected_files=expected_files(case),
                    actual_files=(),
                    response=None,
                    error=str(error),
                )
            results.append(result)
            status = "PASS" if result.passed else "FAIL"
            print(
                f"{status} {result.case_id}: expected={list(result.expected_files)} "
                f"actual={list(result.actual_files)}"
                + (f" error={result.error}" if result.error else ""),
                flush=True,
            )
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    passed = sum(result.passed for result in results)
    print(f"\nLive evaluation: {passed}/{len(results)} cases passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
