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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence, TextIO

import watch_files
from evals.agent_changes import replay, scoring


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WATCHER_PATH = REPOSITORY_ROOT / "watch_files.py"
DEFAULT_RESULTS_DIRECTORY = REPOSITORY_ROOT / "eval-results"


@dataclass(frozen=True)
class ProviderOutcome:
    status: str
    latency_ms: int
    transcript: str
    raw_response: str | None
    parsed_response: dict[str, Any] | None
    error: str | None


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def manifest_sha256() -> str:
    return hashlib.sha256(replay.MANIFEST_PATH.read_bytes()).hexdigest()


def evaluation_configuration(
    *, model: str, reasoning_effort: str | None, prompt: str,
    fixture_revision: int, cases: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    schema_text = watch_files.REVIEW_SCHEMA_JSON
    return {
        "model": model,
        "model_options": {"reasoning_effort": reasoning_effort},
        "prompt": {
            "revision": watch_files.PROMPT_REVISION,
            "sha256": sha256_text(prompt),
            "text": prompt,
        },
        "schema": {
            "revision": watch_files.REVIEW_SCHEMA_REVISION,
            "sha256": sha256_text(schema_text),
            "value": watch_files.REVIEW_SCHEMA,
        },
        "fixture": {
            "revision": fixture_revision,
            "manifest_sha256": manifest_sha256(),
            "case_ids": [case["id"] for case in cases],
        },
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


def validate_response(value: Any) -> str | None:
    if not isinstance(value, dict) or not isinstance(value.get("findings"), list):
        return "response must contain a findings array"
    required = set(
        watch_files.REVIEW_SCHEMA["properties"]["findings"]["items"]["required"]
    )
    for index, finding in enumerate(value["findings"]):
        if not isinstance(finding, dict):
            return f"finding {index} is not an object"
        missing = required - finding.keys()
        if missing:
            return f"finding {index} is missing {sorted(missing)}"
        if not isinstance(finding["file"], str) or not finding["file"]:
            return f"finding {index} has an invalid file"
        if "line" in finding and (
            isinstance(finding["line"], bool)
            or not isinstance(finding["line"], int)
            or finding["line"] < 1
        ):
            return f"finding {index} has an invalid line"
        if finding["severity"] not in {"critical", "high", "medium", "low"}:
            return f"finding {index} has an invalid severity"
        confidence = finding["confidence"]
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or confidence < 0.95
            or confidence > 1
        ):
            return f"finding {index} has invalid confidence"
        for field in ("title", "explanation", "suggested_fix"):
            if not isinstance(finding[field], str) or not finding[field]:
                return f"finding {index} has an invalid {field}"
        if len(finding["suggested_fix"]) > 2000:
            return f"finding {index} has an overlong suggested_fix"
    return None


def wait_for_outcome(output: queue.Queue[str], *, timeout: float) -> ProviderOutcome:
    started = time.monotonic()
    deadline = started + timeout
    transcript: list[str] = []
    raw_lines: list[str] = []
    review_started = False

    while True:
        try:
            line = _next_line(output, deadline)
        except (queue.Empty, TimeoutError):
            phase = "provider response" if review_started else "filesystem event"
            return ProviderOutcome(
                "timeout", round((time.monotonic() - started) * 1000),
                "".join(transcript), "".join(raw_lines) or None, None,
                f"timed out waiting for {phase}",
            )

        print(line, end="", flush=True)
        transcript.append(line)
        if line.startswith("Reviewing ") or line.startswith("\nReviewing "):
            review_started = True
        try:
            event_wrapper = json.loads(line)
        except json.JSONDecodeError:
            event_wrapper = None
        if isinstance(event_wrapper, dict) and isinstance(
            event_wrapper.get("quodet_evaluation_event"), dict
        ):
            event = event_wrapper["quodet_evaluation_event"]
            raw_response = event.get("raw_response")
            if event.get("status") != "success":
                return ProviderOutcome(
                    str(event.get("status", "provider-error")),
                    round((time.monotonic() - started) * 1000),
                    "".join(transcript),
                    raw_response if isinstance(raw_response, str) else None,
                    None,
                    str(event.get("stderr") or "provider invocation failed"),
                )
            try:
                parsed = json.loads(raw_response)
            except (TypeError, json.JSONDecodeError) as error:
                return ProviderOutcome(
                    "schema-error", round((time.monotonic() - started) * 1000),
                    "".join(transcript),
                    raw_response if isinstance(raw_response, str) else None,
                    None, str(error),
                )
            schema_error = validate_response(parsed)
            return ProviderOutcome(
                "schema-error" if schema_error else "schema-valid",
                round((time.monotonic() - started) * 1000),
                "".join(transcript), raw_response,
                parsed if isinstance(parsed, dict) else None, schema_error,
            )
        if "llm review timed out" in line:
            return ProviderOutcome(
                "timeout", round((time.monotonic() - started) * 1000),
                "".join(transcript), "".join(raw_lines) or None, None, line.strip(),
            )
        if "llm exited with status" in line or "Could not run llm" in line:
            return ProviderOutcome(
                "provider-error", round((time.monotonic() - started) * 1000),
                "".join(transcript), "".join(raw_lines) or None, None, line.strip(),
            )

        stripped = line.strip()
        if not raw_lines and stripped.startswith("{"):
            raw_lines.append(line)
        elif raw_lines:
            raw_lines.append(line)
        else:
            continue

        raw_response = "".join(raw_lines)
        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError:
            continue
        schema_error = validate_response(parsed)
        return ProviderOutcome(
            "schema-error" if schema_error else "schema-valid",
            round((time.monotonic() - started) * 1000),
            "".join(transcript), raw_response,
            parsed if isinstance(parsed, dict) else None, schema_error,
        )


def case_outcome(case: dict[str, Any], provider: ProviderOutcome) -> dict[str, Any]:
    expected_files = sorted(finding["file"] for finding in case["expected_findings"])
    reported_files = sorted(
        Path(finding.get("file", "")).name
        for finding in (provider.parsed_response or {}).get("findings", [])
        if isinstance(finding, dict)
    )
    return {
        "case_id": case["id"],
        "evaluation_split": case["evaluation_split"],
        "failure_families": case["failure_families"],
        "scope": case["scope"],
        "expected_evidence_depth": case["expected_evidence_depth"],
        "expected_finding_ids": [finding["id"] for finding in case["expected_findings"]],
        "expected_findings": case["expected_findings"],
        "status": provider.status,
        "latency_ms": provider.latency_ms,
        "raw_response": provider.raw_response,
        "parsed_response": provider.parsed_response,
        "transcript": provider.transcript,
        "error": provider.error,
        "diagnostics": {
            "expected_files": expected_files,
            "reported_files": reported_files,
            "filename_match": reported_files == expected_files,
            "note": "Filename equality is diagnostic only, never a true-positive decision.",
        },
    }


def watcher_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable, "-u", str(WATCHER_PATH), str(args.destination),
        "--model", args.model, "--debounce", str(args.debounce),
        "--review-timeout", str(args.review_timeout),
        "--reasoning-effort", args.reasoning_effort, "--poll",
        "--evaluation-events",
    ]
    if args.log:
        command.append("--log")
    return command


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run raw live Quodet evaluations in one process tree."
    )
    parser.add_argument("case", help="case ID, evaluation split, or 'all'")
    parser.add_argument("--destination", type=Path, default=replay.DEFAULT_DESTINATION)
    parser.add_argument("--results-directory", type=Path, default=DEFAULT_RESULTS_DIRECTORY)
    parser.add_argument("--model", default=watch_files.DEFAULT_MODEL)
    parser.add_argument(
        "--reasoning-effort", choices=("auto", "low", "medium", "high"),
        default="auto",
    )
    parser.add_argument("--debounce", type=float, default=3.0)
    parser.add_argument("--review-timeout", type=float, default=60.0)
    parser.add_argument("--inter-file-delay", type=float, default=0.25)
    parser.add_argument("--settle", type=float, default=1.5)
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args(argv)
    for name in ("debounce", "review_timeout", "settle"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be greater than zero")
    if args.inter_file_delay < 0:
        parser.error("--inter-file-delay cannot be negative")
    return args


def select_cases(manifest: dict[str, Any], selector: str) -> list[dict[str, Any]]:
    if selector == "all":
        return list(manifest["cases"])
    by_split = [case for case in manifest["cases"] if case["evaluation_split"] == selector]
    if by_split or selector in scoring.EVALUATION_SPLITS:
        return by_split
    return [replay.case_by_id(manifest, selector)]


def write_raw_run(
    *, results_directory: Path, configuration: dict[str, Any],
    cases: list[dict[str, Any]], started_at: str,
) -> Path:
    run_id = datetime.fromisoformat(started_at).strftime("%Y%m%dT%H%M%S.%fZ")
    run = {
        "format_version": 1,
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "configuration": configuration,
        "cases": cases,
    }
    run["metrics"] = scoring.score_run(run, None)
    results_directory.mkdir(parents=True, exist_ok=True)
    destination = results_directory / f"{run_id}.raw.json"
    destination.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = replay.load_manifest()
    cases = select_cases(manifest, args.case)
    args.destination = args.destination.expanduser().resolve()
    args.results_directory = args.results_directory.expanduser().resolve()
    configuration = evaluation_configuration(
        model=args.model,
        reasoning_effort=watch_files.resolve_reasoning_effort(
            args.model, args.reasoning_effort
        ),
        prompt=watch_files.DEFAULT_PROMPT,
        fixture_revision=manifest["version"],
        cases=cases,
    )
    print(f"Evaluation configuration: {json.dumps(configuration, sort_keys=True)}")
    started_at = datetime.now(UTC).isoformat()
    if not cases:
        artifact = write_raw_run(
            results_directory=args.results_directory,
            configuration=configuration,
            cases=[],
            started_at=started_at,
        )
        print(f"No fixtures are exposed for the {args.case!r} boundary.")
        print(f"Raw evaluation artifact: {artifact}")
        return 0

    args.destination.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        watcher_command(args), cwd=REPOSITORY_ROOT, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    assert process.stdout is not None
    output: queue.Queue[str] = queue.Queue()
    threading.Thread(target=_read_lines, args=(process.stdout, output), daemon=True).start()
    outcomes: list[dict[str, Any]] = []
    try:
        wait_for_startup(output)
        time.sleep(args.settle)
        for case in cases:
            print(f"\nReplaying {case['id']} ({case['difficulty']})", flush=True)
            replay.replay_case(
                case, destination=args.destination,
                inter_file_delay=args.inter_file_delay,
            )
            provider = wait_for_outcome(
                output, timeout=args.debounce + args.review_timeout + 15
            )
            outcome = case_outcome(case, provider)
            outcomes.append(outcome)
            print(
                f"RAW {case['id']}: status={provider.status}; "
                f"filename_match={outcome['diagnostics']['filename_match']}; "
                "semantic adjudication=pending", flush=True,
            )
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    artifact = write_raw_run(
        results_directory=args.results_directory, configuration=configuration,
        cases=outcomes, started_at=started_at,
    )
    print(f"\nRaw evaluation artifact: {artifact}")
    print("Semantic adjudication is required before TP/FP/FN are reported.")
    return 0 if all(outcome["status"] == "schema-valid" for outcome in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
