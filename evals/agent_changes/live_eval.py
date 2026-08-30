"""Run Quodet and coding-agent change replays in one process tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import queue
import re
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
from evals.agent_changes import artifacts, benchmark, replay, scoring
from model_runner import (
    ModelDocument,
    ModelRunConfig,
    ModelRunRequest,
    load_model_run_config,
    model_run_config_sha256,
    preflight_model_run,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OLLAMA_REGISTRY_PREFIX = "Ollama:"
OLLAMA_LOCAL_BLOB = re.compile(
    r"^FROM\s+(?:.*/)?sha256-([0-9a-f]{64})\s*$", re.MULTILINE,
)
WATCHER_PATH = REPOSITORY_ROOT / "watch_files.py"
DEFAULT_RESULTS_DIRECTORY = REPOSITORY_ROOT / "eval-results"
MAX_RUNTIME_ATTESTATION_SECONDS = 10.0


@dataclass(frozen=True)
class ProviderOutcome:
    status: str
    latency_ms: int
    transcript: str
    raw_response: str | None
    parsed_response: dict[str, Any] | None
    error: str | None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    maximum_cost_usd: float | None = None
    resource_usage: dict[str, Any] | None = None
    effective_config: dict[str, Any] | None = None
    model_latency_ms: int | None = None
    model_attempt_count: int | None = None
    runtime_attestation: dict[str, Any] | None = None


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def manifest_sha256() -> str:
    return hashlib.sha256(replay.MANIFEST_PATH.read_bytes()).hexdigest()


def evaluation_configuration(
    *, model: str, reasoning_effort: str | None, prompt: str,
    fixture_revision: int, cases: Sequence[dict[str, Any]],
    debounce_seconds: float, inter_file_delay_seconds: float,
    model_run_config: ModelRunConfig | None = None,
    benchmark_plan: dict[str, Any] | None = None,
    maximum_authorized_cost_usd: float | None = None,
    runtime_attestation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    schema_text = watch_files.REVIEW_SCHEMA_JSON
    configuration = {
        "model": model,
        "model_options": {"reasoning_effort": reasoning_effort},
        "batching": {
            "debounce_seconds": debounce_seconds,
            "inter_file_delay_seconds": inter_file_delay_seconds,
        },
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
            "fixture_tree_sha256": replay.fixture_tree_sha256(),
            "provider_payload_sha256": benchmark.provider_fixture_payload_sha256(),
            "case_ids": [case["id"] for case in cases],
        },
    }
    if model_run_config is not None:
        configuration["benchmark"] = {
            "experiment_id": (
                benchmark_plan or {}
            ).get("experiment_id"),
            "plan_sha256": (
                benchmark.plan_sha256(benchmark_plan)
                if benchmark_plan is not None else None
            ),
            "candidate_id": model_run_config.candidate_id,
            "model_run_config": model_run_config.to_dict(),
            "maximum_authorized_cost_usd": maximum_authorized_cost_usd,
            "runtime_attestation": runtime_attestation,
        }
    return configuration


def _model_result_fields(event: dict[str, Any]) -> dict[str, Any]:
    result = event.get("model_run_result")
    if not isinstance(result, dict):
        return {"runtime_attestation": event.get("runtime_attestation")}
    return {
        "input_tokens": result.get("input_tokens"),
        "output_tokens": result.get("output_tokens"),
        "cost_usd": result.get("cost_usd"),
        "maximum_cost_usd": result.get("maximum_cost_usd"),
        "resource_usage": result.get("resource_usage"),
        "effective_config": result.get("effective_config"),
        "model_latency_ms": result.get("latency_ms"),
        "model_attempt_count": result.get("attempt_count"),
        "runtime_attestation": event.get("runtime_attestation"),
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
    schema = watch_files.REVIEW_SCHEMA
    if not isinstance(value, dict) or not isinstance(value.get("findings"), list):
        return "response must contain a findings array"
    item_schema = schema["properties"]["findings"]["items"]
    properties = item_schema["properties"]
    required = set(item_schema["required"])
    for index, finding in enumerate(value["findings"]):
        if not isinstance(finding, dict):
            return f"finding {index} is not an object"
        missing = required - finding.keys()
        if missing:
            return f"finding {index} is missing {sorted(missing)}"
        if not isinstance(finding["file"], str):
            return f"finding {index} has an invalid file"
        if "line" in finding and (
            isinstance(finding["line"], bool)
            or not isinstance(finding["line"], int)
            or finding["line"] < properties["line"]["minimum"]
        ):
            return f"finding {index} has an invalid line"
        if finding["severity"] not in properties["severity"]["enum"]:
            return f"finding {index} has an invalid severity"
        confidence = finding["confidence"]
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence)
            or confidence < properties["confidence"]["minimum"]
            or confidence > properties["confidence"]["maximum"]
        ):
            return f"finding {index} has invalid confidence"
        for field in ("title", "explanation"):
            if not isinstance(finding[field], str):
                return f"finding {index} has an invalid {field}"
        suggested_fix = finding["suggested_fix"]
        fix_schema = properties["suggested_fix"]
        if (
            not isinstance(suggested_fix, str)
            or len(suggested_fix) < fix_schema["minLength"]
            or len(suggested_fix) > fix_schema["maxLength"]
        ):
            return f"finding {index} has an invalid suggested_fix"
    return None


def parse_provider_response(raw_response: object) -> Any:
    """Parse strict JSON, rejecting JavaScript constants accepted by json.loads."""
    if not isinstance(raw_response, str):
        raise TypeError("provider response is not text")

    def reject_constant(value: str) -> None:
        raise ValueError(f"invalid JSON constant: {value}")

    return json.loads(raw_response, parse_constant=reject_constant)


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
            result_fields = _model_result_fields(event)
            if event.get("status") != "success":
                return ProviderOutcome(
                    str(event.get("status", "provider-error")),
                    round((time.monotonic() - started) * 1000),
                    "".join(transcript),
                    raw_response if isinstance(raw_response, str) else None,
                    None,
                    str(event.get("stderr") or "provider invocation failed"),
                    **result_fields,
                )
            try:
                parsed = parse_provider_response(raw_response)
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                return ProviderOutcome(
                    "schema-error", round((time.monotonic() - started) * 1000),
                    "".join(transcript),
                    raw_response if isinstance(raw_response, str) else None,
                    None, str(error), **result_fields,
                )
            schema_error = validate_response(parsed)
            return ProviderOutcome(
                "schema-error" if schema_error else "schema-valid",
                round((time.monotonic() - started) * 1000),
                "".join(transcript), raw_response,
                parsed if isinstance(parsed, dict) else None, schema_error,
                **result_fields,
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
            parsed = parse_provider_response(raw_response)
        except json.JSONDecodeError:
            continue
        except (TypeError, ValueError) as error:
            return ProviderOutcome(
                "schema-error", round((time.monotonic() - started) * 1000),
                "".join(transcript), raw_response, None, str(error),
            )
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
        "input_tokens": provider.input_tokens,
        "output_tokens": provider.output_tokens,
        "cost_usd": provider.cost_usd,
        "maximum_cost_usd": provider.maximum_cost_usd,
        "resource_usage": provider.resource_usage,
        "effective_model_config": provider.effective_config,
        "model_latency_ms": provider.model_latency_ms,
        "model_attempt_count": provider.model_attempt_count,
        "runtime_attestation": provider.runtime_attestation,
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
    if getattr(args, "model_run_config", None) is not None:
        command.extend(["--model-run-config", str(args.model_run_config)])
        command.extend([
            "--model-run-config-sha256", args.model_run_config_sha256,
            "--benchmark-plan", str(args.benchmark_plan),
            "--benchmark-plan-sha256", args.benchmark_plan_sha256,
        ])
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
    parser.add_argument("--model-run-config", type=Path)
    parser.add_argument(
        "--benchmark-plan", type=Path, default=benchmark.DEFAULT_PLAN,
    )
    args = parser.parse_args(argv)
    for name in ("debounce", "review_timeout", "settle"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be greater than zero")
    if args.inter_file_delay < 0:
        parser.error("--inter-file-delay cannot be negative")
    return args


def benchmark_cost_preflight(
    config: ModelRunConfig,
    cases: Sequence[dict[str, Any]],
) -> float | None:
    maximum_costs: list[float] = []
    for case in cases:
        request = ModelRunRequest(
            documents=tuple(
                ModelDocument(replay.CASES_ROOT / case["id"] / filename)
                for filename in case["files"]
            ),
            prompt=watch_files.DEFAULT_PROMPT,
            schema_json=watch_files.REVIEW_SCHEMA_JSON,
            cwd=REPOSITORY_ROOT,
        )
        result = preflight_model_run(config, request)
        if not result.allowed:
            raise ValueError(f"{case['id']} preflight failed: {result.reason}")
        if result.maximum_cost_usd is not None:
            maximum_costs.append(result.maximum_cost_usd)
    if not maximum_costs:
        return None
    total = sum(maximum_costs)
    assert config.max_cost_usd is not None
    if total > config.max_cost_usd:
        raise ValueError(
            f"benchmark maximum cost ${total:.6f} exceeds the configured "
            f"experiment cap ${config.max_cost_usd:.6f}"
        )
    return total


def attest_runtime(config: ModelRunConfig) -> dict[str, Any]:
    deadline = time.monotonic() + min(
        MAX_RUNTIME_ATTESTATION_SECONDS, config.timeout_seconds,
    )

    def run_attestation_command(
        command: Sequence[str],
    ) -> subprocess.CompletedProcess[str]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError("runtime attestation timed out")
        try:
            return subprocess.run(
                command, check=False, capture_output=True, text=True,
                timeout=remaining,
            )
        except subprocess.TimeoutExpired as error:
            raise ValueError("runtime attestation timed out") from error

    version_result = run_attestation_command(["llm", "--version"])
    plugins_result = run_attestation_command(["llm", "plugins"])
    models_result = run_attestation_command(["llm", "models", "list"])
    if any(
        result.returncode != 0
        for result in (version_result, plugins_result, models_result)
    ):
        raise ValueError("could not attest installed llm runtime and model registry")
    cli_version = version_result.stdout.strip().rsplit(" ", 1)[-1]
    try:
        plugins = json.loads(plugins_result.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("llm plugins did not return valid JSON") from error
    if config.runtime == "llm":
        runtime_version = cli_version
        runtime_entry: dict[str, Any] = {"name": "llm", "version": cli_version}
    else:
        matching = [
            plugin for plugin in plugins
            if isinstance(plugin, dict) and plugin.get("name") == config.runtime
        ]
        if len(matching) != 1:
            raise ValueError(
                f"configured runtime {config.runtime!r} is not installed exactly once"
            )
        runtime_entry = matching[0]
        runtime_version = str(runtime_entry.get("version"))
    if runtime_version != config.runtime_version:
        raise ValueError(
            f"configured runtime version {config.runtime_version!r} differs from "
            f"installed {runtime_version!r}"
        )
    model_entry = watch_files._listed_model_entries(models_result.stdout).get(
        config.model
    )
    if model_entry is None:
        raise ValueError(f"configured model alias {config.model!r} is not installed")
    local_model_attestation = None
    if config.locality == "local":
        if config.runtime != "llm-ollama" or not model_entry.startswith(
            OLLAMA_REGISTRY_PREFIX
        ):
            raise ValueError(
                "local execution requires llm-ollama and an Ollama registry entry"
            )
        runtime_model_id = str(config.hardware.get("runtime_model_id", ""))
        registered_model_id = model_entry[len(OLLAMA_REGISTRY_PREFIX):].strip()
        registered_model_id = registered_model_id.split(" (aliases:", 1)[0].strip()
        if (
            not runtime_model_id
            or runtime_model_id.endswith(":cloud")
            or registered_model_id != runtime_model_id
        ):
            raise ValueError(
                "local llm alias does not resolve to the configured non-cloud "
                "runtime_model_id"
            )
        model_result = run_attestation_command(
            ["ollama", "show", "--modelfile", runtime_model_id],
        )
        if model_result.returncode != 0:
            raise ValueError("could not attest the configured Ollama model")
        digest_match = OLLAMA_LOCAL_BLOB.search(model_result.stdout)
        if digest_match is None:
            raise ValueError(
                "configured Ollama model does not resolve to an immutable local blob"
            )
        actual_digest = digest_match.group(1)
        expected_digest = config.hardware.get("runtime_artifact_sha256")
        if actual_digest != expected_digest:
            raise ValueError(
                "configured Ollama model blob differs from runtime_artifact_sha256"
            )
        local_model_attestation = {
            "runtime_model_id": runtime_model_id,
            "runtime_artifact_sha256": actual_digest,
        }
    attestation = {
        "llm_cli_version": cli_version,
        "runtime": runtime_entry,
        "model_registry_entry": model_entry,
    }
    if local_model_attestation is not None:
        attestation["local_model"] = local_model_attestation
    return attestation


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
    destination = results_directory / f"{run_id}.raw.json"
    artifacts.write_private_json(destination, run)
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = replay.load_manifest()
    cases = select_cases(manifest, args.case)
    args.destination = args.destination.expanduser().resolve()
    args.results_directory = args.results_directory.expanduser().resolve()
    run_config_path = getattr(args, "model_run_config", None)
    model_run_config = (
        load_model_run_config(run_config_path.expanduser().resolve())
        if run_config_path is not None else None
    )
    benchmark_plan = None
    maximum_authorized_cost = None
    runtime_attestation = None
    if model_run_config is not None:
        args.model_run_config = run_config_path.expanduser().resolve()
        args.benchmark_plan = getattr(
            args, "benchmark_plan", benchmark.DEFAULT_PLAN,
        ).expanduser().resolve()
        benchmark_plan = benchmark.load_plan(args.benchmark_plan)
        benchmark.validate_model_run_config_against_plan(
            benchmark_plan, model_run_config,
        )
        args.model_run_config_sha256 = model_run_config_sha256(model_run_config)
        args.benchmark_plan_sha256 = benchmark.plan_sha256(benchmark_plan)
        watch_files.validate_model_run_config_privacy(model_run_config)
        maximum_authorized_cost = benchmark_cost_preflight(model_run_config, cases)
        runtime_attestation = attest_runtime(model_run_config)
        args.model = model_run_config.model
        args.review_timeout = model_run_config.timeout_seconds
        args.reasoning_effort = "auto"
    configuration = evaluation_configuration(
        model=args.model,
        reasoning_effort=watch_files.resolve_reasoning_effort(
            args.model, args.reasoning_effort
        ),
        prompt=watch_files.DEFAULT_PROMPT,
        fixture_revision=manifest["version"],
        cases=cases,
        debounce_seconds=args.debounce,
        inter_file_delay_seconds=args.inter_file_delay,
        model_run_config=model_run_config,
        benchmark_plan=benchmark_plan,
        maximum_authorized_cost_usd=maximum_authorized_cost,
        runtime_attestation=runtime_attestation,
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

    outcomes: list[dict[str, Any]] = []
    process: subprocess.Popen[str] | None = None
    interrupted = False
    artifact: Path | None = None
    try:
        args.destination.mkdir(parents=True, exist_ok=True)
        process = subprocess.Popen(
            watcher_command(args), cwd=REPOSITORY_ROOT, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert process.stdout is not None
        output: queue.Queue[str] = queue.Queue()
        threading.Thread(
            target=_read_lines, args=(process.stdout, output), daemon=True
        ).start()
        wait_for_startup(output)
        time.sleep(args.settle)
        for case in cases:
            print(f"\nReplaying {case['id']} ({case['difficulty']})", flush=True)
            outcome_index = len(outcomes)
            outcomes.append(case_outcome(
                case,
                ProviderOutcome(
                    status="interrupted", latency_ms=0, transcript="",
                    raw_response=None, parsed_response=None,
                    error="attempt started but did not produce a complete outcome",
                ),
            ))
            try:
                replay.replay_case(
                    case, destination=args.destination,
                    inter_file_delay=args.inter_file_delay,
                )
                provider = wait_for_outcome(
                    output, timeout=args.debounce + args.review_timeout + 15
                )
            except BaseException as error:
                if not isinstance(error, KeyboardInterrupt):
                    outcomes[outcome_index] = case_outcome(
                        case,
                        ProviderOutcome(
                            status="harness-error", latency_ms=0, transcript="",
                            raw_response=None, parsed_response=None,
                            error=f"{type(error).__name__}: {error}",
                        ),
                    )
                raise
            outcome = case_outcome(case, provider)
            outcomes[outcome_index] = outcome
            print(
                f"RAW {case['id']}: status={provider.status}; "
                f"filename_match={outcome['diagnostics']['filename_match']}; "
                "semantic adjudication=pending", flush=True,
            )
    except BaseException:
        interrupted = True
        raise
    finally:
        if process is not None and process.poll() is None:
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
        if interrupted:
            print(f"\nPartial raw evaluation artifact: {artifact}", file=sys.stderr)

    assert artifact is not None
    print(f"\nRaw evaluation artifact: {artifact}")
    print("Semantic adjudication is required before TP/FP/FN are reported.")
    return 0 if all(outcome["status"] == "schema-valid" for outcome in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
