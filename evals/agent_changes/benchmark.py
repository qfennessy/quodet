"""Prepare exact model configs and aggregate the frozen issue #5 comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import watch_files
from evals.agent_changes import artifacts, replay, scoring
from model_runner import ModelRunConfig, Pricing


ROOT = Path(__file__).resolve().parent
DEFAULT_PLAN = ROOT / "model_benchmark_plan.json"
DEFAULT_SCORECARD = ROOT / "model_benchmark_scorecard.json"
ORDINARY_PLAN_SCOPE = "ordinary-benchmark"
CHALLENGE_PLAN_SCOPE = "challenge-qualification"
CHALLENGE_QUALIFICATION_ATTEMPTS = 3


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan_sha256(plan: Mapping[str, Any]) -> str:
    payload = json.dumps(
        plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def provider_fixture_payload_sha256(
    manifest: Mapping[str, Any] | None = None,
    *,
    cases: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """Hash the exact deterministic document text staged for the provider."""
    if manifest is not None and cases is not None:
        raise ValueError("pass either manifest or cases, not both")
    selected_cases = cases if cases is not None else (
        manifest or replay.load_manifest()
    )["cases"]
    digest = hashlib.sha256()
    for case in selected_cases:
        source_root = Path(
            case.get("_source_root", replay.CASES_ROOT / case["id"])
        )
        # collect_attachments sorts a debounced batch by path before staging it.
        for filename in sorted(case["files"]):
            relative_path = replay.replay_relative_directory(case) / filename
            contents = (source_root / filename).read_text(encoding="utf-8")
            sanitized, _ = watch_files.redact_sensitive_values(contents)
            sanitized_path, path_redactions = watch_files.redact_sensitive_path(
                relative_path
            )
            # sanitize_attachments excludes a file whose path itself is sensitive.
            # Such a file contributes no provider document bytes.
            if path_redactions:
                continue
            provider_contents = (
                f"Original relative path: {sanitized_path}\n\n{sanitized}"
            ).encode()
            digest.update(sanitized_path.encode())
            digest.update(b"\0")
            digest.update(len(provider_contents).to_bytes(8, "big"))
            digest.update(provider_contents)
    return digest.hexdigest()


def _matches_frozen_model_options(
    model_options: Mapping[str, Any], expected_temperature: Any,
    expected_reasoning_effort: Any = None,
) -> bool:
    temperature = model_options.get("temperature")
    expected_keys: set[str] = set()
    temperature_matches = expected_temperature is None
    if expected_temperature is not None:
        expected_keys.add("temperature")
        temperature_matches = (
            isinstance(expected_temperature, (int, float))
            and not isinstance(expected_temperature, bool)
            and isinstance(temperature, (int, float))
            and not isinstance(temperature, bool)
            and temperature == expected_temperature
        )
    if expected_reasoning_effort is not None:
        expected_keys.add("reasoning_effort")
    return (
        bool(expected_keys)
        and set(model_options) == expected_keys
        and temperature_matches
        and (
            expected_reasoning_effort is None
            or (
                expected_reasoning_effort in {"low", "medium", "high"}
                and model_options.get("reasoning_effort")
                == expected_reasoning_effort
            )
        )
    )


def load_plan(path: Path = DEFAULT_PLAN) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("benchmark plan must be a JSON object")
    validate_plan(value)
    return value


def validate_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("format_version") != 1:
        raise ValueError("unsupported benchmark plan format")
    fixture = plan.get("fixture")
    contract = plan.get("review_contract")
    candidates = plan.get("candidates")
    if not isinstance(fixture, Mapping) or not isinstance(contract, Mapping):
        raise ValueError("benchmark plan requires fixture and review_contract objects")
    if not isinstance(candidates, Mapping) or set(candidates) != set(
        plan.get("candidate_order", [])
    ):
        raise ValueError("candidate_order must name every candidate exactly once")
    scope = plan.get("evaluation_scope")
    challenge_fixture = plan.get("challenge_fixture")
    attempts = contract.get("attempts_per_case")
    if scope == ORDINARY_PLAN_SCOPE:
        if challenge_fixture is not None:
            raise ValueError("ordinary benchmark plan cannot bind challenge fixtures")
        if attempts != 1:
            raise ValueError("ordinary benchmark must freeze exactly one attempt")
    elif scope == CHALLENGE_PLAN_SCOPE:
        if attempts != CHALLENGE_QUALIFICATION_ATTEMPTS:
            raise ValueError(
                "challenge qualification plan must freeze exactly three attempts"
            )
        if not isinstance(challenge_fixture, Mapping) or set(challenge_fixture) != {
            "revision", "manifest_sha256", "content_sha256",
            "provider_payload_sha256", "case_ids",
        }:
            raise ValueError(
                "challenge qualification plan requires an exact challenge_fixture binding"
            )
        case_ids = challenge_fixture.get("case_ids")
        if (
            not isinstance(challenge_fixture.get("revision"), int)
            or isinstance(challenge_fixture.get("revision"), bool)
            or challenge_fixture["revision"] < 1
            or not isinstance(case_ids, list)
            or not case_ids
            or any(not isinstance(case_id, str) or not case_id for case_id in case_ids)
            or len(case_ids) != len(set(case_ids))
        ):
            raise ValueError("challenge fixture revision or ordered case IDs are invalid")
        for key in (
            "manifest_sha256", "content_sha256", "provider_payload_sha256",
        ):
            digest = challenge_fixture.get(key)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"challenge fixture {key} is not a lowercase SHA-256")
    else:
        raise ValueError(
            "benchmark plan evaluation_scope must be ordinary-benchmark or "
            "challenge-qualification"
        )
    if fixture.get("revision") != replay.load_manifest().get("version"):
        raise ValueError("benchmark fixture revision does not match the corpus")
    if fixture.get("manifest_sha256") != _sha256(replay.MANIFEST_PATH):
        raise ValueError("benchmark fixture manifest hash has drifted")
    if fixture.get("fixture_tree_sha256") != replay.fixture_tree_sha256():
        raise ValueError("benchmark fixture file bytes have drifted")
    if fixture.get("provider_payload_sha256") != provider_fixture_payload_sha256():
        raise ValueError("benchmark provider fixture payload has drifted")
    expected_contract = {
        "prompt_revision": watch_files.PROMPT_REVISION,
        "prompt_sha256": hashlib.sha256(watch_files.DEFAULT_PROMPT.encode()).hexdigest(),
        "schema_revision": watch_files.REVIEW_SCHEMA_REVISION,
        "schema_sha256": hashlib.sha256(
            watch_files.REVIEW_SCHEMA_JSON.encode()
        ).hexdigest(),
    }
    for key, expected in expected_contract.items():
        if contract.get(key) != expected:
            raise ValueError(f"benchmark review contract {key} has drifted")
    if contract.get("repair_or_fallback"):
        raise ValueError("benchmark must freeze no repair or fallback")
    privacy = plan.get("privacy", {})
    if not privacy.get("hosted_requires_external_upload_consent") or not privacy.get(
        "local_requires_immutable_runtime_artifact_attestation"
    ):
        raise ValueError("benchmark privacy boundary is not fail-closed")
    for candidate_id, candidate in candidates.items():
        if not isinstance(candidate, Mapping):
            raise ValueError(f"candidate {candidate_id} must be an object")
        for key in (
            "model_artifact", "model_revision", "license", "published_quantization",
            "published_context_limit", "metadata_source", "required_locality",
        ):
            if not candidate.get(key):
                raise ValueError(f"candidate {candidate_id} is missing {key}")
        revision = candidate["model_revision"]
        if not isinstance(revision, str) or len(revision) != 40:
            raise ValueError(f"candidate {candidate_id} revision is not an exact SHA")
        if candidate["required_locality"] == "local":
            runtime_artifact = candidate.get("runtime_artifact")
            if not isinstance(runtime_artifact, Mapping) or runtime_artifact.get(
                "status"
            ) not in {"unregistered", "approved"}:
                raise ValueError(
                    f"local candidate {candidate_id} requires runtime_artifact status"
                )
            if runtime_artifact.get("runtime") != "llm-ollama":
                raise ValueError(
                    f"local candidate {candidate_id} requires llm-ollama binding"
                )
            if runtime_artifact["status"] == "unregistered" and (
                runtime_artifact.get("runtime_version") is not None
                or runtime_artifact.get("model") is not None
                or runtime_artifact.get("runtime_model_id") is not None
                or runtime_artifact.get("runtime_artifact_sha256") is not None
                or runtime_artifact.get("quantization") is not None
                or runtime_artifact.get("max_output_tokens_option") is not None
            ):
                raise ValueError(
                    f"local candidate {candidate_id} has a partial runtime binding"
                )
            if runtime_artifact["status"] == "approved":
                runtime_model_id = runtime_artifact.get("runtime_model_id")
                runtime_digest = runtime_artifact.get("runtime_artifact_sha256")
                source_binding = runtime_artifact.get("source_binding")
                for key in (
                    "runtime_version", "model", "quantization",
                    "max_output_tokens_option",
                ):
                    if not isinstance(runtime_artifact.get(key), str) or not (
                        runtime_artifact[key]
                    ):
                        raise ValueError(
                            f"local candidate {candidate_id} approved {key} is invalid"
                        )
                if not isinstance(runtime_model_id, str) or not runtime_model_id:
                    raise ValueError(
                        f"local candidate {candidate_id} approved model ID is invalid"
                    )
                if (
                    not isinstance(runtime_digest, str)
                    or len(runtime_digest) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in runtime_digest
                    )
                ):
                    raise ValueError(
                        f"local candidate {candidate_id} approved digest is invalid"
                    )
                if not isinstance(source_binding, Mapping) or (
                    source_binding.get("model_artifact")
                    != candidate["model_artifact"]
                    or source_binding.get("model_revision")
                    != candidate["model_revision"]
                    or not source_binding.get("conversion_tool")
                    or not source_binding.get("conversion_tool_version")
                    or not isinstance(source_binding.get("recipe_sha256"), str)
                    or len(source_binding["recipe_sha256"]) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in source_binding["recipe_sha256"]
                    )
                ):
                    raise ValueError(
                        f"local candidate {candidate_id} source binding is invalid"
                    )
        else:
            runtime_artifact = candidate.get("runtime_artifact")
            if not isinstance(runtime_artifact, Mapping) or runtime_artifact.get(
                "status"
            ) not in {"unregistered", "approved"}:
                raise ValueError(
                    f"hosted candidate {candidate_id} requires runtime_artifact status"
                )
            hosted_identity_keys = (
                "provider", "runtime", "runtime_version", "model",
                "provider_model_revision", "quantization",
                "max_output_tokens_option",
            )
            if runtime_artifact["status"] == "unregistered" and any(
                runtime_artifact.get(key) is not None for key in hosted_identity_keys
            ):
                raise ValueError(
                    f"hosted candidate {candidate_id} has a partial runtime binding"
                )
            if runtime_artifact["status"] == "approved":
                if any(
                    not isinstance(runtime_artifact.get(key), str)
                    or not runtime_artifact[key]
                    for key in hosted_identity_keys
                ):
                    raise ValueError(
                        f"hosted candidate {candidate_id} approved identity is invalid"
                    )
                source_binding = runtime_artifact.get("source_binding")
                if not isinstance(source_binding, Mapping) or (
                    source_binding.get("model_artifact")
                    != candidate["model_artifact"]
                    or source_binding.get("model_revision")
                    != candidate["model_revision"]
                    or not str(source_binding.get("evidence_url", "")).startswith(
                        "https://"
                    )
                    or not source_binding.get("evidence_as_of")
                ):
                    raise ValueError(
                        f"hosted candidate {candidate_id} source binding is invalid"
                    )
                try:
                    date.fromisoformat(str(source_binding["evidence_as_of"]))
                except ValueError as error:
                    raise ValueError(
                        f"hosted candidate {candidate_id} evidence date is invalid"
                    ) from error


def prepare_run_config(
    plan: Mapping[str, Any],
    *,
    candidate_id: str,
    model: str,
    provider: str,
    runtime: str,
    runtime_version: str,
    quantization: str,
    model_options: Mapping[str, str | int | float | bool],
    timeout_seconds: float,
    max_output_tokens: int,
    max_output_tokens_option: str,
    max_output_bytes: int,
    pricing: Pricing,
    max_cost_usd: float | None,
    external_upload_consent: bool,
    hardware: Mapping[str, str | int | float | bool],
) -> ModelRunConfig:
    validate_plan(plan)
    try:
        candidate = plan["candidates"][candidate_id]
    except KeyError as error:
        raise ValueError(f"unknown benchmark candidate: {candidate_id}") from error
    locality = candidate["required_locality"]
    runtime_artifact = candidate["runtime_artifact"]
    if runtime_artifact["status"] != "approved":
        raise ValueError(
            f"candidate {candidate_id} has no approved runtime artifact; "
            "register its exact execution identity in the frozen plan"
        )
    execution = plan["execution_contract"]
    if timeout_seconds != execution["timeout_seconds"]:
        raise ValueError("candidate timeout differs from frozen execution contract")
    if max_output_tokens != execution["max_output_tokens"]:
        raise ValueError("candidate output-token cap differs from frozen contract")
    if max_output_bytes != execution["max_output_bytes"]:
        raise ValueError("candidate output-byte cap differs from frozen contract")
    if not _matches_frozen_model_options(
        model_options,
        execution["temperature"],
        execution.get("reasoning_effort"),
    ):
        raise ValueError("candidate model options differ from frozen execution contract")
    if locality == "hosted" and not external_upload_consent:
        raise ValueError("hosted candidate requires explicit external-upload consent")
    if locality == "hosted" and max_cost_usd is None:
        raise ValueError("hosted candidate requires --max-cost-usd")
    if locality == "hosted" and (
        pricing.input_usd_per_million_tokens is None
        or pricing.output_usd_per_million_tokens is None
    ):
        raise ValueError("hosted candidate requires explicit input and output pricing")
    if locality == "hosted" and not pricing.source.startswith("https://"):
        raise ValueError("hosted candidate requires an HTTPS pricing source")
    if locality == "hosted":
        try:
            date.fromisoformat(pricing.as_of)
        except ValueError as error:
            raise ValueError(
                "hosted candidate requires an ISO pricing as-of date"
            ) from error
    if locality == "local" and external_upload_consent:
        raise ValueError("local candidate config cannot request external-upload consent")
    if locality == "local" and provider != "local":
        raise ValueError("local candidate requires --provider local")
    if locality == "hosted" and provider == "local":
        raise ValueError("hosted candidate cannot use --provider local")
    if locality == "local":
        missing_measurements = {
            "amortized_hourly_cost_usd", "model_load_ms", "peak_memory_bytes",
            "runtime_model_id", "runtime_artifact_sha256",
        } - set(hardware)
        if missing_measurements:
            raise ValueError(
                "local candidate hardware metadata is missing "
                f"{sorted(missing_measurements)}"
            )
        if runtime != "llm-ollama":
            raise ValueError(
                "local benchmark candidates currently require runtime llm-ollama "
                "for immutable local-blob attestation"
            )
        runtime_model_id = hardware["runtime_model_id"]
        runtime_artifact_sha256 = hardware["runtime_artifact_sha256"]
        if (
            not isinstance(runtime_model_id, str)
            or not runtime_model_id.strip()
            or runtime_model_id.endswith(":cloud")
        ):
            raise ValueError("local runtime_model_id must name a non-cloud model")
        if (
            not isinstance(runtime_artifact_sha256, str)
            or len(runtime_artifact_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in runtime_artifact_sha256
            )
        ):
            raise ValueError(
                "local runtime_artifact_sha256 must be a lowercase SHA-256"
            )
        if (
            runtime != runtime_artifact["runtime"]
            or runtime_version != runtime_artifact["runtime_version"]
            or model != runtime_artifact["model"]
            or runtime_model_id != runtime_artifact["runtime_model_id"]
            or runtime_artifact_sha256
            != runtime_artifact["runtime_artifact_sha256"]
            or quantization != runtime_artifact["quantization"]
            or max_output_tokens_option
            != runtime_artifact["max_output_tokens_option"]
        ):
            raise ValueError(
                "local runtime identity differs from the candidate binding "
                "approved in the frozen plan"
            )
    else:
        configured_hosted_identity = {
            "provider": provider,
            "runtime": runtime,
            "runtime_version": runtime_version,
            "model": model,
            "quantization": quantization,
            "provider_model_revision": hardware.get("provider_model_revision"),
            "max_output_tokens_option": max_output_tokens_option,
        }
        if any(
            configured_hosted_identity[key] != runtime_artifact[key]
            for key in configured_hosted_identity
        ):
            raise ValueError(
                "hosted execution identity differs from the candidate binding "
                "approved in the frozen plan"
            )
    return ModelRunConfig(
        candidate_id=candidate_id,
        model=model,
        model_artifact=candidate["model_artifact"],
        model_revision=candidate["model_revision"],
        provider=provider,
        runtime=runtime,
        runtime_version=runtime_version,
        locality=locality,
        quantization=quantization,
        model_options=model_options,
        context_limit=int(candidate["published_context_limit"]),
        timeout_seconds=timeout_seconds,
        max_output_bytes=max_output_bytes,
        max_output_tokens=max_output_tokens,
        max_output_tokens_option=max_output_tokens_option,
        pricing=pricing,
        max_cost_usd=max_cost_usd,
        external_upload_consent=external_upload_consent,
        hardware=hardware,
    )


def validate_model_run_config_against_plan(
    plan: Mapping[str, Any], config: ModelRunConfig,
) -> str:
    """Validate the complete frozen binding before any model invocation."""
    expected = prepare_run_config(
        plan,
        candidate_id=config.candidate_id,
        model=config.model,
        provider=config.provider,
        runtime=config.runtime,
        runtime_version=config.runtime_version,
        quantization=config.quantization,
        model_options=config.model_options,
        timeout_seconds=config.timeout_seconds,
        max_output_tokens=config.max_output_tokens,
        max_output_tokens_option=config.max_output_tokens_option,
        max_output_bytes=config.max_output_bytes,
        pricing=config.pricing,
        max_cost_usd=config.max_cost_usd,
        external_upload_consent=config.external_upload_consent,
        hardware=config.hardware,
    )
    actual_values = config.to_dict()
    expected_values = expected.to_dict()
    differing = sorted(
        key for key in set(actual_values) | set(expected_values)
        if actual_values.get(key) != expected_values.get(key)
    )
    if differing:
        raise ValueError(
            "model run config differs from the complete frozen benchmark binding "
            f"in {differing}"
        )
    return config.candidate_id


def _validate_runtime_attestation(
    model_config: Mapping[str, Any],
    attestation: object,
    *,
    label: str,
    expected_cli_version: str | None = None,
) -> str:
    """Bind a recorded runtime observation to the exact effective config."""
    if not isinstance(attestation, Mapping):
        raise ValueError(f"{label} is missing live runtime attestation")
    runtime_entry = attestation.get("runtime")
    if not isinstance(runtime_entry, Mapping) or (
        runtime_entry.get("name") != model_config.get("runtime")
        or str(runtime_entry.get("version")) != model_config.get("runtime_version")
    ):
        raise ValueError(f"{label} runtime attestation differs from exact config")
    llm_cli_version = attestation.get("llm_cli_version")
    if not isinstance(llm_cli_version, str) or not llm_cli_version.strip():
        raise ValueError(f"{label} is missing its exact llm CLI version")
    if (
        expected_cli_version is not None
        and llm_cli_version != expected_cli_version
    ):
        raise ValueError(
            f"{label} llm CLI version differs from the startup attestation"
        )
    registry_entry = attestation.get("model_registry_entry")
    if not isinstance(registry_entry, str) or (
        model_config.get("model") not in watch_files._listed_model_entries(
            registry_entry
        )
    ):
        raise ValueError(
            f"{label} model registry entry differs from exact config"
        )
    if model_config.get("locality") == "local":
        local_model = attestation.get("local_model")
        hardware = model_config.get("hardware", {})
        if not isinstance(local_model, Mapping) or (
            local_model.get("runtime_model_id") != hardware.get("runtime_model_id")
            or local_model.get("runtime_artifact_sha256")
            != hardware.get("runtime_artifact_sha256")
        ):
            raise ValueError(
                f"{label} local-model attestation differs from exact config"
            )
    return llm_cli_version


def validate_run_against_plan(
    plan: Mapping[str, Any], run: Mapping[str, Any]
) -> str:
    configuration = run.get("configuration", {})
    benchmark = configuration.get("benchmark", {})
    if benchmark.get("plan_sha256") != plan_sha256(plan):
        raise ValueError("run benchmark plan hash differs from the scoring plan")
    if configuration.get("attempts_per_case") != plan["review_contract"][
        "attempts_per_case"
    ]:
        raise ValueError("run attempt count differs from the frozen plan")
    candidate_id = benchmark.get("candidate_id")
    if candidate_id not in plan["candidates"]:
        raise ValueError("run does not name a candidate from this benchmark")
    candidate = plan["candidates"][candidate_id]
    model_config = benchmark.get("model_run_config", {})
    execution = plan["execution_contract"]
    for key, expected in (
        ("timeout_seconds", execution["timeout_seconds"]),
        ("max_output_tokens", execution["max_output_tokens"]),
        ("max_output_bytes", execution["max_output_bytes"]),
    ):
        if model_config.get(key) != expected:
            raise ValueError(f"run model config {key} differs from frozen contract")
    model_options = model_config.get("model_options")
    if not isinstance(model_options, Mapping) or not _matches_frozen_model_options(
        model_options,
        execution["temperature"],
        execution.get("reasoning_effort"),
    ):
        raise ValueError("run model options differ from frozen contract")
    for key in ("model_artifact", "model_revision", "locality"):
        expected = candidate[
            "required_locality" if key == "locality" else key
        ]
        if model_config.get(key) != expected:
            raise ValueError(f"run candidate {key} does not match the frozen plan")
    if model_config.get("locality") == "local":
        runtime_artifact = candidate.get("runtime_artifact", {})
        if runtime_artifact.get("status") != "approved" or (
            model_config.get("model") != runtime_artifact.get("model")
            or model_config.get("runtime_version")
            != runtime_artifact.get("runtime_version")
            or model_config.get("runtime") != runtime_artifact.get("runtime")
            or model_config.get("hardware", {}).get("runtime_model_id")
            != runtime_artifact.get("runtime_model_id")
            or model_config.get("hardware", {}).get("runtime_artifact_sha256")
            != runtime_artifact.get("runtime_artifact_sha256")
            or model_config.get("quantization")
            != runtime_artifact.get("quantization")
            or model_config.get("max_output_tokens_option")
            != runtime_artifact.get("max_output_tokens_option")
        ):
            raise ValueError(
                "run local-model identity differs from the frozen candidate binding"
            )
    else:
        runtime_artifact = candidate.get("runtime_artifact", {})
        hosted_identity = {
            "provider": model_config.get("provider"),
            "runtime": model_config.get("runtime"),
            "runtime_version": model_config.get("runtime_version"),
            "model": model_config.get("model"),
            "quantization": model_config.get("quantization"),
            "provider_model_revision": model_config.get("hardware", {}).get(
                "provider_model_revision"
            ),
            "max_output_tokens_option": model_config.get(
                "max_output_tokens_option"
            ),
        }
        if runtime_artifact.get("status") != "approved" or any(
            hosted_identity[key] != runtime_artifact.get(key)
            for key in hosted_identity
        ):
            raise ValueError(
                "run hosted identity differs from the frozen candidate binding"
            )
    startup_attestation = benchmark.get("runtime_attestation")
    startup_cli_version = _validate_runtime_attestation(
        model_config,
        startup_attestation,
        label="run",
    )
    fixture = configuration.get("fixture", {})
    if fixture.get("revision") != plan["fixture"]["revision"]:
        raise ValueError("run fixture revision does not match the frozen plan")
    if fixture.get("manifest_sha256") != plan["fixture"]["manifest_sha256"]:
        raise ValueError("run fixture hash does not match the frozen plan")
    if fixture.get("fixture_tree_sha256") != plan["fixture"]["fixture_tree_sha256"]:
        raise ValueError("run fixture file hash does not match the frozen plan")
    case_ids = fixture.get("case_ids")
    manifest_cases = {case["id"]: case for case in replay.load_manifest()["cases"]}
    if (
        not isinstance(case_ids, list)
        or any(not isinstance(case_id, str) for case_id in case_ids)
        or len(case_ids) != len(set(case_ids))
        or any(case_id not in manifest_cases for case_id in case_ids)
    ):
        raise ValueError("run fixture case IDs are invalid or outside the frozen corpus")
    selected_cases = [manifest_cases[case_id] for case_id in case_ids]
    if fixture.get("provider_payload_sha256") != provider_fixture_payload_sha256(
        cases=selected_cases
    ):
        raise ValueError("run provider payload hash does not match selected fixture bytes")
    for name in ("prompt", "schema"):
        expected = plan["review_contract"][f"{name}_sha256"]
        if configuration.get(name, {}).get("sha256") != expected:
            raise ValueError(f"run {name} hash does not match the frozen plan")
    adjudication = run.get("adjudication")
    if not isinstance(adjudication, dict):
        raise ValueError("scored run must embed its complete adjudication")
    if run.get("adjudication_sha256") != scoring.adjudication_sha256(adjudication):
        raise ValueError("scored run adjudication hash does not match")
    recomputed_metrics = scoring.score_run(dict(run), adjudication)
    if run.get("metrics") != recomputed_metrics:
        raise ValueError("scored run metrics do not match retained adjudication")

    valid_statuses = {
        "schema-valid", "schema-error", "timeout", "provider-error",
        "output-limit", "budget-blocked", "harness-error", "interrupted",
    }
    seen_case_ids: set[str] = set()
    for outcome in run.get("cases", []):
        case_id = outcome.get("case_id")
        if case_id not in manifest_cases:
            raise ValueError(f"scored run contains unknown case {case_id!r}")
        if case_id not in case_ids:
            raise ValueError(f"scored run case {case_id!r} was not in its fixture binding")
        if case_id in seen_case_ids:
            raise ValueError("ordinary benchmark contains a repeated case attempt")
        seen_case_ids.add(case_id)
        fixture_case = manifest_cases[case_id]
        for key, expected in (
            ("evaluation_split", fixture_case["evaluation_split"]),
            ("failure_families", fixture_case["failure_families"]),
            (
                "expected_finding_ids",
                [finding["id"] for finding in fixture_case["expected_findings"]],
            ),
            ("expected_findings", fixture_case["expected_findings"]),
        ):
            if outcome.get(key) != expected:
                raise ValueError(f"scored run case {case_id} has altered {key}")
        if outcome.get("status") not in valid_statuses:
            raise ValueError(f"scored run case {case_id} has invalid status")
        model_attempted = outcome.get("model_attempted")
        if model_attempted is True:
            attempt_count = outcome.get("model_attempt_count")
            if (
                isinstance(attempt_count, bool)
                or not isinstance(attempt_count, int)
                or attempt_count != 1
            ):
                raise ValueError(
                    f"scored run case {case_id} must retain its single model attempt"
                )
            if outcome.get("effective_model_config") != model_config:
                raise ValueError(
                    f"scored run case {case_id} effective model config differs "
                    "from the approved run config"
                )
            _validate_runtime_attestation(
                model_config,
                outcome.get("runtime_attestation"),
                label=f"scored run case {case_id}",
                expected_cli_version=startup_cli_version,
            )
        elif model_attempted is False:
            if outcome.get("model_attempt_count") is not None:
                raise ValueError(
                    f"scored run case {case_id} records a contradictory model attempt"
                )
            if outcome.get("runtime_attestation") is not None:
                raise ValueError(
                    f"scored run case {case_id} records an unattached runtime attestation"
                )
            if outcome.get("effective_model_config") is not None:
                raise ValueError(
                    f"scored run case {case_id} records an unattached effective config"
                )
            if outcome.get("status") not in {"provider-error", "timeout"}:
                raise ValueError(
                    f"scored run case {case_id} has an invalid pre-inference status"
                )
        elif model_attempted is not None:
            raise ValueError(
                f"scored run case {case_id} has invalid model-attempt provenance"
            )
        elif outcome.get("status") not in {"harness-error", "interrupted"}:
            raise ValueError(
                f"scored run case {case_id} is missing model-attempt provenance"
            )
        elif (
            outcome.get("model_attempt_count") is not None
            or outcome.get("runtime_attestation") is not None
            or outcome.get("effective_model_config") is not None
        ):
            raise ValueError(
                f"scored run case {case_id} has contradictory attempt provenance"
            )
    return str(candidate_id)


def _percentile(values: list[int], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = math.ceil(percentile * len(ordered)) - 1
    return float(ordered[max(0, index)])


def summarize_run(
    run: Mapping[str, Any], *, required_case_ids: set[str]
) -> dict[str, Any]:
    metrics = run.get("metrics", {})
    if metrics.get("adjudication_status") != "complete":
        return {"status": "awaiting-adjudication"}
    cases = list(run.get("cases", []))
    latencies = [
        int(case["latency_ms"]) for case in cases
        if isinstance(case.get("latency_ms"), (int, float))
    ]
    tp = metrics.get("tp", 0)
    fp = metrics.get("fp", 0)
    fn = metrics.get("fn", 0)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    holdout = metrics.get("split_metrics", {}).get("holdout", {})
    decision_cases = [
        case for case in cases
        if case.get("evaluation_split") in {"holdout", "clean-control"}
    ]
    decision_latencies = [
        int(case["latency_ms"]) for case in decision_cases
        if isinstance(case.get("latency_ms"), (int, float))
    ]
    decision_schema_valid = sum(
        case.get("status") == "schema-valid" for case in decision_cases
    )
    usage_cases = [case for case in cases if case.get("input_tokens") is not None]
    known_costs = [case["cost_usd"] for case in cases if case.get("cost_usd") is not None]
    observed_case_ids = [str(case.get("case_id")) for case in cases]
    suite_complete = (
        len(observed_case_ids) == len(set(observed_case_ids))
        and set(observed_case_ids) == required_case_ids
    )
    model_config = run.get("configuration", {}).get("benchmark", {}).get(
        "model_run_config", {}
    )
    usage_complete = bool(cases) and all(
        case.get("input_tokens") is not None
        and case.get("output_tokens") is not None
        for case in cases
    )
    cost_complete = bool(cases) and all(
        case.get("cost_usd") is not None for case in cases
    )
    resource_complete = True
    if model_config.get("locality") == "local":
        resource_complete = bool(cases) and all(
            isinstance(case.get("resource_usage"), Mapping)
            and case["resource_usage"].get("model_load_ms") is not None
            and case["resource_usage"].get("peak_memory_bytes") is not None
            for case in cases
        )
    measurement_complete = usage_complete and cost_complete and resource_complete
    if not suite_complete:
        status = "incomplete-run"
    elif not measurement_complete:
        status = "incomplete-measurements"
    else:
        status = "complete"
    return {
        "status": status,
        "attempted_cases": len(cases),
        "status_counts": {
            status: sum(case.get("status") == status for case in cases)
            for status in sorted({str(case.get("status")) for case in cases})
        },
        "finding_precision": holdout.get("finding_precision"),
        "finding_recall": holdout.get("finding_recall"),
        "schema_valid_rate": (
            decision_schema_valid / len(decision_cases) if decision_cases else None
        ),
        "clean_control_false_positive_rate": metrics.get(
            "clean_control_false_positive_rate"
        ),
        "fix_quality_score": holdout.get("fix_quality_score"),
        "latency_ms": {
            "p50": statistics.median(decision_latencies) if decision_latencies else None,
            "p95": _percentile(decision_latencies, 0.95),
        },
        "aggregate_report_only": {
            "finding_precision": precision,
            "finding_recall": recall,
            "schema_valid_rate": metrics.get("schema_valid_rate"),
            "fix_quality_score": metrics.get("fix_quality_score"),
            "latency_ms": {
                "p50": statistics.median(latencies) if latencies else None,
                "p95": _percentile(latencies, 0.95),
            },
        },
        "usage": {
            "reported_cases": len(usage_cases),
            "input_tokens": sum(case.get("input_tokens") or 0 for case in cases),
            "output_tokens": sum(case.get("output_tokens") or 0 for case in cases),
        },
        "cost": {
            "known_cases": len(known_costs),
            "unknown_cases": len(cases) - len(known_costs),
            "total_usd": sum(known_costs) if len(known_costs) == len(cases) else None,
            "maximum_authorized_usd": run.get("configuration", {}).get(
                "benchmark", {}
            ).get("model_run_config", {}).get("max_cost_usd"),
        },
        "resource_usage": {
            "records": [case.get("resource_usage") for case in cases],
            "hardware": run.get("configuration", {}).get("benchmark", {}).get(
                "model_run_config", {}
            ).get("hardware"),
        },
        "measurement_completeness": {
            "usage": usage_complete,
            "cost": cost_complete,
            "local_resources": resource_complete,
        },
        "by_split": metrics.get("by_split", {}),
        "split_metrics": metrics.get("split_metrics", {}),
        "by_family": metrics.get("by_family", {}),
    }


def _eligible(summary: Mapping[str, Any], rule: Mapping[str, Any]) -> bool:
    if summary.get("status") != "complete":
        return False
    gates = rule["eligibility_gates"]
    values = (
        summary.get("schema_valid_rate"), summary.get("finding_precision"),
        summary.get("clean_control_false_positive_rate"),
        summary.get("fix_quality_score"), summary.get("latency_ms", {}).get("p95"),
    )
    if any(value is None for value in values):
        return False
    return bool(
        values[0] >= gates["minimum_schema_valid_rate"]
        and values[1] >= gates["minimum_finding_precision"]
        and values[2] <= gates["maximum_clean_control_false_positive_rate"]
        and values[3] >= gates["minimum_fix_quality_score"]
        and values[4] <= gates["maximum_p95_latency_ms"]
    )


def build_scorecard(
    plan: Mapping[str, Any], runs: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    validate_plan(plan)
    if plan["evaluation_scope"] != ORDINARY_PLAN_SCOPE:
        raise ValueError("challenge qualification plans cannot build ordinary scorecards")
    summaries = {
        candidate_id: {"status": "not-run"}
        for candidate_id in plan["candidate_order"]
    }
    sources: list[str] = []
    fixture_manifest = replay.load_manifest()
    required_splits = set(plan["fixture"]["required_splits"])
    required_case_ids = {
        case["id"] for case in fixture_manifest["cases"]
        if case["evaluation_split"] in required_splits
    }
    for run in runs:
        candidate_id = validate_run_against_plan(plan, run)
        if summaries[candidate_id]["status"] != "not-run":
            raise ValueError(f"duplicate run for candidate {candidate_id}")
        summaries[candidate_id] = summarize_run(
            run, required_case_ids=required_case_ids
        )
        sources.append(str(run.get("run_id", "unknown")))
    eligible = [
        candidate_id for candidate_id in plan["candidate_order"]
        if _eligible(summaries[candidate_id], plan["decision_rule"])
    ]
    complete = all(summary["status"] == "complete" for summary in summaries.values())
    selection = None
    outcome = "comparison is incomplete; retain the current default"
    if complete and not eligible:
        outcome = "no candidate established eligibility; retain the current default"
    elif complete:
        def quality_key(candidate_id: str) -> tuple[float, float, float, float, float]:
            summary = summaries[candidate_id]
            return (
                -summary["finding_precision"],
                summary["clean_control_false_positive_rate"],
                -summary["fix_quality_score"],
                -summary["finding_recall"],
                summary["latency_ms"]["p95"],
            )

        ranked = sorted(eligible, key=quality_key)
        best_key = quality_key(ranked[0])
        tied = [candidate_id for candidate_id in ranked if quality_key(candidate_id) == best_key]
        if len(tied) == 1:
            selection = tied[0]
            outcome = (
                f"{selection} wins at the first differing pre-registered ranking measure; "
                "review the evidence before changing any production default"
            )
        else:
            known_costs = {
                candidate_id: summaries[candidate_id]["cost"]["total_usd"]
                for candidate_id in tied
            }
            if all(cost is not None for cost in known_costs.values()):
                lowest = min(known_costs.values())
                cheapest = [
                    candidate_id for candidate_id, cost in known_costs.items()
                    if cost == lowest
                ]
                if len(cheapest) == 1:
                    selection = cheapest[0]
                    outcome = (
                        f"{selection} wins the final normalized-cost tie-break; "
                        "review the evidence before changing any production default"
                    )
            if selection is None:
                outcome = (
                    "eligible candidates remain tied without comparable normalized cost; "
                    "retain the current default"
                )
    return {
        "format_version": 1,
        "experiment_id": plan["experiment_id"],
        "status": "complete" if complete else "incomplete",
        "generated_from": sources,
        "candidates": summaries,
        "decision": {
            "selection": selection,
            "current_default": plan["current_default"],
            "outcome": outcome,
            "eligible_candidates": eligible,
        },
    }


def _json_object(value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("must be a JSON object")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare or score the frozen model benchmark")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="write one exact run config")
    prepare.add_argument("candidate", choices=(
        "qwen35-a3b-local", "deepseek-v4-flash-hosted", "devstral-small-2-local"
    ))
    prepare.add_argument("--model", required=True, help="configured llm model ID or alias")
    prepare.add_argument("--provider", required=True)
    prepare.add_argument("--runtime", required=True)
    prepare.add_argument("--runtime-version", required=True)
    prepare.add_argument("--quantization", required=True)
    prepare.add_argument("--hardware", type=_json_object, required=True)
    prepare.add_argument("--model-options", type=_json_object, default={})
    prepare.add_argument("--timeout", type=float, default=60.0)
    prepare.add_argument("--max-output-tokens", type=int, default=4096)
    prepare.add_argument("--max-output-tokens-option", required=True)
    prepare.add_argument("--max-output-bytes", type=int, default=262144)
    prepare.add_argument("--input-usd-per-million", type=float)
    prepare.add_argument("--output-usd-per-million", type=float)
    prepare.add_argument("--pricing-source", default="not-applicable")
    prepare.add_argument("--pricing-as-of", default="not-applicable")
    prepare.add_argument("--max-cost-usd", type=float)
    prepare.add_argument("--allow-external-upload", action="store_true")
    prepare.add_argument("--output", type=Path, required=True)
    score = subparsers.add_parser("scorecard", help="aggregate adjudicated run artifacts")
    score.add_argument("runs", nargs="*", type=Path)
    score.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan = load_plan(args.plan)
    if args.command == "prepare":
        config = prepare_run_config(
            plan, candidate_id=args.candidate, model=args.model,
            provider=args.provider, runtime=args.runtime,
            runtime_version=args.runtime_version, quantization=args.quantization,
            model_options=args.model_options, timeout_seconds=args.timeout,
            max_output_tokens=args.max_output_tokens,
            max_output_tokens_option=args.max_output_tokens_option,
            max_output_bytes=args.max_output_bytes,
            pricing=Pricing(
                args.input_usd_per_million, args.output_usd_per_million,
                args.pricing_source, args.pricing_as_of,
            ),
            max_cost_usd=args.max_cost_usd,
            external_upload_consent=args.allow_external_upload,
            hardware=args.hardware,
        )
        artifacts.write_private_json(args.output, config.to_dict())
    else:
        runs = [json.loads(path.read_text(encoding="utf-8")) for path in args.runs]
        artifacts.write_private_json(args.output, build_scorecard(plan, runs))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
