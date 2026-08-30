"""One-shot, bounded model invocation with explicit privacy and cost policy.

This module deliberately knows nothing about Quodet's review prompt or finding
schema.  Callers supply immutable, already-sanitized documents and the exact
prompt/schema they want the configured model to process.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence


MAX_PROVIDER_OUTPUT_BYTES = 262_144
RUN_STATUSES = frozenset(
    {"success", "timeout", "provider-error", "output-limit", "budget-blocked"}
)
Locality = Literal["local", "hosted"]
Scalar = str | int | float | bool
USAGE_LINE = re.compile(
    r"Token usage:\s*(?:(?P<input>[\d,]+) input)?"
    r"(?:,\s*)?(?:(?P<output>[\d,]+) output)?"
)


@dataclass(frozen=True)
class ModelDocument:
    path: Path
    media_type: str = "text/plain"


@dataclass(frozen=True)
class ModelRunRequest:
    documents: tuple[ModelDocument, ...]
    prompt: str
    schema_json: str
    cwd: Path
    log: bool = False


@dataclass(frozen=True)
class Pricing:
    input_usd_per_million_tokens: float | None
    output_usd_per_million_tokens: float | None
    source: str
    as_of: str


@dataclass(frozen=True)
class ModelRunConfig:
    candidate_id: str
    model: str
    model_artifact: str
    model_revision: str
    provider: str
    runtime: str
    runtime_version: str
    locality: Locality
    quantization: str
    model_options: Mapping[str, Scalar] = field(default_factory=dict)
    context_limit: int = 0
    timeout_seconds: float = 60.0
    max_output_bytes: int = MAX_PROVIDER_OUTPUT_BYTES
    max_output_tokens: int = 4096
    max_output_tokens_option: str = "max_tokens"
    pricing: Pricing = field(
        default_factory=lambda: Pricing(None, None, "not-applicable", "not-applicable")
    )
    max_cost_usd: float | None = None
    external_upload_consent: bool = False
    hardware: Mapping[str, Scalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        required = {
            "candidate_id": self.candidate_id,
            "model": self.model,
            "model_artifact": self.model_artifact,
            "model_revision": self.model_revision,
            "provider": self.provider,
            "runtime": self.runtime,
            "runtime_version": self.runtime_version,
            "quantization": self.quantization,
        }
        missing = [name for name, value in required.items() if not value.strip()]
        if missing:
            raise ValueError(f"model run config has blank required fields: {missing}")
        if self.locality not in {"local", "hosted"}:
            raise ValueError("locality must be 'local' or 'hosted'")
        if self.context_limit <= 0:
            raise ValueError("context_limit must be greater than zero")
        if self.timeout_seconds <= 0 or not math.isfinite(self.timeout_seconds):
            raise ValueError("timeout_seconds must be finite and greater than zero")
        if self.max_output_bytes <= 0 or self.max_output_tokens <= 0:
            raise ValueError("output limits must be greater than zero")
        if not self.max_output_tokens_option.strip():
            raise ValueError("max_output_tokens_option must be explicit")
        if (
            self.max_output_tokens_option in self.model_options
            and self.model_options[self.max_output_tokens_option]
            != self.max_output_tokens
        ):
            raise ValueError(
                "model_options output-token limit differs from max_output_tokens"
            )
        if self.max_cost_usd is not None and (
            not math.isfinite(self.max_cost_usd) or self.max_cost_usd <= 0
        ):
            raise ValueError("max_cost_usd must be finite and greater than zero")
        if not self.pricing.source.strip() or not self.pricing.as_of.strip():
            raise ValueError("pricing source and as_of must be explicit")
        for name, value in asdict(self.pricing).items():
            if name.endswith("_tokens") and value is not None and (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"pricing {name} must be a non-negative number")
        for collection_name, values in (
            ("model_options", self.model_options),
            ("hardware", self.hardware),
        ):
            if not isinstance(values, Mapping) or any(
                not isinstance(key, str) or not isinstance(value, (str, int, float, bool))
                for key, value in values.items()
            ):
                raise ValueError(f"{collection_name} must contain scalar values")
            if any(
                isinstance(value, float) and not math.isfinite(value)
                for value in values.values()
            ):
                raise ValueError(f"{collection_name} cannot contain non-finite values")
        for name in (
            "amortized_hourly_cost_usd", "model_load_ms", "peak_memory_bytes"
        ):
            value = self.hardware.get(name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value < 0
            ):
                raise ValueError(f"hardware {name} must be a non-negative number")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["model_options"] = dict(sorted(self.model_options.items()))
        value["hardware"] = dict(sorted(self.hardware.items()))
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelRunConfig":
        expected = {
            "candidate_id", "model", "model_artifact", "model_revision",
            "provider", "runtime", "runtime_version", "locality", "quantization",
            "model_options", "context_limit", "timeout_seconds", "max_output_bytes",
            "max_output_tokens", "max_output_tokens_option", "pricing", "max_cost_usd",
            "external_upload_consent", "hardware",
        }
        unexpected = set(value) - expected
        missing = expected - set(value)
        if missing or unexpected:
            raise ValueError(
                f"model run config keys differ: missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )
        pricing = value["pricing"]
        if not isinstance(pricing, Mapping):
            raise ValueError("pricing must be an object")
        if not isinstance(value["external_upload_consent"], bool):
            raise ValueError("external_upload_consent must be a boolean")
        for name in (
            "context_limit", "max_output_bytes", "max_output_tokens"
        ):
            if isinstance(value[name], bool) or not isinstance(value[name], int):
                raise ValueError(f"{name} must be an integer")
        if (
            isinstance(value["timeout_seconds"], bool)
            or not isinstance(value["timeout_seconds"], (int, float))
        ):
            raise ValueError("timeout_seconds must be a number")
        return cls(
            candidate_id=str(value["candidate_id"]),
            model=str(value["model"]),
            model_artifact=str(value["model_artifact"]),
            model_revision=str(value["model_revision"]),
            provider=str(value["provider"]),
            runtime=str(value["runtime"]),
            runtime_version=str(value["runtime_version"]),
            locality=value["locality"],
            quantization=str(value["quantization"]),
            model_options=dict(value["model_options"]),
            context_limit=int(value["context_limit"]),
            timeout_seconds=float(value["timeout_seconds"]),
            max_output_bytes=int(value["max_output_bytes"]),
            max_output_tokens=int(value["max_output_tokens"]),
            max_output_tokens_option=str(value["max_output_tokens_option"]),
            pricing=Pricing(
                pricing.get("input_usd_per_million_tokens"),
                pricing.get("output_usd_per_million_tokens"),
                str(pricing.get("source", "")),
                str(pricing.get("as_of", "")),
            ),
            max_cost_usd=(
                None if value["max_cost_usd"] is None
                else float(value["max_cost_usd"])
            ),
            external_upload_consent=bool(value["external_upload_consent"]),
            hardware=dict(value["hardware"]),
        )


def model_run_config_sha256(config: ModelRunConfig) -> str:
    """Hash the canonical execution config used across process boundaries."""
    payload = json.dumps(
        config.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class PreflightResult:
    allowed: bool
    reason: str | None
    conservative_input_tokens: int
    maximum_cost_usd: float | None


@dataclass(frozen=True)
class ModelRunResult:
    status: str
    returncode: int | None
    stdout: str
    stderr: str
    latency_ms: int
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    maximum_cost_usd: float | None
    resource_usage: Mapping[str, Scalar | None]
    effective_config: Mapping[str, Any]
    attempt_count: int = 1

    def __post_init__(self) -> None:
        if self.status not in RUN_STATUSES:
            raise ValueError(f"invalid model run status: {self.status}")
        if self.attempt_count != 1:
            raise ValueError("the model runner performs exactly one attempt")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BoundedProcessResult:
    returncode: int
    stdout: str
    stderr: str
    output_exceeded: bool = False
    timed_out: bool = False
    latency_ms: int = 0


def load_model_run_config(path: Path) -> ModelRunConfig:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("model run config must be a JSON object")
    return ModelRunConfig.from_dict(value)


def conservative_request_tokens(request: ModelRunRequest) -> int:
    """Return a tokenizer-independent upper bound suitable for cost gating.

    For the UTF-8 text accepted by Quodet, the number of model tokens cannot
    exceed the number of input bytes.  Counting every byte as one token is
    intentionally conservative and avoids provider calls during preflight.
    """
    total = len(request.prompt.encode()) + len(request.schema_json.encode())
    for document in request.documents:
        total += document.path.stat().st_size
    return total


def preflight_model_run(
    config: ModelRunConfig,
    request: ModelRunRequest,
    *,
    cost_allowance_usd: float | None = None,
) -> PreflightResult:
    input_tokens = conservative_request_tokens(request)
    if input_tokens + config.max_output_tokens > config.context_limit:
        return PreflightResult(
            False, "request plus maximum output exceeds the configured context limit",
            input_tokens, None,
        )
    if config.locality == "local":
        if config.provider != "local":
            return PreflightResult(
                False,
                "local execution requires provider='local'; hosted aliases must "
                "use locality='hosted' with consent and a cost cap",
                input_tokens,
                None,
            )
        return PreflightResult(True, None, input_tokens, None)
    if config.provider == "local":
        return PreflightResult(
            False, "hosted execution cannot claim provider='local'", input_tokens, None
        )
    if not config.external_upload_consent:
        return PreflightResult(
            False, "hosted execution requires explicit external-upload consent",
            input_tokens, None,
        )
    if config.max_cost_usd is None:
        return PreflightResult(
            False, "hosted execution requires an explicit positive cost cap",
            input_tokens, None,
        )
    if (
        config.pricing.input_usd_per_million_tokens is None
        or config.pricing.output_usd_per_million_tokens is None
    ):
        return PreflightResult(
            False, "hosted execution requires known input and output pricing",
            input_tokens, None,
        )
    if not config.pricing.source.startswith("https://"):
        return PreflightResult(
            False, "hosted execution requires an HTTPS pricing source",
            input_tokens, None,
        )
    try:
        date.fromisoformat(config.pricing.as_of)
    except ValueError:
        return PreflightResult(
            False, "hosted execution requires an ISO pricing as-of date",
            input_tokens, None,
        )
    # Provider/plugin framing is outside Quodet's control. Cost against the
    # entire configured context window, not a token estimate of raw files, so
    # hidden wrappers or system text cannot escape the authorization ceiling.
    maximum_billable_input_tokens = config.context_limit - config.max_output_tokens
    maximum_cost = (
        maximum_billable_input_tokens * config.pricing.input_usd_per_million_tokens
        + config.max_output_tokens * config.pricing.output_usd_per_million_tokens
    ) / 1_000_000
    allowance = config.max_cost_usd
    if cost_allowance_usd is not None:
        allowance = min(allowance, cost_allowance_usd)
    if maximum_cost > allowance:
        return PreflightResult(
            False,
            f"maximum request cost ${maximum_cost:.6f} exceeds remaining cap "
            f"${allowance:.6f}",
            input_tokens,
            maximum_cost,
        )
    return PreflightResult(True, None, input_tokens, maximum_cost)


def build_llm_command(config: ModelRunConfig, request: ModelRunRequest) -> list[str]:
    command = [
        "llm", "prompt", "--model", config.model, "--no-stream",
        "--schema", request.schema_json, "--usage",
    ]
    options = dict(config.model_options)
    options[config.max_output_tokens_option] = config.max_output_tokens
    for name, value in sorted(options.items()):
        option_value = str(value).lower() if isinstance(value, bool) else str(value)
        command.extend(["--option", name, option_value])
    command.append("--log" if request.log else "--no-log")
    for document in request.documents:
        command.extend(["--fragment", os.fspath(document.path)])
    command.append(request.prompt)
    return command


def run_bounded_command(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout: float,
    output_limit: int,
) -> BoundedProcessResult:
    """Run once, retaining at most ``output_limit + 1`` bytes per stream."""
    started = time.monotonic()
    with (
        tempfile.TemporaryFile(mode="w+b") as stdout_file,
        tempfile.TemporaryFile(mode="w+b") as stderr_file,
    ):
        process = subprocess.Popen(
            command, cwd=cwd, stdout=stdout_file, stderr=stderr_file,
        )
        deadline = started + timeout
        output_exceeded = False
        timed_out = False
        while process.poll() is None:
            if (
                os.fstat(stdout_file.fileno()).st_size > output_limit
                or os.fstat(stderr_file.fileno()).st_size > output_limit
            ):
                output_exceeded = True
                process.kill()
                process.wait()
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                process.kill()
                process.wait()
                break
            try:
                process.wait(timeout=min(0.05, remaining))
            except subprocess.TimeoutExpired:
                pass
        if (
            os.fstat(stdout_file.fileno()).st_size > output_limit
            or os.fstat(stderr_file.fileno()).st_size > output_limit
        ):
            output_exceeded = True
        stdout_file.seek(0)
        stderr_file.seek(0)
        return BoundedProcessResult(
            returncode=process.returncode,
            stdout=stdout_file.read(output_limit + 1).decode("utf-8", errors="replace"),
            stderr=stderr_file.read(output_limit + 1).decode("utf-8", errors="replace"),
            output_exceeded=output_exceeded,
            timed_out=timed_out,
            latency_ms=round((time.monotonic() - started) * 1000),
        )


def run_model(
    config: ModelRunConfig,
    request: ModelRunRequest,
    *,
    cost_allowance_usd: float | None = None,
) -> ModelRunResult:
    """Perform one preflighted attempt; never retry, repair, or fall back."""
    effective_config = config.to_dict()
    preflight = preflight_model_run(
        config, request, cost_allowance_usd=cost_allowance_usd
    )
    if not preflight.allowed:
        return ModelRunResult(
            status="budget-blocked", returncode=None, stdout="",
            stderr=preflight.reason or "preflight rejected the request", latency_ms=0,
            input_tokens=None, output_tokens=None, cost_usd=None,
            maximum_cost_usd=preflight.maximum_cost_usd,
            resource_usage={
                "model_load_ms": None, "peak_memory_bytes": None,
                "measurement_status": "not-started",
            },
            effective_config=effective_config,
        )
    try:
        result = run_bounded_command(
            build_llm_command(config, request), cwd=request.cwd,
            timeout=config.timeout_seconds, output_limit=config.max_output_bytes,
        )
    except OSError as error:
        return ModelRunResult(
            status="provider-error", returncode=None, stdout="", stderr=str(error),
            latency_ms=0, input_tokens=None, output_tokens=None, cost_usd=None,
            maximum_cost_usd=preflight.maximum_cost_usd,
            resource_usage={
                "model_load_ms": None, "peak_memory_bytes": None,
                "measurement_status": "unavailable",
            },
            effective_config=effective_config,
        )
    if result.timed_out:
        status = "timeout"
    elif result.output_exceeded:
        status = "output-limit"
    elif result.returncode != 0:
        status = "provider-error"
    else:
        status = "success"
    input_tokens = None
    output_tokens = None
    for match in USAGE_LINE.finditer(result.stderr):
        input_tokens = (
            int(match.group("input").replace(",", ""))
            if match.group("input") else None
        )
        output_tokens = (
            int(match.group("output").replace(",", ""))
            if match.group("output") else None
        )
    cost_usd = None
    if input_tokens is not None and output_tokens is not None:
        if (
            config.pricing.input_usd_per_million_tokens is not None
            and config.pricing.output_usd_per_million_tokens is not None
        ):
            cost_usd = (
                input_tokens * config.pricing.input_usd_per_million_tokens
                + output_tokens * config.pricing.output_usd_per_million_tokens
            ) / 1_000_000
    if config.locality == "local" and isinstance(
        config.hardware.get("amortized_hourly_cost_usd"), (int, float)
    ):
        cost_usd = (
            result.latency_ms
            * float(config.hardware["amortized_hourly_cost_usd"])
            / 3_600_000
        )
    model_load_ms = config.hardware.get("model_load_ms")
    peak_memory_bytes = config.hardware.get("peak_memory_bytes")
    measured_resources = (
        isinstance(model_load_ms, (int, float))
        and isinstance(peak_memory_bytes, (int, float))
    )
    return ModelRunResult(
        status=status, returncode=result.returncode, stdout=result.stdout,
        stderr=result.stderr, latency_ms=result.latency_ms,
        input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost_usd,
        maximum_cost_usd=preflight.maximum_cost_usd,
        resource_usage={
            "wall_time_ms": result.latency_ms,
            "model_load_ms": model_load_ms if measured_resources else None,
            "peak_memory_bytes": peak_memory_bytes if measured_resources else None,
            "measurement_status": (
                "config-supplied" if measured_resources else "runtime-did-not-report"
            ),
        },
        effective_config=effective_config,
    )
