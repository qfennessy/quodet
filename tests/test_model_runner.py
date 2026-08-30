from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from model_runner import (
    BoundedProcessResult,
    ModelDocument,
    ModelRunConfig,
    ModelRunRequest,
    Pricing,
    preflight_model_run,
    run_model,
)


def config(*, locality: str = "local", consent: bool = False) -> ModelRunConfig:
    return ModelRunConfig(
        candidate_id="candidate",
        model="alias",
        model_artifact="owner/model",
        model_revision="a" * 40,
        provider="local" if locality == "local" else "provider",
        runtime="runtime",
        runtime_version="1.0.0",
        locality=locality,  # type: ignore[arg-type]
        quantization="fp8",
        model_options={"temperature": 0},
        context_limit=10000,
        timeout_seconds=5,
        max_output_bytes=1024,
        max_output_tokens=100,
        pricing=Pricing(
            1.0 if locality == "hosted" else None,
            2.0 if locality == "hosted" else None,
            "https://provider.example/pricing" if locality == "hosted" else "test",
            "2026-08-30",
        ),
        max_cost_usd=1.0 if locality == "hosted" else None,
        external_upload_consent=consent,
        hardware={"device": "fixture"},
    )


class ModelRunnerTests(unittest.TestCase):
    def request(self, root: Path) -> ModelRunRequest:
        document = root / "sanitized.txt"
        document.write_text("safe source", encoding="utf-8")
        return ModelRunRequest(
            documents=(ModelDocument(document),),
            prompt="review",
            schema_json='{"type":"object"}',
            cwd=root,
        )

    def test_hosted_preflight_fails_closed_without_consent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = preflight_model_run(
                config(locality="hosted", consent=False),
                self.request(Path(temporary_directory)),
            )
        self.assertFalse(result.allowed)
        self.assertIn("consent", result.reason or "")

    def test_hosted_preflight_fails_closed_without_dated_pricing_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            hosted = config(locality="hosted", consent=True)
            invalid = ModelRunConfig.from_dict({
                **hosted.to_dict(),
                "pricing": {
                    **hosted.to_dict()["pricing"],
                    "source": "not-applicable",
                    "as_of": "not-applicable",
                },
            })
            result = preflight_model_run(
                invalid, self.request(Path(temporary_directory)),
            )
        self.assertFalse(result.allowed)
        self.assertIn("pricing source", result.reason or "")

    def test_run_model_attempts_once_and_retains_exact_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            model_config = config()
            with mock.patch(
                "model_runner.run_bounded_command",
                return_value=BoundedProcessResult(
                    returncode=0, stdout='{"findings": []}',
                    stderr="Token usage: 123 input, 45 output\n",
                    latency_ms=12,
                ),
            ) as command:
                result = run_model(model_config, self.request(root))
        self.assertEqual(result.status, "success")
        self.assertEqual(result.attempt_count, 1)
        self.assertEqual(result.effective_config, model_config.to_dict())
        self.assertEqual((result.input_tokens, result.output_tokens), (123, 45))
        self.assertIsNone(result.cost_usd)
        command.assert_called_once()
        invoked = command.call_args.args[0]
        option_index = invoked.index("max_tokens")
        self.assertEqual(invoked[option_index - 1 : option_index + 2], [
            "--option", "max_tokens", "100",
        ])
        self.assertIn("--usage", invoked)

    def test_output_limit_is_a_retained_failure_without_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with mock.patch(
                "model_runner.run_bounded_command",
                return_value=BoundedProcessResult(
                    returncode=-9, stdout="x" * 1025, stderr="",
                    output_exceeded=True, latency_ms=7,
                ),
            ) as command:
                result = run_model(config(), self.request(root))
        self.assertEqual(result.status, "output-limit")
        self.assertEqual(len(result.stdout), 1025)
        command.assert_called_once()


if __name__ == "__main__":
    unittest.main()
