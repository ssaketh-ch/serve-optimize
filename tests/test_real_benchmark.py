import subprocess

import pytest

import serve_optimize.real_benchmark as real_benchmark
from serve_optimize.real_benchmark import _choose_device, _missing_power_result, query_nvidia_smi_power
from serve_optimize.schemas import ServingConfig


class _CudaAvailable:
    @staticmethod
    def is_available() -> bool:
        return True


class _TorchWithCuda:
    cuda = _CudaAvailable()


def test_choose_device_preserves_the_requested_gpu_index() -> None:
    assert _choose_device(_TorchWithCuda(), "auto", device_index=2) == "cuda:2"
    assert _choose_device(_TorchWithCuda(), "cpu", device_index=2) == "cpu"


def test_power_query_targets_the_same_gpu_index(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        return subprocess.CompletedProcess(command, 0, stdout="123.5\n", stderr="")

    monkeypatch.setattr(real_benchmark.subprocess, "run", fake_run)

    assert query_nvidia_smi_power(device_index=3) == pytest.approx(123.5)
    assert captured["command"][1:3] == ["-i", "3"]


def test_missing_power_keeps_throughput_but_is_not_recommendation_eligible() -> None:
    result = _missing_power_result(
        _config(),
        throughput=42.0,
        generated_tokens=84,
        raw={"mode": "test"},
    )

    assert result.throughput_tok_s == pytest.approx(42.0)
    assert result.generated_tokens == 84
    assert result.feasible is False
    assert result.average_power_watts == 0.0
    assert result.joules_per_token == 0.0
    assert result.raw["power_source"] == "unavailable"


def _config() -> ServingConfig:
    return ServingConfig(
        id="config",
        backend="transformers",
        model_id="model",
        dtype="fp16",
        quantization="none",
        max_batch_size=1,
        max_context_tokens=128,
        kv_cache_policy="default",
        scheduler="eager",
    )
