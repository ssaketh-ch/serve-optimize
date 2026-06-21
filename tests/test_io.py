import json

import pytest

from serve_optimize.io import load_result_jsonl, write_jsonl
from serve_optimize.schemas import BenchmarkResult, ServingConfig


def test_load_result_jsonl_round_trips_current_benchmark_result(tmp_path) -> None:
    result = BenchmarkResult(
        config=_config("cfg_current"),
        throughput_tok_s=100.0,
        average_power_watts=200.0,
        joules_per_token=2.0,
        tokens_per_watt=0.5,
        ttft_ms=12.0,
        p95_latency_ms=45.0,
        raw={"mode": "test"},
    )
    path = tmp_path / "results.jsonl"
    write_jsonl(path, [result])

    loaded = load_result_jsonl(path)

    assert loaded == [result]


def test_load_result_jsonl_supports_legacy_aliases_and_blank_lines(tmp_path) -> None:
    path = tmp_path / "legacy.jsonl"
    row = {
        "config": {
            "config_id": "cfg_legacy",
            "backend": "vllm",
            "model": "model_path",
            "dtype": "fp16",
            "quantization": "none",
            "batch_size": 4,
            "max_model_len": 1024,
            "tp": 2,
        },
        "throughput_tokens_per_sec": "100.0",
        "average_power_w": "200.0",
        "peak_power_w": 225.0,
        "completion_tokens": 256,
        "feasible": "true",
    }
    path.write_text(json.dumps(row) + "\n\n", encoding="utf-8")

    loaded = load_result_jsonl(path)

    assert len(loaded) == 1
    result = loaded[0]
    assert result.config.id == "cfg_legacy"
    assert result.config.model_id == "model_path"
    assert result.config.max_batch_size == 4
    assert result.config.max_context_tokens == 1024
    assert result.config.tensor_parallelism == 2
    assert result.throughput_tok_s == pytest.approx(100.0)
    assert result.average_power_watts == pytest.approx(200.0)
    assert result.joules_per_token == pytest.approx(2.0)
    assert result.tokens_per_watt == pytest.approx(0.5)
    assert result.peak_power_watts == pytest.approx(225.0)
    assert result.generated_tokens == 256
    assert result.feasible is True


def test_load_result_jsonl_reports_invalid_rows(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"config": {}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON"):
        load_result_jsonl(path)


def test_load_result_jsonl_requires_numeric_throughput(tmp_path) -> None:
    path = tmp_path / "missing_metric.jsonl"
    path.write_text(json.dumps({"config": {"id": "cfg"}}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="throughput_tok_s"):
        load_result_jsonl(path)


def _config(config_id: str) -> ServingConfig:
    return ServingConfig(
        id=config_id,
        backend="vllm",
        model_id="model_path",
        dtype="fp16",
        quantization="none",
        max_batch_size=1,
        max_context_tokens=2048,
        kv_cache_policy="paged",
        scheduler="continuous_batching",
    )
