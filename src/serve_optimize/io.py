"""Artifact serialization helpers."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .schemas import BenchmarkResult, ServingConfig, to_dict


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_dict(row), sort_keys=True) + "\n")


def load_result_jsonl(path: Path) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object row in {path} at line {line_number}.")
            results.append(_benchmark_result_from_row(row, line_number=line_number))
    return results


def _benchmark_result_from_row(row: dict[str, Any], *, line_number: int) -> BenchmarkResult:
    config_payload = row.get("config")
    if not isinstance(config_payload, dict):
        raise ValueError(f"Benchmark result row {line_number} is missing config object.")
    config = _serving_config_from_row(config_payload, line_number=line_number)
    throughput = _required_float(row, "throughput_tok_s", "throughput_tokens_per_sec", "tokens_s", line_number=line_number)
    average_power = _optional_float(row, "average_power_watts", "average_power_w", "power_watts", "power_w")
    average_power = 0.0 if average_power is None else average_power
    joules_per_token = _optional_float(row, "joules_per_token", "energy_per_token_j")
    if joules_per_token is None:
        joules_per_token = average_power / throughput if throughput > 0 else 0.0
    tokens_per_watt = _optional_float(row, "tokens_per_watt", "tokens_per_second_per_watt")
    if tokens_per_watt is None:
        tokens_per_watt = throughput / average_power if average_power > 0 else 0.0
    raw = row.get("raw") if isinstance(row.get("raw"), dict) else {}
    return BenchmarkResult(
        config=config,
        throughput_tok_s=throughput,
        average_power_watts=average_power,
        joules_per_token=joules_per_token,
        tokens_per_watt=tokens_per_watt,
        ttft_ms=_optional_float(row, "ttft_ms"),
        p95_latency_ms=_optional_float(row, "p95_latency_ms"),
        peak_power_watts=_optional_float(row, "peak_power_watts", "peak_power_w"),
        total_energy_joules=_optional_float(row, "total_energy_joules", "energy_joules"),
        generated_tokens=_optional_int(row, "generated_tokens", "completion_tokens"),
        feasible=_optional_bool(row, "feasible", default=True),
        reason=_optional_str(row, "reason"),
        raw=dict(raw),
    )


def _serving_config_from_row(row: dict[str, Any], *, line_number: int) -> ServingConfig:
    return ServingConfig(
        id=_optional_str(row, "id", "config_id") or f"row_{line_number}",
        backend=_optional_str(row, "backend") or "unknown",
        model_id=_optional_str(row, "model_id", "model") or "unknown",
        dtype=_optional_str(row, "dtype") or "unknown",
        quantization=_optional_str(row, "quantization") or "none",
        max_batch_size=_optional_int(row, "max_batch_size", "max_num_seqs", "batch_size", "concurrency") or 1,
        max_context_tokens=_optional_int(row, "max_context_tokens", "max_model_len", "context_length") or 1,
        kv_cache_policy=_optional_str(row, "kv_cache_policy") or "unknown",
        scheduler=_optional_str(row, "scheduler") or "unknown",
        tensor_parallelism=_optional_int(row, "tensor_parallelism", "tensor_parallel_size", "tp") or 1,
        gpu_memory_utilization=_optional_float(row, "gpu_memory_utilization", "mem_fraction_static") or 0.9,
        block_size=_optional_int(row, "block_size"),
        kv_cache_dtype=_optional_str(row, "kv_cache_dtype"),
        enforce_eager=_optional_bool(row, "enforce_eager"),
        max_num_batched_tokens=_optional_int(row, "max_num_batched_tokens"),
        enable_chunked_prefill=_optional_bool(row, "enable_chunked_prefill"),
        max_cudagraph_capture_size=_optional_int(row, "max_cudagraph_capture_size"),
        enable_prefix_caching=_optional_bool(row, "enable_prefix_caching"),
        power_limit_watts=_optional_float(row, "power_limit_watts"),
        estimated_vram_mb=_optional_int(row, "estimated_vram_mb"),
        notes=_optional_str_list(row, "notes"),
        extra=_optional_dict(row, "extra"),
    )


def _required_float(row: dict[str, Any], *keys: str, line_number: int) -> float:
    value = _optional_float(row, *keys)
    if value is None:
        joined = ", ".join(keys)
        raise ValueError(f"Benchmark result row {line_number} is missing numeric field: {joined}.")
    return value


def _optional_float(row: dict[str, Any], *keys: str) -> float | None:
    value = _first_present(row, keys)
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(row: dict[str, Any], *keys: str) -> int | None:
    value = _first_present(row, keys)
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(row: dict[str, Any], *keys: str, default: bool | None = None) -> bool | None:
    value = _first_present(row, keys)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return default


def _optional_str(row: dict[str, Any], *keys: str) -> str | None:
    value = _first_present(row, keys)
    if value is None:
        return None
    return str(value)


def _optional_str_list(row: dict[str, Any], key: str) -> list[str]:
    value = row.get(key)
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _optional_dict(row: dict[str, Any], key: str) -> dict[str, Any]:
    value = row.get(key)
    return dict(value) if isinstance(value, dict) else {}


def _first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return None
