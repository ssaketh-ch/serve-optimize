"""Benchmark runner interfaces and a deterministic synthetic simulator."""

from __future__ import annotations

import math
import random

from .schemas import BenchmarkResult, GpuDevice, HardwareSnapshot, ModelSpec, ServingConfig


def run_dry_benchmark(config: ServingConfig, hardware: HardwareSnapshot, model: ModelSpec, seed: int = 7) -> BenchmarkResult:
    """Return deterministic synthetic metrics for CI and offline optimizer validation."""

    gpu = hardware.best_gpu
    rng = random.Random(f"{seed}:{config.id}")
    throughput = _synthetic_throughput(config, model, gpu)
    throughput *= rng.uniform(0.965, 1.035)
    average_power = _synthetic_power(config, gpu, throughput)
    average_power *= rng.uniform(0.98, 1.02)
    joules_per_token = average_power / max(throughput, 1e-6)
    tokens_per_watt = throughput / max(average_power, 1e-6)
    ttft_ms = _synthetic_ttft(config, model)
    p95_latency_ms = ttft_ms + (config.max_context_tokens / max(throughput, 1.0)) * 1000.0 * 0.18
    generated_tokens = 4096

    return BenchmarkResult(
        config=config,
        throughput_tok_s=round(throughput, 3),
        average_power_watts=round(average_power, 3),
        joules_per_token=round(joules_per_token, 6),
        tokens_per_watt=round(tokens_per_watt, 6),
        ttft_ms=round(ttft_ms, 3),
        p95_latency_ms=round(p95_latency_ms, 3),
        peak_power_watts=round(average_power * 1.12, 3),
        total_energy_joules=round(joules_per_token * generated_tokens, 3),
        generated_tokens=generated_tokens,
        raw={"mode": "dry-run", "seed": seed},
    )


def _synthetic_throughput(config: ServingConfig, model: ModelSpec, gpu: GpuDevice | None) -> float:
    gpu_factor = _gpu_factor(gpu)
    model_factor = 7.0 / max(model.parameter_count_b, 0.5)
    batch_factor = math.log2(config.max_batch_size + 1) / math.log2(17)
    context_penalty = 1.0 / (1.0 + max(config.max_context_tokens - 4096, 0) / 32768)
    quant_factor = {
        "none": 1.0,
        "awq-int4": 1.28,
        "gptq-int4": 1.18,
        "bnb-int8": 1.08,
    }.get(config.quantization, 1.0)
    backend_factor = {
        "vllm": 1.0,
        "sglang": 1.05,
        "trt-llm": 1.2,
        "llama.cpp": 0.48,
        "dry-run": 0.92,
    }.get(config.backend, 1.0)
    dtype_factor = 1.04 if config.dtype == "bf16" else 1.0
    power_factor = 1.0
    if config.power_limit_watts and gpu and gpu.power_limit_watts:
        cap_ratio = min(config.power_limit_watts / gpu.power_limit_watts, 1.0)
        power_factor = 0.78 + 0.22 * cap_ratio
    return max(5.0, 1900.0 * gpu_factor * model_factor * batch_factor * context_penalty * quant_factor * backend_factor * dtype_factor * power_factor)


def _synthetic_power(config: ServingConfig, gpu: GpuDevice | None, throughput: float) -> float:
    default_limit = _default_power_limit(gpu)
    limit = config.power_limit_watts or default_limit
    quant_power_factor = {
        "none": 1.0,
        "awq-int4": 0.86,
        "gptq-int4": 0.88,
        "bnb-int8": 0.92,
    }.get(config.quantization, 1.0)
    utilization = min(0.96, 0.42 + math.log10(max(throughput, 10.0)) / 6.0)
    idle = max(18.0, limit * 0.18)
    active = idle + (limit - idle) * utilization * quant_power_factor
    return min(limit, max(idle, active))


def _synthetic_ttft(config: ServingConfig, model: ModelSpec) -> float:
    quant_penalty = 1.08 if config.quantization != "none" else 1.0
    return (85.0 + model.parameter_count_b * 9.0 + config.max_context_tokens * 0.018) * quant_penalty


def _gpu_factor(gpu: GpuDevice | None) -> float:
    if not gpu:
        return 0.08
    name = gpu.name.upper()
    memory = gpu.total_memory_mb or 0
    if "H200" in name:
        return 1.35 if memory < 100_000 else 1.75
    if "H100" in name:
        return 1.45
    if "A100" in name:
        return 1.05
    if "RTX PRO 6000" in name or "6000" in name:
        return 0.92
    if "4090" in name:
        return 0.78
    if "3090" in name:
        return 0.48
    if "1660" in name:
        return 0.12
    return min(0.8, max(0.1, memory / 80_000))


def _default_power_limit(gpu: GpuDevice | None) -> float:
    if gpu and gpu.power_limit_watts:
        return gpu.power_limit_watts
    if not gpu:
        return 65.0
    name = gpu.name.upper()
    if "H200" in name:
        return 700.0
    if "H100" in name:
        return 700.0
    if "RTX PRO 6000" in name:
        return 600.0
    if "4090" in name:
        return 450.0
    if "1660" in name:
        return 120.0
    return 300.0
