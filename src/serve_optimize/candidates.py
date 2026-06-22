"""Heuristic candidate generation for early research iterations."""

from __future__ import annotations

import hashlib
from itertools import product

from .schemas import Backend, Goal, HardwareSnapshot, ModelSpec, ServingConfig

DTYPE_BYTES = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
}

QUANT_WEIGHT_BYTES = {
    "none": None,
    "awq-int4": 0.58,
    "gptq-int4": 0.58,
    "bnb-int8": 1.05,
}


def generate_candidates(
    hardware: HardwareSnapshot,
    model: ModelSpec,
    goal: Goal = Goal.BALANCED,
    include_experimental: bool = False,
    limit: int = 48,
) -> list[ServingConfig]:
    gpu = hardware.best_gpu
    visible_vram_mb = gpu.total_memory_mb if gpu else None
    backends = _candidate_backends(gpu_name=gpu.name if gpu else None, has_gpu=bool(gpu), include_experimental=include_experimental)
    dtypes = _candidate_dtypes(gpu.name if gpu else "")
    quantizations = _candidate_quantizations(model, visible_vram_mb)
    batch_sizes = _candidate_batch_sizes(visible_vram_mb, model, goal)
    contexts = _candidate_contexts(model, visible_vram_mb)
    power_limits = _candidate_power_limits(gpu.power_limit_watts if gpu else None, goal)

    candidates: list[ServingConfig] = []
    for backend, dtype, quant, batch_size, context, power_limit in product(
        backends,
        dtypes,
        quantizations,
        batch_sizes,
        contexts,
        power_limits,
    ):
        estimated_vram = estimate_vram_mb(model, dtype, quant, batch_size, context)
        if visible_vram_mb and estimated_vram > visible_vram_mb * 0.94:
            continue
        if backend == Backend.TRT_LLM.value and quant == "bnb-int8":
            continue
        if backend == Backend.LLAMA_CPP.value and dtype == "bf16":
            continue

        config = ServingConfig(
            id=_config_id(backend, model.model_id, dtype, quant, batch_size, context, power_limit),
            backend=backend,
            model_id=model.model_id,
            dtype=dtype,
            quantization=quant,
            max_batch_size=batch_size,
            max_context_tokens=context,
            kv_cache_policy="paged" if backend in {Backend.VLLM.value, Backend.SGLANG.value, Backend.DRY_RUN.value} else "static",
            scheduler="continuous-batching" if backend in {Backend.VLLM.value, Backend.SGLANG.value} else "backend-default",
            tensor_parallelism=1,
            gpu_memory_utilization=0.9,
            power_limit_watts=power_limit,
            estimated_vram_mb=estimated_vram,
            notes=_notes_for_config(backend, quant, power_limit, gpu.name if gpu else None),
        )
        candidates.append(config)

    return _rank_candidates(candidates, goal)[:limit]


def estimate_vram_mb(
    model: ModelSpec,
    dtype: str,
    quantization: str,
    batch_size: int,
    context_tokens: int,
) -> int:
    weight_bytes = QUANT_WEIGHT_BYTES.get(quantization)
    if weight_bytes is None:
        weight_bytes = DTYPE_BYTES.get(dtype, 2.0)

    weights_mb = model.parameter_count * weight_bytes / (1024 * 1024)
    # Early heuristic: KV footprint scales with params, batch, and context.
    # This is intentionally conservative until model-specific hidden sizes are added.
    kv_mb = model.parameter_count_b * batch_size * context_tokens * 0.0018
    runtime_overhead_mb = max(1024.0, weights_mb * 0.12)
    return int(weights_mb + kv_mb + runtime_overhead_mb)


def _candidate_backends(gpu_name: str | None, has_gpu: bool, include_experimental: bool) -> list[str]:
    if not has_gpu:
        return [Backend.TRANSFORMERS.value, Backend.DRY_RUN.value]
    backends = [Backend.TRANSFORMERS.value, Backend.VLLM.value, Backend.SGLANG.value, Backend.DRY_RUN.value]
    if include_experimental:
        backends.append(Backend.LLAMA_CPP.value)
        if gpu_name and any(token in gpu_name.upper() for token in ("H100", "H200", "A100", "L40", "RTX")):
            backends.append(Backend.TRT_LLM.value)
    return backends


def _candidate_dtypes(gpu_name: str) -> list[str]:
    upper = gpu_name.upper()
    if any(token in upper for token in ("H100", "H200", "A100", "A800", "B100", "B200", "RTX PRO", "4090", "6000")):
        return ["bf16", "fp16"]
    return ["fp16"]


def _candidate_quantizations(model: ModelSpec, visible_vram_mb: int | None) -> list[str]:
    quantizations = ["none", "awq-int4", "gptq-int4", "bnb-int8"]
    if visible_vram_mb is None:
        return ["awq-int4", "none"]
    if model.parameter_count_b >= 20 and visible_vram_mb < 48_000:
        return ["awq-int4", "gptq-int4", "bnb-int8"]
    if model.parameter_count_b >= 7 and visible_vram_mb < 16_000:
        return ["awq-int4", "gptq-int4"]
    return quantizations


def _candidate_batch_sizes(visible_vram_mb: int | None, model: ModelSpec, goal: Goal) -> list[int]:
    if not visible_vram_mb:
        return [1, 2, 4, 8]
    if visible_vram_mb < 12_000:
        base = [1, 2, 4]
    elif visible_vram_mb < 32_000:
        base = [1, 2, 4, 8]
    else:
        base = [1, 2, 4, 8, 16, 32]
    if goal == Goal.EFFICIENT:
        return [item for item in base if item <= 8]
    if goal == Goal.PERFORMANCE:
        return base[-4:]
    if model.parameter_count_b >= 30:
        return [item for item in base if item <= 16]
    return base


def _candidate_contexts(model: ModelSpec, visible_vram_mb: int | None) -> list[int]:
    upper = model.max_context_tokens
    contexts = [2048, 4096, 8192, 16384, 32768]
    if visible_vram_mb and visible_vram_mb < 16_000:
        contexts = [2048, 4096]
    return [context for context in contexts if context <= upper]


def _candidate_power_limits(power_limit_watts: float | None, goal: Goal) -> list[float | None]:
    if not power_limit_watts:
        return [None]
    if goal == Goal.PERFORMANCE:
        return [None, round(power_limit_watts, 1)]
    if goal == Goal.EFFICIENT:
        return [round(power_limit_watts * factor, 1) for factor in (0.7, 0.8, 0.9)]
    return [None, round(power_limit_watts * 0.8, 1), round(power_limit_watts * 0.9, 1)]


def _rank_candidates(candidates: list[ServingConfig], goal: Goal) -> list[ServingConfig]:
    def key(config: ServingConfig) -> tuple[float, int, int]:
        quant_bonus = 1 if config.quantization != "none" else 0
        if goal == Goal.PERFORMANCE:
            return (config.max_batch_size, config.max_context_tokens, -quant_bonus)
        if goal == Goal.EFFICIENT:
            return (quant_bonus, -(config.power_limit_watts or 9999), -config.max_batch_size)
        return (quant_bonus, config.max_batch_size, -(config.power_limit_watts or 0))

    return sorted(candidates, key=key, reverse=True)


def _config_id(*parts: object) -> str:
    payload = "|".join(str(part) for part in parts)
    digest = hashlib.sha1(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:10]
    return f"cfg-{digest}"


def _notes_for_config(backend: str, quantization: str, power_limit: float | None, gpu_name: str | None) -> list[str]:
    notes: list[str] = []
    if backend == Backend.DRY_RUN.value:
        notes.append("Synthetic candidate for CI, smoke tests, and offline optimizer validation.")
    if backend == Backend.TRANSFORMERS.value:
        notes.append("Reference backend for functional smoke tests and correctness checks.")
    if quantization != "none":
        notes.append("Requires a compatible quantized checkpoint or backend quantization path.")
    if power_limit:
        notes.append("Power limit requires permission to set GPU power management controls.")
    if gpu_name and "1660" in gpu_name:
        notes.append("GTX 1660-class GPUs are useful for detection and telemetry validation but not target LLM serving.")
    return notes
