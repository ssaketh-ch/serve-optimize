"""Capability-aware candidate generation for Managed Evaluation Mode."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field, replace

from .backends.sglang import SGLangArgumentCapabilities
from .backends.vllm import VLLMArgumentCapabilities
from .candidates import estimate_vram_mb, generate_candidates
from .modeling import infer_model_spec
from .schemas import Goal, HardwareSnapshot, ModelCapabilityMetadata, ModelSpec, ServingConfig, WorkloadProfile, to_dict
from .validation import normalize_quantization


@dataclass(frozen=True)
class CapabilityContext:
    backend: str
    model: str
    goal: Goal
    hardware: HardwareSnapshot
    model_spec: ModelSpec | None = None
    model_metadata: ModelCapabilityMetadata | None = None
    backend_metadata: dict[str, object] = field(default_factory=dict)
    telemetry_capabilities: dict[str, object] = field(default_factory=dict)
    vllm_argument_capabilities: VLLMArgumentCapabilities | None = None
    sglang_argument_capabilities: SGLangArgumentCapabilities | None = None
    workload_profile: WorkloadProfile | None = None
    managed_mode: bool = True


@dataclass(frozen=True)
class ManagedCandidateGenerationResult:
    candidates: list[ServingConfig]
    candidate_source_counts: dict[str, int]
    capability_filtered_count: int = 0
    invalid_quantization_filtered_count: int = 0
    safe_baseline_added: bool = False


def generate_managed_candidates_from_capabilities(
    context: CapabilityContext,
    *,
    limit: int,
) -> ManagedCandidateGenerationResult:
    """Generate a compact managed candidate pool from known model and runtime capabilities."""

    model_spec = context.model_spec or infer_model_spec(context.model)
    metadata = context.model_metadata or ModelCapabilityMetadata(model_id=context.model)
    if context.backend == "sglang":
        return _generate_sglang_candidates(context, model_spec, metadata, limit=limit)
    allowed_quantizations = _allowed_quantizations(metadata)
    dtype = _native_dtype(metadata) or _hardware_default_dtype(context.hardware)
    workload_profile = context.workload_profile or WorkloadProfile()

    candidates: list[ServingConfig] = []
    baseline = _candidate(
        context=context,
        model_spec=model_spec,
        dtype=dtype,
        quantization=allowed_quantizations[0],
        max_context_tokens=_bounded_context(model_spec, 2048),
        max_batch_size=1,
        gpu_memory_utilization=0.9,
        workload_concurrency=1,
        source="safe_baseline",
        baseline=True,
        engine_options={"backend_defaults": True},
    )
    candidates.append(baseline)

    for quantization in allowed_quantizations:
        for max_context_tokens, max_batch_size, gpu_memory_utilization, workload_concurrency, engine_options in _candidate_shapes(
            model_spec,
            context.goal,
            dtype,
            context.vllm_argument_capabilities,
            workload_profile,
        ):
            candidates.append(
                _candidate(
                    context=context,
                    model_spec=model_spec,
                    dtype=dtype,
                    quantization=quantization,
                    max_context_tokens=max_context_tokens,
                    max_batch_size=max_batch_size,
                    gpu_memory_utilization=gpu_memory_utilization,
                    workload_concurrency=workload_concurrency,
                    source="capability_aware",
                    baseline=False,
                    engine_options=engine_options,
                )
            )

    legacy_candidates = generate_candidates(
        context.hardware,
        model_spec,
        goal=context.goal,
        limit=max(96, max(1, limit) * 24),
    )
    capability_filtered_count = 0
    invalid_quantization_filtered_count = 0
    for config in legacy_candidates:
        if config.backend != context.backend:
            capability_filtered_count += 1
            continue
        if normalize_quantization(config.quantization) not in {normalize_quantization(item) for item in allowed_quantizations}:
            capability_filtered_count += 1
            invalid_quantization_filtered_count += 1
            continue
        candidates.append(_from_legacy(config, source="legacy_filtered"))

    candidates = _dedupe_candidates(candidates)
    selected = candidates[: max(1, limit)]
    source_counts = Counter(str((config.extra or {}).get("candidate_source") or "unknown") for config in selected)
    return ManagedCandidateGenerationResult(
        candidates=selected,
        candidate_source_counts=dict(sorted(source_counts.items())),
        capability_filtered_count=capability_filtered_count,
        invalid_quantization_filtered_count=invalid_quantization_filtered_count,
        safe_baseline_added=bool(selected and selected[0].extra.get("candidate_source") == "safe_baseline"),
    )


def _generate_sglang_candidates(
    context: CapabilityContext,
    model_spec: ModelSpec,
    metadata: ModelCapabilityMetadata,
    *,
    limit: int,
) -> ManagedCandidateGenerationResult:
    dtype = _sglang_dtype(_native_dtype(metadata) or _hardware_default_dtype(context.hardware), context.sglang_argument_capabilities)
    quantization = _allowed_quantizations(metadata)[0]
    capabilities = context.sglang_argument_capabilities
    engine_options: dict[str, object] = {}
    if (
        capabilities is not None
        and capabilities.detection_status == "success"
        and capabilities.supports("--disable-piecewise-cuda-graph")
    ):
        engine_options["disable_piecewise_cuda_graph"] = True
    memory_fraction_supported = bool(
        capabilities is not None
        and capabilities.detection_status == "success"
        and capabilities.supports("--mem-fraction-static")
    )
    running_requests_supported = bool(
        capabilities is not None
        and capabilities.detection_status == "success"
        and capabilities.supports("--max-running-requests")
    )
    chunked_prefill_supported = bool(
        capabilities is not None
        and capabilities.detection_status == "success"
        and capabilities.supports("--chunked-prefill-size")
    )
    contexts = _unique_positive([_bounded_context(model_spec, item) for item in (2048, 4096)])
    primary_context = contexts[0]
    secondary_context = contexts[1] if len(contexts) > 1 else primary_context
    candidates = [
        _candidate(
            context=context,
            model_spec=model_spec,
            dtype=dtype,
            quantization=quantization,
            max_context_tokens=primary_context,
            max_batch_size=1,
            gpu_memory_utilization=0.8 if memory_fraction_supported else 0.0,
            workload_concurrency=1,
            source="safe_baseline",
            baseline=True,
            engine_options={"backend_defaults": True},
        )
    ]
    candidate_batch_size = 4 if running_requests_supported else 1
    shapes = [
        (primary_context, candidate_batch_size, 0.8, 2, chunked_prefill_supported),
        (secondary_context, 1, 0.9, 1, False),
        (secondary_context, candidate_batch_size, 0.9, 4, chunked_prefill_supported),
    ]
    if context.goal == Goal.EFFICIENT:
        shapes = shapes[:2]
    elif context.goal == Goal.PERFORMANCE:
        shapes = [shapes[0], shapes[-1]]
    for max_context_tokens, max_batch_size, memory_fraction, workload_concurrency, use_chunked_prefill in shapes:
        shape_options = dict(engine_options)
        if use_chunked_prefill:
            shape_options["chunked_prefill_size"] = primary_context
        candidates.append(
            _candidate(
                context=context,
                model_spec=model_spec,
                dtype=dtype,
                quantization=quantization,
                max_context_tokens=max_context_tokens,
                max_batch_size=max_batch_size,
                gpu_memory_utilization=memory_fraction if memory_fraction_supported else 0.0,
                workload_concurrency=workload_concurrency,
                source="sglang_capability_aware",
                baseline=False,
                engine_options=shape_options,
            )
        )
    if _profile_payload(context):
        candidates = [
            _with_profile(config, _profile_payload(context))
            for config in candidates
        ]
    selected = _dedupe_candidates(candidates)[: max(1, min(limit, 4))]
    source_counts = Counter(str((config.extra or {}).get("candidate_source") or "unknown") for config in selected)
    return ManagedCandidateGenerationResult(
        candidates=selected,
        candidate_source_counts=dict(sorted(source_counts.items())),
        safe_baseline_added=bool(selected and selected[0].extra.get("candidate_source") == "safe_baseline"),
    )


def _allowed_quantizations(metadata: ModelCapabilityMetadata) -> list[str]:
    quantization = normalize_quantization(metadata.quantization_method)
    if quantization == "awq":
        return ["awq-int4"]
    if quantization == "gptq":
        return ["gptq-int4"]
    return ["none"]


def _native_dtype(metadata: ModelCapabilityMetadata) -> str | None:
    if metadata.torch_dtype is None:
        return None
    normalized = metadata.torch_dtype.strip().lower().removeprefix("torch.")
    return {
        "bfloat16": "bf16",
        "bf16": "bf16",
        "float16": "fp16",
        "fp16": "fp16",
        "half": "fp16",
        "float32": "fp32",
        "fp32": "fp32",
    }.get(normalized)


def _hardware_default_dtype(hardware: HardwareSnapshot) -> str:
    del hardware
    return "fp16"


def _candidate_shapes(
    model_spec: ModelSpec,
    goal: Goal,
    dtype: str,
    capabilities: VLLMArgumentCapabilities | None,
    workload_profile: WorkloadProfile,
) -> list[tuple[int, int, float, int, dict[str, object]]]:
    contexts = [_bounded_context(model_spec, item) for item in (2048, 4096, 8192)]
    contexts = _unique_positive(contexts)
    primary_context = contexts[0]
    secondary_context = contexts[1] if len(contexts) > 1 else contexts[0]
    tertiary_context = contexts[2] if len(contexts) > 2 else secondary_context
    dtype_option = _kv_cache_dtype_for_model_dtype(dtype)

    shapes: list[tuple[int, int, float, int, dict[str, object]]] = [
        (primary_context, 1, 0.7, 1, {}),
        (
            primary_context,
            4,
            0.8,
            2,
            {
                "max_num_batched_tokens": _batched_tokens_for(primary_context, multiplier=2),
                "enable_chunked_prefill": True,
            },
        ),
        (
            primary_context,
            4,
            0.9,
            4,
            {
                "block_size": 16,
                "max_cudagraph_capture_size": 32,
            },
        ),
        (
            secondary_context,
            4,
            0.8,
            2,
            {
                "max_num_batched_tokens": _batched_tokens_for(secondary_context, multiplier=2),
                "enable_chunked_prefill": False,
            },
        ),
        (
            secondary_context,
            8,
            0.8,
            4,
            {
                "max_num_batched_tokens": _batched_tokens_for(secondary_context, multiplier=2),
                "enable_chunked_prefill": True,
                **({"kv_cache_dtype": dtype_option} if dtype_option else {}),
            },
        ),
        (
            tertiary_context,
            8,
            0.9,
            8,
            {
                "max_num_batched_tokens": _batched_tokens_for(tertiary_context, multiplier=1),
                "enable_chunked_prefill": True,
                "block_size": 16,
            },
        ),
        (primary_context, 1, 0.8, 1, {"enforce_eager": True}),
    ]
    shapes = [
        (context, batch_size, memory, workload, _filter_engine_options(options, capabilities))
        for context, batch_size, memory, workload, options in shapes
    ]
    if _prefix_cache_profile(workload_profile) and capabilities is not None and capabilities.supports("--enable-prefix-caching"):
        shapes = _with_prefix_cache_shape(shapes)
    if goal == Goal.EFFICIENT:
        shapes = [shape for shape in shapes if shape[2] <= 0.8 and shape[1] <= 8]
    elif goal == Goal.PERFORMANCE:
        shapes = [shape for shape in shapes if shape[1] >= 4 or shape[4].get("enforce_eager") is True]
    return shapes


def _filter_engine_options(options: dict[str, object], capabilities: VLLMArgumentCapabilities | None) -> dict[str, object]:
    if not options or capabilities is None or capabilities.detection_status != "success":
        return {}
    filtered: dict[str, object] = {}
    if "block_size" in options and capabilities.supports("--block-size"):
        filtered["block_size"] = options["block_size"]
    if "max_num_batched_tokens" in options and capabilities.supports("--max-num-batched-tokens"):
        filtered["max_num_batched_tokens"] = options["max_num_batched_tokens"]
    if options.get("enable_chunked_prefill") is True and capabilities.supports("--enable-chunked-prefill"):
        filtered["enable_chunked_prefill"] = True
    if options.get("enable_chunked_prefill") is False and capabilities.supports("--no-enable-chunked-prefill"):
        filtered["enable_chunked_prefill"] = False
    if "kv_cache_dtype" in options and _kv_cache_dtype_allowed(str(options["kv_cache_dtype"]), capabilities):
        filtered["kv_cache_dtype"] = options["kv_cache_dtype"]
    if options.get("enforce_eager") is True and capabilities.supports("--enforce-eager"):
        filtered["enforce_eager"] = True
    if "max_cudagraph_capture_size" in options and capabilities.cudagraph_capture_flag() is not None:
        filtered["max_cudagraph_capture_size"] = options["max_cudagraph_capture_size"]
    if options.get("enable_prefix_caching") is True and capabilities.supports("--enable-prefix-caching"):
        filtered["enable_prefix_caching"] = True
    if options.get("enable_prefix_caching") is False and capabilities.supports("--no-enable-prefix-caching"):
        filtered["enable_prefix_caching"] = False
    return filtered


def _prefix_cache_profile(profile: WorkloadProfile) -> bool:
    return profile.profile_name in {"repeated-prefix", "repeated_prefix"} or profile.prefix_reuse_expected is True


def _with_prefix_cache_shape(
    shapes: list[tuple[int, int, float, int, dict[str, object]]],
) -> list[tuple[int, int, float, int, dict[str, object]]]:
    updated = list(shapes)
    for index, (context, batch_size, memory, workload, options) in enumerate(updated):
        if index == 0:
            continue
        next_options = dict(options)
        next_options["enable_prefix_caching"] = True
        updated[index] = (context, batch_size, memory, workload, next_options)
        break
    return updated


def _profile_payload(context: CapabilityContext) -> dict[str, object]:
    profile = context.workload_profile
    if profile is None or profile.profile_name == "default":
        return {}
    return to_dict(profile)


def _kv_cache_dtype_allowed(value: str, capabilities: VLLMArgumentCapabilities) -> bool:
    if not capabilities.supports("--kv-cache-dtype"):
        return False
    choices = capabilities.choices_for("--kv-cache-dtype")
    return bool(choices and value in choices)


def _sglang_dtype(dtype: str, capabilities: SGLangArgumentCapabilities | None) -> str:
    rendered = {
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
    }.get(dtype, dtype)
    if capabilities is None or capabilities.detection_status != "success" or not capabilities.supports("--dtype"):
        return dtype
    choices = capabilities.choices_for("--dtype")
    if not choices or rendered in choices:
        return dtype
    if dtype in choices:
        return dtype
    return "fp16" if "float16" in choices or "fp16" in choices else dtype


def _with_profile(config: ServingConfig, profile_payload: dict[str, object]) -> ServingConfig:
    extra = dict(config.extra or {})
    extra["workload_profile"] = profile_payload
    return replace(config, extra=extra)


def _candidate(
    *,
    context: CapabilityContext,
    model_spec: ModelSpec,
    dtype: str,
    quantization: str,
    max_context_tokens: int,
    max_batch_size: int,
    gpu_memory_utilization: float,
    workload_concurrency: int,
    source: str,
    baseline: bool,
    engine_options: dict[str, object] | None = None,
) -> ServingConfig:
    engine_options = dict(engine_options or {})
    extra = {
        "candidate_source": source,
        "model_native": normalize_quantization(quantization) == "none",
        "workload_concurrency": workload_concurrency,
        "max_new_tokens": 128,
    }
    profile_payload = _profile_payload(context)
    if profile_payload:
        extra["workload_profile"] = profile_payload
    for field_name in (
        "disable_piecewise_cuda_graph",
        "disable_radix_cache",
        "disable_cuda_graph",
        "chunked_prefill_size",
        "cuda_graph_max_bs",
        "backend_defaults",
    ):
        if field_name in engine_options:
            extra[field_name] = engine_options[field_name]
    notes = ["Generated from managed capability context before validation."]
    if baseline:
        extra["baseline"] = True
        notes = ["Backend default baseline inserted before managed validation."]
    config_id = _config_id(
        context.backend,
        context.model,
        dtype,
        quantization,
        max_batch_size,
        max_context_tokens,
        gpu_memory_utilization,
        workload_concurrency,
        tuple(sorted(engine_options.items())),
        source,
    )
    return ServingConfig(
        id=config_id,
        backend=context.backend,
        model_id=context.model,
        dtype=dtype,
        quantization=quantization,
        max_batch_size=max_batch_size,
        max_context_tokens=max_context_tokens,
        kv_cache_policy="paged" if context.backend == "vllm" else "backend-default",
        scheduler="continuous-batching" if context.backend == "vllm" else "backend-default",
        tensor_parallelism=1,
        gpu_memory_utilization=gpu_memory_utilization,
        block_size=_optional_int(engine_options.get("block_size")),
        kv_cache_dtype=_optional_str(engine_options.get("kv_cache_dtype")),
        enforce_eager=_optional_bool(engine_options.get("enforce_eager")),
        max_num_batched_tokens=_optional_int(engine_options.get("max_num_batched_tokens")),
        enable_chunked_prefill=_optional_bool(engine_options.get("enable_chunked_prefill")),
        max_cudagraph_capture_size=_optional_int(engine_options.get("max_cudagraph_capture_size")),
        enable_prefix_caching=_optional_bool(engine_options.get("enable_prefix_caching")),
        estimated_vram_mb=estimate_vram_mb(model_spec, dtype, quantization, max_batch_size, max_context_tokens),
        notes=notes,
        extra=extra,
    )


def _from_legacy(config: ServingConfig, *, source: str) -> ServingConfig:
    extra = dict(config.extra or {})
    extra.setdefault("candidate_source", source)
    extra.setdefault("model_native", normalize_quantization(config.quantization) == "none")
    extra.setdefault("workload_concurrency", max(1, config.max_batch_size))
    extra.setdefault("max_new_tokens", 128)
    notes = list(config.notes)
    notes.append("Legacy generated candidate retained after capability filtering.")
    return ServingConfig(
        id=f"{config.id}-managed",
        backend=config.backend,
        model_id=config.model_id,
        dtype=config.dtype,
        quantization=config.quantization,
        max_batch_size=config.max_batch_size,
        max_context_tokens=config.max_context_tokens,
        kv_cache_policy=config.kv_cache_policy,
        scheduler=config.scheduler,
        tensor_parallelism=config.tensor_parallelism,
        gpu_memory_utilization=config.gpu_memory_utilization,
        block_size=config.block_size,
        kv_cache_dtype=config.kv_cache_dtype,
        enforce_eager=config.enforce_eager,
        max_num_batched_tokens=config.max_num_batched_tokens,
        enable_chunked_prefill=config.enable_chunked_prefill,
        max_cudagraph_capture_size=config.max_cudagraph_capture_size,
        enable_prefix_caching=config.enable_prefix_caching,
        power_limit_watts=config.power_limit_watts,
        estimated_vram_mb=config.estimated_vram_mb,
        notes=notes,
        extra=extra,
    )


def _dedupe_candidates(candidates: list[ServingConfig]) -> list[ServingConfig]:
    seen_ids: set[str] = set()
    seen_shapes: set[tuple[object, ...]] = set()
    result: list[ServingConfig] = []
    for config in candidates:
        extra = config.extra or {}
        shape = (
            config.backend,
            config.model_id,
            config.dtype,
            normalize_quantization(config.quantization),
            config.max_batch_size,
            config.max_context_tokens,
            config.gpu_memory_utilization,
            config.tensor_parallelism,
            config.block_size,
            config.kv_cache_dtype,
            config.enforce_eager,
            config.max_num_batched_tokens,
            config.enable_chunked_prefill,
            config.max_cudagraph_capture_size,
            config.enable_prefix_caching,
            extra.get("disable_piecewise_cuda_graph"),
            extra.get("workload_concurrency"),
        )
        if config.id in seen_ids or shape in seen_shapes:
            continue
        seen_ids.add(config.id)
        seen_shapes.add(shape)
        result.append(config)
    return result


def _bounded_context(model_spec: ModelSpec, requested: int) -> int:
    return max(1, min(model_spec.max_context_tokens, requested))


def _unique_positive(values: list[int]) -> list[int]:
    result: list[int] = []
    for value in values:
        if value > 0 and value not in result:
            result.append(value)
    return result


def _batched_tokens_for(max_context_tokens: int, *, multiplier: int) -> int:
    lower_bound = max(max_context_tokens + 1, 2048)
    target = max(max_context_tokens * multiplier, 4096)
    if max_context_tokens <= 4096:
        target = max(target, 4096)
    else:
        target = max(target, 8192)
    return max(lower_bound, target)


def _kv_cache_dtype_for_model_dtype(dtype: str) -> str | None:
    return {
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp16": "float16",
        "float16": "float16",
    }.get(dtype)


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _config_id(*parts: object) -> str:
    payload = "|".join(str(part) for part in parts)
    digest = hashlib.sha1(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:10]
    return f"cfg-managed-{digest}"
