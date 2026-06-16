"""Candidate validation helpers shared by managed evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from .backends.sglang import SGLANG_UNSUPPORTED_VLLM_FIELDS, SGLangArgumentCapabilities
from .backends.vllm import VLLMArgumentCapabilities
from .schemas import ModelCapabilityMetadata, ServingConfig


@dataclass(frozen=True)
class CandidateValidationResult:
    config_id: str
    valid: bool
    reason: str | None = None


QUANTIZATION_ALIASES = {
    "": "none",
    "none": "none",
    "null": "none",
    "awq": "awq",
    "awq-int4": "awq",
    "gptq": "gptq",
    "gptq-int4": "gptq",
}

SUPPORTED_KV_CACHE_DTYPES = {"auto", "bfloat16", "float16"}


def normalize_quantization(quantization: object) -> str:
    if quantization is None:
        return "none"
    text = str(quantization).strip().lower()
    return QUANTIZATION_ALIASES.get(text, text)


def validate_quantization_compatibility(
    config: ServingConfig,
    model_metadata: ModelCapabilityMetadata,
    *,
    managed_mode: bool = True,
) -> CandidateValidationResult:
    quantization = normalize_quantization(config.quantization)
    if quantization == "none":
        return CandidateValidationResult(config_id=config.id, valid=True)
    if quantization in {"awq", "gptq"}:
        if not model_metadata.metadata_known:
            return CandidateValidationResult(
                config_id=config.id,
                valid=False,
                reason=f"quantization {quantization} requires model config quantization_config.quant_method={quantization}",
            )
        if normalize_quantization(model_metadata.quantization_method) == quantization:
            return CandidateValidationResult(config_id=config.id, valid=True)
        return CandidateValidationResult(
            config_id=config.id,
            valid=False,
            reason=f"quantization {quantization} requires model config quantization_config.quant_method={quantization}",
        )
    if managed_mode:
        return CandidateValidationResult(
            config_id=config.id,
            valid=False,
            reason=f"quantization {config.quantization} is not supported by managed validation without explicit compatible model metadata",
        )
    return CandidateValidationResult(config_id=config.id, valid=True)


def validate_managed_candidate(
    config: ServingConfig,
    *,
    backend: str,
    model_metadata: ModelCapabilityMetadata,
    vllm_argument_capabilities: VLLMArgumentCapabilities | None = None,
    sglang_argument_capabilities: SGLangArgumentCapabilities | None = None,
) -> CandidateValidationResult:
    if config.backend != backend:
        return CandidateValidationResult(
            config_id=config.id,
            valid=False,
            reason=f"Candidate backend '{config.backend}' is not supported by managed backend '{backend}'.",
        )
    quantization = validate_quantization_compatibility(config, model_metadata, managed_mode=True)
    if not quantization.valid:
        return quantization
    if backend == "sglang":
        return validate_sglang_engine_options(config, sglang_argument_capabilities=sglang_argument_capabilities)
    return validate_engine_options(config, vllm_argument_capabilities=vllm_argument_capabilities)


def validate_sglang_engine_options(
    config: ServingConfig,
    *,
    sglang_argument_capabilities: SGLangArgumentCapabilities | None = None,
) -> CandidateValidationResult:
    if not config.model_id.strip():
        return CandidateValidationResult(
            config_id=config.id,
            valid=False,
            reason="model_id must not be empty.",
        )
    if config.max_context_tokens <= 0:
        return CandidateValidationResult(
            config_id=config.id,
            valid=False,
            reason="max_context_tokens must be greater than zero.",
        )
    if config.max_batch_size <= 0:
        return CandidateValidationResult(config_id=config.id, valid=False, reason="max_batch_size must be greater than zero.")
    if config.tensor_parallelism <= 0:
        return CandidateValidationResult(
            config_id=config.id,
            valid=False,
            reason="tensor_parallelism must be greater than zero.",
        )
    if not 0.0 <= config.gpu_memory_utilization <= 1.0:
        return CandidateValidationResult(
            config_id=config.id,
            valid=False,
            reason="gpu_memory_utilization must be between 0.0 and 1.0.",
        )

    extra = config.extra or {}
    boolean_options = (
        "disable_piecewise_cuda_graph",
        "disable_radix_cache",
        "disable_cuda_graph",
        "trust_remote_code",
    )
    for field_name in boolean_options:
        value = extra.get(field_name)
        if value is not None and not isinstance(value, bool):
            return CandidateValidationResult(
                config_id=config.id,
                valid=False,
                reason=f"{field_name} must be a boolean.",
            )
    served_model_name = extra.get("served_model_name")
    if served_model_name is not None and (
        not isinstance(served_model_name, str) or not served_model_name.strip()
    ):
        return CandidateValidationResult(
            config_id=config.id,
            valid=False,
            reason="served_model_name must be a non-empty string.",
        )
    chunked_prefill_size = extra.get("chunked_prefill_size")
    if chunked_prefill_size is not None and (
        not isinstance(chunked_prefill_size, int) or isinstance(chunked_prefill_size, bool) or chunked_prefill_size == 0 or chunked_prefill_size < -1
    ):
        return CandidateValidationResult(
            config_id=config.id,
            valid=False,
            reason="chunked_prefill_size must be -1 or a positive integer.",
        )
    cuda_graph_max_bs = extra.get("cuda_graph_max_bs")
    if cuda_graph_max_bs is not None and (
        not isinstance(cuda_graph_max_bs, int) or isinstance(cuda_graph_max_bs, bool) or cuda_graph_max_bs <= 0
    ):
        return CandidateValidationResult(
            config_id=config.id,
            valid=False,
            reason="cuda_graph_max_bs must be a positive integer.",
        )
    if extra.get("disable_cuda_graph") is True and cuda_graph_max_bs is not None:
        return CandidateValidationResult(
            config_id=config.id,
            valid=False,
            reason="disable_cuda_graph=true cannot be combined with cuda_graph_max_bs.",
        )

    for field_name in SGLANG_UNSUPPORTED_VLLM_FIELDS:
        if getattr(config, field_name) is not None:
            return CandidateValidationResult(
                config_id=config.id,
                valid=False,
                reason=f"{field_name} is a vLLM field without a direct SGLang translation.",
            )
    if sglang_argument_capabilities is not None and sglang_argument_capabilities.detection_status == "success":
        if not sglang_argument_capabilities.supports("--model-path"):
            return CandidateValidationResult(
                config_id=config.id,
                valid=False,
                reason="SGLang launch requires detected support for --model-path.",
            )
        if sglang_argument_capabilities.context_length_flag() is None:
            return CandidateValidationResult(
                config_id=config.id,
                valid=False,
                reason="SGLang launch requires a detected context length flag.",
            )
        if config.dtype and not sglang_argument_capabilities.supports("--dtype"):
            return CandidateValidationResult(
                config_id=config.id,
                valid=False,
                reason="dtype requires detected SGLang support for --dtype.",
            )
        if config.dtype:
            dtype = _sglang_dtype(config.dtype)
            choices = sglang_argument_capabilities.choices_for("--dtype")
            if choices and dtype not in choices and config.dtype not in choices:
                return CandidateValidationResult(
                    config_id=config.id,
                    valid=False,
                    reason=f"dtype '{config.dtype}' is not listed by installed SGLang. Supported values: {', '.join(sorted(choices))}.",
                )
        if config.tensor_parallelism > 1 and sglang_argument_capabilities.tensor_parallel_flag() is None:
            return CandidateValidationResult(
                config_id=config.id,
                valid=False,
                reason="tensor_parallel_size > 1 requires detected SGLang tensor parallel flag support.",
            )
        if config.max_batch_size > 1 and not sglang_argument_capabilities.supports("--max-running-requests"):
            return CandidateValidationResult(
                config_id=config.id,
                valid=False,
                reason="max_num_seqs > 1 requires detected SGLang support for --max-running-requests.",
            )
        if config.gpu_memory_utilization > 0 and not sglang_argument_capabilities.supports("--mem-fraction-static"):
            return CandidateValidationResult(
                config_id=config.id,
                valid=False,
                reason="gpu_memory_utilization > 0 requires detected SGLang support for --mem-fraction-static.",
            )
        quantization = normalize_quantization(config.quantization)
        if quantization != "none":
            if not sglang_argument_capabilities.supports("--quantization"):
                return CandidateValidationResult(
                    config_id=config.id,
                    valid=False,
                    reason="quantization requires detected SGLang support for --quantization.",
                )
            choices = sglang_argument_capabilities.choices_for("--quantization")
            if choices and quantization not in choices:
                return CandidateValidationResult(
                    config_id=config.id,
                    valid=False,
                    reason=(
                        f"quantization '{quantization}' is not listed by installed SGLang. "
                        f"Supported values: {', '.join(sorted(choices))}."
                    ),
                )
        capability_options = (
            ("disable_piecewise_cuda_graph", "--disable-piecewise-cuda-graph"),
            ("disable_radix_cache", "--disable-radix-cache"),
            ("disable_cuda_graph", "--disable-cuda-graph"),
            ("trust_remote_code", "--trust-remote-code"),
            ("served_model_name", "--served-model-name"),
            ("chunked_prefill_size", "--chunked-prefill-size"),
            ("cuda_graph_max_bs", "--cuda-graph-max-bs"),
        )
        for field_name, flag in capability_options:
            if extra.get(field_name) not in (None, False) and not sglang_argument_capabilities.supports(flag):
                return CandidateValidationResult(
                    config_id=config.id,
                    valid=False,
                    reason=f"{field_name} requires detected SGLang support for {flag}.",
                )
    return CandidateValidationResult(config_id=config.id, valid=True)


def validate_engine_options(
    config: ServingConfig,
    *,
    vllm_argument_capabilities: VLLMArgumentCapabilities | None = None,
) -> CandidateValidationResult:
    checks = [
        _positive_optional_int("block_size", config.block_size),
        _positive_optional_int("max_num_batched_tokens", config.max_num_batched_tokens),
        _positive_optional_int("max_cudagraph_capture_size", config.max_cudagraph_capture_size),
        _optional_bool("enforce_eager", config.enforce_eager),
        _optional_bool("enable_chunked_prefill", config.enable_chunked_prefill),
        _optional_bool("enable_prefix_caching", config.enable_prefix_caching),
    ]
    for reason in checks:
        if reason is not None:
            return CandidateValidationResult(config_id=config.id, valid=False, reason=reason)

    if config.kv_cache_dtype is not None:
        if not isinstance(config.kv_cache_dtype, str):
            return CandidateValidationResult(config_id=config.id, valid=False, reason="kv_cache_dtype must be a string.")
        installed_choices = (
            vllm_argument_capabilities.choices_for("--kv-cache-dtype")
            if vllm_argument_capabilities is not None and vllm_argument_capabilities.detection_status == "success"
            else frozenset()
        )
        if not installed_choices and config.kv_cache_dtype not in SUPPORTED_KV_CACHE_DTYPES:
            return CandidateValidationResult(
                config_id=config.id,
                valid=False,
                reason=f"kv_cache_dtype '{config.kv_cache_dtype}' is not supported. Supported values: auto, bfloat16, float16.",
            )

    capability_result = _validate_vllm_argument_capabilities(config, vllm_argument_capabilities)
    if capability_result is not None:
        return capability_result

    if config.enable_chunked_prefill is False:
        if config.max_num_batched_tokens is None:
            return CandidateValidationResult(
                config_id=config.id,
                valid=False,
                reason="enable_chunked_prefill=false requires max_num_batched_tokens to be set.",
            )
        if config.max_num_batched_tokens <= config.max_context_tokens:
            return CandidateValidationResult(
                config_id=config.id,
                valid=False,
                reason="enable_chunked_prefill=false requires max_num_batched_tokens > max_model_len.",
            )

    if config.enforce_eager is True and config.max_cudagraph_capture_size is not None:
        return CandidateValidationResult(
            config_id=config.id,
            valid=False,
            reason="enforce_eager=true cannot be combined with max_cudagraph_capture_size.",
        )

    return CandidateValidationResult(config_id=config.id, valid=True)


def _validate_vllm_argument_capabilities(
    config: ServingConfig,
    capabilities: VLLMArgumentCapabilities | None,
) -> CandidateValidationResult | None:
    if capabilities is None or capabilities.detection_status != "success":
        return None
    field_flags = [
        ("block_size", config.block_size, "--block-size"),
        ("enforce_eager", config.enforce_eager is True, "--enforce-eager"),
        ("max_num_batched_tokens", config.max_num_batched_tokens, "--max-num-batched-tokens"),
    ]
    for field_name, value, flag in field_flags:
        if value is not None and value is not False and not capabilities.supports(flag):
            return CandidateValidationResult(
                config_id=config.id,
                valid=False,
                reason=f"{field_name} requires installed vLLM support for {flag}.",
            )

    if config.kv_cache_dtype is not None:
        if not capabilities.supports("--kv-cache-dtype"):
            return CandidateValidationResult(
                config_id=config.id,
                valid=False,
                reason="kv_cache_dtype requires installed vLLM support for --kv-cache-dtype.",
            )
        choices = capabilities.choices_for("--kv-cache-dtype")
        if choices and config.kv_cache_dtype not in choices:
            return CandidateValidationResult(
                config_id=config.id,
                valid=False,
                reason=f"kv_cache_dtype '{config.kv_cache_dtype}' is not listed by installed vLLM. Supported values: {', '.join(sorted(choices))}.",
            )

    if config.max_cudagraph_capture_size is not None and capabilities.cudagraph_capture_flag() is None:
        return CandidateValidationResult(
            config_id=config.id,
            valid=False,
            reason="max_cudagraph_capture_size requires installed vLLM support for --max-cudagraph-capture-size or --cuda-graph-sizes.",
        )

    if config.enable_chunked_prefill is True and not capabilities.supports("--enable-chunked-prefill"):
        return CandidateValidationResult(
            config_id=config.id,
            valid=False,
            reason="enable_chunked_prefill=true requires installed vLLM support for --enable-chunked-prefill.",
        )
    if config.enable_chunked_prefill is False and not capabilities.supports("--no-enable-chunked-prefill"):
        return CandidateValidationResult(
            config_id=config.id,
            valid=False,
            reason="enable_chunked_prefill=false requires installed vLLM support for --no-enable-chunked-prefill.",
        )

    if config.enable_prefix_caching is True and not capabilities.supports("--enable-prefix-caching"):
        return CandidateValidationResult(
            config_id=config.id,
            valid=False,
            reason="enable_prefix_caching=true requires installed vLLM support for --enable-prefix-caching.",
        )
    if config.enable_prefix_caching is False and not capabilities.supports("--no-enable-prefix-caching"):
        return CandidateValidationResult(
            config_id=config.id,
            valid=False,
            reason="enable_prefix_caching=false requires installed vLLM support for --no-enable-prefix-caching.",
        )

    return None


def _positive_optional_int(name: str, value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        return f"{name} must be an integer."
    if value <= 0:
        return f"{name} must be greater than 0."
    return None


def _optional_bool(name: str, value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        return f"{name} must be a boolean."
    return None


def _sglang_dtype(dtype: str) -> str:
    return {
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
    }.get(dtype, dtype)
