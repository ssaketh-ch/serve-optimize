import json

from serve_optimize.backends.vllm import VLLMArgumentCapabilities
from serve_optimize.modeling import infer_model_capability_metadata
from serve_optimize.schemas import ServingConfig
from serve_optimize.validation import (
    normalize_quantization,
    validate_managed_candidate,
    validate_quantization_compatibility,
)


def test_normal_bf16_local_config_rejects_awq_candidate(tmp_path) -> None:
    model_dir = _model_dir(tmp_path, {"torch_dtype": "bfloat16"})
    metadata = infer_model_capability_metadata(str(model_dir))

    result = validate_quantization_compatibility(_config("awq"), metadata)

    assert result.valid is False
    assert "quantization_config.quant_method=awq" in str(result.reason)


def test_normal_bf16_local_config_accepts_none_quantization(tmp_path) -> None:
    model_dir = _model_dir(tmp_path, {"torch_dtype": "bfloat16"})
    metadata = infer_model_capability_metadata(str(model_dir))

    result = validate_quantization_compatibility(_config("none"), metadata)

    assert result.valid is True


def test_awq_local_config_accepts_awq(tmp_path) -> None:
    model_dir = _model_dir(tmp_path, {"quantization_config": {"quant_method": "awq"}})
    metadata = infer_model_capability_metadata(str(model_dir))

    result = validate_quantization_compatibility(_config("awq"), metadata)

    assert result.valid is True


def test_gptq_local_config_accepts_gptq(tmp_path) -> None:
    model_dir = _model_dir(tmp_path, {"quantization_config": {"quant_method": "gptq"}})
    metadata = infer_model_capability_metadata(str(model_dir))

    result = validate_quantization_compatibility(_config("gptq"), metadata)

    assert result.valid is True


def test_quantization_aliases_normalize() -> None:
    assert normalize_quantization("awq-int4") == "awq"
    assert normalize_quantization("awq") == "awq"
    assert normalize_quantization("gptq-int4") == "gptq"
    assert normalize_quantization("gptq") == "gptq"
    assert normalize_quantization(None) == "none"
    assert normalize_quantization("") == "none"
    assert normalize_quantization("null") == "none"
    assert normalize_quantization("none") == "none"


def test_awq_int4_alias_validates_against_awq_metadata(tmp_path) -> None:
    model_dir = _model_dir(tmp_path, {"quantization_config": {"quant_method": "awq"}})
    metadata = infer_model_capability_metadata(str(model_dir))

    result = validate_quantization_compatibility(_config("awq-int4"), metadata)

    assert result.valid is True


def test_gptq_int4_alias_validates_against_gptq_metadata(tmp_path) -> None:
    model_dir = _model_dir(tmp_path, {"quantization_config": {"quant_method": "gptq"}})
    metadata = infer_model_capability_metadata(str(model_dir))

    result = validate_quantization_compatibility(_config("gptq-int4"), metadata)

    assert result.valid is True


def test_unknown_remote_model_metadata_rejects_explicit_quantization_without_crashing() -> None:
    metadata = infer_model_capability_metadata("org/model-id")

    result = validate_quantization_compatibility(_config("awq"), metadata)

    assert metadata.metadata_known is False
    assert result.valid is False
    assert "quantization_config.quant_method=awq" in str(result.reason)


def test_invalid_block_size_is_rejected() -> None:
    result = validate_managed_candidate(
        _config("none", block_size=0),
        backend="vllm",
        model_metadata=infer_model_capability_metadata("org/model-id"),
    )

    assert result.valid is False
    assert "block_size must be greater than 0" in str(result.reason)


def test_invalid_kv_cache_dtype_is_rejected() -> None:
    result = validate_managed_candidate(
        _config("none", kv_cache_dtype="fp8_e4m3"),
        backend="vllm",
        model_metadata=infer_model_capability_metadata("org/model-id"),
    )

    assert result.valid is False
    assert "kv_cache_dtype 'fp8_e4m3' is not supported" in str(result.reason)


def test_kv_cache_dtype_rejected_when_installed_vllm_choices_exclude_it() -> None:
    result = validate_managed_candidate(
        _config("none", kv_cache_dtype="bfloat16"),
        backend="vllm",
        model_metadata=infer_model_capability_metadata("org/model-id"),
        vllm_argument_capabilities=_caps(
            "--kv-cache-dtype",
            kv_cache_dtype_choices=("auto", "fp8", "fp8_e4m3", "fp8_e5m2", "fp8_inc"),
        ),
    )

    assert result.valid is False
    assert "kv_cache_dtype 'bfloat16' is not listed by installed vLLM" in str(result.reason)


def test_kv_cache_dtype_allowed_when_installed_vllm_lists_it() -> None:
    result = validate_managed_candidate(
        _config("none", kv_cache_dtype="auto"),
        backend="vllm",
        model_metadata=infer_model_capability_metadata("org/model-id"),
        vllm_argument_capabilities=_caps("--kv-cache-dtype", kv_cache_dtype_choices=("auto", "fp8")),
    )

    assert result.valid is True


def test_unsupported_engine_argument_is_rejected_before_launch() -> None:
    result = validate_managed_candidate(
        _config("none", block_size=16),
        backend="vllm",
        model_metadata=infer_model_capability_metadata("org/model-id"),
        vllm_argument_capabilities=_caps(),
    )

    assert result.valid is False
    assert "block_size requires installed vLLM support for --block-size" in str(result.reason)


def test_cudagraph_capture_size_accepts_cuda_graph_sizes_alias() -> None:
    result = validate_managed_candidate(
        _config("none", max_cudagraph_capture_size=32),
        backend="vllm",
        model_metadata=infer_model_capability_metadata("org/model-id"),
        vllm_argument_capabilities=_caps("--cuda-graph-sizes"),
    )

    assert result.valid is True


def test_chunked_prefill_true_rejected_when_installed_flag_missing() -> None:
    result = validate_managed_candidate(
        _config("none", enable_chunked_prefill=True),
        backend="vllm",
        model_metadata=infer_model_capability_metadata("org/model-id"),
        vllm_argument_capabilities=_caps(),
    )

    assert result.valid is False
    assert "enable_chunked_prefill=true requires installed vLLM support" in str(result.reason)


def test_chunked_prefill_false_requires_batched_tokens_above_model_len() -> None:
    result = validate_managed_candidate(
        _config("none", enable_chunked_prefill=False, max_num_batched_tokens=2048),
        backend="vllm",
        model_metadata=infer_model_capability_metadata("org/model-id"),
    )

    assert result.valid is False
    assert "max_num_batched_tokens > max_model_len" in str(result.reason)


def test_enforce_eager_rejects_cudagraph_capture_size() -> None:
    result = validate_managed_candidate(
        _config("none", enforce_eager=True, max_cudagraph_capture_size=32),
        backend="vllm",
        model_metadata=infer_model_capability_metadata("org/model-id"),
    )

    assert result.valid is False
    assert "enforce_eager=true cannot be combined" in str(result.reason)


def _caps(*flags: str, kv_cache_dtype_choices: tuple[str, ...] = ()) -> VLLMArgumentCapabilities:
    option_choices = {"--kv-cache-dtype": frozenset(kv_cache_dtype_choices)} if kv_cache_dtype_choices else {}
    return VLLMArgumentCapabilities(
        executable="vllm",
        version="test",
        supported_flags=frozenset(flags),
        option_choices=option_choices,
        help_hash="test",
        detection_status="success",
    )


def _model_dir(tmp_path, payload: dict[str, object]):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    return model_dir


def _config(quantization: str, **kwargs) -> ServingConfig:
    return ServingConfig(
        id=f"cfg-{quantization}",
        backend="vllm",
        model_id="model-path",
        dtype="bf16",
        quantization=quantization,
        max_batch_size=1,
        max_context_tokens=2048,
        kv_cache_policy="paged",
        scheduler="continuous-batching",
        **kwargs,
    )
