from pathlib import Path

from serve_optimize.aiconfig_parser import parse_aiconfigurator_text_candidates
from serve_optimize.aiconfigurator_bridge import AIConfiguratorRun
from serve_optimize.backends.vllm import VLLMArgumentCapabilities
from serve_optimize.schemas import Goal, GpuDevice, HardwareSnapshot, ModelCapabilityMetadata, ServingConfig
from serve_optimize.synthesis import (
    SYNTHESIS_SOURCE,
    AIConfiguratorSynthesisProvider,
    CandidateSynthesisContext,
    resolve_aiconfigurator_system_key,
)


def test_aiconfigurator_system_resolution_maps_h200_mig_to_h200_sxm() -> None:
    resolution = resolve_aiconfigurator_system_key(
        _hardware(name="NVIDIA H200 141GB HBM3 MIG 1g.18gb", uuid="MIG-1", mig_profile="1g.18gb")
    )

    assert resolution.system_key == "h200_sxm"
    assert resolution.is_mig is True


def test_aiconfigurator_system_resolution_maps_common_gpu_names() -> None:
    cases = {
        "NVIDIA H200 SXM": "h200_sxm",
        "NVIDIA H100 PCIe": "h100_pcie",
        "NVIDIA H100 SXM": "h100_sxm",
        "NVIDIA A100 PCIe": "a100_pcie",
        "NVIDIA A100 SXM": "a100_sxm",
        "NVIDIA L40S": "l40s",
    }

    for gpu_name, system_key in cases.items():
        assert resolve_aiconfigurator_system_key(_hardware(name=gpu_name)).system_key == system_key


def test_aiconfigurator_system_resolution_unknown_skips_cleanly() -> None:
    resolution = resolve_aiconfigurator_system_key(_hardware(name="Generic CUDA GPU"))

    assert resolution.system_key is None
    assert resolution.warnings


def test_aiconfigurator_text_table_parser_extracts_candidates() -> None:
    text = """
    rank backend throughput ttft_ms request_latency_ms concurrency batch_size tp pp dp
    1 vllm 1234.5 10 20 4 4 1 1 1
    """

    parsed = parse_aiconfigurator_text_candidates(text)

    assert len(parsed) == 1
    assert parsed[0].backend == "vllm"
    assert parsed[0].predicted_tokens_s == 1234.5
    assert parsed[0].predicted_ttft_ms == 10.0
    assert parsed[0].predicted_request_latency_ms == 20.0
    assert parsed[0].concurrency == 4
    assert parsed[0].batch_size == 4


def test_aiconfigurator_synthesis_provider_converts_bounded_candidates(tmp_path) -> None:
    seen_kwargs = {}

    def runner(**kwargs):
        seen_kwargs.update(kwargs)
        output_dir = Path(kwargs["output_dir"])
        csv_dir = output_dir / "run" / "agg"
        csv_dir.mkdir(parents=True)
        (csv_dir / "best_config_topn.csv").write_text(
            "backend,model,isl,osl,concurrency,bs,tokens/s,request_latency\n"
            "vllm,model-path,1024,128,4,4,120.5,40\n"
            "vllm,model-path,2048,128,8,8,130.5,50\n",
            encoding="utf-8",
        )
        return AIConfiguratorRun(command=["aiconfigurator"], returncode=0, stdout="", stderr="")

    provider = AIConfiguratorSynthesisProvider(runner=runner, max_candidates=1)
    result = provider.synthesize(context=_context(hardware=_hardware(name="NVIDIA H200 SXM")), out_dir=tmp_path)

    assert result.available is True
    assert result.used is True
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.extra["candidate_source"] == SYNTHESIS_SOURCE
    assert candidate.extra["synthesis_confidence"] == 1.0
    assert candidate.quantization == "none"
    assert candidate.dtype == "bf16"
    assert candidate.max_batch_size == 4
    assert candidate.extra["aiconfigurator_system_key"] == "h200_sxm"
    assert candidate.extra["aiconfigurator_rank"] == 1
    assert candidate.extra["aiconfigurator_predicted_metrics"]["tokens_per_sec"] == 120.5
    assert seen_kwargs["system"] == "h200_sxm"
    assert "local_gpu" not in result.raw_metadata["command"]


def test_aiconfigurator_synthesis_provider_parses_text_output(tmp_path) -> None:
    def runner(**kwargs):
        return AIConfiguratorRun(
            command=["aiconfigurator", "--system", kwargs["system"]],
            returncode=0,
            stdout="rank backend throughput ttft_ms request_latency_ms concurrency batch_size tp pp dp\n1 vllm 777 11 22 4 4 1 1 1\n",
            stderr="",
        )

    result = AIConfiguratorSynthesisProvider(runner=runner, max_candidates=1).synthesize(
        context=_context(hardware=_hardware(name="NVIDIA H200 SXM")),
        out_dir=tmp_path,
    )

    assert result.used is True
    assert result.raw_metadata["parser_source"] == "stdout_stderr"
    assert result.candidates[0].extra["aiconfigurator_predicted_metrics"]["tokens_per_sec"] == 777.0


def test_aiconfigurator_synthesis_unknown_system_is_nonfatal(tmp_path) -> None:
    calls = []

    def runner(**kwargs):
        calls.append(kwargs)
        return AIConfiguratorRun(command=["aiconfigurator"], returncode=0, stdout="", stderr="")

    result = AIConfiguratorSynthesisProvider(runner=runner).synthesize(
        context=_context(hardware=_hardware(name="Generic CUDA GPU")),
        out_dir=tmp_path,
    )

    assert result.available is False
    assert result.used is False
    assert result.skipped_reason == "unable_to_resolve_aiconfigurator_system"
    assert calls == []


def test_aiconfigurator_synthesis_provider_error_is_nonfatal(tmp_path) -> None:
    def runner(**_kwargs):
        raise RuntimeError("boom")

    result = AIConfiguratorSynthesisProvider(runner=runner).synthesize(context=_context(), out_dir=tmp_path)

    assert result.available is False
    assert result.used is False
    assert result.candidates == []
    assert "boom" in result.warnings[0]


def _context(hardware: HardwareSnapshot | None = None) -> CandidateSynthesisContext:
    baseline = _config("cfg-baseline", max_batch_size=1, max_context_tokens=2048)
    existing = [
        baseline,
        _config("cfg-wide", max_batch_size=8, max_context_tokens=4096, max_num_batched_tokens=4097, enable_chunked_prefill=True),
    ]
    return CandidateSynthesisContext(
        hardware=hardware or _hardware(name="NVIDIA H200 SXM"),
        backend="vllm",
        backend_argument_capabilities=_caps(
            "--max-num-batched-tokens",
            "--enable-chunked-prefill",
        ),
        model="model-path",
        model_metadata=ModelCapabilityMetadata(model_id="model-path", metadata_known=True, torch_dtype="bfloat16"),
        goal=Goal.BALANCED,
        evidence_summary={"exact_fresh_candidate_ids": [], "evidence_prior_count": 0},
        safe_baseline=baseline,
        existing_candidates=existing,
        max_candidates=3,
    )


def _config(config_id: str, *, max_batch_size: int, max_context_tokens: int, **kwargs) -> ServingConfig:
    return ServingConfig(
        id=config_id,
        backend="vllm",
        model_id="model-path",
        dtype="bf16",
        quantization="none",
        max_batch_size=max_batch_size,
        max_context_tokens=max_context_tokens,
        kv_cache_policy="paged",
        scheduler="continuous-batching",
        gpu_memory_utilization=0.9,
        extra={
            "candidate_source": "safe_baseline" if config_id == "cfg-baseline" else "capability_aware",
            "baseline": config_id == "cfg-baseline",
            "model_native": True,
            "workload_concurrency": max_batch_size,
            "max_new_tokens": 128,
        },
        **kwargs,
    )


def _hardware(name: str = "Generic CUDA GPU", uuid: str = "GPU-1", mig_profile: str | None = None) -> HardwareSnapshot:
    return HardwareSnapshot(
        hostname="host",
        platform="linux",
        python_version="3.12",
        detected_at="2026-01-01T00:00:00+00:00",
        gpus=[
            GpuDevice(
                index=0,
                name=name,
                uuid=uuid,
                total_memory_mb=80_000,
                compute_capability="9.0",
                mig_profile=mig_profile,
                driver_version="1",
                cuda_version="12",
            )
        ],
    )


def _caps(*flags: str) -> VLLMArgumentCapabilities:
    return VLLMArgumentCapabilities(
        executable="vllm",
        version="test",
        supported_flags=frozenset(flags),
        help_hash="test",
        detection_status="success",
    )
