from pathlib import Path

from serve_optimize.aiconfigurator_bridge import AIConfiguratorRun
from serve_optimize.evidence import EvidenceHitType, EvidenceLookupResult
from serve_optimize.priors import (
    AIConfiguratorPriorProvider,
    ManagedPriorPolicy,
    apply_managed_prior_policy,
    evidence_lookup_to_prior,
)
from serve_optimize.schemas import Goal, PriorCandidate, PriorResult, PriorSource, ServingConfig


def test_aiconfigurator_prior_provider_unavailable_does_not_crash(tmp_path) -> None:
    def runner(**_kwargs):
        raise RuntimeError("missing cli")

    result = AIConfiguratorPriorProvider(runner=runner).collect_priors(
        model="model-path",
        backend="vllm",
        goal=Goal.BALANCED,
        candidates=[_config()],
        out_dir=tmp_path,
    )

    assert result.available is False
    assert result.used is False
    assert "missing cli" in result.warnings[0]


def test_aiconfigurator_prior_candidate_labels_source_and_confidence(tmp_path) -> None:
    def runner(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        csv_dir = output_dir / "run" / "agg"
        csv_dir.mkdir(parents=True)
        (csv_dir / "best_config_topn.csv").write_text(
            "backend,model,tokens/s,ttft,tpot,request_latency,memory,power_w\n"
            "vllm,model-path,120.5,10,2,40,12.5,210\n",
            encoding="utf-8",
        )
        return AIConfiguratorRun(command=["aiconfigurator"], returncode=0, stdout="", stderr="")

    result = AIConfiguratorPriorProvider(runner=runner).collect_priors(
        model="model-path",
        backend="vllm",
        goal=Goal.BALANCED,
        candidates=[_config()],
        out_dir=tmp_path,
    )

    assert result.available is True
    assert result.used is True
    assert result.candidates[0].source == PriorSource.AICONFIGURATOR.value
    assert result.candidates[0].confidence == 1.0
    assert result.candidates[0].predicted_throughput_tokens_per_sec == 120.5
    assert result.candidates[0].raw_prior_payload["predicted_power_w"] == 210.0


def test_prior_pruning_preserves_baseline_and_low_memory_candidate() -> None:
    candidates = [
        _config(config_id="cfg-a", estimated_vram_mb=16_000),
        _config(config_id="cfg-b", dtype="bf16", estimated_vram_mb=8_000),
        _config(config_id="cfg-c", dtype="fp32", estimated_vram_mb=24_000),
    ]

    result = apply_managed_prior_policy(
        candidates,
        prior_results=[],
        evidence_priors=[],
        policy=ManagedPriorPolicy(max_prior_candidates=1, preserve_diversity=False),
    )

    assert [candidate.id for candidate in result.candidates] == ["cfg-a", "cfg-b"]
    assert result.candidates_pruned_by_prior == 1


def test_prior_pruning_preserves_near_compatible_evidence_metadata() -> None:
    candidates = [
        _config(config_id="cfg-a"),
        _config(config_id="cfg-b", dtype="bf16"),
        _config(config_id="cfg-c", dtype="fp32"),
    ]
    evidence_prior = PriorCandidate(
        source=PriorSource.EVIDENCE_NEAR_HIT.value,
        candidate_id="cfg-c",
        config_id="cfg-c",
        confidence=0.5,
        notes=["near compatible"],
    )

    result = apply_managed_prior_policy(
        candidates,
        prior_results=[],
        evidence_priors=[evidence_prior],
        policy=ManagedPriorPolicy(max_prior_candidates=1, preserve_diversity=False),
    )

    assert "cfg-c" in [candidate.id for candidate in result.candidates]
    cfg_c = next(candidate for candidate in result.candidates if candidate.id == "cfg-c")
    assert cfg_c.extra["prior_source"] == PriorSource.EVIDENCE_NEAR_HIT.value
    assert cfg_c.extra["prior_confidence"] == 0.5


def test_evidence_lookup_to_prior_uses_stale_and_near_only_as_metadata() -> None:
    stale = evidence_lookup_to_prior(
        _config(),
        EvidenceLookupResult(
            hit_type=EvidenceHitType.EXACT_STALE_HIT,
            reason="stale",
            measurement={"measurement_id": "meas-1", "throughput_tokens_per_sec": 10.0},
        ),
    )
    near = evidence_lookup_to_prior(
        _config(config_id="cfg-near"),
        EvidenceLookupResult(
            hit_type=EvidenceHitType.NEAR_COMPATIBLE_HIT,
            reason="near",
            near_matches=[{"measurement_id": "meas-2", "throughput_tokens_per_sec": 20.0}],
        ),
    )

    assert stale is not None
    assert stale.source == PriorSource.EVIDENCE_STALE_HIT.value
    assert near is not None
    assert near.source == PriorSource.EVIDENCE_NEAR_HIT.value


def test_prior_pruning_uses_matching_aiconfigurator_prior() -> None:
    candidates = [_config(config_id="cfg-a"), _config(config_id="cfg-b"), _config(config_id="cfg-c")]
    result = apply_managed_prior_policy(
        candidates,
        prior_results=[
            PriorResult(
                source=PriorSource.AICONFIGURATOR.value,
                available=True,
                used=True,
                candidates=[
                    PriorCandidate(
                        source=PriorSource.AICONFIGURATOR.value,
                        candidate_id="cfg-c",
                        config_id="cfg-c",
                        confidence=0.9,
                        notes=["estimated"],
                    )
                ],
            )
        ],
        evidence_priors=[],
        policy=ManagedPriorPolicy(max_prior_candidates=1, preserve_low_memory_candidate=False, preserve_diversity=False),
    )

    assert "cfg-c" in [candidate.id for candidate in result.candidates]
    cfg_c = next(candidate for candidate in result.candidates if candidate.id == "cfg-c")
    assert cfg_c.extra["prior_source"] == PriorSource.AICONFIGURATOR.value


def _config(
    config_id: str = "cfg-test",
    dtype: str = "fp16",
    estimated_vram_mb: int | None = None,
) -> ServingConfig:
    return ServingConfig(
        id=config_id,
        backend="vllm",
        model_id="model-path",
        dtype=dtype,
        quantization="none",
        max_batch_size=2,
        max_context_tokens=2048,
        kv_cache_policy="paged",
        scheduler="continuous-batching",
        estimated_vram_mb=estimated_vram_mb,
    )
