from scripts.run_research_validation import _baseline_first, _search_pool
from serve_optimize.schemas import ServingConfig


def _config(identifier: str, *, baseline: bool = False, batch: int = 1) -> ServingConfig:
    return ServingConfig(
        id=identifier,
        backend="vllm",
        model_id="model",
        dtype="bfloat16",
        quantization="none",
        max_batch_size=batch,
        max_context_tokens=2048,
        kv_cache_policy="paged",
        scheduler="continuous batching",
        extra={"baseline": baseline, "candidate_source": "safe_baseline" if baseline else "capability_aware"},
    )


def test_validation_search_pool_keeps_backend_default_first() -> None:
    ordered = _baseline_first([_config("candidate", batch=8), _config("default", baseline=True)])
    assert [item.id for item in ordered] == ["default", "candidate"]
    assert _search_pool(ordered)[0].baseline is True
