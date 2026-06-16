from serve_optimize.pareto import pareto_frontier, select_recommendation
from serve_optimize.schemas import BenchmarkResult, Goal, ServingConfig


def cfg(name: str) -> ServingConfig:
    return ServingConfig(
        id=name,
        backend="dry-run",
        model_id="mistral-7b",
        dtype="fp16",
        quantization="none",
        max_batch_size=1,
        max_context_tokens=2048,
        kv_cache_policy="paged",
        scheduler="continuous-batching",
    )


def result(name: str, throughput: float, jpt: float) -> BenchmarkResult:
    return BenchmarkResult(
        config=cfg(name),
        throughput_tok_s=throughput,
        average_power_watts=throughput * jpt,
        joules_per_token=jpt,
        tokens_per_watt=1 / jpt,
    )


def test_pareto_filters_dominated_results() -> None:
    rows = [
        result("fast", 100, 2.0),
        result("efficient", 80, 1.0),
        result("dominated", 70, 1.5),
    ]
    frontier = pareto_frontier(rows)
    assert {item.config.id for item in frontier} == {"fast", "efficient"}


def test_goal_selection() -> None:
    rows = [result("fast", 100, 2.0), result("efficient", 80, 1.0)]
    assert select_recommendation(rows, Goal.PERFORMANCE).config.id == "fast"
    assert select_recommendation(rows, Goal.EFFICIENT).config.id == "efficient"

