from serve_optimize.managed_summary import baseline_comparison, format_recommendation_summary_text
from serve_optimize.schemas import RecommendationResult


def test_baseline_comparison_reports_directional_improvements() -> None:
    recommendation = _recommendation(
        [
            _row("baseline", source="safe_baseline", throughput=100.0, latency=1.0, power=200.0, energy=2.0, efficiency=0.5),
            _row("selected", source="generated", throughput=120.0, latency=0.8, power=180.0, energy=1.5, efficiency=0.8),
        ]
    )

    comparison = baseline_comparison(recommendation)

    assert comparison["available"] is True
    assert comparison["selected_is_baseline"] is False
    assert comparison["metrics"]["throughput_tokens_per_sec"]["improvement_percent"] == 20.0
    assert comparison["metrics"]["p95_latency_ms"]["improvement_percent"] == 20.0
    assert comparison["metrics"]["average_power_w"]["improvement_percent"] == 10.0
    assert comparison["metrics"]["joules_per_token"]["improvement_percent"] == 25.0
    assert comparison["metrics"]["tokens_per_watt"]["improvement_percent"] == 60.0


def test_baseline_comparison_uses_matching_backend_default_variant() -> None:
    recommendation = _recommendation(
        [
            _row("baseline", source="safe_baseline", throughput=100.0, concurrency=8),
            _row("baseline-16", source="backend_default_variant", throughput=150.0, concurrency=16),
            _row("selected", source="generated", throughput=180.0, concurrency=16),
        ]
    )

    comparison = baseline_comparison(recommendation)

    assert comparison["baseline_candidate_id"] == "baseline-16"
    assert comparison["metrics"]["throughput_tokens_per_sec"]["baseline"] == 150.0
    assert comparison["metrics"]["throughput_tokens_per_sec"]["improvement_percent"] == 20.0


def test_baseline_comparison_explains_missing_baseline() -> None:
    comparison = baseline_comparison(
        _recommendation([_row("selected", source="generated", throughput=120.0)])
    )

    assert comparison["available"] is False
    assert "baseline" in comparison["reason"].lower()


def test_summary_text_includes_baseline_comparison() -> None:
    payload = {
        "status": "success",
        "goal": "balanced",
        "confidence": "high",
        "recommendation_type": "comparative measured recommendation",
        "recommended_command": "vllm serve model",
        "selected": {},
        "metrics": {},
        "evaluated_set_fidelity": {},
        "baseline_comparison": {
            "available": True,
            "baseline_candidate_id": "baseline",
            "selected_is_baseline": False,
            "metrics": {
                "throughput_tokens_per_sec": {
                    "baseline": 100.0,
                    "selected": 120.0,
                    "improvement_percent": 20.0,
                }
            },
        },
    }

    text = format_recommendation_summary_text(payload)

    assert "Default baseline comparison:" in text
    assert "throughput_tokens_per_sec: +20.00%" in text


def _recommendation(rows: list[dict[str, object]]) -> RecommendationResult:
    return RecommendationResult(
        recommended_candidate_id="selected",
        goal="balanced",
        selected_score=None,
        selected_config=None,
        selected_serve_command=None,
        selected_benchmark_plan=None,
        candidate_table=rows,
    )


def _row(
    candidate_id: str,
    *,
    source: str,
    throughput: float | None = None,
    latency: float | None = None,
    power: float | None = None,
    energy: float | None = None,
    efficiency: float | None = None,
    concurrency: int | None = None,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "candidate_source": source,
        "benchmark_concurrency": concurrency,
        "total_tokens_s": throughput,
        "p95_latency_s": latency,
        "average_power_watts": power,
        "joules_per_token": energy,
        "tokens_per_second_per_watt": efficiency,
    }
