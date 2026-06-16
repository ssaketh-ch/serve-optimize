import json

import pytest

from serve_optimize.aiconfigurator_bridge import AIConfiguratorRun
from serve_optimize.cli import _parse_concurrency_sweep, main
from serve_optimize.recommendation import (
    ATTACH_MODE_LIMITATION,
    RecommendationRun,
    audit_recommendation_quality,
    build_attach_preflight,
    compute_evaluated_set_fidelity,
    compute_optimizer_quality,
    generate_attach_mode_candidate_set,
    generate_attach_mode_candidates,
    generate_heuristic_candidates,
    generate_sweep_candidates,
    recommend_attach_mode,
    score_recommendation_inputs,
)
from serve_optimize.schemas import (
    EndpointBenchmarkPlan,
    RecommendationGoal,
    RecommendationInput,
    RecommendationResult,
    RecommendationScore,
    RequestRecord,
    ServeCandidate,
    VllmServePlan,
)


def test_throughput_goal_prefers_highest_measured_tokens() -> None:
    inputs = [
        _input("c1", total_tokens_s=100.0, p95_latency_s=1.0, request_rate=5.0),
        _input("c2", total_tokens_s=150.0, p95_latency_s=1.2, request_rate=5.0),
    ]

    scores, result = score_recommendation_inputs(inputs, goal=RecommendationGoal.THROUGHPUT)

    assert result.recommended_candidate_id == "c2"
    assert scores[0].candidate_id == "c2"


def test_evaluated_set_fidelity_selected_rank_one() -> None:
    inputs = [
        _input("c1", total_tokens_s=100.0, p95_latency_s=1.0, tokens_per_second_per_watt=1.0),
        _input("c2", total_tokens_s=150.0, p95_latency_s=1.2, tokens_per_second_per_watt=1.5),
    ]

    _, result = score_recommendation_inputs(inputs, goal=RecommendationGoal.THROUGHPUT)
    fidelity = result.evaluated_set_fidelity

    assert fidelity["scope"] == "evaluated_candidates_only"
    assert fidelity["selected_candidate_id"] == "c2"
    assert fidelity["selected_rank"] == 1
    assert fidelity["best_candidate_id"] == "c2"
    assert fidelity["selected_score_over_best_score"] == pytest.approx(1.0)
    assert fidelity["gap_to_best_score"] == pytest.approx(0.0)
    assert fidelity["selected_is_best_evaluated"] is True
    assert fidelity["metric_winners"]["throughput"]["candidate_id"] == "c2"
    assert result.optimizer_quality["scope"] == "evaluated_candidates_only"
    assert result.optimizer_quality["search_regret"]["score_gap_to_best"] == pytest.approx(0.0)
    assert "evaluated-set comparison" in fidelity["notes"][1]


def test_recommendation_quality_audit_is_evaluated_set_scoped() -> None:
    inputs = [
        _input("c1", total_tokens_s=100.0, p95_latency_s=1.0),
        _input("c2", total_tokens_s=150.0, p95_latency_s=1.2),
    ]

    _, result = score_recommendation_inputs(inputs, goal=RecommendationGoal.THROUGHPUT)
    audit = audit_recommendation_quality(result)

    assert audit["scope"] == "evaluated_candidates_only"
    assert audit["selected_candidate_id"] == "c2"
    assert audit["selected_is_best_evaluated"] is True
    assert audit["wording_policy"] == "evaluated_set_only"
    assert "exhaustive search" in audit["notes"][1]


def test_compute_evaluated_set_fidelity_selected_not_rank_one() -> None:
    ranked_scores = [
        _score("best", 0.9),
        _score("selected", 0.6),
    ]
    candidate_table = [
        {"candidate_id": "best", "total_tokens_s": 100.0, "p95_latency_s": 0.5},
        {"candidate_id": "selected", "total_tokens_s": 80.0, "p95_latency_s": 0.7},
    ]

    fidelity = compute_evaluated_set_fidelity(
        selected_candidate_id="selected",
        ranked_scores=ranked_scores,
        candidate_table=candidate_table,
        pareto_ids={"best"},
    )

    assert fidelity["selected_rank"] == 2
    assert fidelity["best_candidate_id"] == "best"
    assert fidelity["selected_is_best_evaluated"] is False
    assert fidelity["selected_is_pareto_optimal"] is False
    assert fidelity["selected_score_over_best_score"] == pytest.approx(0.6 / 0.9)
    assert fidelity["gap_to_best_score"] == pytest.approx(0.3)


def test_compute_evaluated_set_fidelity_zero_best_score() -> None:
    fidelity = compute_evaluated_set_fidelity(
        selected_candidate_id="zero",
        ranked_scores=[_score("zero", 0.0)],
        candidate_table=[{"candidate_id": "zero", "total_tokens_s": 0.0}],
        pareto_ids={"zero"},
    )

    assert fidelity["selected_rank"] == 1
    assert fidelity["best_score"] == 0.0
    assert fidelity["selected_score_over_best_score"] is None
    assert fidelity["gap_to_best_score"] == 0.0


def test_compute_optimizer_quality_reports_bounded_regret() -> None:
    quality = compute_optimizer_quality(
        goal="balanced",
        selected_candidate_id="selected",
        ranked_scores=[_score("best", 0.9), _score("selected", 0.6)],
        candidate_table=[
            {"candidate_id": "best", "source": "managed_measured", "total_tokens_s": 100.0, "p95_latency_s": 0.5, "score": 0.9, "status": "eligible"},
            {"candidate_id": "selected", "source": "managed_measured", "total_tokens_s": 80.0, "p95_latency_s": 0.7, "score": 0.6, "status": "eligible"},
        ],
    )

    assert quality["scope"] == "evaluated_candidates_only"
    assert quality["baseline_type"] == "bounded_evaluated_candidate_baseline"
    assert quality["search_regret"]["score_gap_to_best"] == pytest.approx(0.3)
    assert quality["search_regret"]["relative_score_regret"] == pytest.approx(1 / 3)
    assert quality["metric_regret_percent"]["throughput"] == pytest.approx(20.0)
    assert quality["metric_regret_percent"]["p95_latency"] == pytest.approx(40.0)
    assert quality["policy_coverage"]["candidate_source_counts"]["managed_measured"] == 2


def test_evaluated_set_fidelity_missing_power_metric_winners() -> None:
    inputs = [
        _input("c1", total_tokens_s=100.0, p95_latency_s=1.0),
        _input("c2", total_tokens_s=150.0, p95_latency_s=1.2),
    ]

    _, result = score_recommendation_inputs(inputs, goal=RecommendationGoal.THROUGHPUT)
    winners = result.evaluated_set_fidelity["metric_winners"]

    assert winners["throughput"]["candidate_id"] == "c2"
    assert winners["lowest_latency"]["candidate_id"] == "c1"
    assert winners["lowest_energy_per_token"] is None
    assert winners["best_tokens_per_watt"] is None
    assert winners["lowest_power"] is None


def test_latency_goal_prefers_lowest_p95_latency() -> None:
    inputs = [
        _input("c1", total_tokens_s=120.0, p95_latency_s=0.9),
        _input("c2", total_tokens_s=150.0, p95_latency_s=0.4),
    ]

    _, result = score_recommendation_inputs(inputs, goal=RecommendationGoal.LATENCY)

    assert result.recommended_candidate_id == "c2"


def test_balanced_goal_without_power_warns_and_uses_performance_weights() -> None:
    inputs = [
        _input("c1", total_tokens_s=80.0, p95_latency_s=0.8),
        _input("c2", total_tokens_s=120.0, p95_latency_s=1.5),
    ]

    _, result = score_recommendation_inputs(inputs, goal=RecommendationGoal.BALANCED)

    assert result.recommended_candidate_id == "c2"
    assert "Balanced scoring used performance-only weights because power telemetry was unavailable." in result.warnings


def test_balanced_goal_with_power_uses_efficiency_signal() -> None:
    inputs = [
        _input("c1", total_tokens_s=110.0, p95_latency_s=0.8, tokens_per_second_per_watt=0.6),
        _input("c2", total_tokens_s=100.0, p95_latency_s=0.9, tokens_per_second_per_watt=1.4),
    ]

    _, result = score_recommendation_inputs(inputs, goal=RecommendationGoal.BALANCED)

    assert result.recommended_candidate_id == "c2"
    assert result.telemetry_used_in_scoring is True
    assert result.score_weights["power"] == 0.30
    assert "Balanced scoring used performance-only weights because power telemetry was unavailable." not in result.warnings


def test_balanced_goal_without_power_can_pick_throughput_candidate() -> None:
    inputs = [
        _input("fast", total_tokens_s=140.0, p95_latency_s=1.0),
        _input("efficient", total_tokens_s=100.0, p95_latency_s=0.9),
    ]

    _, result = score_recommendation_inputs(inputs, goal=RecommendationGoal.BALANCED)

    assert result.recommended_candidate_id == "fast"
    assert result.telemetry_used_in_scoring is False


def test_balanced_goal_with_power_can_change_selection() -> None:
    inputs = [
        _input("fast", total_tokens_s=140.0, p95_latency_s=1.0, tokens_per_second_per_watt=0.4, joules_per_token=0.04),
        _input("efficient", total_tokens_s=100.0, p95_latency_s=0.9, tokens_per_second_per_watt=1.8, joules_per_token=0.006),
    ]

    _, result = score_recommendation_inputs(inputs, goal=RecommendationGoal.BALANCED)

    assert result.recommended_candidate_id == "efficient"
    assert result.power_aware is True


def test_efficiency_goal_with_power_prefers_best_tokens_per_watt() -> None:
    inputs = [
        _input("c1", total_tokens_s=90.0, p95_latency_s=1.0, tokens_per_second_per_watt=0.8),
        _input("c2", total_tokens_s=80.0, p95_latency_s=1.1, tokens_per_second_per_watt=1.5),
    ]

    _, result = score_recommendation_inputs(inputs, goal=RecommendationGoal.EFFICIENCY)

    assert result.recommended_candidate_id == "c2"
    assert result.alternative_recommendations["efficiency"]["candidate_id"] == "c2"


def test_efficiency_goal_without_power_returns_unavailable_result() -> None:
    inputs = [_input("c1", total_tokens_s=90.0, p95_latency_s=1.0)]

    scores, result = score_recommendation_inputs(inputs, goal=RecommendationGoal.EFFICIENCY)

    assert result.recommended_candidate_id is None
    assert scores[0].disqualifiers == ["missing_power_telemetry"]
    assert result.power_missing_reason == "No candidate had usable power telemetry."


def test_all_candidates_failed_produce_no_recommendation() -> None:
    inputs = [
        _input("c1", total_tokens_s=0.0, p95_latency_s=None, successful_requests=0, failed_requests=4, total_requests=4),
        _input("c2", total_tokens_s=0.0, p95_latency_s=None, successful_requests=0, failed_requests=4, total_requests=4),
    ]

    scores, result = score_recommendation_inputs(inputs, goal=RecommendationGoal.THROUGHPUT)

    assert result.recommended_candidate_id is None
    assert all(score.final_score is None for score in scores)
    assert all("no_successful_requests" in score.disqualifiers for score in scores)


def test_slo_latency_violation_is_ineligible_for_recommendation() -> None:
    inputs = [
        _input("slow", total_tokens_s=200.0, p95_latency_s=1.2, raw=_slo_raw({"p95_latency_ms": 800})),
        _input("ok", total_tokens_s=100.0, p95_latency_s=0.7, raw=_slo_raw({"p95_latency_ms": 800})),
    ]

    scores, result = score_recommendation_inputs(inputs, goal=RecommendationGoal.THROUGHPUT)
    score_by_id = {score.candidate_id: score for score in scores}

    assert result.recommended_candidate_id == "ok"
    assert "slo_p95_latency_ms_exceeded" in score_by_id["slow"].disqualifiers
    assert result.candidate_table[1]["status"] == "slo_p95_latency_ms_exceeded"


def test_slo_throughput_violation_is_ineligible_for_recommendation() -> None:
    inputs = [
        _input("low", total_tokens_s=90.0, p95_latency_s=0.4, raw=_slo_raw({"min_throughput_tokens_per_sec": 100})),
        _input("ok", total_tokens_s=120.0, p95_latency_s=0.7, raw=_slo_raw({"min_throughput_tokens_per_sec": 100})),
    ]

    scores, result = score_recommendation_inputs(inputs, goal=RecommendationGoal.LATENCY)
    score_by_id = {score.candidate_id: score for score in scores}

    assert result.recommended_candidate_id == "ok"
    assert "slo_min_throughput_tokens_per_sec_not_met" in score_by_id["low"].disqualifiers


def test_slo_failed_request_rate_violation_is_ineligible_for_recommendation() -> None:
    inputs = [
        _input(
            "flaky",
            total_tokens_s=160.0,
            p95_latency_s=0.5,
            successful_requests=8,
            failed_requests=2,
            total_requests=10,
            raw=_slo_raw({"max_failed_request_rate": 0.1}),
        ),
        _input("ok", total_tokens_s=100.0, p95_latency_s=0.6, raw=_slo_raw({"max_failed_request_rate": 0.1})),
    ]

    scores, result = score_recommendation_inputs(inputs, goal=RecommendationGoal.THROUGHPUT)
    score_by_id = {score.candidate_id: score for score in scores}

    assert result.recommended_candidate_id == "ok"
    assert "slo_max_failed_request_rate_exceeded" in score_by_id["flaky"].disqualifiers


def test_slo_ttft_and_tpot_constraints_require_metrics() -> None:
    inputs = [_input("missing", total_tokens_s=100.0, p95_latency_s=0.5, raw=_slo_raw({"ttft_ms": 100, "tpot_ms": 20}))]

    scores, result = score_recommendation_inputs(inputs, goal=RecommendationGoal.THROUGHPUT)

    assert result.recommended_candidate_id is None
    assert "slo_ttft_ms_missing" in scores[0].disqualifiers
    assert "slo_tpot_ms_missing" in scores[0].disqualifiers


def test_warning_propagation_is_preserved_in_recommendation() -> None:
    inputs = [_input("c1", total_tokens_s=100.0, p95_latency_s=1.0, warnings=["telemetry degraded"])]

    _, result = score_recommendation_inputs(
        inputs,
        goal=RecommendationGoal.THROUGHPUT,
        extra_warnings=[ATTACH_MODE_LIMITATION],
    )

    assert "telemetry degraded" in result.warnings
    assert ATTACH_MODE_LIMITATION in result.warnings


def test_deterministic_ranking_uses_candidate_id_tiebreaker() -> None:
    inputs = [
        _input("c2", total_tokens_s=100.0, p95_latency_s=1.0),
        _input("c1", total_tokens_s=100.0, p95_latency_s=1.0),
    ]

    scores, _ = score_recommendation_inputs(inputs, goal=RecommendationGoal.THROUGHPUT)

    assert [score.candidate_id for score in scores] == ["c1", "c2"]


def test_throughput_goal_uses_power_as_small_tiebreaker_when_available() -> None:
    inputs = [
        _input("wasteful", total_tokens_s=101.0, p95_latency_s=1.0, tokens_per_second_per_watt=0.3, joules_per_token=0.05),
        _input("efficient", total_tokens_s=100.0, p95_latency_s=1.0, tokens_per_second_per_watt=2.0, joules_per_token=0.006),
    ]

    _, result = score_recommendation_inputs(inputs, goal=RecommendationGoal.THROUGHPUT)

    assert result.recommended_candidate_id == "efficient"
    assert result.score_weights["power"] == 0.05


def test_latency_goal_uses_power_as_small_tiebreaker_when_available() -> None:
    inputs = [
        _input("wasteful", total_tokens_s=100.0, p95_latency_s=0.99, tokens_per_second_per_watt=0.3, joules_per_token=0.05),
        _input("efficient", total_tokens_s=100.0, p95_latency_s=1.0, tokens_per_second_per_watt=2.0, joules_per_token=0.006),
    ]

    _, result = score_recommendation_inputs(inputs, goal=RecommendationGoal.LATENCY)

    assert result.recommended_candidate_id == "efficient"
    assert result.score_weights["power"] == 0.05


def test_pareto_frontier_uses_power_when_available() -> None:
    inputs = [
        _input("dominator", total_tokens_s=120.0, p95_latency_s=0.8, tokens_per_second_per_watt=2.0, joules_per_token=0.006),
        _input("dominated", total_tokens_s=100.0, p95_latency_s=1.0, tokens_per_second_per_watt=1.0, joules_per_token=0.01),
        _input("low-latency", total_tokens_s=80.0, p95_latency_s=0.4, tokens_per_second_per_watt=1.2, joules_per_token=0.008),
    ]

    scores, result = score_recommendation_inputs(inputs, goal=RecommendationGoal.BALANCED)

    pareto_ids = {row["candidate_id"] for row in result.pareto_frontier}
    assert "dominator" in pareto_ids
    assert "low-latency" in pareto_ids
    assert "dominated" not in pareto_ids
    assert next(score for score in scores if score.candidate_id == "dominator").pareto_optimal is True


def test_alternative_recommendations_include_objective_winners() -> None:
    inputs = [
        _input("throughput", total_tokens_s=150.0, p95_latency_s=1.4, tokens_per_second_per_watt=0.8, joules_per_token=0.02),
        _input("latency", total_tokens_s=80.0, p95_latency_s=0.3, tokens_per_second_per_watt=1.0, joules_per_token=0.01),
        _input("efficiency", total_tokens_s=100.0, p95_latency_s=0.8, tokens_per_second_per_watt=2.5, joules_per_token=0.005),
    ]

    _, result = score_recommendation_inputs(inputs, goal=RecommendationGoal.BALANCED)

    assert result.alternative_recommendations["throughput"]["candidate_id"] == "throughput"
    assert result.alternative_recommendations["latency"]["candidate_id"] == "latency"
    assert result.alternative_recommendations["efficiency"]["candidate_id"] == "efficiency"


def test_recommendation_confidence_high_with_good_telemetry_and_clear_margin() -> None:
    inputs = [
        _input("winner", total_tokens_s=160.0, p95_latency_s=0.5, tokens_per_second_per_watt=2.0, joules_per_token=0.005, telemetry_quality="good"),
        _input("runner", total_tokens_s=80.0, p95_latency_s=1.5, tokens_per_second_per_watt=0.8, joules_per_token=0.02, telemetry_quality="good"),
    ]

    _, result = score_recommendation_inputs(inputs, goal=RecommendationGoal.BALANCED)

    assert result.recommended_candidate_id == "winner"
    assert result.confidence_level == "high"
    assert any("Telemetry quality" in reason for reason in result.confidence_reasons)


def test_recommendation_confidence_limited_when_telemetry_is_poor() -> None:
    inputs = [
        _input("winner", total_tokens_s=160.0, p95_latency_s=0.5, tokens_per_second_per_watt=2.0, joules_per_token=0.005, telemetry_quality="poor"),
        _input("runner", total_tokens_s=80.0, p95_latency_s=1.5, tokens_per_second_per_watt=0.8, joules_per_token=0.02, telemetry_quality="good"),
    ]

    _, result = score_recommendation_inputs(inputs, goal=RecommendationGoal.BALANCED)

    assert result.recommended_candidate_id == "winner"
    assert result.confidence_level == "low"
    assert any("poor" in reason for reason in result.confidence_reasons)


def test_recommendation_confidence_mentions_missing_utilization_capability() -> None:
    inputs = [
        _input(
            "winner",
            total_tokens_s=160.0,
            p95_latency_s=0.5,
            tokens_per_second_per_watt=2.0,
            joules_per_token=0.005,
            telemetry_quality="good",
            telemetry_capabilities={
                "available_fields": ["power", "temperature", "clocks"],
                "unavailable_fields": ["gpu_utilization", "memory_utilization"],
            },
        ),
        _input("runner", total_tokens_s=80.0, p95_latency_s=1.5, tokens_per_second_per_watt=0.8, joules_per_token=0.02),
    ]

    _, result = score_recommendation_inputs(inputs, goal=RecommendationGoal.BALANCED)

    assert result.confidence_level == "high"
    assert any("GPU utilization was unavailable" in reason for reason in result.confidence_reasons)


def test_generate_attach_mode_candidates_auto_falls_back_to_sweep(tmp_path) -> None:
    candidates, source, warnings = generate_attach_mode_candidates(
        base_url="http://127.0.0.1:8080/v1",
        model="model-path",
        backend="vllm",
        system="sys",
        total_gpus=1,
        isl=512,
        osl=128,
        ttft=2000.0,
        tpot=30.0,
        candidate_source="auto",
        top_k=2,
        run_dir=tmp_path,
        aiconfigurator_runner=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("missing aiconfigurator")),
    )

    assert source == "sweep"
    assert len(candidates) == 6
    assert any("AIConfigurator candidate generation unavailable" in warning for warning in warnings)


def test_generate_attach_mode_candidate_set_aiconfigurator_notes_are_metadata(tmp_path) -> None:
    result = generate_attach_mode_candidate_set(
        base_url="http://127.0.0.1:8080/v1",
        model="model-path",
        backend="vllm",
        system="sys",
        total_gpus=1,
        isl=512,
        osl=128,
        ttft=2000.0,
        tpot=30.0,
        candidate_source="aiconfigurator",
        top_k=1,
        concurrency_sweep=(16, 32),
        disable_sweep=True,
        run_dir=tmp_path,
        aiconfigurator_runner=_fake_aic_runner,
    )

    assert result.resolved_source == "aiconfigurator"
    assert len(result.candidates) == 1
    assert result.warnings == []
    assert any("AIConfigurator candidates were loaded" in note for note in result.metadata_notes)


def test_generate_sweep_candidates() -> None:
    candidates = generate_sweep_candidates(
        model="model-path",
        backend="sglang",
        system="sys",
        isl=256,
        osl=64,
        concurrencies=(16, 32, 128),
    )

    assert [candidate.candidate_id for candidate in candidates] == ["sweep-c016", "sweep-c032", "sweep-c128"]
    assert [candidate.source for candidate in candidates] == ["heuristic_sweep", "heuristic_sweep", "heuristic_sweep"]
    assert [candidate.concurrency for candidate in candidates] == [16, 32, 128]


def test_parse_concurrency_sweep() -> None:
    assert _parse_concurrency_sweep("16,32, 128") == (16, 32, 128)
    with pytest.raises(SystemExit):
        _parse_concurrency_sweep("16,bad")


def test_generate_heuristic_candidates_respects_top_k() -> None:
    candidates = generate_heuristic_candidates(
        model="model-path",
        backend="sglang",
        system="sys",
        isl=256,
        osl=64,
        top_k=3,
    )

    assert [candidate.concurrency for candidate in candidates] == [16, 32, 64]


def test_recommend_attach_mode_writes_artifacts_with_heuristic_candidates(tmp_path) -> None:
    run = recommend_attach_mode(
        base_url="http://127.0.0.1:8080/v1",
        model="model-path",
        backend="vllm",
        system="sys",
        total_gpus=1,
        isl=32,
        osl=8,
        ttft=None,
        tpot=None,
        goal=RecommendationGoal.THROUGHPUT,
        telemetry="none",
        out_dir=tmp_path / "recommendations",
        candidate_source="heuristic",
        top_k=2,
        request_fn=_ok_request,
    )

    assert (run.run_dir / "recommendation.json").exists()
    assert (run.run_dir / "scores.jsonl").exists()
    assert (run.run_dir / "summary.json").exists()
    payload = json.loads((run.run_dir / "recommendation.json").read_text(encoding="utf-8"))
    assert payload["recommended_candidate_id"] is not None
    assert ATTACH_MODE_LIMITATION in payload["warnings"]
    assert "candidate_table" in payload
    assert payload["candidate_count"] == 2
    assert payload["valid_candidate_count"] == 2
    assert "pareto_frontier" in payload
    assert (run.run_dir / "pareto_frontier.json").exists()
    assert (run.run_dir / "pareto_frontier.csv").exists()


def test_attach_preflight_writes_plan_without_endpoint_request(tmp_path) -> None:
    run = build_attach_preflight(
        base_url="http://127.0.0.1:65535/v1",
        model="model-path",
        backend="vllm",
        system="sys",
        total_gpus=1,
        isl=32,
        osl=8,
        ttft=None,
        tpot=None,
        goal=RecommendationGoal.THROUGHPUT,
        telemetry="none",
        out_dir=tmp_path / "recommendations",
        candidate_source="heuristic",
        top_k=2,
    )

    payload = json.loads((run.run_dir / "preflight.json").read_text(encoding="utf-8"))
    text = (run.run_dir / "preflight.txt").read_text(encoding="utf-8")

    assert payload["mode"] == "attach"
    assert payload["safety"]["will_call_endpoint"] is False
    assert payload["safety"]["will_launch_servers"] is False
    assert payload["candidates"]["valid_count"] == 2
    assert payload["budget"]["planned_workload_measurements"] == 2
    assert (run.run_dir / "plan" / "evaluation_plans.jsonl").exists()
    assert not (run.run_dir / "evaluation").exists()
    assert "will call endpoint: no" in text


def test_recommend_attach_mode_aiconfigurator_check_is_ok(tmp_path) -> None:
    run = recommend_attach_mode(
        base_url="http://127.0.0.1:8080/v1",
        model="model-path",
        backend="vllm",
        system="sys",
        total_gpus=1,
        isl=32,
        osl=8,
        ttft=None,
        tpot=None,
        goal=RecommendationGoal.THROUGHPUT,
        telemetry="none",
        out_dir=tmp_path / "recommendations",
        candidate_source="aiconfigurator",
        top_k=1,
        request_fn=_ok_request,
        aiconfigurator_runner=_fake_aic_runner,
    )

    check = next(item for item in run.result.checks if item.name == "candidate_generation")
    assert check.status == "ok"
    assert any("AIConfigurator candidates were loaded" in note for note in run.result.metadata_notes)
    assert not any("AIConfigurator candidates were loaded" in reason for reason in run.result.selection_reasons)


def test_balanced_scoring_with_multiple_sweep_candidates_has_comparative_reasoning() -> None:
    inputs = [
        _input("sweep-c064", source="heuristic_sweep", concurrency=64, total_tokens_s=90.0, p95_latency_s=0.6),
        _input("sweep-c128", source="heuristic_sweep", concurrency=128, total_tokens_s=120.0, p95_latency_s=0.8),
        _input("sweep-c512", source="heuristic_sweep", concurrency=512, total_tokens_s=125.0, p95_latency_s=2.0),
    ]

    _, result = score_recommendation_inputs(inputs, goal=RecommendationGoal.BALANCED)

    assert result.was_comparative is True
    assert result.recommended_candidate_id == "sweep-c128"
    assert any("highest balanced score" in reason for reason in result.selection_reasons)
    assert any("compared with sweep-c512" in reason for reason in result.selection_reasons)


def test_single_candidate_recommendation_is_marked_as_validation() -> None:
    _, result = score_recommendation_inputs(
        [_input("sweep-c064", source="heuristic_sweep", concurrency=64, total_tokens_s=90.0, p95_latency_s=0.6)],
        goal=RecommendationGoal.BALANCED,
    )

    assert result.was_comparative is False
    assert any("validation result rather than a comparative search" in reason for reason in result.selection_reasons)


def test_recommend_cli_smoke_with_mocked_runner(monkeypatch, tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    def fake_recommend_attach_mode(**kwargs) -> RecommendationRun:
        run_dir = tmp_path / "recommendations" / "mocked"
        run_dir.mkdir(parents=True, exist_ok=True)
        result = RecommendationResult(
            recommended_candidate_id="heuristic-rank-0001",
            goal="balanced",
            selected_score=RecommendationScore(
                candidate_id="heuristic-rank-0001",
                goal="balanced",
                throughput_score=1.0,
                latency_score=1.0,
                efficiency_score=None,
                reliability_score=1.0,
                prediction_accuracy_score=None,
                balanced_score=1.0,
                final_score=1.0,
                reasons=["mocked"],
                disqualifiers=[],
            ),
            selected_config=ServeCandidate(
                candidate_id="heuristic-rank-0001",
                rank=1,
                source="heuristic",
                model="model-path",
                backend="vllm",
            ),
            selected_serve_command="vllm serve model-path",
            selected_benchmark_plan=EndpointBenchmarkPlan(
                candidate_id="heuristic-rank-0001",
                base_url="http://127.0.0.1:8080/v1",
                model="model-path",
                concurrency=16,
                num_requests=128,
                max_tokens=8,
                expected_input_tokens=32,
                expected_output_tokens=8,
            ),
            predicted_metrics={},
            measured_metrics={"total_tokens_s": 100.0, "request_rate_req_s": 10.0, "p95_latency_s": 0.5},
            telemetry_metrics={},
            comparison_metrics={},
            warnings=[],
            artifacts={"run_dir": str(run_dir), "report_txt": str(run_dir / "report.txt")},
            alternatives=[],
            rationale=["mocked"],
        )
        return RecommendationRun(run_dir=run_dir, result=result, scores=[], summary={}, checks=[], failed=False)

    monkeypatch.setattr("serve_optimize.cli.recommend_attach_mode", fake_recommend_attach_mode)

    main(
        [
            "recommend",
            "--base-url",
            "http://127.0.0.1:8080/v1",
            "--model",
            "model-path",
            "--backend",
            "vllm",
            "--system",
            "sys",
            "--total-gpus",
            "1",
            "--isl",
            "32",
            "--osl",
            "8",
            "--goal",
            "balanced",
            "--out",
            str(tmp_path / "recommendations"),
        ]
    )
    output = capsys.readouterr().out
    assert "Serve Optimize Recommendation" in output
    assert "Recommended Configuration" in output


def _score(candidate_id: str, final_score: float | None) -> RecommendationScore:
    return RecommendationScore(
        candidate_id=candidate_id,
        goal=RecommendationGoal.BALANCED.value,
        throughput_score=final_score,
        latency_score=final_score,
        efficiency_score=None,
        reliability_score=1.0 if final_score is not None else None,
        prediction_accuracy_score=None,
        balanced_score=final_score,
        final_score=final_score,
        power_score=None,
    )


def _input(
    candidate_id: str,
    *,
    total_tokens_s: float,
    p95_latency_s: float | None,
    source: str = "test",
    concurrency: int = 16,
    request_rate: float = 10.0,
    successful_requests: int = 4,
    failed_requests: int = 0,
    total_requests: int = 4,
    tokens_per_second_per_watt: float | None = None,
    joules_per_token: float | None = None,
    telemetry_quality: str | None = None,
    telemetry_capabilities: dict[str, object] | None = None,
    warnings: list[str] | None = None,
    raw: dict[str, object] | None = None,
) -> RecommendationInput:
    candidate = ServeCandidate(
        candidate_id=candidate_id,
        rank=1,
        source=source,
        model="model-path",
        backend="vllm",
        concurrency=concurrency,
        request_rate=request_rate,
        predicted_tokens_s=100.0,
        predicted_request_latency_ms=1000.0,
        raw=raw or {},
    )
    return RecommendationInput(
        candidate_id=candidate_id,
        candidate_rank=1,
        candidate_source=source,
        model="model-path",
        backend="vllm",
        candidate=candidate,
        serve_plan=VllmServePlan(
            candidate_id=candidate_id,
            model="model-path",
            host="127.0.0.1",
            port=8080,
            dtype="bfloat16",
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            max_model_len=2048,
            gpu_memory_utilization=0.9,
            command=["vllm", "serve", "model-path"],
            shell_command="vllm serve model-path",
        ),
        benchmark_plan=EndpointBenchmarkPlan(
            candidate_id=candidate_id,
            base_url="http://127.0.0.1:8080/v1",
            model="model-path",
            concurrency=concurrency,
            num_requests=128,
            max_tokens=8,
            expected_input_tokens=32,
            expected_output_tokens=8,
        ),
        predicted_metrics={
            "predicted_tokens_s": 100.0,
            "predicted_request_rate": request_rate,
            "predicted_request_latency_ms": 1000.0,
        },
        measured_metrics={
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "request_rate_req_s": request_rate,
            "total_tokens_s": total_tokens_s,
            "avg_latency_s": p95_latency_s,
            "p95_latency_s": p95_latency_s,
        },
        telemetry_metrics={
            "tokens_per_second_per_watt": tokens_per_second_per_watt,
            "joules_per_token": joules_per_token,
            "average_power_watts": total_tokens_s / tokens_per_second_per_watt if tokens_per_second_per_watt else None,
            "telemetry_quality": telemetry_quality or ("good" if tokens_per_second_per_watt is not None else "unavailable"),
            "power_sample_count": 10 if tokens_per_second_per_watt is not None else 0,
            "power_sampling_rate_hz": 5.0 if tokens_per_second_per_watt is not None else None,
            "power_stddev_watts": 5.0 if tokens_per_second_per_watt is not None else None,
            "telemetry_capabilities": telemetry_capabilities or {},
        },
        comparison_metrics={
            "measured_avg_latency_ms": p95_latency_s * 1000.0 if p95_latency_s is not None else None,
        },
        warnings=warnings or [],
    )


def _slo_raw(constraints: dict[str, object]) -> dict[str, object]:
    return {
        "workload_profile": {
            "profile_name": "short",
            "slo_constraints": constraints,
        },
        "slo_constraints": constraints,
    }


def _ok_request(config, request_id: int) -> RequestRecord:
    return RequestRecord(
        request_id=request_id,
        start_time=float(request_id),
        end_time=float(request_id) + 0.1,
        latency_s=0.1,
        status="ok",
        prompt_tokens=8,
        completion_tokens=8,
        total_tokens=16,
    )


def _fake_aic_runner(**kwargs) -> AIConfiguratorRun:
    output_dir = kwargs["output_dir"]
    agg_dir = output_dir / "run" / "agg"
    agg_dir.mkdir(parents=True, exist_ok=True)
    (agg_dir / "best_config_topn.csv").write_text(
        "model,isl,osl,concurrency,request_rate,bs,global_bs,ttft,tpot,request_latency,"
        "tokens/s,tokens/s/gpu,tokens/s/user,tp,pp,dp,parallel,gemm,kvcache,fmha,backend,version,system,memory,power_w\n"
        "model-path,32,8,16,10.0,16,16,50.0,10.0,1000.0,"
        "100.0,100.0,6.25,1,1,1,tp1pp1dp1,bfloat16,bfloat16,bfloat16,vllm,0.1,sys,4.0,0.0\n",
        encoding="utf-8",
    )
    return AIConfiguratorRun(command=["aiconfigurator"], returncode=0, stdout="", stderr="")
