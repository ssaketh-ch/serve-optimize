
import pytest
from rich.console import Console

from serve_optimize.cli import main
from serve_optimize.recommendation import RecommendationRun
from serve_optimize.reporting import RichReporter, format_recommendation_report
from serve_optimize.schemas import (
    CheckRecord,
    EndpointBenchmarkPlan,
    RecommendationResult,
    RecommendationScore,
    ServeCandidate,
)


def test_format_recommendation_report_with_telemetry() -> None:
    report = format_recommendation_report(_result())

    assert "Serve Optimize Recommendation" in report
    assert "Recommended Configuration" in report
    assert "Prediction vs Measurement" in report
    assert "Candidates Evaluated" in report
    assert "Scoring Policy" in report
    assert "Pareto Frontier" in report
    assert "Alternative Recommendations" in report
    assert "Power and Efficiency" in report
    assert "Resource Telemetry" in report
    assert "Telemetry Capabilities" in report
    assert "Recommendation Confidence" in report
    assert "average power" in report
    assert "missing fields" in report
    assert "gpu_util_percent" in report
    assert "gpu_utilization" in report
    assert "nearly flat" in report
    assert "Checks Performed" in report


def test_format_recommendation_report_without_telemetry() -> None:
    result = _result()
    result = RecommendationResult(
        recommended_candidate_id=result.recommended_candidate_id,
        goal=result.goal,
        selected_score=result.selected_score,
        selected_config=result.selected_config,
        selected_serve_command=result.selected_serve_command,
        selected_benchmark_plan=result.selected_benchmark_plan,
        status=result.status,
        mode=result.mode,
        endpoint=result.endpoint,
        model=result.model,
        backend=result.backend,
        candidate_source=result.candidate_source,
        telemetry_requested="auto",
        telemetry_provider=None,
        predicted_metrics=result.predicted_metrics,
        measured_metrics=result.measured_metrics,
        telemetry_metrics={},
        comparison_metrics=result.comparison_metrics,
        warnings=result.warnings,
        checks=result.checks,
        limitations=result.limitations,
        artifacts=result.artifacts,
        alternatives=result.alternatives,
        rationale=result.rationale,
    )

    report = format_recommendation_report(result)

    assert "Power telemetry: unavailable" in report
    assert "average power" not in report


def test_warning_and_check_display_in_report() -> None:
    report = format_recommendation_report(_result(warnings=["telemetry degraded"]))

    assert "telemetry degraded" in report
    assert "[WARN]" in report
    assert "Metadata Notes" not in report


def test_rich_reporter_renders_sections() -> None:
    console = Console(record=True, width=120)
    reporter = RichReporter(console=console)
    reporter.render(result=_result())
    output = console.export_text()

    assert "Serve Optimize Recommendation" in output
    assert "Recommended Configuration" in output
    assert "Candidates Evaluated" in output
    assert "Scoring Policy" in output
    assert "Pareto Frontier" in output
    assert "Alternative Recommendations" in output
    assert "Resource Telemetry" in output
    assert "Checks Performed" in output


def test_recommend_cli_writes_report_txt(monkeypatch, tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    run_dir = tmp_path / "recommendations" / "mocked"
    run_dir.mkdir(parents=True, exist_ok=True)
    result = _result()
    result = RecommendationResult(
        recommended_candidate_id=result.recommended_candidate_id,
        goal=result.goal,
        selected_score=result.selected_score,
        selected_config=result.selected_config,
        selected_serve_command=result.selected_serve_command,
        selected_benchmark_plan=result.selected_benchmark_plan,
        status=result.status,
        mode=result.mode,
        endpoint=result.endpoint,
        model=result.model,
        backend=result.backend,
        candidate_source=result.candidate_source,
        telemetry_requested=result.telemetry_requested,
        telemetry_provider=result.telemetry_provider,
        predicted_metrics=result.predicted_metrics,
        measured_metrics=result.measured_metrics,
        telemetry_metrics=result.telemetry_metrics,
        comparison_metrics=result.comparison_metrics,
        warnings=result.warnings,
        checks=result.checks,
        limitations=result.limitations,
        artifacts={**result.artifacts, "run_dir": str(run_dir), "report_txt": str(run_dir / "report.txt")},
        alternatives=result.alternatives,
        rationale=result.rationale,
    )

    def fake_recommend_attach_mode(**kwargs) -> RecommendationRun:
        return RecommendationRun(
            run_dir=run_dir,
            result=result,
            scores=result.alternatives,
            summary={"status": "success"},
            checks=result.checks,
            failed=False,
        )

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
    capsys.readouterr()
    report_path = run_dir / "report.txt"
    assert report_path.exists()
    assert "Serve Optimize Recommendation" in report_path.read_text(encoding="utf-8")


def test_telemetry_check_help_includes_duration(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["telemetry-check", "--help"])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "--duration" in output
    assert "--interval" in output


def _result(warnings: list[str] | None = None) -> RecommendationResult:
    selected_config = ServeCandidate(
        candidate_id="aic-rank-0001",
        rank=1,
        source="aiconfigurator",
        model="model-path",
        backend="vllm",
        parallel="tp1pp1dp1",
    )
    selected_score = RecommendationScore(
        candidate_id="aic-rank-0001",
        goal="balanced",
        throughput_score=0.9,
        latency_score=0.8,
        efficiency_score=0.7,
        power_score=0.72,
        reliability_score=1.0,
        prediction_accuracy_score=0.85,
        balanced_score=0.84,
        final_score=0.84,
        weights_used={"throughput": 0.30, "latency": 0.25, "power": 0.30, "reliability": 0.15},
        score_breakdown={
            "throughput_score": 0.9,
            "latency_score": 0.8,
            "efficiency_score": 0.7,
            "power_score": 0.72,
            "reliability_score": 1.0,
        },
        pareto_optimal=True,
        reasons=["Balanced scoring combined measured throughput, p95 latency, reliability, and efficiency."],
        disqualifiers=[],
    )
    return RecommendationResult(
        recommended_candidate_id="aic-rank-0001",
        goal="balanced",
        selected_score=selected_score,
        selected_config=selected_config,
        selected_serve_command="vllm serve model-path --host 127.0.0.1 --port 8080",
        selected_benchmark_plan=EndpointBenchmarkPlan(
            candidate_id="aic-rank-0001",
            base_url="http://127.0.0.1:8080/v1",
            model="model-path",
            concurrency=512,
            num_requests=1024,
            max_tokens=128,
            expected_input_tokens=512,
            expected_output_tokens=128,
        ),
        status="success",
        mode="attach",
        endpoint="http://127.0.0.1:8080/v1",
        model="model-path",
        backend="vllm",
        candidate_source="aiconfigurator",
        telemetry_requested="auto",
        telemetry_provider="nvml",
        candidate_count=2,
        valid_candidate_count=2,
        was_comparative=True,
        predicted_metrics={
            "predicted_tokens_s": 35612.52,
            "predicted_request_rate": 278.22,
            "predicted_request_latency_ms": 1496.106,
        },
        measured_metrics={
            "failed_requests": 0,
            "total_requests": 1024,
            "request_rate_req_s": 250.0,
            "total_tokens_s": 30077.97,
            "avg_latency_s": 1.7,
            "p95_latency_s": 10.528,
        },
        telemetry_metrics={
            "average_power_watts": 177.169,
            "peak_power_watts": 184.0,
            "power_stddev_watts": 3.2,
            "power_sampling_rate_hz": 5.0,
            "energy_joules": 1000.0,
            "joules_per_token": 0.006,
            "tokens_per_second_per_watt": 169.77,
            "power_sample_count": 12,
            "average_gpu_util_percent": 91.2,
            "max_gpu_util_percent": 97.0,
            "average_memory_util_percent": 65.0,
            "average_temperature_c": 67.0,
            "power_limit_watts": 600.0,
            "telemetry_quality": "good",
            "missing_fields": ["gpu_util_percent"],
            "telemetry_warnings": ["GPU utilization was unavailable, so power interpretation is limited."],
            "telemetry_notes": ["Power readings are nearly flat. Efficiency differences may mainly reflect throughput differences."],
            "telemetry_capabilities": {
                "provider": "nvml",
                "device_name": "Generic GPU",
                "available_fields": ["power", "temperature", "memory_usage", "clocks", "power_limit"],
                "unavailable_fields": ["gpu_utilization", "memory_utilization", "throttle_reasons"],
                "notes": ["Platform reports utilization metrics as unavailable."],
                "warnings": [],
            },
        },
        comparison_metrics={
            "predicted_tokens_s": 35612.52,
            "measured_total_tokens_s": 30077.97,
            "measured_over_predicted_tokens_ratio": 0.845,
            "measured_over_predicted_request_rate_ratio": 0.899,
            "latency_delta_percent": 13.6,
        },
        score_weights={"throughput": 0.30, "latency": 0.25, "power": 0.30, "reliability": 0.15},
        score_breakdown=selected_score.score_breakdown,
        ranked_candidates=[],
        pareto_frontier=[
            {
                "candidate_id": "aic-rank-0001",
                "source": "aiconfigurator",
                "concurrency": 512,
                "total_tokens_s": 30077.97,
                "p95_latency_s": 10.528,
                "average_power_watts": 177.169,
                "joules_per_token": 0.006,
                "tokens_per_second_per_watt": 169.77,
                "score": 0.84,
            }
        ],
        alternative_recommendations={
            "throughput": {
                "candidate_id": "aic-rank-0001",
                "concurrency": 512,
                "total_tokens_s": 30077.97,
                "p95_latency_s": 10.528,
                "tokens_per_second_per_watt": 169.77,
                "reason": "highest measured total_tokens_s",
            }
        },
        telemetry_used_in_scoring=True,
        power_aware=True,
        confidence_level="high",
        confidence_reasons=["Telemetry quality for the selected candidate was good."],
        selection_reasons=[
            "Selected aic-rank-0001 because it had the highest balanced score among 2 valid candidates.",
            "It completed with 0 failed requests.",
        ],
        metadata_notes=[],
        warnings=warnings or ["Attach Mode cannot verify server launch flags"],
        checks=[
            CheckRecord(name="endpoint_health", status="ok", message="Endpoint health check passed."),
            CheckRecord(name="telemetry", status="ok", message="Telemetry collected using nvml."),
            CheckRecord(name="attach_mode_caveat", status="warn", message="Attach Mode cannot verify server launch flags"),
        ],
        limitations=[
            "Attach Mode benchmarks the currently running endpoint.",
            "It cannot prove the endpoint was launched with the generated serve command.",
        ],
        artifacts={
            "run_dir": "results/recommendations/recommend-1",
            "report_txt": "results/recommendations/recommend-1/report.txt",
            "recommendation_json": "results/recommendations/recommend-1/recommendation.json",
            "scores_jsonl": "results/recommendations/recommend-1/scores.jsonl",
            "pareto_frontier_json": "results/recommendations/recommend-1/pareto_frontier.json",
            "pareto_frontier_csv": "results/recommendations/recommend-1/pareto_frontier.csv",
            "summary_json": "results/recommendations/recommend-1/summary.json",
            "metadata_json": "results/recommendations/recommend-1/metadata.json",
            "evaluation_run_dir": "results/recommendations/recommend-1/evaluation",
            "telemetry_capabilities_json": "results/recommendations/recommend-1/evaluation/per_candidate/aic-rank-0001/telemetry_capabilities.json",
        },
        candidate_table=[
            {
                "candidate_id": "aic-rank-0001",
                "source": "aiconfigurator",
                "concurrency": 512,
                "total_tokens_s": 30077.97,
                "p95_latency_s": 10.528,
                "average_power_watts": 177.169,
                "joules_per_token": 0.006,
                "tokens_per_second_per_watt": 169.77,
                "failed_requests": 0,
                "throughput_score": 0.9,
                "latency_score": 0.8,
                "power_score": 0.72,
                "reliability_score": 1.0,
                "score": 0.84,
                "pareto_optimal": True,
                "status": "eligible",
            },
            {
                "candidate_id": "sweep-c128",
                "source": "heuristic_sweep",
                "concurrency": 128,
                "total_tokens_s": 28000.0,
                "p95_latency_s": 4.0,
                "average_power_watts": 170.0,
                "joules_per_token": 0.0065,
                "tokens_per_second_per_watt": 164.7,
                "failed_requests": 0,
                "throughput_score": 0.84,
                "latency_score": 1.0,
                "power_score": 0.70,
                "reliability_score": 1.0,
                "score": 0.82,
                "pareto_optimal": True,
                "status": "eligible",
            },
        ],
        alternatives=[selected_score],
        rationale=["It had the best balanced score among evaluated candidates."],
    )
