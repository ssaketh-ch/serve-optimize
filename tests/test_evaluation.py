import json

import pytest

from serve_optimize.cli import main
from serve_optimize.endpoint_benchmark import summarize_requests
from serve_optimize.evaluation import (
    compare_candidate_to_summary,
    load_evaluation_plans,
    run_evaluation_plan_dir,
)
from serve_optimize.schemas import EndpointBenchmarkConfig, PowerSampleRecord, RequestRecord
from serve_optimize.telemetry import TelemetryCapture


def test_load_evaluation_plans_jsonl(tmp_path) -> None:
    plan_dir = _write_plan_dir(tmp_path)

    plans = load_evaluation_plans(plan_dir)

    assert len(plans) == 1
    assert plans[0].candidate_id == "aic-rank-0001"
    assert plans[0].candidate.predicted_tokens_s == pytest.approx(1000.0)
    assert plans[0].benchmark_plan is not None
    assert plans[0].benchmark_plan.concurrency == 2


def test_run_evaluation_plan_with_mocked_endpoint_writes_artifacts(tmp_path) -> None:
    plan_dir = _write_plan_dir(tmp_path)

    result = run_evaluation_plan_dir(
        plan_dir=plan_dir,
        out_dir=tmp_path / "evaluations",
        request_fn=_ok_request,
    )

    candidate_dir = result.run_dir / "per_candidate" / "aic-rank-0001"
    assert (result.run_dir / "summary.json").exists()
    assert (candidate_dir / "config.json").exists()
    assert (candidate_dir / "requests.jsonl").exists()
    assert (candidate_dir / "summary.json").exists()
    assert (candidate_dir / "comparison.json").exists()

    summary = json.loads((candidate_dir / "summary.json").read_text(encoding="utf-8"))
    comparison = json.loads((candidate_dir / "comparison.json").read_text(encoding="utf-8"))
    assert summary["total_requests"] == 4
    assert summary["failed_requests"] == 0
    assert comparison["predicted_tokens_s"] == pytest.approx(1000.0)
    assert comparison["measured_total_tokens_s"] > 0
    assert result.summary["successful_requests"] == 4
    assert result.failed is False


def test_run_evaluation_plan_with_mocked_telemetry_writes_power_artifacts(tmp_path) -> None:
    result = run_evaluation_plan_dir(
        plan_dir=_write_plan_dir(tmp_path),
        out_dir=tmp_path / "evaluations",
        telemetry="nvml",
        request_fn=_ok_request,
        telemetry_collector_factory=lambda telemetry, device_index, interval_s: _FakeCollector(
            TelemetryCapture(
                provider=telemetry,
                samples=[PowerSampleRecord(0.0, "measured", 120.0, telemetry, device_index=device_index)],
                warnings=[],
            )
        ),
    )

    candidate_dir = result.run_dir / "per_candidate" / "aic-rank-0001"
    summary = json.loads((candidate_dir / "summary.json").read_text(encoding="utf-8"))
    assert (candidate_dir / "power_samples.jsonl").exists()
    assert (candidate_dir / "telemetry_summary.json").exists()
    assert summary["power_sample_count"] == 1
    assert summary["average_power_watts"] == pytest.approx(120.0)
    assert summary["telemetry_provider"] == "nvml"
    assert summary["telemetry_quality"] == "poor"


def test_evaluation_comparison_metric_calculation(tmp_path) -> None:
    plan = load_evaluation_plans(_write_plan_dir(tmp_path))[0]
    summary = summarize_requests(
        "run",
        [
            RequestRecord(0, 0.0, 1.0, 1.0, "ok", prompt_tokens=10, completion_tokens=20, total_tokens=30),
            RequestRecord(1, 0.0, 3.0, 3.0, "ok", prompt_tokens=10, completion_tokens=20, total_tokens=30),
        ],
        wall_time_s=2.0,
    )

    comparison = compare_candidate_to_summary(plan.candidate, summary)

    assert comparison["predicted_tokens_s"] == pytest.approx(1000.0)
    assert comparison["measured_total_tokens_s"] == pytest.approx(30.0)
    assert comparison["measured_over_predicted_tokens_ratio"] == pytest.approx(0.03)
    assert comparison["predicted_request_rate"] == pytest.approx(10.0)
    assert comparison["measured_request_rate"] == pytest.approx(1.0)
    assert comparison["measured_over_predicted_request_rate_ratio"] == pytest.approx(0.1)
    assert comparison["predicted_request_latency_ms"] == pytest.approx(1000.0)
    assert comparison["measured_avg_latency_ms"] == pytest.approx(2000.0)
    assert comparison["measured_p95_latency_ms"] == pytest.approx(2900.0)
    assert comparison["latency_delta_percent"] == pytest.approx(100.0)


def test_evaluation_failure_accounting(tmp_path) -> None:
    result = run_evaluation_plan_dir(
        plan_dir=_write_plan_dir(tmp_path),
        out_dir=tmp_path / "evaluations",
        request_fn=_failed_request,
    )

    candidate_dir = result.run_dir / "per_candidate" / "aic-rank-0001"
    summary = json.loads((candidate_dir / "summary.json").read_text(encoding="utf-8"))
    top_summary = json.loads((result.run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["successful_requests"] == 0
    assert summary["failed_requests"] == 4
    assert top_summary["all_requests_failed"] is True
    assert result.failed is True


def test_evaluation_telemetry_failure_keeps_benchmark_results(tmp_path) -> None:
    result = run_evaluation_plan_dir(
        plan_dir=_write_plan_dir(tmp_path),
        out_dir=tmp_path / "evaluations",
        telemetry="auto",
        request_fn=_ok_request,
        telemetry_collector_factory=lambda telemetry, device_index, interval_s: _FakeCollector(
            TelemetryCapture(
                provider="nvidia-smi",
                samples=[PowerSampleRecord(0.0, "measured", None, "nvidia-smi", error="mock failure")],
                warnings=["nvidia-smi telemetry sample failed: mock failure"],
            )
        ),
    )

    candidate_dir = result.run_dir / "per_candidate" / "aic-rank-0001"
    summary = json.loads((candidate_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["successful_requests"] == 4
    assert summary["average_power_watts"] is None
    assert "nvidia-smi telemetry sample failed: mock failure" in summary["warnings"]
    assert summary["telemetry_quality"] == "poor"
    assert result.failed is False


def test_evaluation_overrides(tmp_path) -> None:
    result = run_evaluation_plan_dir(
        plan_dir=_write_plan_dir(tmp_path),
        out_dir=tmp_path / "evaluations",
        override_concurrency=1,
        override_num_requests=2,
        request_fn=_ok_request,
    )

    config = json.loads(
        (result.run_dir / "per_candidate" / "aic-rank-0001" / "config.json").read_text(encoding="utf-8")
    )
    assert config["concurrency"] == 1
    assert config["num_requests"] == 2


def test_run_evaluation_plan_cli_smoke() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["run-evaluation-plan", "--help"])
    assert exc.value.code == 0


def test_run_evaluation_plan_help_includes_telemetry_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["run-evaluation-plan", "--help"])
    output = capsys.readouterr().out
    assert "--telemetry" in output


def _ok_request(_config: EndpointBenchmarkConfig, request_id: int) -> RequestRecord:
    return RequestRecord(
        request_id=request_id,
        start_time=float(request_id),
        end_time=float(request_id) + 0.1,
        latency_s=0.1,
        status="ok",
        prompt_tokens=5,
        completion_tokens=10,
        total_tokens=15,
    )


def _failed_request(_config: EndpointBenchmarkConfig, request_id: int) -> RequestRecord:
    return RequestRecord(
        request_id=request_id,
        start_time=float(request_id),
        end_time=float(request_id) + 0.1,
        latency_s=0.1,
        status="error",
        error="mock failure",
    )


def _write_plan_dir(tmp_path):
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    row = {
        "candidate_id": "aic-rank-0001",
        "rank": 1,
        "candidate": {
            "candidate_id": "aic-rank-0001",
            "rank": 1,
            "source": "test",
            "model": "model-path",
            "backend": "vllm",
            "backend_version": "0.1",
            "system": "test-system",
            "isl": 32,
            "osl": 8,
            "prefix": 0,
            "concurrency": 2,
            "request_rate": 10.0,
            "batch_size": 2,
            "global_batch_size": 2,
            "tp": 1,
            "pp": 1,
            "dp": 1,
            "moe_tp": 1,
            "moe_ep": 1,
            "parallel": "tp1pp1dp1",
            "gemm": "bfloat16",
            "kvcache": "bfloat16",
            "fmha": "bfloat16",
            "moe": "bfloat16",
            "comm": "half",
            "predicted_ttft_ms": 50.0,
            "predicted_tpot_ms": 10.0,
            "predicted_request_latency_ms": 1000.0,
            "predicted_seq_s": 10.0,
            "predicted_tokens_s": 1000.0,
            "predicted_tokens_s_per_gpu": 1000.0,
            "predicted_tokens_s_per_user": 500.0,
            "predicted_memory_gb": 12.5,
            "predicted_power_w": 0.0,
            "raw": {"model": "model-path"},
        },
        "serve_plan": None,
        "benchmark_plan": {
            "candidate_id": "aic-rank-0001",
            "base_url": "http://127.0.0.1:8080/v1",
            "model": "model-path",
            "concurrency": 2,
            "num_requests": 4,
            "max_tokens": 8,
            "expected_input_tokens": 32,
            "expected_output_tokens": 8,
        },
        "notes": [],
    }
    (plan_dir / "evaluation_plans.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    return plan_dir


class _FakeCollector:
    def __init__(self, capture: TelemetryCapture):
        self.capture = capture

    def start(self) -> None:
        return None

    def stop(self) -> TelemetryCapture:
        return self.capture
