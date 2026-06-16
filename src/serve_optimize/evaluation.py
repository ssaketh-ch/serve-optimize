"""Execute generated evaluation plans against already-running endpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .endpoint_benchmark import RequestFn, TelemetryCollectorFactory, make_run_id, run_endpoint_benchmark
from .io import write_json
from .schemas import (
    CandidateEvaluationPlan,
    EndpointBenchmarkConfig,
    EndpointBenchmarkPlan,
    EndpointBenchmarkSummary,
    ServeCandidate,
    VllmServePlan,
)


@dataclass(frozen=True)
class EvaluationResult:
    run_dir: Path
    summary: dict[str, object]
    failed: bool


def run_evaluation_plan_dir(
    plan_dir: Path,
    out_dir: Path,
    limit_candidates: int | None = None,
    override_concurrency: int | None = None,
    override_num_requests: int | None = None,
    timeout_s: float = 120.0,
    telemetry: str = "none",
    request_fn: RequestFn | None = None,
    telemetry_collector_factory: TelemetryCollectorFactory | None = None,
) -> EvaluationResult:
    plans = load_evaluation_plans(plan_dir)
    if limit_candidates is not None:
        plans = plans[:limit_candidates]

    run_id = make_run_id(prefix="evaluation")
    run_dir = out_dir / run_id
    per_candidate_dir = run_dir / "per_candidate"
    per_candidate_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    total_requests = 0
    successful_requests = 0
    failed_requests = 0

    for plan in plans:
        if plan.benchmark_plan is None:
            rows.append(
                {
                    "candidate_id": plan.candidate_id,
                    "rank": plan.rank,
                    "status": "skipped",
                    "reason": "Evaluation plan does not contain a benchmark plan.",
                }
            )
            continue
        benchmark_plan = _apply_overrides(plan.benchmark_plan, override_concurrency, override_num_requests)
        config = EndpointBenchmarkConfig(
            run_id=plan.candidate_id,
            base_url=benchmark_plan.base_url,
            model=benchmark_plan.model,
            concurrency=benchmark_plan.concurrency,
            num_requests=benchmark_plan.num_requests,
            max_tokens=benchmark_plan.max_tokens,
            prompt=make_synthetic_prompt(benchmark_plan.expected_input_tokens),
            timeout_s=timeout_s,
            telemetry=telemetry,
        )
        run = run_endpoint_benchmark(
            config=config,
            out_dir=per_candidate_dir,
            prediction=None,
            hardware=None,
            request_fn=request_fn,
            telemetry_collector_factory=telemetry_collector_factory,
        )
        comparison = compare_candidate_to_summary(plan.candidate, run.summary)
        write_json(run.run_dir / "comparison.json", comparison)
        row = {
            "candidate_id": plan.candidate_id,
            "rank": plan.rank,
            "concurrency": config.concurrency,
            "summary_path": str(run.run_dir / "summary.json"),
            "telemetry_summary_path": str(run.run_dir / "telemetry_summary.json") if telemetry != "none" else None,
            "telemetry_capabilities_path": str(run.run_dir / "telemetry_capabilities.json") if telemetry != "none" else None,
            "comparison_path": str(run.run_dir / "comparison.json"),
            **comparison,
            "failed_requests": run.summary.failed_requests,
            "average_power_watts": run.summary.average_power_watts,
            "peak_power_watts": run.summary.peak_power_watts,
            "power_sample_count": run.summary.power_sample_count,
            "telemetry_quality": run.summary.telemetry_quality,
            "telemetry_available": run.summary.telemetry_available,
            "power_stddev_watts": run.summary.power_stddev_watts,
            "power_sampling_rate_hz": run.summary.power_sampling_rate_hz,
        }
        rows.append(row)
        total_requests += run.summary.total_requests
        successful_requests += run.summary.successful_requests
        failed_requests += run.summary.failed_requests

    failed = total_requests > 0 and successful_requests == 0
    summary: dict[str, object] = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "plan_dir": str(plan_dir),
        "candidate_count": len(plans),
        "total_requests": total_requests,
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "all_requests_failed": failed,
        "candidates": rows,
    }
    write_json(run_dir / "summary.json", summary)
    return EvaluationResult(run_dir=run_dir, summary=summary, failed=failed)


def load_evaluation_plans(plan_dir: Path) -> list[CandidateEvaluationPlan]:
    path = plan_dir / "evaluation_plans.jsonl"
    plans = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            plans.append(_evaluation_plan_from_dict(json.loads(line)))
    return plans


def compare_candidate_to_summary(candidate: ServeCandidate, summary: EndpointBenchmarkSummary) -> dict[str, float | int | str | None]:
    predicted_tokens = candidate.predicted_tokens_s
    predicted_rate = candidate.request_rate
    predicted_latency_ms = candidate.predicted_request_latency_ms
    measured_avg_latency_ms = summary.avg_latency_s * 1000.0 if summary.avg_latency_s is not None else None
    measured_p95_latency_ms = summary.p95_latency_s * 1000.0 if summary.p95_latency_s is not None else None
    return {
        "candidate_id": candidate.candidate_id,
        "predicted_tokens_s": predicted_tokens,
        "measured_total_tokens_s": summary.total_tokens_s,
        "measured_over_predicted_tokens_ratio": _ratio(summary.total_tokens_s, predicted_tokens),
        "predicted_request_rate": predicted_rate,
        "measured_request_rate": summary.request_rate_req_s,
        "measured_over_predicted_request_rate_ratio": _ratio(summary.request_rate_req_s, predicted_rate),
        "predicted_request_latency_ms": predicted_latency_ms,
        "measured_avg_latency_ms": measured_avg_latency_ms,
        "measured_p95_latency_ms": measured_p95_latency_ms,
        "latency_delta_percent": _delta_percent(measured_avg_latency_ms, predicted_latency_ms),
    }


def make_synthetic_prompt(expected_input_tokens: int | None) -> str:
    token_count = max(1, expected_input_tokens or 32)
    return " ".join(["benchmark"] * token_count)


def _apply_overrides(
    plan: EndpointBenchmarkPlan,
    override_concurrency: int | None,
    override_num_requests: int | None,
) -> EndpointBenchmarkPlan:
    return EndpointBenchmarkPlan(
        candidate_id=plan.candidate_id,
        base_url=plan.base_url,
        model=plan.model,
        concurrency=override_concurrency or plan.concurrency,
        num_requests=override_num_requests or plan.num_requests,
        max_tokens=plan.max_tokens,
        expected_input_tokens=plan.expected_input_tokens,
        expected_output_tokens=plan.expected_output_tokens,
    )


def _evaluation_plan_from_dict(row: dict[str, object]) -> CandidateEvaluationPlan:
    candidate = _candidate_from_dict(row["candidate"])
    benchmark_plan = None
    benchmark_row = row.get("benchmark_plan")
    if isinstance(benchmark_row, dict):
        benchmark_plan = _benchmark_plan_from_dict(benchmark_row)
    serve_plan = None
    serve_row = row.get("serve_plan")
    if isinstance(serve_row, dict):
        serve_plan = _serve_plan_from_dict(serve_row)
    return CandidateEvaluationPlan(
        candidate_id=str(row.get("candidate_id") or candidate.candidate_id),
        rank=int(row.get("rank") or candidate.rank),
        candidate=candidate,
        serve_plan=serve_plan,
        benchmark_plan=benchmark_plan,
        notes=[str(item) for item in row.get("notes", [])] if isinstance(row.get("notes"), list) else [],
    )


def _candidate_from_dict(row: object) -> ServeCandidate:
    if not isinstance(row, dict):
        raise ValueError("Evaluation plan candidate must be an object.")
    return ServeCandidate(**row)


def _benchmark_plan_from_dict(row: dict[str, object]) -> EndpointBenchmarkPlan:
    return EndpointBenchmarkPlan(
        candidate_id=str(row["candidate_id"]),
        base_url=str(row["base_url"]),
        model=str(row["model"]),
        concurrency=int(row["concurrency"]),
        num_requests=int(row["num_requests"]),
        max_tokens=int(row["max_tokens"]),
        expected_input_tokens=_optional_int(row.get("expected_input_tokens")),
        expected_output_tokens=_optional_int(row.get("expected_output_tokens")),
    )


def _serve_plan_from_dict(row: dict[str, object]) -> VllmServePlan:
    return VllmServePlan(
        candidate_id=str(row["candidate_id"]),
        model=str(row["model"]),
        host=str(row["host"]),
        port=int(row["port"]),
        dtype=str(row["dtype"]),
        tensor_parallel_size=int(row["tensor_parallel_size"]),
        pipeline_parallel_size=_optional_int(row.get("pipeline_parallel_size")),
        max_model_len=int(row["max_model_len"]),
        gpu_memory_utilization=float(row["gpu_memory_utilization"]),
        command=[str(item) for item in row.get("command", [])],
        shell_command=str(row["shell_command"]),
    )


def _optional_int(value: object) -> int | None:
    return int(value) if value is not None else None


def _ratio(measured: float | None, predicted: float | None) -> float | None:
    if measured is None or predicted in {None, 0}:
        return None
    return measured / predicted


def _delta_percent(measured: float | None, predicted: float | None) -> float | None:
    if measured is None or predicted in {None, 0}:
        return None
    return (measured - predicted) / predicted * 100.0
