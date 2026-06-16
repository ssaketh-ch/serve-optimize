"""Attach Mode recommendation orchestration and scoring."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .aiconfig_parser import parse_aiconfigurator_best_configs
from .aiconfig_plans import candidate_to_endpoint_benchmark_plan, candidate_to_vllm_serve_plan
from .aiconfigurator_bridge import AIConfiguratorRun, run_aiconfigurator
from .endpoint_benchmark import (
    RequestFn,
    TelemetryCollectorFactory,
    make_run_id,
    send_chat_completion_request,
)
from .evaluation import run_evaluation_plan_dir
from .hardware import detect_hardware
from .io import write_json, write_jsonl
from .preflight import PreflightRun, write_preflight_artifacts
from .schemas import (
    CandidateEvaluationPlan,
    CheckRecord,
    EndpointBenchmarkConfig,
    EndpointBenchmarkPlan,
    EndpointBenchmarkSummary,
    RecommendationGoal,
    RecommendationInput,
    RecommendationResult,
    RecommendationScore,
    ServeCandidate,
    VllmServePlan,
    WorkloadProfile,
)
from .workloads import slo_disqualifiers, slo_note, workload_profile_to_payload

# Goal weights stay explicit so scoring remains inspectable in artifacts and tests.
THROUGHPUT_WEIGHTS = {
    "throughput": 0.85,
    "reliability": 0.10,
    "latency": 0.05,
}
THROUGHPUT_WITH_POWER_WEIGHTS = {
    "throughput": 0.80,
    "reliability": 0.10,
    "latency": 0.05,
    "power": 0.05,
}
LATENCY_WEIGHTS = {
    "latency": 0.85,
    "reliability": 0.15,
}
LATENCY_WITH_POWER_WEIGHTS = {
    "latency": 0.80,
    "reliability": 0.15,
    "power": 0.05,
}
EFFICIENCY_WEIGHTS = {
    "power": 0.70,
    "reliability": 0.20,
    "latency": 0.10,
}

# Balanced mode makes power a first-class signal when it is available.
BALANCED_WITH_POWER_WEIGHTS = {
    "throughput": 0.30,
    "latency": 0.25,
    "power": 0.30,
    "reliability": 0.15,
}
BALANCED_NO_POWER_WEIGHTS = {
    "throughput": 0.45,
    "latency": 0.30,
    "reliability": 0.25,
}

DEFAULT_HEURISTIC_CONCURRENCIES = (16, 32, 64, 128)
DEFAULT_SWEEP_CONCURRENCIES = (16, 32, 64, 128, 256, 512)
CONFIDENCE_CLEAR_WIN_MARGIN = 0.05
CONFIDENCE_SMALL_WIN_MARGIN = 0.02
ATTACH_MODE_LIMITATION = (
    "Attach Mode benchmarks the running endpoint. It cannot prove that the running server matches the "
    "generated serve command for a candidate."
)
GROSS_ENERGY_LIMITATION = (
    "Energy metrics currently use gross active energy from average measured power over wall time. "
    "Idle subtraction is not implemented yet."
)


AIConfiguratorRunner = Callable[..., AIConfiguratorRun]


@dataclass(frozen=True)
class RecommendationRun:
    run_dir: Path
    result: RecommendationResult
    scores: list[RecommendationScore]
    summary: dict[str, object]
    checks: list[CheckRecord]
    failed: bool


@dataclass(frozen=True)
class CandidateGenerationResult:
    candidates: list[ServeCandidate]
    resolved_source: str
    metadata_notes: list[str]
    warnings: list[str]


def build_attach_preflight(
    *,
    base_url: str,
    model: str,
    backend: str,
    system: str,
    total_gpus: int,
    isl: int,
    osl: int,
    ttft: float | None,
    tpot: float | None,
    goal: RecommendationGoal,
    telemetry: str,
    out_dir: Path,
    candidate_source: str = "auto",
    top_k: int = 4,
    concurrency_sweep: tuple[int, ...] = DEFAULT_SWEEP_CONCURRENCIES,
    disable_sweep: bool = False,
    override_concurrency: int | None = None,
    override_num_requests: int | None = None,
    timeout_s: float = 120.0,
    allow_efficiency_fallback: bool = False,
    aiconfigurator_runner: AIConfiguratorRunner = run_aiconfigurator,
    workload_profile: WorkloadProfile | None = None,
) -> PreflightRun:
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    run_id = make_run_id(prefix="recommend-preflight")
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    hardware = detect_hardware()
    candidate_generation = generate_attach_mode_candidate_set(
        base_url=base_url,
        model=model,
        backend=backend,
        system=system,
        total_gpus=total_gpus,
        isl=isl,
        osl=osl,
        ttft=ttft,
        tpot=tpot,
        candidate_source=candidate_source,
        top_k=top_k,
        concurrency_sweep=concurrency_sweep,
        disable_sweep=disable_sweep,
        run_dir=run_dir,
        aiconfigurator_runner=aiconfigurator_runner,
    )
    candidates = _with_workload_profile(candidate_generation.candidates, workload_profile)
    plan_dir, evaluation_plans = _write_attach_mode_plan_bundle(
        run_dir=run_dir,
        candidates=candidates,
        base_url=base_url,
        backend=backend,
    )
    benchmark_plans = [plan.benchmark_plan for plan in evaluation_plans if plan.benchmark_plan is not None]
    total_requests = sum(
        override_num_requests if override_num_requests is not None else plan.num_requests
        for plan in benchmark_plans
    )
    max_concurrency = max(
        [
            override_concurrency if override_concurrency is not None else plan.concurrency
            for plan in benchmark_plans
        ],
        default=None,
    )
    payload = {
        "schema_version": "serve-optimize-preflight/v1",
        "run_id": run_id,
        "mode": "attach",
        "status": "planned",
        "dry_run": True,
        "backend": backend,
        "model": model,
        "goal": goal.value,
        "telemetry": telemetry,
        "endpoint": base_url,
        "candidate_source": candidate_generation.resolved_source,
        "workload_profile": workload_profile_to_payload(workload_profile),
        "created_hardware_snapshot": hardware,
        "candidates": {
            "requested_limit": top_k,
            "generated_count": len(candidates),
            "valid_count": len(evaluation_plans),
            "rejected_count": 0,
            "source": candidate_generation.resolved_source,
        },
        "budget": {
            "launch_group_count": 0,
            "planned_workload_measurements": len(benchmark_plans),
            "planned_requests": total_requests,
            "max_concurrency": max_concurrency,
            "timeout_s": timeout_s,
            "allow_efficiency_fallback": allow_efficiency_fallback,
        },
        "evidence": {
            "db_path": None,
            "write_enabled": False,
            "exact_reuse": "not used by Attach Mode",
        },
        "safety": {
            "will_call_endpoint": False,
            "will_launch_servers": False,
            "will_write_measured_evidence": False,
        },
        "outputs": {
            "plan_dir": str(plan_dir),
            "candidate_plan": str(plan_dir / "evaluation_plans.jsonl"),
        },
        "guidance": {
            "execute": "Run the same command without --dry-run to perform endpoint health checks and measurements.",
            "repeat": "Attach Mode creates a new measured run each time. Use repeatability after collecting multiple runs.",
            "resume": "Attach Mode resume is not implemented. Keep the preflight and measured run directories as separate artifacts.",
        },
        "warnings": candidate_generation.warnings,
        "notes": candidate_generation.metadata_notes + [note for note in [slo_note(workload_profile)] if note] + [ATTACH_MODE_LIMITATION],
    }
    return write_preflight_artifacts(run_dir, payload)


def recommend_attach_mode(
    *,
    base_url: str,
    model: str,
    backend: str,
    system: str,
    total_gpus: int,
    isl: int,
    osl: int,
    ttft: float | None,
    tpot: float | None,
    goal: RecommendationGoal,
    telemetry: str,
    out_dir: Path,
    candidate_source: str = "auto",
    top_k: int = 4,
    concurrency_sweep: tuple[int, ...] = DEFAULT_SWEEP_CONCURRENCIES,
    disable_sweep: bool = False,
    override_concurrency: int | None = None,
    override_num_requests: int | None = None,
    timeout_s: float = 120.0,
    allow_efficiency_fallback: bool = False,
    request_fn: RequestFn | None = None,
    telemetry_collector_factory: TelemetryCollectorFactory | None = None,
    aiconfigurator_runner: AIConfiguratorRunner = run_aiconfigurator,
    workload_profile: WorkloadProfile | None = None,
) -> RecommendationRun:
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    hardware = detect_hardware()
    request_fn = request_fn or send_chat_completion_request
    checks: list[CheckRecord] = []
    _check_endpoint_health(base_url=base_url, model=model, timeout_s=timeout_s, request_fn=request_fn)
    checks.append(CheckRecord(name="endpoint_health", status="ok", message="Endpoint health check passed."))

    run_id = make_run_id(prefix="recommend")
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    candidate_generation = generate_attach_mode_candidate_set(
        base_url=base_url,
        model=model,
        backend=backend,
        system=system,
        total_gpus=total_gpus,
        isl=isl,
        osl=osl,
        ttft=ttft,
        tpot=tpot,
        candidate_source=candidate_source,
        top_k=top_k,
        concurrency_sweep=concurrency_sweep,
        disable_sweep=disable_sweep,
        run_dir=run_dir,
        aiconfigurator_runner=aiconfigurator_runner,
    )
    candidates = _with_workload_profile(candidate_generation.candidates, workload_profile)
    resolved_source = candidate_generation.resolved_source
    if not candidates:
        raise RuntimeError("No candidates were generated for Attach Mode recommendation.")
    checks.append(
        CheckRecord(
            name="candidate_generation",
            status="ok" if candidates else "fail",
            message=f"Candidate generation completed using {resolved_source}.",
            details={"candidate_count": len(candidates)},
        )
    )

    plan_dir, evaluation_plans = _write_attach_mode_plan_bundle(
        run_dir=run_dir,
        candidates=candidates,
        base_url=base_url,
        backend=backend,
    )
    evaluation = run_evaluation_plan_dir(
        plan_dir=plan_dir,
        out_dir=run_dir / "evaluation",
        override_concurrency=override_concurrency,
        override_num_requests=override_num_requests,
        timeout_s=timeout_s,
        telemetry=telemetry,
        request_fn=request_fn,
        telemetry_collector_factory=telemetry_collector_factory,
    )
    checks.append(
        _benchmark_check(
            candidate_count=len(evaluation_plans),
            total_requests=int(evaluation.summary.get("total_requests", 0)),
            failed_requests=int(evaluation.summary.get("failed_requests", 0)),
            all_failed=evaluation.failed,
        )
    )
    inputs = build_recommendation_inputs(evaluation_plans, evaluation.run_dir)
    checks.append(_telemetry_check(telemetry=telemetry, inputs=inputs))
    scores, recommendation = score_recommendation_inputs(
        inputs,
        goal=goal,
        allow_efficiency_fallback=allow_efficiency_fallback,
        extra_warnings=candidate_generation.warnings + [ATTACH_MODE_LIMITATION],
        metadata_notes=candidate_generation.metadata_notes + [note for note in [slo_note(workload_profile)] if note],
    )
    checks.append(
        CheckRecord(
            name="scoring",
            status="ok" if recommendation.recommended_candidate_id is not None else "warn",
            message=(
                f"Scoring completed for goal '{goal.value}'."
                if recommendation.recommended_candidate_id is not None
                else f"Scoring completed but no candidate could be recommended for goal '{goal.value}'."
            ),
        )
    )
    checks.append(CheckRecord(name="attach_mode_caveat", status="warn", message=ATTACH_MODE_LIMITATION))
    confidence_level, confidence_reasons = _recommendation_confidence(
        recommendation=recommendation,
        inputs=inputs,
        scores=scores,
    )

    recommendation = RecommendationResult(
        recommended_candidate_id=recommendation.recommended_candidate_id,
        goal=recommendation.goal,
        selected_score=recommendation.selected_score,
        selected_config=recommendation.selected_config,
        selected_serve_command=recommendation.selected_serve_command,
        selected_benchmark_plan=recommendation.selected_benchmark_plan,
        status="success" if recommendation.recommended_candidate_id is not None else "warning",
        mode="attach",
        endpoint=base_url,
        model=model,
        backend=backend,
        candidate_source=resolved_source,
        telemetry_requested=telemetry,
        telemetry_provider=_selected_telemetry_provider(inputs, recommendation.recommended_candidate_id),
        candidate_count=recommendation.candidate_count,
        valid_candidate_count=recommendation.valid_candidate_count,
        was_comparative=recommendation.was_comparative,
        predicted_metrics=recommendation.predicted_metrics,
        measured_metrics=recommendation.measured_metrics,
        telemetry_metrics=recommendation.telemetry_metrics,
        comparison_metrics=recommendation.comparison_metrics,
        score_weights=recommendation.score_weights,
        score_breakdown=recommendation.score_breakdown,
        ranked_candidates=recommendation.ranked_candidates,
        pareto_frontier=recommendation.pareto_frontier,
        alternative_recommendations=recommendation.alternative_recommendations,
        telemetry_used_in_scoring=recommendation.telemetry_used_in_scoring,
        power_aware=recommendation.power_aware,
        power_missing_reason=recommendation.power_missing_reason,
        confidence_level=confidence_level,
        confidence_reasons=confidence_reasons,
        selection_reasons=recommendation.selection_reasons,
        metadata_notes=recommendation.metadata_notes,
        warnings=recommendation.warnings,
        checks=checks,
        limitations=[ATTACH_MODE_LIMITATION, GROSS_ENERGY_LIMITATION],
        artifacts={},
        candidate_table=recommendation.candidate_table,
        alternatives=recommendation.alternatives,
        rationale=recommendation.rationale,
        evaluated_set_fidelity=recommendation.evaluated_set_fidelity,
        optimizer_quality=recommendation.optimizer_quality,
    )

    artifacts = {
        "run_dir": str(run_dir),
        "report_txt": str(run_dir / "report.txt"),
        "recommendation_json": str(run_dir / "recommendation.json"),
        "scores_jsonl": str(run_dir / "scores.jsonl"),
        "pareto_frontier_json": str(run_dir / "pareto_frontier.json"),
        "pareto_frontier_csv": str(run_dir / "pareto_frontier.csv"),
        "summary_json": str(run_dir / "summary.json"),
        "metadata_json": str(run_dir / "metadata.json"),
        "plan_dir": str(plan_dir),
        "evaluation_run_dir": str(evaluation.run_dir),
    }
    selected_telemetry_summary_path = _selected_telemetry_summary_path(inputs, recommendation.recommended_candidate_id)
    if selected_telemetry_summary_path is not None:
        artifacts["telemetry_summary_json"] = selected_telemetry_summary_path
    selected_telemetry_capabilities_path = _selected_telemetry_capabilities_path(inputs, recommendation.recommended_candidate_id)
    if selected_telemetry_capabilities_path is not None:
        artifacts["telemetry_capabilities_json"] = selected_telemetry_capabilities_path
    recommendation = RecommendationResult(
        recommended_candidate_id=recommendation.recommended_candidate_id,
        goal=recommendation.goal,
        selected_score=recommendation.selected_score,
        selected_config=recommendation.selected_config,
        selected_serve_command=recommendation.selected_serve_command,
        selected_benchmark_plan=recommendation.selected_benchmark_plan,
        status=recommendation.status,
        mode=recommendation.mode,
        endpoint=recommendation.endpoint,
        model=recommendation.model,
        backend=recommendation.backend,
        candidate_source=recommendation.candidate_source,
        telemetry_requested=recommendation.telemetry_requested,
        telemetry_provider=recommendation.telemetry_provider,
        candidate_count=recommendation.candidate_count,
        valid_candidate_count=recommendation.valid_candidate_count,
        was_comparative=recommendation.was_comparative,
        predicted_metrics=recommendation.predicted_metrics,
        measured_metrics=recommendation.measured_metrics,
        telemetry_metrics=recommendation.telemetry_metrics,
        comparison_metrics=recommendation.comparison_metrics,
        score_weights=recommendation.score_weights,
        score_breakdown=recommendation.score_breakdown,
        ranked_candidates=recommendation.ranked_candidates,
        pareto_frontier=recommendation.pareto_frontier,
        alternative_recommendations=recommendation.alternative_recommendations,
        telemetry_used_in_scoring=recommendation.telemetry_used_in_scoring,
        power_aware=recommendation.power_aware,
        power_missing_reason=recommendation.power_missing_reason,
        confidence_level=recommendation.confidence_level,
        confidence_reasons=recommendation.confidence_reasons,
        selection_reasons=recommendation.selection_reasons,
        metadata_notes=recommendation.metadata_notes,
        warnings=recommendation.warnings,
        checks=recommendation.checks,
        limitations=recommendation.limitations,
        artifacts=artifacts,
        candidate_table=recommendation.candidate_table,
        alternatives=recommendation.alternatives,
        rationale=recommendation.rationale,
        evaluated_set_fidelity=recommendation.evaluated_set_fidelity,
        optimizer_quality=recommendation.optimizer_quality,
    )

    write_json(run_dir / "recommendation.json", recommendation)
    write_jsonl(run_dir / "scores.jsonl", scores)
    write_json(run_dir / "pareto_frontier.json", recommendation.pareto_frontier)
    _write_pareto_csv(run_dir / "pareto_frontier.csv", recommendation.pareto_frontier)
    summary = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "goal": goal.value,
        "candidate_source": resolved_source,
        "candidate_count": len(inputs),
        "valid_candidate_count": recommendation.valid_candidate_count,
        "was_comparative": recommendation.was_comparative,
        "telemetry": telemetry,
        "workload_profile": workload_profile_to_payload(workload_profile),
        "plan_dir": str(plan_dir),
        "evaluation_run_dir": str(evaluation.run_dir),
        "recommended_candidate_id": recommendation.recommended_candidate_id,
        "all_requests_failed": evaluation.failed,
        "warnings": recommendation.warnings,
        "status": recommendation.status,
        "telemetry_used_in_scoring": recommendation.telemetry_used_in_scoring,
        "power_aware": recommendation.power_aware,
        "confidence_level": recommendation.confidence_level,
        "confidence_reasons": recommendation.confidence_reasons,
        "pareto_candidate_count": len(recommendation.pareto_frontier),
    }
    write_json(run_dir / "summary.json", summary)
    metadata = {
        "schema_version": "attach-recommendation/v1",
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hardware": hardware,
        "artifact_files": sorted(path.name for path in run_dir.iterdir() if path.is_file()),
    }
    write_json(run_dir / "metadata.json", metadata)
    return RecommendationRun(
        run_dir=run_dir,
        result=recommendation,
        scores=scores,
        summary=summary,
        checks=checks,
        failed=evaluation.failed or recommendation.recommended_candidate_id is None,
    )


def generate_attach_mode_candidates(
    *,
    base_url: str,
    model: str,
    backend: str,
    system: str,
    total_gpus: int,
    isl: int,
    osl: int,
    ttft: float | None,
    tpot: float | None,
    candidate_source: str,
    top_k: int,
    run_dir: Path,
    aiconfigurator_runner: AIConfiguratorRunner,
) -> tuple[list[ServeCandidate], str, list[str]]:
    result = generate_attach_mode_candidate_set(
        base_url=base_url,
        model=model,
        backend=backend,
        system=system,
        total_gpus=total_gpus,
        isl=isl,
        osl=osl,
        ttft=ttft,
        tpot=tpot,
        candidate_source=candidate_source,
        top_k=top_k,
        concurrency_sweep=DEFAULT_SWEEP_CONCURRENCIES,
        disable_sweep=False,
        run_dir=run_dir,
        aiconfigurator_runner=aiconfigurator_runner,
    )
    return result.candidates, result.resolved_source, result.warnings


def generate_attach_mode_candidate_set(
    *,
    base_url: str,
    model: str,
    backend: str,
    system: str,
    total_gpus: int,
    isl: int,
    osl: int,
    ttft: float | None,
    tpot: float | None,
    candidate_source: str,
    top_k: int,
    concurrency_sweep: tuple[int, ...],
    disable_sweep: bool,
    run_dir: Path,
    aiconfigurator_runner: AIConfiguratorRunner,
) -> CandidateGenerationResult:
    del base_url
    warnings: list[str] = []
    metadata_notes: list[str] = []
    if candidate_source == "heuristic":
        return CandidateGenerationResult(
            candidates=generate_heuristic_candidates(model=model, backend=backend, system=system, isl=isl, osl=osl, top_k=top_k),
            resolved_source="heuristic",
            metadata_notes=["Generated heuristic candidates from user-provided Attach Mode inputs."],
            warnings=[],
        )
    if candidate_source == "sweep":
        return CandidateGenerationResult(
            candidates=generate_sweep_candidates(
                model=model,
                backend=backend,
                system=system,
                isl=isl,
                osl=osl,
                concurrencies=concurrency_sweep,
            ),
            resolved_source="sweep",
            metadata_notes=[f"Generated concurrency sweep candidates: {_join_ints(concurrency_sweep)}."],
            warnings=[],
        )
    if candidate_source == "aiconfigurator":
        candidates, aic_notes = _generate_aiconfigurator_candidates(
            model=model,
            backend=backend,
            system=system,
            total_gpus=total_gpus,
            isl=isl,
            osl=osl,
            ttft=ttft,
            tpot=tpot,
            top_k=top_k,
            run_dir=run_dir,
            aiconfigurator_runner=aiconfigurator_runner,
        )
        return CandidateGenerationResult(
            candidates=candidates,
            resolved_source="aiconfigurator",
            metadata_notes=aic_notes,
            warnings=[],
        )
    if candidate_source == "auto":
        try:
            candidates, aic_notes = _generate_aiconfigurator_candidates(
                model=model,
                backend=backend,
                system=system,
                total_gpus=total_gpus,
                isl=isl,
                osl=osl,
                ttft=ttft,
                tpot=tpot,
                top_k=top_k,
                run_dir=run_dir,
                aiconfigurator_runner=aiconfigurator_runner,
            )
            metadata_notes.extend(aic_notes)
            resolved_source = "aiconfigurator"
        except Exception as exc:
            warnings.append(f"AIConfigurator candidate generation unavailable: {exc}")
            candidates = []
            resolved_source = "auto"
        if not disable_sweep:
            sweep = generate_sweep_candidates(
                model=model,
                backend=backend,
                system=system,
                isl=isl,
                osl=osl,
                concurrencies=concurrency_sweep,
            )
            candidates.extend(sweep)
            metadata_notes.append(f"Included concurrency sweep candidates: {_join_ints(concurrency_sweep)}.")
            resolved_source = "aiconfigurator+sweep" if resolved_source == "aiconfigurator" else "sweep"
        if not candidates:
            metadata_notes.append("Falling back to heuristic candidates because no AIConfigurator or sweep candidates were available.")
            candidates = generate_heuristic_candidates(model=model, backend=backend, system=system, isl=isl, osl=osl, top_k=top_k)
            resolved_source = "heuristic"
        return CandidateGenerationResult(
            candidates=candidates,
            resolved_source=resolved_source,
            metadata_notes=metadata_notes,
            warnings=warnings,
        )
    raise ValueError(f"Unsupported candidate source: {candidate_source}")


def _with_workload_profile(candidates: list[ServeCandidate], profile: WorkloadProfile | None) -> list[ServeCandidate]:
    payload = workload_profile_to_payload(profile)
    if not payload:
        return candidates
    updated = []
    for candidate in candidates:
        raw = dict(candidate.raw or {})
        raw["workload_profile"] = payload
        if payload.get("slo_constraints"):
            raw["slo_constraints"] = payload["slo_constraints"]
        updated.append(replace(candidate, raw=raw))
    return updated


def generate_heuristic_candidates(
    *,
    model: str,
    backend: str,
    system: str,
    isl: int,
    osl: int,
    top_k: int,
    concurrencies: tuple[int, ...] = DEFAULT_HEURISTIC_CONCURRENCIES,
) -> list[ServeCandidate]:
    rows: list[ServeCandidate] = []
    for rank, concurrency in enumerate(concurrencies[:top_k], start=1):
        rows.append(
            ServeCandidate(
                candidate_id=f"heuristic-rank-{rank:04d}",
                rank=rank,
                source="heuristic",
                model=model,
                backend=backend,
                system=system,
                isl=isl,
                osl=osl,
                concurrency=concurrency,
                batch_size=concurrency,
                global_batch_size=concurrency,
                raw={"candidate_source": "heuristic", "concurrency_seed": concurrency},
            )
        )
    return rows


def generate_sweep_candidates(
    *,
    model: str,
    backend: str,
    system: str,
    isl: int,
    osl: int,
    concurrencies: tuple[int, ...] = DEFAULT_SWEEP_CONCURRENCIES,
) -> list[ServeCandidate]:
    rows: list[ServeCandidate] = []
    for rank, concurrency in enumerate(concurrencies, start=1):
        rows.append(
            ServeCandidate(
                candidate_id=f"sweep-c{concurrency:03d}",
                rank=rank,
                source="heuristic_sweep",
                model=model,
                backend=backend,
                system=system,
                isl=isl,
                osl=osl,
                concurrency=concurrency,
                batch_size=concurrency,
                global_batch_size=concurrency,
                raw={"candidate_source": "heuristic_sweep", "concurrency": concurrency},
            )
        )
    return rows


def build_recommendation_inputs(
    evaluation_plans: list[CandidateEvaluationPlan],
    evaluation_run_dir: Path,
) -> list[RecommendationInput]:
    plan_by_candidate = {plan.candidate_id: plan for plan in evaluation_plans}
    summary = json.loads((evaluation_run_dir / "summary.json").read_text(encoding="utf-8"))
    inputs: list[RecommendationInput] = []
    for row in summary.get("candidates", []):
        if not isinstance(row, dict):
            continue
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str):
            continue
        plan = plan_by_candidate.get(candidate_id)
        if plan is None:
            continue
        summary_path = row.get("summary_path")
        comparison_path = row.get("comparison_path")
        if not isinstance(summary_path, str):
            continue
        benchmark_summary = EndpointBenchmarkSummary(**json.loads(Path(summary_path).read_text(encoding="utf-8")))
        comparison_metrics = {}
        if isinstance(comparison_path, str) and Path(comparison_path).exists():
            comparison_metrics = json.loads(Path(comparison_path).read_text(encoding="utf-8"))
        telemetry_metrics = _telemetry_metrics(benchmark_summary)
        telemetry_summary_path = row.get("telemetry_summary_path")
        if isinstance(telemetry_summary_path, str):
            telemetry_metrics["telemetry_summary_path"] = telemetry_summary_path
        telemetry_capabilities_path = row.get("telemetry_capabilities_path")
        if isinstance(telemetry_capabilities_path, str):
            telemetry_metrics["telemetry_capabilities_path"] = telemetry_capabilities_path
        inputs.append(
            RecommendationInput(
                candidate_id=plan.candidate_id,
                candidate_rank=plan.rank,
                candidate_source=plan.candidate.source,
                model=plan.candidate.model,
                backend=plan.candidate.backend,
                candidate=plan.candidate,
                serve_plan=plan.serve_plan,
                benchmark_plan=plan.benchmark_plan,
                predicted_metrics=_predicted_metrics(plan.candidate),
                measured_metrics=_measured_metrics(benchmark_summary),
                telemetry_metrics=telemetry_metrics,
                comparison_metrics=comparison_metrics,
                warnings=list(plan.notes) + list(benchmark_summary.warnings),
            )
        )
    return inputs


def score_recommendation_inputs(
    inputs: list[RecommendationInput],
    *,
    goal: RecommendationGoal,
    allow_efficiency_fallback: bool = False,
    extra_warnings: list[str] | None = None,
    metadata_notes: list[str] | None = None,
) -> tuple[list[RecommendationScore], RecommendationResult]:
    warnings = list(extra_warnings or [])
    notes = list(metadata_notes or [])
    if not inputs:
        result = RecommendationResult(
            recommended_candidate_id=None,
            goal=goal.value,
            selected_score=None,
            selected_config=None,
            selected_serve_command=None,
            selected_benchmark_plan=None,
            status="warning",
            warnings=["No recommendation inputs were available."],
            selection_reasons=["No candidates were available for scoring."],
            confidence_level="low",
            confidence_reasons=["No candidates were available for scoring."],
            metadata_notes=notes,
            rationale=["No candidates were available for scoring."],
        )
        return [], result

    has_any_power = any(_positive_number(item.telemetry_metrics.get("tokens_per_second_per_watt")) for item in inputs)
    effective_goal = goal
    if goal == RecommendationGoal.EFFICIENCY and not has_any_power:
        if not allow_efficiency_fallback:
            warnings.append("Efficiency recommendation is unavailable because no candidate has usable power telemetry.")
            scores = [_make_unavailable_score(item, goal, "missing_power_telemetry") for item in inputs]
            result = RecommendationResult(
                recommended_candidate_id=None,
                goal=goal.value,
                selected_score=None,
                selected_config=None,
                selected_serve_command=None,
                selected_benchmark_plan=None,
                status="warning",
                warnings=warnings,
                alternatives=scores,
                selection_reasons=["Efficiency mode requires measured power telemetry."],
                metadata_notes=notes,
                candidate_count=len(inputs),
                valid_candidate_count=0,
                was_comparative=False,
                score_weights=_weights_for_goal(goal, has_any_power=False),
                ranked_candidates=_candidate_table(inputs, scores),
                pareto_frontier=[],
                alternative_recommendations={},
                telemetry_used_in_scoring=False,
                power_aware=False,
                power_missing_reason="No candidate had usable power telemetry.",
                confidence_level="low",
                confidence_reasons=["No usable power telemetry was available for the requested efficiency goal."],
                candidate_table=_candidate_table(inputs, scores),
                rationale=["Efficiency mode requires measured power telemetry."],
            )
            result = replace(
                result,
                optimizer_quality=compute_optimizer_quality(
                    goal=goal.value,
                    selected_candidate_id=None,
                    ranked_scores=scores,
                    candidate_table=result.candidate_table,
                ),
            )
            return scores, result
        effective_goal = RecommendationGoal.BALANCED
        warnings.append("Efficiency goal fell back to balanced scoring because no candidate had usable power telemetry.")

    throughput_scores = _normalize_higher(inputs, lambda item: _optional_float(item.measured_metrics.get("total_tokens_s")))
    latency_scores = _normalize_lower(inputs, lambda item: _latency_value(item))
    efficiency_scores = _normalize_higher(inputs, lambda item: _optional_float(item.telemetry_metrics.get("tokens_per_second_per_watt")))
    joules_scores = _normalize_lower(inputs, lambda item: _optional_float(item.telemetry_metrics.get("joules_per_token")))
    reliability_scores = {item.candidate_id: _reliability_score(item) for item in inputs}
    prediction_accuracy_scores = {item.candidate_id: _prediction_accuracy_score(item) for item in inputs}
    power_scores = {
        item.candidate_id: _power_score(efficiency_scores.get(item.candidate_id), joules_scores.get(item.candidate_id))
        for item in inputs
    }

    if effective_goal == RecommendationGoal.BALANCED and not has_any_power:
        warnings.append("Balanced scoring used performance-only weights because power telemetry was unavailable.")
    weights = _weights_for_goal(effective_goal, has_any_power=has_any_power)

    scores: list[RecommendationScore] = []
    for item in inputs:
        disqualifiers = _disqualifiers(item, effective_goal, has_any_power)
        reasons: list[str] = []
        missing_metric_penalties = _missing_metric_penalties(item, effective_goal, has_any_power)
        if disqualifiers:
            reasons.append("Candidate was not eligible for recommendation scoring.")
            scores.append(
                RecommendationScore(
                    candidate_id=item.candidate_id,
                    goal=goal.value,
                    throughput_score=_round_or_none(throughput_scores.get(item.candidate_id)),
                    latency_score=_round_or_none(latency_scores.get(item.candidate_id)),
                    efficiency_score=_round_or_none(efficiency_scores.get(item.candidate_id)),
                    reliability_score=_round_or_none(reliability_scores.get(item.candidate_id)),
                    prediction_accuracy_score=_round_or_none(prediction_accuracy_scores.get(item.candidate_id)),
                    balanced_score=None,
                    final_score=None,
                    power_score=_round_or_none(power_scores.get(item.candidate_id)),
                    weights_used=weights,
                    missing_metric_penalties=missing_metric_penalties,
                    score_breakdown=_score_breakdown(
                        throughput_scores.get(item.candidate_id),
                        latency_scores.get(item.candidate_id),
                        efficiency_scores.get(item.candidate_id),
                        power_scores.get(item.candidate_id),
                        reliability_scores.get(item.candidate_id),
                        prediction_accuracy_scores.get(item.candidate_id),
                    ),
                    reasons=reasons,
                    disqualifiers=disqualifiers,
                )
            )
            continue

        throughput_score = throughput_scores.get(item.candidate_id)
        latency_score = latency_scores.get(item.candidate_id)
        efficiency_score = efficiency_scores.get(item.candidate_id)
        power_score = power_scores.get(item.candidate_id)
        reliability_score = reliability_scores.get(item.candidate_id)
        prediction_accuracy_score = prediction_accuracy_scores.get(item.candidate_id)
        balanced_score = None

        if effective_goal == RecommendationGoal.THROUGHPUT:
            final_score = _weighted_sum(
                (
                    (throughput_score, weights["throughput"]),
                    (reliability_score, weights["reliability"]),
                    (latency_score, weights["latency"]),
                    (power_score, weights.get("power", 0.0)),
                )
            )
            reasons.append(
                "Throughput scoring emphasized measured total_tokens_s with reliability, p95 latency, and power as tie-breakers when available."
            )
        elif effective_goal == RecommendationGoal.LATENCY:
            final_score = _weighted_sum(
                (
                    (latency_score, weights["latency"]),
                    (reliability_score, weights["reliability"]),
                    (power_score, weights.get("power", 0.0)),
                )
            )
            reasons.append("Latency scoring emphasized measured p95 latency, penalized failures, and used power as a tie-breaker when available.")
        elif effective_goal == RecommendationGoal.EFFICIENCY:
            final_score = _weighted_sum(
                (
                    (power_score, weights["power"]),
                    (reliability_score, weights["reliability"]),
                    (latency_score, weights["latency"]),
                )
            )
            reasons.append("Efficiency scoring emphasized measured tokens/sec/watt and joules/token, then penalized failures and high p95 latency.")
        else:
            balanced_score = _weighted_sum(
                (
                    (throughput_score, weights["throughput"]),
                    (latency_score, weights["latency"]),
                    (reliability_score, weights["reliability"]),
                    (power_score, weights.get("power", 0.0)),
                )
            )
            final_score = balanced_score
            if has_any_power:
                reasons.append("Balanced scoring combined measured throughput, p95 latency, power efficiency, and reliability.")
            else:
                reasons.append("Balanced scoring combined measured throughput, p95 latency, and reliability.")

        scores.append(
            RecommendationScore(
                candidate_id=item.candidate_id,
                goal=goal.value,
                throughput_score=_round_or_none(throughput_score),
                latency_score=_round_or_none(latency_score),
                efficiency_score=_round_or_none(efficiency_score),
                reliability_score=_round_or_none(reliability_score),
                prediction_accuracy_score=_round_or_none(prediction_accuracy_score),
                balanced_score=_round_or_none(balanced_score),
                final_score=_round_or_none(final_score),
                power_score=_round_or_none(power_score),
                weights_used=weights,
                missing_metric_penalties=missing_metric_penalties,
                score_breakdown=_score_breakdown(
                    throughput_score,
                    latency_score,
                    efficiency_score,
                    power_score,
                    reliability_score,
                    prediction_accuracy_score,
                ),
                reasons=reasons,
                disqualifiers=disqualifiers,
            )
        )

    pareto_frontier = _pareto_frontier(inputs, scores, has_any_power=has_any_power)
    pareto_ids = {str(row["candidate_id"]) for row in pareto_frontier}
    scores = [
        _with_pareto_flag(score, score.candidate_id in pareto_ids)
        for score in scores
    ]
    ranked_scores = sorted(
        scores,
        key=lambda item: (
            item.final_score is None,
            -(item.final_score or -1.0),
            not item.pareto_optimal,
            -(item.reliability_score or -1.0),
            -(item.throughput_score or -1.0),
            -(item.latency_score or -1.0),
            item.candidate_id,
        ),
    )
    selected_score = next((item for item in ranked_scores if item.final_score is not None), None)
    selected_input = next((item for item in inputs if selected_score and item.candidate_id == selected_score.candidate_id), None)
    valid_candidate_count = sum(1 for item in ranked_scores if item.final_score is not None)
    was_comparative = valid_candidate_count > 1
    selection_reasons = _selection_reasons(
        selected_input=selected_input,
        inputs=inputs,
        ranked_scores=ranked_scores,
        goal=goal,
        has_any_power=has_any_power,
        was_comparative=was_comparative,
    )
    rationale = list(selection_reasons)
    if selected_input is None or selected_score is None:
        rationale.append("No eligible candidate produced a usable measured recommendation.")
    candidate_table = _candidate_table(inputs, ranked_scores, pareto_ids=pareto_ids)
    evaluated_set_fidelity = compute_evaluated_set_fidelity(
        selected_candidate_id=selected_score.candidate_id if selected_score else None,
        ranked_scores=ranked_scores,
        candidate_table=candidate_table,
        pareto_ids=pareto_ids,
    )
    optimizer_quality = compute_optimizer_quality(
        goal=goal.value,
        selected_candidate_id=selected_score.candidate_id if selected_score else None,
        ranked_scores=ranked_scores,
        candidate_table=candidate_table,
    )
    result = RecommendationResult(
        recommended_candidate_id=selected_score.candidate_id if selected_score else None,
        goal=goal.value,
        selected_score=selected_score,
        selected_config=selected_input.candidate if selected_input else None,
        selected_serve_command=selected_input.serve_plan.shell_command if selected_input and selected_input.serve_plan else None,
        selected_benchmark_plan=selected_input.benchmark_plan if selected_input else None,
        status="success" if selected_score is not None else "warning",
        predicted_metrics=selected_input.predicted_metrics if selected_input else {},
        measured_metrics=selected_input.measured_metrics if selected_input else {},
        telemetry_metrics=selected_input.telemetry_metrics if selected_input else {},
        comparison_metrics=selected_input.comparison_metrics if selected_input else {},
        candidate_count=len(inputs),
        valid_candidate_count=valid_candidate_count,
        was_comparative=was_comparative,
        score_weights=weights,
        score_breakdown=selected_score.score_breakdown if selected_score else {},
        ranked_candidates=candidate_table,
        pareto_frontier=pareto_frontier,
        alternative_recommendations=_alternative_recommendations(inputs, ranked_scores, has_any_power=has_any_power),
        telemetry_used_in_scoring=has_any_power and any(key in weights for key in ("power", "efficiency")),
        power_aware=has_any_power,
        power_missing_reason=None if has_any_power else "No candidate had usable power telemetry.",
        selection_reasons=selection_reasons,
        metadata_notes=notes,
        warnings=warnings + (selected_input.warnings if selected_input else []),
        candidate_table=candidate_table,
        alternatives=ranked_scores,
        rationale=rationale,
        evaluated_set_fidelity=evaluated_set_fidelity,
        optimizer_quality=optimizer_quality,
    )
    confidence_level, confidence_reasons = _recommendation_confidence(
        recommendation=result,
        inputs=inputs,
        scores=ranked_scores,
    )
    result = replace(result, confidence_level=confidence_level, confidence_reasons=confidence_reasons)
    return ranked_scores, result


def compute_evaluated_set_fidelity(
    *,
    selected_candidate_id: str | None,
    ranked_scores: list[RecommendationScore],
    candidate_table: list[dict[str, float | int | str | None]],
    pareto_ids: set[str] | None = None,
) -> dict[str, object]:
    pareto_ids = set(pareto_ids or set())
    valid_scores = [score for score in ranked_scores if score.final_score is not None]
    score_by_id = {score.candidate_id: score for score in valid_scores}
    selected_score = score_by_id.get(selected_candidate_id or "")
    best_score = valid_scores[0] if valid_scores else None
    second_score = valid_scores[1] if len(valid_scores) > 1 else None
    selected_rank = None
    if selected_candidate_id is not None:
        for index, score in enumerate(valid_scores, start=1):
            if score.candidate_id == selected_candidate_id:
                selected_rank = index
                break
    selected_value = selected_score.final_score if selected_score else None
    best_value = best_score.final_score if best_score else None
    second_value = second_score.final_score if second_score else None
    return {
        "scope": "evaluated_candidates_only",
        "selected_candidate_id": selected_candidate_id,
        "selected_rank": selected_rank,
        "selected_score": selected_value,
        "best_candidate_id": best_score.candidate_id if best_score else None,
        "best_score": best_value,
        "selected_score_over_best_score": _safe_ratio(selected_value, best_value),
        "gap_to_best_score": _score_gap(best_value, selected_value),
        "gap_to_second_best_score": _score_gap(selected_value, second_value),
        "selected_is_best_evaluated": bool(selected_candidate_id and best_score and selected_candidate_id == best_score.candidate_id),
        "selected_is_pareto_optimal": bool(selected_candidate_id and selected_candidate_id in pareto_ids),
        "valid_candidate_count": len(valid_scores),
        "pareto_candidate_count": len(pareto_ids),
        "metric_winners": _metric_winners(candidate_table),
        "notes": [
            "Evaluated-set fidelity compares only candidates that were evaluated or satisfied by exact fresh measured evidence.",
            "This is an evaluated-set comparison, not an exhaustive search claim.",
        ],
    }


def compute_optimizer_quality(
    *,
    goal: str,
    selected_candidate_id: str | None,
    ranked_scores: list[RecommendationScore],
    candidate_table: list[dict[str, float | int | str | None]],
) -> dict[str, object]:
    valid_scores = [score for score in ranked_scores if score.final_score is not None]
    selected_score = next((score for score in valid_scores if score.candidate_id == selected_candidate_id), None)
    best_score = valid_scores[0] if valid_scores else None
    selected_row = _candidate_row(candidate_table, selected_candidate_id)
    metric_winners = _metric_winners(candidate_table)
    source_counts: dict[str, int] = {}
    eligible_count = 0
    rejected_count = 0
    for row in candidate_table:
        source = str(row.get("candidate_source") or row.get("source") or "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1
        if row.get("status") == "eligible":
            eligible_count += 1
        else:
            rejected_count += 1
    selected_value = selected_score.final_score if selected_score else None
    best_value = best_score.final_score if best_score else None
    return {
        "schema_version": "optimizer_quality/v1",
        "scope": "evaluated_candidates_only",
        "baseline_type": "bounded_evaluated_candidate_baseline",
        "goal": goal,
        "selected_candidate_id": selected_candidate_id,
        "best_score_candidate_id": best_score.candidate_id if best_score else None,
        "search_regret": {
            "score_gap_to_best": _score_gap(best_value, selected_value),
            "relative_score_regret": _relative_regret(best_value, selected_value, maximize=True),
        },
        "bounded_baselines": {
            "score": {
                "candidate_id": best_score.candidate_id,
                "metric": "score",
                "value": best_value,
            } if best_score else None,
            **metric_winners,
        },
        "metric_regret_percent": _metric_regret_percent(selected_row, metric_winners),
        "policy_coverage": {
            "candidate_count": len(candidate_table),
            "valid_candidate_count": len(valid_scores),
            "eligible_candidate_count": eligible_count,
            "rejected_candidate_count": rejected_count,
            "candidate_source_counts": source_counts,
            "safe_baseline_present": source_counts.get("safe_baseline", 0) > 0,
            "synthesized_candidate_count": source_counts.get("aiconfigurator_synthesis", 0),
        },
        "notes": [
            "Optimizer quality is bounded to evaluated candidates and exact fresh measured evidence hits.",
            "Regret values compare the selected candidate with bounded evaluated baselines, not all possible configurations.",
        ],
    }


def audit_recommendation_quality(recommendation: RecommendationResult) -> dict[str, object]:
    fidelity = dict(recommendation.evaluated_set_fidelity or {})
    optimizer_quality = dict(recommendation.optimizer_quality or {})
    selected_score = recommendation.selected_score.final_score if recommendation.selected_score else None
    return {
        "schema_version": "recommendation-quality-audit/v1",
        "scope": "evaluated_candidates_only",
        "selected_candidate_id": recommendation.recommended_candidate_id,
        "selected_rank": fidelity.get("selected_rank"),
        "selected_score": selected_score,
        "best_candidate_id": fidelity.get("best_candidate_id"),
        "best_score": fidelity.get("best_score"),
        "gap_to_best_score": fidelity.get("gap_to_best_score"),
        "selected_is_best_evaluated": fidelity.get("selected_is_best_evaluated"),
        "selected_is_pareto_optimal": fidelity.get("selected_is_pareto_optimal"),
        "candidate_count": recommendation.candidate_count,
        "valid_candidate_count": recommendation.valid_candidate_count,
        "pareto_candidate_count": len(recommendation.pareto_frontier),
        "fidelity_present": bool(fidelity),
        "optimizer_quality_present": bool(optimizer_quality),
        "search_regret": optimizer_quality.get("search_regret"),
        "bounded_baselines": optimizer_quality.get("bounded_baselines"),
        "wording_policy": "evaluated_set_only",
        "stability_inputs": [
            "selected_canonical_config",
            "selected_command",
            "candidate_table",
            "pareto_frontier",
        ],
        "notes": [
            "Quality audit is scoped to evaluated candidates and exact fresh measured evidence hits.",
            "It does not claim exhaustive search coverage.",
        ],
    }


def _metric_winners(candidate_table: list[dict[str, float | int | str | None]]) -> dict[str, dict[str, object] | None]:
    return {
        "throughput": _metric_winner(candidate_table, "total_tokens_s", maximize=True),
        "lowest_latency": _metric_winner(candidate_table, "p95_latency_s", maximize=False),
        "lowest_energy_per_token": _metric_winner(candidate_table, "joules_per_token", maximize=False),
        "best_tokens_per_watt": _metric_winner(candidate_table, "tokens_per_second_per_watt", maximize=True),
        "lowest_power": _metric_winner(candidate_table, "average_power_watts", maximize=False),
    }


def _metric_winner(
    candidate_table: list[dict[str, float | int | str | None]],
    metric: str,
    *,
    maximize: bool,
) -> dict[str, object] | None:
    rows = [
        (str(row.get("candidate_id")), _optional_float(row.get(metric)))
        for row in candidate_table
        if row.get("candidate_id") is not None and _optional_float(row.get(metric)) is not None
    ]
    if not rows:
        return None
    candidate_id, value = max(rows, key=lambda item: item[1] or 0.0) if maximize else min(rows, key=lambda item: item[1] or 0.0)
    return {"candidate_id": candidate_id, "metric": metric, "value": value}


def _candidate_row(
    candidate_table: list[dict[str, float | int | str | None]],
    candidate_id: str | None,
) -> dict[str, float | int | str | None] | None:
    if candidate_id is None:
        return None
    return next((row for row in candidate_table if row.get("candidate_id") == candidate_id), None)


def _metric_regret_percent(
    selected_row: dict[str, float | int | str | None] | None,
    metric_winners: dict[str, dict[str, object] | None],
) -> dict[str, float | None]:
    if selected_row is None:
        return {
            "throughput": None,
            "p95_latency": None,
            "tokens_per_watt": None,
            "joules_per_token": None,
        }
    return {
        "throughput": _metric_percent_regret(
            selected_row,
            metric_winners.get("throughput"),
            "total_tokens_s",
            maximize=True,
        ),
        "p95_latency": _metric_percent_regret(
            selected_row,
            metric_winners.get("lowest_latency"),
            "p95_latency_s",
            maximize=False,
        ),
        "tokens_per_watt": _metric_percent_regret(
            selected_row,
            metric_winners.get("best_tokens_per_watt"),
            "tokens_per_second_per_watt",
            maximize=True,
        ),
        "joules_per_token": _metric_percent_regret(
            selected_row,
            metric_winners.get("lowest_energy_per_token"),
            "joules_per_token",
            maximize=False,
        ),
    }


def _metric_percent_regret(
    selected_row: dict[str, float | int | str | None],
    winner: dict[str, object] | None,
    metric: str,
    *,
    maximize: bool,
) -> float | None:
    if winner is None:
        return None
    selected_value = _optional_float(selected_row.get(metric))
    winner_value = _optional_float(winner.get("value"))
    regret = _relative_regret(winner_value, selected_value, maximize=maximize)
    return round(regret * 100.0, 6) if regret is not None else None


def _relative_regret(best_value: float | None, selected_value: float | None, *, maximize: bool) -> float | None:
    if best_value is None or selected_value is None or best_value == 0.0:
        return None
    if maximize:
        regret = (best_value - selected_value) / abs(best_value)
    else:
        regret = (selected_value - best_value) / abs(best_value)
    return max(0.0, regret)


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in {None, 0.0}:
        return None
    return numerator / denominator


def _score_gap(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _write_attach_mode_plan_bundle(
    *,
    run_dir: Path,
    candidates: list[ServeCandidate],
    base_url: str,
    backend: str,
) -> tuple[Path, list[CandidateEvaluationPlan]]:
    plan_dir = run_dir / "plan"
    plans: list[CandidateEvaluationPlan] = []
    serve_plans: list[VllmServePlan] = []
    benchmark_plans: list[EndpointBenchmarkPlan] = []
    for candidate in candidates:
        notes: list[str] = []
        serve_plan = None
        if (candidate.backend or backend).lower() == "vllm":
            serve_plan = candidate_to_vllm_serve_plan(candidate)
            serve_plans.append(serve_plan)
        else:
            notes.append("No backend-specific serve plan adapter is implemented for this candidate backend.")
        benchmark_plan = candidate_to_endpoint_benchmark_plan(candidate, base_url=base_url)
        benchmark_plans.append(benchmark_plan)
        plans.append(
            CandidateEvaluationPlan(
                candidate_id=candidate.candidate_id,
                rank=candidate.rank,
                candidate=candidate,
                serve_plan=serve_plan,
                benchmark_plan=benchmark_plan,
                notes=notes,
            )
        )

    write_jsonl(plan_dir / "candidates.jsonl", candidates)
    write_jsonl(plan_dir / "serve_plans.jsonl", serve_plans)
    write_jsonl(plan_dir / "benchmark_plans.jsonl", benchmark_plans)
    write_jsonl(plan_dir / "evaluation_plans.jsonl", plans)
    write_json(
        plan_dir / "summary.json",
        {
            "candidate_count": len(candidates),
            "serve_plan_count": len(serve_plans),
            "artifact_files": [
                "candidates.jsonl",
                "serve_plans.jsonl",
                "benchmark_plans.jsonl",
                "evaluation_plans.jsonl",
                "summary.json",
            ],
        },
    )
    return plan_dir, plans


def _write_pareto_csv(path: Path, rows: list[dict[str, float | int | str | None]]) -> None:
    fieldnames = [
        "candidate_id",
        "source",
        "concurrency",
        "total_tokens_s",
        "p95_latency_s",
        "failed_requests",
        "average_power_watts",
        "joules_per_token",
        "tokens_per_second_per_watt",
        "score",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _generate_aiconfigurator_candidates(
    *,
    model: str,
    backend: str,
    system: str,
    total_gpus: int,
    isl: int,
    osl: int,
    ttft: float | None,
    tpot: float | None,
    top_k: int,
    run_dir: Path,
    aiconfigurator_runner: AIConfiguratorRunner,
) -> tuple[list[ServeCandidate], list[str]]:
    aic_dir = run_dir / "aiconfigurator"
    aic_dir.mkdir(parents=True, exist_ok=True)
    run = aiconfigurator_runner(
        mode="default",
        model=model,
        system=system,
        backend=backend,
        output_dir=aic_dir,
        isl=isl,
        osl=osl,
        ttft=ttft,
        tpot=tpot,
        total_gpus=total_gpus,
    )
    warnings: list[str] = []
    if run.returncode != 0:
        detail = run.stderr.strip() or run.stdout.strip() or f"returncode={run.returncode}"
        raise RuntimeError(f"AIConfigurator default run failed: {detail}")
    best_config_csv = _find_best_config_csv(aic_dir)
    if best_config_csv is None:
        raise RuntimeError("AIConfigurator did not produce best_config_topn.csv under the save directory.")
    warnings.append(f"AIConfigurator candidates were loaded from {best_config_csv}.")
    if best_config_csv.parent.name != "agg":
        warnings.append("AIConfigurator best_config_topn.csv did not come from an agg directory; results may be harder to compare in Attach Mode.")
    return parse_aiconfigurator_best_configs(str(best_config_csv), top_k=top_k), warnings


def _check_endpoint_health(
    *,
    base_url: str,
    model: str,
    timeout_s: float,
    request_fn: RequestFn,
) -> None:
    config = EndpointBenchmarkConfig(
        run_id="health-check",
        base_url=base_url,
        model=model,
        concurrency=1,
        num_requests=1,
        max_tokens=1,
        prompt="health check",
        timeout_s=timeout_s,
    )
    record = request_fn(config, 0)
    if record.status != "ok":
        detail = record.error or record.status
        raise RuntimeError(f"Endpoint health check failed: {detail}")


def _find_best_config_csv(aic_dir: Path) -> Path | None:
    paths = sorted(aic_dir.rglob("best_config_topn.csv"))
    if not paths:
        return None
    agg_paths = [path for path in paths if path.parent.name == "agg"]
    return agg_paths[0] if agg_paths else paths[0]


def _predicted_metrics(candidate: ServeCandidate) -> dict[str, float | int | str | None]:
    return {
        "predicted_tokens_s": candidate.predicted_tokens_s,
        "predicted_request_rate": candidate.request_rate,
        "predicted_request_latency_ms": candidate.predicted_request_latency_ms,
        "predicted_power_w": candidate.predicted_power_w,
        "predicted_memory_gb": candidate.predicted_memory_gb,
    }


def _measured_metrics(summary: EndpointBenchmarkSummary) -> dict[str, float | int | str | None]:
    return {
        "total_requests": summary.total_requests,
        "successful_requests": summary.successful_requests,
        "failed_requests": summary.failed_requests,
        "wall_time_s": summary.wall_time_s,
        "request_rate_req_s": summary.request_rate_req_s,
        "prompt_tokens": summary.prompt_tokens,
        "completion_tokens": summary.completion_tokens,
        "total_tokens": summary.total_tokens,
        "output_tokens_s": summary.output_tokens_s,
        "total_tokens_s": summary.total_tokens_s,
        "avg_latency_s": summary.avg_latency_s,
        "p50_latency_s": summary.p50_latency_s,
        "p95_latency_s": summary.p95_latency_s,
        "p99_latency_s": summary.p99_latency_s,
    }


def _telemetry_metrics(summary: EndpointBenchmarkSummary) -> dict[str, Any]:
    telemetry_summary = summary.telemetry_summary
    return {
        "average_power_watts": summary.average_power_watts,
        "min_power_watts": summary.min_power_watts,
        "max_power_watts": summary.max_power_watts,
        "peak_power_watts": summary.peak_power_watts,
        "power_stddev_watts": summary.power_stddev_watts,
        "power_sampling_duration_s": summary.power_sampling_duration_s,
        "power_sampling_rate_hz": summary.power_sampling_rate_hz,
        "energy_joules": summary.energy_joules,
        "joules_per_token": summary.joules_per_token,
        "tokens_per_second_per_watt": summary.tokens_per_second_per_watt,
        "power_sample_count": summary.power_sample_count,
        "observed_memory_mb": summary.observed_memory_mb,
        "average_gpu_util_percent": summary.average_gpu_util_percent,
        "max_gpu_util_percent": summary.max_gpu_util_percent,
        "average_memory_util_percent": summary.average_memory_util_percent,
        "max_memory_util_percent": summary.max_memory_util_percent,
        "average_temperature_c": summary.average_temperature_c,
        "max_temperature_c": summary.max_temperature_c,
        "average_sm_clock_mhz": summary.average_sm_clock_mhz,
        "average_memory_clock_mhz": summary.average_memory_clock_mhz,
        "power_limit_watts": telemetry_summary.get("power_limit_watts"),
        "enforced_power_limit_watts": telemetry_summary.get("enforced_power_limit_watts"),
        "telemetry_provider": summary.telemetry_provider,
        "telemetry_available": summary.telemetry_available,
        "telemetry_quality": summary.telemetry_quality,
        "missing_fields": telemetry_summary.get("missing_fields", []),
        "telemetry_warnings": telemetry_summary.get("warnings") or summary.warnings,
        "telemetry_notes": telemetry_summary.get("notes") or summary.telemetry_notes,
        "provider_info": telemetry_summary.get("provider_info", {}),
        "power_stats": telemetry_summary.get("power_stats", {}),
        "utilization_stats": telemetry_summary.get("utilization_stats", {}),
        "thermal_stats": telemetry_summary.get("thermal_stats", {}),
        "clock_stats": telemetry_summary.get("clock_stats", {}),
        "telemetry_capabilities": telemetry_summary.get("telemetry_capabilities", {}),
        "telemetry_summary": telemetry_summary,
    }


def _disqualifiers(item: RecommendationInput, goal: RecommendationGoal, has_any_power: bool) -> list[str]:
    del has_any_power
    disqualifiers: list[str] = []
    successful = _optional_int(item.measured_metrics.get("successful_requests")) or 0
    total = _optional_int(item.measured_metrics.get("total_requests")) or 0
    if total <= 0 or successful <= 0:
        disqualifiers.append("no_successful_requests")
    if goal == RecommendationGoal.LATENCY and _latency_value(item) is None:
        disqualifiers.append("missing_latency_metric")
    if goal == RecommendationGoal.EFFICIENCY and not _positive_number(item.telemetry_metrics.get("tokens_per_second_per_watt")):
        disqualifiers.append("missing_power_telemetry")
    disqualifiers.extend(slo_disqualifiers(item))
    return disqualifiers


def _reliability_score(item: RecommendationInput) -> float | None:
    total = _optional_int(item.measured_metrics.get("total_requests"))
    successful = _optional_int(item.measured_metrics.get("successful_requests"))
    if total in {None, 0} or successful is None:
        return None
    return successful / total


def _prediction_accuracy_score(item: RecommendationInput) -> float | None:
    metrics = []
    token_ratio = _ratio(
        _optional_float(item.measured_metrics.get("total_tokens_s")),
        _optional_float(item.predicted_metrics.get("predicted_tokens_s")),
    )
    request_ratio = _ratio(
        _optional_float(item.measured_metrics.get("request_rate_req_s")),
        _optional_float(item.predicted_metrics.get("predicted_request_rate")),
    )
    latency_ratio = _ratio(
        _optional_float(item.comparison_metrics.get("measured_avg_latency_ms")),
        _optional_float(item.predicted_metrics.get("predicted_request_latency_ms")),
    )
    for ratio in (token_ratio, request_ratio, latency_ratio):
        if ratio is None or ratio <= 0:
            continue
        metrics.append(1.0 / (1.0 + abs(1.0 - ratio)))
    if not metrics:
        return None
    return sum(metrics) / len(metrics)


def _normalize_higher(
    inputs: list[RecommendationInput],
    value_fn: Callable[[RecommendationInput], float | None],
) -> dict[str, float | None]:
    values = {item.candidate_id: value_fn(item) for item in inputs}
    valid = [value for value in values.values() if value is not None]
    if not valid:
        return {candidate_id: None for candidate_id in values}
    low = min(valid)
    high = max(valid)
    if high == low:
        return {candidate_id: 1.0 if value is not None else None for candidate_id, value in values.items()}
    if low >= 0 and high > 0:
        return {
            candidate_id: (value / high) if value is not None else None
            for candidate_id, value in values.items()
        }
    return {
        candidate_id: ((value - low) / (high - low)) if value is not None else None
        for candidate_id, value in values.items()
    }


def _normalize_lower(
    inputs: list[RecommendationInput],
    value_fn: Callable[[RecommendationInput], float | None],
) -> dict[str, float | None]:
    values = {item.candidate_id: value_fn(item) for item in inputs}
    valid = [value for value in values.values() if value is not None]
    if not valid:
        return {candidate_id: None for candidate_id in values}
    low = min(valid)
    high = max(valid)
    if high == low:
        return {candidate_id: 1.0 if value is not None else None for candidate_id, value in values.items()}
    if low >= 0:
        return {
            candidate_id: ((low / value) if value not in {None, 0} else None)
            for candidate_id, value in values.items()
        }
    return {
        candidate_id: ((high - value) / (high - low)) if value is not None else None
        for candidate_id, value in values.items()
    }


def _latency_value(item: RecommendationInput) -> float | None:
    return _optional_float(item.measured_metrics.get("p95_latency_s")) or _optional_float(item.measured_metrics.get("avg_latency_s"))


def _weighted_sum(parts: tuple[tuple[float | None, float], ...]) -> float | None:
    total = 0.0
    weight_sum = 0.0
    for value, weight in parts:
        if value is None or weight == 0:
            continue
        total += value * weight
        weight_sum += weight
    if weight_sum == 0:
        return None
    return total / weight_sum


def _weights_for_goal(goal: RecommendationGoal, *, has_any_power: bool) -> dict[str, float]:
    if goal == RecommendationGoal.THROUGHPUT:
        return THROUGHPUT_WITH_POWER_WEIGHTS if has_any_power else THROUGHPUT_WEIGHTS
    if goal == RecommendationGoal.LATENCY:
        return LATENCY_WITH_POWER_WEIGHTS if has_any_power else LATENCY_WEIGHTS
    if goal == RecommendationGoal.EFFICIENCY:
        return EFFICIENCY_WEIGHTS
    return BALANCED_WITH_POWER_WEIGHTS if has_any_power else BALANCED_NO_POWER_WEIGHTS


def _power_score(efficiency_score: float | None, joules_score: float | None) -> float | None:
    return _weighted_sum(
        (
            (efficiency_score, 0.60),
            (joules_score, 0.40),
        )
    )


def _missing_metric_penalties(item: RecommendationInput, goal: RecommendationGoal, has_any_power: bool) -> list[str]:
    penalties: list[str] = []
    if _optional_float(item.measured_metrics.get("total_tokens_s")) is None:
        penalties.append("missing_total_tokens_s")
    if _latency_value(item) is None:
        penalties.append("missing_latency")
    if _reliability_score(item) is None:
        penalties.append("missing_reliability")
    if has_any_power and not _positive_number(item.telemetry_metrics.get("tokens_per_second_per_watt")):
        penalties.append("missing_power_telemetry")
    if goal == RecommendationGoal.EFFICIENCY and not _positive_number(item.telemetry_metrics.get("tokens_per_second_per_watt")):
        penalties.append("missing_efficiency_metric")
    return penalties


def _score_breakdown(
    throughput_score: float | None,
    latency_score: float | None,
    efficiency_score: float | None,
    power_score: float | None,
    reliability_score: float | None,
    prediction_accuracy_score: float | None,
) -> dict[str, float | int | str | None]:
    return {
        "throughput_score": _round_or_none(throughput_score),
        "latency_score": _round_or_none(latency_score),
        "efficiency_score": _round_or_none(efficiency_score),
        "power_score": _round_or_none(power_score),
        "reliability_score": _round_or_none(reliability_score),
        "prediction_accuracy_score": _round_or_none(prediction_accuracy_score),
    }


def _with_pareto_flag(score: RecommendationScore, pareto_optimal: bool) -> RecommendationScore:
    return RecommendationScore(
        candidate_id=score.candidate_id,
        goal=score.goal,
        throughput_score=score.throughput_score,
        latency_score=score.latency_score,
        efficiency_score=score.efficiency_score,
        reliability_score=score.reliability_score,
        prediction_accuracy_score=score.prediction_accuracy_score,
        balanced_score=score.balanced_score,
        final_score=score.final_score,
        power_score=score.power_score,
        weights_used=score.weights_used,
        missing_metric_penalties=score.missing_metric_penalties,
        score_breakdown=score.score_breakdown,
        pareto_optimal=pareto_optimal,
        reasons=score.reasons,
        disqualifiers=score.disqualifiers,
    )


def _make_unavailable_score(item: RecommendationInput, goal: RecommendationGoal, disqualifier: str) -> RecommendationScore:
    return RecommendationScore(
        candidate_id=item.candidate_id,
        goal=goal.value,
        throughput_score=None,
        latency_score=None,
        efficiency_score=None,
        reliability_score=_round_or_none(_reliability_score(item)),
        prediction_accuracy_score=_round_or_none(_prediction_accuracy_score(item)),
        balanced_score=None,
        final_score=None,
        power_score=None,
        weights_used=_weights_for_goal(goal, has_any_power=False),
        missing_metric_penalties=[disqualifier],
        score_breakdown=_score_breakdown(None, None, None, None, _reliability_score(item), _prediction_accuracy_score(item)),
        reasons=list(item.warnings) + ["Candidate could not be scored for the requested goal."],
        disqualifiers=[disqualifier],
    )


def _selection_reasons(
    *,
    selected_input: RecommendationInput | None,
    inputs: list[RecommendationInput],
    ranked_scores: list[RecommendationScore],
    goal: RecommendationGoal,
    has_any_power: bool,
    was_comparative: bool,
) -> list[str]:
    selected_score = next((score for score in ranked_scores if selected_input and score.candidate_id == selected_input.candidate_id), None)
    if selected_input is None or selected_score is None:
        return ["No eligible candidate produced a usable measured recommendation."]

    reasons: list[str] = []
    valid_inputs = [
        item for item in inputs if next((score for score in ranked_scores if score.candidate_id == item.candidate_id and score.final_score is not None), None)
    ]
    if was_comparative:
        reasons.append(
            f"Selected {selected_input.candidate_id} because it had the highest {goal.value} score among {len(valid_inputs)} valid candidates."
        )
    else:
        reasons.append("Only one valid candidate was evaluated, so this recommendation is a validation result rather than a comparative search.")
    if has_any_power and goal in {RecommendationGoal.BALANCED, RecommendationGoal.EFFICIENCY}:
        reasons.append("Power telemetry was included in the scoring policy for this goal.")

    throughput = _optional_float(selected_input.measured_metrics.get("total_tokens_s"))
    if throughput is not None:
        peak_throughput = max((_optional_float(item.measured_metrics.get("total_tokens_s")) or 0.0) for item in valid_inputs)
        if was_comparative and throughput >= peak_throughput:
            reasons.append(f"It achieved the highest measured throughput at {_format_rate(throughput)} tokens/s.")
        else:
            reasons.append(f"It achieved {_format_rate(throughput)} measured tokens/s.")
        if was_comparative and peak_throughput > 0 and throughput < peak_throughput:
            reasons.append(f"It retained {_format_percent(throughput / peak_throughput)} of peak measured throughput.")

    failed = _optional_int(selected_input.measured_metrics.get("failed_requests"))
    if failed == 0:
        reasons.append("It completed with 0 failed requests.")
    elif failed is not None:
        reasons.append(f"It completed with {failed} failed requests, which reduced its reliability score.")

    selected_p95 = _optional_float(selected_input.measured_metrics.get("p95_latency_s"))
    if selected_p95 is not None and was_comparative:
        valid_latencies = [_optional_float(item.measured_metrics.get("p95_latency_s")) for item in valid_inputs]
        valid_latencies = [value for value in valid_latencies if value is not None]
        if valid_latencies and selected_p95 <= min(valid_latencies):
            reasons.append(f"It had the lowest measured p95 latency at {_format_seconds(selected_p95)}.")
        high_concurrency = _highest_concurrency_input(valid_inputs, exclude_candidate_id=selected_input.candidate_id)
        high_p95 = _optional_float(high_concurrency.measured_metrics.get("p95_latency_s")) if high_concurrency else None
        high_throughput = _optional_float(high_concurrency.measured_metrics.get("total_tokens_s")) if high_concurrency else None
        if high_p95 is not None and high_p95 > selected_p95 and high_throughput not in {None, 0} and throughput is not None:
            improvement = (high_p95 - selected_p95) / high_p95
            retained = throughput / high_throughput
            reasons.append(
                f"It improved p95 latency by {_format_percent(improvement)} compared with {high_concurrency.candidate_id} "
                f"while retaining {_format_percent(retained)} of that candidate's throughput."
            )

    efficiency = _optional_float(selected_input.telemetry_metrics.get("tokens_per_second_per_watt"))
    if efficiency is not None:
        valid_efficiency = [
            _optional_float(item.telemetry_metrics.get("tokens_per_second_per_watt"))
            for item in valid_inputs
        ]
        valid_efficiency = [value for value in valid_efficiency if value is not None]
        if has_any_power and valid_efficiency and efficiency >= max(valid_efficiency):
            reasons.append(f"It had the best measured efficiency at {_format_rate(efficiency)} tokens/s/W.")
        elif has_any_power:
            reasons.append(f"Power telemetry was available and measured {_format_rate(efficiency)} tokens/s/W.")

    predicted_ratio = _optional_float(selected_input.comparison_metrics.get("measured_over_predicted_tokens_ratio"))
    if predicted_ratio is not None:
        reasons.append(f"Its measured throughput reached {_format_percent(predicted_ratio)} of the predicted tokens/sec.")

    return reasons


def _candidate_table(
    inputs: list[RecommendationInput],
    scores: list[RecommendationScore],
    pareto_ids: set[str] | None = None,
) -> list[dict[str, float | int | str | None]]:
    pareto_ids = pareto_ids or set()
    score_by_id = {score.candidate_id: score for score in scores}
    rows = []
    for item in inputs:
        score = score_by_id.get(item.candidate_id)
        row: dict[str, float | int | str | None] = {
            "candidate_id": item.candidate_id,
            "source": item.candidate_source,
            "concurrency": item.benchmark_plan.concurrency if item.benchmark_plan else None,
            "total_tokens_s": _optional_float(item.measured_metrics.get("total_tokens_s")),
            "p95_latency_s": _optional_float(item.measured_metrics.get("p95_latency_s")),
            "average_power_watts": _optional_float(item.telemetry_metrics.get("average_power_watts")),
            "power_stddev_watts": _optional_float(item.telemetry_metrics.get("power_stddev_watts")),
            "power_sampling_rate_hz": _optional_float(item.telemetry_metrics.get("power_sampling_rate_hz")),
            "joules_per_token": _optional_float(item.telemetry_metrics.get("joules_per_token")),
            "tokens_per_second_per_watt": _optional_float(item.telemetry_metrics.get("tokens_per_second_per_watt")),
            "telemetry_quality": str(item.telemetry_metrics.get("telemetry_quality") or "unavailable"),
            "failed_requests": _optional_int(item.measured_metrics.get("failed_requests")),
            "throughput_score": score.throughput_score if score else None,
            "latency_score": score.latency_score if score else None,
            "efficiency_score": score.efficiency_score if score else None,
            "power_score": score.power_score if score else None,
            "reliability_score": score.reliability_score if score else None,
            "score": score.final_score if score else None,
            "pareto_optimal": item.candidate_id in pareto_ids,
            "status": ",".join(score.disqualifiers) if score and score.disqualifiers else "eligible",
        }
        row.update(_candidate_engine_fields(item))
        rows.append(row)
    return sorted(rows, key=lambda row: ((row["score"] is None), -(float(row["score"] or -1.0)), str(row["candidate_id"])))


def _candidate_engine_fields(item: RecommendationInput) -> dict[str, float | int | str | None]:
    raw = item.candidate.raw or {}
    values: dict[str, float | int | str | None] = {
        "candidate_source": _optional_str(raw.get("managed_candidate_source")),
        "dtype": _optional_str(raw.get("dtype")),
        "quantization": _optional_str(raw.get("quantization")),
        "max_model_len": _optional_int(raw.get("max_context_tokens")),
        "gpu_memory_utilization": _optional_float(raw.get("gpu_memory_utilization")),
        "max_num_seqs": _optional_int(item.candidate.batch_size),
        "tensor_parallel_size": _optional_int(item.candidate.tp),
        "benchmark_concurrency": _optional_int(item.candidate.concurrency),
        "block_size": _optional_int(raw.get("block_size")),
        "kv_cache_dtype": _optional_str(raw.get("kv_cache_dtype")),
        "enforce_eager": _optional_bool(raw.get("enforce_eager")),
        "max_num_batched_tokens": _optional_int(raw.get("max_num_batched_tokens")),
        "enable_chunked_prefill": _optional_bool(raw.get("enable_chunked_prefill")),
        "max_cudagraph_capture_size": _optional_int(raw.get("max_cudagraph_capture_size")),
        "enable_prefix_caching": _optional_bool(raw.get("enable_prefix_caching")),
        "synthesis_rationale": _optional_str(raw.get("synthesis_rationale")),
        "synthesis_confidence": _optional_float(raw.get("synthesis_confidence")),
        "synthesis_status": _optional_str(raw.get("synthesis_status")),
        "workload_profile": _workload_profile_name(raw.get("workload_profile")),
        "slo_constraints": _slo_constraint_names(raw),
    }
    return {key: value for key, value in values.items() if value is not None}


def _workload_profile_name(value: object) -> str | None:
    if isinstance(value, dict):
        return _optional_str(value.get("profile_name"))
    return None


def _slo_constraint_names(raw: dict[str, Any]) -> str | None:
    constraints = raw.get("slo_constraints")
    if not isinstance(constraints, dict):
        profile = raw.get("workload_profile")
        constraints = profile.get("slo_constraints") if isinstance(profile, dict) else None
    if not isinstance(constraints, dict) or not constraints:
        return None
    return ",".join(sorted(str(key) for key in constraints))


def _optional_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _pareto_frontier(
    inputs: list[RecommendationInput],
    scores: list[RecommendationScore],
    *,
    has_any_power: bool,
) -> list[dict[str, float | int | str | None]]:
    score_by_id = {score.candidate_id: score for score in scores}
    valid_inputs = [
        item for item in inputs
        if score_by_id.get(item.candidate_id) is not None and score_by_id[item.candidate_id].final_score is not None
    ]
    frontier: list[RecommendationInput] = []
    for candidate in valid_inputs:
        if not any(_dominates(other, candidate, has_any_power=has_any_power) for other in valid_inputs if other.candidate_id != candidate.candidate_id):
            frontier.append(candidate)
    return [
        {
            "candidate_id": item.candidate_id,
            "source": item.candidate_source,
            "concurrency": item.benchmark_plan.concurrency if item.benchmark_plan else None,
            "total_tokens_s": _optional_float(item.measured_metrics.get("total_tokens_s")),
            "p95_latency_s": _optional_float(item.measured_metrics.get("p95_latency_s")),
            "failed_requests": _optional_int(item.measured_metrics.get("failed_requests")),
            "average_power_watts": _optional_float(item.telemetry_metrics.get("average_power_watts")),
            "power_stddev_watts": _optional_float(item.telemetry_metrics.get("power_stddev_watts")),
            "power_sampling_rate_hz": _optional_float(item.telemetry_metrics.get("power_sampling_rate_hz")),
            "joules_per_token": _optional_float(item.telemetry_metrics.get("joules_per_token")),
            "tokens_per_second_per_watt": _optional_float(item.telemetry_metrics.get("tokens_per_second_per_watt")),
            "telemetry_quality": str(item.telemetry_metrics.get("telemetry_quality") or "unavailable"),
            "score": score_by_id[item.candidate_id].final_score,
        }
        for item in sorted(frontier, key=lambda row: (-(score_by_id[row.candidate_id].final_score or 0.0), row.candidate_id))
    ]


def _dominates(left: RecommendationInput, right: RecommendationInput, *, has_any_power: bool) -> bool:
    left_values = _pareto_values(left, has_any_power=has_any_power)
    right_values = _pareto_values(right, has_any_power=has_any_power)
    return all(left >= right for left, right in zip(left_values, right_values, strict=True)) and any(
        left > right for left, right in zip(left_values, right_values, strict=True)
    )


def _pareto_values(item: RecommendationInput, *, has_any_power: bool) -> tuple[float, ...]:
    values = [
        _optional_float(item.measured_metrics.get("total_tokens_s")) or 0.0,
        -(_latency_value(item) if _latency_value(item) is not None else float("inf")),
        -float(_optional_int(item.measured_metrics.get("failed_requests")) or 0),
    ]
    if has_any_power:
        values.extend(
            [
                _optional_float(item.telemetry_metrics.get("tokens_per_second_per_watt")) or 0.0,
                -(_optional_float(item.telemetry_metrics.get("joules_per_token")) or float("inf")),
            ]
        )
    return tuple(values)


def _alternative_recommendations(
    inputs: list[RecommendationInput],
    scores: list[RecommendationScore],
    *,
    has_any_power: bool,
) -> dict[str, dict[str, float | int | str | None]]:
    score_by_id = {score.candidate_id: score for score in scores}
    valid = [item for item in inputs if score_by_id.get(item.candidate_id) and score_by_id[item.candidate_id].final_score is not None]
    alternatives: dict[str, dict[str, float | int | str | None]] = {}
    if not valid:
        return alternatives
    alternatives["throughput"] = _objective_row(
        max(valid, key=lambda item: (_optional_float(item.measured_metrics.get("total_tokens_s")) or -1.0, item.candidate_id)),
        "highest measured total_tokens_s",
    )
    latency_valid = [item for item in valid if _latency_value(item) is not None]
    if latency_valid:
        alternatives["latency"] = _objective_row(
            min(latency_valid, key=lambda item: (_latency_value(item) or float("inf"), item.candidate_id)),
            "lowest measured p95 or average latency",
        )
    if has_any_power:
        efficiency_valid = [item for item in valid if _positive_number(item.telemetry_metrics.get("tokens_per_second_per_watt"))]
        if efficiency_valid:
            alternatives["efficiency"] = _objective_row(
                max(efficiency_valid, key=lambda item: (_optional_float(item.telemetry_metrics.get("tokens_per_second_per_watt")) or -1.0, item.candidate_id)),
                "highest measured tokens/sec/watt",
            )
        energy_valid = [item for item in valid if _positive_number(item.telemetry_metrics.get("joules_per_token"))]
        if energy_valid:
            alternatives["lowest_energy"] = _objective_row(
                min(energy_valid, key=lambda item: (_optional_float(item.telemetry_metrics.get("joules_per_token")) or float("inf"), item.candidate_id)),
                "lowest measured joules/token",
            )
    balanced = _balanced_winner_input(valid, has_any_power=has_any_power)
    if balanced is not None:
        alternatives["balanced"] = _objective_row(balanced, "highest balanced score")
    return alternatives


def _objective_row(item: RecommendationInput, reason: str) -> dict[str, float | int | str | None]:
    return {
        "candidate_id": item.candidate_id,
        "source": item.candidate_source,
        "concurrency": item.benchmark_plan.concurrency if item.benchmark_plan else None,
        "total_tokens_s": _optional_float(item.measured_metrics.get("total_tokens_s")),
        "p95_latency_s": _optional_float(item.measured_metrics.get("p95_latency_s")),
        "tokens_per_second_per_watt": _optional_float(item.telemetry_metrics.get("tokens_per_second_per_watt")),
        "joules_per_token": _optional_float(item.telemetry_metrics.get("joules_per_token")),
        "reason": reason,
    }


def _balanced_winner_input(inputs: list[RecommendationInput], *, has_any_power: bool) -> RecommendationInput | None:
    if not inputs:
        return None
    throughput_scores = _normalize_higher(inputs, lambda item: _optional_float(item.measured_metrics.get("total_tokens_s")))
    latency_scores = _normalize_lower(inputs, lambda item: _latency_value(item))
    efficiency_scores = _normalize_higher(inputs, lambda item: _optional_float(item.telemetry_metrics.get("tokens_per_second_per_watt")))
    joules_scores = _normalize_lower(inputs, lambda item: _optional_float(item.telemetry_metrics.get("joules_per_token")))
    reliability_scores = {item.candidate_id: _reliability_score(item) for item in inputs}
    weights = BALANCED_WITH_POWER_WEIGHTS if has_any_power else BALANCED_NO_POWER_WEIGHTS

    def score(item: RecommendationInput) -> tuple[float, str]:
        power_score = _power_score(efficiency_scores.get(item.candidate_id), joules_scores.get(item.candidate_id))
        value = _weighted_sum(
            (
                (throughput_scores.get(item.candidate_id), weights["throughput"]),
                (latency_scores.get(item.candidate_id), weights["latency"]),
                (reliability_scores.get(item.candidate_id), weights["reliability"]),
                (power_score, weights.get("power", 0.0)),
            )
        )
        return (value if value is not None else -1.0, item.candidate_id)

    return max(inputs, key=score)


def _highest_concurrency_input(inputs: list[RecommendationInput], exclude_candidate_id: str) -> RecommendationInput | None:
    candidates = [item for item in inputs if item.candidate_id != exclude_candidate_id and item.benchmark_plan is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.benchmark_plan.concurrency if item.benchmark_plan else 0)


def _selected_telemetry_provider(inputs: list[RecommendationInput], candidate_id: str | None) -> str | None:
    if candidate_id is None:
        return None
    for item in inputs:
        if item.candidate_id == candidate_id:
            provider = item.telemetry_metrics.get("telemetry_provider")
            return str(provider) if provider is not None else None
    return None


def _selected_telemetry_summary_path(inputs: list[RecommendationInput], candidate_id: str | None) -> str | None:
    if candidate_id is None:
        return None
    for item in inputs:
        if item.candidate_id == candidate_id:
            path = item.telemetry_metrics.get("telemetry_summary_path")
            return str(path) if path is not None else None
    return None


def _selected_telemetry_capabilities_path(inputs: list[RecommendationInput], candidate_id: str | None) -> str | None:
    if candidate_id is None:
        return None
    for item in inputs:
        if item.candidate_id == candidate_id:
            path = item.telemetry_metrics.get("telemetry_capabilities_path")
            return str(path) if path is not None else None
    return None


def _benchmark_check(candidate_count: int, total_requests: int, failed_requests: int, all_failed: bool) -> CheckRecord:
    if all_failed:
        return CheckRecord(
            name="benchmark_execution",
            status="fail",
            message="Benchmark completed but all requests failed.",
            details={"candidate_count": candidate_count, "total_requests": total_requests, "failed_requests": failed_requests},
        )
    if failed_requests > 0:
        return CheckRecord(
            name="benchmark_execution",
            status="warn",
            message="Benchmark completed with some failed requests.",
            details={"candidate_count": candidate_count, "total_requests": total_requests, "failed_requests": failed_requests},
        )
    return CheckRecord(
        name="benchmark_execution",
        status="ok",
        message="Benchmark completed.",
        details={"candidate_count": candidate_count, "total_requests": total_requests, "failed_requests": failed_requests},
    )


def _telemetry_check(telemetry: str, inputs: list[RecommendationInput]) -> CheckRecord:
    if telemetry == "none":
        return CheckRecord(name="telemetry", status="skip", message="Telemetry was disabled for this run.")
    providers = sorted(
        {
            str(item.telemetry_metrics.get("telemetry_provider"))
            for item in inputs
            if item.telemetry_metrics.get("telemetry_provider") is not None
        }
    )
    warnings = [warning for item in inputs for warning in item.warnings]
    power_samples = sum(int(item.telemetry_metrics.get("power_sample_count") or 0) for item in inputs)
    if providers and power_samples > 0 and not warnings:
        return CheckRecord(
            name="telemetry",
            status="ok",
            message=f"Telemetry collected using {', '.join(providers)}.",
            details={"power_sample_count": power_samples},
        )
    if providers:
        return CheckRecord(
            name="telemetry",
            status="warn",
            message=f"Telemetry attempted using {', '.join(providers)} with warnings.",
            details={"power_sample_count": power_samples},
        )
    return CheckRecord(
        name="telemetry",
        status="warn",
        message="Telemetry was requested but no provider returned usable samples.",
        details={"power_sample_count": power_samples},
    )


def _recommendation_confidence(
    *,
    recommendation: RecommendationResult,
    inputs: list[RecommendationInput],
    scores: list[RecommendationScore],
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if recommendation.recommended_candidate_id is None:
        return "low", ["No candidate was recommended."]

    valid_scores = [score for score in scores if score.final_score is not None]
    total_failed = sum(_optional_int(item.measured_metrics.get("failed_requests")) or 0 for item in inputs)
    top_scores = sorted(valid_scores, key=lambda score: score.final_score or 0.0, reverse=True)
    margin = None
    if len(top_scores) >= 2 and top_scores[0].final_score is not None and top_scores[1].final_score is not None:
        margin = top_scores[0].final_score - top_scores[1].final_score

    level = "high"
    if len(valid_scores) < 2:
        level = "low"
        reasons.append("Only one valid candidate was evaluated.")
    else:
        reasons.append(f"{len(valid_scores)} valid candidates were evaluated.")

    if total_failed > 0:
        level = "low"
        reasons.append(f"{total_failed} benchmark requests failed across evaluated candidates.")
    else:
        reasons.append("No benchmark request failures were recorded.")

    if recommendation.telemetry_used_in_scoring:
        quality = str(recommendation.telemetry_metrics.get("telemetry_quality") or "unavailable")
        if quality == "good":
            reasons.append("Telemetry quality for the selected candidate was good.")
        elif quality == "limited":
            level = _min_confidence(level, "medium")
            reasons.append("Telemetry quality for the selected candidate was limited.")
        else:
            level = "low"
            reasons.append(f"Telemetry quality for the selected candidate was {quality}.")
        unavailable_capabilities = _unavailable_telemetry_capabilities(recommendation)
        if "gpu_utilization" in unavailable_capabilities:
            reasons.append("GPU utilization was unavailable from the selected telemetry provider.")
        if "memory_utilization" in unavailable_capabilities:
            reasons.append("Memory utilization was unavailable from the selected telemetry provider.")
    elif recommendation.power_missing_reason:
        level = _min_confidence(level, "medium")
        reasons.append(recommendation.power_missing_reason)

    if margin is None:
        reasons.append("Winner margin could not be compared against a runner-up.")
    elif margin < CONFIDENCE_SMALL_WIN_MARGIN:
        level = "low"
        reasons.append(f"Winner margin was small at {margin:.3f}.")
    elif margin < CONFIDENCE_CLEAR_WIN_MARGIN:
        level = _min_confidence(level, "medium")
        reasons.append(f"Winner margin was moderate at {margin:.3f}.")
    else:
        reasons.append(f"Winner margin was clear at {margin:.3f}.")
    return level, reasons


def _min_confidence(current: str, candidate: str) -> str:
    rank = {"low": 0, "medium": 1, "high": 2}
    return current if rank[current] <= rank[candidate] else candidate


def _unavailable_telemetry_capabilities(recommendation: RecommendationResult) -> set[str]:
    payload = recommendation.telemetry_metrics.get("telemetry_capabilities")
    if not isinstance(payload, dict):
        return set()
    fields = payload.get("unavailable_fields")
    if not isinstance(fields, list):
        return set()
    return {str(field) for field in fields}


def _join_ints(values: tuple[int, ...]) -> str:
    return ",".join(str(value) for value in values)


def _format_rate(value: float) -> str:
    return f"{value:,.2f}"


def _format_percent(value: float) -> str:
    return f"{value * 100.0:.1f}%"


def _format_seconds(value: float) -> str:
    return f"{value:.3f}s"


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _positive_number(value: object) -> bool:
    parsed = _optional_float(value)
    return parsed is not None and parsed > 0


def _ratio(measured: float | None, predicted: float | None) -> float | None:
    if measured is None or predicted in {None, 0}:
        return None
    return measured / predicted


def _round_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 6)
