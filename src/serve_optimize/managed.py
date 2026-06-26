"""Managed Evaluation Mode orchestration."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .backends.base import ManagedBackendAdapter
from .backends.factory import (
    create_managed_backend_adapter,
    normalize_managed_backend_name,
    validate_managed_backend_supported,
)
from .backends.sglang import SGLangArgumentCapabilities, render_sglang_launch
from .backends.vllm import VLLMArgumentCapabilities, render_vllm_launch
from .budget import ManagedBudgetPolicy, select_promotion_decisions, summarize_promotions
from .endpoint_benchmark import (
    DEFAULT_ENDPOINT_PROMPT,
    RequestFn,
    TelemetryCollectorFactory,
    aggregate_benchmark_summaries,
    make_run_id,
    run_endpoint_benchmark,
)
from .evidence import (
    DEFAULT_EVIDENCE_DB_PATH,
    EvidenceRecommendationRecord,
    EvidenceRunRecord,
    EvidenceStore,
    build_evidence_request_context,
    classify_evidence_lookup,
    launch_config_hash,
    measurement_from_summary,
    workload_config_hash,
)
from .hardware import detect_hardware
from .io import write_json, write_jsonl
from .managed_candidates import (
    CapabilityContext,
    ManagedCandidateGenerationResult,
    generate_managed_candidates_from_capabilities,
)
from .managed_summary import write_recommendation_summary_artifacts
from .modeling import infer_model_capability_metadata
from .preflight import PreflightRun, write_preflight_artifacts
from .priors import (
    AIConfiguratorPriorProvider,
    ManagedPriorPolicy,
    PriorProvider,
    apply_managed_prior_policy,
    evidence_lookup_to_prior,
)
from .recommendation import audit_recommendation_quality, score_recommendation_inputs
from .reporting import format_recommendation_report
from .runtime_environment import (
    collect_runtime_environment,
    stable_payload_hash,
)
from .schemas import (
    CandidateFailureRecord,
    EndpointBenchmarkConfig,
    EndpointBenchmarkPlan,
    EndpointBenchmarkSummary,
    EvaluationRung,
    Goal,
    HealthCheckResult,
    LaunchConfig,
    LaunchGroup,
    ManagedCandidateResult,
    ManagedLifecycleRecord,
    ManagedRunSummary,
    ModelCapabilityMetadata,
    PriorCandidate,
    PromotionDecision,
    RecommendationGoal,
    RecommendationInput,
    RecommendationResult,
    RungResult,
    ServeCandidate,
    ServerHandle,
    ServerLaunchSpec,
    ServingConfig,
    VllmServePlan,
    WorkloadConfig,
    WorkloadProfile,
    to_dict,
)
from .synthesis import (
    SYNTHESIS_SCHEMA_VERSION,
    SYNTHESIS_SOURCE,
    AIConfiguratorSynthesisProvider,
    CandidateSynthesisContext,
    CandidateSynthesisProvider,
    CandidateSynthesisResult,
    resolve_aiconfigurator_system_key,
    synthesis_result_to_artifact,
)
from .validation import CandidateValidationResult, validate_managed_candidate
from .workloads import slo_note

CandidateProvider = Callable[[], list[ServingConfig]]
ManagedProgressCallback = Callable[[str, dict[str, Any]], None]
WORKLOAD_EXTRA_KEYS = {
    "benchmark_duration_s",
    "dataset",
    "input_length",
    "max_new_tokens",
    "num_prompts",
    "num_requests",
    "num_requests_adjusted_reason",
    "output_length",
    "request_rate",
    "timeout_s",
    "trials",
    "warmup_duration_s",
    "warmup_requests",
    "idle_baseline_duration_s",
    "idle_power_watts",
    "soak_duration_s",
    "stream",
    "workload_concurrency",
    "workload_extra",
    "workload_id",
    "workload_prompt",
    "workload_profile",
    "prior_confidence",
    "prior_notes",
    "prior_source",
    "requested_num_requests",
    "requested_rung_num_requests",
    "raw_aiconfigurator_candidate",
    "base_workload_id",
    "measured_or_evidence_source",
    "promotion_reason",
    "promotion_status",
    "rung",
    "rung_index",
    "synthesis_confidence",
    "synthesis_constraints",
    "synthesis_rationale",
    "synthesis_status",
    "aiconfigurator_predicted_metrics",
    "aiconfigurator_rank",
    "aiconfigurator_system_key",
}
CLIENT_LIMITED_MIN_CONFIGS = 3
CLIENT_LIMITED_FLAT_THROUGHPUT_CV = 0.05
CLIENT_LIMITED_CPU_THRESHOLD_PERCENT = 90.0
LOAD_SUFFICIENCY_MIN_CONCURRENCY_LEVELS = 3
LOAD_SUFFICIENCY_FLAT_THROUGHPUT_CV = 0.05
LOAD_SUFFICIENCY_GPU_THRESHOLD_PERCENT = 85.0
LOAD_SUFFICIENCY_PRESSURE_GROWTH_RATIO = 1.2


class _ManagedExecutionState:
    def __init__(self) -> None:
        self.candidate_results: list[ManagedCandidateResult] = []
        self.failures: list[CandidateFailureRecord] = []
        self.completed = 0
        self.evidence_hits = 0
        self.resume_skips = 0
        self.cold_launch_count = 0
        self.workload_measurement_count = 0
        self.launch_group_rows: list[dict[str, object]] = []
        self.rung_results: list[RungResult] = []


ValidationRejection = tuple[ServingConfig, CandidateValidationResult]


@dataclass(frozen=True)
class _ManagedRecommendationArtifacts:
    status: str
    reason: str | None
    selected_config_id: str | None = None
    selected_evidence_key: str | None = None
    selected_measurement_id: str | None = None
    recommendation_score: float | None = None
    recommendation_confidence: str | None = None
    pareto_candidate_count: int = 0
    recommendation_artifact_path: str | None = None
    pareto_artifact_path: str | None = None
    report_artifact_path: str | None = None
    recommendation_summary_txt_path: str | None = None
    recommendation_summary_json_path: str | None = None
    recommendation_quality_audit: dict[str, Any] | None = None
    optimizer_quality: dict[str, Any] | None = None


@dataclass(frozen=True)
class _ResumeCandidateRecord:
    candidate_result: ManagedCandidateResult
    rung_result: RungResult
    summary_path: str
    launch_config_hash: str
    workload_config_hash: str


@dataclass(frozen=True)
class _ManagedResumeState:
    source_run_dir: Path
    source_run_id: str | None
    records: dict[tuple[str, str, str, str], _ResumeCandidateRecord]
    warnings: list[str]


def run_managed_evaluation(
    *,
    backend: str,
    model: str,
    goal: Goal,
    limit: int,
    trials: int,
    startup_timeout_s: float,
    cooldown_s: float,
    host: str,
    port: int | None,
    out_dir: Path,
    telemetry: str = "auto",
    request_timeout_s: float = 120.0,
    adapter: ManagedBackendAdapter | None = None,
    request_fn: RequestFn | None = None,
    telemetry_collector_factory: TelemetryCollectorFactory | None = None,
    candidate_provider: CandidateProvider | None = None,
    evidence_db_path: Path | None = DEFAULT_EVIDENCE_DB_PATH,
    evidence_write: bool = True,
    evidence_freshness_hours: float = 168.0,
    prior_provider: PriorProvider | None = None,
    prior_policy: ManagedPriorPolicy | None = None,
    synthesis_provider: CandidateSynthesisProvider | None = None,
    budget_policy: ManagedBudgetPolicy | None = None,
    command: list[str] | None = None,
    workload_profile: WorkloadProfile | None = None,
    warmup_requests: int = 0,
    steady_state_duration_s: float | None = None,
    idle_baseline_duration_s: float = 0.0,
    idle_power_watts: float | None = None,
    soak_duration_s: float | None = None,
    stream: bool = False,
    resume_from: Path | None = None,
    allow_remote_model_config_download: bool = False,
    progress_callback: ManagedProgressCallback | None = None,
) -> ManagedRunSummary:
    backend = normalize_managed_backend_name(backend)
    _validate_run_inputs(backend, limit, trials, startup_timeout_s, cooldown_s)
    adapter = adapter or _adapter_for_backend(backend)
    run_id = make_run_id(prefix="managed")
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _progress(
        progress_callback,
        "run_created",
        run_id=run_id,
        run_dir=str(run_dir),
        backend=backend,
        model=model,
        goal=goal.value,
    )
    resume_state = _load_managed_resume_state(
        resume_from,
        backend=backend,
        model=model,
        goal=goal,
    )

    launch_specs_path = run_dir / "launch_specs.jsonl"
    launch_groups_path = run_dir / "launch_groups.json"
    workload_configs_path = run_dir / "workload_configs.jsonl"
    lifecycle_path = run_dir / "server_lifecycle.jsonl"
    failures_path = run_dir / "candidate_failures.jsonl"
    prior_candidates_path = run_dir / "prior_candidates.json"
    prior_summary_path = run_dir / "prior_summary.json"
    evidence_decisions_path = run_dir / "evidence_decisions.jsonl"
    candidate_synthesis_path = run_dir / "candidate_synthesis.json"
    evaluation_rungs_path = run_dir / "evaluation_rungs.json"
    promotion_decisions_path = run_dir / "promotion_decisions.jsonl"
    recommendation_path = run_dir / "managed_recommendation.json"
    pareto_frontier_path = run_dir / "managed_pareto_frontier.json"
    pareto_frontier_csv_path = run_dir / "managed_pareto_frontier.csv"
    report_path = run_dir / "managed_report.txt"
    recommendation_summary_txt_path = run_dir / "recommendation_summary.txt"
    recommendation_summary_json_path = run_dir / "recommendation_summary.json"
    optimizer_quality_path = run_dir / "optimizer_quality.json"
    optimizer_failure_cache_path = run_dir / "optimizer_failure_cache.json"
    client_saturation_path = run_dir / "client_saturation.json"
    load_sufficiency_path = run_dir / "load_sufficiency.json"
    vllm_argument_capabilities_path = run_dir / "vllm_argument_capabilities.json"
    sglang_argument_capabilities_path = run_dir / "sglang_argument_capabilities.json"
    rendered_launch_configs_path = run_dir / "rendered_launch_configs.jsonl"
    runtime_environment_path = run_dir / "runtime_environment.json"
    write_jsonl(launch_specs_path, [])
    write_jsonl(rendered_launch_configs_path, [])
    write_json(launch_groups_path, [])
    write_jsonl(workload_configs_path, [])
    write_jsonl(lifecycle_path, [])
    write_jsonl(failures_path, [])
    write_jsonl(evidence_decisions_path, [])
    write_json(prior_candidates_path, [])
    write_json(prior_summary_path, {})
    write_json(candidate_synthesis_path, _empty_synthesis_summary())
    write_json(evaluation_rungs_path, [])
    write_jsonl(promotion_decisions_path, [])
    write_json(optimizer_quality_path, {})
    write_json(optimizer_failure_cache_path, {})
    write_json(client_saturation_path, {})
    write_json(load_sufficiency_path, {})

    hardware = detect_hardware()
    workload_profile = workload_profile or WorkloadProfile()
    backend_metadata = _backend_metadata(adapter, backend)
    vllm_argument_capabilities = _backend_argument_capabilities(adapter, backend)
    sglang_argument_capabilities = _backend_sglang_argument_capabilities(adapter, backend)
    runtime_environment = collect_runtime_environment(
        backend_name=backend,
        backend_version=_optional_str(backend_metadata.get("version")),
    ).to_artifact()
    _progress(
        progress_callback,
        "environment_collected",
        backend=backend,
        backend_version=_optional_str(backend_metadata.get("version")),
        runtime_fingerprint=_optional_str(runtime_environment.get("environment_fingerprint")),
    )
    write_json(runtime_environment_path, runtime_environment)
    if vllm_argument_capabilities is not None:
        write_json(vllm_argument_capabilities_path, vllm_argument_capabilities.to_artifact())
    if sglang_argument_capabilities is not None:
        write_json(sglang_argument_capabilities_path, sglang_argument_capabilities.to_artifact())
    evidence_store, evidence_warnings = _open_evidence_store(
        evidence_db_path=evidence_db_path,
        evidence_write=evidence_write,
    )
    _progress(
        progress_callback,
        "evidence_ready",
        db_path=str(evidence_db_path) if evidence_db_path is not None else None,
        write_enabled=evidence_write and evidence_db_path is not None,
        freshness_hours=evidence_freshness_hours,
        warning_count=len(evidence_warnings),
    )
    if evidence_store is not None:
        try:
            context_for_run = build_evidence_request_context(
                hardware=hardware,
                backend=backend,
                backend_metadata=backend_metadata,
                model=model,
                telemetry=telemetry,
                launch_config={},
                workload_config={},
                goal=goal.value,
                trials=trials,
                runtime_environment=runtime_environment,
                rendered_launch_command=[],
                backend_capability_help_hash=_backend_capability_help_hash(
                    vllm_argument_capabilities,
                    sglang_argument_capabilities,
                ),
            )
            evidence_store.insert_run(
                EvidenceRunRecord(
                    run_id=run_id,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    command=json.dumps(command or [], sort_keys=True),
                    mode="managed",
                    hardware_fingerprint=context_for_run.hardware_fingerprint,
                    backend_fingerprint=context_for_run.backend_fingerprint,
                    model_fingerprint=context_for_run.model_fingerprint,
                    telemetry_fingerprint=context_for_run.telemetry_fingerprint,
                    runtime_fingerprint=runtime_environment.get(
                        "environment_fingerprint"
                    ),
                    runtime_environment_json=runtime_environment,
                    output_dir=str(run_dir),
                    metadata_json={
                        "schema_version": "managed-evidence-run/v1",
                        "backend": backend,
                        "goal": goal.value,
                        "freshness_hours": evidence_freshness_hours,
                        "runtime_environment": runtime_environment,
                    },
                )
            )
        except Exception as exc:
            evidence_warnings.append(f"Evidence DB run write failed: {exc.__class__.__name__}: {exc}")
            evidence_store.close()
            evidence_store = None

    model_metadata = infer_model_capability_metadata(
        model,
        allow_remote_download=allow_remote_model_config_download,
    )
    _progress(
        progress_callback,
        "model_metadata",
        metadata_known=model_metadata.metadata_known,
        config_path=model_metadata.config_path,
        quantization_method=model_metadata.quantization_method,
    )
    candidate_generation = _provided_candidate_generation()
    if candidate_provider is not None:
        candidate_pool = candidate_provider()
        candidate_pool = _with_measurement_quality_options(
            candidate_pool,
            warmup_requests=warmup_requests,
            steady_state_duration_s=steady_state_duration_s,
            idle_baseline_duration_s=idle_baseline_duration_s,
            idle_power_watts=idle_power_watts,
            soak_duration_s=soak_duration_s,
            stream=stream,
        )
        candidate_pool = candidate_pool[:limit]
        candidate_generation = _provided_candidate_generation(candidate_pool)
    else:
        candidate_generation = _generate_managed_candidate_generation(
            backend=backend,
            model=model,
            goal=goal,
            limit=limit,
            hardware=hardware,
            model_metadata=model_metadata,
            backend_metadata=backend_metadata,
            vllm_argument_capabilities=vllm_argument_capabilities,
            sglang_argument_capabilities=sglang_argument_capabilities,
            workload_profile=workload_profile,
        )
        candidate_pool = candidate_generation.candidates
        candidate_pool = _with_measurement_quality_options(
            candidate_pool,
            warmup_requests=warmup_requests,
            steady_state_duration_s=steady_state_duration_s,
            idle_baseline_duration_s=idle_baseline_duration_s,
            idle_power_watts=idle_power_watts,
            soak_duration_s=soak_duration_s,
            stream=stream,
        )
    _progress(
        progress_callback,
        "candidates_generated",
        candidate_count=len(candidate_pool),
        source_counts=candidate_generation.candidate_source_counts,
    )
    valid_candidates, validation_rejections = _validate_managed_candidate_pool(
        candidate_pool,
        backend=backend,
        model_metadata=model_metadata,
        limit=limit,
        backfill_valid_candidates=candidate_provider is None,
        vllm_argument_capabilities=vllm_argument_capabilities,
        sglang_argument_capabilities=sglang_argument_capabilities,
    )
    canonical_candidates, canonical_rejections, rendered_launch_rows = _canonicalize_valid_candidates(
        valid_candidates,
        backend=backend,
        vllm_argument_capabilities=vllm_argument_capabilities,
        sglang_argument_capabilities=sglang_argument_capabilities,
        runtime_environment=runtime_environment,
    )
    valid_candidates = canonical_candidates
    validation_rejections.extend(canonical_rejections)
    _progress(
        progress_callback,
        "candidates_validated",
        valid_count=len(valid_candidates),
        rejected_count=len(validation_rejections),
    )
    synthesis_summary = _empty_synthesis_summary()
    if candidate_provider is None:
        synthesis_summary = _synthesize_managed_candidates(
            provider=synthesis_provider or AIConfiguratorSynthesisProvider(),
            run_dir=run_dir,
            initial_candidates=valid_candidates,
            validation_rejections=validation_rejections,
            rendered_launch_rows=rendered_launch_rows,
            backend=backend,
            model=model,
            goal=goal,
            hardware=hardware,
            model_metadata=model_metadata,
            vllm_argument_capabilities=vllm_argument_capabilities,
            sglang_argument_capabilities=sglang_argument_capabilities,
            telemetry=telemetry,
            trials=trials,
            request_timeout_s=request_timeout_s,
            evidence_store=evidence_store,
            evidence_freshness_hours=evidence_freshness_hours,
            evidence_warnings=evidence_warnings,
            backend_metadata=backend_metadata,
            runtime_environment=runtime_environment,
            evidence_decisions_path=evidence_decisions_path,
        )
        valid_candidates = list(synthesis_summary.pop("_valid_candidates"))
        validation_rejections = list(synthesis_summary.pop("_validation_rejections"))
        rendered_launch_rows = list(synthesis_summary.pop("_rendered_launch_rows"))
        _progress(
            progress_callback,
            "candidates_synthesized",
            valid_count=len(valid_candidates),
            rejected_count=len(validation_rejections),
        )
    _append_jsonl(rendered_launch_configs_path, rendered_launch_rows)
    valid_candidate_count_before_prior_pruning = len(valid_candidates)
    rejected_candidate_count_before_prior_pruning = len(validation_rejections)
    candidate_source_count_configs = [config for config, _validation in validation_rejections] + valid_candidates
    preflight = _preflight_evidence(
        candidates=valid_candidates,
        hardware=hardware,
        backend=backend,
        backend_metadata=backend_metadata,
        runtime_environment=runtime_environment,
        model=model,
        telemetry=telemetry,
        goal=goal,
        trials=trials,
        request_timeout_s=request_timeout_s,
        evidence_store=evidence_store,
        evidence_freshness_hours=evidence_freshness_hours,
        evidence_warnings=evidence_warnings,
        evidence_decisions_path=evidence_decisions_path,
        vllm_argument_capabilities=vllm_argument_capabilities,
        sglang_argument_capabilities=sglang_argument_capabilities,
    )
    exact_fresh_ids = set(preflight["exact_fresh_ids"])
    evidence_priors = list(preflight["evidence_priors"])
    _progress(
        progress_callback,
        "evidence_preflight",
        exact_fresh_count=len(exact_fresh_ids),
        evidence_prior_count=len(evidence_priors),
    )
    synthesis_summary["final_preflight"] = {
        "exact_fresh_candidate_ids": sorted(exact_fresh_ids),
        "evidence_prior_count": len(evidence_priors),
    }
    prior_results = []
    should_collect_priors = bool(valid_candidates) and len(exact_fresh_ids) < len(valid_candidates)
    if should_collect_priors:
        active_prior_provider = prior_provider
        if active_prior_provider is None and candidate_provider is None:
            system_resolution = resolve_aiconfigurator_system_key(hardware)
            if system_resolution.system_key:
                active_prior_provider = AIConfiguratorPriorProvider(system=system_resolution.system_key)
        if active_prior_provider is not None:
            prior_results.append(
                active_prior_provider.collect_priors(
                    model=model,
                    backend=backend,
                    goal=goal,
                    candidates=valid_candidates,
                    out_dir=run_dir / "priors",
                )
            )
    pruning = apply_managed_prior_policy(
        valid_candidates,
        prior_results=prior_results,
        evidence_priors=evidence_priors,
        exact_fresh_candidate_ids=exact_fresh_ids,
        policy=prior_policy,
    )
    valid_candidates = pruning.candidates
    _progress(
        progress_callback,
        "prior_pruning",
        remaining_count=len(valid_candidates),
        pruned_count=pruning.candidates_pruned_by_prior,
        prior_sources_used=pruning.prior_sources_used,
    )
    all_prior_candidates = [prior for result in prior_results for prior in result.candidates] + evidence_priors
    write_json(prior_candidates_path, all_prior_candidates)
    write_json(
        prior_summary_path,
        {
            **pruning.summary,
            "prior_sources_used": pruning.prior_sources_used,
            "prior_candidate_count": pruning.prior_candidate_count,
            "candidates_pruned_by_prior": pruning.candidates_pruned_by_prior,
            "ai_configurator_available": pruning.ai_configurator_available,
            "ai_configurator_used": pruning.ai_configurator_used,
            "provider_results": prior_results,
            "evidence_prior_count": len(evidence_priors),
            "all_exact_fresh_evidence": bool(valid_candidates) and len(exact_fresh_ids) == len(valid_candidates),
        },
    )
    pruned_ids = set(pruning.summary.get("pruned_candidate_ids", [])) if isinstance(pruning.summary, dict) else set()
    synthesis_summary = _finalize_synthesis_summary(
        synthesis_summary,
        valid_candidate_ids={config.id for config in valid_candidates},
        pruned_candidate_ids={str(candidate_id) for candidate_id in pruned_ids},
        validation_rejections=validation_rejections,
    )
    write_json(candidate_synthesis_path, synthesis_summary)
    candidates = [config for config, _validation in validation_rejections] + valid_candidates
    budget_policy = budget_policy or ManagedBudgetPolicy.default()
    staged_evaluation = budget_policy.should_stage(len(valid_candidates))
    active_rungs = budget_policy.rungs if staged_evaluation else [_regular_measure_rung()]
    write_json(evaluation_rungs_path, active_rungs)

    state = _ManagedExecutionState()
    promotion_decisions_by_rung: dict[str, list[PromotionDecision]] = {}
    configs_by_id = {config.id: config for config in candidates}

    for config, validation in validation_rejections:
        failure = _candidate_failure(
            run_id,
            config.id,
            "validation",
            validation.reason or "Candidate failed managed validation.",
            details={
                "reason": _validation_failure_reason(validation),
                "backend": config.backend,
                "candidate_source": (config.extra or {}).get("candidate_source"),
                "model_metadata_known": model_metadata.metadata_known,
                "model_config_path": model_metadata.config_path,
                "model_quantization_method": model_metadata.quantization_method,
                "candidate_quantization": config.quantization,
                "synthesis_rationale": (config.extra or {}).get("synthesis_rationale"),
                "synthesis_confidence": (config.extra or {}).get("synthesis_confidence"),
                "synthesis_constraints": (config.extra or {}).get("synthesis_constraints"),
            },
        )
        state.failures.append(failure)
        _append_jsonl(failures_path, [failure])
        state.candidate_results.append(_rejected_result(config, failure))
        _progress(
            progress_callback,
            "candidate_rejected",
            candidate_id=config.id,
            stage="validation",
            reason=failure.error,
        )

    if staged_evaluation:
        active_candidates = valid_candidates
        for index, rung in enumerate(active_rungs):
            if not active_candidates:
                break
            rung_configs = _configs_for_rung(
                active_candidates,
                rung=rung,
                base_trials=trials,
                request_timeout_s=request_timeout_s,
                telemetry=telemetry,
                promotion_status="probe" if rung.index == 0 else "promoted",
                promotion_reason="initial probe" if rung.index == 0 else "promoted from previous rung",
            )
            configs_by_id.update({config.id: config for config in rung_configs})
            launch_groups = group_candidates_by_launch_config(
                rung_configs,
                trials=trials,
                request_timeout_s=request_timeout_s,
                telemetry=telemetry,
            )
            launch_groups = [_rung_launch_group(group, rung) for group in launch_groups]
            _append_jsonl(workload_configs_path, [workload for group in launch_groups for workload in group.workload_configs])
            _progress(
                progress_callback,
                "rung_start",
                rung=rung.name,
                rung_index=rung.index,
                candidate_count=len(rung_configs),
                launch_group_count=len(launch_groups),
                workload_count=sum(len(group.workload_configs) for group in launch_groups),
            )
            _evaluate_launch_groups(
                state=state,
                launch_groups=launch_groups,
                configs_by_id=configs_by_id,
                run_id=run_id,
                run_dir=run_dir,
                backend=backend,
                model=model,
                goal=goal,
                telemetry=telemetry,
                backend_metadata=backend_metadata,
                runtime_environment=runtime_environment,
                hardware=hardware,
                adapter=adapter,
                host=host,
                port=port,
                startup_timeout_s=startup_timeout_s,
                cooldown_s=cooldown_s,
                request_fn=request_fn,
                telemetry_collector_factory=telemetry_collector_factory,
                evidence_store=evidence_store,
                evidence_freshness_hours=evidence_freshness_hours,
                evidence_warnings=evidence_warnings,
                evidence_decisions_path=evidence_decisions_path,
                vllm_argument_capabilities=vllm_argument_capabilities,
                sglang_argument_capabilities=sglang_argument_capabilities,
                lifecycle_path=lifecycle_path,
                failures_path=failures_path,
                launch_specs_path=launch_specs_path,
                resume_state=resume_state,
                progress_callback=progress_callback,
            )
            next_rung = active_rungs[index + 1] if index + 1 < len(active_rungs) else None
            if next_rung is None:
                continue
            rung_results = [result for result in state.rung_results if result.rung == rung.name]
            decisions = select_promotion_decisions(
                candidates=active_candidates,
                results=rung_results,
                prior_by_config_id=pruning.prior_by_config_id,
                goal=goal,
                from_rung=rung,
                to_rung=next_rung,
                policy=budget_policy,
            )
            promotion_decisions_by_rung[rung.name] = decisions
            _append_jsonl(promotion_decisions_path, decisions)
            promoted_ids = {decision.candidate_id for decision in decisions if decision.promoted}
            _progress(
                progress_callback,
                "promotion",
                from_rung=rung.name,
                to_rung=next_rung.name,
                promoted_count=len(promoted_ids),
                candidate_count=len(decisions),
            )
            active_candidates = [config for config in active_candidates if config.id in promoted_ids]
    else:
        rung = active_rungs[0]
        rung_configs = _configs_for_rung(
            valid_candidates,
            rung=rung,
            base_trials=trials,
            request_timeout_s=request_timeout_s,
            telemetry=telemetry,
            promotion_status="exhaustive",
            promotion_reason="single regular measurement rung",
        )
        configs_by_id.update({config.id: config for config in rung_configs})
        launch_groups = group_candidates_by_launch_config(
            rung_configs,
            trials=trials,
            request_timeout_s=request_timeout_s,
            telemetry=telemetry,
        )
        launch_groups = [_rung_launch_group(group, rung) for group in launch_groups]
        _append_jsonl(workload_configs_path, [workload for group in launch_groups for workload in group.workload_configs])
        _progress(
            progress_callback,
            "rung_start",
            rung=rung.name,
            rung_index=rung.index,
            candidate_count=len(rung_configs),
            launch_group_count=len(launch_groups),
            workload_count=sum(len(group.workload_configs) for group in launch_groups),
        )
        _evaluate_launch_groups(
            state=state,
            launch_groups=launch_groups,
            configs_by_id=configs_by_id,
            run_id=run_id,
            run_dir=run_dir,
            backend=backend,
            model=model,
            goal=goal,
            telemetry=telemetry,
            backend_metadata=backend_metadata,
            runtime_environment=runtime_environment,
            hardware=hardware,
            adapter=adapter,
            host=host,
            port=port,
            startup_timeout_s=startup_timeout_s,
            cooldown_s=cooldown_s,
            request_fn=request_fn,
            telemetry_collector_factory=telemetry_collector_factory,
            evidence_store=evidence_store,
            evidence_freshness_hours=evidence_freshness_hours,
            evidence_warnings=evidence_warnings,
            evidence_decisions_path=evidence_decisions_path,
            vllm_argument_capabilities=vllm_argument_capabilities,
            sglang_argument_capabilities=sglang_argument_capabilities,
            lifecycle_path=lifecycle_path,
            failures_path=failures_path,
            launch_specs_path=launch_specs_path,
            resume_state=resume_state,
            progress_callback=progress_callback,
        )

    promotion_summary = summarize_promotions(
        policy_name=budget_policy.name if staged_evaluation else "exhaustive",
        candidate_count=len(valid_candidates),
        rung_count=len(active_rungs),
        probe_candidate_count=len(valid_candidates),
        decisions_by_rung=promotion_decisions_by_rung,
    )
    synthesis_summary = _apply_synthesis_execution_status(synthesis_summary, state.candidate_results)
    write_json(candidate_synthesis_path, synthesis_summary)
    write_json(launch_groups_path, state.launch_group_rows)
    client_saturation = _client_saturation_summary(state.rung_results)
    write_json(client_saturation_path, client_saturation)
    _progress(
        progress_callback,
        "client_saturation",
        classification=client_saturation.get("classification"),
        candidate_count=client_saturation.get("candidate_count"),
        max_client_cpu_utilization_percent=client_saturation.get("max_client_cpu_utilization_percent"),
        throughput_coefficient_of_variation=client_saturation.get("throughput_coefficient_of_variation"),
    )
    load_sufficiency = _load_sufficiency_summary(
        state.rung_results,
        goal=goal,
        client_saturation=client_saturation,
    )
    write_json(load_sufficiency_path, load_sufficiency)
    _progress(
        progress_callback,
        "load_sufficiency",
        classification=load_sufficiency.get("classification"),
        concurrency_level_count=load_sufficiency.get("concurrency_level_count"),
        max_gpu_util_percent=load_sufficiency.get("max_gpu_util_percent"),
        pressure_growth_ratio=load_sufficiency.get("pressure_growth_ratio"),
    )
    _progress(
        progress_callback,
        "recommendation_start",
        measured_row_count=len(state.rung_results),
    )
    managed_recommendation = _write_managed_recommendation_artifacts(
        run_id=run_id,
        run_dir=run_dir,
        backend=backend,
        model=model,
        goal=goal,
        telemetry=telemetry,
        configs_by_id=configs_by_id,
        rung_results=state.rung_results,
        evidence_store=evidence_store,
        evidence_warnings=evidence_warnings,
        recommendation_path=recommendation_path,
        pareto_frontier_path=pareto_frontier_path,
        pareto_frontier_csv_path=pareto_frontier_csv_path,
        report_path=report_path,
        recommendation_summary_txt_path=recommendation_summary_txt_path,
        recommendation_summary_json_path=recommendation_summary_json_path,
        optimizer_quality_path=optimizer_quality_path,
        vllm_argument_capabilities=vllm_argument_capabilities,
        sglang_argument_capabilities=sglang_argument_capabilities,
        runtime_environment=runtime_environment,
        client_saturation=client_saturation,
        load_sufficiency=load_sufficiency,
    )
    _progress(
        progress_callback,
        "recommendation_complete",
        status=managed_recommendation.status,
        selected_config_id=managed_recommendation.selected_config_id,
        confidence=managed_recommendation.recommendation_confidence,
        reason=managed_recommendation.reason,
    )
    optimizer_failure_cache = _write_optimizer_failure_cache(
        optimizer_failure_cache_path,
        failures=state.failures,
        configs_by_id=configs_by_id,
    )

    if evidence_store is not None:
        evidence_store.close()

    completed_candidate_count = _unique_result_count(state.candidate_results, statuses={"completed", "evidence_hit", "resumed"})
    status = _run_status(completed_candidate_count, len(state.failures))
    artifacts = {
        "run_dir": str(run_dir),
        "managed_run_json": str(run_dir / "managed_run.json"),
        "launch_specs_jsonl": str(launch_specs_path),
        "rendered_launch_configs_jsonl": str(rendered_launch_configs_path),
        "runtime_environment_json": str(runtime_environment_path),
        "launch_groups_json": str(launch_groups_path),
        "workload_configs_jsonl": str(workload_configs_path),
        "server_lifecycle_jsonl": str(lifecycle_path),
        "candidate_failures_jsonl": str(failures_path),
        "prior_candidates_json": str(prior_candidates_path),
        "prior_summary_json": str(prior_summary_path),
        "evidence_decisions_jsonl": str(evidence_decisions_path),
        "candidate_synthesis_json": str(candidate_synthesis_path),
        "evaluation_rungs_json": str(evaluation_rungs_path),
        "promotion_decisions_jsonl": str(promotion_decisions_path),
        "managed_recommendation_json": str(recommendation_path),
        "managed_pareto_frontier_json": str(pareto_frontier_path),
        "managed_pareto_frontier_csv": str(pareto_frontier_csv_path),
        "managed_report_txt": str(report_path),
        "recommendation_summary_txt": str(recommendation_summary_txt_path),
        "recommendation_summary_json": str(recommendation_summary_json_path),
        "optimizer_quality_json": str(optimizer_quality_path),
        "optimizer_failure_cache_json": str(optimizer_failure_cache_path),
        "client_saturation_json": str(client_saturation_path),
        "load_sufficiency_json": str(load_sufficiency_path),
        "logs_dir": str(run_dir / "logs"),
    }
    if vllm_argument_capabilities is not None:
        artifacts["vllm_argument_capabilities_json"] = str(vllm_argument_capabilities_path)
    if sglang_argument_capabilities is not None:
        artifacts["sglang_argument_capabilities_json"] = str(sglang_argument_capabilities_path)
    if evidence_db_path is not None and evidence_write:
        artifacts["evidence_db"] = str(evidence_db_path)

    summary = ManagedRunSummary(
        run_id=run_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        backend=backend,
        model=model,
        goal=goal.value,
        candidate_count=len(candidates),
        completed_candidate_count=completed_candidate_count,
        failed_candidate_count=len(state.failures),
        startup_timeout_s=startup_timeout_s,
        cooldown_s=cooldown_s,
        trials=trials,
        status=status,
        artifacts=artifacts,
        candidates=state.candidate_results,
        evidence_db_path=str(evidence_db_path) if evidence_db_path is not None and evidence_write else None,
        evidence_write_enabled=evidence_write and evidence_db_path is not None,
        evidence_hit_candidate_count=state.evidence_hits,
        evidence_hits=state.evidence_hits,
        evidence_warnings=evidence_warnings,
        evidence_decision_summary=_evidence_decision_summary(evidence_decisions_path),
        cold_launch_count=state.cold_launch_count,
        cold_launches=state.cold_launch_count,
        workload_measurement_count=state.workload_measurement_count,
        workload_measurements=state.workload_measurement_count,
        skipped_by_evidence_count=state.evidence_hits,
        launch_groups_count=len(state.launch_group_rows),
        average_workloads_per_launch=round(state.workload_measurement_count / state.cold_launch_count, 6) if state.cold_launch_count else 0.0,
        backend_metadata=backend_metadata,
        runtime_environment=runtime_environment,
        prior_sources_used=pruning.prior_sources_used,
        prior_candidate_count=pruning.prior_candidate_count,
        candidates_after_prior_pruning=len(valid_candidates),
        candidates_pruned_by_prior=pruning.candidates_pruned_by_prior,
        ai_configurator_available=pruning.ai_configurator_available,
        ai_configurator_used=pruning.ai_configurator_used,
        candidate_source_counts=_candidate_source_counts(candidate_source_count_configs),
        capability_filtered_count=candidate_generation.capability_filtered_count,
        invalid_quantization_filtered_count=candidate_generation.invalid_quantization_filtered_count,
        safe_baseline_added=candidate_generation.safe_baseline_added,
        workload_profile=to_dict(workload_profile),
        valid_candidate_count_before_prior_pruning=valid_candidate_count_before_prior_pruning,
        rejected_candidate_count_before_prior_pruning=rejected_candidate_count_before_prior_pruning,
        budget_policy_name=promotion_summary.policy_name,
        rung_count=len(active_rungs),
        probe_measurement_count=_rung_measurement_count(state.rung_results, "probe"),
        promoted_measurement_count=_rung_measurement_count(state.rung_results, "measure"),
        validation_measurement_count=_rung_measurement_count(state.rung_results, "validate"),
        pruned_after_probe_count=promotion_summary.pruned_after_probe_count,
        promotion_summary=promotion_summary,
        recommendation_status=managed_recommendation.status,
        recommendation_reason=managed_recommendation.reason,
        selected_config_id=managed_recommendation.selected_config_id,
        selected_evidence_key=managed_recommendation.selected_evidence_key,
        selected_measurement_id=managed_recommendation.selected_measurement_id,
        recommendation_score=managed_recommendation.recommendation_score,
        recommendation_confidence=managed_recommendation.recommendation_confidence,
        pareto_candidate_count=managed_recommendation.pareto_candidate_count,
        recommendation_artifact_path=managed_recommendation.recommendation_artifact_path,
        pareto_artifact_path=managed_recommendation.pareto_artifact_path,
        recommendation_summary_txt_path=managed_recommendation.recommendation_summary_txt_path,
        recommendation_summary_json_path=managed_recommendation.recommendation_summary_json_path,
        recommendation_quality_audit=managed_recommendation.recommendation_quality_audit or {},
        optimizer_quality={
            **(managed_recommendation.optimizer_quality or {}),
            "failure_cache": optimizer_failure_cache.get("summary", {}),
        },
        synthesis_summary=synthesis_summary,
        resume_from_run_dir=str(resume_state.source_run_dir) if resume_state is not None else None,
        resume_source_run_id=resume_state.source_run_id if resume_state is not None else None,
        resume_loaded_candidate_count=len(resume_state.records) if resume_state is not None else 0,
        resume_skipped_candidate_count=state.resume_skips,
        resume_warnings=resume_state.warnings if resume_state is not None else [],
        client_saturation=client_saturation,
        load_sufficiency=load_sufficiency,
    )
    write_json(run_dir / "managed_run.json", summary)
    _progress(
        progress_callback,
        "run_complete",
        status=summary.status,
        completed_candidate_count=summary.completed_candidate_count,
        candidate_count=summary.candidate_count,
        failed_candidate_count=summary.failed_candidate_count,
        run_dir=str(run_dir),
    )
    return summary


def build_managed_preflight(
    *,
    backend: str,
    model: str,
    goal: Goal,
    limit: int,
    trials: int,
    startup_timeout_s: float,
    cooldown_s: float,
    host: str,
    port: int | None,
    out_dir: Path,
    telemetry: str = "auto",
    request_timeout_s: float = 120.0,
    adapter: ManagedBackendAdapter | None = None,
    candidate_provider: CandidateProvider | None = None,
    evidence_db_path: Path | None = DEFAULT_EVIDENCE_DB_PATH,
    evidence_write: bool = True,
    evidence_freshness_hours: float = 168.0,
    budget_policy: ManagedBudgetPolicy | None = None,
    workload_profile: WorkloadProfile | None = None,
    warmup_requests: int = 0,
    steady_state_duration_s: float | None = None,
    idle_baseline_duration_s: float = 0.0,
    idle_power_watts: float | None = None,
    soak_duration_s: float | None = None,
    stream: bool = False,
    resume_from: Path | None = None,
) -> PreflightRun:
    backend = normalize_managed_backend_name(backend)
    _validate_run_inputs(backend, limit, trials, startup_timeout_s, cooldown_s)
    adapter = adapter or _adapter_for_backend(backend)
    run_id = make_run_id(prefix="managed-preflight")
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    hardware = detect_hardware()
    workload_profile = workload_profile or WorkloadProfile()
    backend_metadata = _backend_metadata(adapter, backend)
    vllm_argument_capabilities = _backend_argument_capabilities(adapter, backend)
    sglang_argument_capabilities = _backend_sglang_argument_capabilities(adapter, backend)
    runtime_environment = collect_runtime_environment(
        backend_name=backend,
        backend_version=_optional_str(backend_metadata.get("version")),
    ).to_artifact()
    model_metadata = infer_model_capability_metadata(model)
    if candidate_provider is not None:
        candidate_pool = candidate_provider()[:limit]
        candidate_pool = _with_measurement_quality_options(
            candidate_pool,
            warmup_requests=warmup_requests,
            steady_state_duration_s=steady_state_duration_s,
            idle_baseline_duration_s=idle_baseline_duration_s,
            idle_power_watts=idle_power_watts,
            soak_duration_s=soak_duration_s,
            stream=stream,
        )
        candidate_generation = _provided_candidate_generation(candidate_pool)
    else:
        candidate_generation = _generate_managed_candidate_generation(
            backend=backend,
            model=model,
            goal=goal,
            limit=limit,
            hardware=hardware,
            model_metadata=model_metadata,
            backend_metadata=backend_metadata,
            vllm_argument_capabilities=vllm_argument_capabilities,
            sglang_argument_capabilities=sglang_argument_capabilities,
            workload_profile=workload_profile,
        )
        candidate_pool = candidate_generation.candidates
        candidate_pool = _with_measurement_quality_options(
            candidate_pool,
            warmup_requests=warmup_requests,
            steady_state_duration_s=steady_state_duration_s,
            idle_baseline_duration_s=idle_baseline_duration_s,
            idle_power_watts=idle_power_watts,
            soak_duration_s=soak_duration_s,
            stream=stream,
        )
    valid_candidates, validation_rejections = _validate_managed_candidate_pool(
        candidate_pool,
        backend=backend,
        model_metadata=model_metadata,
        limit=limit,
        backfill_valid_candidates=candidate_provider is None,
        vllm_argument_capabilities=vllm_argument_capabilities,
        sglang_argument_capabilities=sglang_argument_capabilities,
    )
    valid_candidates, canonical_rejections, rendered_launch_rows = _canonicalize_valid_candidates(
        valid_candidates,
        backend=backend,
        vllm_argument_capabilities=vllm_argument_capabilities,
        sglang_argument_capabilities=sglang_argument_capabilities,
        runtime_environment=runtime_environment,
    )
    validation_rejections.extend(canonical_rejections)

    budget_policy = budget_policy or ManagedBudgetPolicy.default()
    staged_evaluation = budget_policy.should_stage(len(valid_candidates))
    active_rungs = budget_policy.rungs if staged_evaluation else [_regular_measure_rung()]
    preview_rung = active_rungs[0]
    preview_candidates = _configs_for_rung(
        valid_candidates,
        rung=preview_rung,
        base_trials=trials,
        request_timeout_s=request_timeout_s,
        telemetry=telemetry,
        promotion_status="probe" if staged_evaluation else "exhaustive",
        promotion_reason="initial probe" if staged_evaluation else "single regular measurement rung",
    )
    launch_groups = [
        _rung_launch_group(group, preview_rung)
        for group in group_candidates_by_launch_config(
            preview_candidates,
            trials=trials,
            request_timeout_s=request_timeout_s,
            telemetry=telemetry,
        )
    ]
    workload_configs = [workload for group in launch_groups for workload in group.workload_configs]
    planned_measurements = sum(workload.trials for workload in workload_configs)
    failures = [
        _candidate_failure(
            run_id,
            config.id,
            "validation",
            validation.reason or "Candidate failed managed validation.",
            details={"backend": config.backend},
        )
        for config, validation in validation_rejections
    ]

    rendered_launch_configs_path = run_dir / "rendered_launch_configs.jsonl"
    workload_configs_path = run_dir / "workload_configs.jsonl"
    launch_groups_path = run_dir / "launch_groups.json"
    failures_path = run_dir / "candidate_failures.jsonl"
    write_jsonl(rendered_launch_configs_path, rendered_launch_rows)
    write_jsonl(workload_configs_path, workload_configs)
    write_json(launch_groups_path, launch_groups)
    write_jsonl(failures_path, failures)

    payload = {
        "schema_version": "serve-optimize-preflight/v1",
        "run_id": run_id,
        "mode": "managed",
        "status": "planned",
        "dry_run": True,
        "backend": backend,
        "model": model,
        "goal": goal.value,
        "telemetry": telemetry,
        "host": host,
        "port": port,
        "backend_available": bool(adapter.is_available()),
        "backend_metadata": backend_metadata,
        "runtime_environment": runtime_environment,
        "capability_help_hash": _backend_capability_help_hash(vllm_argument_capabilities, sglang_argument_capabilities),
        "workload_profile": to_dict(workload_profile),
        "candidates": {
            "requested_limit": limit,
            "generated_count": len(candidate_pool),
            "valid_count": len(valid_candidates),
            "rejected_count": len(validation_rejections),
            "source_counts": candidate_generation.candidate_source_counts,
        },
        "budget": {
            "policy": budget_policy.name if staged_evaluation else "exhaustive",
            "staged": staged_evaluation,
            "rung_count": len(active_rungs),
            "preview_rung": preview_rung.name,
            "launch_group_count": len(launch_groups),
            "planned_workload_measurements": planned_measurements,
            "startup_timeout_s": startup_timeout_s,
            "cooldown_s": cooldown_s,
            "later_rungs_depend_on_promotions": staged_evaluation,
        },
        "evidence": {
            "db_path": str(evidence_db_path) if evidence_db_path is not None else None,
            "write_enabled": evidence_write and evidence_db_path is not None,
            "freshness_hours": evidence_freshness_hours,
            "exact_reuse": "planned for execution; dry run does not read or write measured evidence",
            "resume_from": str(resume_from) if resume_from is not None else None,
        },
        "safety": {
            "will_call_endpoint": False,
            "will_launch_servers": False,
            "will_write_measured_evidence": False,
        },
        "outputs": {
            "rendered_launch_configs": str(rendered_launch_configs_path),
            "workload_configs": str(workload_configs_path),
            "launch_groups": str(launch_groups_path),
            "candidate_failures": str(failures_path),
        },
        "guidance": {
            "execute": "Run the same command without --dry-run to launch managed servers, health check, benchmark, and write measured evidence.",
            "repeat": "Run the same measured command again with the same evidence database to allow exact fresh evidence reuse.",
            "resume": "Pass --resume-from with a previous managed run directory to reuse completed workloads whose launch and workload identities still match.",
        },
        "warnings": [],
        "notes": [
            "Dry run renders commands and workload plans only.",
            "No backend process is started and no endpoint request is sent.",
        ],
    }
    return write_preflight_artifacts(run_dir, payload)


def _regular_measure_rung() -> EvaluationRung:
    return EvaluationRung(
        index=0,
        name="measure",
        purpose="Regular exhaustive managed measurement.",
        num_requests_scale=1.0,
        min_num_requests=1,
        trials=None,
        promotion_fraction=1.0,
    )


def _configs_for_rung(
    candidates: list[ServingConfig],
    *,
    rung: EvaluationRung,
    base_trials: int,
    request_timeout_s: float,
    telemetry: str,
    promotion_status: str,
    promotion_reason: str,
) -> list[ServingConfig]:
    rung_configs: list[ServingConfig] = []
    for config in candidates:
        base_workload = serving_config_to_workload_config(
            config,
            trials=base_trials,
            request_timeout_s=request_timeout_s,
            telemetry=telemetry,
        )
        num_requests = max(rung.min_num_requests, int(base_workload.num_requests * rung.num_requests_scale))
        if rung.max_num_requests is not None:
            num_requests = min(num_requests, rung.max_num_requests)
        requested_rung_num_requests = num_requests
        minimum_rung_requests = base_workload.concurrency + base_workload.warmup_requests
        num_requests = max(num_requests, minimum_rung_requests)
        trial_count = max(1, rung.trials if rung.trials is not None else base_workload.trials)
        extra = dict(config.extra or {})
        extra.update(
            {
                "base_workload_id": base_workload.workload_id,
                "num_requests": num_requests,
                "workload_id": f"{base_workload.workload_id}-{rung.name}",
                "rung": rung.name,
                "rung_index": rung.index,
                "promotion_status": promotion_status,
                "promotion_reason": promotion_reason,
                "measured_or_evidence_source": "pending",
                "trials": trial_count,
            }
        )
        if requested_rung_num_requests < minimum_rung_requests:
            extra["requested_rung_num_requests"] = requested_rung_num_requests
            extra["num_requests_adjusted_reason"] = "raised_to_match_concurrency"
        rung_configs.append(replace(config, extra=extra))
    return rung_configs


def _load_managed_resume_state(
    resume_from: Path | None,
    *,
    backend: str,
    model: str,
    goal: Goal,
) -> _ManagedResumeState | None:
    if resume_from is None:
        return None
    source_run_dir = resume_from.resolve()
    managed_run_path = source_run_dir / "managed_run.json"
    if not managed_run_path.is_file():
        raise ValueError(f"resume run directory is missing managed_run.json: {source_run_dir}")
    payload = _read_json_object(managed_run_path)
    if str(payload.get("backend") or "") != backend:
        raise ValueError("resume run backend does not match current managed evaluation.")
    if str(payload.get("model") or "") != model:
        raise ValueError("resume run model does not match current managed evaluation.")
    if str(payload.get("goal") or "") != goal.value:
        raise ValueError("resume run goal does not match current managed evaluation.")

    warnings: list[str] = []
    workload_hashes = _resume_workload_hashes(source_run_dir, warnings)
    launch_hashes = _resume_launch_hashes(source_run_dir, warnings)
    records: dict[tuple[str, str, str, str], _ResumeCandidateRecord] = {}
    for row in payload.get("candidates", []):
        if not isinstance(row, dict) or row.get("status") != "completed":
            continue
        candidate = _dataclass_from_dict(ManagedCandidateResult, row)
        if candidate is None:
            warnings.append("Skipped a resume candidate with an unreadable result row.")
            continue
        summary_path = _resume_summary_path(candidate.summary_paths, source_run_dir)
        if summary_path is None:
            warnings.append(f"Skipped resume candidate {candidate.config_id}: no readable summary path.")
            continue
        summary = _load_endpoint_summary(summary_path)
        if summary is None:
            warnings.append(f"Skipped resume candidate {candidate.config_id}: summary could not be loaded.")
            continue
        workload_id = summary.run_id
        workload_key = (candidate.config_id, workload_id)
        launch_hash = launch_hashes.get(workload_key)
        workload_hash = workload_hashes.get(workload_key)
        if not launch_hash or not workload_hash:
            warnings.append(f"Skipped resume candidate {candidate.config_id}: missing launch or workload identity.")
            continue
        resumed_candidate = replace(
            candidate,
            status="resumed",
            measured_or_evidence_source="resume",
        )
        rung_result = RungResult(
            candidate_id=candidate.config_id,
            workload_id=workload_id,
            rung=candidate.rung or "measure",
            rung_index=_rung_index_for_name(candidate.rung),
            status="resumed",
            measured_or_evidence_source="resume",
            evidence_key=candidate.evidence_key,
            evidence_measurement_id=candidate.evidence_measurement_id,
            metrics=_summary_metrics(summary),
        )
        records[(candidate.config_id, workload_id, launch_hash, workload_hash)] = _ResumeCandidateRecord(
            candidate_result=resumed_candidate,
            rung_result=rung_result,
            summary_path=str(summary_path),
            launch_config_hash=launch_hash,
            workload_config_hash=workload_hash,
        )
    return _ManagedResumeState(
        source_run_dir=source_run_dir,
        source_run_id=_optional_str(payload.get("run_id")),
        records=records,
        warnings=warnings,
    )


def _resume_record_for_workload(
    resume_state: _ManagedResumeState | None,
    *,
    workload: WorkloadConfig,
    launch_config_hash: str,
) -> _ResumeCandidateRecord | None:
    if resume_state is None:
        return None
    key = (
        workload.candidate_id,
        workload.workload_id,
        launch_config_hash,
        workload_config_hash(workload),
    )
    return resume_state.records.get(key)


def _resume_workload_hashes(source_run_dir: Path, warnings: list[str]) -> dict[tuple[str, str], str]:
    hashes: dict[tuple[str, str], str] = {}
    for row in _read_jsonl_objects(source_run_dir / "workload_configs.jsonl", warnings):
        candidate_id = _optional_str(row.get("candidate_id"))
        workload_id = _optional_str(row.get("workload_id"))
        if candidate_id and workload_id:
            hashes[(candidate_id, workload_id)] = workload_config_hash(row)
    return hashes


def _resume_launch_hashes(source_run_dir: Path, warnings: list[str]) -> dict[tuple[str, str], str]:
    hashes: dict[tuple[str, str], str] = {}
    payload = _read_json_value(source_run_dir / "launch_groups.json", warnings)
    if not isinstance(payload, list):
        return hashes
    for group in payload:
        if not isinstance(group, dict):
            continue
        launch_hash = _optional_str(group.get("launch_config_hash"))
        if not launch_hash:
            continue
        workloads = group.get("workload_configs")
        if not isinstance(workloads, list):
            continue
        for workload in workloads:
            if not isinstance(workload, dict):
                continue
            candidate_id = _optional_str(workload.get("candidate_id"))
            workload_id = _optional_str(workload.get("workload_id"))
            if candidate_id and workload_id:
                hashes[(candidate_id, workload_id)] = launch_hash
    return hashes


def _resume_summary_path(paths: list[str], source_run_dir: Path) -> Path | None:
    for raw_path in reversed(paths):
        path = Path(raw_path)
        candidates = [path] if path.is_absolute() else [path, source_run_dir / path]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    return None


def _load_endpoint_summary(path: Path) -> EndpointBenchmarkSummary | None:
    try:
        row = _read_json_object(path)
        return EndpointBenchmarkSummary(**{field.name: row[field.name] for field in fields(EndpointBenchmarkSummary) if field.name in row})
    except (OSError, TypeError, ValueError, KeyError):
        return None


def _rung_index_for_name(name: str | None) -> int:
    return {"probe": 0, "measure": 1, "validate": 2}.get(str(name or "measure"), 0)


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return payload


def _read_json_value(path: Path, warnings: list[str]) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        warnings.append(f"Could not read resume artifact {path.name}: {exc.__class__.__name__}: {exc}")
    except json.JSONDecodeError as exc:
        warnings.append(f"Could not parse resume artifact {path.name}: {exc.msg}")
    return None


def _read_jsonl_objects(path: Path, warnings: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        warnings.append(f"Could not read resume artifact {path.name}: {exc.__class__.__name__}: {exc}")
        return rows
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            warnings.append(f"Could not parse {path.name} line {line_number}: {exc.msg}")
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _dataclass_from_dict(cls: type[Any], row: dict[str, Any]) -> Any | None:
    names = {field.name for field in fields(cls)}
    try:
        return cls(**{key: value for key, value in row.items() if key in names})
    except TypeError:
        return None


def _rung_launch_group(group: LaunchGroup, rung: EvaluationRung) -> LaunchGroup:
    return replace(group, group_id=f"{group.group_id}-{rung.name}")


def _evaluate_launch_groups(
    *,
    state: _ManagedExecutionState,
    launch_groups: list[LaunchGroup],
    configs_by_id: dict[str, ServingConfig],
    run_id: str,
    run_dir: Path,
    backend: str,
    model: str,
    goal: Goal,
    telemetry: str,
    backend_metadata: dict[str, object],
    runtime_environment: dict[str, object],
    hardware,
    adapter: ManagedBackendAdapter,
    host: str,
    port: int | None,
    startup_timeout_s: float,
    cooldown_s: float,
    request_fn: RequestFn | None,
    telemetry_collector_factory: TelemetryCollectorFactory | None,
    evidence_store: EvidenceStore | None,
    evidence_freshness_hours: float,
    evidence_warnings: list[str],
    evidence_decisions_path: Path,
    vllm_argument_capabilities: VLLMArgumentCapabilities | None,
    sglang_argument_capabilities: SGLangArgumentCapabilities | None,
    lifecycle_path: Path,
    failures_path: Path,
    launch_specs_path: Path,
    resume_state: _ManagedResumeState | None = None,
    progress_callback: ManagedProgressCallback | None = None,
) -> None:
    del telemetry
    for group in launch_groups:
        _progress(
            progress_callback,
            "launch_group_start",
            group_id=group.group_id,
            rung=_group_rung(group),
            workload_count=len(group.workload_configs),
        )
        group_metadata: list[dict[str, object]] = []
        pending_workloads: list[tuple[ServingConfig, WorkloadConfig, object]] = []
        handle: ServerHandle | None = None
        resumed_workload_count = 0
        representative_config = configs_by_id[group.original_config_ids[0]]
        try:
            for workload in group.workload_configs:
                config = configs_by_id[workload.candidate_id]
                resume_record = _resume_record_for_workload(
                    resume_state,
                    workload=workload,
                    launch_config_hash=group.launch_config_hash,
                )
                if resume_record is not None:
                    resumed_workload_count += 1
                    state.resume_skips += 1
                    state.completed += 1
                    state.rung_results.append(resume_record.rung_result)
                    state.candidate_results.append(resume_record.candidate_result)
                    metadata = {
                        "candidate_id": workload.candidate_id,
                        "workload_id": workload.workload_id,
                        "rung": workload.rung,
                        "source_run_id": resume_state.source_run_id if resume_state is not None else None,
                        "source_run_dir": str(resume_state.source_run_dir) if resume_state is not None else None,
                        "summary_path": resume_record.summary_path,
                        "launch_config_hash": resume_record.launch_config_hash,
                        "workload_config_hash": resume_record.workload_config_hash,
                        "used_as_resume": True,
                    }
                    group_metadata.append(metadata)
                    _append_jsonl(
                        lifecycle_path,
                        [
                            _lifecycle(
                                run_id,
                                workload.candidate_id,
                                backend,
                                "resume_skip",
                                "completed",
                                message="Previous completed measured workload matched current launch and workload identity.",
                                details={**metadata, "group_id": group.group_id},
                            )
                        ],
                    )
                    _progress(
                        progress_callback,
                        "resume_hit",
                        candidate_id=workload.candidate_id,
                        workload_id=workload.workload_id,
                        rung=workload.rung,
                        summary_path=resume_record.summary_path,
                    )
                    continue
                evidence_context = build_evidence_request_context(
                    hardware=hardware,
                    backend=backend,
                    backend_metadata=backend_metadata,
                    model=model,
                    telemetry=workload.telemetry,
                    launch_config=group.launch_config,
                    workload_config=workload,
                    goal=goal.value,
                    trials=workload.trials,
                    runtime_environment=runtime_environment,
                    rendered_launch_command=_rendered_launch_command(
                        config,
                        backend=backend,
                        vllm_argument_capabilities=vllm_argument_capabilities,
                        sglang_argument_capabilities=sglang_argument_capabilities,
                    ),
                    backend_capability_help_hash=_backend_capability_help_hash(
                        vllm_argument_capabilities,
                        sglang_argument_capabilities,
                    ),
                )
                if evidence_store is not None:
                    try:
                        lookup = evidence_store.lookup_evidence(evidence_context, freshness_hours=evidence_freshness_hours)
                        decision = classify_evidence_lookup(
                            lookup,
                            candidate_id=workload.candidate_id,
                            context=evidence_context,
                            current_backend_argument_capabilities=(
                                vllm_argument_capabilities
                                or sglang_argument_capabilities
                            ),
                            goal=goal.value,
                        )
                        _append_jsonl(evidence_decisions_path, [decision.to_artifact()])
                        metadata = {
                            "candidate_id": workload.candidate_id,
                            "workload_id": workload.workload_id,
                            "rung": workload.rung,
                            "evidence_key": evidence_context.evidence_key,
                            "hit_type": lookup.hit_type.value,
                            "classification": decision.classification.value,
                            "used_as_exact": decision.used_as_exact,
                            "used_as_prior": decision.used_as_prior,
                            "measurement_id": lookup.measurement.get("measurement_id") if lookup.measurement else None,
                        }
                        group_metadata.append(metadata)
                        _progress(
                            progress_callback,
                            "evidence_lookup",
                            candidate_id=workload.candidate_id,
                            workload_id=workload.workload_id,
                            rung=workload.rung,
                            hit_type=lookup.hit_type.value,
                            classification=decision.classification.value,
                            used_as_exact=decision.used_as_exact,
                        )
                        _append_jsonl(
                            lifecycle_path,
                            [
                                _lifecycle(
                                    run_id,
                                    workload.candidate_id,
                                    backend,
                                    "evidence_lookup",
                                    lookup.hit_type.value,
                                    message=lookup.reason,
                                    details={**metadata, "group_id": group.group_id},
                                )
                            ],
                        )
                        if decision.used_as_exact:
                            state.evidence_hits += 1
                            state.completed += 1
                            state.rung_results.append(
                                _rung_result_from_evidence(
                                    workload,
                                    evidence_context,
                                    lookup,
                                )
                            )
                            state.candidate_results.append(
                                ManagedCandidateResult(
                                    config_id=workload.candidate_id,
                                    backend=config.backend,
                                    status="evidence_hit",
                                    evidence_key=evidence_context.evidence_key,
                                    evidence_hit_type=lookup.hit_type.value,
                                    evidence_measurement_id=str(lookup.measurement.get("measurement_id")) if lookup.measurement else None,
                                    **_prior_result_fields(config),
                                    **_workload_result_fields(workload, source="evidence"),
                                )
                            )
                            _progress(
                                progress_callback,
                                "evidence_hit",
                                candidate_id=workload.candidate_id,
                                workload_id=workload.workload_id,
                                rung=workload.rung,
                                evidence_key=evidence_context.evidence_key,
                                measurement_id=lookup.measurement.get("measurement_id") if lookup.measurement else None,
                            )
                            continue
                    except Exception as exc:
                        warning = f"Evidence DB lookup failed for {workload.candidate_id}: {exc.__class__.__name__}: {exc}"
                        evidence_warnings.append(warning)
                        _progress(
                            progress_callback,
                            "warning",
                            candidate_id=workload.candidate_id,
                            workload_id=workload.workload_id,
                            message=warning,
                        )
                        group_metadata.append(
                            {
                                "candidate_id": workload.candidate_id,
                                "workload_id": workload.workload_id,
                                "rung": workload.rung,
                                "hit_type": "lookup_failed",
                                "error": warning,
                            }
                        )
                        _append_jsonl(lifecycle_path, [_lifecycle(run_id, workload.candidate_id, backend, "evidence_lookup", "failed", message=warning, details={"group_id": group.group_id})])
                pending_workloads.append((config, workload, evidence_context))

            if not pending_workloads:
                if resumed_workload_count:
                    status = "resume_completed"
                    message = "All remaining workloads in launch group were completed in the resumed run."
                else:
                    status = "evidence_hit"
                    message = "All workloads in launch group had exact fresh measured evidence."
                _append_jsonl(
                    lifecycle_path,
                    [
                        _lifecycle(
                            run_id,
                            group.group_id,
                            backend,
                            "launch_skipped",
                            status,
                            message=message,
                            details={"group_id": group.group_id, "workload_count": len(group.workload_configs), "rung": _group_rung(group)},
                        )
                    ],
                )
                _progress(
                    progress_callback,
                    "launch_skipped",
                    group_id=group.group_id,
                    rung=_group_rung(group),
                    status=status,
                    message=message,
                    workload_count=len(group.workload_configs),
                )
                continue

            if not adapter.is_available():
                _progress(
                    progress_callback,
                    "backend_unavailable",
                    backend=backend,
                    group_id=group.group_id,
                    pending_workload_count=len(pending_workloads),
                )
                for config, workload, _evidence_context in pending_workloads:
                    failure = _candidate_failure(
                        run_id,
                        workload.candidate_id,
                        "availability",
                        f"Managed backend '{backend}' is not available.",
                        details={"group_id": group.group_id, "rung": workload.rung, "reason": "backend_unavailable"},
                    )
                    state.failures.append(failure)
                    _append_jsonl(failures_path, [failure])
                    state.candidate_results.append(_failed_result(config, failure, workload=workload))
                continue

            launch_config = replace(representative_config, id=group.group_id)
            spec = adapter.build_launch_spec(launch_config, host=host, port=port, log_dir=run_dir / "logs")
            launch_provenance = _launch_provenance_from_spec(spec)
            _append_jsonl(launch_specs_path, [spec])
            _append_jsonl(lifecycle_path, [_lifecycle(run_id, group.group_id, backend, "launch_spec", "ok", details={"base_url": spec.base_url, "group_id": group.group_id, "rung": _group_rung(group), **launch_provenance})])
            _progress(
                progress_callback,
                "launch_start",
                group_id=group.group_id,
                rung=_group_rung(group),
                base_url=spec.base_url,
                command=" ".join(spec.command),
                backend_version=launch_provenance.get("backend_version"),
                effective_values=launch_provenance.get("backend_effective_values"),
                stdout_log_path=spec.stdout_log_path,
                stderr_log_path=spec.stderr_log_path,
                pending_workload_count=len(pending_workloads),
            )

            handle = adapter.launch_server(spec)
            state.cold_launch_count += 1
            _progress(
                progress_callback,
                "launch_started",
                group_id=group.group_id,
                pid=handle.pid,
                pgid=handle.pgid,
                base_url=handle.base_url,
            )
            _append_jsonl(
                lifecycle_path,
                [
                    _lifecycle(
                        run_id,
                        group.group_id,
                        backend,
                        "launch",
                        "ok",
                        message="Server process launched for launch group.",
                        pid=handle.pid,
                        pgid=handle.pgid,
                        details={"group_id": group.group_id, "workload_count": len(pending_workloads), "rung": _group_rung(group)},
                    )
                ],
            )

            _progress(
                progress_callback,
                "health_wait",
                group_id=group.group_id,
                base_url=handle.base_url,
                timeout_s=startup_timeout_s,
            )
            health = adapter.wait_for_health(handle, model=model, timeout_s=startup_timeout_s, request_fn=request_fn)
            _append_jsonl(lifecycle_path, [_lifecycle(run_id, group.group_id, backend, "health", "ok" if health.healthy else "failed", details={**to_dict(health), "group_id": group.group_id, "rung": _group_rung(group)})])
            _progress(
                progress_callback,
                "health_result",
                group_id=group.group_id,
                healthy=health.healthy,
                status=health.status,
                attempts=health.attempts,
                error=health.error,
            )
            if not health.healthy:
                failure_reason = _failure_reason_for_health(health)
                for config, workload, _evidence_context in pending_workloads:
                    failure = _candidate_failure(
                        run_id,
                        workload.candidate_id,
                        "health",
                        health.error or health.status,
                        details={**to_dict(health), "group_id": group.group_id, "rung": workload.rung, "reason": failure_reason},
                    )
                    state.failures.append(failure)
                    _append_jsonl(failures_path, [failure])
                    state.candidate_results.append(_failed_result(config, failure, workload=workload))
                continue

            for config, workload, evidence_context in pending_workloads:
                _progress(
                    progress_callback,
                    "workload_start",
                    candidate_id=workload.candidate_id,
                    workload_id=workload.workload_id,
                    rung=workload.rung,
                    concurrency=workload.concurrency,
                    num_requests=workload.num_requests,
                    warmup_requests=workload.warmup_requests,
                    trials=workload.trials,
                    max_new_tokens=workload.max_new_tokens,
                )
                benchmark_run_dirs: list[str] = []
                summary_paths: list[str] = []
                trial_summaries: list[EndpointBenchmarkSummary] = []
                last_summary = None
                measurement_id = None
                for trial in range(workload.trials):
                    benchmark_config = _benchmark_config_from_workload(
                        workload=workload,
                        base_url=handle.base_url,
                        model=model,
                        trial=trial,
                        launch_provenance=launch_provenance,
                    )
                    _progress(
                        progress_callback,
                        "trial_start",
                        candidate_id=workload.candidate_id,
                        workload_id=workload.workload_id,
                        rung=workload.rung,
                        trial=trial + 1,
                        trials=workload.trials,
                    )
                    benchmark = run_endpoint_benchmark(
                        config=benchmark_config,
                        out_dir=run_dir / "per_candidate",
                        prediction=None,
                        hardware=None,
                        request_fn=request_fn,
                        telemetry_collector_factory=telemetry_collector_factory,
                    )
                    last_summary = benchmark.summary
                    trial_summaries.append(benchmark.summary)
                    benchmark_run_dirs.append(str(benchmark.run_dir))
                    summary_paths.append(str(benchmark.run_dir / "summary.json"))
                    state.workload_measurement_count += 1
                    _progress(
                        progress_callback,
                        "trial_complete",
                        candidate_id=workload.candidate_id,
                        workload_id=workload.workload_id,
                        rung=workload.rung,
                        trial=trial + 1,
                        trials=workload.trials,
                        run_dir=str(benchmark.run_dir),
                        **_summary_progress_fields(benchmark.summary),
                    )

                if trial_summaries:
                    last_summary = aggregate_benchmark_summaries(workload.workload_id, trial_summaries)
                    aggregate_dir = run_dir / "per_candidate" / f"{workload.workload_id}-aggregate"
                    aggregate_dir.mkdir(parents=True, exist_ok=True)
                    write_json(aggregate_dir / "summary.json", last_summary)
                    summary_paths.append(str(aggregate_dir / "summary.json"))
                    if evidence_store is not None:
                        try:
                            measurement = measurement_from_summary(
                                run_id=run_id,
                                context=evidence_context,
                                summary=last_summary,
                                raw_json={
                                    "summary": last_summary,
                                    "trial_summaries": trial_summaries,
                                    "config": _benchmark_config_from_workload(
                                        workload=workload,
                                        base_url=handle.base_url,
                                        model=model,
                                        trial=0,
                                        launch_provenance=launch_provenance,
                                    ),
                                    "candidate": config,
                                    "launch_group": group,
                                    "workload": replace(workload, measured_or_evidence_source="measured"),
                                    "run_dirs": benchmark_run_dirs,
                                    "aggregate_run_dir": str(aggregate_dir),
                                    "rung": workload.rung,
                                    "promotion_status": workload.promotion_status,
                                    "promotion_reason": workload.promotion_reason,
                                    "runtime_fingerprint": evidence_context.runtime_fingerprint,
                                    "runtime_environment": evidence_context.runtime_environment,
                                },
                            )
                            evidence_store.insert_measurement(measurement)
                            measurement_id = measurement.measurement_id
                            _progress(
                                progress_callback,
                                "evidence_write",
                                candidate_id=workload.candidate_id,
                                workload_id=workload.workload_id,
                                status="ok",
                                evidence_key=evidence_context.evidence_key,
                                measurement_id=measurement.measurement_id,
                            )
                            _append_jsonl(
                                lifecycle_path,
                                [
                                    _lifecycle(
                                        run_id,
                                        workload.candidate_id,
                                        backend,
                                        "evidence_write",
                                        "ok",
                                        details={
                                            "evidence_key": evidence_context.evidence_key,
                                            "measurement_id": measurement.measurement_id,
                                            "group_id": group.group_id,
                                            "workload_id": workload.workload_id,
                                            "rung": workload.rung,
                                            "trial_count": len(trial_summaries),
                                        },
                                    )
                                ],
                            )
                        except Exception as exc:
                            warning = f"Evidence DB measurement write failed for {workload.candidate_id}: {exc.__class__.__name__}: {exc}"
                            evidence_warnings.append(warning)
                            _progress(
                                progress_callback,
                                "evidence_write",
                                candidate_id=workload.candidate_id,
                                workload_id=workload.workload_id,
                                status="failed",
                                message=warning,
                            )
                            _append_jsonl(lifecycle_path, [_lifecycle(run_id, workload.candidate_id, backend, "evidence_write", "failed", message=warning, details={"group_id": group.group_id, "rung": workload.rung})])

                state.completed += 1
                if last_summary is not None:
                    state.rung_results.append(
                        _rung_result_from_summary(
                            workload,
                            evidence_context,
                            last_summary,
                            measurement_id=measurement_id,
                        )
                    )
                state.candidate_results.append(
                    ManagedCandidateResult(
                        config_id=workload.candidate_id,
                        backend=config.backend,
                        status="completed",
                        benchmark_run_dirs=benchmark_run_dirs,
                        summary_paths=summary_paths,
                        evidence_key=evidence_context.evidence_key,
                        evidence_measurement_id=measurement_id,
                        **_prior_result_fields(config),
                        **_workload_result_fields(workload, source="measured"),
                    )
                )
                _append_jsonl(lifecycle_path, [_lifecycle(run_id, workload.candidate_id, backend, "benchmark", "ok", details={"trials": workload.trials, "group_id": group.group_id, "workload_id": workload.workload_id, "rung": workload.rung})])
                _progress(
                    progress_callback,
                    "workload_complete",
                    candidate_id=workload.candidate_id,
                    workload_id=workload.workload_id,
                    rung=workload.rung,
                    summary_path=summary_paths[-1] if summary_paths else None,
                    evidence_measurement_id=measurement_id,
                    **_summary_progress_fields(last_summary),
                )
        except KeyboardInterrupt as exc:
            failed_workloads = pending_workloads or [
                (configs_by_id[candidate_id], workload, None)
                for candidate_id in group.original_config_ids
                for workload in group.workload_configs
                if workload.candidate_id == candidate_id
            ]
            message = f"{exc.__class__.__name__}: {exc or 'interrupted by operator'}"
            for config, workload, _evidence_context in failed_workloads:
                failure = _candidate_failure(
                    run_id,
                    workload.candidate_id,
                    "interruption",
                    message,
                    details={"group_id": group.group_id, "rung": workload.rung},
                )
                state.failures.append(failure)
                _append_jsonl(failures_path, [failure])
                state.candidate_results.append(
                    _failed_result(config, failure, workload=workload)
                )
            _append_jsonl(
                lifecycle_path,
                [
                    _lifecycle(
                        run_id,
                        group.group_id,
                        backend,
                        "interruption",
                        "failed",
                        message=message,
                        pid=handle.pid if handle is not None else None,
                        pgid=handle.pgid if handle is not None else None,
                        details={
                            "group_id": group.group_id,
                            "rung": _group_rung(group),
                        },
                    )
                ],
            )
            raise
        except Exception as exc:
            failure_stage = "launch" if handle is None else "benchmark"
            failure_reason = _failure_reason_for_exception(exc, stage=failure_stage)
            _progress(
                progress_callback,
                "launch_group_failed",
                group_id=group.group_id,
                rung=_group_rung(group),
                stage=failure_stage,
                error=f"{exc.__class__.__name__}: {exc}",
            )
            failed_workloads = pending_workloads or [(configs_by_id[candidate_id], workload, None) for candidate_id in group.original_config_ids for workload in group.workload_configs if workload.candidate_id == candidate_id]
            for config, workload, _evidence_context in failed_workloads:
                failure = _candidate_failure(run_id, workload.candidate_id, failure_stage, f"{exc.__class__.__name__}: {exc}", details={"group_id": group.group_id, "rung": workload.rung, "reason": failure_reason})
                state.failures.append(failure)
                _append_jsonl(failures_path, [failure])
                state.candidate_results.append(_failed_result(config, failure, workload=workload))
        finally:
            state.launch_group_rows.append(
                {
                    **to_dict(group),
                    "rung": _group_rung(group),
                    "evidence_lookup_metadata": group_metadata,
                    "pending_workload_count": len(pending_workloads),
                    "resumed_workload_count": resumed_workload_count,
                }
            )
            if handle is not None:
                try:
                    stop_record = replace(adapter.stop_server(handle), run_id=run_id, config_id=group.group_id)
                except Exception as exc:
                    stop_record = _lifecycle(run_id, group.group_id, backend, "stop", "failed", message=f"{exc.__class__.__name__}: {exc}", pid=handle.pid, pgid=handle.pgid)
                _append_jsonl(lifecycle_path, [stop_record])
                _progress(
                    progress_callback,
                    "server_stop",
                    group_id=group.group_id,
                    status=stop_record.status,
                    returncode=stop_record.returncode,
                )
                if cooldown_s > 0:
                    _progress(
                        progress_callback,
                        "cooldown",
                        group_id=group.group_id,
                        seconds=cooldown_s,
                    )
                    time.sleep(cooldown_s)


def _progress(
    progress_callback: ManagedProgressCallback | None,
    event: str,
    **details: Any,
) -> None:
    if progress_callback is None:
        return
    progress_callback(
        event,
        {
            "event": event,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **details,
        },
    )


def _summary_progress_fields(summary: object | None) -> dict[str, object]:
    if summary is None:
        return {}
    measurement_quality = getattr(summary, "measurement_quality", None)
    if not isinstance(measurement_quality, dict):
        measurement_quality = {}
    measured_requests = getattr(summary, "measured_requests", None)
    if measured_requests is None:
        measured_requests = getattr(summary, "total_requests", None)
    return {
        "throughput_tokens_per_sec": _optional_float(getattr(summary, "total_tokens_s", None)),
        "request_rate": _optional_float(getattr(summary, "request_rate_req_s", None)),
        "p95_latency_s": _optional_float(getattr(summary, "p95_latency_s", None)),
        "failed_requests": _optional_int(getattr(summary, "failed_requests", None)),
        "measured_requests": _optional_int(measured_requests),
        "successful_requests": _optional_int(getattr(summary, "successful_requests", None)),
        "concurrency_coverage": _optional_str(measurement_quality.get("concurrency_coverage")),
        "client_cpu_utilization_percent": _optional_float(getattr(summary, "client_cpu_utilization_percent", None)),
        "p95_client_queue_s": _optional_float(getattr(summary, "p95_client_queue_s", None)),
        "load_saturation_signal": _optional_str(getattr(summary, "load_saturation_signal", None)),
        "client_issue_rate_req_s": _optional_float(getattr(summary, "client_issue_rate_req_s", None)),
    }


def _rung_result_from_summary(
    workload: WorkloadConfig,
    evidence_context: object,
    summary: object,
    *,
    measurement_id: str | None = None,
) -> RungResult:
    return RungResult(
        candidate_id=workload.candidate_id,
        workload_id=workload.workload_id,
        rung=workload.rung or "measure",
        rung_index=workload.rung_index or 0,
        status="completed",
        measured_or_evidence_source="measured",
        evidence_key=getattr(evidence_context, "evidence_key", None),
        evidence_measurement_id=measurement_id,
        runtime_fingerprint=getattr(
            evidence_context,
            "runtime_fingerprint",
            None,
        ),
        runtime_environment=dict(
            getattr(evidence_context, "runtime_environment", {}) or {}
        ),
        metrics=_summary_metrics(summary),
    )


def _rung_result_from_evidence(
    workload: WorkloadConfig,
    evidence_context: object,
    lookup: object,
) -> RungResult:
    measurement = getattr(lookup, "measurement", None) or {}
    return RungResult(
        candidate_id=workload.candidate_id,
        workload_id=workload.workload_id,
        rung=workload.rung or "measure",
        rung_index=workload.rung_index or 0,
        status="evidence_hit",
        measured_or_evidence_source="evidence",
        evidence_key=getattr(evidence_context, "evidence_key", None),
        evidence_hit_type=getattr(getattr(lookup, "hit_type", None), "value", None),
        evidence_measurement_id=str(measurement.get("measurement_id")) if measurement.get("measurement_id") else None,
        runtime_fingerprint=getattr(
            evidence_context,
            "runtime_fingerprint",
            None,
        ),
        runtime_environment=dict(
            getattr(evidence_context, "runtime_environment", {}) or {}
        ),
        metrics=_measurement_metrics(measurement),
    )


def _write_managed_recommendation_artifacts(
    *,
    run_id: str,
    run_dir: Path,
    backend: str,
    model: str,
    goal: Goal,
    telemetry: str,
    configs_by_id: dict[str, ServingConfig],
    rung_results: list[RungResult],
    evidence_store: EvidenceStore | None,
    evidence_warnings: list[str],
    recommendation_path: Path,
    pareto_frontier_path: Path,
    pareto_frontier_csv_path: Path,
    report_path: Path,
    recommendation_summary_txt_path: Path,
    recommendation_summary_json_path: Path,
    optimizer_quality_path: Path,
    vllm_argument_capabilities: VLLMArgumentCapabilities | None,
    sglang_argument_capabilities: SGLangArgumentCapabilities | None,
    runtime_environment: dict[str, object],
    client_saturation: dict[str, Any],
    load_sufficiency: dict[str, Any],
) -> _ManagedRecommendationArtifacts:
    inputs, input_metadata = _managed_recommendation_inputs(
        configs_by_id=configs_by_id,
        rung_results=rung_results,
        model=model,
        backend=backend,
        vllm_argument_capabilities=vllm_argument_capabilities,
        sglang_argument_capabilities=sglang_argument_capabilities,
    )
    recommendation_goal = _managed_recommendation_goal(goal)
    scores, recommendation = score_recommendation_inputs(
        inputs,
        goal=recommendation_goal,
        metadata_notes=[
            "Managed Mode recommendations use only measured results and exact fresh measured evidence hits.",
            *([note] if (note := slo_note(_workload_profile_from_candidates(list(configs_by_id.values())))) else []),
            *_client_saturation_notes(client_saturation),
            *_load_sufficiency_notes(load_sufficiency),
        ],
    )
    recommendation = _managed_recommendation_result(
        recommendation,
        backend=backend,
        model=model,
        telemetry=telemetry,
        artifacts={
            "run_dir": str(run_dir),
            "managed_recommendation_json": str(recommendation_path),
            "managed_pareto_frontier_json": str(pareto_frontier_path),
            "managed_pareto_frontier_csv": str(pareto_frontier_csv_path),
            "managed_report_txt": str(report_path),
            "recommendation_summary_txt": str(recommendation_summary_txt_path),
            "recommendation_summary_json": str(recommendation_summary_json_path),
        },
    )
    selected = input_metadata.get(recommendation.recommended_candidate_id or "")
    recommendation_quality_audit = audit_recommendation_quality(recommendation)
    optimizer_quality = recommendation.optimizer_quality
    write_json(optimizer_quality_path, optimizer_quality)
    payload = {
        "schema_version": "managed-recommendation/v1",
        "run_id": run_id,
        "status": recommendation.status if recommendation.recommended_candidate_id else "unavailable",
        "reason": None if recommendation.recommended_candidate_id else _unavailable_recommendation_reason(recommendation),
        "selected_evidence_key": selected.get("evidence_key") if selected else None,
        "selected_measurement_id": selected.get("measurement_id") if selected else None,
        "selected_source": selected.get("source") if selected else None,
        "selected_runtime_fingerprint": (
            selected.get("runtime_fingerprint") if selected else None
        ),
        "runtime_environment": runtime_environment,
        "client_saturation": client_saturation,
        "load_sufficiency": load_sufficiency,
        "recommendation_quality_audit": recommendation_quality_audit,
        "optimizer_quality": optimizer_quality,
        "recommendation": recommendation,
    }
    write_json(recommendation_path, payload)
    write_json(pareto_frontier_path, recommendation.pareto_frontier)
    _write_managed_pareto_csv(pareto_frontier_csv_path, recommendation.pareto_frontier)
    report_path.write_text(
        format_recommendation_report(recommendation, metadata={"run_id": run_id, "schema_version": "managed-recommendation/v1"}),
        encoding="utf-8",
    )
    selected_config = configs_by_id.get(recommendation.recommended_candidate_id or "")
    write_recommendation_summary_artifacts(
        txt_path=recommendation_summary_txt_path,
        json_path=recommendation_summary_json_path,
        recommendation=recommendation,
        selected_config=selected_config,
        selected_source=selected.get("source") if selected else None,
        reason=payload["reason"],
        runtime_environment=runtime_environment,
        selected_runtime_fingerprint=(
            selected.get("runtime_fingerprint") if selected else None
        ),
        artifacts={
            "managed_recommendation_json": str(recommendation_path),
            "managed_pareto_frontier_json": str(pareto_frontier_path),
            "managed_report_txt": str(report_path),
            "managed_run_json": str(run_dir / "managed_run.json"),
            "recommendation_summary_txt": str(recommendation_summary_txt_path),
            "recommendation_summary_json": str(recommendation_summary_json_path),
            "optimizer_quality_json": str(optimizer_quality_path),
        },
    )
    if recommendation.recommended_candidate_id is not None:
        _write_evidence_recommendation(
            evidence_store=evidence_store,
            evidence_warnings=evidence_warnings,
            run_id=run_id,
            goal=recommendation.goal,
            recommendation=recommendation,
            recommendation_payload=payload,
            selected=selected or {},
        )
    selected_score = recommendation.selected_score.final_score if recommendation.selected_score else None
    status = "success" if recommendation.recommended_candidate_id is not None else "unavailable"
    return _ManagedRecommendationArtifacts(
        status=status,
        reason=payload["reason"],
        selected_config_id=recommendation.recommended_candidate_id,
        selected_evidence_key=selected.get("evidence_key") if selected else None,
        selected_measurement_id=selected.get("measurement_id") if selected else None,
        recommendation_score=selected_score,
        recommendation_confidence=recommendation.confidence_level,
        pareto_candidate_count=len(recommendation.pareto_frontier),
        recommendation_artifact_path=str(recommendation_path),
        pareto_artifact_path=str(pareto_frontier_path),
        report_artifact_path=str(report_path),
        recommendation_summary_txt_path=str(recommendation_summary_txt_path),
        recommendation_summary_json_path=str(recommendation_summary_json_path),
        recommendation_quality_audit=recommendation_quality_audit,
        optimizer_quality=optimizer_quality,
    )


def _managed_recommendation_inputs(
    *,
    configs_by_id: dict[str, ServingConfig],
    rung_results: list[RungResult],
    model: str,
    backend: str,
    vllm_argument_capabilities: VLLMArgumentCapabilities | None,
    sglang_argument_capabilities: SGLangArgumentCapabilities | None,
) -> tuple[list[RecommendationInput], dict[str, dict[str, Any]]]:
    selected_results = _latest_recommendable_rung_results(rung_results)
    inputs: list[RecommendationInput] = []
    metadata: dict[str, dict[str, Any]] = {}
    for rank, result in enumerate(selected_results, start=1):
        config = configs_by_id.get(result.candidate_id)
        if config is None:
            continue
        if result.measured_or_evidence_source == "evidence":
            source = "managed_evidence_hit"
        elif result.measured_or_evidence_source == "resume":
            source = "managed_resume"
        else:
            source = "managed_measured"
        workload = serving_config_to_workload_config(config, trials=1, request_timeout_s=120.0, telemetry="none")
        candidate = _managed_serve_candidate(
            config,
            rank=rank,
            source=source,
            model=model,
            backend=backend,
            vllm_argument_capabilities=vllm_argument_capabilities,
            sglang_argument_capabilities=sglang_argument_capabilities,
        )
        benchmark_plan = EndpointBenchmarkPlan(
            candidate_id=config.id,
            base_url="managed",
            model=model,
            concurrency=workload.concurrency,
            num_requests=workload.num_requests,
            max_tokens=workload.max_new_tokens,
            expected_input_tokens=workload.input_length,
            expected_output_tokens=workload.output_length,
        )
        inputs.append(
            RecommendationInput(
                candidate_id=config.id,
                candidate_rank=rank,
                candidate_source=source,
                model=model,
                backend=backend,
                candidate=candidate,
                serve_plan=_managed_serve_plan(
                    config,
                    model=model,
                    vllm_argument_capabilities=vllm_argument_capabilities,
                    sglang_argument_capabilities=sglang_argument_capabilities,
                ),
                benchmark_plan=benchmark_plan,
                predicted_metrics={},
                measured_metrics=_managed_measured_metrics(result.metrics),
                telemetry_metrics=_managed_telemetry_metrics(result.metrics),
                comparison_metrics={},
                warnings=[],
            )
        )
        metadata[config.id] = {
            "evidence_key": result.evidence_key,
            "measurement_id": result.evidence_measurement_id,
            "source": source,
            "runtime_fingerprint": result.runtime_fingerprint,
            "runtime_environment": result.runtime_environment,
        }
    return inputs, metadata


def _latest_recommendable_rung_results(rung_results: list[RungResult]) -> list[RungResult]:
    by_candidate: dict[str, RungResult] = {}
    for result in rung_results:
        if result.status not in {"completed", "evidence_hit", "resumed"}:
            continue
        if result.measured_or_evidence_source not in {"measured", "evidence", "resume"}:
            continue
        previous = by_candidate.get(result.candidate_id)
        if previous is None or result.rung_index >= previous.rung_index:
            by_candidate[result.candidate_id] = result
    return sorted(by_candidate.values(), key=lambda result: (result.rung_index, result.candidate_id))


def _managed_serve_candidate(
    config: ServingConfig,
    *,
    rank: int,
    source: str,
    model: str,
    backend: str,
    vllm_argument_capabilities: VLLMArgumentCapabilities | None,
    sglang_argument_capabilities: SGLangArgumentCapabilities | None,
) -> ServeCandidate:
    extra = dict(config.extra or {})
    rendered_metadata = _rendered_launch_metadata(
        config,
        backend=backend,
        vllm_argument_capabilities=vllm_argument_capabilities,
        sglang_argument_capabilities=sglang_argument_capabilities,
    )
    return ServeCandidate(
        candidate_id=config.id,
        rank=rank,
        source=source,
        model=model,
        backend=backend,
        concurrency=_positive_int(extra.get("workload_concurrency"), default=max(1, config.max_batch_size)),
        batch_size=config.max_batch_size,
        global_batch_size=config.max_batch_size,
        tp=config.tensor_parallelism,
        raw={
            "candidate_source": source,
            "managed_candidate_source": extra.get("candidate_source"),
            "backend_defaults": extra.get("backend_defaults"),
            "dtype": config.dtype,
            "quantization": config.quantization,
            "max_context_tokens": config.max_context_tokens,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "block_size": config.block_size,
            "kv_cache_dtype": config.kv_cache_dtype,
            "enforce_eager": config.enforce_eager,
            "max_num_batched_tokens": config.max_num_batched_tokens,
            "enable_chunked_prefill": config.enable_chunked_prefill,
            "max_cudagraph_capture_size": config.max_cudagraph_capture_size,
            "enable_prefix_caching": config.enable_prefix_caching,
            "synthesis_rationale": extra.get("synthesis_rationale"),
            "synthesis_confidence": extra.get("synthesis_confidence"),
            "synthesis_constraints": extra.get("synthesis_constraints"),
            "synthesis_status": extra.get("synthesis_status"),
            "workload_profile": extra.get("workload_profile"),
            "slo_constraints": (
                extra.get("workload_profile", {}).get("slo_constraints")
                if isinstance(extra.get("workload_profile"), dict)
                else None
            ),
            **rendered_metadata,
            "extra": extra,
        },
    )


def _managed_serve_plan(
    config: ServingConfig,
    *,
    model: str,
    vllm_argument_capabilities: VLLMArgumentCapabilities | None,
    sglang_argument_capabilities: SGLangArgumentCapabilities | None,
) -> VllmServePlan | None:
    if config.backend == "vllm":
        rendered = render_vllm_launch(config, capabilities=vllm_argument_capabilities)
    elif config.backend == "sglang":
        rendered = render_sglang_launch(config, capabilities=sglang_argument_capabilities)
    else:
        return None
    canonical_config = rendered.canonical_config
    return VllmServePlan(
        candidate_id=config.id,
        model=model,
        host="managed",
        port=0,
        dtype=canonical_config.dtype,
        tensor_parallel_size=canonical_config.tensor_parallelism,
        pipeline_parallel_size=None,
        max_model_len=canonical_config.max_context_tokens,
        gpu_memory_utilization=canonical_config.gpu_memory_utilization,
        command=rendered.command,
        shell_command=" ".join(rendered.command),
        block_size=canonical_config.block_size,
        kv_cache_dtype=canonical_config.kv_cache_dtype,
        enforce_eager=canonical_config.enforce_eager,
        max_num_batched_tokens=canonical_config.max_num_batched_tokens,
        enable_chunked_prefill=canonical_config.enable_chunked_prefill,
        max_cudagraph_capture_size=canonical_config.max_cudagraph_capture_size,
        enable_prefix_caching=canonical_config.enable_prefix_caching,
    )


def _managed_measured_metrics(metrics: dict[str, Any]) -> dict[str, float | int | str | None]:
    return {
        "total_tokens_s": _optional_float(metrics.get("throughput_tokens_per_sec")),
        "request_rate_req_s": _optional_float(metrics.get("requests_per_sec")),
        "p50_latency_s": _seconds_from_ms(metrics.get("p50_latency_ms")),
        "p95_latency_s": _seconds_from_ms(metrics.get("p95_latency_ms")),
        "p99_latency_s": _seconds_from_ms(metrics.get("p99_latency_ms")),
        "ttft_ms": _optional_float(metrics.get("p95_ttft_ms")),
        "time_to_first_token_ms": _optional_float(metrics.get("p95_ttft_ms")),
        "avg_ttft_ms": _optional_float(metrics.get("avg_ttft_ms")),
        "p50_ttft_ms": _optional_float(metrics.get("p50_ttft_ms")),
        "p95_ttft_ms": _optional_float(metrics.get("p95_ttft_ms")),
        "tpot_ms": _optional_float(metrics.get("p95_tpot_ms")),
        "time_per_output_token_ms": _optional_float(metrics.get("p95_tpot_ms")),
        "avg_tpot_ms": _optional_float(metrics.get("avg_tpot_ms")),
        "p50_tpot_ms": _optional_float(metrics.get("p50_tpot_ms")),
        "p95_tpot_ms": _optional_float(metrics.get("p95_tpot_ms")),
        "ttft_sample_count": _optional_int(metrics.get("ttft_sample_count")) or 0,
        "tpot_sample_count": _optional_int(metrics.get("tpot_sample_count")) or 0,
        "timing_source": _optional_str(metrics.get("timing_source")),
        "total_requests": _optional_int(metrics.get("total_requests")),
        "successful_requests": _optional_int(metrics.get("successful_requests")),
        "failed_requests": _optional_int(metrics.get("failed_requests")) or 0,
        "configured_concurrency": _optional_int(metrics.get("configured_concurrency")),
        "effective_concurrency_limit": _optional_int(metrics.get("effective_concurrency_limit")),
        "concurrency_coverage": _optional_str(metrics.get("concurrency_coverage")),
        "client_cpu_time_s": _optional_float(metrics.get("client_cpu_time_s")),
        "client_cpu_utilization_percent": _optional_float(metrics.get("client_cpu_utilization_percent")),
        "client_queue_sample_count": _optional_int(metrics.get("client_queue_sample_count")) or 0,
        "avg_client_queue_s": _optional_float(metrics.get("avg_client_queue_s")),
        "p50_client_queue_s": _optional_float(metrics.get("p50_client_queue_s")),
        "p95_client_queue_s": _optional_float(metrics.get("p95_client_queue_s")),
        "p99_client_queue_s": _optional_float(metrics.get("p99_client_queue_s")),
        "max_client_queue_s": _optional_float(metrics.get("max_client_queue_s")),
        "client_saturation_signal": _optional_str(metrics.get("client_saturation_signal")),
        "client_issue_rate_req_s": _optional_float(metrics.get("client_issue_rate_req_s")),
        "avg_request_backlog": _optional_float(metrics.get("avg_request_backlog")),
        "max_request_backlog": _optional_float(metrics.get("max_request_backlog")),
        "avg_token_backlog": _optional_float(metrics.get("avg_token_backlog")),
        "max_token_backlog": _optional_float(metrics.get("max_token_backlog")),
        "load_saturation_signal": _optional_str(metrics.get("load_saturation_signal")),
        "load_sufficiency": metrics.get("load_sufficiency") if isinstance(metrics.get("load_sufficiency"), dict) else {},
        "average_gpu_util_percent": _optional_float(metrics.get("average_gpu_util_percent")),
        "max_gpu_util_percent": _optional_float(metrics.get("max_gpu_util_percent")),
    }


def _managed_telemetry_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    active_tokens_per_watt = _optional_float(metrics.get("active_tokens_per_watt"))
    gross_tokens_per_watt = _optional_float(metrics.get("tokens_per_watt"))
    active_tokens_per_joule = _optional_float(metrics.get("active_tokens_per_joule"))
    raw_tokens_per_joule = _optional_float(metrics.get("tokens_per_joule"))
    active_joules_per_token = _optional_float(metrics.get("active_joules_per_token"))
    gross_joules_per_token = _optional_float(metrics.get("joules_per_token"))
    active_joules_per_generated_token = _optional_float(metrics.get("active_joules_per_generated_token"))
    raw_joules_per_generated_token = _optional_float(metrics.get("joules_per_generated_token"))
    energy_accounting = _optional_str(metrics.get("energy_accounting"))
    if energy_accounting is None:
        energy_accounting = "idle_subtracted" if active_joules_per_token is not None else "raw"
    return {
        "average_power_watts": _optional_float(metrics.get("average_power_w")),
        "joules_per_token": (
            active_joules_per_token if active_joules_per_token is not None else gross_joules_per_token
        ),
        "joules_per_generated_token": (
            active_joules_per_generated_token
            if active_joules_per_generated_token is not None
            else raw_joules_per_generated_token
        ),
        "tokens_per_second_per_watt": (
            active_tokens_per_watt if active_tokens_per_watt is not None else gross_tokens_per_watt
        ),
        "tokens_per_joule": (
            active_tokens_per_joule if active_tokens_per_joule is not None else raw_tokens_per_joule
        ),
        "active_joules_per_token": active_joules_per_token,
        "gross_joules_per_token": gross_joules_per_token,
        "active_joules_per_generated_token": active_joules_per_generated_token,
        "raw_joules_per_generated_token": raw_joules_per_generated_token,
        "active_tokens_per_second_per_watt": active_tokens_per_watt,
        "gross_tokens_per_second_per_watt": gross_tokens_per_watt,
        "active_tokens_per_joule": active_tokens_per_joule,
        "raw_tokens_per_joule": raw_tokens_per_joule,
        "energy_accounting": energy_accounting,
        "warmup_power_sample_count": _optional_int(metrics.get("warmup_power_sample_count")) or 0,
        "measurement_power_sample_count": _optional_int(metrics.get("measurement_power_sample_count")) or 0,
        "average_gpu_util_percent": _optional_float(metrics.get("average_gpu_util_percent")),
        "max_gpu_util_percent": _optional_float(metrics.get("max_gpu_util_percent")),
        "temperature_rise_c": _optional_float(metrics.get("temperature_rise_c")),
        "temperature_slope_c_per_min": _optional_float(metrics.get("temperature_slope_c_per_min")),
        "thermal_stability_classification": _optional_str(metrics.get("thermal_stability_classification")),
        "telemetry_quality": _optional_str(metrics.get("telemetry_quality"))
        or ("limited" if metrics.get("average_power_w") is not None else "unavailable"),
    }


def _managed_recommendation_goal(goal: Goal) -> RecommendationGoal:
    if goal == Goal.PERFORMANCE:
        return RecommendationGoal.THROUGHPUT
    if goal == Goal.EFFICIENT:
        return RecommendationGoal.EFFICIENCY
    return RecommendationGoal.BALANCED


def _managed_recommendation_result(
    recommendation: RecommendationResult,
    *,
    backend: str,
    model: str,
    telemetry: str,
    artifacts: dict[str, str],
) -> RecommendationResult:
    return replace(
        recommendation,
        mode="managed",
        endpoint="managed",
        model=model,
        backend=backend,
        candidate_source="managed",
        telemetry_requested=telemetry,
        artifacts=artifacts,
        limitations=["Managed recommendations use measured results or exact fresh measured evidence only."],
    )


def _write_evidence_recommendation(
    *,
    evidence_store: EvidenceStore | None,
    evidence_warnings: list[str],
    run_id: str,
    goal: str,
    recommendation: RecommendationResult,
    recommendation_payload: dict[str, Any],
    selected: dict[str, Any],
) -> None:
    if evidence_store is None:
        return
    try:
        evidence_store.insert_recommendation(
            EvidenceRecommendationRecord(
                recommendation_id=f"rec-{hashlib.sha1(f'{run_id}|{recommendation.recommended_candidate_id}'.encode(), usedforsecurity=False).hexdigest()[:16]}",
                run_id=run_id,
                created_at=datetime.now(timezone.utc).isoformat(),
                goal=goal,
                evidence_key=selected.get("evidence_key"),
                selected_measurement_id=selected.get("measurement_id"),
                selected_config_id=recommendation.recommended_candidate_id,
                score=recommendation.selected_score.final_score if recommendation.selected_score else None,
                confidence=recommendation.confidence_level,
                recommendation_json=recommendation_payload,
            )
        )
    except Exception as exc:
        evidence_warnings.append(f"Evidence DB recommendation write failed: {exc.__class__.__name__}: {exc}")


def _write_managed_pareto_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "candidate_id",
        "source",
        "concurrency",
        "total_tokens_s",
        "p95_latency_s",
        "failed_requests",
        "client_cpu_utilization_percent",
        "p95_client_queue_s",
        "client_saturation_signal",
        "client_issue_rate_req_s",
        "max_request_backlog",
        "max_token_backlog",
        "load_saturation_signal",
        "max_gpu_util_percent",
        "average_power_watts",
        "joules_per_token",
        "joules_per_generated_token",
        "tokens_per_second_per_watt",
        "tokens_per_joule",
        "energy_accounting",
        "score",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _unavailable_recommendation_reason(recommendation: RecommendationResult) -> str:
    if recommendation.selection_reasons:
        return recommendation.selection_reasons[0]
    if recommendation.warnings:
        return recommendation.warnings[0]
    return "No measured or exact fresh evidence-hit candidates were available for recommendation."


def _seconds_from_ms(value: object) -> float | None:
    latency_ms = _optional_float(value)
    return latency_ms / 1000.0 if latency_ms is not None else None


def _summary_metrics(summary: object) -> dict[str, float | int | str | None]:
    row = to_dict(summary)
    if not isinstance(row, dict):
        row = {}
    measurement_quality = row.get("measurement_quality") if isinstance(row.get("measurement_quality"), dict) else {}
    throughput = _optional_float(row.get("total_tokens_s"))
    request_rate = _optional_float(row.get("request_rate_req_s"))
    stable_rates = _stable_rates_from_request_latencies(row) if row.get("measurement_duration_s") is None else None
    if stable_rates is not None:
        throughput = stable_rates["throughput_tokens_per_sec"]
        request_rate = stable_rates["requests_per_sec"]
    return {
        "throughput_tokens_per_sec": throughput,
        "requests_per_sec": request_rate,
        "p50_latency_ms": _latency_s_to_ms(row.get("p50_latency_s")),
        "p95_latency_ms": _latency_s_to_ms(row.get("p95_latency_s")),
        "p99_latency_ms": _latency_s_to_ms(row.get("p99_latency_s")),
        "average_power_w": _optional_float(row.get("average_power_watts")),
        "joules_per_token": (
            _optional_float(row.get("active_joules_per_token"))
            if row.get("active_joules_per_token") is not None
            else _optional_float(row.get("joules_per_token"))
        ),
        "active_joules_per_token": _optional_float(row.get("active_joules_per_token")),
        "joules_per_generated_token": (
            _optional_float(row.get("active_joules_per_generated_token"))
            if row.get("active_joules_per_generated_token") is not None
            else _optional_float(row.get("joules_per_generated_token"))
        ),
        "active_joules_per_generated_token": _optional_float(row.get("active_joules_per_generated_token")),
        "stability_classification": _optional_str(row.get("stability_classification")),
        "tokens_per_watt": _optional_float(row.get("tokens_per_second_per_watt")),
        "active_tokens_per_watt": _optional_float(row.get("active_tokens_per_second_per_watt")),
        "tokens_per_joule": (
            _optional_float(row.get("active_tokens_per_joule"))
            if row.get("active_tokens_per_joule") is not None
            else _optional_float(row.get("tokens_per_joule"))
        ),
        "active_tokens_per_joule": _optional_float(row.get("active_tokens_per_joule")),
        "energy_accounting": _optional_str(row.get("energy_accounting"))
        or _optional_str(measurement_quality.get("energy_accounting")),
        "warmup_power_sample_count": _optional_int(row.get("warmup_power_sample_count")) or 0,
        "measurement_power_sample_count": _optional_int(row.get("measurement_power_sample_count")) or 0,
        "warmup_average_power_watts": _optional_float(row.get("warmup_average_power_watts")),
        "measurement_average_power_watts": _optional_float(row.get("measurement_average_power_watts")),
        "telemetry_quality": _optional_str(row.get("telemetry_quality")) or "unavailable",
        "total_requests": (
            _optional_int(row.get("measured_requests"))
            if row.get("measured_requests") is not None
            else _optional_int(row.get("total_requests"))
        ),
        "successful_requests": (
            _optional_int(row.get("measured_successful_requests"))
            if row.get("measured_successful_requests") is not None
            else _optional_int(row.get("successful_requests"))
        ),
        "failed_requests": (
            _optional_int(row.get("measured_failed_requests"))
            if row.get("measured_failed_requests") is not None
            else _optional_int(row.get("failed_requests"))
        ),
        "configured_concurrency": _optional_int(measurement_quality.get("configured_concurrency")),
        "effective_concurrency_limit": _optional_int(measurement_quality.get("effective_concurrency_limit")),
        "concurrency_coverage": _optional_str(measurement_quality.get("concurrency_coverage")),
        "client_cpu_time_s": _optional_float(row.get("client_cpu_time_s")),
        "client_cpu_utilization_percent": _optional_float(row.get("client_cpu_utilization_percent")),
        "client_queue_sample_count": _optional_int(row.get("client_queue_sample_count")) or 0,
        "avg_client_queue_s": _optional_float(row.get("avg_client_queue_s")),
        "p50_client_queue_s": _optional_float(row.get("p50_client_queue_s")),
        "p95_client_queue_s": _optional_float(row.get("p95_client_queue_s")),
        "p99_client_queue_s": _optional_float(row.get("p99_client_queue_s")),
        "max_client_queue_s": _optional_float(row.get("max_client_queue_s")),
        "client_saturation_signal": _optional_str(measurement_quality.get("client_saturation_signal")),
        "client_issue_rate_req_s": _optional_float(row.get("client_issue_rate_req_s")),
        "avg_request_backlog": _optional_float(row.get("avg_request_backlog")),
        "max_request_backlog": _optional_float(row.get("max_request_backlog")),
        "avg_token_backlog": _optional_float(row.get("avg_token_backlog")),
        "max_token_backlog": _optional_float(row.get("max_token_backlog")),
        "load_saturation_signal": _optional_str(row.get("load_saturation_signal")),
        "load_sufficiency": row.get("load_sufficiency") if isinstance(row.get("load_sufficiency"), dict) else {},
        "average_gpu_util_percent": _optional_float(row.get("average_gpu_util_percent")),
        "max_gpu_util_percent": _optional_float(row.get("max_gpu_util_percent")),
    }


def _stable_rates_from_request_latencies(row: dict[str, Any]) -> dict[str, float] | None:
    wall_time_s = _optional_float(row.get("wall_time_s"))
    p95_latency_s = _optional_float(row.get("p95_latency_s"))
    avg_latency_s = _optional_float(row.get("avg_latency_s"))
    successful_requests = _optional_int(row.get("successful_requests")) or 0
    total_tokens = _optional_int(row.get("total_tokens")) or 0
    if (
        wall_time_s is None
        or p95_latency_s is None
        or avg_latency_s is None
        or wall_time_s >= p95_latency_s
        or successful_requests <= 0
        or avg_latency_s <= 0
    ):
        return None
    total_request_time_s = avg_latency_s * successful_requests
    return {
        "throughput_tokens_per_sec": total_tokens / total_request_time_s if total_request_time_s > 0 else 0.0,
        "requests_per_sec": successful_requests / total_request_time_s if total_request_time_s > 0 else 0.0,
    }


def _measurement_metrics(measurement: dict[str, Any]) -> dict[str, float | int | str | None]:
    raw_json = measurement.get("raw_json") if isinstance(measurement.get("raw_json"), dict) else {}
    raw_summary = raw_json.get("summary") if isinstance(raw_json.get("summary"), dict) else {}
    measurement_quality = raw_summary.get("measurement_quality") if isinstance(raw_summary.get("measurement_quality"), dict) else {}
    throughput = _optional_float(measurement.get("throughput_tokens_per_sec"))
    request_rate = _optional_float(measurement.get("requests_per_sec"))
    stable_rates = _stable_rates_from_request_latencies(raw_summary) if raw_summary.get("measurement_duration_s") is None else None
    if stable_rates is not None:
        throughput = stable_rates["throughput_tokens_per_sec"]
        request_rate = stable_rates["requests_per_sec"]
    return {
        "throughput_tokens_per_sec": throughput,
        "requests_per_sec": request_rate,
        "p50_latency_ms": _optional_float(measurement.get("p50_latency_ms")),
        "p95_latency_ms": _optional_float(measurement.get("p95_latency_ms")),
        "p99_latency_ms": _optional_float(measurement.get("p99_latency_ms")),
        "avg_ttft_ms": _optional_float(raw_summary.get("avg_ttft_ms")),
        "p50_ttft_ms": _optional_float(raw_summary.get("p50_ttft_ms")),
        "p95_ttft_ms": _optional_float(raw_summary.get("p95_ttft_ms")),
        "avg_tpot_ms": _optional_float(raw_summary.get("avg_tpot_ms")),
        "p50_tpot_ms": _optional_float(raw_summary.get("p50_tpot_ms")),
        "p95_tpot_ms": _optional_float(raw_summary.get("p95_tpot_ms")),
        "ttft_sample_count": _optional_int(raw_summary.get("ttft_sample_count")),
        "tpot_sample_count": _optional_int(raw_summary.get("tpot_sample_count")),
        "timing_source": _optional_str(raw_summary.get("timing_source")),
        "average_power_w": _optional_float(measurement.get("average_power_w")),
        "joules_per_token": _optional_float(measurement.get("joules_per_token")),
        "active_joules_per_token": _optional_float(raw_summary.get("active_joules_per_token")),
        "joules_per_generated_token": (
            _optional_float(raw_summary.get("active_joules_per_generated_token"))
            if raw_summary.get("active_joules_per_generated_token") is not None
            else _optional_float(raw_summary.get("joules_per_generated_token"))
        ),
        "active_joules_per_generated_token": _optional_float(raw_summary.get("active_joules_per_generated_token")),
        "stability_classification": _optional_str(raw_summary.get("stability_classification")),
        "temperature_rise_c": _optional_float(raw_summary.get("temperature_rise_c")),
        "temperature_slope_c_per_min": _optional_float(raw_summary.get("temperature_slope_c_per_min")),
        "thermal_stability_classification": _optional_str(raw_summary.get("thermal_stability_classification")),
        "tokens_per_watt": _optional_float(measurement.get("tokens_per_watt")),
        "active_tokens_per_watt": _optional_float(raw_summary.get("active_tokens_per_second_per_watt")),
        "tokens_per_joule": (
            _optional_float(raw_summary.get("active_tokens_per_joule"))
            if raw_summary.get("active_tokens_per_joule") is not None
            else _optional_float(raw_summary.get("tokens_per_joule"))
        ),
        "active_tokens_per_joule": _optional_float(raw_summary.get("active_tokens_per_joule")),
        "energy_accounting": _optional_str(raw_summary.get("energy_accounting"))
        or _optional_str(measurement_quality.get("energy_accounting")),
        "warmup_power_sample_count": _optional_int(raw_summary.get("warmup_power_sample_count")) or 0,
        "measurement_power_sample_count": _optional_int(raw_summary.get("measurement_power_sample_count")) or 0,
        "warmup_average_power_watts": _optional_float(raw_summary.get("warmup_average_power_watts")),
        "measurement_average_power_watts": _optional_float(raw_summary.get("measurement_average_power_watts")),
        "telemetry_quality": (
            _optional_str(raw_summary.get("telemetry_quality"))
            or _optional_str(measurement.get("confidence"))
            or "unavailable"
        ),
        "total_requests": (
            _optional_int(raw_summary.get("measured_requests"))
            if raw_summary.get("measured_requests") is not None
            else _optional_int(raw_summary.get("total_requests"))
        ),
        "successful_requests": (
            _optional_int(raw_summary.get("measured_successful_requests"))
            if raw_summary.get("measured_successful_requests") is not None
            else _optional_int(raw_summary.get("successful_requests"))
        ),
        "failed_requests": (
            _optional_int(raw_summary.get("measured_failed_requests"))
            if raw_summary.get("measured_failed_requests") is not None
            else _optional_int(raw_summary.get("failed_requests"))
        ),
        "configured_concurrency": _optional_int(measurement_quality.get("configured_concurrency")),
        "effective_concurrency_limit": _optional_int(measurement_quality.get("effective_concurrency_limit")),
        "concurrency_coverage": _optional_str(measurement_quality.get("concurrency_coverage")),
        "client_cpu_time_s": _optional_float(raw_summary.get("client_cpu_time_s")),
        "client_cpu_utilization_percent": _optional_float(raw_summary.get("client_cpu_utilization_percent")),
        "client_queue_sample_count": _optional_int(raw_summary.get("client_queue_sample_count")) or 0,
        "avg_client_queue_s": _optional_float(raw_summary.get("avg_client_queue_s")),
        "p50_client_queue_s": _optional_float(raw_summary.get("p50_client_queue_s")),
        "p95_client_queue_s": _optional_float(raw_summary.get("p95_client_queue_s")),
        "p99_client_queue_s": _optional_float(raw_summary.get("p99_client_queue_s")),
        "max_client_queue_s": _optional_float(raw_summary.get("max_client_queue_s")),
        "client_saturation_signal": _optional_str(measurement_quality.get("client_saturation_signal")),
        "client_issue_rate_req_s": _optional_float(raw_summary.get("client_issue_rate_req_s")),
        "avg_request_backlog": _optional_float(raw_summary.get("avg_request_backlog")),
        "max_request_backlog": _optional_float(raw_summary.get("max_request_backlog")),
        "avg_token_backlog": _optional_float(raw_summary.get("avg_token_backlog")),
        "max_token_backlog": _optional_float(raw_summary.get("max_token_backlog")),
        "load_saturation_signal": _optional_str(raw_summary.get("load_saturation_signal")),
        "load_sufficiency": raw_summary.get("load_sufficiency") if isinstance(raw_summary.get("load_sufficiency"), dict) else {},
        "average_gpu_util_percent": _optional_float(raw_summary.get("average_gpu_util_percent")),
        "max_gpu_util_percent": _optional_float(raw_summary.get("max_gpu_util_percent")),
    }


def _workload_result_fields(workload: WorkloadConfig, *, source: str) -> dict[str, object]:
    return {
        "rung": workload.rung,
        "promotion_status": workload.promotion_status,
        "promotion_reason": workload.promotion_reason,
        "measured_or_evidence_source": source,
    }


def _group_rung(group: LaunchGroup) -> str | None:
    if not group.workload_configs:
        return None
    return group.workload_configs[0].rung


def _rung_measurement_count(results: list[RungResult], rung: str) -> int:
    return sum(1 for result in results if result.rung == rung and result.measured_or_evidence_source == "measured")


def _client_saturation_summary(rung_results: list[RungResult]) -> dict[str, Any]:
    selected_results = _latest_recommendable_rung_results(rung_results)
    rows = []
    for result in selected_results:
        throughput = _optional_float(result.metrics.get("throughput_tokens_per_sec"))
        if throughput is None or throughput <= 0:
            continue
        rows.append(
            {
                "candidate_id": result.candidate_id,
                "throughput_tokens_per_sec": throughput,
                "client_cpu_utilization_percent": _optional_float(result.metrics.get("client_cpu_utilization_percent")),
                "p95_client_queue_s": _optional_float(result.metrics.get("p95_client_queue_s")),
                "client_saturation_signal": _optional_str(result.metrics.get("client_saturation_signal")),
            }
        )
    throughputs = [float(row["throughput_tokens_per_sec"]) for row in rows]
    cpu_values = [
        value
        for row in rows
        if (value := _optional_float(row.get("client_cpu_utilization_percent"))) is not None
    ]
    queue_values = [
        value
        for row in rows
        if (value := _optional_float(row.get("p95_client_queue_s"))) is not None
    ]
    mean_throughput = sum(throughputs) / len(throughputs) if throughputs else None
    throughput_cv = _coefficient_of_variation(throughputs)
    flat_throughput = (
        len(throughputs) >= CLIENT_LIMITED_MIN_CONFIGS
        and throughput_cv is not None
        and throughput_cv <= CLIENT_LIMITED_FLAT_THROUGHPUT_CV
    )
    max_cpu = max(cpu_values, default=None)
    client_saturated = max_cpu is not None and max_cpu >= CLIENT_LIMITED_CPU_THRESHOLD_PERCENT
    if flat_throughput and client_saturated:
        classification = "client_limited"
    elif len(throughputs) < CLIENT_LIMITED_MIN_CONFIGS:
        classification = "insufficient_evidence"
    elif not cpu_values:
        classification = "insufficient_client_cpu_evidence"
    else:
        classification = "not_client_limited"
    return {
        "schema_version": "client-saturation/v1",
        "classification": classification,
        "candidate_count": len(throughputs),
        "min_required_candidate_count": CLIENT_LIMITED_MIN_CONFIGS,
        "mean_throughput_tokens_per_sec": _round_or_none(mean_throughput),
        "throughput_coefficient_of_variation": _round_or_none(throughput_cv),
        "flat_throughput": flat_throughput,
        "flat_throughput_cv_threshold": CLIENT_LIMITED_FLAT_THROUGHPUT_CV,
        "client_saturated": client_saturated,
        "client_cpu_saturation_threshold_percent": CLIENT_LIMITED_CPU_THRESHOLD_PERCENT,
        "max_client_cpu_utilization_percent": _round_or_none(max_cpu),
        "max_p95_client_queue_s": _round_or_none(max(queue_values, default=None)),
        "rows": rows,
        "reason": _client_saturation_reason(
            classification,
            candidate_count=len(throughputs),
            flat_throughput=flat_throughput,
            client_saturated=client_saturated,
            has_cpu=bool(cpu_values),
        ),
    }


def _client_saturation_reason(
    classification: str,
    *,
    candidate_count: int,
    flat_throughput: bool,
    client_saturated: bool,
    has_cpu: bool,
) -> str:
    if classification == "client_limited":
        return "Throughput was flat across measured configurations while client CPU was saturated."
    if candidate_count < CLIENT_LIMITED_MIN_CONFIGS:
        return "Fewer than three measured configurations were available, so flat throughput across many configurations was not established."
    if not has_cpu:
        return "Client CPU utilization was unavailable, so client saturation could not be established."
    if not flat_throughput:
        return "Throughput was not flat across the measured configurations."
    if not client_saturated:
        return "Client CPU utilization did not reach the saturation threshold."
    return "Client saturation was not established."


def _client_saturation_notes(client_saturation: dict[str, Any]) -> list[str]:
    if client_saturation.get("classification") != "client_limited":
        return []
    return ["Experiment marked client_limited because throughput was flat while client CPU was saturated."]


def _load_sufficiency_summary(
    rung_results: list[RungResult],
    *,
    goal: Goal,
    client_saturation: dict[str, Any],
) -> dict[str, Any]:
    selected_results = _latest_recommendable_rung_results(rung_results)
    rows = [_load_sufficiency_row(result) for result in selected_results]
    rows = [row for row in rows if row is not None]
    throughputs = [float(row["throughput_tokens_per_sec"]) for row in rows]
    concurrency_values = sorted(
        {
            value
            for row in rows
            if (value := _optional_int(row.get("configured_concurrency"))) is not None and value > 0
        }
    )
    mean_throughput = sum(throughputs) / len(throughputs) if throughputs else None
    throughput_cv = _coefficient_of_variation(throughputs)
    flat_throughput = (
        len(throughputs) >= LOAD_SUFFICIENCY_MIN_CONCURRENCY_LEVELS
        and throughput_cv is not None
        and throughput_cv <= LOAD_SUFFICIENCY_FLAT_THROUGHPUT_CV
    )
    max_gpu = _max_optional(
        [
            value
            for row in rows
            for value in (
                _optional_float(row.get("average_gpu_util_percent")),
                _optional_float(row.get("max_gpu_util_percent")),
            )
            if value is not None
        ]
    )
    gpu_saturated = max_gpu is not None and max_gpu >= LOAD_SUFFICIENCY_GPU_THRESHOLD_PERCENT
    pressure_growth = _load_pressure_growth(rows)
    pressure_growth_ratio = _max_optional(list(pressure_growth.values()))
    pressure_applied = (
        pressure_growth_ratio is not None
        and pressure_growth_ratio >= LOAD_SUFFICIENCY_PRESSURE_GROWTH_RATIO
    )
    client_limited = client_saturation.get("classification") == "client_limited"
    if goal != Goal.PERFORMANCE:
        classification = "not_evaluated_non_throughput_goal"
    elif not rows:
        classification = "insufficient_evidence"
    elif client_limited:
        classification = "client_limited"
    elif gpu_saturated:
        classification = "load_sufficient_gpu_saturated"
    elif len(concurrency_values) < LOAD_SUFFICIENCY_MIN_CONCURRENCY_LEVELS:
        classification = "insufficient_sweep"
    elif flat_throughput and pressure_applied:
        classification = "load_sufficient_throughput_plateau"
    elif flat_throughput:
        classification = "load_insufficient_pressure"
    else:
        classification = "not_saturated"
    zero_change_check = {
        "zero_change_detected": flat_throughput,
        "load_generator_applied_pressure": pressure_applied,
        "pressure_growth_ratio": _round_or_none(pressure_growth_ratio),
        "pressure_growth_threshold": LOAD_SUFFICIENCY_PRESSURE_GROWTH_RATIO,
    }
    return {
        "schema_version": "load-sufficiency/v1",
        "classification": classification,
        "goal": goal.value,
        "candidate_count": len(rows),
        "concurrency_levels": concurrency_values,
        "concurrency_level_count": len(concurrency_values),
        "min_required_concurrency_levels": LOAD_SUFFICIENCY_MIN_CONCURRENCY_LEVELS,
        "mean_throughput_tokens_per_sec": _round_or_none(mean_throughput),
        "throughput_coefficient_of_variation": _round_or_none(throughput_cv),
        "flat_throughput": flat_throughput,
        "flat_throughput_cv_threshold": LOAD_SUFFICIENCY_FLAT_THROUGHPUT_CV,
        "gpu_saturated": gpu_saturated,
        "gpu_saturation_threshold_percent": LOAD_SUFFICIENCY_GPU_THRESHOLD_PERCENT,
        "max_gpu_util_percent": _round_or_none(max_gpu),
        "pressure_applied": pressure_applied,
        "pressure_growth_ratio": _round_or_none(pressure_growth_ratio),
        "pressure_growth_threshold": LOAD_SUFFICIENCY_PRESSURE_GROWTH_RATIO,
        "pressure_growth": {key: _round_or_none(value) for key, value in pressure_growth.items()},
        "zero_change_pressure_check": zero_change_check,
        "client_saturation_classification": client_saturation.get("classification"),
        "rows": rows,
        "reason": _load_sufficiency_reason(
            classification,
            goal=goal,
            row_count=len(rows),
            concurrency_level_count=len(concurrency_values),
            flat_throughput=flat_throughput,
            pressure_applied=pressure_applied,
            gpu_saturated=gpu_saturated,
            client_limited=client_limited,
        ),
    }


def _load_sufficiency_row(result: RungResult) -> dict[str, object] | None:
    throughput = _optional_float(result.metrics.get("throughput_tokens_per_sec"))
    if throughput is None or throughput <= 0:
        return None
    return {
        "candidate_id": result.candidate_id,
        "workload_id": result.workload_id,
        "rung": result.rung,
        "rung_index": result.rung_index,
        "measured_or_evidence_source": result.measured_or_evidence_source,
        "throughput_tokens_per_sec": throughput,
        "request_rate_req_s": _optional_float(result.metrics.get("requests_per_sec")),
        "configured_concurrency": _optional_int(result.metrics.get("configured_concurrency")),
        "effective_concurrency_limit": _optional_int(result.metrics.get("effective_concurrency_limit")),
        "client_issue_rate_req_s": _optional_float(result.metrics.get("client_issue_rate_req_s")),
        "avg_request_backlog": _optional_float(result.metrics.get("avg_request_backlog")),
        "max_request_backlog": _optional_float(result.metrics.get("max_request_backlog")),
        "avg_token_backlog": _optional_float(result.metrics.get("avg_token_backlog")),
        "max_token_backlog": _optional_float(result.metrics.get("max_token_backlog")),
        "average_gpu_util_percent": _optional_float(result.metrics.get("average_gpu_util_percent")),
        "max_gpu_util_percent": _optional_float(result.metrics.get("max_gpu_util_percent")),
        "load_saturation_signal": _optional_str(result.metrics.get("load_saturation_signal")),
        "client_saturation_signal": _optional_str(result.metrics.get("client_saturation_signal")),
    }


def _load_pressure_growth(rows: list[dict[str, object]]) -> dict[str, float]:
    growth = {}
    for key in (
        "configured_concurrency",
        "client_issue_rate_req_s",
        "max_request_backlog",
        "max_token_backlog",
    ):
        ratio = _growth_ratio(
            [
                value
                for row in rows
                if (value := _optional_float(row.get(key))) is not None
            ]
        )
        if ratio is not None:
            growth[key] = ratio
    return growth


def _growth_ratio(values: list[float]) -> float | None:
    positive = [value for value in values if value > 0]
    if len(positive) < 2:
        return None
    minimum = min(positive)
    maximum = max(positive)
    if minimum <= 0:
        return None
    return maximum / minimum


def _max_optional(values: Iterable[float | None]) -> float | None:
    concrete = [value for value in values if value is not None]
    return max(concrete, default=None)


def _load_sufficiency_reason(
    classification: str,
    *,
    goal: Goal,
    row_count: int,
    concurrency_level_count: int,
    flat_throughput: bool,
    pressure_applied: bool,
    gpu_saturated: bool,
    client_limited: bool,
) -> str:
    if classification == "not_evaluated_non_throughput_goal":
        return f"Load sufficiency is only evaluated for throughput mode, and this run used goal {goal.value}."
    if row_count == 0:
        return "No successful throughput measurements were available for load sufficiency classification."
    if client_limited:
        return "Client saturation was detected, so the run cannot prove server load sufficiency."
    if gpu_saturated:
        return "GPU utilization reached the server saturation threshold."
    if concurrency_level_count < LOAD_SUFFICIENCY_MIN_CONCURRENCY_LEVELS:
        return "Fewer than three concurrency levels were measured and no saturation signal was observed."
    if flat_throughput and pressure_applied:
        return "Throughput was flat while the load generator applied increasing pressure."
    if flat_throughput:
        return "Throughput was flat, but the load generator did not show enough pressure growth."
    return "Throughput did not plateau and no GPU saturation signal was observed."


def _load_sufficiency_notes(load_sufficiency: dict[str, Any]) -> list[str]:
    classification = load_sufficiency.get("classification")
    if classification == "load_insufficient_pressure":
        return ["Load sufficiency was not established because flat throughput lacked pressure growth evidence."]
    if classification == "insufficient_sweep":
        return ["Load sufficiency was not established because the throughput sweep covered fewer than three concurrency levels."]
    if classification == "client_limited":
        return ["Load sufficiency was not established because the experiment was client_limited."]
    return []


def _coefficient_of_variation(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    if mean == 0:
        return None
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return variance**0.5 / mean


def _unique_result_count(results: list[ManagedCandidateResult], *, statuses: set[str]) -> int:
    return len({result.config_id for result in results if result.status in statuses})


def _latency_s_to_ms(value: object) -> float | None:
    latency = _optional_float(value)
    return latency * 1000.0 if latency is not None else None


def _adapter_for_backend(backend: str) -> ManagedBackendAdapter:
    return create_managed_backend_adapter(backend)


def group_candidates_by_launch_config(
    candidates: list[ServingConfig],
    *,
    trials: int,
    request_timeout_s: float,
    telemetry: str,
) -> list[LaunchGroup]:
    groups: dict[str, dict[str, object]] = {}
    order: list[str] = []
    for config in candidates:
        launch_config = serving_config_to_launch_config(config)
        launch_hash = launch_config_hash(launch_config)
        if launch_hash not in groups:
            groups[launch_hash] = {
                "launch_config": launch_config,
                "workloads": [],
                "config_ids": [],
            }
            order.append(launch_hash)
        workload = serving_config_to_workload_config(
            config,
            trials=trials,
            request_timeout_s=request_timeout_s,
            telemetry=telemetry,
        )
        groups[launch_hash]["workloads"].append(workload)
        groups[launch_hash]["config_ids"].append(config.id)

    launch_groups: list[LaunchGroup] = []
    for launch_hash in order:
        row = groups[launch_hash]
        group_id = f"launch-{launch_hash[:12]}"
        launch_groups.append(
            LaunchGroup(
                group_id=group_id,
                launch_config=row["launch_config"],
                workload_configs=list(row["workloads"]),
                original_config_ids=list(row["config_ids"]),
                launch_config_hash=launch_hash,
            )
        )
    return launch_groups


def serving_config_to_launch_config(config: ServingConfig) -> LaunchConfig:
    extra = dict(config.extra or {})
    launch_extra = {
        key: value
        for key, value in extra.items()
        if key not in WORKLOAD_EXTRA_KEYS
    }
    return LaunchConfig(
        backend=config.backend,
        model=config.model_id,
        dtype=config.dtype,
        quantization=config.quantization,
        max_model_len=config.max_context_tokens,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_num_seqs=config.max_batch_size,
        block_size=config.block_size,
        max_num_batched_tokens=config.max_num_batched_tokens or _optional_int(extra.get("max_num_batched_tokens")),
        tensor_parallel_size=config.tensor_parallelism,
        kv_cache_dtype=config.kv_cache_dtype or _optional_str(extra.get("kv_cache_dtype")),
        enforce_eager=config.enforce_eager,
        enable_chunked_prefill=config.enable_chunked_prefill,
        max_cudagraph_capture_size=config.max_cudagraph_capture_size,
        enable_prefix_caching=config.enable_prefix_caching,
        kv_cache_policy=config.kv_cache_policy,
        scheduler=config.scheduler,
        power_limit_watts=config.power_limit_watts,
        extra=launch_extra,
    )


def serving_config_to_workload_config(
    config: ServingConfig,
    *,
    trials: int,
    request_timeout_s: float,
    telemetry: str,
) -> WorkloadConfig:
    extra = dict(config.extra or {})
    profile = _workload_profile_payload(extra.get("workload_profile"))
    profile_concurrency = _positive_int(profile.get("concurrency"), default=0)
    candidate_concurrency = _positive_int(extra.get("workload_concurrency"), default=0)
    concurrency = max(profile_concurrency, candidate_concurrency)
    if concurrency <= 0:
        concurrency = max(1, config.max_batch_size)
    max_new_tokens = _positive_int(
        profile.get("max_new_tokens") or profile.get("output_tokens") or extra.get("max_new_tokens") or extra.get("output_length"),
        default=128,
    )
    warmup_requests = _positive_int(extra.get("warmup_requests"), default=0)
    requested_num_requests = _positive_int(extra.get("num_requests") or profile.get("num_requests"), default=max(128, 2 * concurrency))
    minimum_num_requests = concurrency + warmup_requests
    num_requests = max(requested_num_requests, minimum_num_requests)
    workload_extra = dict(extra.get("workload_extra") or {}) if isinstance(extra.get("workload_extra"), dict) else {}
    if requested_num_requests < minimum_num_requests:
        workload_extra["requested_num_requests"] = requested_num_requests
        workload_extra["num_requests_adjusted_reason"] = "raised_to_match_concurrency"
    if profile and profile.get("profile_name") != "default":
        workload_extra["workload_profile"] = profile
    elif profile and (profile.get("token_distribution") or profile.get("slo_constraints")):
        workload_extra["workload_profile"] = profile
    return WorkloadConfig(
        workload_id=str(extra.get("workload_id") or config.id),
        candidate_id=config.id,
        concurrency=concurrency,
        num_requests=num_requests,
        max_new_tokens=max_new_tokens,
        prompt=str(extra.get("workload_prompt") or DEFAULT_ENDPOINT_PROMPT),
        timeout_s=_positive_float(extra.get("timeout_s"), default=request_timeout_s),
        trials=_positive_int(extra.get("trials"), default=trials),
        request_rate=_optional_float(extra.get("request_rate")),
        input_length=_optional_int(extra.get("input_length")) or _optional_int(profile.get("input_tokens")),
        output_length=_optional_int(extra.get("output_length")) or _optional_int(profile.get("output_tokens")) or max_new_tokens,
        num_prompts=_optional_int(extra.get("num_prompts")) or num_requests,
        dataset=_optional_str(extra.get("dataset")) or _optional_str(profile.get("dataset")),
        warmup_duration_s=_optional_float(extra.get("warmup_duration_s")),
        benchmark_duration_s=_optional_float(extra.get("benchmark_duration_s")),
        warmup_requests=warmup_requests,
        idle_baseline_duration_s=_positive_float(extra.get("idle_baseline_duration_s"), default=0.0),
        idle_power_watts=_optional_float(extra.get("idle_power_watts")),
        soak_duration_s=_optional_float(extra.get("soak_duration_s")),
        stream=_optional_bool(extra.get("stream"), default=False),
        telemetry=telemetry,
        prior_source=_optional_str(extra.get("prior_source")),
        prior_confidence=_optional_float(extra.get("prior_confidence")),
        prior_notes=[str(note) for note in extra.get("prior_notes", [])] if isinstance(extra.get("prior_notes"), list) else [],
        rung=_optional_str(extra.get("rung")),
        rung_index=_optional_int(extra.get("rung_index")),
        promotion_status=_optional_str(extra.get("promotion_status")),
        promotion_reason=_optional_str(extra.get("promotion_reason")),
        measured_or_evidence_source=_optional_str(extra.get("measured_or_evidence_source")),
        extra=workload_extra,
    )


def _generate_managed_candidates(
    *,
    backend: str,
    model: str,
    goal: Goal,
    limit: int,
    hardware,
    model_metadata: ModelCapabilityMetadata | None = None,
) -> list[ServingConfig]:
    return _generate_managed_candidate_generation(
        backend=backend,
        model=model,
        goal=goal,
        limit=limit,
        hardware=hardware,
        model_metadata=model_metadata,
    ).candidates


def _generate_managed_candidate_generation(
    *,
    backend: str,
    model: str,
    goal: Goal,
    limit: int,
    hardware,
    model_metadata: ModelCapabilityMetadata | None,
    backend_metadata: dict[str, object] | None = None,
    vllm_argument_capabilities: VLLMArgumentCapabilities | None = None,
    sglang_argument_capabilities: SGLangArgumentCapabilities | None = None,
    workload_profile: WorkloadProfile | None = None,
) -> ManagedCandidateGenerationResult:
    return generate_managed_candidates_from_capabilities(
        CapabilityContext(
            backend=backend,
            model=model,
            goal=goal,
            hardware=hardware,
            model_metadata=model_metadata,
            backend_metadata=backend_metadata or {},
            vllm_argument_capabilities=vllm_argument_capabilities,
            sglang_argument_capabilities=sglang_argument_capabilities,
            workload_profile=workload_profile,
            managed_mode=True,
        ),
        limit=limit,
    )


def _provided_candidate_generation(candidates: list[ServingConfig] | None = None) -> ManagedCandidateGenerationResult:
    candidates = candidates or []
    source_counts: dict[str, int] = {}
    for config in candidates:
        source = str((config.extra or {}).get("candidate_source") or "provided")
        source_counts[source] = source_counts.get(source, 0) + 1
    return ManagedCandidateGenerationResult(
        candidates=candidates,
        candidate_source_counts=source_counts,
        safe_baseline_added=any(
            config.quantization == "none"
            and (config.extra or {}).get("baseline") is True
            and (config.extra or {}).get("model_native") is True
            for config in candidates
        ),
    )


def _canonicalize_valid_candidates(
    candidates: list[ServingConfig],
    *,
    backend: str,
    vllm_argument_capabilities: VLLMArgumentCapabilities | None,
    sglang_argument_capabilities: SGLangArgumentCapabilities | None,
    runtime_environment: dict[str, object],
) -> tuple[list[ServingConfig], list[ValidationRejection], list[dict[str, object]]]:
    canonical_candidates: list[ServingConfig] = []
    rejections: list[ValidationRejection] = []
    rows: list[dict[str, object]] = []
    for config in candidates:
        if backend == "vllm":
            rendered = render_vllm_launch(config, capabilities=vllm_argument_capabilities)
        elif backend == "sglang":
            rendered = render_sglang_launch(config, capabilities=sglang_argument_capabilities)
        else:
            canonical_candidates.append(config)
            continue
        if rendered.unsupported_fields:
            reason = f"Candidate has {backend} fields that cannot be rendered: " + ", ".join(sorted(rendered.unsupported_fields))
            rejections.append((config, CandidateValidationResult(config_id=config.id, valid=False, reason=reason)))
            continue
        canonical = rendered.canonical_config
        canonical_candidates.append(canonical)
        rows.append(
            _rendered_launch_config_row(
                config,
                rendered,
                runtime_environment=runtime_environment,
            )
        )
    return canonical_candidates, rejections, rows


def _rendered_launch_config_row(
    config: ServingConfig,
    rendered,
    *,
    runtime_environment: dict[str, object],
) -> dict[str, object]:
    logical_launch_config = serving_config_to_launch_config(config)
    canonical_launch_config = serving_config_to_launch_config(rendered.canonical_config)
    return {
        "schema_version": "rendered-launch-config/v1",
        "logical_config_id": config.id,
        "canonical_config_id": rendered.canonical_config.id,
        "logical_launch_config_hash": launch_config_hash(logical_launch_config),
        "canonical_launch_config_hash": launch_config_hash(canonical_launch_config),
        "command": rendered.command,
        "rendered_launch_command_hash": stable_payload_hash(
            {"command": rendered.command}
        ),
        "runtime_environment": runtime_environment,
        "canonical_config": rendered.canonical_config,
        "rendered_fields": rendered.rendered_fields,
        "omitted_fields": rendered.omitted_fields,
        "unsupported_fields": rendered.unsupported_fields,
        "unavailable_fields": getattr(rendered, "unavailable_fields", {}),
        "flag_aliases": rendered.flag_aliases,
        "capabilities_help_hash": rendered.capabilities_help_hash,
    }


def _launch_provenance_from_spec(spec: ServerLaunchSpec) -> dict[str, object]:
    metadata = spec.metadata or {}
    rendered = metadata.get("rendered_launch")
    rendered_payload = rendered if isinstance(rendered, dict) else {}
    command = [str(item) for item in spec.command]
    return {
        "backend_name": spec.backend,
        "backend_version": _optional_str(metadata.get("version")),
        "backend_launch_command": command,
        "backend_launch_command_hash": stable_payload_hash({"command": command}),
        "backend_effective_values": _dict_payload(rendered_payload.get("rendered_fields")),
        "backend_applied_configuration": _dict_payload(rendered_payload.get("canonical_config")),
        "backend_omitted_values": _dict_payload(rendered_payload.get("omitted_fields")),
        "backend_unsupported_values": _dict_payload(rendered_payload.get("unsupported_fields")),
        "backend_unavailable_values": _dict_payload(rendered_payload.get("unavailable_fields")),
        "backend_flag_aliases": _dict_payload(rendered_payload.get("flag_aliases")),
        "backend_capabilities_help_hash": _optional_str(rendered_payload.get("capabilities_help_hash")),
    }


def _dict_payload(value: object) -> dict[str, object]:
    payload = to_dict(value)
    return dict(payload) if isinstance(payload, dict) else {}


def _rendered_launch_metadata(
    config: ServingConfig,
    *,
    backend: str,
    vllm_argument_capabilities: VLLMArgumentCapabilities | None,
    sglang_argument_capabilities: SGLangArgumentCapabilities | None,
) -> dict[str, object]:
    if backend == "vllm":
        rendered = render_vllm_launch(config, capabilities=vllm_argument_capabilities)
    elif backend == "sglang":
        rendered = render_sglang_launch(config, capabilities=sglang_argument_capabilities)
    else:
        return {}
    return {
        "rendered_engine_fields": rendered.rendered_fields,
        "omitted_engine_fields": rendered.omitted_fields,
        "unsupported_engine_fields": rendered.unsupported_fields,
        "unavailable_engine_fields": getattr(rendered, "unavailable_fields", {}),
        "flag_aliases": rendered.flag_aliases,
        "capabilities_help_hash": rendered.capabilities_help_hash,
    }


def _validate_managed_candidate_pool(
    candidates: list[ServingConfig],
    *,
    backend: str,
    model_metadata: ModelCapabilityMetadata,
    limit: int,
    backfill_valid_candidates: bool,
    vllm_argument_capabilities: VLLMArgumentCapabilities | None = None,
    sglang_argument_capabilities: SGLangArgumentCapabilities | None = None,
) -> tuple[list[ServingConfig], list[ValidationRejection]]:
    validation_rejections: list[ValidationRejection] = []
    valid_candidates: list[ServingConfig] = []
    for config in candidates:
        validation = validate_managed_candidate(
            config,
            backend=backend,
            model_metadata=model_metadata,
            vllm_argument_capabilities=vllm_argument_capabilities,
            sglang_argument_capabilities=sglang_argument_capabilities,
        )
        if validation.valid:
            valid_candidates.append(config)
            if backfill_valid_candidates and len(valid_candidates) >= limit:
                break
        else:
            validation_rejections.append((config, validation))
    return valid_candidates[:limit], validation_rejections


def _synthesize_managed_candidates(
    *,
    provider: CandidateSynthesisProvider,
    run_dir: Path,
    initial_candidates: list[ServingConfig],
    validation_rejections: list[ValidationRejection],
    rendered_launch_rows: list[dict[str, object]],
    backend: str,
    model: str,
    goal: Goal,
    hardware,
    model_metadata: ModelCapabilityMetadata,
    vllm_argument_capabilities: VLLMArgumentCapabilities | None,
    sglang_argument_capabilities: SGLangArgumentCapabilities | None,
    telemetry: str,
    trials: int,
    request_timeout_s: float,
    evidence_store: EvidenceStore | None,
    evidence_freshness_hours: float,
    evidence_warnings: list[str],
    backend_metadata: dict[str, object],
    runtime_environment: dict[str, object],
    evidence_decisions_path: Path,
) -> dict[str, object]:
    safe_baseline = _safe_baseline_candidate(initial_candidates)
    initial_preflight = _preflight_evidence(
        candidates=initial_candidates,
        hardware=hardware,
        backend=backend,
        backend_metadata=backend_metadata,
        runtime_environment=runtime_environment,
        model=model,
        telemetry=telemetry,
        goal=goal,
        trials=trials,
        request_timeout_s=request_timeout_s,
        evidence_store=evidence_store,
        evidence_freshness_hours=evidence_freshness_hours,
        evidence_warnings=evidence_warnings,
        evidence_decisions_path=evidence_decisions_path,
        vllm_argument_capabilities=vllm_argument_capabilities,
        sglang_argument_capabilities=sglang_argument_capabilities,
    )
    result = provider.synthesize(
        context=CandidateSynthesisContext(
            hardware=hardware,
            backend=backend,
            backend_argument_capabilities=vllm_argument_capabilities,
            model=model,
            model_metadata=model_metadata,
            goal=goal,
            evidence_summary={
                "exact_fresh_candidate_ids": sorted(initial_preflight["exact_fresh_ids"]),
                "evidence_prior_count": len(initial_preflight["evidence_priors"]),
                "lookup_metadata": initial_preflight["lookup_metadata"],
            },
            safe_baseline=safe_baseline,
            existing_candidates=initial_candidates,
            workload_profile=_workload_profile_from_candidates(initial_candidates),
            max_candidates=3,
        ),
        out_dir=run_dir / "synthesis",
    )
    summary = _synthesis_summary(result, initial_preflight)
    if not result.candidates:
        return {
            **summary,
            "_valid_candidates": initial_candidates,
            "_validation_rejections": validation_rejections,
            "_rendered_launch_rows": rendered_launch_rows,
        }

    synth_valid, synth_rejections = _validate_managed_candidate_pool(
        result.candidates,
        backend=backend,
        model_metadata=model_metadata,
        limit=max(1, len(result.candidates)),
        backfill_valid_candidates=False,
        vllm_argument_capabilities=vllm_argument_capabilities,
        sglang_argument_capabilities=sglang_argument_capabilities,
    )
    synth_canonical, synth_canonical_rejections, synth_rendered_rows = _canonicalize_valid_candidates(
        synth_valid,
        backend=backend,
        vllm_argument_capabilities=vllm_argument_capabilities,
        sglang_argument_capabilities=sglang_argument_capabilities,
        runtime_environment=runtime_environment,
    )
    validation_rejections.extend(synth_rejections)
    validation_rejections.extend(synth_canonical_rejections)
    existing_keys = {
        _candidate_equivalence_key(
            config,
            trials=trials,
            request_timeout_s=request_timeout_s,
            telemetry=telemetry,
        )
        for config in initial_candidates
    }
    merged_candidates = list(initial_candidates)
    merged_rendered_rows = list(rendered_launch_rows)
    for config, row in zip(synth_canonical, synth_rendered_rows, strict=False):
        key = _candidate_equivalence_key(
            config,
            trials=trials,
            request_timeout_s=request_timeout_s,
            telemetry=telemetry,
        )
        record = _synthesis_candidate_record(config, status="validated", reason=None)
        if key in existing_keys:
            record["status"] = "deduped"
            record["reason"] = "Matched an existing canonical launch and workload config."
            summary["candidate_records"].append(record)
            continue
        existing_keys.add(key)
        merged_candidates.append(_with_synthesis_status(config, "validated"))
        merged_rendered_rows.append(row)
        summary["candidate_records"].append(record)
    for config, validation in synth_rejections + synth_canonical_rejections:
        summary["candidate_records"].append(
            _synthesis_candidate_record(config, status="rejected", reason=validation.reason)
        )
    return {
        **summary,
        "_valid_candidates": merged_candidates,
        "_validation_rejections": validation_rejections,
        "_rendered_launch_rows": merged_rendered_rows,
    }


def _empty_synthesis_summary() -> dict[str, object]:
    return {
        "schema_version": SYNTHESIS_SCHEMA_VERSION,
        "provider_results": [],
        "candidate_records": [],
        "summary": {
            "proposed_candidate_count": 0,
            "validated_candidate_count": 0,
            "deduped_candidate_count": 0,
            "rejected_candidate_count": 0,
            "pruned_candidate_count": 0,
            "selected_for_evaluation_count": 0,
        },
    }


def _synthesis_summary(result: CandidateSynthesisResult, preflight: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": SYNTHESIS_SCHEMA_VERSION,
        "provider_results": [synthesis_result_to_artifact(result)],
        "initial_preflight": {
            "exact_fresh_candidate_ids": sorted(preflight["exact_fresh_ids"]),
            "evidence_prior_count": len(preflight["evidence_priors"]),
        },
        "candidate_records": [
            _synthesis_candidate_record(config, status="proposed", reason=None)
            for config in result.candidates
        ],
        "summary": {
            "proposed_candidate_count": len(result.candidates),
            "validated_candidate_count": 0,
            "deduped_candidate_count": 0,
            "rejected_candidate_count": 0,
            "pruned_candidate_count": 0,
            "selected_for_evaluation_count": 0,
        },
    }


def _finalize_synthesis_summary(
    summary: dict[str, object],
    *,
    valid_candidate_ids: set[str],
    pruned_candidate_ids: set[str],
    validation_rejections: list[ValidationRejection],
) -> dict[str, object]:
    rejection_by_id = {config.id: validation.reason for config, validation in validation_rejections}
    raw_records = [record for record in summary.get("candidate_records", []) if isinstance(record, dict)]
    final_status_by_id: dict[str, tuple[str, str | None]] = {}
    for record in raw_records:
        candidate_id = str(record.get("candidate_id"))
        status = str(record.get("status") or "proposed")
        reason = record.get("reason") if isinstance(record.get("reason"), str) else None
        if status == "validated":
            if candidate_id in valid_candidate_ids:
                status = "selected_for_evaluation"
            elif candidate_id in pruned_candidate_ids:
                status = "pruned_by_prior"
            elif candidate_id in rejection_by_id:
                status = "rejected"
                reason = rejection_by_id[candidate_id]
        if status != "proposed":
            final_status_by_id[candidate_id] = (status, reason)
    records = []
    seen: set[tuple[str, str]] = set()
    for record in raw_records:
        candidate_id = str(record.get("candidate_id"))
        status = str(record.get("status") or "proposed")
        reason = record.get("reason") if isinstance(record.get("reason"), str) else None
        if candidate_id in final_status_by_id:
            status, reason = final_status_by_id[candidate_id]
        elif status == "validated" and candidate_id in rejection_by_id:
            status = "rejected"
            reason = rejection_by_id[candidate_id]
        key = (candidate_id, status)
        if key in seen:
            continue
        seen.add(key)
        updated = dict(record)
        updated["status"] = status
        updated["reason"] = reason
        records.append(updated)
    summary["candidate_records"] = records
    summary["summary"] = {
        "proposed_candidate_count": sum(1 for record in records if record.get("status") == "proposed"),
        "validated_candidate_count": sum(1 for record in records if record.get("status") == "validated"),
        "deduped_candidate_count": sum(1 for record in records if record.get("status") == "deduped"),
        "rejected_candidate_count": sum(1 for record in records if record.get("status") == "rejected"),
        "pruned_candidate_count": sum(1 for record in records if record.get("status") == "pruned_by_prior"),
        "selected_for_evaluation_count": sum(1 for record in records if record.get("status") == "selected_for_evaluation"),
    }
    return summary


def _apply_synthesis_execution_status(
    summary: dict[str, object],
    candidate_results: list[ManagedCandidateResult],
) -> dict[str, object]:
    result_by_id = {result.config_id: result for result in candidate_results}
    records = []
    for record in summary.get("candidate_records", []):
        if not isinstance(record, dict):
            continue
        candidate_id = str(record.get("candidate_id"))
        status = str(record.get("status") or "")
        result = result_by_id.get(candidate_id)
        updated = dict(record)
        if status == "selected_for_evaluation" and result is not None:
            if result.status in {"completed", "resumed"}:
                updated["status"] = "measured"
            elif result.status == "evidence_hit":
                updated["status"] = "evidence_hit"
            elif result.status == "failed":
                updated["status"] = "failed"
                updated["reason"] = result.error
        records.append(updated)
    summary["candidate_records"] = records
    summary["summary"] = {
        "proposed_candidate_count": sum(1 for record in records if record.get("status") == "proposed"),
        "validated_candidate_count": sum(1 for record in records if record.get("status") == "validated"),
        "deduped_candidate_count": sum(1 for record in records if record.get("status") == "deduped"),
        "rejected_candidate_count": sum(1 for record in records if record.get("status") == "rejected"),
        "pruned_candidate_count": sum(1 for record in records if record.get("status") == "pruned_by_prior"),
        "selected_for_evaluation_count": sum(1 for record in records if record.get("status") == "selected_for_evaluation"),
        "measured_candidate_count": sum(1 for record in records if record.get("status") == "measured"),
        "evidence_hit_candidate_count": sum(1 for record in records if record.get("status") == "evidence_hit"),
        "failed_candidate_count": sum(1 for record in records if record.get("status") == "failed"),
    }
    return summary


def _synthesis_candidate_record(config: ServingConfig, *, status: str, reason: str | None) -> dict[str, object]:
    extra = dict(config.extra or {})
    return {
        "candidate_id": config.id,
        "candidate_source": extra.get("candidate_source"),
        "status": status,
        "reason": reason,
        "synthesis_rationale": extra.get("synthesis_rationale"),
        "synthesis_confidence": extra.get("synthesis_confidence"),
        "synthesis_constraints": extra.get("synthesis_constraints"),
        "aiconfigurator_system_key": extra.get("aiconfigurator_system_key"),
        "aiconfigurator_rank": extra.get("aiconfigurator_rank"),
        "aiconfigurator_predicted_metrics": extra.get("aiconfigurator_predicted_metrics"),
        "dtype": config.dtype,
        "quantization": config.quantization,
        "max_model_len": config.max_context_tokens,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "max_num_seqs": config.max_batch_size,
        "tensor_parallel_size": config.tensor_parallelism,
        "block_size": config.block_size,
        "kv_cache_dtype": config.kv_cache_dtype,
        "enforce_eager": config.enforce_eager,
        "max_num_batched_tokens": config.max_num_batched_tokens,
        "enable_chunked_prefill": config.enable_chunked_prefill,
        "max_cudagraph_capture_size": config.max_cudagraph_capture_size,
        "enable_prefix_caching": config.enable_prefix_caching,
        "workload_concurrency": extra.get("workload_concurrency"),
    }


def _with_synthesis_status(config: ServingConfig, status: str) -> ServingConfig:
    extra = dict(config.extra or {})
    if extra.get("candidate_source") == SYNTHESIS_SOURCE:
        extra["synthesis_status"] = status
    return replace(config, extra=extra)


def _workload_profile_from_candidates(candidates: list[ServingConfig]) -> dict[str, object]:
    if not candidates:
        return {}
    profiles = [
        _workload_profile_payload((config.extra or {}).get("workload_profile"))
        for config in candidates
        if _workload_profile_payload((config.extra or {}).get("workload_profile"))
    ]
    if profiles:
        return profiles[0]
    return {
        "profile_name": "default",
        "max_existing_concurrency": max(_positive_int((config.extra or {}).get("workload_concurrency"), default=config.max_batch_size) for config in candidates),
        "max_existing_context": max(config.max_context_tokens for config in candidates),
        "max_existing_batch": max(config.max_batch_size for config in candidates),
        "max_new_tokens": max(_positive_int((config.extra or {}).get("max_new_tokens"), default=128) for config in candidates),
    }


def _workload_profile_payload(value: object) -> dict[str, object]:
    if isinstance(value, WorkloadProfile):
        return to_dict(value)
    if isinstance(value, dict):
        return dict(value)
    return {}


def _candidate_equivalence_key(
    config: ServingConfig,
    *,
    trials: int,
    request_timeout_s: float,
    telemetry: str,
) -> tuple[str, str]:
    comparable = _strip_candidate_metadata(config)
    launch_hash = launch_config_hash(serving_config_to_launch_config(comparable))
    workload_hash = workload_config_hash(
        serving_config_to_workload_config(
            comparable,
            trials=trials,
            request_timeout_s=request_timeout_s,
            telemetry=telemetry,
        )
    )
    return launch_hash, workload_hash


def _strip_candidate_metadata(config: ServingConfig) -> ServingConfig:
    extra = {
        key: value
        for key, value in dict(config.extra or {}).items()
        if key
        not in {
            "baseline",
            "candidate_source",
            "model_native",
            "prior_confidence",
            "prior_notes",
            "prior_source",
            "raw_aiconfigurator_candidate",
            "synthesis_confidence",
            "synthesis_constraints",
            "synthesis_rationale",
            "synthesis_status",
            "aiconfigurator_predicted_metrics",
            "aiconfigurator_rank",
            "aiconfigurator_system_key",
        }
    }
    return replace(config, extra=extra)


def _safe_baseline_candidate(candidates: list[ServingConfig]) -> ServingConfig | None:
    for config in candidates:
        extra = config.extra or {}
        if extra.get("candidate_source") == "safe_baseline" and extra.get("baseline") is True:
            return config
    return candidates[0] if candidates else None


def _candidate_source_counts(candidates: list[ServingConfig]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for config in candidates:
        source = str((config.extra or {}).get("candidate_source") or "unknown")
        counts[source] = counts.get(source, 0) + 1
    return dict(sorted(counts.items()))


def _open_evidence_store(*, evidence_db_path: Path | None, evidence_write: bool) -> tuple[EvidenceStore | None, list[str]]:
    if not evidence_write or evidence_db_path is None:
        return None, []
    try:
        return EvidenceStore(evidence_db_path), []
    except Exception as exc:
        return None, [f"Evidence DB unavailable: {exc.__class__.__name__}: {exc}"]


def _backend_metadata(adapter: ManagedBackendAdapter, backend: str) -> dict[str, object]:
    metadata_fn = getattr(adapter, "backend_metadata", None)
    if callable(metadata_fn):
        metadata = metadata_fn()
        if isinstance(metadata, dict):
            return dict(metadata)
    return {"adapter": getattr(adapter, "name", backend), "backend": backend}


def _backend_argument_capabilities(adapter: ManagedBackendAdapter, backend: str) -> VLLMArgumentCapabilities | None:
    if backend != "vllm":
        return None
    capabilities_fn = getattr(adapter, "argument_capabilities", None)
    if callable(capabilities_fn):
        capabilities = capabilities_fn()
        if isinstance(capabilities, VLLMArgumentCapabilities):
            return capabilities
    return None


def _backend_sglang_argument_capabilities(adapter: ManagedBackendAdapter, backend: str) -> SGLangArgumentCapabilities | None:
    if backend != "sglang":
        return None
    capabilities_fn = getattr(adapter, "argument_capabilities", None)
    if callable(capabilities_fn):
        capabilities = capabilities_fn()
        if isinstance(capabilities, SGLangArgumentCapabilities):
            return capabilities
    return None


def _backend_capability_help_hash(
    vllm_argument_capabilities: VLLMArgumentCapabilities | None,
    sglang_argument_capabilities: SGLangArgumentCapabilities | None,
) -> str | None:
    capabilities = (
        vllm_argument_capabilities
        or sglang_argument_capabilities
    )
    return capabilities.help_hash if capabilities is not None else None


def _rendered_launch_command(
    config: ServingConfig,
    *,
    backend: str,
    vllm_argument_capabilities: VLLMArgumentCapabilities | None,
    sglang_argument_capabilities: SGLangArgumentCapabilities | None,
) -> list[str]:
    if backend == "vllm":
        return render_vllm_launch(
            config,
            capabilities=vllm_argument_capabilities,
        ).command
    if backend == "sglang":
        return render_sglang_launch(
            config,
            capabilities=sglang_argument_capabilities,
        ).command
    return []


def _preflight_evidence(
    *,
    candidates: list[ServingConfig],
    hardware,
    backend: str,
    backend_metadata: dict[str, object],
    runtime_environment: dict[str, object],
    model: str,
    telemetry: str,
    goal: Goal,
    trials: int,
    request_timeout_s: float,
    evidence_store: EvidenceStore | None,
    evidence_freshness_hours: float,
    evidence_warnings: list[str],
    evidence_decisions_path: Path,
    vllm_argument_capabilities: VLLMArgumentCapabilities | None,
    sglang_argument_capabilities: SGLangArgumentCapabilities | None,
) -> dict[str, object]:
    exact_fresh_ids: set[str] = set()
    evidence_priors: list[PriorCandidate] = []
    lookup_metadata: list[dict[str, object]] = []
    if evidence_store is None:
        return {
            "exact_fresh_ids": exact_fresh_ids,
            "evidence_priors": evidence_priors,
            "lookup_metadata": lookup_metadata,
        }
    for config in candidates:
        launch_config = serving_config_to_launch_config(config)
        workload = serving_config_to_workload_config(
            config,
            trials=trials,
            request_timeout_s=request_timeout_s,
            telemetry=telemetry,
        )
        context = build_evidence_request_context(
            hardware=hardware,
            backend=backend,
            backend_metadata=backend_metadata,
            model=model,
            telemetry=telemetry,
            launch_config=launch_config,
            workload_config=workload,
            goal=goal.value,
            trials=workload.trials,
            runtime_environment=runtime_environment,
            rendered_launch_command=_rendered_launch_command(
                config,
                backend=backend,
                vllm_argument_capabilities=vllm_argument_capabilities,
                sglang_argument_capabilities=sglang_argument_capabilities,
            ),
            backend_capability_help_hash=_backend_capability_help_hash(
                vllm_argument_capabilities,
                sglang_argument_capabilities,
            ),
        )
        try:
            lookup = evidence_store.lookup_evidence(context, freshness_hours=evidence_freshness_hours)
        except Exception as exc:
            warning = f"Evidence DB preflight lookup failed for {config.id}: {exc.__class__.__name__}: {exc}"
            evidence_warnings.append(warning)
            lookup_metadata.append({"candidate_id": config.id, "hit_type": "lookup_failed", "error": warning})
            continue
        decision = classify_evidence_lookup(
            lookup,
            candidate_id=config.id,
            context=context,
            current_backend_argument_capabilities=(
                vllm_argument_capabilities
                or sglang_argument_capabilities
            ),
            goal=goal.value,
        )
        _append_jsonl(evidence_decisions_path, [decision.to_artifact()])
        lookup_metadata.append(
            {
                "candidate_id": config.id,
                "workload_id": workload.workload_id,
                "evidence_key": context.evidence_key,
                "hit_type": lookup.hit_type.value,
                "classification": decision.classification.value,
                "used_as_exact": decision.used_as_exact,
                "used_as_prior": decision.used_as_prior,
                "measurement_id": lookup.measurement.get("measurement_id") if lookup.measurement else None,
            }
        )
        if decision.used_as_exact:
            exact_fresh_ids.add(config.id)
            continue
        prior = evidence_lookup_to_prior(config, lookup)
        if prior is not None:
            evidence_priors.append(prior)
    return {
        "exact_fresh_ids": exact_fresh_ids,
        "evidence_priors": evidence_priors,
        "lookup_metadata": lookup_metadata,
    }


def _prior_result_fields(config: ServingConfig) -> dict[str, object]:
    extra = config.extra or {}
    notes = extra.get("prior_notes")
    return {
        "prior_source": _optional_str(extra.get("prior_source")),
        "prior_confidence": _optional_float(extra.get("prior_confidence")),
        "prior_notes": [str(note) for note in notes] if isinstance(notes, list) else [],
    }


def _validate_run_inputs(backend: str, limit: int, trials: int, startup_timeout_s: float, cooldown_s: float) -> None:
    validate_managed_backend_supported(backend)
    if limit < 1:
        raise ValueError("limit must be at least 1.")
    if trials < 1:
        raise ValueError("trials must be at least 1.")
    if startup_timeout_s <= 0:
        raise ValueError("startup_timeout_s must be greater than 0.")
    if cooldown_s < 0:
        raise ValueError("cooldown_s must be at least 0.")


def _benchmark_config(
    *,
    config: ServingConfig,
    base_url: str,
    model: str,
    trial: int,
    trials: int,
    telemetry: str,
    timeout_s: float,
) -> EndpointBenchmarkConfig:
    run_id = config.id if trials == 1 else f"{config.id}-trial-{trial + 1:02d}"
    concurrency = max(1, config.max_batch_size)
    return EndpointBenchmarkConfig(
        run_id=run_id,
        base_url=base_url,
        model=model,
        concurrency=concurrency,
        num_requests=max(128, 2 * concurrency),
        max_tokens=128,
        prompt=DEFAULT_ENDPOINT_PROMPT,
        timeout_s=timeout_s,
        telemetry=telemetry,
    )


def _benchmark_config_from_workload(
    *,
    workload: WorkloadConfig,
    base_url: str,
    model: str,
    trial: int,
    launch_provenance: dict[str, object] | None = None,
) -> EndpointBenchmarkConfig:
    run_id = workload.workload_id if workload.trials == 1 else f"{workload.workload_id}-trial-{trial + 1:02d}"
    launch_provenance = launch_provenance or {}
    return EndpointBenchmarkConfig(
        run_id=run_id,
        base_url=base_url,
        model=model,
        concurrency=workload.concurrency,
        num_requests=workload.num_requests,
        max_tokens=workload.max_new_tokens,
        prompt=workload.prompt,
        timeout_s=workload.timeout_s,
        endpoint=workload.endpoint,
        telemetry=workload.telemetry,
        warmup_requests=workload.warmup_requests,
        steady_state_duration_s=workload.benchmark_duration_s,
        idle_baseline_duration_s=workload.idle_baseline_duration_s,
        idle_power_watts=workload.idle_power_watts,
        soak_duration_s=workload.soak_duration_s,
        stream=workload.stream,
        backend_name=_optional_str(launch_provenance.get("backend_name")),
        backend_version=_optional_str(launch_provenance.get("backend_version")),
        backend_launch_command=(
            [str(item) for item in launch_provenance.get("backend_launch_command", [])]
            if isinstance(launch_provenance.get("backend_launch_command"), list)
            else []
        ),
        backend_launch_command_hash=_optional_str(launch_provenance.get("backend_launch_command_hash")),
        backend_effective_values=_dict_payload(launch_provenance.get("backend_effective_values")),
        backend_applied_configuration=_dict_payload(launch_provenance.get("backend_applied_configuration")),
        backend_omitted_values=_dict_payload(launch_provenance.get("backend_omitted_values")),
        backend_unsupported_values=_dict_payload(launch_provenance.get("backend_unsupported_values")),
        backend_unavailable_values=_dict_payload(launch_provenance.get("backend_unavailable_values")),
        backend_flag_aliases=_dict_payload(launch_provenance.get("backend_flag_aliases")),
        backend_capabilities_help_hash=_optional_str(launch_provenance.get("backend_capabilities_help_hash")),
    )


def _with_measurement_quality_options(
    candidates: list[ServingConfig],
    *,
    warmup_requests: int,
    steady_state_duration_s: float | None,
    idle_baseline_duration_s: float,
    idle_power_watts: float | None,
    soak_duration_s: float | None,
    stream: bool,
) -> list[ServingConfig]:
    if (
        warmup_requests <= 0
        and steady_state_duration_s is None
        and idle_baseline_duration_s <= 0
        and idle_power_watts is None
        and soak_duration_s is None
        and not stream
    ):
        return candidates
    updated = []
    for config in candidates:
        extra = dict(config.extra or {})
        if warmup_requests > 0:
            extra["warmup_requests"] = warmup_requests
        if steady_state_duration_s is not None:
            extra["benchmark_duration_s"] = steady_state_duration_s
        if idle_baseline_duration_s > 0:
            extra["idle_baseline_duration_s"] = idle_baseline_duration_s
        if idle_power_watts is not None:
            extra["idle_power_watts"] = idle_power_watts
        if soak_duration_s is not None:
            extra["soak_duration_s"] = soak_duration_s
        if stream:
            extra["stream"] = True
        updated.append(replace(config, extra=extra))
    return updated


def _positive_int(value: object, *, default: int) -> int:
    try:
        parsed = int(value) if value is not None else default
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _optional_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _positive_float(value: object, *, default: float) -> float:
    try:
        parsed = float(value) if value is not None else default
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _optional_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _round_or_none(value: float | int | None) -> float | None:
    return round(float(value), 6) if value is not None else None


def _optional_bool(value: object, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _write_optimizer_failure_cache(
    path: Path,
    *,
    failures: list[CandidateFailureRecord],
    configs_by_id: dict[str, ServingConfig],
) -> dict[str, object]:
    stage_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    entries: list[dict[str, object]] = []
    for failure in failures:
        config = configs_by_id.get(failure.config_id)
        failure_reason = _failure_reason_from_details(failure.details)
        stage_counts[failure.stage] = stage_counts.get(failure.stage, 0) + 1
        reason_counts[failure_reason] = reason_counts.get(failure_reason, 0) + 1
        cache_payload = {
            "backend": config.backend if config is not None else None,
            "stage": failure.stage,
            "reason": failure_reason,
            "config": _failure_cache_config_payload(config),
        }
        entries.append(
            {
                "cache_key": stable_payload_hash(cache_payload),
                "config_id": failure.config_id,
                "backend": config.backend if config is not None else None,
                "stage": failure.stage,
                "reason": failure_reason,
                "error": failure.error,
                "details": failure.details,
            }
        )
    payload = {
        "schema_version": "optimizer_failure_cache/v1",
        "scope": "managed_candidate_failures",
        "entry_count": len(entries),
        "summary": {
            "entry_count": len(entries),
            "stage_counts": dict(sorted(stage_counts.items())),
            "reason_counts": dict(sorted(reason_counts.items())),
            "cache_key_policy": "backend_stage_reason_and_serving_config_without_candidate_id",
        },
        "entries": entries,
        "notes": [
            "Failure cache entries are artifact backed and scoped to managed candidate failures.",
            "Cache keys intentionally exclude candidate ids so equivalent failed configs can be recognized.",
        ],
    }
    write_json(path, payload)
    return payload


def _failure_cache_config_payload(config: ServingConfig | None) -> dict[str, object]:
    if config is None:
        return {}
    payload = dict(to_dict(config))
    payload.pop("id", None)
    return payload


def _candidate_failure(
    run_id: str,
    config_id: str,
    stage: str,
    error: str,
    details: dict[str, object] | None = None,
) -> CandidateFailureRecord:
    return CandidateFailureRecord(
        run_id=run_id,
        config_id=config_id,
        stage=stage,
        error=error,
        timestamp=datetime.now(timezone.utc).isoformat(),
        details=details or {},
    )


def _validation_failure_reason(validation: CandidateValidationResult) -> str:
    reason = validation.reason or ""
    classified = _failure_reason_from_text(reason)
    if classified is not None:
        return classified
    markers = (
        "requires detected",
        "requires installed",
        "cannot be rendered",
        "does not support",
        "not listed by installed",
        "without a direct",
        "unsupported",
    )
    if any(marker in reason for marker in markers):
        return "invalid_config"
    return "validation_failed"


def _failure_reason_for_exception(exc: Exception, *, stage: str) -> str:
    classified = _failure_reason_from_text(f"{exc.__class__.__name__}: {exc}")
    if classified is not None:
        return classified
    if stage == "launch" and isinstance(exc, ValueError):
        return "invalid_config"
    if stage == "launch":
        return "backend_failed_to_start"
    if stage == "benchmark" and (isinstance(exc, TimeoutError) or _is_timeout_text(exc)):
        return "benchmark_timeout"
    return stage


def _failure_reason_for_health(health: HealthCheckResult) -> str:
    payload = f"{health.status} {health.error or ''}"
    details = health.details if isinstance(health.details, dict) else {}
    if details:
        payload = f"{payload} {json.dumps(details, sort_keys=True)}"
    classified = _failure_reason_from_text(payload)
    if classified is not None:
        return classified
    if health.status == "process_exited" or "process_returncode" in details:
        return "backend_crashed_during_load"
    return "backend_failed_to_start"


def _failure_reason_from_details(details: dict[str, object]) -> str:
    reason = details.get("reason") if isinstance(details, dict) else None
    if reason is not None and str(reason).strip():
        return str(reason)
    return "unclassified"


def _failure_reason_from_text(value: object) -> str | None:
    text = str(value).lower()
    if _has_any(
        text,
        (
            "out of memory",
            "cuda oom",
            "cuda error: out of memory",
            " oom",
            "oom ",
            "(oom)",
            "[oom]",
            ": oom",
            "failed to allocate",
            "not enough memory",
            "insufficient memory",
        ),
    ):
        return "out_of_memory"
    if _has_any(text, ("gated repo", "gated model", "requires authentication", "unauthorized", "forbidden", "permission denied", "access denied", "http_401", "http_403", " 401", " 403")):
        return "unavailable_gated_access"
    if _has_any(text, ("model not found", "repository not found", "repo not found", "not found", "does not exist", "unavailable model", "http_404", " 404")):
        return "unavailable_model"
    return None


def _has_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _is_timeout_text(value: object) -> bool:
    text = str(value).lower()
    return _has_any(text, ("timed out", "timeout", "deadline exceeded"))


def _failed_result(config: ServingConfig, failure: CandidateFailureRecord, workload: WorkloadConfig | None = None) -> ManagedCandidateResult:
    workload_fields = _workload_result_fields(workload, source="failed") if workload is not None else {}
    return ManagedCandidateResult(
        config_id=config.id,
        backend=config.backend,
        status="failed",
        failure_stage=failure.stage,
        error=failure.error,
        **_prior_result_fields(config),
        **workload_fields,
    )


def _rejected_result(config: ServingConfig, failure: CandidateFailureRecord) -> ManagedCandidateResult:
    return ManagedCandidateResult(
        config_id=config.id,
        backend=config.backend,
        status="rejected",
        failure_stage=failure.stage,
        error=failure.error,
        **_prior_result_fields(config),
    )


def _lifecycle(
    run_id: str,
    config_id: str,
    backend: str,
    event: str,
    status: str,
    message: str | None = None,
    pid: int | None = None,
    pgid: int | None = None,
    returncode: int | None = None,
    details: dict[str, object] | None = None,
) -> ManagedLifecycleRecord:
    return ManagedLifecycleRecord(
        run_id=run_id,
        config_id=config_id,
        backend=backend,
        event=event,
        status=status,
        timestamp=datetime.now(timezone.utc).isoformat(),
        message=message,
        pid=pid,
        pgid=pgid,
        returncode=returncode,
        details=details or {},
    )


def _run_status(completed: int, failure_count: int) -> str:
    if completed == 0 and failure_count > 0:
        return "failed"
    if failure_count > 0:
        return "warning"
    return "success"


def _evidence_decision_summary(path: Path) -> dict[str, object]:
    counts: dict[str, int] = {}
    exact_count = 0
    prior_count = 0
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {"decision_count": 0, "classifications": {}, "used_as_exact_count": 0, "used_as_prior_count": 0}
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        classification = str(row.get("classification") or "unknown")
        counts[classification] = counts.get(classification, 0) + 1
        if row.get("used_as_exact") is True:
            exact_count += 1
        if row.get("used_as_prior") is True:
            prior_count += 1
    return {
        "decision_count": sum(counts.values()),
        "classifications": dict(sorted(counts.items())),
        "used_as_exact_count": exact_count,
        "used_as_prior_count": prior_count,
    }


def _append_jsonl(path: Path, rows: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_dict(row), sort_keys=True) + "\n")
