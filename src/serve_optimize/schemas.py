"""Typed records shared by the optimizer pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class Goal(str, Enum):
    PERFORMANCE = "performance"
    BALANCED = "balanced"
    EFFICIENT = "efficient"


class RecommendationGoal(str, Enum):
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    EFFICIENCY = "efficiency"
    BALANCED = "balanced"


class Backend(str, Enum):
    DRY_RUN = "dry-run"
    TRANSFORMERS = "transformers"
    VLLM = "vllm"
    SGLANG = "sglang"
    TRT_LLM = "trt-llm"
    LLAMA_CPP = "llama.cpp"


class PriorSource(str, Enum):
    AICONFIGURATOR = "aiconfigurator"
    EVIDENCE_NEAR_HIT = "evidence_near_hit"
    EVIDENCE_STALE_HIT = "evidence_stale_hit"
    HEURISTIC = "heuristic"
    BACKEND_DEFAULT = "backend_default"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class GpuDevice:
    index: int
    name: str
    uuid: str | None = None
    total_memory_mb: int | None = None
    free_memory_mb: int | None = None
    compute_capability: str | None = None
    mig_mode: str | None = None
    mig_parent_uuid: str | None = None
    mig_profile: str | None = None
    power_limit_watts: float | None = None
    current_power_watts: float | None = None
    sm_clock_mhz: int | None = None
    mem_clock_mhz: int | None = None
    driver_version: str | None = None
    cuda_version: str | None = None
    source: str = "unknown"
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_mig(self) -> bool:
        return bool(self.mig_profile or (self.uuid and self.uuid.startswith("MIG-")))


@dataclass(frozen=True)
class HardwareSnapshot:
    hostname: str
    platform: str
    python_version: str
    detected_at: str
    gpus: list[GpuDevice] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @classmethod
    def empty(cls, hostname: str, platform: str, python_version: str, note: str) -> HardwareSnapshot:
        return cls(
            hostname=hostname,
            platform=platform,
            python_version=python_version,
            detected_at=datetime.now(timezone.utc).isoformat(),
            gpus=[],
            notes=[note],
        )

    @property
    def best_gpu(self) -> GpuDevice | None:
        if not self.gpus:
            return None
        return max(self.gpus, key=lambda gpu: gpu.total_memory_mb or 0)


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    parameter_count_b: float
    max_context_tokens: int = 4096
    family: str = "unknown"

    @property
    def parameter_count(self) -> float:
        return self.parameter_count_b * 1_000_000_000


@dataclass(frozen=True)
class ModelCapabilityMetadata:
    model_id: str
    metadata_known: bool = False
    is_local_path: bool = False
    config_path: str | None = None
    torch_dtype: str | None = None
    quantization_method: str | None = None
    quantization_config: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ServingConfig:
    id: str
    backend: str
    model_id: str
    dtype: str
    quantization: str
    max_batch_size: int
    max_context_tokens: int
    kv_cache_policy: str
    scheduler: str
    tensor_parallelism: int = 1
    gpu_memory_utilization: float = 0.9
    block_size: int | None = None
    kv_cache_dtype: str | None = None
    enforce_eager: bool | None = None
    max_num_batched_tokens: int | None = None
    enable_chunked_prefill: bool | None = None
    max_cudagraph_capture_size: int | None = None
    enable_prefix_caching: bool | None = None
    power_limit_watts: float | None = None
    estimated_vram_mb: int | None = None
    notes: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LaunchConfig:
    backend: str
    model: str
    dtype: str
    quantization: str
    max_model_len: int
    gpu_memory_utilization: float
    max_num_seqs: int
    tensor_parallel_size: int = 1
    block_size: int | None = None
    max_num_batched_tokens: int | None = None
    kv_cache_dtype: str | None = None
    enforce_eager: bool | None = None
    enable_chunked_prefill: bool | None = None
    max_cudagraph_capture_size: int | None = None
    enable_prefix_caching: bool | None = None
    kv_cache_policy: str | None = None
    scheduler: str | None = None
    power_limit_watts: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkloadConfig:
    workload_id: str
    candidate_id: str
    concurrency: int
    num_requests: int
    max_new_tokens: int
    prompt: str
    timeout_s: float
    trials: int = 1
    request_rate: float | None = None
    input_length: int | None = None
    output_length: int | None = None
    num_prompts: int | None = None
    dataset: str | None = None
    warmup_duration_s: float | None = None
    benchmark_duration_s: float | None = None
    warmup_requests: int = 0
    idle_baseline_duration_s: float = 0.0
    idle_power_watts: float | None = None
    soak_duration_s: float | None = None
    stream: bool = False
    endpoint: str = "/v1/chat/completions"
    telemetry: str = "none"
    prior_source: str | None = None
    prior_confidence: float | None = None
    prior_notes: list[str] = field(default_factory=list)
    rung: str | None = None
    rung_index: int | None = None
    promotion_status: str | None = None
    promotion_reason: str | None = None
    measured_or_evidence_source: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkloadProfile:
    profile_name: str = "default"
    input_tokens: int | None = None
    output_tokens: int | None = None
    concurrency: int | None = None
    num_requests: int | None = None
    max_new_tokens: int | None = None
    dataset: str | None = None
    token_distribution: dict[str, Any] = field(default_factory=dict)
    slo_constraints: dict[str, Any] = field(default_factory=dict)
    prefix_reuse_expected: bool = False
    repeated_prefix_ratio: float | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LaunchGroup:
    group_id: str
    launch_config: LaunchConfig
    workload_configs: list[WorkloadConfig]
    original_config_ids: list[str]
    launch_config_hash: str
    evidence_lookup_metadata: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class PriorCandidate:
    source: str
    candidate_id: str
    config_id: str | None = None
    support_status: str = "unknown"
    confidence: float | None = None
    predicted_throughput_tokens_per_sec: float | None = None
    predicted_ttft_ms: float | None = None
    predicted_tpot_ms: float | None = None
    predicted_latency_ms: float | None = None
    predicted_memory_gb: float | None = None
    notes: list[str] = field(default_factory=list)
    raw_prior_payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PriorResult:
    source: str
    available: bool
    used: bool
    candidates: list[PriorCandidate] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    raw_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationRung:
    index: int
    name: str
    purpose: str
    num_requests_scale: float = 1.0
    min_num_requests: int = 1
    max_num_requests: int | None = None
    trials: int | None = None
    promotion_fraction: float = 0.5
    max_promotions: int | None = None


@dataclass(frozen=True)
class PromotionDecision:
    candidate_id: str
    from_rung: str
    to_rung: str | None
    promoted: bool
    reason: str
    metrics: dict[str, Any] = field(default_factory=dict)
    prior_source: str | None = None
    prior_confidence: float | None = None


@dataclass(frozen=True)
class RungResult:
    candidate_id: str
    workload_id: str
    rung: str
    rung_index: int
    status: str
    measured_or_evidence_source: str
    evidence_key: str | None = None
    evidence_hit_type: str | None = None
    evidence_measurement_id: str | None = None
    runtime_fingerprint: str | None = None
    runtime_environment: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidatePromotionSummary:
    policy_name: str
    candidate_count: int
    rung_count: int
    probe_candidate_count: int
    promoted_candidate_count: int
    validation_candidate_count: int
    pruned_after_probe_count: int
    decisions_by_rung: dict[str, dict[str, int]] = field(default_factory=dict)


@dataclass(frozen=True)
class ServerLaunchSpec:
    config_id: str
    backend: str
    model_id: str
    host: str
    port: int
    base_url: str
    command: list[str]
    environment: dict[str, str] = field(default_factory=dict)
    stdout_log_path: str | None = None
    stderr_log_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "server-launch-spec/v1"


@dataclass(frozen=True)
class ServerHandle:
    config_id: str
    backend: str
    pid: int
    pgid: int
    host: str
    port: int
    base_url: str
    started_at: str
    stdout_log_path: str | None = None
    stderr_log_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HealthCheckResult:
    config_id: str
    backend: str
    base_url: str
    healthy: bool
    status: str
    attempts: int
    started_at: str
    ended_at: str
    latency_s: float | None = None
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ManagedLifecycleRecord:
    run_id: str
    config_id: str
    backend: str
    event: str
    status: str
    timestamp: str
    message: str | None = None
    pid: int | None = None
    pgid: int | None = None
    returncode: int | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateFailureRecord:
    run_id: str
    config_id: str
    stage: str
    error: str
    timestamp: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ManagedCandidateResult:
    config_id: str
    backend: str
    status: str
    benchmark_run_dirs: list[str] = field(default_factory=list)
    summary_paths: list[str] = field(default_factory=list)
    failure_stage: str | None = None
    error: str | None = None
    evidence_key: str | None = None
    evidence_hit_type: str | None = None
    evidence_measurement_id: str | None = None
    prior_source: str | None = None
    prior_confidence: float | None = None
    prior_notes: list[str] = field(default_factory=list)
    rung: str | None = None
    promotion_status: str | None = None
    promotion_reason: str | None = None
    measured_or_evidence_source: str | None = None


@dataclass(frozen=True)
class ManagedRunSummary:
    run_id: str
    created_at: str
    backend: str
    model: str
    goal: str
    candidate_count: int
    completed_candidate_count: int
    failed_candidate_count: int
    startup_timeout_s: float
    cooldown_s: float
    trials: int
    status: str
    artifacts: dict[str, str] = field(default_factory=dict)
    candidates: list[ManagedCandidateResult] = field(default_factory=list)
    evidence_db_path: str | None = None
    evidence_write_enabled: bool = False
    evidence_hit_candidate_count: int = 0
    evidence_hits: int = 0
    evidence_warnings: list[str] = field(default_factory=list)
    evidence_decision_summary: dict[str, Any] = field(default_factory=dict)
    cold_launch_count: int = 0
    cold_launches: int = 0
    workload_measurement_count: int = 0
    workload_measurements: int = 0
    skipped_by_evidence_count: int = 0
    launch_groups_count: int = 0
    average_workloads_per_launch: float = 0.0
    backend_metadata: dict[str, Any] = field(default_factory=dict)
    runtime_environment: dict[str, Any] = field(default_factory=dict)
    prior_sources_used: list[str] = field(default_factory=list)
    prior_candidate_count: int = 0
    candidates_after_prior_pruning: int = 0
    candidates_pruned_by_prior: int = 0
    ai_configurator_available: bool = False
    ai_configurator_used: bool = False
    candidate_source_counts: dict[str, int] = field(default_factory=dict)
    synthesis_summary: dict[str, Any] = field(default_factory=dict)
    workload_profile: dict[str, Any] = field(default_factory=dict)
    capability_filtered_count: int = 0
    invalid_quantization_filtered_count: int = 0
    safe_baseline_added: bool = False
    valid_candidate_count_before_prior_pruning: int = 0
    rejected_candidate_count_before_prior_pruning: int = 0
    budget_policy_name: str | None = None
    rung_count: int = 0
    probe_measurement_count: int = 0
    promoted_measurement_count: int = 0
    validation_measurement_count: int = 0
    pruned_after_probe_count: int = 0
    promotion_summary: CandidatePromotionSummary | dict[str, Any] | None = None
    recommendation_status: str = "unavailable"
    recommendation_reason: str | None = None
    selected_config_id: str | None = None
    selected_evidence_key: str | None = None
    selected_measurement_id: str | None = None
    recommendation_score: float | None = None
    recommendation_confidence: str | None = None
    pareto_candidate_count: int = 0
    recommendation_artifact_path: str | None = None
    pareto_artifact_path: str | None = None
    recommendation_summary_txt_path: str | None = None
    recommendation_summary_json_path: str | None = None
    recommendation_quality_audit: dict[str, Any] = field(default_factory=dict)
    optimizer_quality: dict[str, Any] = field(default_factory=dict)
    resume_from_run_dir: str | None = None
    resume_source_run_id: str | None = None
    resume_loaded_candidate_count: int = 0
    resume_skipped_candidate_count: int = 0
    resume_warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AICPrediction:
    backend: str | None = None
    version: str | None = None
    system: str | None = None
    model: str | None = None
    isl: int | None = None
    osl: int | None = None
    concurrency: int | None = None
    request_rate: float | None = None
    bs: int | None = None
    ttft: float | None = None
    tpot: float | None = None
    request_latency: float | None = None
    tokens_s: float | None = None
    tokens_s_gpu: float | None = None
    tokens_s_user: float | None = None
    tp: int | None = None
    pp: int | None = None
    dp: int | None = None
    parallel: str | None = None
    memory: float | None = None
    power_w: float | None = None
    source_path: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EndpointBenchmarkConfig:
    run_id: str
    base_url: str
    model: str
    concurrency: int
    num_requests: int
    max_tokens: int
    prompt: str
    timeout_s: float
    endpoint: str = "/v1/chat/completions"
    telemetry: str = "none"
    device_index: int = 0
    prediction_csv: str | None = None
    warmup_requests: int = 0
    steady_state_duration_s: float | None = None
    idle_power_watts: float | None = None
    idle_baseline_duration_s: float = 0.0
    soak_duration_s: float | None = None
    stream: bool = False
    api_key_env: str | None = None
    schema_version: str = "endpoint-benchmark/v1"


@dataclass(frozen=True)
class RequestRecord:
    request_id: int
    start_time: float
    end_time: float
    latency_s: float
    status: str
    error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    ttft_s: float | None = None
    tpot_s: float | None = None
    timing_source: str | None = None
    token_count_source: str | None = None


@dataclass(frozen=True)
class PowerSampleRecord:
    timestamp_s: float
    phase: str
    watts: float | None
    source: str
    device_index: int | None = None
    device_id: str | None = None
    provider: str | None = None
    power_watts: float | None = None
    gpu_memory_used_mb: int | None = None
    memory_used_mb: int | None = None
    memory_total_mb: int | None = None
    memory_util_percent: float | None = None
    gpu_utilization_pct: float | None = None
    gpu_util_percent: float | None = None
    gpu_temperature_c: float | None = None
    temperature_c: float | None = None
    graphics_clock_mhz: int | None = None
    sm_clock_mhz: int | None = None
    memory_clock_mhz: int | None = None
    power_limit_watts: float | None = None
    enforced_power_limit_watts: float | None = None
    throttle_reasons: str | None = None
    mig_mode: str | None = None
    mig_profile: str | None = None
    device_name: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class TelemetryCapabilities:
    provider: str | None = None
    device_name: str | None = None
    available_fields: list[str] = field(default_factory=list)
    unavailable_fields: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TelemetrySummary:
    telemetry_provider: str | None = None
    telemetry_available: bool = False
    telemetry_quality: str = "unavailable"
    telemetry_warnings: list[str] = field(default_factory=list)
    telemetry_notes: list[str] = field(default_factory=list)
    sample_count: int = 0
    duration_s: float | None = None
    sampling_rate_hz: float | None = None
    missing_fields: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    provider_info: dict[str, Any] = field(default_factory=dict)
    telemetry_capabilities: TelemetryCapabilities | dict[str, Any] | None = None
    power_stats: dict[str, Any] = field(default_factory=dict)
    utilization_stats: dict[str, Any] = field(default_factory=dict)
    thermal_stats: dict[str, Any] = field(default_factory=dict)
    clock_stats: dict[str, Any] = field(default_factory=dict)
    power_sample_count: int = 0
    valid_power_sample_count: int = 0
    power_sampling_duration_s: float | None = None
    power_sampling_rate_hz: float | None = None
    average_power_watts: float | None = None
    min_power_watts: float | None = None
    max_power_watts: float | None = None
    peak_power_watts: float | None = None
    power_stddev_watts: float | None = None
    energy_joules: float | None = None
    joules_per_token: float | None = None
    tokens_per_second_per_watt: float | None = None
    average_gpu_util_percent: float | None = None
    max_gpu_util_percent: float | None = None
    average_memory_util_percent: float | None = None
    max_memory_util_percent: float | None = None
    average_temperature_c: float | None = None
    max_temperature_c: float | None = None
    temperature_rise_c: float | None = None
    temperature_slope_c_per_min: float | None = None
    thermal_stability_classification: str | None = None
    average_sm_clock_mhz: float | None = None
    average_memory_clock_mhz: float | None = None
    average_memory_used_mb: float | None = None
    max_memory_used_mb: int | None = None
    memory_total_mb: int | None = None
    power_limit_watts: float | None = None
    enforced_power_limit_watts: float | None = None
    device_name: str | None = None
    mig_mode: str | None = None
    mig_profile: str | None = None


@dataclass(frozen=True)
class EndpointBenchmarkSummary:
    run_id: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    wall_time_s: float
    request_rate_req_s: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    output_tokens_s: float
    total_tokens_s: float
    avg_latency_s: float | None
    p50_latency_s: float | None
    p95_latency_s: float | None
    p99_latency_s: float | None
    avg_ttft_ms: float | None = None
    p50_ttft_ms: float | None = None
    p95_ttft_ms: float | None = None
    avg_tpot_ms: float | None = None
    p50_tpot_ms: float | None = None
    p95_tpot_ms: float | None = None
    ttft_sample_count: int = 0
    tpot_sample_count: int = 0
    timing_source: str | None = None
    power_sample_count: int = 0
    average_power_watts: float | None = None
    min_power_watts: float | None = None
    max_power_watts: float | None = None
    peak_power_watts: float | None = None
    power_stddev_watts: float | None = None
    power_sampling_duration_s: float | None = None
    power_sampling_rate_hz: float | None = None
    idle_power_watts: float | None = None
    active_power_watts: float | None = None
    active_energy_joules: float | None = None
    energy_joules: float | None = None
    joules_per_token: float | None = None
    active_joules_per_token: float | None = None
    tokens_per_second_per_watt: float | None = None
    active_tokens_per_second_per_watt: float | None = None
    warmup_requests: int = 0
    steady_state_requests: int | None = None
    steady_state_duration_s: float | None = None
    steady_state_total_tokens: int | None = None
    steady_state_total_tokens_s: float | None = None
    steady_state_request_rate_req_s: float | None = None
    measurement_quality: dict[str, Any] = field(default_factory=dict)
    trial_statistics: dict[str, Any] = field(default_factory=dict)
    stability_classification: str = "single_trial"
    confidence_intervals: dict[str, Any] = field(default_factory=dict)
    observed_memory_mb: int | None = None
    average_gpu_util_percent: float | None = None
    max_gpu_util_percent: float | None = None
    average_memory_util_percent: float | None = None
    max_memory_util_percent: float | None = None
    average_temperature_c: float | None = None
    max_temperature_c: float | None = None
    temperature_rise_c: float | None = None
    temperature_slope_c_per_min: float | None = None
    thermal_stability_classification: str | None = None
    average_sm_clock_mhz: float | None = None
    average_memory_clock_mhz: float | None = None
    telemetry_provider: str | None = None
    telemetry_available: bool = False
    telemetry_quality: str = "unavailable"
    telemetry_notes: list[str] = field(default_factory=list)
    telemetry_summary: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    measurement_duration_s: float | None = None
    measured_requests: int | None = None
    measured_successful_requests: int | None = None
    measured_failed_requests: int | None = None
    token_count_source: str | None = None


@dataclass(frozen=True)
class PredictionComparison:
    run_id: str
    metrics: dict[str, dict[str, float | None]]
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ServeCandidate:
    candidate_id: str
    rank: int
    source: str
    model: str | None = None
    backend: str | None = None
    backend_version: str | None = None
    system: str | None = None
    isl: int | None = None
    osl: int | None = None
    prefix: int | None = None
    concurrency: int | None = None
    request_rate: float | None = None
    batch_size: int | None = None
    global_batch_size: int | None = None
    tp: int | None = None
    pp: int | None = None
    dp: int | None = None
    moe_tp: int | None = None
    moe_ep: int | None = None
    parallel: str | None = None
    gemm: str | None = None
    kvcache: str | None = None
    fmha: str | None = None
    moe: str | None = None
    comm: str | None = None
    predicted_ttft_ms: float | None = None
    predicted_tpot_ms: float | None = None
    predicted_request_latency_ms: float | None = None
    predicted_seq_s: float | None = None
    predicted_tokens_s: float | None = None
    predicted_tokens_s_per_gpu: float | None = None
    predicted_tokens_s_per_user: float | None = None
    predicted_memory_gb: float | None = None
    predicted_power_w: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VllmServePlan:
    candidate_id: str
    model: str
    host: str
    port: int
    dtype: str
    tensor_parallel_size: int
    pipeline_parallel_size: int | None
    max_model_len: int
    gpu_memory_utilization: float
    command: list[str]
    shell_command: str
    block_size: int | None = None
    kv_cache_dtype: str | None = None
    enforce_eager: bool | None = None
    max_num_batched_tokens: int | None = None
    enable_chunked_prefill: bool | None = None
    max_cudagraph_capture_size: int | None = None
    enable_prefix_caching: bool | None = None


@dataclass(frozen=True)
class EndpointBenchmarkPlan:
    candidate_id: str
    base_url: str
    model: str
    concurrency: int
    num_requests: int
    max_tokens: int
    expected_input_tokens: int | None
    expected_output_tokens: int | None


@dataclass(frozen=True)
class CandidateEvaluationPlan:
    candidate_id: str
    rank: int
    candidate: ServeCandidate
    serve_plan: VllmServePlan | None = None
    benchmark_plan: EndpointBenchmarkPlan | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CheckRecord:
    name: str
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecommendationInput:
    candidate_id: str
    candidate_rank: int
    candidate_source: str
    model: str | None
    backend: str | None
    candidate: ServeCandidate
    serve_plan: VllmServePlan | None
    benchmark_plan: EndpointBenchmarkPlan | None
    predicted_metrics: dict[str, float | int | str | None] = field(default_factory=dict)
    measured_metrics: dict[str, float | int | str | None] = field(default_factory=dict)
    telemetry_metrics: dict[str, Any] = field(default_factory=dict)
    comparison_metrics: dict[str, float | int | str | None] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RecommendationScore:
    candidate_id: str
    goal: str
    throughput_score: float | None
    latency_score: float | None
    efficiency_score: float | None
    reliability_score: float | None
    prediction_accuracy_score: float | None
    balanced_score: float | None
    final_score: float | None
    power_score: float | None = None
    weights_used: dict[str, float] = field(default_factory=dict)
    missing_metric_penalties: list[str] = field(default_factory=list)
    score_breakdown: dict[str, float | int | str | None] = field(default_factory=dict)
    pareto_optimal: bool = False
    reasons: list[str] = field(default_factory=list)
    disqualifiers: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RecommendationResult:
    recommended_candidate_id: str | None
    goal: str
    selected_score: RecommendationScore | None
    selected_config: ServeCandidate | None
    selected_serve_command: str | None
    selected_benchmark_plan: EndpointBenchmarkPlan | None
    status: str = "success"
    mode: str = "attach"
    endpoint: str | None = None
    model: str | None = None
    backend: str | None = None
    candidate_source: str | None = None
    telemetry_requested: str | None = None
    telemetry_provider: str | None = None
    candidate_count: int = 0
    valid_candidate_count: int = 0
    was_comparative: bool = False
    predicted_metrics: dict[str, float | int | str | None] = field(default_factory=dict)
    measured_metrics: dict[str, float | int | str | None] = field(default_factory=dict)
    telemetry_metrics: dict[str, Any] = field(default_factory=dict)
    comparison_metrics: dict[str, float | int | str | None] = field(default_factory=dict)
    score_weights: dict[str, float] = field(default_factory=dict)
    score_breakdown: dict[str, float | int | str | None] = field(default_factory=dict)
    ranked_candidates: list[dict[str, float | int | str | None]] = field(default_factory=list)
    pareto_frontier: list[dict[str, float | int | str | None]] = field(default_factory=list)
    alternative_recommendations: dict[str, dict[str, float | int | str | None]] = field(default_factory=dict)
    telemetry_used_in_scoring: bool = False
    power_aware: bool = False
    power_missing_reason: str | None = None
    confidence_level: str | None = None
    confidence_reasons: list[str] = field(default_factory=list)
    selection_reasons: list[str] = field(default_factory=list)
    metadata_notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: list[CheckRecord] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    candidate_table: list[dict[str, float | int | str | None]] = field(default_factory=list)
    alternatives: list[RecommendationScore] = field(default_factory=list)
    rationale: list[str] = field(default_factory=list)
    evaluated_set_fidelity: dict[str, Any] = field(default_factory=dict)
    optimizer_quality: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkResult:
    config: ServingConfig
    throughput_tok_s: float
    average_power_watts: float
    joules_per_token: float
    tokens_per_watt: float
    ttft_ms: float | None = None
    p95_latency_ms: float | None = None
    peak_power_watts: float | None = None
    total_energy_joules: float | None = None
    generated_tokens: int | None = None
    feasible: bool = True
    reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BackendStatus:
    name: str
    available: bool
    version: str | None = None
    command: str | None = None
    reason: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Recommendation:
    goal: str
    selected: BenchmarkResult | None
    frontier: list[BenchmarkResult]
    evaluated: list[BenchmarkResult]
    hardware: HardwareSnapshot
    model: ModelSpec
    notes: list[str] = field(default_factory=list)


def to_dict(value: Any) -> Any:
    """Convert nested dataclasses and enums into JSON-serializable objects."""

    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_dict(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_dict(item) for item in value]
    return value
