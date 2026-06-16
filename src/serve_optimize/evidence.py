"""Persistent measured evidence storage and lookup."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from .runtime_environment import build_runtime_evidence_fingerprint
from .schemas import (
    EndpointBenchmarkConfig,
    EndpointBenchmarkSummary,
    HardwareSnapshot,
    LaunchConfig,
    ModelSpec,
    ServingConfig,
    TelemetryCapabilities,
    TelemetrySummary,
    WorkloadConfig,
    to_dict,
)

DEFAULT_EVIDENCE_DB_PATH = Path("results/serve_optimize_evidence.sqlite")
EVIDENCE_SCHEMA_VERSION = 2


class EvidenceHitType(str, Enum):
    EXACT_FRESH_HIT = "EXACT_FRESH_HIT"
    EXACT_STALE_HIT = "EXACT_STALE_HIT"
    NEAR_COMPATIBLE_HIT = "NEAR_COMPATIBLE_HIT"
    PRIOR_ONLY_HIT = "PRIOR_ONLY_HIT"
    MISS = "MISS"


class EvidenceCompatibilityClassification(str, Enum):
    EXACT_FRESH = "exact_fresh"
    EXACT_STALE = "exact_stale"
    NEAR_COMPATIBLE = "near_compatible"
    RUNTIME_DRIFT = "runtime_drift"
    MISSING_RUNTIME_FINGERPRINT = "missing_runtime_fingerprint"
    INCOMPATIBLE = "incompatible"
    UNSUPPORTED_UNDER_CURRENT_BACKEND = "unsupported_under_current_backend"
    MISSING = "missing"


@dataclass(frozen=True)
class EvidenceRequestContext:
    hardware_fingerprint: str
    backend_fingerprint: str
    model_fingerprint: str
    telemetry_fingerprint: str
    launch_config_hash: str
    workload_config_hash: str
    runtime_fingerprint: str | None
    runtime_environment: dict[str, Any]
    evidence_key: str
    backend: str
    model: str
    goal: str


@dataclass(frozen=True)
class EvidenceLookupResult:
    hit_type: EvidenceHitType
    reason: str
    measurement: dict[str, Any] | None = None
    near_matches: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_exact_fresh_hit(self) -> bool:
        return self.hit_type == EvidenceHitType.EXACT_FRESH_HIT


@dataclass(frozen=True)
class EvidenceCompatibilityDecision:
    candidate_id: str
    evidence_key: str
    classification: EvidenceCompatibilityClassification
    used_as_exact: bool
    used_as_prior: bool
    freshness_status: str
    compatibility_reasons: list[str] = field(default_factory=list)
    rejection_reason: str | None = None
    source_evidence_metadata: dict[str, Any] = field(default_factory=dict)
    current_runtime_fingerprint: str | None = None

    def to_artifact(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "evidence_key": self.evidence_key,
            "classification": self.classification.value,
            "used_as_exact": self.used_as_exact,
            "used_as_prior": self.used_as_prior,
            "freshness_status": self.freshness_status,
            "compatibility_reasons": self.compatibility_reasons,
            "rejection_reason": self.rejection_reason,
            "source_evidence_metadata": self.source_evidence_metadata,
            "current_runtime_fingerprint": self.current_runtime_fingerprint,
        }


@dataclass(frozen=True)
class EvidenceRunRecord:
    run_id: str
    created_at: str
    command: str | None
    mode: str
    hardware_fingerprint: str
    backend_fingerprint: str
    model_fingerprint: str
    telemetry_fingerprint: str
    runtime_fingerprint: str | None = None
    runtime_environment_json: dict[str, Any] = field(default_factory=dict)
    output_dir: str | None = None
    metadata_json: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceMeasurementRecord:
    measurement_id: str
    run_id: str
    created_at: str
    evidence_key: str
    hardware_fingerprint: str
    backend_fingerprint: str
    model_fingerprint: str
    telemetry_fingerprint: str
    launch_config_hash: str
    workload_config_hash: str
    backend: str
    model: str
    goal: str
    runtime_fingerprint: str | None = None
    runtime_environment_json: dict[str, Any] = field(default_factory=dict)
    throughput_tokens_per_sec: float | None = None
    requests_per_sec: float | None = None
    p50_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    p99_latency_ms: float | None = None
    average_power_w: float | None = None
    peak_power_w: float | None = None
    joules_per_token: float | None = None
    tokens_per_watt: float | None = None
    total_energy_j: float | None = None
    stability_score: float | None = None
    confidence: str | None = None
    power_measurement_type: str = "unavailable"
    telemetry_source: str | None = None
    is_measured: bool = True
    is_synthetic: bool = False
    is_stable: bool | None = None
    raw_json: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceRecommendationRecord:
    recommendation_id: str
    run_id: str
    created_at: str
    goal: str
    evidence_key: str | None
    selected_measurement_id: str | None
    selected_config_id: str | None
    score: float | None
    confidence: str | None
    recommendation_json: dict[str, Any]


def canonical_json(value: object) -> str:
    return json.dumps(_canonical_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def hardware_fingerprint(hardware: HardwareSnapshot | dict[str, Any] | None) -> str:
    row = to_dict(hardware) if hardware is not None else {}
    gpus = row.get("gpus", []) if isinstance(row, dict) else []
    stable_gpus = []
    if isinstance(gpus, list):
        for gpu in gpus:
            if not isinstance(gpu, dict):
                continue
            stable_gpus.append(
                {
                    "compute_capability": gpu.get("compute_capability"),
                    "cuda_version": gpu.get("cuda_version"),
                    "driver_version": gpu.get("driver_version"),
                    "mig_mode": gpu.get("mig_mode"),
                    "mig_profile": gpu.get("mig_profile"),
                    "name": gpu.get("name"),
                    "total_memory_mb": gpu.get("total_memory_mb"),
                    "uuid": gpu.get("uuid"),
                }
            )
    stable_gpus = sorted(stable_gpus, key=lambda item: canonical_json(item))
    return stable_hash({"schema": "hardware-fingerprint/v1", "gpus": stable_gpus})


def backend_fingerprint(
    backend: str,
    *,
    version: str | None = None,
    adapter_name: str | None = None,
    adapter_version: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    metadata = dict(metadata or {})
    return stable_hash(
        {
            "schema": "backend-fingerprint/v1",
            "backend": backend,
            "version": version or metadata.get("version"),
            "adapter_name": adapter_name or metadata.get("adapter") or backend,
            "adapter_version": adapter_version or metadata.get("adapter_version"),
        }
    )


def model_fingerprint(model: ModelSpec | str | dict[str, Any]) -> str:
    if isinstance(model, ModelSpec):
        payload = {
            "model_id": model.model_id,
            "parameter_count_b": model.parameter_count_b,
            "max_context_tokens": model.max_context_tokens,
            "family": model.family,
        }
    elif isinstance(model, str):
        payload = {"model_id": model}
    else:
        payload = _without_unstable_fields(model)
    return stable_hash({"schema": "model-fingerprint/v1", "model": payload})


def telemetry_fingerprint(telemetry: str | TelemetrySummary | TelemetryCapabilities | dict[str, Any] | None) -> str:
    payload = _telemetry_payload(telemetry)
    return stable_hash({"schema": "telemetry-fingerprint/v1", "telemetry": payload})


def launch_config_hash(config: LaunchConfig | ServingConfig | dict[str, Any]) -> str:
    row = to_dict(config)
    if not isinstance(row, dict):
        row = {}
    payload = {
        "backend": row.get("backend"),
        "model": row.get("model_id") or row.get("model"),
        "dtype": row.get("dtype"),
        "quantization": row.get("quantization"),
        "max_model_len": row.get("max_context_tokens") or row.get("max_model_len"),
        "gpu_memory_utilization": row.get("gpu_memory_utilization"),
        "max_num_seqs": row.get("max_batch_size") or row.get("max_num_seqs"),
        "max_num_batched_tokens": row.get("max_num_batched_tokens"),
        "tensor_parallel_size": row.get("tensor_parallelism") or row.get("tensor_parallel_size"),
        "kv_cache_dtype": row.get("kv_cache_dtype"),
        "kv_cache_policy": row.get("kv_cache_policy"),
        "scheduler": row.get("scheduler"),
        "power_limit_watts": row.get("power_limit_watts"),
        "extra": _without_unstable_fields(row.get("extra", {})),
    }
    for key in (
        "block_size",
        "enforce_eager",
        "enable_chunked_prefill",
        "max_cudagraph_capture_size",
        "enable_prefix_caching",
    ):
        value = row.get(key)
        if value is not None:
            payload[key] = value
    return stable_hash({"schema": "launch-config/v1", "config": payload})


def workload_config_hash(config: WorkloadConfig | EndpointBenchmarkConfig | dict[str, Any], *, trials: int = 1, extra: dict[str, Any] | None = None) -> str:
    row = to_dict(config)
    if not isinstance(row, dict):
        row = {}
    prompt = str(row.get("prompt") or "")
    max_new_tokens = row.get("max_tokens") or row.get("max_new_tokens")
    extra_payload = {}
    if isinstance(row.get("extra"), dict):
        extra_payload.update(row.get("extra", {}))
    extra_payload.update(extra or {})
    workload_profile = extra_payload.get("workload_profile") if isinstance(extra_payload.get("workload_profile"), dict) else {}
    token_distribution = {}
    slo_constraints = {}
    workload_profile_name = None
    if isinstance(workload_profile, dict):
        workload_profile_name = workload_profile.get("profile_name")
        if isinstance(workload_profile.get("token_distribution"), dict):
            token_distribution = dict(workload_profile["token_distribution"])
        if isinstance(workload_profile.get("slo_constraints"), dict):
            slo_constraints = dict(workload_profile["slo_constraints"])
    payload = {
        "concurrency": row.get("concurrency"),
        "request_rate": row.get("request_rate"),
        "num_requests": row.get("num_requests"),
        "num_prompts": row.get("num_prompts") or row.get("num_requests"),
        "dataset": row.get("dataset"),
        "workload_profile_name": workload_profile_name,
        "token_distribution": token_distribution,
        "slo_constraints": slo_constraints,
        "benchmark_duration_s": row.get("benchmark_duration_s"),
        "warmup_duration_s": row.get("warmup_duration_s"),
        "warmup_requests": row.get("warmup_requests"),
        "idle_baseline_duration_s": row.get("idle_baseline_duration_s"),
        "idle_power_watts": row.get("idle_power_watts"),
        "trials": row.get("trials") or trials,
        "max_new_tokens": max_new_tokens,
        "input_length": row.get("expected_input_tokens") or row.get("input_length"),
        "output_length": row.get("expected_output_tokens") or row.get("output_length") or max_new_tokens,
        "endpoint": row.get("endpoint"),
        "timeout_s": row.get("timeout_s"),
        "prompt_hash": stable_hash({"prompt": prompt}) if prompt else None,
        "prompt_length_chars": len(prompt),
        "extra": _without_unstable_fields(extra_payload),
    }
    return stable_hash({"schema": "workload-config/v1", "workload": payload})


def evidence_key(
    *,
    hardware_fingerprint: str,
    backend_fingerprint: str,
    model_fingerprint: str,
    telemetry_fingerprint: str,
    launch_config_hash: str,
    workload_config_hash: str,
    runtime_fingerprint: str | None,
) -> str:
    return stable_hash(
        {
            "schema": "evidence-key/v2",
            "hardware_fingerprint": hardware_fingerprint,
            "backend_fingerprint": backend_fingerprint,
            "model_fingerprint": model_fingerprint,
            "telemetry_fingerprint": telemetry_fingerprint,
            "launch_config_hash": launch_config_hash,
            "workload_config_hash": workload_config_hash,
            "runtime_fingerprint": runtime_fingerprint or "missing",
        }
    )


def build_evidence_request_context(
    *,
    hardware: HardwareSnapshot | dict[str, Any] | None,
    backend: str,
    backend_metadata: dict[str, Any] | None,
    model: ModelSpec | str | dict[str, Any],
    telemetry: str | TelemetrySummary | TelemetryCapabilities | dict[str, Any] | None,
    launch_config: LaunchConfig | ServingConfig | dict[str, Any],
    workload_config: WorkloadConfig | EndpointBenchmarkConfig | dict[str, Any],
    goal: str,
    trials: int = 1,
    runtime_environment: dict[str, Any] | None = None,
    rendered_launch_command: list[str] | None = None,
    backend_capability_help_hash: str | None = None,
) -> EvidenceRequestContext:
    hardware_fp = hardware_fingerprint(hardware)
    backend_fp = backend_fingerprint(backend, metadata=backend_metadata)
    model_fp = model_fingerprint(model)
    telemetry_fp = telemetry_fingerprint(telemetry)
    launch_hash = launch_config_hash(launch_config)
    workload_hash = workload_config_hash(workload_config, trials=trials)
    runtime_payload: dict[str, Any] = {}
    runtime_fp = None
    if runtime_environment is not None and rendered_launch_command is not None:
        runtime_payload = build_runtime_evidence_fingerprint(
            runtime_environment,
            rendered_launch_command=rendered_launch_command,
            backend_capability_help_hash=backend_capability_help_hash,
            canonical_launch_config_identity=launch_hash,
            model_identity=model_fp,
            workload_identity=workload_hash,
        )
        runtime_fp = str(runtime_payload["fingerprint"])
    key = evidence_key(
        hardware_fingerprint=hardware_fp,
        backend_fingerprint=backend_fp,
        model_fingerprint=model_fp,
        telemetry_fingerprint=telemetry_fp,
        launch_config_hash=launch_hash,
        workload_config_hash=workload_hash,
        runtime_fingerprint=runtime_fp,
    )
    model_value = model.model_id if isinstance(model, ModelSpec) else str(model if not isinstance(model, dict) else model.get("model_id") or model.get("model") or "")
    return EvidenceRequestContext(
        hardware_fingerprint=hardware_fp,
        backend_fingerprint=backend_fp,
        model_fingerprint=model_fp,
        telemetry_fingerprint=telemetry_fp,
        launch_config_hash=launch_hash,
        workload_config_hash=workload_hash,
        runtime_fingerprint=runtime_fp,
        runtime_environment=runtime_payload,
        evidence_key=key,
        backend=backend,
        model=model_value,
        goal=goal,
    )


def measurement_from_summary(
    *,
    run_id: str,
    context: EvidenceRequestContext,
    summary: EndpointBenchmarkSummary | dict[str, Any],
    raw_json: dict[str, Any] | None = None,
) -> EvidenceMeasurementRecord:
    row = to_dict(summary)
    if not isinstance(row, dict):
        row = {}
    telemetry_summary = row.get("telemetry_summary") if isinstance(row.get("telemetry_summary"), dict) else {}
    power_type = "idle_subtracted" if row.get("active_energy_joules") is not None else ("measured" if row.get("average_power_watts") is not None else "unavailable")
    telemetry_source = row.get("telemetry_provider") or telemetry_summary.get("telemetry_provider")
    created_at = datetime.now(timezone.utc).isoformat()
    return EvidenceMeasurementRecord(
        measurement_id=f"meas-{uuid.uuid4().hex}",
        run_id=run_id,
        created_at=created_at,
        evidence_key=context.evidence_key,
        hardware_fingerprint=context.hardware_fingerprint,
        backend_fingerprint=context.backend_fingerprint,
        model_fingerprint=context.model_fingerprint,
        telemetry_fingerprint=context.telemetry_fingerprint,
        launch_config_hash=context.launch_config_hash,
        workload_config_hash=context.workload_config_hash,
        runtime_fingerprint=context.runtime_fingerprint,
        runtime_environment_json=context.runtime_environment,
        backend=context.backend,
        model=context.model,
        goal=context.goal,
        throughput_tokens_per_sec=_optional_float(row.get("total_tokens_s")),
        requests_per_sec=_optional_float(row.get("request_rate_req_s")),
        p50_latency_ms=_latency_ms(row.get("p50_latency_s")),
        p95_latency_ms=_latency_ms(row.get("p95_latency_s")),
        p99_latency_ms=_latency_ms(row.get("p99_latency_s")),
        average_power_w=_optional_float(row.get("average_power_watts")),
        peak_power_w=_optional_float(row.get("peak_power_watts")),
        joules_per_token=_optional_float(row.get("active_joules_per_token")) or _optional_float(row.get("joules_per_token")),
        tokens_per_watt=_optional_float(row.get("tokens_per_second_per_watt")),
        total_energy_j=_optional_float(row.get("active_energy_joules")) or _optional_float(row.get("energy_joules")),
        stability_score=_stability_score(row.get("stability_classification")),
        confidence=str(row.get("telemetry_quality")) if row.get("telemetry_quality") is not None else None,
        power_measurement_type=power_type,
        telemetry_source=str(telemetry_source) if telemetry_source is not None else None,
        is_measured=True,
        is_synthetic=False,
        is_stable=_is_stable(row.get("stability_classification")),
        raw_json=raw_json or row,
    )


class EvidenceStore:
    def __init__(self, path: Path, *, initialize: bool = True):
        self.path = Path(path)
        if initialize:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path))
        self.connection.row_factory = sqlite3.Row
        if initialize:
            self.initialize()

    def close(self) -> None:
        self.connection.close()

    def initialize(self) -> None:
        self.connection.executescript(SCHEMA_SQL)
        _ensure_column(
            self.connection,
            "evidence_runs",
            "runtime_fingerprint",
            "TEXT",
        )
        _ensure_column(
            self.connection,
            "evidence_runs",
            "runtime_environment_json",
            "TEXT",
        )
        _ensure_column(
            self.connection,
            "evidence_measurements",
            "runtime_fingerprint",
            "TEXT",
        )
        _ensure_column(
            self.connection,
            "evidence_measurements",
            "runtime_environment_json",
            "TEXT",
        )
        self.connection.commit()

    def insert_run(self, record: EvidenceRunRecord) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO evidence_runs (
                run_id, created_at, command, mode, hardware_fingerprint, backend_fingerprint,
                model_fingerprint, telemetry_fingerprint, runtime_fingerprint,
                runtime_environment_json, output_dir, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.run_id,
                record.created_at,
                record.command,
                record.mode,
                record.hardware_fingerprint,
                record.backend_fingerprint,
                record.model_fingerprint,
                record.telemetry_fingerprint,
                record.runtime_fingerprint,
                canonical_json(record.runtime_environment_json),
                record.output_dir,
                canonical_json(record.metadata_json),
            ),
        )
        self.connection.commit()

    def insert_measurement(self, record: EvidenceMeasurementRecord) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO evidence_measurements (
                measurement_id, run_id, created_at, evidence_key, hardware_fingerprint,
                backend_fingerprint, model_fingerprint, telemetry_fingerprint, launch_config_hash,
                workload_config_hash, runtime_fingerprint, runtime_environment_json,
                backend, model, goal, throughput_tokens_per_sec,
                requests_per_sec, p50_latency_ms, p95_latency_ms, p99_latency_ms,
                average_power_w, peak_power_w, joules_per_token, tokens_per_watt,
                total_energy_j, stability_score, confidence, power_measurement_type,
                telemetry_source, is_measured, is_synthetic, is_stable, raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.measurement_id,
                record.run_id,
                record.created_at,
                record.evidence_key,
                record.hardware_fingerprint,
                record.backend_fingerprint,
                record.model_fingerprint,
                record.telemetry_fingerprint,
                record.launch_config_hash,
                record.workload_config_hash,
                record.runtime_fingerprint,
                canonical_json(record.runtime_environment_json),
                record.backend,
                record.model,
                record.goal,
                record.throughput_tokens_per_sec,
                record.requests_per_sec,
                record.p50_latency_ms,
                record.p95_latency_ms,
                record.p99_latency_ms,
                record.average_power_w,
                record.peak_power_w,
                record.joules_per_token,
                record.tokens_per_watt,
                record.total_energy_j,
                record.stability_score,
                record.confidence,
                record.power_measurement_type,
                record.telemetry_source,
                _bool_to_int(record.is_measured),
                _bool_to_int(record.is_synthetic),
                _bool_to_int(record.is_stable),
                canonical_json(record.raw_json),
            ),
        )
        self.connection.commit()

    def insert_recommendation(self, record: EvidenceRecommendationRecord) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO evidence_recommendations (
                recommendation_id, run_id, created_at, goal, evidence_key, selected_measurement_id,
                selected_config_id, score, confidence, recommendation_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.recommendation_id,
                record.run_id,
                record.created_at,
                record.goal,
                record.evidence_key,
                record.selected_measurement_id,
                record.selected_config_id,
                record.score,
                record.confidence,
                canonical_json(record.recommendation_json),
            ),
        )
        self.connection.commit()

    def lookup_evidence(self, context: EvidenceRequestContext, *, freshness_hours: float) -> EvidenceLookupResult:
        exact_rows = self._fetch_measurements(
            """
            SELECT * FROM evidence_measurements
            WHERE evidence_key = ?
            ORDER BY created_at DESC
            """,
            (context.evidence_key,),
        )
        usable_exact = [_normalize_measurement_row(row) for row in exact_rows if _is_usable_exact(row)]
        if usable_exact:
            latest = usable_exact[0]
            if _is_fresh(str(latest["created_at"]), freshness_hours):
                return EvidenceLookupResult(EvidenceHitType.EXACT_FRESH_HIT, "Fresh measured evidence matched exactly.", latest)
            return EvidenceLookupResult(EvidenceHitType.EXACT_STALE_HIT, "Measured evidence matched exactly but is stale.", latest)
        if exact_rows:
            return EvidenceLookupResult(EvidenceHitType.PRIOR_ONLY_HIT, "Exact evidence exists but is not usable measured power evidence.", _normalize_measurement_row(exact_rows[0]))

        near_rows = self._fetch_measurements(
            """
            SELECT * FROM evidence_measurements
            WHERE hardware_fingerprint = ?
              AND backend_fingerprint = ?
              AND model_fingerprint = ?
              AND is_measured = 1
            ORDER BY created_at DESC
            LIMIT 5
            """,
            (context.hardware_fingerprint, context.backend_fingerprint, context.model_fingerprint),
        )
        if near_rows:
            return EvidenceLookupResult(
                EvidenceHitType.NEAR_COMPATIBLE_HIT,
                "Measured evidence exists for the same hardware, backend, and model, but launch or workload config differs.",
                None,
                [_normalize_measurement_row(row) for row in near_rows],
            )

        prior_rows = self._fetch_measurements(
            """
            SELECT * FROM evidence_measurements
            WHERE backend = ?
              AND model = ?
            ORDER BY created_at DESC
            LIMIT 5
            """,
            (context.backend, context.model),
        )
        if prior_rows:
            return EvidenceLookupResult(
                EvidenceHitType.PRIOR_ONLY_HIT,
                "Evidence exists for the same backend and model, but core fingerprints differ.",
                _normalize_measurement_row(prior_rows[0]),
                [_normalize_measurement_row(row) for row in prior_rows],
            )
        return EvidenceLookupResult(EvidenceHitType.MISS, "No matching evidence was found.")

    def list_measurements(self, *, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._fetch_measurements(
            """
            SELECT * FROM evidence_measurements
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [_normalize_measurement_row(row) for row in rows]

    def _fetch_measurements(self, query: str, params: tuple[object, ...]) -> list[sqlite3.Row]:
        return list(self.connection.execute(query, params))


def initialize_evidence_db(path: Path) -> None:
    store = EvidenceStore(path)
    store.close()


def list_evidence_measurements(path: Path, *, limit: int = 20) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    store = EvidenceStore(path, initialize=False)
    try:
        return store.list_measurements(limit=limit)
    finally:
        store.close()


def lookup_evidence(path: Path, context: EvidenceRequestContext, *, freshness_hours: float) -> EvidenceLookupResult:
    if not Path(path).exists():
        return EvidenceLookupResult(EvidenceHitType.MISS, "Evidence database does not exist.")
    store = EvidenceStore(path, initialize=False)
    try:
        return store.lookup_evidence(context, freshness_hours=freshness_hours)
    finally:
        store.close()


def classify_evidence_lookup(
    lookup: EvidenceLookupResult,
    *,
    candidate_id: str,
    context: EvidenceRequestContext,
    current_backend_argument_capabilities: Any | None = None,
    goal: str | None = None,
) -> EvidenceCompatibilityDecision:
    measurement = lookup.measurement or (lookup.near_matches[0] if lookup.near_matches else {})
    source_metadata = _source_evidence_metadata(measurement)
    runtime_rejection = _runtime_compatibility_rejection(
        context,
        measurement,
    )
    if runtime_rejection is not None:
        classification, rejection_reason = runtime_rejection
        return EvidenceCompatibilityDecision(
            candidate_id=candidate_id,
            evidence_key=context.evidence_key,
            classification=classification,
            used_as_exact=False,
            used_as_prior=bool(measurement),
            freshness_status=_freshness_status(lookup),
            compatibility_reasons=[lookup.reason],
            rejection_reason=rejection_reason,
            source_evidence_metadata=source_metadata,
            current_runtime_fingerprint=context.runtime_fingerprint,
        )
    unsupported = _unsupported_under_current_backend(measurement, current_backend_argument_capabilities)
    if unsupported:
        return EvidenceCompatibilityDecision(
            candidate_id=candidate_id,
            evidence_key=context.evidence_key,
            classification=EvidenceCompatibilityClassification.UNSUPPORTED_UNDER_CURRENT_BACKEND,
            used_as_exact=False,
            used_as_prior=lookup.hit_type in {EvidenceHitType.EXACT_STALE_HIT, EvidenceHitType.NEAR_COMPATIBLE_HIT, EvidenceHitType.PRIOR_ONLY_HIT},
            freshness_status=_freshness_status(lookup),
            compatibility_reasons=[lookup.reason],
            rejection_reason=unsupported,
            source_evidence_metadata=source_metadata,
            current_runtime_fingerprint=context.runtime_fingerprint,
        )
    telemetry_rejection = _telemetry_rejection_for_goal(measurement, goal)
    if lookup.hit_type == EvidenceHitType.EXACT_FRESH_HIT and telemetry_rejection is None:
        return EvidenceCompatibilityDecision(
            candidate_id=candidate_id,
            evidence_key=context.evidence_key,
            classification=EvidenceCompatibilityClassification.EXACT_FRESH,
            used_as_exact=True,
            used_as_prior=False,
            freshness_status="fresh",
            compatibility_reasons=[
                lookup.reason,
                "Canonical launch config, workload, and runtime fingerprint matched.",
            ],
            source_evidence_metadata=source_metadata,
            current_runtime_fingerprint=context.runtime_fingerprint,
        )
    if lookup.hit_type == EvidenceHitType.EXACT_FRESH_HIT:
        return EvidenceCompatibilityDecision(
            candidate_id=candidate_id,
            evidence_key=context.evidence_key,
            classification=EvidenceCompatibilityClassification.INCOMPATIBLE,
            used_as_exact=False,
            used_as_prior=False,
            freshness_status="fresh",
            compatibility_reasons=[lookup.reason],
            rejection_reason=telemetry_rejection,
            source_evidence_metadata=source_metadata,
            current_runtime_fingerprint=context.runtime_fingerprint,
        )
    if lookup.hit_type == EvidenceHitType.EXACT_STALE_HIT:
        return EvidenceCompatibilityDecision(
            candidate_id=candidate_id,
            evidence_key=context.evidence_key,
            classification=EvidenceCompatibilityClassification.EXACT_STALE,
            used_as_exact=False,
            used_as_prior=True,
            freshness_status="stale",
            compatibility_reasons=[lookup.reason],
            source_evidence_metadata=source_metadata,
            current_runtime_fingerprint=context.runtime_fingerprint,
        )
    if lookup.hit_type == EvidenceHitType.NEAR_COMPATIBLE_HIT:
        return EvidenceCompatibilityDecision(
            candidate_id=candidate_id,
            evidence_key=context.evidence_key,
            classification=EvidenceCompatibilityClassification.NEAR_COMPATIBLE,
            used_as_exact=False,
            used_as_prior=True,
            freshness_status="unknown",
            compatibility_reasons=[lookup.reason],
            source_evidence_metadata=source_metadata,
            current_runtime_fingerprint=context.runtime_fingerprint,
        )
    if lookup.hit_type == EvidenceHitType.MISS:
        return EvidenceCompatibilityDecision(
            candidate_id=candidate_id,
            evidence_key=context.evidence_key,
            classification=EvidenceCompatibilityClassification.MISSING,
            used_as_exact=False,
            used_as_prior=False,
            freshness_status="missing",
            compatibility_reasons=[lookup.reason],
            source_evidence_metadata={},
            current_runtime_fingerprint=context.runtime_fingerprint,
        )
    return EvidenceCompatibilityDecision(
        candidate_id=candidate_id,
        evidence_key=context.evidence_key,
        classification=EvidenceCompatibilityClassification.INCOMPATIBLE,
        used_as_exact=False,
        used_as_prior=False,
        freshness_status=_freshness_status(lookup),
        compatibility_reasons=[lookup.reason],
        rejection_reason=telemetry_rejection or "Evidence does not satisfy exact or near compatibility policy.",
        source_evidence_metadata=source_metadata,
        current_runtime_fingerprint=context.runtime_fingerprint,
    )


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS evidence_runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    command TEXT,
    mode TEXT NOT NULL,
    hardware_fingerprint TEXT NOT NULL,
    backend_fingerprint TEXT NOT NULL,
    model_fingerprint TEXT NOT NULL,
    telemetry_fingerprint TEXT NOT NULL,
    runtime_fingerprint TEXT,
    runtime_environment_json TEXT,
    output_dir TEXT,
    metadata_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evidence_measurements (
    measurement_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    evidence_key TEXT NOT NULL,
    hardware_fingerprint TEXT NOT NULL,
    backend_fingerprint TEXT NOT NULL,
    model_fingerprint TEXT NOT NULL,
    telemetry_fingerprint TEXT NOT NULL,
    launch_config_hash TEXT NOT NULL,
    workload_config_hash TEXT NOT NULL,
    runtime_fingerprint TEXT,
    runtime_environment_json TEXT,
    backend TEXT NOT NULL,
    model TEXT NOT NULL,
    goal TEXT NOT NULL,
    throughput_tokens_per_sec REAL,
    requests_per_sec REAL,
    p50_latency_ms REAL,
    p95_latency_ms REAL,
    p99_latency_ms REAL,
    average_power_w REAL,
    peak_power_w REAL,
    joules_per_token REAL,
    tokens_per_watt REAL,
    total_energy_j REAL,
    stability_score REAL,
    confidence TEXT,
    power_measurement_type TEXT,
    telemetry_source TEXT,
    is_measured INTEGER NOT NULL DEFAULT 1,
    is_synthetic INTEGER NOT NULL DEFAULT 0,
    is_stable INTEGER,
    raw_json TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES evidence_runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_evidence_measurements_key_created
ON evidence_measurements(evidence_key, created_at);

CREATE INDEX IF NOT EXISTS idx_evidence_measurements_core
ON evidence_measurements(hardware_fingerprint, backend_fingerprint, model_fingerprint);

CREATE TABLE IF NOT EXISTS evidence_recommendations (
    recommendation_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    goal TEXT NOT NULL,
    evidence_key TEXT,
    selected_measurement_id TEXT,
    selected_config_id TEXT,
    score REAL,
    confidence TEXT,
    recommendation_json TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES evidence_runs(run_id)
);

CREATE TABLE IF NOT EXISTS evidence_drift_checks (
    drift_check_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    original_measurement_id TEXT NOT NULL,
    new_measurement_id TEXT NOT NULL,
    throughput_delta REAL,
    latency_delta REAL,
    joules_per_token_delta REAL,
    tokens_per_watt_delta REAL,
    passed INTEGER,
    raw_json TEXT NOT NULL
);
"""


def _ensure_column(
    connection: sqlite3.Connection,
    table: str,
    column: str,
    declaration: str,
) -> None:
    columns = {
        str(row[1])
        for row in connection.execute(f"PRAGMA table_info({table})")
    }
    if column not in columns:
        connection.execute(
            f"ALTER TABLE {table} ADD COLUMN {column} {declaration}"
        )


def _telemetry_payload(telemetry: str | TelemetrySummary | TelemetryCapabilities | dict[str, Any] | None) -> dict[str, Any]:
    if telemetry is None:
        row: dict[str, Any] = {}
    elif isinstance(telemetry, str):
        row = {"provider": telemetry, "requested_provider": telemetry}
    else:
        row = to_dict(telemetry)
        if not isinstance(row, dict):
            row = {}
    capabilities = row.get("telemetry_capabilities") if isinstance(row.get("telemetry_capabilities"), dict) else row
    available = set(capabilities.get("available_fields", [])) if isinstance(capabilities, dict) else set()
    unavailable = set(capabilities.get("unavailable_fields", [])) if isinstance(capabilities, dict) else set()
    return {
        "provider": row.get("provider") or row.get("telemetry_provider") or row.get("requested_provider"),
        "requested_provider": row.get("requested_provider"),
        "power_available": _field_available("power", available, unavailable, row),
        "temperature_available": _field_available("temperature", available, unavailable, row),
        "memory_usage_available": _field_available("memory_usage", available, unavailable, row),
        "clocks_available": _field_available("clocks", available, unavailable, row),
        "power_limit_available": _field_available("power_limit", available, unavailable, row),
        "gpu_utilization_available": _field_available("gpu_utilization", available, unavailable, row),
        "memory_utilization_available": _field_available("memory_utilization", available, unavailable, row),
        "power_measurement_scope": row.get("power_measurement_scope") or row.get("power_measurement_type"),
    }


def _field_available(field_name: str, available: set[str], unavailable: set[str], row: dict[str, Any]) -> bool | None:
    if field_name in available:
        return True
    if field_name in unavailable:
        return False
    direct = row.get(f"{field_name}_available")
    return bool(direct) if direct is not None else None


def _canonical_value(value: object) -> object:
    value = _without_unstable_fields(to_dict(value))
    if isinstance(value, dict):
        return {str(key): _canonical_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, list):
        return [_canonical_value(item) for item in value]
    if isinstance(value, tuple):
        return [_canonical_value(item) for item in value]
    return value


def _without_unstable_fields(value: object) -> object:
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text in UNSTABLE_KEYS:
                continue
            cleaned[key_text] = _without_unstable_fields(item)
        return cleaned
    if isinstance(value, list):
        return [_without_unstable_fields(item) for item in value]
    return value


UNSTABLE_KEYS = {
    "created_at",
    "detected_at",
    "ended_at",
    "start_time",
    "started_at",
    "stdout_log_path",
    "stderr_log_path",
    "timestamp",
    "timestamp_s",
    "run_id",
    "run_dir",
    "output_dir",
    "output_path",
    "pid",
    "pgid",
}


def _bool_to_int(value: bool | None) -> int | None:
    if value is None:
        return None
    return 1 if value else 0


def _is_usable_exact(row: sqlite3.Row) -> bool:
    runtime_fingerprint = (
        row["runtime_fingerprint"]
        if "runtime_fingerprint" in row.keys()
        else None
    )
    return (
        bool(row["is_measured"])
        and row["power_measurement_type"] != "unavailable"
        and bool(runtime_fingerprint)
    )


def _is_fresh(created_at: str, freshness_hours: float) -> bool:
    try:
        created = datetime.fromisoformat(created_at)
    except ValueError:
        return False
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - created <= timedelta(hours=freshness_hours)


def _normalize_measurement_row(row: sqlite3.Row) -> dict[str, Any]:
    payload = dict(row)
    for key in ("is_measured", "is_synthetic", "is_stable"):
        if payload.get(key) is not None:
            payload[key] = bool(payload[key])
    for key in ("raw_json", "runtime_environment_json"):
        value = payload.get(key)
        if isinstance(value, str):
            try:
                payload[key] = json.loads(value)
            except json.JSONDecodeError:
                payload[key] = {}
    return payload


def _source_evidence_metadata(measurement: dict[str, Any]) -> dict[str, Any]:
    if not measurement:
        return {}
    return {
        "measurement_id": measurement.get("measurement_id"),
        "created_at": measurement.get("created_at"),
        "backend": measurement.get("backend"),
        "model": measurement.get("model"),
        "goal": measurement.get("goal"),
        "power_measurement_type": measurement.get("power_measurement_type"),
        "telemetry_source": measurement.get("telemetry_source"),
        "confidence": measurement.get("confidence"),
        "launch_config_hash": measurement.get("launch_config_hash"),
        "workload_config_hash": measurement.get("workload_config_hash"),
        "runtime_fingerprint": measurement.get("runtime_fingerprint"),
        "runtime_environment": measurement.get("runtime_environment_json"),
    }


def _runtime_compatibility_rejection(
    context: EvidenceRequestContext,
    measurement: dict[str, Any],
) -> tuple[EvidenceCompatibilityClassification, str] | None:
    if not measurement:
        return None
    source_fingerprint = measurement.get("runtime_fingerprint")
    if not source_fingerprint:
        return (
            EvidenceCompatibilityClassification.MISSING_RUNTIME_FINGERPRINT,
            "Stored evidence has no runtime fingerprint and is not exact compatible.",
        )
    if not context.runtime_fingerprint:
        return (
            EvidenceCompatibilityClassification.MISSING_RUNTIME_FINGERPRINT,
            "Current evidence request has no runtime fingerprint and cannot reuse exact evidence.",
        )
    source_runtime = measurement.get("runtime_environment_json")
    current_runtime = context.runtime_environment
    if not isinstance(source_runtime, dict) or not source_runtime:
        return (
            EvidenceCompatibilityClassification.MISSING_RUNTIME_FINGERPRINT,
            "Stored evidence has no structured runtime fingerprint payload.",
        )
    compatibility_fields = (
        "rendered_launch_command_hash",
        "backend_capability_help_hash",
        "canonical_launch_config_identity",
    )
    source_environment = source_runtime.get("runtime_environment")
    current_environment = current_runtime.get("runtime_environment")
    source_environment_fingerprint = (
        source_environment.get("environment_fingerprint")
        if isinstance(source_environment, dict)
        else None
    )
    current_environment_fingerprint = (
        current_environment.get("environment_fingerprint")
        if isinstance(current_environment, dict)
        else None
    )
    runtime_changed = (
        source_environment_fingerprint != current_environment_fingerprint
        or any(
            source_runtime.get(field) != current_runtime.get(field)
            for field in compatibility_fields
        )
    )
    if runtime_changed:
        return (
            EvidenceCompatibilityClassification.RUNTIME_DRIFT,
            "Stored evidence runtime fingerprint differs from the current runtime.",
        )
    return None


def _freshness_status(lookup: EvidenceLookupResult) -> str:
    if lookup.hit_type == EvidenceHitType.EXACT_FRESH_HIT:
        return "fresh"
    if lookup.hit_type == EvidenceHitType.EXACT_STALE_HIT:
        return "stale"
    if lookup.hit_type == EvidenceHitType.MISS:
        return "missing"
    return "unknown"


def _telemetry_rejection_for_goal(measurement: dict[str, Any], goal: str | None) -> str | None:
    if str(goal or "").lower() not in {"balanced", "efficient", "efficiency"}:
        return None
    if not measurement:
        return "No measurement metadata was available for power aware evidence reuse."
    if measurement.get("power_measurement_type") == "unavailable":
        return "Power telemetry is unavailable for a power aware goal."
    confidence = str(measurement.get("confidence") or "").lower()
    if confidence in {"poor", "unavailable", "failed"}:
        return f"Telemetry quality '{confidence}' is not acceptable for exact power aware reuse."
    return None


def _unsupported_under_current_backend(measurement: dict[str, Any], capabilities: Any | None) -> str | None:
    if not measurement or capabilities is None or getattr(capabilities, "detection_status", None) != "success":
        return None
    raw_json = measurement.get("raw_json") if isinstance(measurement.get("raw_json"), dict) else {}
    candidate = raw_json.get("candidate") if isinstance(raw_json.get("candidate"), dict) else {}
    launch_group = raw_json.get("launch_group") if isinstance(raw_json.get("launch_group"), dict) else {}
    launch_config = launch_group.get("launch_config") if isinstance(launch_group.get("launch_config"), dict) else {}
    merged = {**launch_config, **candidate}
    required_flags = {
        "block_size": "--block-size",
        "kv_cache_dtype": "--kv-cache-dtype",
        "enforce_eager": "--enforce-eager",
        "max_num_batched_tokens": "--max-num-batched-tokens",
        "enable_prefix_caching": "--enable-prefix-caching",
    }
    for field_name, flag in required_flags.items():
        value = merged.get(field_name)
        if value is not None and value is not False and not capabilities.supports(flag):
            return f"Stored evidence uses {field_name}, but current vLLM cannot render {flag}."
    if merged.get("enable_chunked_prefill") is True and not capabilities.supports("--enable-chunked-prefill"):
        return "Stored evidence uses enable_chunked_prefill=true, but current vLLM cannot render --enable-chunked-prefill."
    if merged.get("enable_chunked_prefill") is False and not capabilities.supports("--no-enable-chunked-prefill"):
        return "Stored evidence uses enable_chunked_prefill=false, but current vLLM cannot render --no-enable-chunked-prefill."
    if merged.get("max_cudagraph_capture_size") is not None and capabilities.cudagraph_capture_flag() is None:
        return "Stored evidence uses max_cudagraph_capture_size, but current vLLM cannot render a CUDA graph capture size flag."
    return None


def _optional_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _latency_ms(value: object) -> float | None:
    latency_s = _optional_float(value)
    return latency_s * 1000.0 if latency_s is not None else None


def _stability_score(value: object) -> float | None:
    text = str(value or "")
    if text == "stable":
        return 1.0
    if text == "mostly_stable":
        return 0.75
    if text == "single_trial":
        return 0.5
    if text == "unstable":
        return 0.0
    return None


def _is_stable(value: object) -> bool | None:
    text = str(value or "")
    if text in {"stable", "mostly_stable"}:
        return True
    if text == "unstable":
        return False
    return None
