import sqlite3
from datetime import datetime, timedelta, timezone

from serve_optimize.backends.vllm import VLLMArgumentCapabilities
from serve_optimize.evidence import (
    EvidenceCompatibilityClassification,
    EvidenceHitType,
    EvidenceMeasurementRecord,
    EvidenceStore,
    build_evidence_request_context,
    classify_evidence_lookup,
    hardware_fingerprint,
    initialize_evidence_db,
    launch_config_hash,
    measurement_from_summary,
    model_fingerprint,
    telemetry_fingerprint,
    workload_config_hash,
)
from serve_optimize.schemas import (
    EndpointBenchmarkConfig,
    EndpointBenchmarkSummary,
    Goal,
    GpuDevice,
    HardwareSnapshot,
    ServingConfig,
)


def test_fingerprint_stability() -> None:
    hardware = _hardware(detected_at="2026-01-01T00:00:00+00:00")

    assert hardware_fingerprint(hardware) == hardware_fingerprint(hardware)
    assert model_fingerprint("model-path") == model_fingerprint("model-path")
    assert telemetry_fingerprint({"provider": "nvml", "available_fields": ["power"]}) == telemetry_fingerprint(
        {"available_fields": ["power"], "provider": "nvml"}
    )


def test_fingerprint_ignores_timestamps_and_output_paths() -> None:
    first = _hardware(detected_at="2026-01-01T00:00:00+00:00")
    second = _hardware(detected_at="2026-02-01T00:00:00+00:00")

    assert hardware_fingerprint(first) == hardware_fingerprint(second)
    assert model_fingerprint({"model_id": "model-path", "output_path": "/tmp/one"}) == model_fingerprint(
        {"model_id": "model-path", "output_path": "/tmp/two"}
    )


def test_runtime_fingerprint_contains_required_identity_fields() -> None:
    context = _context()
    payload = context.runtime_environment

    assert payload["fingerprint"] == context.runtime_fingerprint
    assert payload["rendered_launch_command_hash"]
    assert payload["backend_capability_help_hash"] == "help-a"
    assert payload["canonical_launch_config_identity"] == context.launch_config_hash
    assert payload["model_identity"] == context.model_fingerprint
    assert payload["workload_identity"] == context.workload_config_hash
    environment = payload["runtime_environment"]
    assert environment["backend_name"] == "vllm"
    assert environment["backend_version"] == "0.0"
    assert environment["torch_version"] == "2.7.1"
    assert environment["cuda_runtime_version"] == "12.6"
    assert environment["python_version"] == "3.12.0"
    assert environment["compiler_toolchain_fingerprint"] == "compiler-a"
    assert environment["serve_optimize_git_commit"] == "commit-a"


def test_launch_config_hash_differs_when_launch_field_changes() -> None:
    assert launch_config_hash(_config(dtype="fp16")) != launch_config_hash(_config(dtype="bf16"))


def test_launch_config_hash_differs_for_max_num_batched_tokens() -> None:
    assert launch_config_hash(_config(max_num_batched_tokens=4096)) != launch_config_hash(_config(max_num_batched_tokens=8192))


def test_launch_config_hash_differs_for_block_size() -> None:
    assert launch_config_hash(_config(block_size=16)) != launch_config_hash(_config(block_size=32))


def test_launch_config_hash_differs_for_backend() -> None:
    assert launch_config_hash({"backend": "vllm", "model": "model-path", "dtype": "fp16"}) != launch_config_hash(
        {"backend": "sglang", "model": "model-path", "dtype": "fp16"}
    )


def test_old_style_launch_config_hash_still_works() -> None:
    assert isinstance(launch_config_hash({"backend": "vllm", "model": "model-path", "dtype": "fp16"}), str)


def test_workload_config_hash_differs_when_workload_field_changes() -> None:
    assert workload_config_hash(_workload(concurrency=1)) != workload_config_hash(_workload(concurrency=2))


def test_workload_config_hash_differs_when_token_distribution_changes() -> None:
    first = {
        "concurrency": 1,
        "num_requests": 4,
        "max_tokens": 8,
        "prompt": "hello",
        "timeout_s": 30.0,
        "extra": {
            "workload_profile": {
                "profile_name": "short",
                "token_distribution": {"input_tokens": {"p50": 128}},
            }
        },
    }
    second = {
        "concurrency": 1,
        "num_requests": 4,
        "max_tokens": 8,
        "prompt": "hello",
        "timeout_s": 30.0,
        "extra": {
            "workload_profile": {
                "profile_name": "short",
                "token_distribution": {"input_tokens": {"p50": 256}},
            }
        },
    }

    assert workload_config_hash(first) != workload_config_hash(second)


def test_sqlite_db_initialization_creates_expected_tables(tmp_path) -> None:
    db_path = tmp_path / "evidence.sqlite"

    initialize_evidence_db(db_path)

    with sqlite3.connect(db_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert {"evidence_runs", "evidence_measurements", "evidence_recommendations", "evidence_drift_checks"} <= tables


def test_insert_and_read_measurement_evidence(tmp_path) -> None:
    store = EvidenceStore(tmp_path / "evidence.sqlite")
    context = _context()
    measurement = _measurement(context)

    store.insert_measurement(measurement)
    rows = store.list_measurements(limit=5)
    store.close()

    assert len(rows) == 1
    assert rows[0]["measurement_id"] == measurement.measurement_id
    assert rows[0]["throughput_tokens_per_sec"] == 100.0


def test_measurement_from_summary_uses_idle_subtracted_energy_and_stability() -> None:
    context = _context()
    summary = EndpointBenchmarkSummary(
        run_id="run-idle",
        total_requests=2,
        successful_requests=2,
        failed_requests=0,
        wall_time_s=2.0,
        request_rate_req_s=1.0,
        prompt_tokens=20,
        completion_tokens=40,
        total_tokens=60,
        output_tokens_s=20.0,
        total_tokens_s=30.0,
        avg_latency_s=1.0,
        p50_latency_s=1.0,
        p95_latency_s=1.0,
        p99_latency_s=1.0,
        average_power_watts=120.0,
        energy_joules=240.0,
        joules_per_token=4.0,
        active_energy_joules=40.0,
        active_joules_per_token=40.0 / 60.0,
        telemetry_provider="nvml",
        telemetry_quality="good",
        stability_classification="stable",
    )

    measurement = measurement_from_summary(run_id="run-test", context=context, summary=summary)

    assert measurement.power_measurement_type == "idle_subtracted"
    assert measurement.total_energy_j == 40.0
    assert measurement.joules_per_token == 40.0 / 60.0
    assert measurement.stability_score == 1.0
    assert measurement.is_stable is True


def test_exact_fresh_hit_classification(tmp_path) -> None:
    store = EvidenceStore(tmp_path / "evidence.sqlite")
    context = _context()
    store.insert_measurement(_measurement(context))

    result = store.lookup_evidence(context, freshness_hours=24.0)
    store.close()

    assert result.hit_type == EvidenceHitType.EXACT_FRESH_HIT
    assert result.measurement is not None


def test_exact_fresh_decision_is_reused_as_exact(tmp_path) -> None:
    store = EvidenceStore(tmp_path / "evidence.sqlite")
    context = _context()
    store.insert_measurement(_measurement(context))

    lookup = store.lookup_evidence(context, freshness_hours=24.0)
    decision = classify_evidence_lookup(lookup, candidate_id="cfg-test", context=context, goal=Goal.BALANCED.value)
    store.close()

    assert decision.classification == EvidenceCompatibilityClassification.EXACT_FRESH
    assert decision.used_as_exact is True
    assert decision.used_as_prior is False


def test_exact_stale_hit_classification(tmp_path) -> None:
    store = EvidenceStore(tmp_path / "evidence.sqlite")
    context = _context()
    created_at = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    store.insert_measurement(_measurement(context, created_at=created_at))

    result = store.lookup_evidence(context, freshness_hours=24.0)
    store.close()

    assert result.hit_type == EvidenceHitType.EXACT_STALE_HIT


def test_exact_stale_decision_is_prior_only(tmp_path) -> None:
    store = EvidenceStore(tmp_path / "evidence.sqlite")
    context = _context()
    created_at = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    store.insert_measurement(_measurement(context, created_at=created_at))

    lookup = store.lookup_evidence(context, freshness_hours=24.0)
    decision = classify_evidence_lookup(lookup, candidate_id="cfg-test", context=context, goal=Goal.BALANCED.value)
    store.close()

    assert decision.classification == EvidenceCompatibilityClassification.EXACT_STALE
    assert decision.used_as_exact is False
    assert decision.used_as_prior is True


def test_near_compatible_decision_is_prior_only(tmp_path) -> None:
    store = EvidenceStore(tmp_path / "evidence.sqlite")
    old_context = _context(workload_config=_workload(concurrency=1))
    current_context = _context(workload_config=_workload(concurrency=2))
    store.insert_measurement(_measurement(old_context))

    lookup = store.lookup_evidence(current_context, freshness_hours=24.0)
    decision = classify_evidence_lookup(lookup, candidate_id="cfg-test", context=current_context, goal=Goal.BALANCED.value)
    store.close()

    assert lookup.hit_type == EvidenceHitType.NEAR_COMPATIBLE_HIT
    assert decision.classification == EvidenceCompatibilityClassification.NEAR_COMPATIBLE
    assert decision.used_as_exact is False
    assert decision.used_as_prior is True


def test_unsupported_backend_decision_is_not_exact(tmp_path) -> None:
    store = EvidenceStore(tmp_path / "evidence.sqlite")
    context = _context(launch_config=_config(block_size=16))
    store.insert_measurement(_measurement(context, raw_json={"candidate": {"block_size": 16}}))

    lookup = store.lookup_evidence(context, freshness_hours=24.0)
    decision = classify_evidence_lookup(
        lookup,
        candidate_id="cfg-test",
        context=context,
        current_backend_argument_capabilities=_caps(),
        goal=Goal.BALANCED.value,
    )
    store.close()

    assert decision.classification == EvidenceCompatibilityClassification.UNSUPPORTED_UNDER_CURRENT_BACKEND
    assert decision.used_as_exact is False
    assert "block_size" in str(decision.rejection_reason)


def test_poor_telemetry_prevents_power_goal_exact_reuse(tmp_path) -> None:
    store = EvidenceStore(tmp_path / "evidence.sqlite")
    context = _context()
    store.insert_measurement(_measurement(context, confidence="poor"))

    lookup = store.lookup_evidence(context, freshness_hours=24.0)
    decision = classify_evidence_lookup(lookup, candidate_id="cfg-test", context=context, goal=Goal.EFFICIENT.value)
    store.close()

    assert lookup.hit_type == EvidenceHitType.EXACT_FRESH_HIT
    assert decision.classification == EvidenceCompatibilityClassification.INCOMPATIBLE
    assert decision.used_as_exact is False


def test_miss_classification(tmp_path) -> None:
    store = EvidenceStore(tmp_path / "evidence.sqlite")

    result = store.lookup_evidence(_context(), freshness_hours=24.0)
    store.close()

    assert result.hit_type == EvidenceHitType.MISS


def test_old_unrenderable_launch_field_is_not_exact_fresh(tmp_path) -> None:
    store = EvidenceStore(tmp_path / "evidence.sqlite")
    old_context = _context(launch_config=_config(max_cudagraph_capture_size=32))
    current_context = _context(launch_config=_config())
    store.insert_measurement(_measurement(old_context))

    result = store.lookup_evidence(current_context, freshness_hours=24.0)
    store.close()

    assert result.hit_type != EvidenceHitType.EXACT_FRESH_HIT


def test_backend_version_change_prevents_exact_reuse(tmp_path) -> None:
    decision = _runtime_drift_decision(
        tmp_path,
        _context(backend_version="0.10.0"),
        _context(backend_version="0.23.0"),
    )

    assert decision.classification == EvidenceCompatibilityClassification.RUNTIME_DRIFT
    assert decision.used_as_exact is False


def test_torch_version_change_prevents_exact_reuse(tmp_path) -> None:
    decision = _runtime_drift_decision(
        tmp_path,
        _context(torch_version="2.7.1"),
        _context(torch_version="2.9.1"),
    )

    assert decision.classification == EvidenceCompatibilityClassification.RUNTIME_DRIFT


def test_sglang_compiler_change_prevents_exact_reuse(tmp_path) -> None:
    decision = _runtime_drift_decision(
        tmp_path,
        _context(
            backend="sglang",
            compiler_fingerprint="gcc-12",
        ),
        _context(
            backend="sglang",
            compiler_fingerprint="gcc-13",
        ),
    )

    assert decision.classification == EvidenceCompatibilityClassification.RUNTIME_DRIFT


def test_launch_command_change_prevents_exact_reuse(tmp_path) -> None:
    decision = _runtime_drift_decision(
        tmp_path,
        _context(rendered_launch_command=["vllm", "serve", "model-path"]),
        _context(
            rendered_launch_command=[
                "vllm",
                "serve",
                "model-path",
                "--enforce-eager",
            ]
        ),
    )

    assert decision.classification == EvidenceCompatibilityClassification.RUNTIME_DRIFT


def test_sglang_piecewise_cuda_graph_flag_change_prevents_exact_reuse(tmp_path) -> None:
    base_command = ["python", "-m", "sglang.launch_server", "--model-path", "model-path"]
    decision = _runtime_drift_decision(
        tmp_path,
        _context(
            backend="sglang",
            rendered_launch_command=[
                *base_command,
                "--disable-piecewise-cuda-graph",
            ],
        ),
        _context(
            backend="sglang",
            rendered_launch_command=base_command,
        ),
    )

    assert decision.classification == EvidenceCompatibilityClassification.RUNTIME_DRIFT
    assert decision.used_as_exact is False


def test_backend_help_hash_change_prevents_exact_reuse(tmp_path) -> None:
    decision = _runtime_drift_decision(
        tmp_path,
        _context(help_hash="help-a"),
        _context(help_hash="help-b"),
    )

    assert decision.classification == EvidenceCompatibilityClassification.RUNTIME_DRIFT


def test_missing_legacy_runtime_fingerprint_is_not_exact(tmp_path) -> None:
    store = EvidenceStore(tmp_path / "evidence.sqlite")
    context = _context()
    legacy = _measurement(context)
    store.insert_measurement(
        EvidenceMeasurementRecord(
            **{
                **legacy.__dict__,
                "runtime_fingerprint": None,
                "runtime_environment_json": {},
            }
        )
    )

    lookup = store.lookup_evidence(context, freshness_hours=24.0)
    decision = classify_evidence_lookup(
        lookup,
        candidate_id="cfg-test",
        context=context,
        goal=Goal.BALANCED.value,
    )
    store.close()

    assert decision.classification == EvidenceCompatibilityClassification.MISSING_RUNTIME_FINGERPRINT
    assert decision.used_as_exact is False


def _runtime_drift_decision(tmp_path, old_context, current_context):
    store = EvidenceStore(tmp_path / "evidence.sqlite")
    store.insert_measurement(_measurement(old_context))
    lookup = store.lookup_evidence(current_context, freshness_hours=24.0)
    decision = classify_evidence_lookup(
        lookup,
        candidate_id="cfg-test",
        context=current_context,
        goal=Goal.BALANCED.value,
    )
    store.close()
    return decision


def _context(
    launch_config: ServingConfig | None = None,
    workload_config: EndpointBenchmarkConfig | None = None,
    *,
    backend_version: str = "0.0",
    torch_version: str = "2.7.1",
    compiler_fingerprint: str = "compiler-a",
    rendered_launch_command: list[str] | None = None,
    help_hash: str = "help-a",
    backend: str = "vllm",
):
    config = launch_config or _config(backend=backend)
    return build_evidence_request_context(
        hardware=_hardware(),
        backend=backend,
        backend_metadata={"adapter": backend, "version": backend_version},
        model="model-path",
        telemetry={"provider": "nvml", "available_fields": ["power", "temperature"]},
        launch_config=config,
        workload_config=workload_config or _workload(),
        goal=Goal.BALANCED.value,
        runtime_environment=_runtime_environment(
            backend=backend,
            backend_version=backend_version,
            torch_version=torch_version,
            compiler_fingerprint=compiler_fingerprint,
        ),
        rendered_launch_command=rendered_launch_command
        or [backend, "serve", "model-path"],
        backend_capability_help_hash=help_hash,
    )


def _measurement(
    context,
    created_at: str | None = None,
    confidence: str | None = None,
    raw_json: dict[str, object] | None = None,
) -> EvidenceMeasurementRecord:
    return EvidenceMeasurementRecord(
        measurement_id="meas-test",
        run_id="run-test",
        created_at=created_at or datetime.now(timezone.utc).isoformat(),
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
        throughput_tokens_per_sec=100.0,
        requests_per_sec=10.0,
        p95_latency_ms=20.0,
        average_power_w=200.0,
        joules_per_token=0.5,
        tokens_per_watt=0.2,
        confidence=confidence,
        power_measurement_type="measured",
        telemetry_source="nvml",
        raw_json=raw_json or {"ok": True},
    )


def _caps(*flags: str) -> VLLMArgumentCapabilities:
    return VLLMArgumentCapabilities(
        executable="vllm",
        version="test",
        supported_flags=frozenset(flags),
        help_hash="test",
        detection_status="success",
    )


def _runtime_environment(
    *,
    backend: str,
    backend_version: str,
    torch_version: str,
    compiler_fingerprint: str,
) -> dict[str, object]:
    return {
        "schema_version": "runtime-environment/v1",
        "backend_name": backend,
        "backend_version": backend_version,
        "torch_version": torch_version,
        "cuda_runtime_version": "12.6",
        "python_version": "3.12.0",
        "compiler_toolchain": {},
        "compiler_toolchain_fingerprint": compiler_fingerprint,
        "serve_optimize_git_commit": "commit-a",
        "environment_fingerprint": (
            f"{backend}:{backend_version}:torch:{torch_version}:"
            f"compiler:{compiler_fingerprint}"
        ),
    }


def _hardware(detected_at: str = "2026-01-01T00:00:00+00:00") -> HardwareSnapshot:
    return HardwareSnapshot(
        hostname="host",
        platform="linux",
        python_version="3.12",
        detected_at=detected_at,
        gpus=[
            GpuDevice(
                index=0,
                name="Generic GPU",
                uuid="GPU-1",
                total_memory_mb=1024,
                compute_capability="9.0",
                driver_version="1",
                cuda_version="12",
            )
        ],
    )


def _config(dtype: str = "fp16", **kwargs) -> ServingConfig:
    backend = str(kwargs.pop("backend", "vllm"))
    return ServingConfig(
        id="cfg-test",
        backend=backend,
        model_id="model-path",
        dtype=dtype,
        quantization="none",
        max_batch_size=1,
        max_context_tokens=2048,
        kv_cache_policy="paged",
        scheduler="continuous-batching",
        **kwargs,
    )


def _workload(concurrency: int = 1) -> EndpointBenchmarkConfig:
    return EndpointBenchmarkConfig(
        run_id="bench-run",
        base_url="http://127.0.0.1:8000/v1",
        model="model-path",
        concurrency=concurrency,
        num_requests=4,
        max_tokens=8,
        prompt="hello",
        timeout_s=30.0,
    )
