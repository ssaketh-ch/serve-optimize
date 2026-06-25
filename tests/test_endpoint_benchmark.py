import json

import pytest

import serve_optimize.endpoint_benchmark as endpoint_benchmark
from serve_optimize.aiconfig_parser import parse_aiconfig_prediction_csv
from serve_optimize.endpoint_benchmark import (
    _endpoint_url,
    _run_requests,
    aggregate_benchmark_summaries,
    compare_prediction,
    run_endpoint_benchmark,
    send_chat_completion_request,
    summarize_requests,
)
from serve_optimize.schemas import AICPrediction, EndpointBenchmarkConfig, PowerSampleRecord, RequestRecord
from serve_optimize.telemetry import (
    TelemetryCapture,
    detect_telemetry_capabilities,
    parse_nvidia_smi_sample,
    summarize_telemetry,
)


def test_parse_aiconfig_prediction_csv_preserves_numeric_fields(tmp_path) -> None:
    path = tmp_path / "best_config_topn.csv"
    path.write_text(
        "backend,version,system,model,isl,osl,concurrency,request_rate,bs,ttft,tpot,"
        "request_latency,tokens/s,tokens/s/gpu,tokens/s/user,tp,pp,dp,parallel,memory,power_w\n"
        "vllm,0.6,h200_sxm,example-model,512,128,512,350.931,16,57.922,11.324,"
        "1496.106,44568.232,44568.232,87.047,1,1,1,tp1pp1dp1,65536,184.2\n",
        encoding="utf-8",
    )

    prediction = parse_aiconfig_prediction_csv(path)

    assert prediction.backend == "vllm"
    assert prediction.concurrency == 512
    assert prediction.request_rate == pytest.approx(350.931)
    assert prediction.tokens_s == pytest.approx(44568.232)
    assert prediction.tokens_s_gpu == pytest.approx(44568.232)
    assert prediction.tokens_s_user == pytest.approx(87.047)
    assert prediction.memory == pytest.approx(65536)
    assert prediction.power_w == pytest.approx(184.2)
    assert prediction.raw["tokens/s"] == "44568.232"


def test_summary_metric_calculation_and_failure_accounting() -> None:
    records = [
        RequestRecord(0, 0.0, 1.0, 1.0, "ok", prompt_tokens=10, completion_tokens=20, total_tokens=30),
        RequestRecord(1, 0.0, 2.0, 2.0, "ok", prompt_tokens=15, completion_tokens=30, total_tokens=45),
        RequestRecord(2, 0.0, 3.0, 3.0, "ok", prompt_tokens=25, completion_tokens=50, total_tokens=75),
        RequestRecord(3, 0.0, 1.0, 1.0, "error", error="boom"),
    ]

    summary = summarize_requests("run-1", records, wall_time_s=5.0)

    assert summary.total_requests == 4
    assert summary.successful_requests == 3
    assert summary.failed_requests == 1
    assert summary.request_rate_req_s == pytest.approx(0.6)
    assert summary.prompt_tokens == 50
    assert summary.completion_tokens == 100
    assert summary.total_tokens == 150
    assert summary.output_tokens_s == pytest.approx(20.0)
    assert summary.total_tokens_s == pytest.approx(30.0)
    assert summary.avg_latency_s == pytest.approx(2.0)
    assert summary.p50_latency_s == pytest.approx(2.0)
    assert summary.p95_latency_s == pytest.approx(2.9)
    assert summary.p99_latency_s == pytest.approx(2.98)


def test_summary_flags_insufficient_concurrency_coverage() -> None:
    records = [
        RequestRecord(0, 0.0, 1.0, 1.0, "ok", prompt_tokens=10, completion_tokens=20, total_tokens=30),
        RequestRecord(1, 0.0, 1.0, 1.0, "ok", prompt_tokens=10, completion_tokens=20, total_tokens=30),
    ]

    summary = summarize_requests(
        "run-undercovered",
        records,
        wall_time_s=1.0,
        configured_concurrency=4,
        configured_num_requests=2,
    )

    assert summary.measurement_quality["configured_concurrency"] == 4
    assert summary.measurement_quality["effective_concurrency_limit"] == 2
    assert summary.measurement_quality["concurrency_coverage"] == "insufficient"
    assert any("lower than concurrency" in warning for warning in summary.warnings)

    warmup_summary = summarize_requests(
        "run-undercovered-warmup",
        [
            RequestRecord(index, float(index), float(index) + 1.0, 1.0, "ok", prompt_tokens=10, completion_tokens=20, total_tokens=30)
            for index in range(5)
        ],
        wall_time_s=5.0,
        warmup_requests=2,
        configured_concurrency=4,
        configured_num_requests=5,
    )

    assert warmup_summary.measurement_quality["effective_concurrency_limit"] == 3
    assert warmup_summary.measurement_quality["concurrency_coverage"] == "insufficient"


def test_summary_includes_stream_timing_metrics() -> None:
    records = [
        RequestRecord(
            0,
            0.0,
            1.0,
            1.0,
            "ok",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            ttft_s=0.1,
            tpot_s=0.02,
            timing_source="openai_stream_chunks",
        ),
        RequestRecord(
            1,
            0.0,
            1.2,
            1.2,
            "ok",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            ttft_s=0.2,
            tpot_s=0.04,
            timing_source="openai_stream_chunks",
        ),
    ]

    summary = summarize_requests("run-timing", records, wall_time_s=2.0)

    assert summary.avg_ttft_ms == pytest.approx(150.0)
    assert summary.p95_ttft_ms == pytest.approx(195.0)
    assert summary.avg_tpot_ms == pytest.approx(30.0)
    assert summary.p95_tpot_ms == pytest.approx(39.0)
    assert summary.ttft_sample_count == 2
    assert summary.tpot_sample_count == 2
    assert summary.timing_source == "openai_stream_chunks"
    assert summary.measurement_quality["phase_energy_attribution"] == "unavailable_without_phase_markers"


def test_summary_metric_calculation_with_power_fields() -> None:
    records = [
        RequestRecord(0, 0.0, 1.0, 1.0, "ok", prompt_tokens=10, completion_tokens=20, total_tokens=30),
        RequestRecord(1, 0.0, 2.0, 2.0, "ok", prompt_tokens=10, completion_tokens=30, total_tokens=40),
    ]
    telemetry = TelemetryCapture(
        provider="nvml",
        samples=[
            PowerSampleRecord(0.0, "measured", 100.0, "nvml", gpu_util_percent=70.0),
            PowerSampleRecord(0.2, "measured", 110.0, "nvml", gpu_util_percent=72.0),
            PowerSampleRecord(0.4, "measured", 120.0, "nvml", gpu_util_percent=74.0),
            PowerSampleRecord(0.6, "measured", 130.0, "nvml", gpu_util_percent=76.0),
            PowerSampleRecord(0.8, "measured", 140.0, "nvml", gpu_util_percent=78.0),
        ],
        warnings=[],
    )

    summary = summarize_requests("run-power", records, wall_time_s=2.0, power_samples=telemetry.samples, telemetry=telemetry)

    assert summary.power_sample_count == 5
    assert summary.average_power_watts == pytest.approx(120.0)
    assert summary.peak_power_watts == pytest.approx(140.0)
    assert summary.energy_joules == pytest.approx(240.0)
    assert summary.joules_per_token == pytest.approx(240.0 / 70.0)
    assert summary.tokens_per_second_per_watt == pytest.approx(round(35.0 / 120.0, 6))
    assert summary.telemetry_provider == "nvml"
    assert summary.telemetry_quality == "good"
    assert summary.average_gpu_util_percent == pytest.approx(74.0)
    assert summary.warnings == []


def test_summary_idle_subtraction_fields() -> None:
    records = [
        RequestRecord(0, 0.0, 1.0, 1.0, "ok", prompt_tokens=10, completion_tokens=20, total_tokens=30),
        RequestRecord(1, 0.0, 2.0, 2.0, "ok", prompt_tokens=10, completion_tokens=30, total_tokens=40),
    ]
    telemetry = TelemetryCapture(
        provider="nvml",
        samples=[
            PowerSampleRecord(0.0, "active", 120.0, "nvml", gpu_util_percent=70.0),
            PowerSampleRecord(0.2, "active", 120.0, "nvml", gpu_util_percent=70.0),
            PowerSampleRecord(0.4, "active", 120.0, "nvml", gpu_util_percent=70.0),
            PowerSampleRecord(0.6, "active", 120.0, "nvml", gpu_util_percent=70.0),
            PowerSampleRecord(0.8, "active", 120.0, "nvml", gpu_util_percent=70.0),
        ],
        warnings=[],
    )

    summary = summarize_requests(
        "run-idle",
        records,
        wall_time_s=2.0,
        power_samples=telemetry.samples,
        telemetry=telemetry,
        idle_power_watts=100.0,
    )

    assert summary.energy_joules == pytest.approx(240.0)
    assert summary.idle_power_watts == pytest.approx(100.0)
    assert summary.active_power_watts == pytest.approx(20.0)
    assert summary.active_energy_joules == pytest.approx(40.0)
    assert summary.active_joules_per_token == pytest.approx(40.0 / 70.0)
    assert summary.active_tokens_per_second_per_watt == pytest.approx(35.0 / 20.0)
    assert summary.measurement_quality["energy_accounting"] == "idle_subtracted"


def test_summary_warmup_and_steady_state_window() -> None:
    records = [
        RequestRecord(0, 0.0, 1.0, 1.0, "ok", prompt_tokens=100, completion_tokens=100, total_tokens=200),
        RequestRecord(1, 1.0, 2.0, 1.0, "ok", prompt_tokens=10, completion_tokens=20, total_tokens=30),
        RequestRecord(2, 4.0, 5.0, 1.0, "ok", prompt_tokens=10, completion_tokens=30, total_tokens=40),
    ]

    summary = summarize_requests(
        "run-steady",
        records,
        wall_time_s=5.0,
        warmup_requests=1,
        steady_state_duration_s=1.5,
    )

    assert summary.successful_requests == 3
    assert summary.warmup_requests == 1
    assert summary.steady_state_requests == 1
    assert summary.total_tokens == 30
    assert summary.steady_state_total_tokens == 30
    assert summary.measurement_duration_s == pytest.approx(1.0)
    assert summary.request_rate_req_s == pytest.approx(1.0)
    assert summary.total_tokens_s == pytest.approx(30.0)
    assert summary.measured_requests == 1
    assert summary.measured_successful_requests == 1
    assert summary.measured_failed_requests == 0
    assert summary.measurement_quality["steady_state_requests"] == 1


def test_measurement_window_filters_failures_and_power_with_the_same_boundaries() -> None:
    records = [
        RequestRecord(0, 0.0, 1.0, 1.0, "error", error="warmup failure"),
        RequestRecord(1, 1.0, 2.0, 1.0, "ok", prompt_tokens=10, completion_tokens=20, total_tokens=30),
        RequestRecord(2, 4.0, 5.0, 1.0, "ok", prompt_tokens=10, completion_tokens=30, total_tokens=40),
    ]
    samples = [
        PowerSampleRecord(0.5, "active", 200.0, "nvml"),
        PowerSampleRecord(1.5, "active", 100.0, "nvml"),
        PowerSampleRecord(4.5, "active", 300.0, "nvml"),
    ]

    summary = summarize_requests(
        "run-window",
        records,
        wall_time_s=5.0,
        power_samples=samples,
        telemetry=TelemetryCapture(provider="nvml", samples=samples, warnings=[]),
        warmup_requests=1,
        steady_state_duration_s=1.5,
    )

    assert summary.total_requests == 3
    assert summary.failed_requests == 1
    assert summary.measured_requests == 1
    assert summary.measured_failed_requests == 0
    assert summary.average_power_watts == pytest.approx(100.0)
    assert summary.energy_joules == pytest.approx(100.0)
    assert summary.joules_per_token == pytest.approx(100.0 / 30.0)


def test_all_warmup_requests_leave_the_measurement_window_empty() -> None:
    records = [RequestRecord(0, 0.0, 1.0, 1.0, "ok", prompt_tokens=10, completion_tokens=20, total_tokens=30)]
    samples = [PowerSampleRecord(0.5, "active", 100.0, "nvml")]

    summary = summarize_requests(
        "run-all-warmup",
        records,
        wall_time_s=1.0,
        power_samples=samples,
        telemetry=TelemetryCapture(provider="nvml", samples=samples, warnings=[]),
        warmup_requests=1,
    )

    assert summary.measured_requests == 0
    assert summary.measurement_duration_s is None
    assert summary.total_tokens_s == 0.0
    assert summary.average_power_watts is None
    assert summary.energy_joules is None


def test_aggregate_benchmark_summaries_adds_trial_statistics_and_confidence() -> None:
    first = summarize_requests(
        "trial-1",
        [RequestRecord(0, 0.0, 1.0, 1.0, "ok", prompt_tokens=10, completion_tokens=20, total_tokens=30)],
        wall_time_s=1.0,
    )
    second = summarize_requests(
        "trial-2",
        [RequestRecord(0, 0.0, 1.0, 1.0, "ok", prompt_tokens=10, completion_tokens=26, total_tokens=36)],
        wall_time_s=1.0,
    )

    aggregate = aggregate_benchmark_summaries("aggregate", [first, second])

    assert aggregate.run_id == "aggregate"
    assert aggregate.total_requests == 2
    assert aggregate.total_tokens == 66
    assert aggregate.trial_statistics["trial_count"] == 2
    assert aggregate.confidence_intervals["total_tokens_s"]["count"] == 2
    assert aggregate.stability_classification == "mostly_stable"
    assert aggregate.measurement_quality["measured_requests"] == 2
    assert aggregate.measurement_quality["measurement_duration_s"] == pytest.approx(2.0)


def test_telemetry_summary_with_full_fields_is_good() -> None:
    samples = [
        PowerSampleRecord(
            timestamp_s=float(index),
            phase="measured",
            watts=100.0 + index * 5.0,
            source="nvml",
            provider="nvml",
            gpu_util_percent=60.0 + index,
            memory_util_percent=40.0 + index,
            memory_used_mb=1024 + index,
            memory_total_mb=4096,
            temperature_c=65.0 + index,
            sm_clock_mhz=1500 + index,
            memory_clock_mhz=2000 + index,
            power_limit_watts=300.0,
            device_name="Generic GPU",
        )
        for index in range(5)
    ]

    summary = summarize_telemetry(samples, wall_time_s=5.0, total_tokens=1000, provider="nvml")

    assert summary.telemetry_available is True
    assert summary.telemetry_quality == "good"
    assert summary.power_sample_count == 5
    assert summary.average_power_watts == pytest.approx(110.0)
    assert summary.power_sampling_rate_hz == pytest.approx(1.0)
    assert summary.average_gpu_util_percent == pytest.approx(62.0)
    assert summary.max_memory_util_percent == pytest.approx(44.0)
    assert summary.average_temperature_c == pytest.approx(67.0)
    assert summary.power_limit_watts == pytest.approx(300.0)
    assert summary.telemetry_warnings == []
    assert set(summary.missing_fields) == {"enforced_power_limit_watts", "graphics_clock_mhz", "throttle_reasons"}
    assert summary.power_stats["avg"] == pytest.approx(110.0)
    assert summary.utilization_stats["avg_gpu_util_percent"] == pytest.approx(62.0)
    assert summary.thermal_stats["avg_temperature_c"] == pytest.approx(67.0)
    assert summary.clock_stats["avg_sm_clock_mhz"] == pytest.approx(1502.0)
    assert summary.thermal_stats["temperature_rise_c"] == pytest.approx(4.0)
    assert summary.thermal_stats["stability_classification"] == "limited_window"


def test_telemetry_summary_reports_longer_thermal_soak_stability() -> None:
    samples = [
        PowerSampleRecord(
            timestamp_s=float(index * 30),
            phase="measured",
            watts=120.0,
            source="nvml",
            provider="nvml",
            gpu_util_percent=70.0,
            temperature_c=60.0 + index * 0.25,
        )
        for index in range(5)
    ]

    summary = summarize_telemetry(samples, wall_time_s=120.0, total_tokens=1000, provider="nvml")

    assert summary.temperature_rise_c == pytest.approx(1.0)
    assert summary.temperature_slope_c_per_min == pytest.approx(0.5)
    assert summary.thermal_stability_classification == "stable"


def test_telemetry_summary_with_missing_optional_fields_is_limited() -> None:
    samples = [
        PowerSampleRecord(0.0, "measured", 120.0, "nvml"),
        PowerSampleRecord(0.2, "measured", 121.0, "nvml"),
        PowerSampleRecord(0.4, "measured", 122.0, "nvml"),
        PowerSampleRecord(0.6, "measured", 123.0, "nvml"),
        PowerSampleRecord(0.8, "measured", 124.0, "nvml"),
    ]

    summary = summarize_telemetry(samples, wall_time_s=1.0, total_tokens=100)

    assert summary.telemetry_quality == "limited"
    assert any("GPU utilization was unavailable" in warning for warning in summary.telemetry_warnings)


def test_telemetry_summary_warns_for_low_sample_count_rate_and_flat_power() -> None:
    samples = [
        PowerSampleRecord(0.0, "measured", 100.0, "nvml", gpu_util_percent=50.0),
        PowerSampleRecord(10.0, "measured", 100.5, "nvml", gpu_util_percent=50.0),
    ]

    summary = summarize_telemetry(samples, wall_time_s=10.0, total_tokens=100)

    assert summary.telemetry_quality == "poor"
    assert any("sample count is low" in warning for warning in summary.telemetry_warnings)
    assert any("below 1 Hz" in warning for warning in summary.telemetry_warnings)
    assert any("nearly flat" in note for note in summary.telemetry_notes)


def test_telemetry_summary_mig_note() -> None:
    summary = summarize_telemetry(
        [PowerSampleRecord(0.0, "measured", 100.0, "nvml", gpu_util_percent=50.0, mig_mode="1")],
        wall_time_s=1.0,
        total_tokens=100,
    )

    assert any("MIG power readings" in note for note in summary.telemetry_notes)


def test_telemetry_capability_detection_with_missing_utilization() -> None:
    capabilities = detect_telemetry_capabilities(
        [
            PowerSampleRecord(
                0.0,
                "measured",
                100.0,
                "nvml",
                provider="nvml",
                device_name="Generic GPU",
                temperature_c=65.0,
                memory_used_mb=1024,
                memory_total_mb=4096,
                sm_clock_mhz=1500,
                power_limit_watts=300.0,
            )
        ],
        provider="nvml",
    )

    assert capabilities.provider == "nvml"
    assert "power" in capabilities.available_fields
    assert "temperature" in capabilities.available_fields
    assert "memory_usage" in capabilities.available_fields
    assert "clocks" in capabilities.available_fields
    assert "gpu_utilization" in capabilities.unavailable_fields
    assert "memory_utilization" in capabilities.unavailable_fields
    assert any("utilization metrics" in note for note in capabilities.notes)


def test_telemetry_capability_detection_mig_note() -> None:
    capabilities = detect_telemetry_capabilities(
        [PowerSampleRecord(0.0, "measured", 100.0, "nvml", provider="nvml", mig_mode="1")],
        provider="nvml",
    )

    assert any("MIG environments" in note for note in capabilities.notes)


def test_parse_nvidia_smi_sample_normal_output() -> None:
    fields = [
        "index",
        "name",
        "power.draw",
        "power.limit",
        "memory.used",
        "memory.total",
        "utilization.gpu",
        "utilization.memory",
        "temperature.gpu",
        "clocks.gr",
        "clocks.sm",
        "clocks.mem",
    ]
    sample = parse_nvidia_smi_sample(fields, "0, Generic GPU, 175.5, 300.0, 4096, 8192, 91, 45, 67, 1800, 1700, 5001")

    assert sample.provider == "nvidia-smi"
    assert sample.device_index == 0
    assert sample.device_name == "Generic GPU"
    assert sample.power_watts == pytest.approx(175.5)
    assert sample.power_limit_watts == pytest.approx(300.0)
    assert sample.memory_used_mb == 4096
    assert sample.memory_util_percent == pytest.approx(50.0)
    assert sample.gpu_util_percent == pytest.approx(91.0)
    assert sample.temperature_c == pytest.approx(67.0)
    assert sample.sm_clock_mhz == 1700


def test_parse_nvidia_smi_sample_missing_fields_are_none() -> None:
    fields = [
        "index",
        "name",
        "power.draw",
        "power.limit",
        "memory.used",
        "memory.total",
        "utilization.gpu",
        "temperature.gpu",
    ]
    sample = parse_nvidia_smi_sample(fields, "0, Generic GPU, 175.5, N/A, 4096, 8192, N/A, [Not Supported]")

    assert sample.error is None
    assert sample.power_watts == pytest.approx(175.5)
    assert sample.power_limit_watts is None
    assert sample.gpu_util_percent is None
    assert sample.temperature_c is None


def test_prediction_comparison_calculation() -> None:
    summary = summarize_requests(
        "run-1",
        [
            RequestRecord(0, 0.0, 2.0, 2.0, "ok", prompt_tokens=100, completion_tokens=100, total_tokens=200),
        ],
        wall_time_s=2.0,
    )
    prediction = AICPrediction(tokens_s=200.0, request_rate=2.0, request_latency=1000.0, concurrency=4)
    config = EndpointBenchmarkConfig(
        run_id="run-1",
        base_url="http://127.0.0.1:8080/v1",
        model="example",
        concurrency=2,
        num_requests=1,
        max_tokens=100,
        prompt="hello",
        timeout_s=10.0,
    )

    comparison = compare_prediction("run-1", prediction, summary, config)

    assert comparison.metrics["tokens_s"]["measured"] == pytest.approx(100.0)
    assert comparison.metrics["tokens_s"]["absolute_delta"] == pytest.approx(-100.0)
    assert comparison.metrics["tokens_s"]["percent_delta"] == pytest.approx(-50.0)
    assert comparison.metrics["tokens_s"]["measured_over_predicted_ratio"] == pytest.approx(0.5)
    assert comparison.metrics["request_latency_avg_s"]["predicted"] == pytest.approx(1.0)
    assert comparison.metrics["request_latency_avg_s"]["measured"] == pytest.approx(2.0)
    assert comparison.metrics["concurrency"]["absolute_delta"] == pytest.approx(-2.0)


def test_streaming_request_records_ttft_and_tpot(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        status = 200

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
            return None

        def __iter__(self):
            yield b'data: {"choices":[{"delta":{"content":"a"}}]}\n'
            yield b'data: {"choices":[{"delta":{"content":"b"}}]}\n'
            yield b'data: {"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5},"choices":[]}\n'
            yield b"data: [DONE]\n"

    times = iter([0.0, 0.2, 0.5, 0.6])
    monkeypatch.setattr(endpoint_benchmark.time, "time", lambda: next(times))
    monkeypatch.setattr(endpoint_benchmark.request, "urlopen", lambda request, timeout: FakeResponse())
    config = EndpointBenchmarkConfig(
        run_id="run-stream",
        base_url="http://127.0.0.1:8080/v1",
        model="example",
        concurrency=1,
        num_requests=1,
        max_tokens=8,
        prompt="hello",
        timeout_s=10.0,
        stream=True,
    )

    record = send_chat_completion_request(config, request_id=0)

    assert record.status == "ok"
    assert record.ttft_s == pytest.approx(0.2)
    assert record.tpot_s == pytest.approx(0.3)
    assert record.timing_source == "openai_stream_chunks"
    assert record.prompt_tokens == 3
    assert record.completion_tokens == 2
    assert record.total_tokens == 5
    assert record.token_count_source == "response_usage"


def test_streaming_request_asks_for_usage_without_using_chunks_as_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    class FakeResponse:
        status = 200

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
            return None

        def __iter__(self):
            yield b'data: {"choices":[{"delta":{"reasoning_content":"think"}}]}\n'
            yield b'data: {"choices":[{"text":"answer"}]}\n'
            yield b"data: [DONE]\n"

    def fake_urlopen(http_request, timeout):
        captured["payload"] = json.loads(http_request.data)
        return FakeResponse()

    times = iter([0.0, 0.2, 0.5, 0.6])
    monkeypatch.setattr(endpoint_benchmark.time, "time", lambda: next(times))
    monkeypatch.setattr(endpoint_benchmark.request, "urlopen", fake_urlopen)
    config = EndpointBenchmarkConfig(
        run_id="run-stream-no-usage",
        base_url="http://127.0.0.1:8080/v1",
        model="example",
        concurrency=1,
        num_requests=1,
        max_tokens=8,
        prompt="hello",
        timeout_s=10.0,
        stream=True,
    )

    record = send_chat_completion_request(config, request_id=0)
    summary = summarize_requests("run-stream-no-usage", [record], wall_time_s=0.6)

    assert captured["payload"]["stream_options"] == {"include_usage": True}
    assert record.status == "ok"
    assert record.completion_tokens == 0
    assert record.total_tokens == 0
    assert record.token_count_source is None
    assert summary.total_tokens_s == 0.0
    assert "Streaming usage was unavailable" in summary.warnings[0]


def test_streaming_response_without_output_is_a_failed_request(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        status = 200

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
            return None

        def __iter__(self):
            yield b'data: {"choices":[{"delta":{}}]}\n'
            yield b"data: [DONE]\n"

    times = iter([0.0, 0.2])
    monkeypatch.setattr(endpoint_benchmark.time, "time", lambda: next(times))
    monkeypatch.setattr(endpoint_benchmark.request, "urlopen", lambda request, timeout: FakeResponse())
    config = EndpointBenchmarkConfig(
        run_id="run-empty-stream",
        base_url="http://127.0.0.1:8080/v1",
        model="example",
        concurrency=1,
        num_requests=1,
        max_tokens=8,
        prompt="hello",
        timeout_s=10.0,
        stream=True,
    )

    record = send_chat_completion_request(config, request_id=0)

    assert record.status == "stream_no_output"
    assert record.error == "Streaming response contained no output content."


@pytest.mark.parametrize("base_url", ["file:///tmp/model", "ftp://example.com/v1", "example.com/v1"])
def test_endpoint_url_rejects_non_http_schemes(base_url: str) -> None:
    with pytest.raises(ValueError, match="must use http or https"):
        _endpoint_url(base_url, "/v1/chat/completions")


def test_endpoint_url_rejects_embedded_credentials() -> None:
    with pytest.raises(ValueError, match="must not be embedded"):
        _endpoint_url("https://user:secret@example.com/v1", "/v1/chat/completions")


def test_endpoint_auth_reads_secret_from_environment_without_storing_it(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    captured = {}

    class FakeResponse:
        status = 200

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}}'

    def fake_urlopen(http_request, timeout):
        captured["authorization"] = http_request.get_header("Authorization")
        return FakeResponse()

    monkeypatch.setenv("SERVE_OPTIMIZE_TEST_KEY", "top-secret-value")
    times = iter([0.0, 0.2])
    monkeypatch.setattr(endpoint_benchmark.time, "time", lambda: next(times))
    monkeypatch.setattr(endpoint_benchmark.request, "urlopen", fake_urlopen)
    config = EndpointBenchmarkConfig(
        run_id="run-auth",
        base_url="https://example.com/v1",
        model="example",
        concurrency=1,
        num_requests=1,
        max_tokens=8,
        prompt="hello",
        timeout_s=10.0,
        api_key_env="SERVE_OPTIMIZE_TEST_KEY",
    )

    record = send_chat_completion_request(config, request_id=0)
    run = run_endpoint_benchmark(
        config,
        tmp_path,
        request_fn=lambda _config, request_id: RequestRecord(request_id, 0.0, 0.1, 0.1, "ok"),
    )
    config_text = (run.run_dir / "config.json").read_text(encoding="utf-8")

    assert record.status == "ok"
    assert captured["authorization"] == "Bearer top-secret-value"
    assert "SERVE_OPTIMIZE_TEST_KEY" in config_text
    assert "top-secret-value" not in config_text


def test_endpoint_auth_rejects_an_unset_environment_variable() -> None:
    config = EndpointBenchmarkConfig(
        run_id="run-auth-missing",
        base_url="https://example.com/v1",
        model="example",
        concurrency=1,
        num_requests=1,
        max_tokens=8,
        prompt="hello",
        timeout_s=10.0,
        api_key_env="SERVE_OPTIMIZE_MISSING_KEY",
    )

    with pytest.raises(ValueError, match="unset or empty"):
        send_chat_completion_request(config, request_id=0)


def test_soak_duration_runs_additional_request_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    config = EndpointBenchmarkConfig(
        run_id="run-soak",
        base_url="http://127.0.0.1:8080/v1",
        model="example",
        concurrency=1,
        num_requests=1,
        max_tokens=8,
        prompt="hello",
        timeout_s=10.0,
        soak_duration_s=2.0,
    )

    perf_times = iter([0.0, 1.0, 2.5])
    monkeypatch.setattr(endpoint_benchmark.time, "perf_counter", lambda: next(perf_times))

    def fake_request(_config: EndpointBenchmarkConfig, request_id: int) -> RequestRecord:
        return RequestRecord(request_id, float(request_id), float(request_id) + 0.1, 0.1, "ok")

    records = _run_requests(config, fake_request)

    assert [record.request_id for record in sorted(records, key=lambda item: item.request_id)] == [0, 1]


def test_mocked_endpoint_benchmark_writes_artifacts_and_request_jsonl(tmp_path) -> None:
    config = EndpointBenchmarkConfig(
        run_id="run-mocked",
        base_url="http://127.0.0.1:8080/v1",
        model="example",
        concurrency=2,
        num_requests=3,
        max_tokens=32,
        prompt="hello",
        timeout_s=10.0,
    )
    prediction = AICPrediction(tokens_s=90.0, request_rate=3.0, request_latency=100.0, concurrency=2)

    def fake_request(_config: EndpointBenchmarkConfig, request_id: int) -> RequestRecord:
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

    run = run_endpoint_benchmark(config, tmp_path, prediction=prediction, request_fn=fake_request)
    run_dir = run.run_dir

    assert (run_dir / "config.json").exists()
    assert (run_dir / "prediction.json").exists()
    assert (run_dir / "requests.jsonl").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "comparison.json").exists()
    assert (run_dir / "metadata.json").exists()
    assert not (run_dir / "power_samples.jsonl").exists()

    request_rows = [json.loads(line) for line in (run_dir / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(request_rows) == 3
    assert set(request_rows[0]) >= {
        "request_id",
        "start_time",
        "end_time",
        "latency_s",
        "status",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    }
    assert run.summary.total_requests == 3
    assert run.summary.failed_requests == 0
    assert run.comparison is not None


def test_mocked_endpoint_benchmark_with_telemetry_writes_power_artifacts(tmp_path) -> None:
    config = EndpointBenchmarkConfig(
        run_id="run-telemetry",
        base_url="http://127.0.0.1:8080/v1",
        model="example",
        concurrency=2,
        num_requests=2,
        max_tokens=16,
        prompt="hello",
        timeout_s=10.0,
        telemetry="nvml",
    )

    def fake_request(_config: EndpointBenchmarkConfig, request_id: int) -> RequestRecord:
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

    run = run_endpoint_benchmark(
        config,
        tmp_path,
        request_fn=fake_request,
        telemetry_collector_factory=lambda telemetry, device_index, interval_s: _FakeCollector(
            TelemetryCapture(
                provider=telemetry,
                samples=[
                    PowerSampleRecord(0.0, "measured", 100.0, telemetry, device_index=device_index, gpu_memory_used_mb=1024, gpu_util_percent=70.0),
                    PowerSampleRecord(0.2, "measured", 110.0, telemetry, device_index=device_index, gpu_memory_used_mb=1152, gpu_util_percent=72.0),
                    PowerSampleRecord(0.4, "measured", 120.0, telemetry, device_index=device_index, gpu_memory_used_mb=1280, gpu_util_percent=74.0),
                    PowerSampleRecord(0.6, "measured", 130.0, telemetry, device_index=device_index, gpu_memory_used_mb=1408, gpu_util_percent=76.0),
                    PowerSampleRecord(0.8, "measured", 140.0, telemetry, device_index=device_index, gpu_memory_used_mb=1536, gpu_util_percent=78.0),
                ],
                warnings=[],
            )
        ),
    )

    run_dir = run.run_dir
    assert (run_dir / "power_samples.jsonl").exists()
    assert (run_dir / "telemetry_summary.json").exists()
    assert (run_dir / "telemetry_capabilities.json").exists()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["power_sample_count"] == 5
    assert summary["average_power_watts"] == pytest.approx(120.0)
    assert summary["peak_power_watts"] == pytest.approx(140.0)
    assert summary["energy_joules"] is not None
    assert summary["joules_per_token"] is not None
    assert summary["tokens_per_second_per_watt"] is not None
    assert summary["telemetry_provider"] == "nvml"
    assert summary["telemetry_quality"] == "good"
    assert summary["average_gpu_util_percent"] == pytest.approx(74.0)
    assert summary["warnings"] == []
    capabilities = json.loads((run_dir / "telemetry_capabilities.json").read_text(encoding="utf-8"))
    assert capabilities["provider"] == "nvml"
    assert "power" in capabilities["available_fields"]


def test_mocked_endpoint_benchmark_telemetry_failure_adds_warnings(tmp_path) -> None:
    config = EndpointBenchmarkConfig(
        run_id="run-telemetry-failure",
        base_url="http://127.0.0.1:8080/v1",
        model="example",
        concurrency=1,
        num_requests=1,
        max_tokens=8,
        prompt="hello",
        timeout_s=10.0,
        telemetry="auto",
    )

    run = run_endpoint_benchmark(
        config,
        tmp_path,
        request_fn=lambda _config, request_id: RequestRecord(
            request_id=request_id,
            start_time=0.0,
            end_time=0.2,
            latency_s=0.2,
            status="ok",
            prompt_tokens=4,
            completion_tokens=8,
            total_tokens=12,
        ),
        telemetry_collector_factory=lambda telemetry, device_index, interval_s: _FakeCollector(
            TelemetryCapture(
                provider="nvidia-smi",
                samples=[PowerSampleRecord(0.0, "measured", None, "nvidia-smi", error="mock telemetry failure")],
                warnings=["nvidia-smi telemetry sample failed: mock telemetry failure"],
            )
        ),
    )

    summary = json.loads((run.run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["power_sample_count"] == 1
    assert summary["average_power_watts"] is None
    assert summary["peak_power_watts"] is None
    assert summary["energy_joules"] is None
    assert "nvidia-smi telemetry sample failed: mock telemetry failure" in summary["warnings"]
    assert summary["telemetry_quality"] == "poor"


class _FakeCollector:
    def __init__(self, capture: TelemetryCapture):
        self.capture = capture

    def start(self) -> None:
        return None

    def stop(self) -> TelemetryCapture:
        return self.capture
