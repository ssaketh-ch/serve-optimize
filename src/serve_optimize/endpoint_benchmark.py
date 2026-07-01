"""Endpoint-native OpenAI-compatible benchmark runner."""

from __future__ import annotations

import json
import math
import os
import platform
import socket
import statistics
import subprocess
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from urllib import error, request
from urllib.parse import urlsplit

from .io import write_json, write_jsonl
from .schemas import (
    AICPrediction,
    EndpointBenchmarkConfig,
    EndpointBenchmarkSummary,
    HardwareSnapshot,
    PowerSampleRecord,
    PredictionComparison,
    RequestRecord,
    to_dict,
)
from .telemetry import TelemetryCapture, make_telemetry_collector, summarize_telemetry

DEFAULT_ENDPOINT_PROMPT = "GPU optimization " * 100
CLIENT_CPU_SATURATION_THRESHOLD_PERCENT = 90.0
CLIENT_CPU_WARNING_MIN_DURATION_S = 1.0
LOAD_GPU_SATURATION_THRESHOLD_PERCENT = 85.0


RequestFn = Callable[[EndpointBenchmarkConfig, int], RequestRecord]
TelemetryCollectorFactory = Callable[[str, int, float], object]


@dataclass(frozen=True)
class EndpointBenchmarkRun:
    run_dir: Path
    summary: EndpointBenchmarkSummary
    comparison: PredictionComparison | None


def make_run_id(prefix: str = "endpoint") -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{timestamp}-{uuid.uuid4().hex[:8]}"


def _telemetry_interval_s(config: EndpointBenchmarkConfig) -> float:
    requested_duration = config.steady_state_duration_s or config.soak_duration_s
    if requested_duration is not None and requested_duration <= 30:
        return 0.25
    return 1.0


def run_endpoint_benchmark(
    config: EndpointBenchmarkConfig,
    out_dir: Path,
    prediction: AICPrediction | None = None,
    hardware: HardwareSnapshot | None = None,
    request_fn: RequestFn | None = None,
    telemetry_collector_factory: TelemetryCollectorFactory | None = None,
) -> EndpointBenchmarkRun:
    started_at = datetime.now(timezone.utc).isoformat()
    trial_wall_start = time.perf_counter()
    run_dir = out_dir / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    request_fn = request_fn or send_chat_completion_request

    write_json(run_dir / "config.json", config)
    if prediction is not None:
        write_json(run_dir / "prediction.json", prediction)

    telemetry_collector_factory = telemetry_collector_factory or make_telemetry_collector
    telemetry_interval_s = _telemetry_interval_s(config)
    idle_samples: list[PowerSampleRecord] = []
    idle_power_watts = config.idle_power_watts
    if config.telemetry != "none" and config.idle_baseline_duration_s > 0:
        idle_sampler = telemetry_collector_factory(config.telemetry, config.device_index, telemetry_interval_s)
        idle_sampler.start()
        time.sleep(config.idle_baseline_duration_s)
        idle_capture = idle_sampler.stop()
        idle_samples = [replace(sample, phase="idle") for sample in idle_capture.samples]
        idle_power_watts = idle_power_watts if idle_power_watts is not None else _average_power(idle_samples)

    sampler = telemetry_collector_factory(config.telemetry, config.device_index, telemetry_interval_s)
    sampler.start()
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    records = _run_requests(config, request_fn)
    wall_time_s = max(time.perf_counter() - wall_start, 0.0)
    client_cpu_time_s = max(time.process_time() - cpu_start, 0.0)
    client_cpu_utilization_percent = (client_cpu_time_s / wall_time_s * 100.0) if wall_time_s > 0 else None
    telemetry = sampler.stop()
    ended_at = datetime.now(timezone.utc).isoformat()
    trial_wall_clock_time_s = max(time.perf_counter() - trial_wall_start, 0.0)

    records = sorted(records, key=lambda item: item.request_id)
    power_samples = _phase_power_samples(
        [replace(sample, phase="active") for sample in telemetry.samples],
        records,
        warmup_requests=config.warmup_requests,
        steady_state_duration_s=config.steady_state_duration_s,
    )
    warmup_power_samples = [sample for sample in power_samples if sample.phase == "warmup"]
    measurement_power_samples = [sample for sample in power_samples if sample.phase == "measurement"]
    write_jsonl(run_dir / "requests.jsonl", records)
    if config.telemetry != "none":
        write_jsonl(run_dir / "power_samples.jsonl", [*idle_samples, *power_samples])
        if idle_samples:
            write_jsonl(run_dir / "idle_power_samples.jsonl", idle_samples)
        if warmup_power_samples:
            write_jsonl(run_dir / "warmup_power_samples.jsonl", warmup_power_samples)
        if measurement_power_samples:
            write_jsonl(run_dir / "measurement_power_samples.jsonl", measurement_power_samples)

    summary = summarize_requests(
        config.run_id,
        records,
        wall_time_s,
        power_samples,
        telemetry,
        warmup_requests=config.warmup_requests,
        steady_state_duration_s=config.steady_state_duration_s,
        soak_duration_s=config.soak_duration_s,
        idle_power_watts=idle_power_watts,
        idle_sample_count=len(idle_samples),
        configured_concurrency=config.concurrency,
        configured_num_requests=config.num_requests,
        backend_name=config.backend_name,
        backend_version=config.backend_version,
        backend_launch_command=config.backend_launch_command,
        backend_launch_command_hash=config.backend_launch_command_hash,
        backend_effective_values=config.backend_effective_values,
        backend_applied_configuration=config.backend_applied_configuration,
        backend_omitted_values=config.backend_omitted_values,
        backend_unsupported_values=config.backend_unsupported_values,
        backend_unavailable_values=config.backend_unavailable_values,
        backend_flag_aliases=config.backend_flag_aliases,
        backend_capabilities_help_hash=config.backend_capabilities_help_hash,
        client_cpu_time_s=client_cpu_time_s,
        client_cpu_utilization_percent=client_cpu_utilization_percent,
        config=config,
        started_at=started_at,
        ended_at=ended_at,
        trial_wall_clock_time_s=trial_wall_clock_time_s,
    )
    write_json(run_dir / "summary.json", summary)
    if config.telemetry != "none":
        write_json(run_dir / "telemetry_summary.json", summary.telemetry_summary)
        write_json(run_dir / "telemetry_capabilities.json", summary.telemetry_summary.get("telemetry_capabilities"))

    comparison = None
    if prediction is not None:
        comparison = compare_prediction(config.run_id, prediction, summary, config)
        write_json(run_dir / "comparison.json", comparison)

    metadata = {
        "schema_version": config.schema_version,
        "run_id": config.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact_files": sorted(_artifact_files(run_dir) + ["metadata.json"]),
        "hardware": to_dict(hardware) if hardware is not None else None,
    }
    write_json(run_dir / "metadata.json", metadata)
    return EndpointBenchmarkRun(run_dir=run_dir, summary=summary, comparison=comparison)


def _run_requests(config: EndpointBenchmarkConfig, request_fn: RequestFn) -> list[RequestRecord]:
    workers = max(1, config.concurrency)
    records: list[RequestRecord] = []
    next_request_id = 0
    soak_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        while True:
            request_count = config.num_requests if next_request_id == 0 else workers
            futures = {}
            for request_id in range(next_request_id, next_request_id + request_count):
                submit_time = time.time()
                futures[executor.submit(_timed_request, config, request_fn, request_id, submit_time)] = (
                    request_id,
                    submit_time,
                )
            next_request_id += request_count
            for future in as_completed(futures):
                request_id, submit_time = futures[future]
                try:
                    records.append(future.result())
                except Exception as exc:
                    now = time.time()
                    status = _request_exception_status(exc)
                    records.append(
                        RequestRecord(
                            request_id=request_id,
                            start_time=now,
                            end_time=now,
                            latency_s=0.0,
                            status=status,
                            error=f"{exc.__class__.__name__}: {exc}",
                            error_reason=status,
                            client_status=status,
                            client_submit_time=submit_time,
                            client_queue_s=max(0.0, now - submit_time),
                        )
                    )
            if config.soak_duration_s is None or config.soak_duration_s <= 0:
                break
            if time.perf_counter() - soak_started >= config.soak_duration_s:
                break
    return records


def _timed_request(
    config: EndpointBenchmarkConfig,
    request_fn: RequestFn,
    request_id: int,
    submit_time: float,
) -> RequestRecord:
    client_start = time.time()
    client_queue_s = max(0.0, client_start - submit_time)
    try:
        record = request_fn(config, request_id)
    except Exception as exc:
        end = time.time()
        status = _request_exception_status(exc)
        return RequestRecord(
            request_id=request_id,
            start_time=client_start,
            end_time=end,
            latency_s=max(0.0, end - client_start),
            status=status,
            error=f"{exc.__class__.__name__}: {exc}",
            error_reason=status,
            client_status=status,
            client_submit_time=submit_time,
            client_start_time=client_start,
            client_queue_s=client_queue_s,
        )
    return replace(
        record,
        client_status=record.client_status or record.status,
        client_submit_time=submit_time,
        client_start_time=client_start,
        client_queue_s=client_queue_s,
    )


def send_chat_completion_request(config: EndpointBenchmarkConfig, request_id: int) -> RequestRecord:
    start = time.time()
    payload = {
        "model": config.model,
        "messages": [{"role": "user", "content": config.prompt}],
        "max_tokens": config.max_tokens,
        "temperature": 0,
    }
    if config.stream:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
    data = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        _endpoint_url(config.base_url, config.endpoint),
        data=data,
        headers={"Content-Type": "application/json", "Authorization": _authorization_header(config.api_key_env)},
        method="POST",
    )
    try:
        # The URL scheme and host are validated before request construction.
        with request.urlopen(http_request, timeout=config.timeout_s) as response:  # nosec B310
            if config.stream:
                return _streaming_request_record(response, request_id=request_id, start=start)
            body = response.read().decode("utf-8")
            status_code = response.status
        parsed = json.loads(body) if body else {}
        usage = parsed.get("usage") or {}
        prompt_tokens = _int_value(usage.get("prompt_tokens"))
        completion_tokens = _int_value(usage.get("completion_tokens"))
        total_tokens = _int_value(usage.get("total_tokens")) or prompt_tokens + completion_tokens
        end = time.time()
        status = "ok" if 200 <= status_code < 300 else f"http_{status_code}"
        return RequestRecord(
            request_id=request_id,
            start_time=start,
            end_time=end,
            latency_s=end - start,
            status=status,
            error_reason=None if status == "ok" else status,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            finish_reason=_response_finish_reason(parsed),
            http_status=status_code,
            client_status=status,
            token_count_source="response_usage" if usage else None,
        )
    except error.HTTPError as exc:
        end = time.time()
        status = f"http_{exc.code}"
        return RequestRecord(
            request_id=request_id,
            start_time=start,
            end_time=end,
            latency_s=end - start,
            status=status,
            error=str(exc),
            error_reason=status,
            http_status=exc.code,
            client_status=status,
        )
    except Exception as exc:
        end = time.time()
        status = _request_exception_status(exc)
        return RequestRecord(
            request_id=request_id,
            start_time=start,
            end_time=end,
            latency_s=end - start,
            status=status,
            error=f"{exc.__class__.__name__}: {exc}",
            error_reason=status,
            client_status=status,
        )


def _streaming_request_record(response: object, *, request_id: int, start: float) -> RequestRecord:
    status_code = getattr(response, "status", 200)
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    chunk_count = 0
    usage_received = False
    first_chunk_time: float | None = None
    last_chunk_time: float | None = None
    finish_reason: str | None = None
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").strip() if isinstance(raw_line, bytes) else str(raw_line).strip()
        if not line or not line.startswith("data:"):
            continue
        payload = line.removeprefix("data:").strip()
        if payload == "[DONE]":
            continue
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        usage = parsed.get("usage") if isinstance(parsed, dict) else None
        if isinstance(usage, dict):
            usage_received = any(key in usage for key in ("prompt_tokens", "completion_tokens", "total_tokens"))
            prompt_tokens = _int_value(usage.get("prompt_tokens")) or prompt_tokens
            completion_tokens = _int_value(usage.get("completion_tokens")) or completion_tokens
            total_tokens = _int_value(usage.get("total_tokens")) or total_tokens
        choices = parsed.get("choices") if isinstance(parsed, dict) else None
        if not isinstance(choices, list):
            continue
        finish_reason = finish_reason or _choices_finish_reason(choices)
        if not any(_choice_has_stream_content(choice) for choice in choices if isinstance(choice, dict)):
            continue
        chunk_time = time.time()
        if first_chunk_time is None:
            first_chunk_time = chunk_time
        last_chunk_time = chunk_time
        chunk_count += 1
    end = time.time()
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens
    ttft_s = first_chunk_time - start if first_chunk_time is not None else None
    tpot_s = (
        (last_chunk_time - first_chunk_time) / (chunk_count - 1)
        if first_chunk_time is not None and last_chunk_time is not None and chunk_count > 1
        else None
    )
    has_output = first_chunk_time is not None
    successful_status = 200 <= status_code < 300 and has_output
    status = "ok" if successful_status else ("stream_no_output" if 200 <= status_code < 300 else f"http_{status_code}")
    return RequestRecord(
        request_id=request_id,
        start_time=start,
        end_time=end,
        latency_s=end - start,
        status=status,
        error=None if successful_status else ("Streaming response contained no output content." if 200 <= status_code < 300 else None),
        error_reason=None if successful_status else status,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        ttft_s=ttft_s,
        tpot_s=tpot_s,
        finish_reason=finish_reason,
        http_status=status_code,
        client_status=status,
        timing_source="openai_stream_chunks" if first_chunk_time is not None else None,
        token_count_source="response_usage" if usage_received else None,
    )


def _choice_has_stream_content(choice: dict[str, object]) -> bool:
    if choice.get("text"):
        return True
    delta = choice.get("delta")
    if isinstance(delta, dict) and (delta.get("content") or delta.get("reasoning_content")):
        return True
    message = choice.get("message")
    return isinstance(message, dict) and bool(message.get("content"))


def _response_finish_reason(parsed: dict[str, object]) -> str | None:
    choices = parsed.get("choices")
    return _choices_finish_reason(choices) if isinstance(choices, list) else None


def _choices_finish_reason(choices: list[object]) -> str | None:
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        reason = choice.get("finish_reason")
        if reason is not None:
            return str(reason)
    return None


def _request_exception_status(exc: Exception) -> str:
    if _is_request_timeout(exc):
        return "request_timeout"
    return "error"


def _is_request_timeout(exc: Exception) -> bool:
    if isinstance(exc, TimeoutError | socket.timeout):
        return True
    if isinstance(exc, error.URLError):
        reason = getattr(exc, "reason", None)
        if isinstance(reason, TimeoutError | socket.timeout):
            return True
        return "timed out" in str(reason).lower()
    return "timed out" in str(exc).lower()


def summarize_requests(
    run_id: str,
    records: list[RequestRecord],
    wall_time_s: float,
    power_samples: list[PowerSampleRecord] | None = None,
    telemetry: TelemetryCapture | None = None,
    warmup_requests: int = 0,
    steady_state_duration_s: float | None = None,
    soak_duration_s: float | None = None,
    idle_power_watts: float | None = None,
    idle_sample_count: int = 0,
    configured_concurrency: int | None = None,
    configured_num_requests: int | None = None,
    backend_name: str | None = None,
    backend_version: str | None = None,
    backend_launch_command: list[str] | None = None,
    backend_launch_command_hash: str | None = None,
    backend_effective_values: dict[str, object] | None = None,
    backend_applied_configuration: dict[str, object] | None = None,
    backend_omitted_values: dict[str, object] | None = None,
    backend_unsupported_values: dict[str, object] | None = None,
    backend_unavailable_values: dict[str, object] | None = None,
    backend_flag_aliases: dict[str, object] | None = None,
    backend_capabilities_help_hash: str | None = None,
    client_cpu_time_s: float | None = None,
    client_cpu_utilization_percent: float | None = None,
    config: EndpointBenchmarkConfig | None = None,
    started_at: str | None = None,
    ended_at: str | None = None,
    trial_wall_clock_time_s: float | None = None,
) -> EndpointBenchmarkSummary:
    power_samples = power_samples or []
    telemetry = telemetry or TelemetryCapture(provider=None, samples=[], warnings=[])
    if config is not None:
        backend_name = backend_name or config.backend_name
        backend_version = backend_version or config.backend_version
        backend_launch_command = backend_launch_command or config.backend_launch_command
        backend_launch_command_hash = backend_launch_command_hash or config.backend_launch_command_hash
        backend_effective_values = backend_effective_values or config.backend_effective_values
        backend_applied_configuration = backend_applied_configuration or config.backend_applied_configuration
        backend_omitted_values = backend_omitted_values or config.backend_omitted_values
        backend_unsupported_values = backend_unsupported_values or config.backend_unsupported_values
        backend_unavailable_values = backend_unavailable_values or config.backend_unavailable_values
        backend_flag_aliases = backend_flag_aliases or config.backend_flag_aliases
        backend_capabilities_help_hash = backend_capabilities_help_hash or config.backend_capabilities_help_hash
    successful = [record for record in records if record.status == "ok"]
    measured_records = _measurement_window_records(
        records,
        warmup_requests=warmup_requests,
        steady_state_duration_s=steady_state_duration_s,
    )
    measured_successful = [record for record in measured_records if record.status == "ok"]
    measurement_window_applied = warmup_requests > 0 or bool(steady_state_duration_s and steady_state_duration_s > 0)
    phase_power_samples = _phase_power_samples(
        power_samples,
        records,
        warmup_requests=warmup_requests,
        steady_state_duration_s=steady_state_duration_s,
    )
    warmup_power_samples = [sample for sample in phase_power_samples if sample.phase == "warmup"]
    measurement_power_samples = [sample for sample in phase_power_samples if sample.phase == "measurement"]
    latencies = [record.latency_s for record in measured_successful]
    ttfts_ms = [record.ttft_s * 1000.0 for record in measured_successful if record.ttft_s is not None]
    tpots_ms = [record.tpot_s * 1000.0 for record in measured_successful if record.tpot_s is not None]
    client_queue_s = [record.client_queue_s for record in measured_records if record.client_queue_s is not None]
    prompt_tokens = sum(record.prompt_tokens for record in measured_successful)
    completion_tokens = sum(record.completion_tokens for record in measured_successful)
    total_tokens = sum(record.total_tokens for record in measured_successful)
    measurement_duration = _measurement_duration(measured_records, wall_time_s, window_applied=measurement_window_applied)
    concurrency_coverage = _concurrency_coverage(
        configured_concurrency=configured_concurrency,
        measured_request_count=len(measured_records),
    )
    telemetry_summary = summarize_telemetry(
        measurement_power_samples,
        measurement_duration or 0.0,
        total_tokens,
        provider=telemetry.provider,
        warnings=telemetry.warnings,
    )
    client_issue_rate = _client_issue_rate(measured_records, measurement_duration)
    request_backlog = _interval_backlog(measured_records, measurement_duration, weight="requests")
    token_backlog = _interval_backlog(measured_records, measurement_duration, weight="tokens")
    load_saturation_signal = _load_saturation_signal(
        telemetry_summary.average_gpu_util_percent,
        telemetry_summary.max_gpu_util_percent,
    )
    average_memory_bandwidth_util_percent = telemetry_summary.average_memory_util_percent
    workload_description = _workload_description_with_actual_outputs(config, measured_successful)
    outcome = _trial_outcome(measured_records)
    load_sufficiency = {
        "schema_version": "load-sufficiency-trial/v1",
        "gpu_utilization_available": telemetry_summary.average_gpu_util_percent is not None
        or telemetry_summary.max_gpu_util_percent is not None,
        "average_gpu_util_percent": telemetry_summary.average_gpu_util_percent,
        "max_gpu_util_percent": telemetry_summary.max_gpu_util_percent,
        "gpu_saturation_threshold_percent": LOAD_GPU_SATURATION_THRESHOLD_PERCENT,
        "client_issue_rate_req_s": _round_or_none(client_issue_rate),
        "avg_request_backlog": _round_or_none(request_backlog["avg"]),
        "max_request_backlog": _round_or_none(request_backlog["max"]),
        "request_backlog_source": "client_observed_inflight_requests",
        "avg_token_backlog": _round_or_none(token_backlog["avg"]),
        "max_token_backlog": _round_or_none(token_backlog["max"]),
        "token_backlog_source": "observed_total_tokens_in_flight",
        "load_saturation_signal": load_saturation_signal,
    }
    active_power = _active_power(telemetry_summary.average_power_watts, idle_power_watts)
    active_energy = active_power * measurement_duration if active_power is not None and measurement_duration is not None else None
    active_joules_per_token = active_energy / total_tokens if active_energy is not None and total_tokens > 0 else None
    energy_accounting = "idle_subtracted" if active_energy is not None else "raw"
    joules_per_generated_token = (
        telemetry_summary.energy_joules / completion_tokens
        if telemetry_summary.energy_joules is not None and completion_tokens > 0
        else None
    )
    active_joules_per_generated_token = active_energy / completion_tokens if active_energy is not None and completion_tokens > 0 else None
    tokens_per_joule = total_tokens / telemetry_summary.energy_joules if telemetry_summary.energy_joules not in {None, 0} else None
    active_tokens_per_joule = total_tokens / active_energy if active_energy not in {None, 0} else None
    active_tokens_per_watt = (
        total_tokens / measurement_duration / active_power
        if active_power is not None and active_power > 0 and measurement_duration is not None and measurement_duration > 0
        else None
    )
    measurement_quality = {
        "schema_version": "measurement-quality/v1",
        "warmup_requests": warmup_requests,
        "steady_state_requested_duration_s": steady_state_duration_s,
        "steady_state_requests": len(measured_successful),
        "steady_state_duration_s": _round_or_none(measurement_duration),
        "measurement_duration_s": _round_or_none(measurement_duration),
        "measured_requests": len(measured_records),
        "measured_successful_requests": len(measured_successful),
        "measured_failed_requests": len(measured_records) - len(measured_successful),
        "configured_concurrency": configured_concurrency,
        "configured_num_requests": configured_num_requests,
        "effective_concurrency_limit": _effective_concurrency_limit(
            configured_concurrency,
            configured_num_requests,
            warmup_requests=warmup_requests,
        ),
        "concurrency_coverage": concurrency_coverage,
        "idle_power_watts": _round_or_none(idle_power_watts),
        "idle_sample_count": idle_sample_count,
        "idle_baseline_source": _idle_baseline_source(idle_power_watts, idle_sample_count),
        "idle_baseline_phase": "pre_run" if idle_power_watts is not None or idle_sample_count > 0 else "unavailable",
        "warmup_power_sample_count": len(warmup_power_samples),
        "measurement_power_sample_count": len(measurement_power_samples),
        "warmup_average_power_watts": _round_or_none(_average_power(warmup_power_samples)),
        "measurement_average_power_watts": _round_or_none(_average_power(measurement_power_samples)),
        "energy_window": "measurement",
        "soak_requested_duration_s": soak_duration_s,
        "soak_effective_duration_s": _round_or_none(wall_time_s),
        "ttft_sample_count": len(ttfts_ms),
        "tpot_sample_count": len(tpots_ms),
        "p99_ttft_ms": _round_or_none(_percentile(ttfts_ms, 99)),
        "p99_tpot_ms": _round_or_none(_percentile(tpots_ms, 99)),
        "timing_source": _timing_source(measured_successful),
        "token_count_source": _token_count_source(measured_successful),
        "client_cpu_time_s": _round_or_none(client_cpu_time_s),
        "client_cpu_utilization_percent": _round_or_none(client_cpu_utilization_percent),
        "client_queue_sample_count": len(client_queue_s),
        "avg_client_queue_s": _round_or_none(_mean(client_queue_s)),
        "p50_client_queue_s": _round_or_none(_percentile(client_queue_s, 50)),
        "p95_client_queue_s": _round_or_none(_percentile(client_queue_s, 95)),
        "p99_client_queue_s": _round_or_none(_percentile(client_queue_s, 99)),
        "max_client_queue_s": _round_or_none(max(client_queue_s) if client_queue_s else None),
        "client_saturation_signal": _client_saturation_signal(client_cpu_utilization_percent),
        "client_issue_rate_req_s": _round_or_none(client_issue_rate),
        "avg_request_backlog": _round_or_none(request_backlog["avg"]),
        "max_request_backlog": _round_or_none(request_backlog["max"]),
        "avg_token_backlog": _round_or_none(token_backlog["avg"]),
        "max_token_backlog": _round_or_none(token_backlog["max"]),
        "load_saturation_signal": load_saturation_signal,
        "load_sufficiency": load_sufficiency,
        "energy_accounting": energy_accounting,
        "tokens_per_joule": _round_or_none(tokens_per_joule),
        "active_tokens_per_joule": _round_or_none(active_tokens_per_joule),
        "joules_per_generated_token": _round_or_none(joules_per_generated_token),
        "active_joules_per_generated_token": _round_or_none(active_joules_per_generated_token),
        "peak_gpu_memory_mb": telemetry_summary.max_memory_used_mb,
        "average_memory_bandwidth_util_percent": average_memory_bandwidth_util_percent,
        "phase_energy_attribution": "unavailable_without_phase_markers",
        "outcome": outcome,
        "stability_classification": "single_trial",
    }
    token_warnings = _token_count_warnings(measured_successful)
    concurrency_warnings = _concurrency_warnings(
        configured_concurrency=configured_concurrency,
        configured_num_requests=configured_num_requests,
        measured_request_count=len(measured_records),
    )
    client_warnings = _client_warnings(client_cpu_utilization_percent, measurement_duration)
    return EndpointBenchmarkSummary(
        run_id=run_id,
        total_requests=len(records),
        successful_requests=len(successful),
        failed_requests=len(records) - len(successful),
        wall_time_s=wall_time_s,
        request_rate_req_s=len(measured_successful) / measurement_duration if measurement_duration and measurement_duration > 0 else 0.0,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        output_tokens_s=completion_tokens / measurement_duration if measurement_duration and measurement_duration > 0 else 0.0,
        total_tokens_s=total_tokens / measurement_duration if measurement_duration and measurement_duration > 0 else 0.0,
        avg_latency_s=sum(latencies) / len(latencies) if latencies else None,
        p50_latency_s=_percentile(latencies, 50),
        p95_latency_s=_percentile(latencies, 95),
        p99_latency_s=_percentile(latencies, 99),
        avg_ttft_ms=_round_or_none(_mean(ttfts_ms)),
        p50_ttft_ms=_round_or_none(_percentile(ttfts_ms, 50)),
        p95_ttft_ms=_round_or_none(_percentile(ttfts_ms, 95)),
        p99_ttft_ms=_round_or_none(_percentile(ttfts_ms, 99)),
        avg_tpot_ms=_round_or_none(_mean(tpots_ms)),
        p50_tpot_ms=_round_or_none(_percentile(tpots_ms, 50)),
        p95_tpot_ms=_round_or_none(_percentile(tpots_ms, 95)),
        p99_tpot_ms=_round_or_none(_percentile(tpots_ms, 99)),
        ttft_sample_count=len(ttfts_ms),
        tpot_sample_count=len(tpots_ms),
        timing_source=_timing_source(measured_successful),
        power_sample_count=telemetry_summary.power_sample_count,
        average_power_watts=telemetry_summary.average_power_watts,
        min_power_watts=telemetry_summary.min_power_watts,
        max_power_watts=telemetry_summary.max_power_watts,
        peak_power_watts=telemetry_summary.peak_power_watts,
        power_stddev_watts=telemetry_summary.power_stddev_watts,
        power_sampling_duration_s=telemetry_summary.power_sampling_duration_s,
        power_sampling_rate_hz=telemetry_summary.power_sampling_rate_hz,
        idle_power_watts=_round_or_none(idle_power_watts),
        warmup_power_sample_count=len(warmup_power_samples),
        measurement_power_sample_count=len(measurement_power_samples),
        warmup_average_power_watts=_round_or_none(_average_power(warmup_power_samples)),
        measurement_average_power_watts=_round_or_none(_average_power(measurement_power_samples)),
        active_power_watts=_round_or_none(active_power),
        active_energy_joules=_round_or_none(active_energy),
        idle_subtracted_energy_joules=_round_or_none(active_energy),
        energy_joules=telemetry_summary.energy_joules,
        energy_accounting=energy_accounting,
        joules_per_token=telemetry_summary.joules_per_token,
        active_joules_per_token=_round_or_none(active_joules_per_token),
        joules_per_generated_token=_round_or_none(joules_per_generated_token),
        active_joules_per_generated_token=_round_or_none(active_joules_per_generated_token),
        tokens_per_second_per_watt=telemetry_summary.tokens_per_second_per_watt,
        active_tokens_per_second_per_watt=_round_or_none(active_tokens_per_watt),
        tokens_per_joule=_round_or_none(tokens_per_joule),
        active_tokens_per_joule=_round_or_none(active_tokens_per_joule),
        warmup_requests=warmup_requests,
        steady_state_requests=len(measured_successful),
        steady_state_duration_s=_round_or_none(measurement_duration),
        steady_state_total_tokens=total_tokens,
        steady_state_total_tokens_s=total_tokens / measurement_duration if measurement_duration and measurement_duration > 0 else None,
        steady_state_request_rate_req_s=len(measured_successful) / measurement_duration if measurement_duration and measurement_duration > 0 else None,
        measurement_quality=measurement_quality,
        stability_classification="single_trial",
        observed_memory_mb=telemetry_summary.max_memory_used_mb,
        peak_gpu_memory_mb=telemetry_summary.max_memory_used_mb,
        average_gpu_util_percent=telemetry_summary.average_gpu_util_percent,
        max_gpu_util_percent=telemetry_summary.max_gpu_util_percent,
        average_memory_util_percent=telemetry_summary.average_memory_util_percent,
        max_memory_util_percent=telemetry_summary.max_memory_util_percent,
        average_memory_bandwidth_util_percent=average_memory_bandwidth_util_percent,
        average_temperature_c=telemetry_summary.average_temperature_c,
        max_temperature_c=telemetry_summary.max_temperature_c,
        temperature_rise_c=telemetry_summary.temperature_rise_c,
        temperature_slope_c_per_min=telemetry_summary.temperature_slope_c_per_min,
        thermal_stability_classification=telemetry_summary.thermal_stability_classification,
        average_sm_clock_mhz=telemetry_summary.average_sm_clock_mhz,
        average_memory_clock_mhz=telemetry_summary.average_memory_clock_mhz,
        telemetry_provider=telemetry_summary.telemetry_provider,
        telemetry_available=telemetry_summary.telemetry_available,
        telemetry_quality=telemetry_summary.telemetry_quality,
        telemetry_notes=telemetry_summary.telemetry_notes,
        telemetry_summary=to_dict(telemetry_summary),
        warnings=sorted({*telemetry_summary.telemetry_warnings, *token_warnings, *concurrency_warnings, *client_warnings}),
        measurement_duration_s=_round_or_none(measurement_duration),
        measured_requests=len(measured_records),
        measured_successful_requests=len(measured_successful),
        measured_failed_requests=len(measured_records) - len(measured_successful),
        token_count_source=_token_count_source(measured_successful),
        client_cpu_time_s=_round_or_none(client_cpu_time_s),
        client_cpu_utilization_percent=_round_or_none(client_cpu_utilization_percent),
        client_queue_sample_count=len(client_queue_s),
        avg_client_queue_s=_round_or_none(_mean(client_queue_s)),
        p50_client_queue_s=_round_or_none(_percentile(client_queue_s, 50)),
        p95_client_queue_s=_round_or_none(_percentile(client_queue_s, 95)),
        p99_client_queue_s=_round_or_none(_percentile(client_queue_s, 99)),
        max_client_queue_s=_round_or_none(max(client_queue_s) if client_queue_s else None),
        client_issue_rate_req_s=_round_or_none(client_issue_rate),
        avg_request_backlog=_round_or_none(request_backlog["avg"]),
        max_request_backlog=_round_or_none(request_backlog["max"]),
        avg_token_backlog=_round_or_none(token_backlog["avg"]),
        max_token_backlog=_round_or_none(token_backlog["max"]),
        load_saturation_signal=load_saturation_signal,
        load_sufficiency=load_sufficiency,
        backend_name=backend_name,
        backend_version=backend_version,
        backend_launch_command=list(backend_launch_command or []),
        backend_launch_command_hash=backend_launch_command_hash,
        backend_effective_values=dict(backend_effective_values or {}),
        backend_applied_configuration=dict(backend_applied_configuration or {}),
        backend_omitted_values=dict(backend_omitted_values or {}),
        backend_unsupported_values=dict(backend_unsupported_values or {}),
        backend_unavailable_values=dict(backend_unavailable_values or {}),
        backend_flag_aliases=dict(backend_flag_aliases or {}),
        backend_capabilities_help_hash=backend_capabilities_help_hash,
        trial_id=run_id,
        experiment_campaign_id=config.experiment_campaign_id if config else None,
        parent_run_id=config.parent_run_id if config else None,
        started_at=started_at,
        ended_at=ended_at,
        hostname=socket.gethostname(),
        repository_commit=_git_commit(),
        dirty_tree=_git_dirty(),
        python_version=platform.python_version(),
        cuda_version=_cuda_version(),
        gpu_driver_version=_gpu_driver_version(),
        backend_health_status=config.backend_health_status if config else None,
        model_id=config.model if config else None,
        model_revision=config.model_revision if config else None,
        model_access_status=config.model_access_status if config else None,
        tokenizer_id=config.tokenizer_id if config else None,
        tokenizer_revision=config.tokenizer_revision if config else None,
        objective_mode=config.objective_mode if config else None,
        candidate_source=config.candidate_source if config else None,
        workload_description=workload_description,
        backend_started_at=config.backend_started_at if config else None,
        backend_ready_at=config.backend_ready_at if config else None,
        backend_startup_time_s=config.backend_startup_time_s if config else None,
        model_load_time_s=config.model_load_time_s if config else None,
        trial_wall_clock_time_s=_round_or_none(trial_wall_clock_time_s),
        outcome=outcome,
    )


def aggregate_benchmark_summaries(run_id: str, summaries: list[EndpointBenchmarkSummary]) -> EndpointBenchmarkSummary:
    if not summaries:
        raise ValueError("At least one summary is required for aggregation.")
    if len(summaries) == 1:
        return summaries[0]
    first = summaries[0]
    total_requests = sum(summary.total_requests for summary in summaries)
    successful_requests = sum(summary.successful_requests for summary in summaries)
    failed_requests = sum(summary.failed_requests for summary in summaries)
    wall_time_s = sum(summary.wall_time_s for summary in summaries)
    client_cpu_time_s = sum(summary.client_cpu_time_s for summary in summaries if summary.client_cpu_time_s is not None) or None
    client_cpu_utilization_percent = (
        client_cpu_time_s / wall_time_s * 100.0
        if client_cpu_time_s is not None and wall_time_s > 0
        else _mean([_float(summary.client_cpu_utilization_percent) for summary in summaries if _float(summary.client_cpu_utilization_percent) is not None])
    )
    client_queue_sample_count = sum(summary.client_queue_sample_count for summary in summaries)
    client_issue_rate_req_s = _mean([_float(summary.client_issue_rate_req_s) for summary in summaries if _float(summary.client_issue_rate_req_s) is not None])
    avg_request_backlog = _mean([_float(summary.avg_request_backlog) for summary in summaries if _float(summary.avg_request_backlog) is not None])
    max_request_backlog = max([summary.max_request_backlog for summary in summaries if summary.max_request_backlog is not None], default=None)
    avg_token_backlog = _mean([_float(summary.avg_token_backlog) for summary in summaries if _float(summary.avg_token_backlog) is not None])
    max_token_backlog = max([summary.max_token_backlog for summary in summaries if summary.max_token_backlog is not None], default=None)
    prompt_tokens = sum(summary.prompt_tokens for summary in summaries)
    completion_tokens = sum(summary.completion_tokens for summary in summaries)
    total_tokens = sum(summary.total_tokens for summary in summaries)
    energy_joules = sum([summary.energy_joules for summary in summaries if summary.energy_joules is not None]) or None
    active_energy_joules = sum([summary.active_energy_joules for summary in summaries if summary.active_energy_joules is not None]) or None
    energy_accounting = "idle_subtracted" if active_energy_joules is not None else "raw"
    tokens_per_joule = total_tokens / energy_joules if energy_joules not in {None, 0} else None
    active_tokens_per_joule = total_tokens / active_energy_joules if active_energy_joules not in {None, 0} else None
    joules_per_generated_token = energy_joules / completion_tokens if energy_joules is not None and completion_tokens > 0 else None
    active_joules_per_generated_token = (
        active_energy_joules / completion_tokens
        if active_energy_joules is not None and completion_tokens > 0
        else None
    )
    metrics = {
        "total_tokens_s": [_float(summary.total_tokens_s) for summary in summaries],
        "output_tokens_s": [_float(summary.output_tokens_s) for summary in summaries],
        "request_rate_req_s": [_float(summary.request_rate_req_s) for summary in summaries],
        "p95_latency_s": [_float(summary.p95_latency_s) for summary in summaries],
        "p95_ttft_ms": [_float(summary.p95_ttft_ms) for summary in summaries],
        "p99_ttft_ms": [_float(summary.p99_ttft_ms) for summary in summaries],
        "p95_tpot_ms": [_float(summary.p95_tpot_ms) for summary in summaries],
        "p99_tpot_ms": [_float(summary.p99_tpot_ms) for summary in summaries],
        "joules_per_token": [_float(summary.joules_per_token) for summary in summaries],
        "active_joules_per_token": [_float(summary.active_joules_per_token) for summary in summaries],
        "tokens_per_joule": [_float(summary.tokens_per_joule) for summary in summaries],
        "joules_per_generated_token": [_float(summary.joules_per_generated_token) for summary in summaries],
    }
    metric_values = {key: [value for value in values if value is not None] for key, values in metrics.items()}
    confidence_intervals = {
        key: _confidence_interval(values)
        for key, values in metric_values.items()
        if values
    }
    stability = _stability_classification(metric_values)
    trial_statistics = {
        "schema_version": "trial-statistics/v1",
        "trial_count": len(summaries),
        "stability_classification": stability,
        "metrics": {
            key: _metric_stats(values)
            for key, values in metric_values.items()
            if values
        },
    }
    quality = dict(first.measurement_quality or {})
    quality.update(
        {
            "schema_version": "measurement-quality/v1",
            "trial_count": len(summaries),
            "stability_classification": stability,
            "energy_accounting": energy_accounting,
            "energy_window": "measurement",
            "warmup_power_sample_count": sum(summary.warmup_power_sample_count for summary in summaries),
            "measurement_power_sample_count": sum(summary.measurement_power_sample_count for summary in summaries),
            "warmup_average_power_watts": _round_or_none(
                _mean([_float(summary.warmup_average_power_watts) for summary in summaries if _float(summary.warmup_average_power_watts) is not None])
            ),
            "measurement_average_power_watts": _round_or_none(
                _mean([_float(summary.measurement_average_power_watts) for summary in summaries if _float(summary.measurement_average_power_watts) is not None])
            ),
            "tokens_per_joule": _round_or_none(tokens_per_joule),
            "active_tokens_per_joule": _round_or_none(active_tokens_per_joule),
            "joules_per_generated_token": _round_or_none(joules_per_generated_token),
            "active_joules_per_generated_token": _round_or_none(active_joules_per_generated_token),
            "measurement_duration_s": sum(summary.measurement_duration_s or 0.0 for summary in summaries) or None,
            "measured_requests": sum(summary.measured_requests or 0 for summary in summaries),
            "measured_successful_requests": sum(summary.measured_successful_requests or 0 for summary in summaries),
            "measured_failed_requests": sum(summary.measured_failed_requests or 0 for summary in summaries),
            "token_count_source": _aggregate_token_count_source(summaries),
            "client_cpu_time_s": _round_or_none(client_cpu_time_s),
            "client_cpu_utilization_percent": _round_or_none(client_cpu_utilization_percent),
            "client_queue_sample_count": client_queue_sample_count,
            "avg_client_queue_s": _round_or_none(_weighted_queue_mean(summaries)),
            "p50_client_queue_s": _round_or_none(_mean([_float(summary.p50_client_queue_s) for summary in summaries if _float(summary.p50_client_queue_s) is not None])),
            "p95_client_queue_s": _round_or_none(_mean([_float(summary.p95_client_queue_s) for summary in summaries if _float(summary.p95_client_queue_s) is not None])),
            "p99_client_queue_s": _round_or_none(_mean([_float(summary.p99_client_queue_s) for summary in summaries if _float(summary.p99_client_queue_s) is not None])),
            "max_client_queue_s": _round_or_none(max([summary.max_client_queue_s for summary in summaries if summary.max_client_queue_s is not None], default=None)),
            "client_saturation_signal": _client_saturation_signal(client_cpu_utilization_percent),
            "client_issue_rate_req_s": _round_or_none(client_issue_rate_req_s),
            "avg_request_backlog": _round_or_none(avg_request_backlog),
            "max_request_backlog": _round_or_none(max_request_backlog),
            "avg_token_backlog": _round_or_none(avg_token_backlog),
            "max_token_backlog": _round_or_none(max_token_backlog),
            "load_saturation_signal": _aggregate_load_saturation_signal(summaries),
            "load_sufficiency": _aggregate_load_sufficiency(summaries),
        }
    )
    return replace(
        first,
        run_id=run_id,
        total_requests=total_requests,
        successful_requests=successful_requests,
        failed_requests=failed_requests,
        wall_time_s=wall_time_s,
        request_rate_req_s=_mean(metric_values["request_rate_req_s"]) or 0.0,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        output_tokens_s=_mean(metric_values["output_tokens_s"]) or 0.0,
        total_tokens_s=_mean(metric_values["total_tokens_s"]) or 0.0,
        avg_latency_s=_mean([_float(summary.avg_latency_s) for summary in summaries if _float(summary.avg_latency_s) is not None]),
        p50_latency_s=_mean([_float(summary.p50_latency_s) for summary in summaries if _float(summary.p50_latency_s) is not None]),
        p95_latency_s=_mean(metric_values["p95_latency_s"]),
        p99_latency_s=_mean([_float(summary.p99_latency_s) for summary in summaries if _float(summary.p99_latency_s) is not None]),
        avg_ttft_ms=_mean([_float(summary.avg_ttft_ms) for summary in summaries if _float(summary.avg_ttft_ms) is not None]),
        p50_ttft_ms=_mean([_float(summary.p50_ttft_ms) for summary in summaries if _float(summary.p50_ttft_ms) is not None]),
        p95_ttft_ms=_mean(metric_values["p95_ttft_ms"]),
        p99_ttft_ms=_mean(metric_values["p99_ttft_ms"]),
        avg_tpot_ms=_mean([_float(summary.avg_tpot_ms) for summary in summaries if _float(summary.avg_tpot_ms) is not None]),
        p50_tpot_ms=_mean([_float(summary.p50_tpot_ms) for summary in summaries if _float(summary.p50_tpot_ms) is not None]),
        p95_tpot_ms=_mean(metric_values["p95_tpot_ms"]),
        p99_tpot_ms=_mean(metric_values["p99_tpot_ms"]),
        ttft_sample_count=sum(summary.ttft_sample_count for summary in summaries),
        tpot_sample_count=sum(summary.tpot_sample_count for summary in summaries),
        timing_source=_aggregate_timing_source(summaries),
        power_sample_count=sum(summary.power_sample_count for summary in summaries),
        average_power_watts=_mean([_float(summary.average_power_watts) for summary in summaries if _float(summary.average_power_watts) is not None]),
        peak_power_watts=max([summary.peak_power_watts for summary in summaries if summary.peak_power_watts is not None], default=None),
        warmup_power_sample_count=sum(summary.warmup_power_sample_count for summary in summaries),
        measurement_power_sample_count=sum(summary.measurement_power_sample_count for summary in summaries),
        warmup_average_power_watts=_mean([_float(summary.warmup_average_power_watts) for summary in summaries if _float(summary.warmup_average_power_watts) is not None]),
        measurement_average_power_watts=_mean(
            [_float(summary.measurement_average_power_watts) for summary in summaries if _float(summary.measurement_average_power_watts) is not None]
        ),
        energy_joules=energy_joules,
        energy_accounting=energy_accounting,
        joules_per_token=_mean(metric_values["joules_per_token"]),
        active_energy_joules=active_energy_joules,
        idle_subtracted_energy_joules=active_energy_joules,
        active_joules_per_token=_mean(metric_values["active_joules_per_token"]),
        joules_per_generated_token=_round_or_none(joules_per_generated_token),
        active_joules_per_generated_token=_round_or_none(active_joules_per_generated_token),
        tokens_per_second_per_watt=_mean([_float(summary.tokens_per_second_per_watt) for summary in summaries if _float(summary.tokens_per_second_per_watt) is not None]),
        active_tokens_per_second_per_watt=_mean(
            [
                _float(summary.active_tokens_per_second_per_watt)
                for summary in summaries
                if _float(summary.active_tokens_per_second_per_watt) is not None
            ]
        ),
        tokens_per_joule=_round_or_none(tokens_per_joule),
        active_tokens_per_joule=_round_or_none(active_tokens_per_joule),
        temperature_rise_c=_mean([_float(summary.temperature_rise_c) for summary in summaries if _float(summary.temperature_rise_c) is not None]),
        temperature_slope_c_per_min=_mean([_float(summary.temperature_slope_c_per_min) for summary in summaries if _float(summary.temperature_slope_c_per_min) is not None]),
        thermal_stability_classification=_aggregate_thermal_classification(summaries),
        peak_gpu_memory_mb=max([summary.peak_gpu_memory_mb for summary in summaries if summary.peak_gpu_memory_mb is not None], default=None),
        average_memory_bandwidth_util_percent=_mean(
            [
                _float(summary.average_memory_bandwidth_util_percent)
                for summary in summaries
                if _float(summary.average_memory_bandwidth_util_percent) is not None
            ]
        ),
        steady_state_requests=sum(summary.steady_state_requests or 0 for summary in summaries),
        steady_state_duration_s=sum(summary.steady_state_duration_s or 0.0 for summary in summaries) or None,
        steady_state_total_tokens=sum(summary.steady_state_total_tokens or 0 for summary in summaries),
        steady_state_total_tokens_s=_mean([_float(summary.steady_state_total_tokens_s) for summary in summaries if _float(summary.steady_state_total_tokens_s) is not None]),
        steady_state_request_rate_req_s=_mean([_float(summary.steady_state_request_rate_req_s) for summary in summaries if _float(summary.steady_state_request_rate_req_s) is not None]),
        measurement_quality=quality,
        trial_statistics=trial_statistics,
        stability_classification=stability,
        confidence_intervals=confidence_intervals,
        warnings=sorted({warning for summary in summaries for warning in summary.warnings}),
        measurement_duration_s=sum(summary.measurement_duration_s or 0.0 for summary in summaries) or None,
        measured_requests=sum(summary.measured_requests or 0 for summary in summaries),
        measured_successful_requests=sum(summary.measured_successful_requests or 0 for summary in summaries),
        measured_failed_requests=sum(summary.measured_failed_requests or 0 for summary in summaries),
        token_count_source=_aggregate_token_count_source(summaries),
        client_cpu_time_s=_round_or_none(client_cpu_time_s),
        client_cpu_utilization_percent=_round_or_none(client_cpu_utilization_percent),
        client_queue_sample_count=client_queue_sample_count,
        avg_client_queue_s=_round_or_none(_weighted_queue_mean(summaries)),
        p50_client_queue_s=_round_or_none(_mean([_float(summary.p50_client_queue_s) for summary in summaries if _float(summary.p50_client_queue_s) is not None])),
        p95_client_queue_s=_round_or_none(_mean([_float(summary.p95_client_queue_s) for summary in summaries if _float(summary.p95_client_queue_s) is not None])),
        p99_client_queue_s=_round_or_none(_mean([_float(summary.p99_client_queue_s) for summary in summaries if _float(summary.p99_client_queue_s) is not None])),
        max_client_queue_s=_round_or_none(max([summary.max_client_queue_s for summary in summaries if summary.max_client_queue_s is not None], default=None)),
        client_issue_rate_req_s=_round_or_none(client_issue_rate_req_s),
        avg_request_backlog=_round_or_none(avg_request_backlog),
        max_request_backlog=_round_or_none(max_request_backlog),
        avg_token_backlog=_round_or_none(avg_token_backlog),
        max_token_backlog=_round_or_none(max_token_backlog),
        load_saturation_signal=_aggregate_load_saturation_signal(summaries),
        load_sufficiency=_aggregate_load_sufficiency(summaries),
        trial_id=run_id,
        started_at=_first_available([summary.started_at for summary in summaries]),
        ended_at=_last_available([summary.ended_at for summary in summaries]),
        trial_wall_clock_time_s=sum(summary.trial_wall_clock_time_s or 0.0 for summary in summaries) or None,
        outcome=_aggregate_outcome(summaries),
    )


def compare_prediction(
    run_id: str,
    prediction: AICPrediction,
    summary: EndpointBenchmarkSummary,
    config: EndpointBenchmarkConfig,
) -> PredictionComparison:
    metrics: dict[str, dict[str, float | None]] = {}
    _add_metric(metrics, "tokens_s", prediction.tokens_s, summary.total_tokens_s)
    _add_metric(metrics, "request_rate", prediction.request_rate, summary.request_rate_req_s)
    predicted_latency_s = prediction.request_latency / 1000.0 if prediction.request_latency is not None else None
    _add_metric(metrics, "request_latency_avg_s", predicted_latency_s, summary.avg_latency_s)
    _add_metric(metrics, "request_latency_p50_s", predicted_latency_s, summary.p50_latency_s)
    _add_metric(metrics, "request_latency_p95_s", predicted_latency_s, summary.p95_latency_s)
    _add_metric(metrics, "ttft_ms", prediction.ttft, summary.p95_ttft_ms)
    _add_metric(metrics, "tpot_ms", prediction.tpot, summary.p95_tpot_ms)
    _add_metric(
        metrics,
        "concurrency",
        float(prediction.concurrency) if prediction.concurrency is not None else None,
        float(config.concurrency),
    )
    if prediction.memory is not None and summary.observed_memory_mb is not None:
        _add_metric(metrics, "memory", prediction.memory, float(summary.observed_memory_mb))
    notes = ["AIConfigurator request_latency is interpreted as milliseconds for comparison."]
    if prediction.memory is not None and summary.observed_memory_mb is None:
        notes.append("Memory comparison skipped because no telemetry memory sample was available.")
    return PredictionComparison(run_id=run_id, metrics=metrics, notes=notes)


def _add_metric(metrics: dict[str, dict[str, float | None]], name: str, predicted: float | None, measured: float | None) -> None:
    if predicted is None or measured is None:
        metrics[name] = {
            "predicted": predicted,
            "measured": measured,
            "absolute_delta": None,
            "percent_delta": None,
            "measured_over_predicted_ratio": None,
        }
        return
    delta = measured - predicted
    metrics[name] = {
        "predicted": predicted,
        "measured": measured,
        "absolute_delta": delta,
        "percent_delta": (delta / predicted * 100.0) if predicted else None,
        "measured_over_predicted_ratio": (measured / predicted) if predicted else None,
    }


def _endpoint_url(base_url: str, endpoint: str) -> str:
    base = base_url.rstrip("/")
    parsed = urlsplit(base)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Endpoint base URL must use http or https and include a host.")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Endpoint credentials must not be embedded in the base URL.")
    if base.endswith("/v1") and endpoint.startswith("/v1/"):
        return base + endpoint[3:]
    return base + endpoint


def _authorization_header(api_key_env: str | None) -> str:
    if api_key_env is None:
        return "Bearer EMPTY"
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(f"API key environment variable is unset or empty: {api_key_env}")
    return f"Bearer {api_key}"


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (percentile / 100.0) * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _measurement_window_records(
    records: list[RequestRecord],
    *,
    warmup_requests: int,
    steady_state_duration_s: float | None,
) -> list[RequestRecord]:
    if not records:
        return []
    records = sorted(records, key=lambda record: (record.start_time, record.request_id))
    records = records[max(0, warmup_requests):]
    if steady_state_duration_s is None or steady_state_duration_s <= 0 or not records:
        return records
    start = min(record.start_time for record in records)
    end = start + steady_state_duration_s
    return [record for record in records if record.start_time <= end]


def _measurement_duration(records: list[RequestRecord], wall_time_s: float, *, window_applied: bool) -> float | None:
    if not window_applied:
        return wall_time_s if wall_time_s > 0 else None
    if not records:
        return None
    start = min(record.start_time for record in records)
    end = max(record.end_time for record in records)
    if end > start:
        return end - start
    return None


def _phase_power_samples(
    samples: list[PowerSampleRecord],
    records: list[RequestRecord],
    *,
    warmup_requests: int,
    steady_state_duration_s: float | None,
) -> list[PowerSampleRecord]:
    if not samples:
        return []
    if warmup_requests <= 0 and (steady_state_duration_s is None or steady_state_duration_s <= 0):
        return [sample if sample.phase == "idle" else replace(sample, phase="measurement") for sample in samples]
    warmup_records = _warmup_window_records(records, warmup_requests=warmup_requests)
    measured_records = _measurement_window_records(
        records,
        warmup_requests=warmup_requests,
        steady_state_duration_s=steady_state_duration_s,
    )
    warmup_window = _record_time_window(warmup_records)
    measurement_window = _record_time_window(measured_records)
    phased = []
    for sample in samples:
        if sample.phase == "idle":
            phased.append(sample)
            continue
        phase = "measurement"
        if warmup_window is not None and warmup_window[0] <= sample.timestamp_s <= warmup_window[1]:
            phase = "warmup"
        elif measurement_window is not None and measurement_window[0] <= sample.timestamp_s <= measurement_window[1]:
            phase = "measurement"
        elif warmup_window is not None or measurement_window is not None:
            phase = "active_other"
        phased.append(replace(sample, phase=phase))
    return phased


def _warmup_window_records(records: list[RequestRecord], *, warmup_requests: int) -> list[RequestRecord]:
    if warmup_requests <= 0 or not records:
        return []
    ordered = sorted(records, key=lambda record: (record.start_time, record.request_id))
    return ordered[:warmup_requests]


def _record_time_window(records: list[RequestRecord]) -> tuple[float, float] | None:
    if not records:
        return None
    return min(record.start_time for record in records), max(record.end_time for record in records)


def _effective_concurrency_limit(
    configured_concurrency: int | None,
    configured_num_requests: int | None,
    *,
    warmup_requests: int = 0,
) -> int | None:
    if configured_concurrency is None:
        return None
    if configured_num_requests is None:
        return max(1, configured_concurrency)
    measured_request_budget = max(0, configured_num_requests - max(0, warmup_requests))
    return max(0, min(configured_concurrency, measured_request_budget))


def _concurrency_coverage(configured_concurrency: int | None, measured_request_count: int) -> str | None:
    if configured_concurrency is None:
        return None
    if configured_concurrency <= 1:
        return "complete"
    return "complete" if measured_request_count >= configured_concurrency else "insufficient"


def _concurrency_warnings(
    *,
    configured_concurrency: int | None,
    configured_num_requests: int | None,
    measured_request_count: int,
) -> list[str]:
    if configured_concurrency is None or configured_concurrency <= 1:
        return []
    warnings = []
    if configured_num_requests is not None and configured_num_requests < configured_concurrency:
        warnings.append("Configured request count is lower than concurrency, so the run cannot exercise the requested concurrency.")
    if measured_request_count < configured_concurrency:
        warnings.append("Measured request count is lower than concurrency, so throughput for this candidate is undercovered.")
    return warnings


def _client_saturation_signal(client_cpu_utilization_percent: float | None) -> str:
    if client_cpu_utilization_percent is None:
        return "unknown"
    if client_cpu_utilization_percent >= CLIENT_CPU_SATURATION_THRESHOLD_PERCENT:
        return "cpu_saturated"
    return "not_saturated"


def _client_warnings(client_cpu_utilization_percent: float | None, measurement_duration_s: float | None) -> list[str]:
    if measurement_duration_s is None or measurement_duration_s < CLIENT_CPU_WARNING_MIN_DURATION_S:
        return []
    if _client_saturation_signal(client_cpu_utilization_percent) != "cpu_saturated":
        return []
    return ["Client CPU utilization was high during the benchmark, so throughput may be client limited."]


def _client_issue_rate(records: list[RequestRecord], measurement_duration_s: float | None) -> float | None:
    if not records or measurement_duration_s is None or measurement_duration_s <= 0:
        return None
    return len(records) / measurement_duration_s


def _interval_backlog(records: list[RequestRecord], measurement_duration_s: float | None, *, weight: str) -> dict[str, float | None]:
    if not records or measurement_duration_s is None or measurement_duration_s <= 0:
        return {"avg": None, "max": None}
    events: list[tuple[float, float]] = []
    for record in records:
        start = record.client_start_time if record.client_start_time is not None else record.start_time
        end = record.end_time
        if end < start:
            continue
        value = 1.0 if weight == "requests" else float(max(0, record.total_tokens))
        events.append((start, value))
        events.append((end, -value))
    if not events:
        return {"avg": None, "max": None}
    events.sort(key=lambda item: (item[0], item[1]))
    active = 0.0
    max_active = 0.0
    area = 0.0
    previous = events[0][0]
    for timestamp, delta in events:
        if timestamp > previous:
            area += active * (timestamp - previous)
            previous = timestamp
        active = max(0.0, active + delta)
        max_active = max(max_active, active)
    return {"avg": area / measurement_duration_s, "max": max_active}


def _load_saturation_signal(average_gpu_util_percent: float | None, max_gpu_util_percent: float | None) -> str:
    gpu_value = max(
        [value for value in (average_gpu_util_percent, max_gpu_util_percent) if value is not None],
        default=None,
    )
    if gpu_value is None:
        return "unknown_without_gpu_utilization"
    if gpu_value >= LOAD_GPU_SATURATION_THRESHOLD_PERCENT:
        return "gpu_saturated"
    return "not_saturated"


def _aggregate_load_saturation_signal(summaries: list[EndpointBenchmarkSummary]) -> str:
    signals = [summary.load_saturation_signal for summary in summaries if summary.load_saturation_signal]
    if any(signal == "gpu_saturated" for signal in signals):
        return "gpu_saturated"
    if any(signal == "not_saturated" for signal in signals):
        return "not_saturated"
    return "unknown_without_gpu_utilization"


def _aggregate_load_sufficiency(summaries: list[EndpointBenchmarkSummary]) -> dict[str, object]:
    return {
        "schema_version": "load-sufficiency-trial-aggregate/v1",
        "trial_count": len(summaries),
        "gpu_utilization_available": any(
            (summary.load_sufficiency or {}).get("gpu_utilization_available") is True
            for summary in summaries
        ),
        "average_gpu_util_percent": _round_or_none(
            _mean([_float(summary.average_gpu_util_percent) for summary in summaries if _float(summary.average_gpu_util_percent) is not None])
        ),
        "max_gpu_util_percent": _round_or_none(max([summary.max_gpu_util_percent for summary in summaries if summary.max_gpu_util_percent is not None], default=None)),
        "gpu_saturation_threshold_percent": LOAD_GPU_SATURATION_THRESHOLD_PERCENT,
        "client_issue_rate_req_s": _round_or_none(
            _mean([_float(summary.client_issue_rate_req_s) for summary in summaries if _float(summary.client_issue_rate_req_s) is not None])
        ),
        "avg_request_backlog": _round_or_none(
            _mean([_float(summary.avg_request_backlog) for summary in summaries if _float(summary.avg_request_backlog) is not None])
        ),
        "max_request_backlog": _round_or_none(max([summary.max_request_backlog for summary in summaries if summary.max_request_backlog is not None], default=None)),
        "avg_token_backlog": _round_or_none(
            _mean([_float(summary.avg_token_backlog) for summary in summaries if _float(summary.avg_token_backlog) is not None])
        ),
        "max_token_backlog": _round_or_none(max([summary.max_token_backlog for summary in summaries if summary.max_token_backlog is not None], default=None)),
        "load_saturation_signal": _aggregate_load_saturation_signal(summaries),
    }


def _active_power(average_power_watts: float | None, idle_power_watts: float | None) -> float | None:
    if average_power_watts is None or idle_power_watts is None:
        return None
    return max(0.0, average_power_watts - idle_power_watts)


def _idle_baseline_source(idle_power_watts: float | None, idle_sample_count: int) -> str:
    if idle_sample_count > 0:
        return "sampled_pre_run"
    if idle_power_watts is not None:
        return "configured"
    return "unavailable"


def _average_power(samples: list[PowerSampleRecord]) -> float | None:
    watts = [_power_value(sample) for sample in samples if _power_value(sample) is not None]
    return sum(watts) / len(watts) if watts else None


def _power_value(sample: PowerSampleRecord) -> float | None:
    return _float(sample.power_watts) if sample.power_watts is not None else _float(sample.watts)


def _timing_source(records: list[RequestRecord]) -> str | None:
    sources = sorted({record.timing_source for record in records if record.timing_source})
    if not sources:
        return None
    return sources[0] if len(sources) == 1 else "mixed"


def _token_count_source(records: list[RequestRecord]) -> str | None:
    sources = {record.token_count_source for record in records if record.token_count_source}
    if any(record.total_tokens > 0 and record.token_count_source is None for record in records):
        sources.add("unspecified")
    if not sources:
        return None
    return next(iter(sources)) if len(sources) == 1 else "mixed"


def _token_count_warnings(records: list[RequestRecord]) -> list[str]:
    if any(record.timing_source == "openai_stream_chunks" and record.token_count_source is None for record in records):
        return ["Streaming usage was unavailable, so token throughput and energy per token exclude those responses."]
    if any(record.total_tokens == 0 and record.token_count_source is None for record in records):
        return ["Response usage was unavailable, so token throughput and energy per token exclude those responses."]
    return []


def _aggregate_timing_source(summaries: list[EndpointBenchmarkSummary]) -> str | None:
    sources = sorted({summary.timing_source for summary in summaries if summary.timing_source})
    if not sources:
        return None
    return sources[0] if len(sources) == 1 else "mixed"


def _aggregate_token_count_source(summaries: list[EndpointBenchmarkSummary]) -> str | None:
    sources = sorted({summary.token_count_source for summary in summaries if summary.token_count_source})
    if not sources:
        return None
    return sources[0] if len(sources) == 1 else "mixed"


def _workload_description_with_actual_outputs(
    config: EndpointBenchmarkConfig | None,
    measured_successful: list[RequestRecord],
) -> dict[str, object]:
    description = dict(config.workload_description) if config is not None else {}
    actual_distribution = _token_distribution([record.completion_tokens for record in measured_successful])
    if not description.get("actual_output_length_distribution"):
        description["actual_output_length_distribution"] = actual_distribution
    return description


def _token_distribution(values: list[int]) -> dict[str, object]:
    clean = [float(value) for value in values if value >= 0]
    return {
        "count": len(clean),
        "min": int(min(clean)) if clean else None,
        "p50": _round_or_none(_percentile(clean, 50)),
        "p95": _round_or_none(_percentile(clean, 95)),
        "p99": _round_or_none(_percentile(clean, 99)),
        "max": int(max(clean)) if clean else None,
    }


def _trial_outcome(records: list[RequestRecord]) -> str:
    if not records:
        return "client_failed"
    if all(record.status == "ok" for record in records):
        return "completed"
    reasons = [record.error_reason or record.status for record in records if record.status != "ok"]
    reason_text = " ".join(reasons).lower()
    if "request_timeout" in reasons:
        return "timed_out"
    if "out_of_memory" in reason_text or "cuda out of memory" in reason_text:
        return "out_of_memory"
    if any(reason.startswith("http_404") for reason in reasons):
        return "model_unavailable"
    if any(reason.startswith(("http_401", "http_403")) for reason in reasons) or "gated" in reason_text or "access denied" in reason_text:
        return "access_denied"
    return "client_failed"


def _aggregate_outcome(summaries: list[EndpointBenchmarkSummary]) -> str:
    outcomes = [summary.outcome for summary in summaries if summary.outcome]
    if outcomes and all(outcome == "completed" for outcome in outcomes):
        return "completed"
    priority = [
        "out_of_memory",
        "invalid_config",
        "backend_launch_failed",
        "backend_crashed",
        "timed_out",
        "model_unavailable",
        "access_denied",
        "client_failed",
    ]
    for outcome in priority:
        if outcome in outcomes:
            return outcome
    return outcomes[0] if outcomes else "client_failed"


def _first_available(values: list[str | None]) -> str | None:
    for value in values:
        if value:
            return value
    return None


def _last_available(values: list[str | None]) -> str | None:
    for value in reversed(values):
        if value:
            return value
    return None


def _aggregate_thermal_classification(summaries: list[EndpointBenchmarkSummary]) -> str | None:
    priority = {
        "warming": 4,
        "limited_window": 3,
        "stable": 2,
        "unavailable": 1,
    }
    classes = [summary.thermal_stability_classification for summary in summaries if summary.thermal_stability_classification]
    if not classes:
        return None
    return max(classes, key=lambda item: priority.get(item, 0))


def _weighted_queue_mean(summaries: list[EndpointBenchmarkSummary]) -> float | None:
    weighted_total = 0.0
    sample_count = 0
    for summary in summaries:
        value = _float(summary.avg_client_queue_s)
        if value is None or summary.client_queue_sample_count <= 0:
            continue
        weighted_total += value * summary.client_queue_sample_count
        sample_count += summary.client_queue_sample_count
    return weighted_total / sample_count if sample_count else None


def _metric_stats(values: list[float]) -> dict[str, float | int | None]:
    mean = _mean(values)
    stddev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "mean": _round_or_none(mean),
        "stddev": _round_or_none(stddev),
        "coefficient_of_variation": _round_or_none(stddev / mean) if mean not in {None, 0} else None,
        "confidence_interval_95": _confidence_interval(values),
    }


def _confidence_interval(values: list[float]) -> dict[str, float | int | None]:
    mean = _mean(values)
    if mean is None:
        return {"count": 0, "mean": None, "half_width": None, "low": None, "high": None}
    if len(values) == 1:
        return {"count": 1, "mean": _round_or_none(mean), "half_width": 0.0, "low": _round_or_none(mean), "high": _round_or_none(mean)}
    half_width = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return {
        "count": len(values),
        "mean": _round_or_none(mean),
        "half_width": _round_or_none(half_width),
        "low": _round_or_none(mean - half_width),
        "high": _round_or_none(mean + half_width),
    }


def _stability_classification(metric_values: dict[str, list[float]]) -> str:
    throughput = metric_values.get("total_tokens_s", [])
    p95 = metric_values.get("p95_latency_s", [])
    if len(throughput) < 2 and len(p95) < 2:
        return "single_trial"
    coefficients = []
    for values in (throughput, p95):
        mean = _mean(values)
        if mean not in {None, 0} and len(values) > 1:
            coefficients.append(statistics.stdev(values) / mean)
    if not coefficients:
        return "unknown"
    worst = max(coefficients)
    if worst <= 0.05:
        return "stable"
    if worst <= 0.15:
        return "mostly_stable"
    return "unstable"


def _mean(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def _float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_value(value: object) -> int:
    try:
        return int(value) if value is not None else 0
    except (TypeError, ValueError):
        return 0


def _artifact_files(run_dir: Path) -> list[str]:
    return sorted(path.name for path in run_dir.iterdir() if path.is_file())


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_commit() -> str | None:
    return _git_output(["rev-parse", "HEAD"])


def _git_dirty() -> bool | None:
    status = _git_output(["status", "--short"])
    if status is None:
        return None
    return bool(status.strip())


def _git_output(args: list[str]) -> str | None:
    root = _repo_root()
    if not (root / ".git").exists():
        return None
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else ""


def _cuda_version() -> str | None:
    try:
        import torch
    except (ImportError, OSError):
        return None
    version = getattr(getattr(torch, "version", None), "cuda", None)
    return str(version) if version else None


def _gpu_driver_version() -> str | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    line = completed.stdout.splitlines()[0].strip() if completed.stdout.splitlines() else ""
    return line if completed.returncode == 0 and line else None


def _round_or_none(value: float | int | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)
