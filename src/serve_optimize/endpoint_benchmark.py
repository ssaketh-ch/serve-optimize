"""Endpoint-native OpenAI-compatible benchmark runner."""

from __future__ import annotations

import json
import math
import statistics
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from urllib import error, request

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


def run_endpoint_benchmark(
    config: EndpointBenchmarkConfig,
    out_dir: Path,
    prediction: AICPrediction | None = None,
    hardware: HardwareSnapshot | None = None,
    request_fn: RequestFn | None = None,
    telemetry_collector_factory: TelemetryCollectorFactory | None = None,
) -> EndpointBenchmarkRun:
    run_dir = out_dir / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    request_fn = request_fn or send_chat_completion_request

    write_json(run_dir / "config.json", config)
    if prediction is not None:
        write_json(run_dir / "prediction.json", prediction)

    telemetry_collector_factory = telemetry_collector_factory or make_telemetry_collector
    idle_samples: list[PowerSampleRecord] = []
    idle_power_watts = config.idle_power_watts
    if config.telemetry != "none" and config.idle_baseline_duration_s > 0:
        idle_sampler = telemetry_collector_factory(config.telemetry, config.device_index, 0.2)
        idle_sampler.start()
        time.sleep(config.idle_baseline_duration_s)
        idle_capture = idle_sampler.stop()
        idle_samples = [replace(sample, phase="idle") for sample in idle_capture.samples]
        idle_power_watts = idle_power_watts if idle_power_watts is not None else _average_power(idle_samples)

    sampler = telemetry_collector_factory(config.telemetry, config.device_index, 0.2)
    sampler.start()
    wall_start = time.perf_counter()
    records = _run_requests(config, request_fn)
    wall_time_s = max(time.perf_counter() - wall_start, 0.0)
    telemetry = sampler.stop()
    power_samples = [replace(sample, phase="active") for sample in telemetry.samples]

    records = sorted(records, key=lambda item: item.request_id)
    write_jsonl(run_dir / "requests.jsonl", records)
    if config.telemetry != "none":
        write_jsonl(run_dir / "power_samples.jsonl", [*idle_samples, *power_samples])
        if idle_samples:
            write_jsonl(run_dir / "idle_power_samples.jsonl", idle_samples)

    summary = summarize_requests(
        config.run_id,
        records,
        wall_time_s,
        power_samples,
        telemetry,
        warmup_requests=config.warmup_requests,
        steady_state_duration_s=config.steady_state_duration_s,
        idle_power_watts=idle_power_watts,
        idle_sample_count=len(idle_samples),
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
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(request_fn, config, request_id): request_id for request_id in range(config.num_requests)}
        for future in as_completed(futures):
            request_id = futures[future]
            try:
                records.append(future.result())
            except Exception as exc:
                now = time.time()
                records.append(
                    RequestRecord(
                        request_id=request_id,
                        start_time=now,
                        end_time=now,
                        latency_s=0.0,
                        status="error",
                        error=f"{exc.__class__.__name__}: {exc}",
                    )
                )
    return records


def send_chat_completion_request(config: EndpointBenchmarkConfig, request_id: int) -> RequestRecord:
    start = time.time()
    payload = {
        "model": config.model,
        "messages": [{"role": "user", "content": config.prompt}],
        "max_tokens": config.max_tokens,
        "temperature": 0,
    }
    data = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        _endpoint_url(config.base_url, config.endpoint),
        data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    try:
        with request.urlopen(http_request, timeout=config.timeout_s) as response:
            body = response.read().decode("utf-8")
            status_code = response.status
        parsed = json.loads(body) if body else {}
        usage = parsed.get("usage") or {}
        end = time.time()
        return RequestRecord(
            request_id=request_id,
            start_time=start,
            end_time=end,
            latency_s=end - start,
            status="ok" if 200 <= status_code < 300 else f"http_{status_code}",
            prompt_tokens=_int_value(usage.get("prompt_tokens")),
            completion_tokens=_int_value(usage.get("completion_tokens")),
            total_tokens=_int_value(usage.get("total_tokens")),
        )
    except error.HTTPError as exc:
        end = time.time()
        return RequestRecord(
            request_id=request_id,
            start_time=start,
            end_time=end,
            latency_s=end - start,
            status=f"http_{exc.code}",
            error=str(exc),
        )
    except Exception as exc:
        end = time.time()
        return RequestRecord(
            request_id=request_id,
            start_time=start,
            end_time=end,
            latency_s=end - start,
            status="error",
            error=f"{exc.__class__.__name__}: {exc}",
        )


def summarize_requests(
    run_id: str,
    records: list[RequestRecord],
    wall_time_s: float,
    power_samples: list[PowerSampleRecord] | None = None,
    telemetry: TelemetryCapture | None = None,
    warmup_requests: int = 0,
    steady_state_duration_s: float | None = None,
    idle_power_watts: float | None = None,
    idle_sample_count: int = 0,
) -> EndpointBenchmarkSummary:
    power_samples = power_samples or []
    telemetry = telemetry or TelemetryCapture(provider=None, samples=[], warnings=[])
    successful = [record for record in records if record.status == "ok"]
    measured_successful = _steady_state_records(successful, warmup_requests=warmup_requests, steady_state_duration_s=steady_state_duration_s)
    latencies = [record.latency_s for record in measured_successful]
    prompt_tokens = sum(record.prompt_tokens for record in measured_successful)
    completion_tokens = sum(record.completion_tokens for record in measured_successful)
    total_tokens = sum(record.total_tokens for record in measured_successful)
    steady_state_duration = _steady_state_duration(measured_successful, wall_time_s)
    telemetry_summary = summarize_telemetry(
        power_samples,
        wall_time_s,
        total_tokens,
        provider=telemetry.provider,
        warnings=telemetry.warnings,
    )
    active_power = _active_power(telemetry_summary.average_power_watts, idle_power_watts)
    active_energy = active_power * wall_time_s if active_power is not None else None
    active_joules_per_token = active_energy / total_tokens if active_energy is not None and total_tokens > 0 else None
    measurement_quality = {
        "schema_version": "measurement-quality/v1",
        "warmup_requests": warmup_requests,
        "steady_state_requested_duration_s": steady_state_duration_s,
        "steady_state_requests": len(measured_successful),
        "steady_state_duration_s": _round_or_none(steady_state_duration),
        "idle_power_watts": _round_or_none(idle_power_watts),
        "idle_sample_count": idle_sample_count,
        "energy_accounting": "idle_subtracted" if active_energy is not None else "gross",
        "stability_classification": "single_trial",
    }
    return EndpointBenchmarkSummary(
        run_id=run_id,
        total_requests=len(records),
        successful_requests=len(successful),
        failed_requests=len(records) - len(successful),
        wall_time_s=wall_time_s,
        request_rate_req_s=len(measured_successful) / wall_time_s if wall_time_s > 0 else 0.0,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        output_tokens_s=completion_tokens / wall_time_s if wall_time_s > 0 else 0.0,
        total_tokens_s=total_tokens / wall_time_s if wall_time_s > 0 else 0.0,
        avg_latency_s=sum(latencies) / len(latencies) if latencies else None,
        p50_latency_s=_percentile(latencies, 50),
        p95_latency_s=_percentile(latencies, 95),
        p99_latency_s=_percentile(latencies, 99),
        power_sample_count=telemetry_summary.power_sample_count,
        average_power_watts=telemetry_summary.average_power_watts,
        min_power_watts=telemetry_summary.min_power_watts,
        max_power_watts=telemetry_summary.max_power_watts,
        peak_power_watts=telemetry_summary.peak_power_watts,
        power_stddev_watts=telemetry_summary.power_stddev_watts,
        power_sampling_duration_s=telemetry_summary.power_sampling_duration_s,
        power_sampling_rate_hz=telemetry_summary.power_sampling_rate_hz,
        idle_power_watts=_round_or_none(idle_power_watts),
        active_power_watts=_round_or_none(active_power),
        active_energy_joules=_round_or_none(active_energy),
        energy_joules=telemetry_summary.energy_joules,
        joules_per_token=telemetry_summary.joules_per_token,
        active_joules_per_token=_round_or_none(active_joules_per_token),
        tokens_per_second_per_watt=telemetry_summary.tokens_per_second_per_watt,
        warmup_requests=warmup_requests,
        steady_state_requests=len(measured_successful),
        steady_state_duration_s=_round_or_none(steady_state_duration),
        steady_state_total_tokens=total_tokens,
        steady_state_total_tokens_s=total_tokens / steady_state_duration if steady_state_duration and steady_state_duration > 0 else None,
        steady_state_request_rate_req_s=len(measured_successful) / steady_state_duration if steady_state_duration and steady_state_duration > 0 else None,
        measurement_quality=measurement_quality,
        stability_classification="single_trial",
        observed_memory_mb=telemetry_summary.max_memory_used_mb,
        average_gpu_util_percent=telemetry_summary.average_gpu_util_percent,
        max_gpu_util_percent=telemetry_summary.max_gpu_util_percent,
        average_memory_util_percent=telemetry_summary.average_memory_util_percent,
        max_memory_util_percent=telemetry_summary.max_memory_util_percent,
        average_temperature_c=telemetry_summary.average_temperature_c,
        max_temperature_c=telemetry_summary.max_temperature_c,
        average_sm_clock_mhz=telemetry_summary.average_sm_clock_mhz,
        average_memory_clock_mhz=telemetry_summary.average_memory_clock_mhz,
        telemetry_provider=telemetry_summary.telemetry_provider,
        telemetry_available=telemetry_summary.telemetry_available,
        telemetry_quality=telemetry_summary.telemetry_quality,
        telemetry_notes=telemetry_summary.telemetry_notes,
        telemetry_summary=to_dict(telemetry_summary),
        warnings=list(telemetry_summary.telemetry_warnings),
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
    prompt_tokens = sum(summary.prompt_tokens for summary in summaries)
    completion_tokens = sum(summary.completion_tokens for summary in summaries)
    total_tokens = sum(summary.total_tokens for summary in summaries)
    metrics = {
        "total_tokens_s": [_float(summary.total_tokens_s) for summary in summaries],
        "request_rate_req_s": [_float(summary.request_rate_req_s) for summary in summaries],
        "p95_latency_s": [_float(summary.p95_latency_s) for summary in summaries],
        "joules_per_token": [_float(summary.joules_per_token) for summary in summaries],
        "active_joules_per_token": [_float(summary.active_joules_per_token) for summary in summaries],
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
            "energy_accounting": "idle_subtracted" if any(summary.active_energy_joules is not None for summary in summaries) else "gross",
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
        output_tokens_s=completion_tokens / wall_time_s if wall_time_s > 0 else 0.0,
        total_tokens_s=_mean(metric_values["total_tokens_s"]) or 0.0,
        avg_latency_s=_mean([_float(summary.avg_latency_s) for summary in summaries if _float(summary.avg_latency_s) is not None]),
        p50_latency_s=_mean([_float(summary.p50_latency_s) for summary in summaries if _float(summary.p50_latency_s) is not None]),
        p95_latency_s=_mean(metric_values["p95_latency_s"]),
        p99_latency_s=_mean([_float(summary.p99_latency_s) for summary in summaries if _float(summary.p99_latency_s) is not None]),
        power_sample_count=sum(summary.power_sample_count for summary in summaries),
        average_power_watts=_mean([_float(summary.average_power_watts) for summary in summaries if _float(summary.average_power_watts) is not None]),
        peak_power_watts=max([summary.peak_power_watts for summary in summaries if summary.peak_power_watts is not None], default=None),
        energy_joules=sum([summary.energy_joules for summary in summaries if summary.energy_joules is not None]) or None,
        joules_per_token=_mean(metric_values["joules_per_token"]),
        active_energy_joules=sum([summary.active_energy_joules for summary in summaries if summary.active_energy_joules is not None]) or None,
        active_joules_per_token=_mean(metric_values["active_joules_per_token"]),
        tokens_per_second_per_watt=_mean([_float(summary.tokens_per_second_per_watt) for summary in summaries if _float(summary.tokens_per_second_per_watt) is not None]),
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
    if base.endswith("/v1") and endpoint.startswith("/v1/"):
        return base + endpoint[3:]
    return base + endpoint


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


def _steady_state_records(
    successful: list[RequestRecord],
    *,
    warmup_requests: int,
    steady_state_duration_s: float | None,
) -> list[RequestRecord]:
    if not successful:
        return []
    records = sorted(successful, key=lambda record: (record.end_time, record.request_id))
    records = records[max(0, warmup_requests):]
    if steady_state_duration_s is None or steady_state_duration_s <= 0 or not records:
        return records
    start = min(record.start_time for record in records)
    end = start + steady_state_duration_s
    return [record for record in records if record.start_time <= end]


def _steady_state_duration(records: list[RequestRecord], wall_time_s: float) -> float | None:
    if not records:
        return None
    start = min(record.start_time for record in records)
    end = max(record.end_time for record in records)
    if end > start:
        return end - start
    return wall_time_s if wall_time_s > 0 else None


def _active_power(average_power_watts: float | None, idle_power_watts: float | None) -> float | None:
    if average_power_watts is None or idle_power_watts is None:
        return None
    return max(0.0, average_power_watts - idle_power_watts)


def _average_power(samples: list[PowerSampleRecord]) -> float | None:
    watts = [_power_value(sample) for sample in samples if _power_value(sample) is not None]
    return sum(watts) / len(watts) if watts else None


def _power_value(sample: PowerSampleRecord) -> float | None:
    return _float(sample.power_watts) if sample.power_watts is not None else _float(sample.watts)


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


def _round_or_none(value: float | int | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)
