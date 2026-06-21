"""Standalone telemetry diagnostics for Attach Mode."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from .io import write_json, write_jsonl
from .schemas import PowerSampleRecord, TelemetrySummary, to_dict
from .telemetry import make_telemetry_collector, summarize_telemetry

TelemetryCollectorFactory = Callable[[str, int, float], object]


@dataclass(frozen=True)
class TelemetryCheckRun:
    run_dir: Path
    samples: list[PowerSampleRecord]
    summary: TelemetrySummary
    report_text: str


def run_telemetry_check(
    *,
    telemetry: str,
    duration_s: float,
    interval_s: float,
    out_dir: Path,
    device_index: int = 0,
    telemetry_collector_factory: TelemetryCollectorFactory | None = None,
) -> TelemetryCheckRun:
    if duration_s <= 0:
        raise ValueError("duration_s must be greater than 0.")
    if interval_s <= 0:
        raise ValueError("interval_s must be greater than 0.")

    run_id = _make_telemetry_check_run_id()
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    factory = telemetry_collector_factory or make_telemetry_collector
    collector = factory(telemetry, device_index, interval_s)
    collector.start()
    start = time.perf_counter()
    time.sleep(duration_s)
    elapsed = max(time.perf_counter() - start, 0.0)
    capture = collector.stop()

    warnings = list(capture.warnings)
    summary = summarize_telemetry(
        capture.samples,
        wall_time_s=elapsed,
        total_tokens=0,
        provider=capture.provider,
        warnings=warnings,
    )
    if summary.duration_s is None:
        summary = replace(summary, duration_s=round(elapsed, 6))
    report_text = _format_telemetry_check_report(summary, run_dir)
    write_jsonl(run_dir / "samples.jsonl", capture.samples)
    write_json(run_dir / "telemetry_summary.json", summary)
    write_json(run_dir / "telemetry_capabilities.json", summary.telemetry_capabilities)
    write_json(
        run_dir / "metadata.json",
        {
            "schema_version": "telemetry-check/v1",
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "telemetry_requested": telemetry,
            "device_index": device_index,
            "duration_s": duration_s,
            "interval_s": interval_s,
            "artifact_files": ["metadata.json", "report.txt", "samples.jsonl", "telemetry_capabilities.json", "telemetry_summary.json"],
        },
    )
    (run_dir / "report.txt").write_text(report_text, encoding="utf-8")
    return TelemetryCheckRun(run_dir=run_dir, samples=capture.samples, summary=summary, report_text=report_text)


def _make_telemetry_check_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"telemetry-check-{timestamp}-{uuid.uuid4().hex[:8]}"


def _format_telemetry_check_report(summary: TelemetrySummary, run_dir: Path) -> str:
    lines = [
        "=" * 60,
        "Serve Optimize Telemetry Check",
        "=" * 60,
        f"Provider: {summary.telemetry_provider or 'unavailable'}",
        f"Device: {summary.device_name or 'unknown'}",
        f"Quality: {summary.telemetry_quality}",
        f"Samples: {summary.sample_count}",
        f"Duration: {_fmt(summary.duration_s, 's')}",
        f"Sampling rate: {_fmt(summary.sampling_rate_hz, 'Hz')}",
        f"Artifacts: {run_dir}",
        "",
        "-" * 60,
        "Power",
        "-" * 60,
        f"Average: {_fmt(summary.power_stats.get('avg'), 'W')}",
        f"Min: {_fmt(summary.power_stats.get('min'), 'W')}",
        f"Max: {_fmt(summary.power_stats.get('max'), 'W')}",
        f"Stddev: {_fmt(summary.power_stats.get('stddev'), 'W')}",
        f"Coefficient of variation: {_fmt(summary.power_stats.get('coefficient_of_variation'), None)}",
        "",
        "-" * 60,
        "Resource Fields",
        "-" * 60,
        f"Average GPU util: {_fmt(summary.utilization_stats.get('avg_gpu_util_percent'), '%')}",
        f"Max GPU util: {_fmt(summary.utilization_stats.get('max_gpu_util_percent'), '%')}",
        f"Average memory util: {_fmt(summary.utilization_stats.get('avg_memory_util_percent'), '%')}",
        f"Average temperature: {_fmt(summary.thermal_stats.get('avg_temperature_c'), 'C')}",
        f"Average SM clock: {_fmt(summary.clock_stats.get('avg_sm_clock_mhz'), 'MHz')}",
        f"Average memory clock: {_fmt(summary.clock_stats.get('avg_memory_clock_mhz'), 'MHz')}",
        f"Power limit: {_fmt(summary.power_limit_watts, 'W')}",
        "",
        "-" * 60,
        "Field Availability",
        "-" * 60,
        f"Missing fields: {', '.join(summary.missing_fields) if summary.missing_fields else 'none'}",
        "",
        "-" * 60,
        "Telemetry Capabilities",
        "-" * 60,
        *_capability_lines(summary.telemetry_capabilities),
        "",
        "-" * 60,
        "Warnings",
        "-" * 60,
        *([f"- {warning}" for warning in summary.warnings] if summary.warnings else ["none"]),
        "",
        "-" * 60,
        "Notes",
        "-" * 60,
        *([f"- {note}" for note in summary.notes] if summary.notes else ["none"]),
    ]
    return "\n".join(lines).rstrip() + "\n"


def _fmt(value: object, unit: str | None) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    rendered = f"{number:.3f}" if abs(number) >= 1 else f"{number:.6f}".rstrip("0").rstrip(".")
    return f"{rendered} {unit}" if unit else rendered


def _capability_lines(capabilities: object) -> list[str]:
    payload = to_dict(capabilities) if capabilities is not None else {}
    available = payload.get("available_fields") if isinstance(payload, dict) else []
    unavailable = payload.get("unavailable_fields") if isinstance(payload, dict) else []
    notes = payload.get("notes") if isinstance(payload, dict) else []
    lines = ["Available:"]
    lines.extend(f"  OK {field}" for field in available or ["none"])
    lines.append("Unavailable:")
    lines.extend(f"  missing {field}" for field in unavailable or ["none"])
    if notes:
        lines.append("Notes:")
        lines.extend(f"  {note}" for note in notes)
    return lines
