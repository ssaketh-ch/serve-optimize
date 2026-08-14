from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

GUIDELLM_VERSION = "0.7.3"
EXPECTED_STREAMS = {16, 32, 64, 128, 256}


def _successful(metric: object) -> dict[str, object]:
    if not isinstance(metric, dict):
        return {}
    value = metric.get("successful")
    return value if isinstance(value, dict) else {}


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _metric(metrics: dict[str, object], name: str, field: str = "mean") -> float | None:
    return _number(_successful(metrics.get(name)).get(field))


def _percentile(metrics: dict[str, object], name: str, percentile: str) -> float | None:
    values = _successful(metrics.get(name)).get("percentiles")
    return _number(values.get(percentile)) if isinstance(values, dict) else None


def audit_report(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata_path = path.parent / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    rows: list[dict[str, object]] = []
    for index, benchmark in enumerate(payload.get("benchmarks", [])):
        metrics = benchmark.get("metrics", {})
        requests = benchmark.get("requests", {})
        scheduler = benchmark.get("scheduler_metrics", {})
        request_totals = metrics.get("request_totals", {})
        successful_records = requests.get("successful", [])
        errored_records = requests.get("errored", [])
        incomplete_records = requests.get("incomplete", [])
        measured_successes = int(request_totals.get("successful") or 0)
        measured_errors = int(request_totals.get("errored") or 0)
        measured_incomplete = int(request_totals.get("incomplete") or 0)
        measured_total = int(request_totals.get("total") or 0)
        raw_successes = len(successful_records)
        raw_errors = len(errored_records)
        raw_incomplete = len(incomplete_records)
        raw_total = len(successful_records) + len(errored_records) + len(incomplete_records)
        scheduler_successes = int((scheduler.get("requests_made") or {}).get("successful") or 0)
        scheduler_errors = int((scheduler.get("requests_made") or {}).get("errored") or 0)
        scheduler_incomplete = int((scheduler.get("requests_made") or {}).get("incomplete") or 0)
        scheduler_total = int((scheduler.get("requests_made") or {}).get("total") or 0)
        prompt_tokens = _metric(metrics, "prompt_token_count", "total_sum")
        output_tokens = _metric(metrics, "output_token_count", "total_sum")
        total_tokens = _metric(metrics, "total_token_count", "total_sum")
        config = benchmark.get("config", {})
        strategy = config.get("strategy", {})
        duration = _number(benchmark.get("duration"))
        streams = strategy.get("streams")
        errors: list[str] = []
        checks = {
            "guidellm_version": payload.get("metadata", {}).get("guidellm_version") == GUIDELLM_VERSION
            and metadata.get("guidellm_version") == GUIDELLM_VERSION,
            "run_metadata": all(
                metadata.get(name)
                for name in (
                    "backend",
                    "backend_version",
                    "backend_command",
                    "backend_target",
                    "guidellm_command",
                    "model",
                    "model_revision",
                    "serve_optimize_command",
                    "workload_profile",
                )
            ),
            "command": bool(metadata.get("guidellm_command")),
            "backend_command": bool(metadata.get("backend_command")),
            "measurement_duration": duration is not None and duration > 0,
            "raw_request_count": raw_total > 0,
            "scheduler_accounting": scheduler_total
            == scheduler_successes + scheduler_errors + scheduler_incomplete,
            "raw_within_scheduler_count": raw_successes <= scheduler_successes
            and raw_errors <= scheduler_errors
            and raw_incomplete <= scheduler_incomplete,
            "measured_request_count": measured_total > 0
            and measured_total == measured_successes + measured_errors + measured_incomplete,
            "measured_within_raw_count": measured_successes <= raw_successes
            and measured_errors <= raw_errors
            and measured_incomplete <= raw_incomplete,
            "successful_requests": measured_successes > 0,
            "no_measured_errors": measured_errors == 0,
            "bounded_terminal_incomplete": isinstance(streams, int)
            and 0 <= measured_incomplete <= streams,
            "nonzero_tokens": all(value is not None and value > 0 for value in (prompt_tokens, output_tokens, total_tokens)),
            "token_accounting": all(value is not None for value in (prompt_tokens, output_tokens, total_tokens))
            and abs(float(prompt_tokens) + float(output_tokens) - float(total_tokens)) < 1e-6,
            "request_rate": _metric(metrics, "requests_per_second") is not None,
            "output_token_rate": _metric(metrics, "output_tokens_per_second") is not None,
            "request_latency": all(
                _percentile(metrics, "request_latency", percentile) is not None
                for percentile in ("p50", "p95", "p99")
            ),
            "ttft": all(
                _percentile(metrics, "time_to_first_token_ms", percentile) is not None
                for percentile in ("p50", "p95", "p99")
            ),
            "tpot": all(
                _percentile(metrics, "time_per_output_token_ms", percentile) is not None
                for percentile in ("p50", "p95", "p99")
            ),
        }
        errors.extend(name for name, passed in checks.items() if not passed)
        rows.append(
            {
                "report": str(path),
                "benchmark_index": index,
                "backend": metadata.get("backend"),
                "backend_version": metadata.get("backend_version"),
                "model": metadata.get("model"),
                "model_revision": metadata.get("model_revision"),
                "workload_profile": metadata.get("workload_profile"),
                "streams": streams,
                "duration_s": duration,
                "raw_requests": raw_total,
                "scheduler_requests": scheduler_total,
                "scheduler_incomplete_requests": scheduler_incomplete,
                "measured_successful_requests": measured_successes,
                "measured_errored_requests": measured_errors,
                "measured_incomplete_requests": measured_incomplete,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "requests_per_second": _metric(metrics, "requests_per_second"),
                "output_tokens_per_second": _metric(metrics, "output_tokens_per_second"),
                "p50_latency_ms": (_percentile(metrics, "request_latency", "p50") or 0.0) * 1000.0,
                "p95_latency_ms": (_percentile(metrics, "request_latency", "p95") or 0.0) * 1000.0,
                "p99_latency_ms": (_percentile(metrics, "request_latency", "p99") or 0.0) * 1000.0,
                "p50_ttft_ms": _percentile(metrics, "time_to_first_token_ms", "p50"),
                "p95_ttft_ms": _percentile(metrics, "time_to_first_token_ms", "p95"),
                "p99_ttft_ms": _percentile(metrics, "time_to_first_token_ms", "p99"),
                "p50_tpot_ms": _percentile(metrics, "time_per_output_token_ms", "p50"),
                "p95_tpot_ms": _percentile(metrics, "time_per_output_token_ms", "p95"),
                "p99_tpot_ms": _percentile(metrics, "time_per_output_token_ms", "p99"),
                "client_queue_seconds": _number(scheduler.get("queued_time_avg")),
                "checks": checks,
                "errors": errors,
            }
        )
    if not rows:
        raise ValueError(f"GuideLLM report has no benchmarks: {path}")
    stream_coverage = len(rows) == len(EXPECTED_STREAMS) and {
        int(row["streams"]) for row in rows if isinstance(row.get("streams"), int)
    } == EXPECTED_STREAMS
    for row in rows:
        row["checks"]["stream_coverage"] = stream_coverage
        if not stream_coverage:
            row["errors"].append("stream_coverage")
    return rows


def audit_root(root: Path) -> dict[str, object]:
    reports = sorted(root.rglob("benchmarks.json"))
    rows = [row for path in reports for row in audit_report(path)]
    return {
        "schema_version": "guidellm-audit/v1",
        "guidellm_version": GUIDELLM_VERSION,
        "expected_streams": sorted(EXPECTED_STREAMS),
        "report_count": len(reports),
        "benchmark_count": len(rows),
        "failed_benchmark_count": sum(bool(row["errors"]) for row in rows),
        "rows": rows,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [key for key in rows[0] if key not in {"checks", "errors"}] if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, required=True)
    args = parser.parse_args()
    payload = audit_root(args.root)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(args.csv_out, payload["rows"])
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, sort_keys=True))
    if payload["failed_benchmark_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
