from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def percentile(values: list[float], value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = value / 100.0 * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def close(actual: object, expected: float | None) -> bool:
    if actual is None or expected is None:
        return actual is expected
    return math.isclose(float(actual), expected, rel_tol=1e-6, abs_tol=1e-6)


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit_trial(summary_path: Path) -> dict[str, object]:
    run_dir = summary_path.parent
    summary = load_json(summary_path)
    config = load_json(run_dir / "config.json")
    records = load_jsonl(run_dir / "requests.jsonl")
    ordered = sorted(records, key=lambda row: (float(row["start_time"]), int(row["request_id"])))
    warmup_count = max(0, int(summary.get("warmup_requests") or 0))
    warmup = ordered[:warmup_count]
    measured = ordered[warmup_count:]
    requested_duration = config.get("steady_state_duration_s")
    if measured and requested_duration is not None and float(requested_duration) > 0:
        cutoff = float(measured[0]["start_time"]) + float(requested_duration)
        measured = [row for row in measured if float(row["start_time"]) <= cutoff]
    successful = [row for row in measured if row.get("status") == "ok"]
    errors: list[str] = []

    checks = {
        "request_count": summary.get("total_requests") == len(records),
        "success_count": summary.get("successful_requests") == sum(row.get("status") == "ok" for row in records),
        "failure_count": summary.get("failed_requests") == sum(row.get("status") != "ok" for row in records),
        "measured_request_count": summary.get("measured_requests") == len(measured),
        "measured_success_count": summary.get("measured_successful_requests") == len(successful),
        "prompt_tokens": summary.get("prompt_tokens") == sum(int(row.get("prompt_tokens") or 0) for row in successful),
        "completion_tokens": summary.get("completion_tokens") == sum(int(row.get("completion_tokens") or 0) for row in successful),
        "total_tokens": summary.get("total_tokens") == sum(int(row.get("total_tokens") or 0) for row in successful),
        "nonzero_tokens": all(float(summary.get(name) or 0) > 0 for name in ("prompt_tokens", "completion_tokens", "total_tokens")),
        "backend_version": bool(summary.get("backend_version")),
        "backend_command": bool(summary.get("backend_launch_command")),
        "effective_configuration": bool(summary.get("backend_effective_values")),
        "measurement_energy": float(summary.get("energy_joules") or 0) > 0,
        "tokens_per_joule": float(summary.get("tokens_per_joule") or 0) > 0,
        "joules_per_generated_token": float(summary.get("joules_per_generated_token") or 0) > 0,
        "energy_accounting": summary.get("energy_accounting") in {"raw", "idle_subtracted"},
        "measurement_power_samples": int(summary.get("measurement_power_sample_count") or 0) > 0,
        "actual_prompt_distribution": bool((summary.get("workload_description") or {}).get("actual_prompt_length_distribution")),
        "actual_output_distribution": bool((summary.get("workload_description") or {}).get("actual_output_length_distribution")),
        "actual_measurement_duration": float((summary.get("workload_description") or {}).get("actual_measurement_duration_s") or 0) > 0,
    }
    if warmup and measured:
        checks["warmup_before_measurement"] = max(float(row["end_time"]) for row in warmup) <= min(
            float(row["start_time"]) for row in measured
        )
        checks["warmup_power_samples"] = int(summary.get("warmup_power_sample_count") or 0) > 0

    metric_inputs = {
        "latency_s": [float(row["latency_s"]) for row in successful],
        "ttft_ms": [float(row["ttft_s"]) * 1000.0 for row in successful if row.get("ttft_s") is not None],
        "tpot_ms": [float(row["tpot_s"]) * 1000.0 for row in successful if row.get("tpot_s") is not None],
    }
    for prefix, values in metric_inputs.items():
        for p in (50, 95, 99):
            field = f"p{p}_{prefix}"
            checks[field] = close(summary.get(field), percentile(values, p))

    duration = float(summary.get("measurement_duration_s") or 0)
    completion_tokens = sum(int(row.get("completion_tokens") or 0) for row in successful)
    total_tokens = sum(int(row.get("total_tokens") or 0) for row in successful)
    checks["output_throughput"] = close(summary.get("output_tokens_s"), completion_tokens / duration if duration else None)
    checks["total_throughput"] = close(summary.get("total_tokens_s"), total_tokens / duration if duration else None)
    for name, passed in checks.items():
        if not passed:
            errors.append(name)
    return {
        "summary_path": str(summary_path),
        "backend": summary.get("backend_name"),
        "run_id": summary.get("run_id"),
        "request_records": len(records),
        "measured_requests": len(measured),
        "checks": checks,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    summary_paths = sorted(
        path
        for root in args.roots
        for path in root.rglob("summary.json")
        if (path.parent / "requests.jsonl").exists()
    )
    trials = [audit_trial(path) for path in summary_paths]
    payload = {
        "schema_version": "live-measurement-audit/v1",
        "trial_summary_count": len(trials),
        "request_record_count": sum(int(row["request_records"]) for row in trials),
        "backend_counts": {
            backend: sum(row["backend"] == backend for row in trials)
            for backend in sorted({str(row["backend"]) for row in trials})
        },
        "failed_check_count": sum(len(row["errors"]) for row in trials),
        "failed_trials": [row for row in trials if row["errors"]],
        "trials": trials,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    if payload["failed_check_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
