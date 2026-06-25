"""Summarize safe baseline comparisons from an overnight campaign."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

METRICS = (
    "throughput_tokens_per_sec",
    "p95_latency_ms",
    "average_power_w",
    "joules_per_token",
    "tokens_per_watt",
)

METRIC_LABELS = {
    "throughput_tokens_per_sec": ("throughput", "tok/s", 1),
    "p95_latency_ms": ("p95 latency", "ms", 1),
    "average_power_w": ("avg power", "W", 1),
    "joules_per_token": ("joules/token", "J/token", 4),
    "tokens_per_watt": ("tokens/watt", "tok/W", 2),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign_dir", type=Path)
    args = parser.parse_args()
    rows = summarize_campaign(args.campaign_dir)
    run_count, failure_count = write_outputs(args.campaign_dir, rows)
    print(
        f"Summarized {run_count} latest comparisons and "
        f"{failure_count} latest skipped or unavailable cells in {args.campaign_dir}"
    )


def summarize_campaign(campaign_dir: Path) -> list[dict[str, Any]]:
    attempts_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for path in sorted(campaign_dir.rglob("recommendation_summary.json")):
        summary = _read_json(path)
        managed = _read_json(path.parent / "managed_run.json")
        comparison = _dict(summary.get("baseline_comparison"))
        run_id = _str(managed.get("run_id")) or path.parent.name
        row = {
            "run_dir": str(path.parent),
            "run_id": run_id,
            "timestamp": _format_timestamp(_timestamp_key(run_id)),
            "model": _dict(summary.get("selected")).get("model") or managed.get("model"),
            "backend": _dict(summary.get("selected")).get("backend") or managed.get("backend"),
            "goal": _normalize_goal(managed.get("goal") or summary.get("goal")),
            "status": summary.get("status"),
            "confidence": summary.get("confidence"),
            "selected_candidate_id": comparison.get("selected_candidate_id"),
            "baseline_candidate_id": comparison.get("baseline_candidate_id"),
            "selected_is_baseline": comparison.get("selected_is_baseline"),
            "comparison_available": comparison.get("available") is True,
            "comparison_reason": comparison.get("reason"),
        }
        metrics = _dict(comparison.get("metrics"))
        for metric in METRICS:
            values = _dict(metrics.get(metric))
            row[f"{metric}_baseline"] = values.get("baseline")
            row[f"{metric}_selected"] = values.get("selected")
            row[f"{metric}_improvement_percent"] = values.get("improvement_percent")
        key = _result_key(row)
        row["_sort_timestamp"] = _timestamp_key(run_id) or _timestamp_key(str(path.parent))
        attempts_by_key.setdefault(key, []).append(row)

    rows = []
    for attempts in attempts_by_key.values():
        latest = max(attempts, key=_row_sort_value)
        latest["attempt_count"] = len(attempts)
        latest.pop("_sort_timestamp", None)
        rows.append(latest)
    return sorted(rows, key=lambda row: (_str(row.get("backend")), _str(row.get("goal")), _str(row.get("model"))))


def write_outputs(campaign_dir: Path, rows: list[dict[str, Any]]) -> tuple[int, int]:
    campaign_dir.mkdir(parents=True, exist_ok=True)
    failure_attempts = _read_failures(campaign_dir / "failures.tsv", campaign_dir)
    latest_rows, failures = _latest_results(rows, failure_attempts)
    managed_attempt_count = sum(_optional_int(row.get("attempt_count"), 1) for row in rows)
    json_path = campaign_dir / "overnight_summary.json"
    csv_path = campaign_dir / "overnight_summary.csv"
    markdown_path = campaign_dir / "overnight_summary.md"
    json_path.write_text(
        json.dumps(
            {
                "run_count": len(latest_rows),
                "attempt_count": managed_attempt_count,
                "failure_count": len(failures),
                "failure_attempt_count": len(failure_attempts),
                "runs": latest_rows,
                "failures": failures,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    fieldnames = list(latest_rows[0]) if latest_rows else _empty_fieldnames()
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(latest_rows)
    markdown_path.write_text(
        format_markdown(
            latest_rows,
            failures,
            managed_attempt_count=managed_attempt_count,
            failure_attempt_count=len(failure_attempts),
        ),
        encoding="utf-8",
    )
    return len(latest_rows), len(failures)


def format_markdown(
    rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    *,
    managed_attempt_count: int | None = None,
    failure_attempt_count: int | None = None,
) -> str:
    lines = [
        "# Overnight Campaign Summary",
        "",
        f"Latest measured comparisons: {len(rows)}",
        f"Managed run attempts found: {managed_attempt_count if managed_attempt_count is not None else len(rows)}",
        f"Latest skipped or unavailable cells: {len(failures)}",
        f"Skipped or failed attempts found: {failure_attempt_count if failure_attempt_count is not None else len(failures)}",
        "",
        "## Baseline Comparison",
        "",
    ]
    if rows:
        lines.append(
            "| Backend | Goal | Model | Status | Throughput | P95 latency | Average power | Joules per token | Tokens per watt | Run |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for row in rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _markdown_cell(row.get("backend")),
                        _markdown_cell(row.get("goal")),
                        _markdown_cell(row.get("model")),
                        _markdown_cell(row.get("status")),
                        _metric_cell(row, "throughput_tokens_per_sec"),
                        _metric_cell(row, "p95_latency_ms"),
                        _metric_cell(row, "average_power_w"),
                        _metric_cell(row, "joules_per_token"),
                        _metric_cell(row, "tokens_per_watt"),
                        _markdown_cell(row.get("run_dir")),
                    ]
                )
                + " |"
            )
    else:
        lines.append("No completed managed recommendations were found.")
    if failures:
        lines.extend(
            [
                "",
                "## Skipped Or Unavailable Cells",
                "",
                "| Backend | Goal | Model | Status | Reason | Log |",
                "|---|---|---|---|---|---|",
            ]
        )
        for failure in failures:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _markdown_cell(failure.get("backend")),
                        _markdown_cell(failure.get("goal")),
                        _markdown_cell(failure.get("model")),
                        _markdown_cell(failure.get("status")),
                        _markdown_cell(failure.get("reason")),
                        _markdown_cell(failure.get("log_path")),
                    ]
                )
                + " |"
            )
    return "\n".join(lines) + "\n"


def _empty_fieldnames() -> list[str]:
    fields = [
        "run_dir",
        "run_id",
        "timestamp",
        "attempt_count",
        "model",
        "backend",
        "goal",
        "status",
        "confidence",
        "selected_candidate_id",
        "baseline_candidate_id",
        "selected_is_baseline",
        "comparison_available",
        "comparison_reason",
    ]
    for metric in METRICS:
        fields.extend(
            [
                f"{metric}_baseline",
                f"{metric}_selected",
                f"{metric}_improvement_percent",
            ]
        )
    return fields


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_failures(path: Path, campaign_dir: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [_enrich_failure(dict(row), campaign_dir) for row in csv.DictReader(handle, delimiter="\t")]


def _dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _enrich_failure(row: dict[str, Any], campaign_dir: Path) -> dict[str, Any]:
    reason = _str(row.get("reason"))
    if reason not in {"", "command_failed"}:
        return row
    evidence = _failure_evidence(row, campaign_dir)
    enriched = _classify_failure_evidence(evidence)
    if enriched:
        row["reason"] = enriched
    return row


def _failure_evidence(row: dict[str, Any], campaign_dir: Path) -> str:
    paths = []
    log_path = _resolve_artifact_path(row.get("log_path"), campaign_dir)
    if log_path is not None:
        paths.append(log_path)
        log_text = _read_text(log_path)
        for match in re.finditer(r"artifacts:\s+(.+)", log_text):
            run_dir = _resolve_artifact_path(match.group(1).strip(), campaign_dir)
            if run_dir is not None:
                paths.extend(
                    [
                        run_dir / "candidate_failures.jsonl",
                        run_dir / "server_lifecycle.jsonl",
                        run_dir / "recommendation_summary.txt",
                    ]
                )
    return "\n".join(_read_text(item) for item in paths if item.exists())


def _resolve_artifact_path(value: object, campaign_dir: Path) -> Path | None:
    text = _str(value)
    if not text:
        return None
    path = Path(text)
    if path.is_absolute() or path.exists():
        return path
    candidate = campaign_dir / path
    if candidate.exists():
        return candidate
    return path


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _classify_failure_evidence(text: str) -> str:
    if not text:
        return ""
    lowered = text.lower()
    if re.search(r"out of memory|cuda out of memory|cannot allocate memory|\boom\b|killed", lowered):
        return "oom"
    if re.search(r"gated repo|restricted|not in the authorized list|401 client error|403 client error|authentication", lowered):
        return "model_access"
    if re.search(
        r"local hf snapshot.*has no files|\.incomplete|snapshot_download|hf_hub_download|huggingface.*download|failed to download|cannot find any model weights|unable to find weights",
        lowered,
    ):
        return "model_download"
    if re.search(r"sglang_grpc_port|server process exited with return code", lowered):
        return "backend_launch"
    if re.search(r"connection refused|health check timed out|no candidates were available for scoring|status.{0,16}timeout", lowered):
        return "startup_timeout"
    return ""


def _latest_results(
    rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    latest: dict[tuple[str, str, str], tuple[str, str, dict[str, Any]]] = {}
    for row in rows:
        key = _result_key(row)
        value = _timestamp_key(row.get("timestamp")) or _timestamp_key(row.get("run_id")) or _timestamp_key(row.get("run_dir"))
        event_type = "run" if _is_comparison_row(row) else "failure"
        payload = row if event_type == "run" else _failure_from_row(row)
        if key not in latest or (value, event_type) >= (latest[key][0], latest[key][1]):
            latest[key] = (value, event_type, payload)
    for failure in failures:
        key = _result_key(failure)
        value = _timestamp_key(failure.get("timestamp")) or _timestamp_key(failure.get("log_path"))
        if key not in latest or (value, "failure") >= (latest[key][0], latest[key][1]):
            latest[key] = (value, "failure", failure)

    latest_rows = [event[2] for event in latest.values() if event[1] == "run"]
    latest_failures = [event[2] for event in latest.values() if event[1] == "failure"]
    latest_rows.sort(key=lambda row: (_str(row.get("backend")), _str(row.get("goal")), _str(row.get("model"))))
    latest_failures.sort(key=lambda failure: (_str(failure.get("backend")), _str(failure.get("goal")), _str(failure.get("model"))))
    return latest_rows, latest_failures


def _result_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (_str(row.get("backend")), _normalize_goal(row.get("goal")), _str(row.get("model")))


def _is_comparison_row(row: dict[str, Any]) -> bool:
    return row.get("status") == "success" and row.get("comparison_available") is True


def _failure_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "timestamp": row.get("timestamp"),
        "backend": row.get("backend"),
        "goal": row.get("goal"),
        "model": row.get("model"),
        "status": row.get("status"),
        "reason": row.get("comparison_reason") or row.get("status") or "unavailable",
        "log_path": row.get("run_dir"),
    }


def _normalize_goal(value: object) -> str:
    goal = _str(value)
    if goal in {"efficiency", "efficient"}:
        return "energy_efficient"
    if goal == "performance":
        return "throughput"
    return goal


def _row_sort_value(row: dict[str, Any]) -> tuple[str, str]:
    return (_str(row.get("_sort_timestamp")), _str(row.get("run_dir")))


def _timestamp_key(value: object) -> str:
    text = _str(value)
    match = re.search(r"(\d{8}T\d{6}Z)", text)
    if match:
        return match.group(1)
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})Z", text)
    if match:
        return "".join(match.groups()[:3]) + "T" + "".join(match.groups()[3:]) + "Z"
    return ""


def _format_timestamp(value: str) -> str:
    match = re.fullmatch(r"(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z", value)
    if not match:
        return ""
    year, month, day, hour, minute, second = match.groups()
    return f"{year}-{month}-{day}T{hour}:{minute}:{second}Z"


def _str(value: object) -> str:
    return "" if value is None else str(value)


def _optional_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _metric_cell(row: dict[str, Any], metric: str) -> str:
    _, suffix, decimals = METRIC_LABELS[metric]
    baseline = _optional_float(row.get(f"{metric}_baseline"))
    selected = _optional_float(row.get(f"{metric}_selected"))
    improvement = _optional_float(row.get(f"{metric}_improvement_percent"))
    if baseline is None or selected is None:
        return "n/a"
    return (
        f"{baseline:.{decimals}f} {suffix} to {selected:.{decimals}f} {suffix}"
        f" ({_format_percent(improvement)})"
    )


def _format_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.1f}%"


def _optional_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _markdown_cell(value: object) -> str:
    text = "n/a" if value is None or value == "" else str(value)
    return text.replace("|", "\\|")


if __name__ == "__main__":
    main()
