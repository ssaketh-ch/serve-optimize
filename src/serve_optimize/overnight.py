"""Summarize safe baseline comparisons from an overnight campaign."""

from __future__ import annotations

import argparse
import csv
import json
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
    write_outputs(args.campaign_dir, rows)
    print(f"Summarized {len(rows)} managed runs in {args.campaign_dir}")


def summarize_campaign(campaign_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(campaign_dir.rglob("recommendation_summary.json")):
        summary = _read_json(path)
        managed = _read_json(path.parent / "managed_run.json")
        comparison = _dict(summary.get("baseline_comparison"))
        row = {
            "run_dir": str(path.parent),
            "model": _dict(summary.get("selected")).get("model") or managed.get("model"),
            "backend": _dict(summary.get("selected")).get("backend") or managed.get("backend"),
            "goal": summary.get("goal") or managed.get("goal"),
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
        rows.append(row)
    return rows


def write_outputs(campaign_dir: Path, rows: list[dict[str, Any]]) -> None:
    campaign_dir.mkdir(parents=True, exist_ok=True)
    failures = _read_failures(campaign_dir / "failures.tsv")
    json_path = campaign_dir / "overnight_summary.json"
    csv_path = campaign_dir / "overnight_summary.csv"
    markdown_path = campaign_dir / "overnight_summary.md"
    json_path.write_text(
        json.dumps(
            {
                "run_count": len(rows),
                "failure_count": len(failures),
                "runs": rows,
                "failures": failures,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    fieldnames = list(rows[0]) if rows else _empty_fieldnames()
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    markdown_path.write_text(format_markdown(rows, failures), encoding="utf-8")


def format_markdown(rows: list[dict[str, Any]], failures: list[dict[str, Any]]) -> str:
    lines = [
        "# Overnight Campaign Summary",
        "",
        f"Completed managed runs: {len(rows)}",
        f"Skipped or failed cells: {len(failures)}",
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
                "## Skipped Cells",
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


def _read_failures(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


def _dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


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
