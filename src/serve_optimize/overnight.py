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
    json_path = campaign_dir / "overnight_summary.json"
    csv_path = campaign_dir / "overnight_summary.csv"
    json_path.write_text(json.dumps({"run_count": len(rows), "runs": rows}, indent=2, sort_keys=True) + "\n")
    fieldnames = list(rows[0]) if rows else _empty_fieldnames()
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _empty_fieldnames() -> list[str]:
    fields = [
        "run_dir",
        "model",
        "backend",
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


def _dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    main()
