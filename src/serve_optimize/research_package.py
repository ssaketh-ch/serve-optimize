"""Research package artifacts built from existing managed run outputs."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import write_json
from .validation_campaign import analyze_validation_campaign

RESEARCH_PACKAGE_SCHEMA_VERSION = "research-package/v1"


def build_research_package(run_dirs: list[Path]) -> dict[str, Any]:
    campaign = analyze_validation_campaign(run_dirs)
    runs = [run for run in campaign.get("runs", []) if isinstance(run, dict)]
    usable = [run for run in runs if run.get("usable")]
    return {
        "schema_version": RESEARCH_PACKAGE_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "type": "existing_managed_run_artifacts",
            "run_dirs": [str(path) for path in run_dirs],
        },
        "summary": {
            "run_count": len(runs),
            "usable_run_count": len(usable),
            "backend_count": len(_values(usable, "backend")),
            "goal_count": len(_values(usable, "goal")),
            "workload_profile_count": len(_values(usable, "workload_profile_name")),
            "model_count": len(_selected_config_values(usable, "model")),
            "quantization_count": len(_selected_config_values(usable, "quantization")),
        },
        "coverage": {
            "backends": _values(usable, "backend"),
            "goals": _values(usable, "goal"),
            "workload_profiles": _values(usable, "workload_profile_name"),
            "models": _selected_config_values(usable, "model"),
            "dtypes": _selected_config_values(usable, "dtype"),
            "quantization": _selected_config_values(usable, "quantization"),
            "telemetry_quality": _values(usable, "telemetry_quality"),
        },
        "validation_campaign": campaign,
        "methodology": [
            "Use existing managed run artifacts only.",
            "Treat measured results and exact fresh measured evidence hits as final recommendation evidence.",
            "Scope recommendation claims to best among evaluated candidates.",
            "Record backend, runtime, workload, telemetry, evidence, and optimizer quality metadata with each package.",
            "Do not infer coverage for models, hardware, workloads, or backends not present in the supplied artifacts.",
        ],
        "required_tables": [
            "runs.csv",
            "coverage.csv",
        ],
        "notes": [
            "Research package creation does not launch servers or run benchmarks.",
            "Broader research claims require additional fresh runtime fingerprinted evidence.",
        ],
    }


def write_research_package_artifacts(run_dirs: list[Path], *, output_dir: Path) -> dict[str, Any]:
    payload = build_research_package(run_dirs)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "research_package.json"
    methodology_path = output_dir / "methodology.md"
    runs_csv_path = output_dir / "runs.csv"
    coverage_csv_path = output_dir / "coverage.csv"
    validation_path = output_dir / "validation_campaign.json"
    payload["artifacts"] = {
        "research_package_json": str(manifest_path),
        "methodology_md": str(methodology_path),
        "runs_csv": str(runs_csv_path),
        "coverage_csv": str(coverage_csv_path),
        "validation_campaign_json": str(validation_path),
    }
    write_json(manifest_path, payload)
    write_json(validation_path, payload["validation_campaign"])
    methodology_path.write_text(format_methodology(payload), encoding="utf-8")
    _write_runs_csv(runs_csv_path, payload["validation_campaign"].get("runs", []))
    _write_coverage_csv(coverage_csv_path, payload["coverage"])
    return payload


def format_methodology(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    lines = [
        "# Serve Optimize Research Package",
        "",
        "## Scope",
        "",
        "This package summarizes existing managed run artifacts. It does not run a new benchmark campaign.",
        "",
        "## Summary",
        "",
        f"* runs: {summary.get('usable_run_count')}/{summary.get('run_count')} usable",
        f"* backends: {summary.get('backend_count')}",
        f"* goals: {summary.get('goal_count')}",
        f"* workload profiles: {summary.get('workload_profile_count')}",
        f"* models: {summary.get('model_count')}",
        f"* quantization modes: {summary.get('quantization_count')}",
        "",
        "## Methodology",
        "",
    ]
    lines.extend(f"* {item}" for item in payload.get("methodology", []))
    lines.extend(["", "## Artifacts", ""])
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    lines.extend(f"* `{key}`: `{value}`" for key, value in artifacts.items())
    return "\n".join(lines) + "\n"


def _write_runs_csv(path: Path, runs: object) -> None:
    rows = [run for run in runs if isinstance(run, dict)]
    columns = [
        "run_dir",
        "backend",
        "goal",
        "status",
        "selected_candidate_id",
        "selected_score",
        "selected_rank",
        "selected_is_best_evaluated",
        "throughput_tokens_per_sec",
        "p95_latency_ms",
        "average_power_w",
        "joules_per_token",
        "tokens_per_watt",
        "telemetry_quality",
        "workload_profile_name",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})


def _write_coverage_csv(path: Path, coverage: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dimension", "value"])
        writer.writeheader()
        for dimension, values in coverage.items():
            if not isinstance(values, list):
                continue
            for value in values:
                writer.writerow({"dimension": dimension, "value": value})


def _values(runs: list[dict[str, Any]], key: str) -> list[str]:
    return sorted({str(run.get(key)) for run in runs if run.get(key) not in {None, ""}})


def _selected_config_values(runs: list[dict[str, Any]], key: str) -> list[str]:
    values = set()
    for run in runs:
        config = run.get("selected_config")
        if isinstance(config, dict) and config.get(key) not in {None, ""}:
            values.add(str(config.get(key)))
    return sorted(values)
