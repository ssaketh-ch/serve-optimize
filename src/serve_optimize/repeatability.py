"""Managed recommendation repeatability analysis."""

from __future__ import annotations

import json
import os
from pathlib import Path
from statistics import mean
from typing import Any

from .evidence import stable_hash
from .io import write_json

REPEATABILITY_SCHEMA_VERSION = "recommendation-repeatability/v1"
CONFIG_FIELDS = (
    "backend",
    "model",
    "dtype",
    "quantization",
    "max_model_len",
    "gpu_memory_utilization",
    "max_num_seqs",
    "tensor_parallel_size",
    "benchmark_concurrency",
    "block_size",
    "kv_cache_dtype",
    "enforce_eager",
    "max_num_batched_tokens",
    "enable_chunked_prefill",
    "max_cudagraph_capture_size",
    "enable_prefix_caching",
)
METRIC_FIELDS = {
    "throughput_tokens_per_sec": ("metrics", "throughput_tokens_per_sec"),
    "p95_latency_ms": ("metrics", "p95_latency_ms"),
    "average_power_w": ("metrics", "average_power_w"),
    "joules_per_token": ("metrics", "joules_per_token"),
    "tokens_per_watt": ("metrics", "tokens_per_watt"),
}


def analyze_repeatability(run_dirs: list[Path]) -> dict[str, Any]:
    runs = [_load_run(Path(run_dir)) for run_dir in run_dirs]
    usable = [run for run in runs if run.get("usable")]
    warnings = [warning for run in runs for warning in run.get("warnings", [])]
    selected_config_fingerprints = [run["selected_config_fingerprint"] for run in usable if run.get("selected_config_fingerprint")]
    selected_candidate_ids = [run["selected_candidate_id"] for run in usable if run.get("selected_candidate_id")]
    selected_commands = [run["selected_command"] for run in usable if run.get("selected_command")]
    top3_overlaps = _pairwise_overlaps([set(run.get("top3_fingerprints", [])) for run in usable])
    pareto_overlaps = _pairwise_overlaps([set(run.get("pareto_fingerprints", [])) for run in usable])
    evidence_reuse = _evidence_reuse_summary(usable)
    return {
        "schema_version": REPEATABILITY_SCHEMA_VERSION,
        "run_count": len(runs),
        "usable_run_count": len(usable),
        "skipped_run_count": len(runs) - len(usable),
        "stability_classification": _stability_classification(usable, top3_overlaps, pareto_overlaps),
        "selected_command_stability": _value_stability(selected_commands),
        "selected_canonical_config_stability": _value_stability(selected_config_fingerprints),
        "selected_candidate_id_stability": _value_stability(selected_candidate_ids),
        "selected_metric_variation": _metric_variation(usable),
        "top3_overlap": _overlap_summary(top3_overlaps),
        "pareto_frontier_overlap": _overlap_summary(pareto_overlaps),
        "evidence_reuse": evidence_reuse,
        "runs": usable,
        "warnings": warnings,
        "notes": [
            "Repeatability compares managed run artifacts and selected canonical configs.",
            "It does not prove exhaustive search coverage.",
        ],
    }


def write_repeatability_artifacts(run_dirs: list[Path], *, output_dir: Path | None = None) -> dict[str, Any]:
    payload = analyze_repeatability(run_dirs)
    output_dir = output_dir or _default_output_dir(run_dirs)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "recommendation_repeatability.json"
    text_path = output_dir / "recommendation_repeatability.txt"
    write_json(json_path, payload)
    text_path.write_text(format_repeatability_text(payload), encoding="utf-8")
    payload["artifacts"] = {
        "recommendation_repeatability_json": str(json_path),
        "recommendation_repeatability_txt": str(text_path),
    }
    write_json(json_path, payload)
    text_path.write_text(format_repeatability_text(payload), encoding="utf-8")
    return payload


def format_repeatability_text(payload: dict[str, Any]) -> str:
    metric_variation = _dict(payload.get("selected_metric_variation"))
    evidence_reuse = _dict(payload.get("evidence_reuse"))
    lines = [
        "Serve Optimize Repeatability",
        "",
        f"run_count: {payload.get('run_count')}",
        f"usable_run_count: {payload.get('usable_run_count')}",
        f"skipped_run_count: {payload.get('skipped_run_count')}",
        f"stability: {payload.get('stability_classification')}",
        f"selected_config_unique_count: {_dict(payload.get('selected_canonical_config_stability')).get('unique_count')}",
        f"selected_command_unique_count: {_dict(payload.get('selected_command_stability')).get('unique_count')}",
        f"reuse_classification: {evidence_reuse.get('reuse_classification')}",
        "",
        "Selected metric variation:",
    ]
    for metric in METRIC_FIELDS:
        row = _dict(metric_variation.get(metric))
        lines.append(
            f"  {metric}: min={_display(row.get('min'))} max={_display(row.get('max'))} "
            f"delta={_display(row.get('absolute_delta'))} relative_delta={_display(row.get('relative_delta'))}"
        )
    if payload.get("warnings"):
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"  {warning}" for warning in payload.get("warnings", []))
    lines.append("")
    lines.append("Note: repeatability and fidelity are evaluated over observed managed run artifacts only.")
    lines.append("This is not an exhaustive search claim.")
    return "\n".join(lines) + "\n"


def _load_run(run_dir: Path) -> dict[str, Any]:
    warnings: list[str] = []
    summary = _dict(_read_json(run_dir / "recommendation_summary.json", warnings))
    recommendation_payload = _dict(_read_json(run_dir / "managed_recommendation.json", warnings))
    managed_run = _dict(_read_json(run_dir / "managed_run.json", warnings))
    pareto = _read_json(run_dir / "managed_pareto_frontier.json", warnings)
    recommendation = _dict(recommendation_payload.get("recommendation"))
    selected = _dict(summary.get("selected"))
    selected_config = _selected_config_payload(selected)
    candidate_table = _list(recommendation.get("candidate_table"))
    run = {
        "run_dir": str(run_dir),
        "usable": bool(summary and recommendation and selected_config),
        "warnings": warnings,
        "selected_candidate_id": selected.get("candidate_id") or recommendation.get("recommended_candidate_id"),
        "selected_command": summary.get("recommended_command") or recommendation.get("selected_serve_command"),
        "selected_config": selected_config,
        "selected_config_fingerprint": stable_hash({"selected_config": selected_config}) if selected_config else None,
        "metrics": _dict(summary.get("metrics")),
        "top3_fingerprints": [_candidate_fingerprint(row) for row in candidate_table[:3]],
        "pareto_fingerprints": [_candidate_fingerprint(row) for row in _list(pareto)],
        "managed_run": {
            "cold_launch_count": _first_int(managed_run, "cold_launch_count", "cold_launches"),
            "workload_measurement_count": _first_int(managed_run, "workload_measurement_count", "workload_measurements"),
            "evidence_hit_candidate_count": _first_int(managed_run, "evidence_hit_candidate_count", "evidence_hits"),
        },
    }
    if not run["usable"]:
        run["warnings"].append(f"{run_dir}: missing usable recommendation artifacts.")
    return run


def _selected_config_payload(selected: dict[str, Any]) -> dict[str, Any]:
    return {field: selected.get(field) for field in CONFIG_FIELDS if selected.get(field) is not None}


def _candidate_fingerprint(row: object) -> str:
    row = _dict(row)
    payload = {field: row.get(field) for field in CONFIG_FIELDS if row.get(field) is not None}
    return stable_hash({"candidate": payload})


def _read_json(path: Path, warnings: list[str]) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        warnings.append(f"{path}: missing.")
        return {}
    except json.JSONDecodeError as exc:
        warnings.append(f"{path}: invalid JSON: {exc}.")
        return {}
    except OSError as exc:
        warnings.append(f"{path}: {exc}.")
        return {}
    return value


def _value_stability(values: list[str]) -> dict[str, Any]:
    return {
        "observed_count": len(values),
        "unique_count": len(set(values)),
        "stable": bool(values and len(set(values)) == 1),
    }


def _metric_variation(runs: list[dict[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    result: dict[str, dict[str, float | int | None]] = {}
    for output_name, path in METRIC_FIELDS.items():
        values = [_optional_float(_nested(run, path)) for run in runs]
        values = [value for value in values if value is not None]
        if not values:
            result[output_name] = {"count": 0, "min": None, "max": None, "mean": None, "absolute_delta": None, "relative_delta": None}
            continue
        minimum = min(values)
        maximum = max(values)
        absolute_delta = maximum - minimum
        result[output_name] = {
            "count": len(values),
            "min": minimum,
            "max": maximum,
            "mean": mean(values),
            "absolute_delta": absolute_delta,
            "relative_delta": absolute_delta / minimum if minimum else None,
        }
    return result


def _pairwise_overlaps(sets: list[set[str]]) -> list[float]:
    overlaps: list[float] = []
    populated = [item for item in sets if item]
    for left_index, left in enumerate(populated):
        for right in populated[left_index + 1 :]:
            union = left | right
            overlaps.append(len(left & right) / len(union) if union else 0.0)
    return overlaps


def _overlap_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"pair_count": 0, "mean": None, "min": None, "max": None}
    return {"pair_count": len(values), "mean": mean(values), "min": min(values), "max": max(values)}


def _stability_classification(
    usable: list[dict[str, Any]],
    top3_overlaps: list[float],
    pareto_overlaps: list[float],
) -> str:
    if len(usable) < 2:
        return "insufficient_runs"
    fingerprints = {run.get("selected_config_fingerprint") for run in usable if run.get("selected_config_fingerprint")}
    if len(fingerprints) == 1:
        return "stable"
    top3_mean = mean(top3_overlaps) if top3_overlaps else 0.0
    pareto_mean = mean(pareto_overlaps) if pareto_overlaps else 0.0
    if top3_mean >= 0.5 or pareto_mean >= 0.5:
        return "mostly_stable"
    return "unstable"


def _evidence_reuse_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        return {
            "reuse_classification": "unavailable",
            "cold_launches": None,
            "workload_measurements": None,
            "evidence_hits": None,
        }
    cold = sum(_optional_int(_nested(run, ("managed_run", "cold_launch_count"))) or 0 for run in runs)
    measurements = sum(_optional_int(_nested(run, ("managed_run", "workload_measurement_count"))) or 0 for run in runs)
    hits = sum(_optional_int(_nested(run, ("managed_run", "evidence_hit_candidate_count"))) or 0 for run in runs)
    if hits > 0 and measurements == 0 and cold == 0:
        classification = "strong_reuse"
    elif hits > 0 and measurements > 0:
        classification = "partial_reuse"
    elif hits > 0:
        classification = "partial_reuse"
    else:
        classification = "weak_reuse"
    return {
        "reuse_classification": classification,
        "cold_launches": cold,
        "workload_measurements": measurements,
        "evidence_hits": hits,
    }


def _default_output_dir(run_dirs: list[Path]) -> Path:
    if not run_dirs:
        return Path.cwd()
    resolved = [Path(run_dir).resolve() for run_dir in run_dirs]
    common = Path.cwd()
    try:
        common = Path(os.path.commonpath([str(path.parent) for path in resolved]))
    except ValueError:
        pass
    return common


def _nested(row: dict[str, Any], path: tuple[str, str]) -> Any:
    current: Any = row
    for key in path:
        current = _dict(current).get(key)
    return current


def _dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _optional_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _optional_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _first_int(row: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = _optional_int(row.get(key))
        if value is not None:
            return value
    return None


def _display(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
