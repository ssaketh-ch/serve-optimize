"""Validation campaign analysis over existing managed run artifacts."""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from .evidence import stable_hash
from .io import write_json
from .repeatability import CONFIG_FIELDS, analyze_repeatability

VALIDATION_CAMPAIGN_SCHEMA_VERSION = "validation-campaign/v1"
CSV_COLUMNS = (
    "run_dir",
    "backend",
    "goal",
    "status",
    "selected_config_fingerprint",
    "selected_candidate_id",
    "selected_score",
    "selected_rank",
    "selected_is_best_evaluated",
    "selected_is_pareto_optimal",
    "valid_candidate_count",
    "pareto_candidate_count",
    "throughput_tokens_per_sec",
    "p95_latency_ms",
    "average_power_w",
    "joules_per_token",
    "tokens_per_watt",
    "telemetry_quality",
    "evidence_exact_count",
    "evidence_prior_count",
    "synthesis_candidate_count",
    "workload_profile_name",
    "warning_count",
)
MEASURED_SOURCES = {"managed_measured", "managed_evidence_hit"}


def analyze_validation_campaign(run_dirs: list[Path]) -> dict[str, Any]:
    runs = [_load_campaign_run(Path(run_dir)) for run_dir in run_dirs]
    usable = [run for run in runs if run.get("usable")]
    warnings = [warning for run in runs for warning in _list(run.get("warnings"))]
    repeatability = analyze_repeatability([Path(run["run_dir"]) for run in runs]) if runs else _empty_repeatability()
    recommendation_quality = _recommendation_quality(runs)
    telemetry_quality = _telemetry_quality(runs)
    evidence_reuse = _evidence_reuse(runs)
    candidate_sources = _candidate_sources(runs)
    synthesis = _aiconfigurator_synthesis(runs)
    workload = _workload_coverage(runs)
    backend = _backend_coverage(runs)
    warnings.extend(_campaign_warnings(recommendation_quality, evidence_reuse, candidate_sources, synthesis, workload, backend))
    return {
        "schema_version": VALIDATION_CAMPAIGN_SCHEMA_VERSION,
        "run_count": len(runs),
        "usable_run_count": len(usable),
        "skipped_run_count": len(runs) - len(usable),
        "warnings": sorted(set(warnings)),
        "summary": {
            "goals": sorted({str(run.get("goal")) for run in usable if run.get("goal")}),
            "backends": sorted({str(run.get("backend")) for run in usable if run.get("backend")}),
            "quality_classification": recommendation_quality["classification"],
            "repeatability_classification": repeatability.get("stability_classification"),
            "telemetry_classification": telemetry_quality["classification"],
            "evidence_reuse_classification": evidence_reuse["classification"],
        },
        "recommendation_quality": recommendation_quality,
        "repeatability": repeatability,
        "telemetry_quality": telemetry_quality,
        "evidence_reuse": evidence_reuse,
        "candidate_sources": candidate_sources,
        "aiconfigurator_synthesis": synthesis,
        "workload_coverage": workload,
        "backend_coverage": backend,
        "runs": runs,
        "notes": [
            "Validation campaign analysis uses existing managed run artifacts only.",
            "Recommendation quality is scoped to evaluated candidates and does not imply exhaustive search coverage.",
        ],
    }


def write_validation_campaign_artifacts(run_dirs: list[Path], *, output_dir: Path | None = None) -> dict[str, Any]:
    payload = analyze_validation_campaign(run_dirs)
    output_dir = output_dir or _default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=False)
    json_path = output_dir / "validation_campaign.json"
    text_path = output_dir / "validation_campaign.txt"
    csv_path = output_dir / "validation_campaign_runs.csv"
    payload["artifacts"] = {
        "validation_campaign_json": str(json_path),
        "validation_campaign_txt": str(text_path),
        "validation_campaign_runs_csv": str(csv_path),
    }
    write_json(json_path, payload)
    text_path.write_text(format_validation_campaign_text(payload), encoding="utf-8")
    _write_campaign_csv(csv_path, _list(payload.get("runs")))
    return payload


def format_validation_campaign_text(payload: dict[str, Any]) -> str:
    summary = _dict(payload.get("summary"))
    repeatability = _dict(payload.get("repeatability"))
    telemetry = _dict(payload.get("telemetry_quality"))
    evidence = _dict(payload.get("evidence_reuse"))
    synthesis = _dict(payload.get("aiconfigurator_synthesis"))
    workload = _dict(payload.get("workload_coverage"))
    backend = _dict(payload.get("backend_coverage"))
    lines = [
        "Serve Optimize Validation Campaign",
        "",
        f"runs: {payload.get('usable_run_count')}/{payload.get('run_count')} usable",
        f"goals: {_join(summary.get('goals'))}",
        f"recommendation_quality: {summary.get('quality_classification')}",
        f"selected_config_stability: {repeatability.get('stability_classification')}",
        f"telemetry_quality: {telemetry.get('classification')}",
        f"evidence_reuse: {evidence.get('classification')}",
        f"aiconfigurator_synthesis: {synthesis.get('classification')}",
        f"workload_profiles: {_join(workload.get('profile_names'))}",
        f"backend_coverage: {backend.get('classification')}",
        "",
        "Key counts:",
        f"  exact_evidence_used: {evidence.get('used_as_exact_count')}",
        f"  prior_evidence_used: {evidence.get('used_as_prior_count')}",
        f"  safe_baseline_runs: {_dict(payload.get('candidate_sources')).get('safe_baseline_run_count')}",
        f"  synthesized_candidates: {synthesis.get('candidate_count')}",
        "",
        "Notes:",
        "  Best means best among evaluated candidates in the supplied artifacts.",
        "  This is not exhaustive search coverage.",
    ]
    warnings = _list(payload.get("warnings"))
    if warnings:
        lines.extend(["", "Warnings:"])
        lines.extend(f"  {warning}" for warning in warnings[:20])
        if len(warnings) > 20:
            lines.append(f"  ... {len(warnings) - 20} more")
    artifacts = _dict(payload.get("artifacts"))
    if artifacts:
        lines.extend(
            [
                "",
                "Artifacts:",
                f"  json: {artifacts.get('validation_campaign_json')}",
                f"  text: {artifacts.get('validation_campaign_txt')}",
                f"  csv: {artifacts.get('validation_campaign_runs_csv')}",
            ]
        )
    return "\n".join(lines) + "\n"


def _load_campaign_run(run_dir: Path) -> dict[str, Any]:
    warnings: list[str] = []
    managed_run = _dict(_read_json(run_dir / "managed_run.json", warnings))
    recommendation_payload = _dict(_read_json(run_dir / "managed_recommendation.json", warnings))
    summary = _dict(_read_json(run_dir / "recommendation_summary.json", warnings))
    pareto = _list(_read_json(run_dir / "managed_pareto_frontier.json", warnings))
    synthesis = _dict(_read_json(run_dir / "candidate_synthesis.json", warnings))
    recommendation = _dict(recommendation_payload.get("recommendation"))
    selected = _dict(summary.get("selected"))
    capabilities = _backend_capabilities(run_dir, managed_run, selected, warnings)
    evidence_decisions = _read_jsonl(run_dir / "evidence_decisions.jsonl", warnings)
    rendered_launches = _read_jsonl(run_dir / "rendered_launch_configs.jsonl", warnings)
    metrics = _dict(summary.get("metrics"))
    fidelity = _dict(summary.get("evaluated_set_fidelity")) or _dict(recommendation.get("evaluated_set_fidelity"))
    quality_audit = _dict(recommendation_payload.get("recommendation_quality_audit")) or _dict(managed_run.get("recommendation_quality_audit"))
    candidate_table = _list(recommendation.get("candidate_table"))
    selected_candidate_id = (
        selected.get("candidate_id")
        or recommendation.get("recommended_candidate_id")
        or quality_audit.get("selected_candidate_id")
    )
    selected_row = _find_candidate(candidate_table, selected_candidate_id)
    selected_config = _selected_config(selected, selected_row)
    selected_source = recommendation_payload.get("selected_source") or selected_row.get("source")
    selected_score = _first_float(quality_audit.get("selected_score"), recommendation.get("selected_score"), selected_row.get("score"))
    best_score = _best_candidate_score(candidate_table)
    evidence_summary = _evidence_decision_counts(evidence_decisions)
    synthesis_summary = _synthesis_summary(synthesis)
    workload_profile_name = _workload_profile_name(managed_run, selected, selected_row)
    backend_hash = capabilities.get("help_hash") or _dict(managed_run.get("backend_metadata")).get("argument_capabilities_help_hash")
    run = {
        "run_dir": str(run_dir),
        "usable": bool(summary and recommendation and selected_config),
        "status": managed_run.get("status") or recommendation_payload.get("status") or summary.get("status") or "unknown",
        "backend": managed_run.get("backend") or selected.get("backend") or selected_row.get("backend"),
        "goal": managed_run.get("goal") or summary.get("goal") or recommendation.get("goal"),
        "selected_candidate_id": selected_candidate_id,
        "selected_config": selected_config,
        "selected_config_fingerprint": stable_hash({"selected_config": selected_config}) if selected_config else None,
        "selected_command": summary.get("recommended_command") or recommendation.get("selected_serve_command"),
        "selected_score": selected_score,
        "best_evaluated_score": best_score,
        "selected_score_ratio_to_best": selected_score / best_score if selected_score is not None and best_score else None,
        "selected_rank": _first_int(fidelity.get("selected_rank"), quality_audit.get("selected_rank")),
        "selected_is_best_evaluated": _first_bool(fidelity.get("selected_is_best_evaluated"), quality_audit.get("selected_is_best_evaluated")),
        "selected_is_pareto_optimal": _first_bool(fidelity.get("selected_is_pareto_optimal"), quality_audit.get("selected_is_pareto_optimal"), selected_row.get("pareto_optimal")),
        "selected_source": selected_source,
        "selected_row_status": selected_row.get("status"),
        "valid_candidate_count": _first_int(fidelity.get("valid_candidate_count"), quality_audit.get("valid_candidate_count"), managed_run.get("completed_candidate_count")),
        "pareto_candidate_count": _first_int(fidelity.get("pareto_candidate_count"), quality_audit.get("pareto_candidate_count"), len(pareto)),
        "throughput_tokens_per_sec": _first_float(metrics.get("throughput_tokens_per_sec"), selected_row.get("total_tokens_s")),
        "p95_latency_ms": _first_float(metrics.get("p95_latency_ms"), _seconds_to_ms(selected_row.get("p95_latency_s"))),
        "average_power_w": _first_float(metrics.get("average_power_w"), selected_row.get("average_power_watts")),
        "joules_per_token": _first_float(metrics.get("joules_per_token"), selected_row.get("joules_per_token")),
        "tokens_per_watt": _first_float(metrics.get("tokens_per_watt"), selected_row.get("tokens_per_second_per_watt")),
        "failed_requests": _first_int(metrics.get("failed_requests"), selected_row.get("failed_requests")),
        "telemetry_quality": _telemetry_label(selected_row, metrics),
        "evidence_decision_summary": evidence_summary,
        "candidate_source_counts": _dict(managed_run.get("candidate_source_counts")),
        "aiconfigurator_synthesis_summary": synthesis_summary,
        "workload_profile": _dict(managed_run.get("workload_profile")),
        "workload_profile_name": workload_profile_name,
        "backend_capability_hash": backend_hash,
        "candidate_table_count": len(candidate_table),
        "rendered_launch_config_count": len(rendered_launches),
        "warnings": warnings,
    }
    run["warnings"].extend(_run_consistency_warnings(run, recommendation_payload, candidate_table, evidence_decisions, synthesis, rendered_launches))
    if not run["usable"]:
        run["warnings"].append(f"{run_dir}: missing usable managed recommendation artifacts.")
    return run


def _recommendation_quality(runs: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [run for run in runs if run.get("usable")]
    failures: list[str] = []
    reviews: list[str] = []
    ratios: list[float] = []
    for run in usable:
        label = run.get("run_dir")
        if run.get("selected_source") not in MEASURED_SOURCES:
            failures.append(f"{label}: selected source is not measured or exact evidence.")
        if run.get("selected_row_status") in {"failed", "rejected", "pruned"}:
            failures.append(f"{label}: selected row status is {run.get('selected_row_status')}.")
        if run.get("selected_is_best_evaluated") is False:
            reviews.append(f"{label}: selected candidate is not marked best among evaluated candidates.")
        if run.get("selected_is_pareto_optimal") is False:
            reviews.append(f"{label}: selected candidate is not marked Pareto optimal.")
        ratio = _optional_float(run.get("selected_score_ratio_to_best"))
        if ratio is not None:
            ratios.append(ratio)
    if not usable:
        classification = "fail"
        failures.append("No usable runs were found.")
    elif failures:
        classification = "fail"
    elif reviews:
        classification = "needs_review"
    else:
        classification = "pass"
    return {
        "classification": classification,
        "usable_run_count": len(usable),
        "selected_best_evaluated_count": sum(1 for run in usable if run.get("selected_is_best_evaluated") is True),
        "selected_pareto_count": sum(1 for run in usable if run.get("selected_is_pareto_optimal") is True),
        "measured_or_exact_selected_count": sum(1 for run in usable if run.get("selected_source") in MEASURED_SOURCES),
        "selected_score_ratio_to_best": {
            "count": len(ratios),
            "min": min(ratios) if ratios else None,
            "mean": mean(ratios) if ratios else None,
        },
        "failures": failures,
        "needs_review": reviews,
    }


def _telemetry_quality(runs: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [str(run.get("telemetry_quality") or "unavailable") for run in runs if run.get("usable")]
    counts = dict(Counter(labels))
    usable_power = sum(1 for run in runs if run.get("average_power_w") is not None and run.get("usable"))
    if not labels:
        classification = "unavailable"
    elif all(label == "good" for label in labels):
        classification = "good"
    elif any(label in {"good", "limited"} for label in labels):
        classification = "mixed"
    elif any(label in {"poor", "unavailable", "missing"} for label in labels):
        classification = "poor"
    else:
        classification = "unavailable"
    power_aware_weak = [
        run["run_dir"]
        for run in runs
        if run.get("goal") in {"balanced", "efficient", "efficiency"} and run.get("telemetry_quality") not in {"good", "limited"}
    ]
    return {
        "classification": classification,
        "counts": counts,
        "usable_power_run_count": usable_power,
        "missing_power_run_count": sum(1 for run in runs if run.get("usable") and run.get("average_power_w") is None),
        "power_aware_runs_with_weak_or_missing_telemetry": power_aware_weak,
    }


def _evidence_reuse(runs: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    exact = 0
    prior = 0
    invalid_exact: list[str] = []
    for run in runs:
        summary = _dict(run.get("evidence_decision_summary"))
        counts.update(_dict(summary.get("classification_counts")))
        exact += int(summary.get("used_as_exact_count") or 0)
        prior += int(summary.get("used_as_prior_count") or 0)
        invalid_exact.extend(_list(summary.get("invalid_exact_decisions")))
    if invalid_exact:
        classification = "fail"
    elif exact and prior:
        classification = "mixed"
    elif exact:
        classification = "exact_reuse"
    elif prior:
        classification = "prior_only"
    else:
        classification = "none"
    return {
        "classification": classification,
        "classification_counts": dict(counts),
        "used_as_exact_count": exact,
        "used_as_prior_count": prior,
        "invalid_exact_decisions": invalid_exact,
    }


def _candidate_sources(runs: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: Counter[str] = Counter()
    missing_safe = []
    for run in runs:
        counts = _dict(run.get("candidate_source_counts"))
        aggregate.update({str(key): int(value) for key, value in counts.items() if _optional_int(value) is not None})
        if run.get("usable") and int(counts.get("safe_baseline") or 0) < 1:
            missing_safe.append(run["run_dir"])
    return {
        "counts": dict(aggregate),
        "safe_baseline_run_count": sum(1 for run in runs if int(_dict(run.get("candidate_source_counts")).get("safe_baseline") or 0) > 0),
        "safe_baseline_missing_runs": missing_safe,
        "classification": "needs_review" if missing_safe else "pass",
    }


def _aiconfigurator_synthesis(runs: list[dict[str, Any]]) -> dict[str, Any]:
    statuses: Counter[str] = Counter()
    system_keys: Counter[str] = Counter()
    local_gpu_runs: list[str] = []
    candidate_count = 0
    for run in runs:
        summary = _dict(run.get("aiconfigurator_synthesis_summary"))
        candidate_count += int(summary.get("candidate_count") or 0)
        statuses.update(_dict(summary.get("statuses")))
        for key in _list(summary.get("system_keys")):
            system_keys[str(key)] += 1
        if summary.get("uses_local_gpu"):
            local_gpu_runs.append(run["run_dir"])
    if local_gpu_runs:
        classification = "fail"
    elif candidate_count:
        classification = "used"
    else:
        classification = "unused"
    return {
        "classification": classification,
        "candidate_count": candidate_count,
        "statuses": dict(statuses),
        "system_keys": dict(system_keys),
        "local_gpu_runs": local_gpu_runs,
    }


def _workload_coverage(runs: list[dict[str, Any]]) -> dict[str, Any]:
    profile_names = [str(run.get("workload_profile_name") or "missing") for run in runs if run.get("usable")]
    counts = dict(Counter(profile_names))
    prefix_default = []
    for run in runs:
        profile = _dict(run.get("workload_profile"))
        if run.get("selected_config", {}).get("enable_prefix_caching") is True:
            profile_name = str(run.get("workload_profile_name") or "default")
            if profile_name == "default" and not profile.get("prefix_reuse_expected"):
                prefix_default.append(run["run_dir"])
    return {
        "profile_names": sorted(counts),
        "profile_counts": counts,
        "default_profile_count": counts.get("default", 0),
        "repeated_prefix_profile_count": counts.get("repeated_prefix", 0),
        "prefix_caching_without_reuse_runs": prefix_default,
        "classification": "needs_review" if prefix_default or counts.get("missing") else "pass",
    }


def _backend_coverage(runs: list[dict[str, Any]]) -> dict[str, Any]:
    statuses: dict[str, Counter[str]] = {}
    failures: list[str] = []
    for run in runs:
        backend = str(run.get("backend") or "unknown")
        statuses.setdefault(backend, Counter())[str(run.get("status") or "unknown")] += 1
        if backend in {"trt-llm", "tensorrt-llm", "tensorrt_llm"} and run.get("status") == "success":
            failures.append(f"{run['run_dir']}: TensorRT-LLM managed runtime success was reported.")
    successful_vllm = statuses.get("vllm", Counter()).get("success", 0)
    successful_sglang = statuses.get("sglang", Counter()).get("success", 0)
    unavailable_sglang = sum(
        count
        for status, count in statuses.get("sglang", Counter()).items()
        if status in {"failed", "unavailable", "unsupported"}
    )
    if failures:
        classification = "fail"
    elif successful_sglang:
        classification = "sglang_present"
    elif statuses.get("vllm"):
        classification = "vllm_only"
    elif unavailable_sglang:
        classification = "sglang_unavailable"
    else:
        classification = "needs_review"
    return {
        "classification": classification,
        "backend_status_counts": {backend: dict(counter) for backend, counter in statuses.items()},
        "successful_vllm_run_count": successful_vllm,
        "successful_sglang_run_count": successful_sglang,
        "unavailable_sglang_run_count": unavailable_sglang,
        "failures": failures,
    }


def _run_consistency_warnings(
    run: dict[str, Any],
    recommendation_payload: dict[str, Any],
    candidate_table: list[Any],
    evidence_decisions: list[dict[str, Any]],
    synthesis: dict[str, Any],
    rendered_launches: list[dict[str, Any]],
) -> list[str]:
    warnings: list[str] = []
    label = run["run_dir"]
    if run.get("selected_source") not in MEASURED_SOURCES:
        warnings.append(f"{label}: selected recommendation is not measured or exact fresh evidence.")
    if not candidate_table:
        warnings.append(f"{label}: managed recommendation candidate table is missing.")
    if run.get("selected_candidate_id") and not _find_candidate(candidate_table, run.get("selected_candidate_id")):
        warnings.append(f"{label}: selected candidate is missing from candidate table.")
    if _dict(run.get("candidate_source_counts")).get("safe_baseline") in {None, 0}:
        warnings.append(f"{label}: safe baseline candidate source was not recorded.")
    for decision in evidence_decisions:
        classification = decision.get("classification")
        if decision.get("used_as_exact") and classification != "exact_fresh":
            warnings.append(f"{label}: {classification} evidence was used as exact.")
    if _synthesis_summary(synthesis).get("uses_local_gpu"):
        warnings.append(f"{label}: AIConfigurator synthesis used local_gpu.")
    if _predicted_metrics_used_as_selected(recommendation_payload):
        warnings.append(f"{label}: selected recommendation appears to use predicted metrics as selected metrics.")
    if _prefix_caching_without_reuse(run, rendered_launches):
        warnings.append(f"{label}: prefix caching appears under a default or non reuse workload profile.")
    if _contains_forbidden_global_claim(run):
        warnings.append(f"{label}: artifact text contains forbidden exhaustive optimum wording.")
    return warnings


def _campaign_warnings(*sections: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    for section in sections:
        for key in ("failures", "needs_review", "invalid_exact_decisions", "safe_baseline_missing_runs", "local_gpu_runs", "prefix_caching_without_reuse_runs"):
            for item in _list(section.get(key)):
                warnings.append(str(item))
    return warnings


def _evidence_decision_counts(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    classification_counts: Counter[str] = Counter()
    used_exact = 0
    used_prior = 0
    invalid_exact = []
    for decision in decisions:
        classification = str(decision.get("classification") or "missing")
        classification_counts[classification] += 1
        if decision.get("used_as_exact"):
            used_exact += 1
            if classification != "exact_fresh":
                invalid_exact.append(f"{decision.get('candidate_id')}: {classification} used as exact")
        if decision.get("used_as_prior"):
            used_prior += 1
    return {
        "classification_counts": dict(classification_counts),
        "used_as_exact_count": used_exact,
        "used_as_prior_count": used_prior,
        "invalid_exact_decisions": invalid_exact,
    }


def _synthesis_summary(synthesis: dict[str, Any]) -> dict[str, Any]:
    records = _list(synthesis.get("candidate_records"))
    statuses = Counter(str(_dict(record).get("status") or "unknown") for record in records)
    keys = []
    uses_local_gpu = False
    for record in records:
        row = _dict(record)
        key = row.get("aiconfigurator_system_key")
        if key:
            keys.append(str(key))
        if key == "local_gpu":
            uses_local_gpu = True
        constraints = _dict(row.get("synthesis_constraints"))
        if constraints.get("aiconfigurator_system_key") == "local_gpu":
            uses_local_gpu = True
    for result in _list(synthesis.get("provider_results")):
        row = _dict(result)
        constraints = _dict(row.get("constraints_used"))
        if constraints.get("aiconfigurator_system_key") == "local_gpu":
            uses_local_gpu = True
        if constraints.get("aiconfigurator_system_key"):
            keys.append(str(constraints.get("aiconfigurator_system_key")))
    return {
        "candidate_count": len(records),
        "statuses": dict(statuses),
        "system_keys": sorted(set(keys)),
        "uses_local_gpu": uses_local_gpu,
    }


def _write_campaign_csv(path: Path, runs: list[Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for run in runs:
            row = _dict(run)
            evidence = _dict(row.get("evidence_decision_summary"))
            writer.writerow(
                {
                    "run_dir": row.get("run_dir"),
                    "backend": row.get("backend"),
                    "goal": row.get("goal"),
                    "status": row.get("status"),
                    "selected_config_fingerprint": row.get("selected_config_fingerprint"),
                    "selected_candidate_id": row.get("selected_candidate_id"),
                    "selected_score": row.get("selected_score"),
                    "selected_rank": row.get("selected_rank"),
                    "selected_is_best_evaluated": row.get("selected_is_best_evaluated"),
                    "selected_is_pareto_optimal": row.get("selected_is_pareto_optimal"),
                    "valid_candidate_count": row.get("valid_candidate_count"),
                    "pareto_candidate_count": row.get("pareto_candidate_count"),
                    "throughput_tokens_per_sec": row.get("throughput_tokens_per_sec"),
                    "p95_latency_ms": row.get("p95_latency_ms"),
                    "average_power_w": row.get("average_power_w"),
                    "joules_per_token": row.get("joules_per_token"),
                    "tokens_per_watt": row.get("tokens_per_watt"),
                    "telemetry_quality": row.get("telemetry_quality"),
                    "evidence_exact_count": evidence.get("used_as_exact_count"),
                    "evidence_prior_count": evidence.get("used_as_prior_count"),
                    "synthesis_candidate_count": _dict(row.get("aiconfigurator_synthesis_summary")).get("candidate_count"),
                    "workload_profile_name": row.get("workload_profile_name"),
                    "warning_count": len(_list(row.get("warnings"))),
                }
            )


def _selected_config(selected: dict[str, Any], selected_row: dict[str, Any]) -> dict[str, Any]:
    row = dict(selected)
    for key in CONFIG_FIELDS:
        if key not in row and selected_row.get(key) is not None:
            row[key] = selected_row.get(key)
    return {key: row.get(key) for key in CONFIG_FIELDS if row.get(key) is not None}


def _find_candidate(candidate_table: list[Any], candidate_id: object) -> dict[str, Any]:
    if candidate_id is None:
        return {}
    for row in candidate_table:
        candidate = _dict(row)
        if candidate.get("candidate_id") == candidate_id:
            return candidate
    return {}


def _best_candidate_score(candidate_table: list[Any]) -> float | None:
    scores = [_optional_float(_dict(row).get("score")) for row in candidate_table]
    scores = [score for score in scores if score is not None]
    return max(scores) if scores else None


def _telemetry_label(selected_row: dict[str, Any], metrics: dict[str, Any]) -> str:
    label = selected_row.get("telemetry_quality")
    if label:
        return str(label)
    return "missing" if metrics.get("average_power_w") is None else "limited"


def _workload_profile_name(managed_run: dict[str, Any], selected: dict[str, Any], selected_row: dict[str, Any]) -> str | None:
    profile = _dict(managed_run.get("workload_profile"))
    return (
        profile.get("profile_name")
        or selected.get("workload_profile")
        or selected_row.get("workload_profile")
        or None
    )


def _prefix_caching_without_reuse(run: dict[str, Any], rendered_launches: list[dict[str, Any]]) -> bool:
    profile = _dict(run.get("workload_profile"))
    profile_name = run.get("workload_profile_name") or profile.get("profile_name") or "default"
    if profile_name != "default" or profile.get("prefix_reuse_expected"):
        return False
    selected = _dict(run.get("selected_config"))
    if selected.get("enable_prefix_caching") is True:
        return True
    for launch in rendered_launches:
        config = _dict(_dict(launch).get("canonical_config"))
        if config.get("enable_prefix_caching") is True:
            return True
    return False


def _predicted_metrics_used_as_selected(recommendation_payload: dict[str, Any]) -> bool:
    recommendation = _dict(recommendation_payload.get("recommendation"))
    selected_source = recommendation_payload.get("selected_source")
    if selected_source and selected_source not in MEASURED_SOURCES:
        return True
    measured = _dict(recommendation.get("measured_metrics"))
    predicted = _dict(recommendation.get("predicted_metrics"))
    return bool(predicted and measured == predicted)


def _contains_forbidden_global_claim(run: dict[str, Any]) -> bool:
    command = str(run.get("selected_command") or "").lower()
    return "global optimum" in command or "global optimality" in command


def _read_json(path: Path, warnings: list[str]) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        warnings.append(f"{path}: missing.")
    except json.JSONDecodeError as exc:
        warnings.append(f"{path}: invalid JSON: {exc}.")
    except OSError as exc:
        warnings.append(f"{path}: {exc}.")
    return {}


def _backend_capabilities(
    run_dir: Path,
    managed_run: dict[str, Any],
    selected: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    backend = str(managed_run.get("backend") or selected.get("backend") or "")
    if backend == "sglang":
        return _dict(_read_json(run_dir / "sglang_argument_capabilities.json", warnings))
    if backend == "vllm":
        return _dict(_read_json(run_dir / "vllm_argument_capabilities.json", warnings))
    return {}


def _read_jsonl(path: Path, warnings: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        warnings.append(f"{path}: missing.")
        return rows
    except OSError as exc:
        warnings.append(f"{path}: {exc}.")
        return rows
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            warnings.append(f"{path}:{index}: invalid JSON: {exc}.")
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = Path("results") / "validation-campaign" / stamp
    if not base.exists():
        return base
    for index in range(2, 1000):
        candidate = base.with_name(f"{base.name}-{index}")
        if not candidate.exists():
            return candidate
    raise RuntimeError("Could not allocate a validation campaign output directory.")


def _empty_repeatability() -> dict[str, Any]:
    return {
        "schema_version": "recommendation-repeatability/v1",
        "run_count": 0,
        "usable_run_count": 0,
        "skipped_run_count": 0,
        "stability_classification": "insufficient_runs",
        "warnings": [],
    }


def _seconds_to_ms(value: object) -> float | None:
    seconds = _optional_float(value)
    return seconds * 1000.0 if seconds is not None else None


def _first_float(*values: object) -> float | None:
    for value in values:
        converted = _optional_float(value)
        if converted is not None:
            return converted
    return None


def _first_int(*values: object) -> int | None:
    for value in values:
        converted = _optional_int(value)
        if converted is not None:
            return converted
    return None


def _first_bool(*values: object) -> bool | None:
    for value in values:
        if isinstance(value, bool):
            return value
    return None


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


def _dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _join(value: object) -> str:
    items = [str(item) for item in _list(value) if item]
    return ", ".join(items) if items else "n/a"
