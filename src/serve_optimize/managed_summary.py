"""Concise Managed Mode recommendation summaries."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

from .schemas import RecommendationResult, ServingConfig

SUMMARY_SCHEMA_VERSION = "recommendation-summary/v1"


def write_recommendation_summary_artifacts(
    *,
    txt_path: Path,
    json_path: Path,
    recommendation: RecommendationResult,
    selected_config: ServingConfig | None,
    selected_source: str | None,
    reason: str | None,
    artifacts: dict[str, str],
    runtime_environment: dict[str, Any] | None = None,
    selected_runtime_fingerprint: str | None = None,
) -> dict[str, Any]:
    payload = build_recommendation_summary(
        recommendation=recommendation,
        selected_config=selected_config,
        selected_source=selected_source,
        reason=reason,
        runtime_environment=runtime_environment,
        selected_runtime_fingerprint=selected_runtime_fingerprint,
        artifacts=artifacts,
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(format_recommendation_summary_text(payload), encoding="utf-8")
    return payload


def build_recommendation_summary(
    *,
    recommendation: RecommendationResult,
    selected_config: ServingConfig | None,
    selected_source: str | None,
    reason: str | None,
    artifacts: dict[str, str],
    runtime_environment: dict[str, Any] | None = None,
    selected_runtime_fingerprint: str | None = None,
) -> dict[str, Any]:
    status = "success" if recommendation.recommended_candidate_id else "unavailable"
    command = recommendation.selected_serve_command or recommended_command(selected_config)
    selected = selected_summary(recommendation, selected_config)
    metrics = metrics_summary(recommendation)
    fidelity = fidelity_summary(recommendation)
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "status": status,
        "reason": None if status == "success" else (reason or "Recommendation unavailable."),
        "goal": _text(recommendation.goal),
        "confidence": _text(recommendation.confidence_level).lower(),
        "recommendation_type": recommendation_type(recommendation, selected_source),
        "recommended_command": command,
        "selected": selected,
        "metrics": metrics,
        "evaluated_set_fidelity": fidelity,
        "runtime_environment": runtime_environment or {},
        "selected_runtime_fingerprint": selected_runtime_fingerprint,
        "why": why_summary(recommendation, selected),
        "artifacts": {
            "summary_text": _artifact_name(artifacts.get("recommendation_summary_txt"), "recommendation_summary.txt"),
            "summary_json": _artifact_name(artifacts.get("recommendation_summary_json"), "recommendation_summary.json"),
            "full_recommendation": _artifact_name(artifacts.get("managed_recommendation_json"), "managed_recommendation.json"),
            "pareto_frontier": _artifact_name(artifacts.get("managed_pareto_frontier_json"), "managed_pareto_frontier.json"),
            "report": _artifact_name(artifacts.get("managed_report_txt"), "managed_report.txt"),
            "run": _artifact_name(artifacts.get("managed_run_json"), "managed_run.json"),
        },
    }


def recommended_command(config: ServingConfig | None) -> str:
    if config is None:
        return "n/a"
    if config.backend == "sglang":
        command = [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            config.model_id,
            "--dtype",
            config.dtype,
            "--context-length",
            str(config.max_context_tokens),
        ]
        if config.tensor_parallelism > 1:
            command.extend(["--tp-size", str(config.tensor_parallelism)])
        return shlex.join(command)
    command = [
        "vllm",
        "serve",
        config.model_id,
        "--dtype",
        config.dtype,
        "--max-model-len",
        str(config.max_context_tokens),
        "--gpu-memory-utilization",
        _number_text(config.gpu_memory_utilization),
        "--max-num-seqs",
        str(config.max_batch_size),
        "--tensor-parallel-size",
        str(config.tensor_parallelism),
    ]
    if config.block_size is not None:
        command.extend(["--block-size", str(config.block_size)])
    if config.kv_cache_dtype is not None:
        command.extend(["--kv-cache-dtype", config.kv_cache_dtype])
    if config.enforce_eager is True:
        command.append("--enforce-eager")
    if config.max_num_batched_tokens is not None:
        command.extend(["--max-num-batched-tokens", str(config.max_num_batched_tokens)])
    if config.enable_chunked_prefill is True:
        command.append("--enable-chunked-prefill")
    elif config.enable_chunked_prefill is False:
        command.append("--no-enable-chunked-prefill")
    if config.max_cudagraph_capture_size is not None:
        command.extend(["--max-cudagraph-capture-size", str(config.max_cudagraph_capture_size)])
    if config.enable_prefix_caching is True:
        command.append("--enable-prefix-caching")
    elif config.enable_prefix_caching is False:
        command.append("--no-enable-prefix-caching")
    if config.quantization != "none":
        command.extend(["--quantization", config.quantization])
    return shlex.join(command)


def selected_summary(recommendation: RecommendationResult, config: ServingConfig | None) -> dict[str, Any]:
    extra = dict(config.extra or {}) if config else {}
    selected = {
        "candidate_id": _text(recommendation.recommended_candidate_id),
        "backend": _text(config.backend if config else recommendation.backend),
        "model": _text(config.model_id if config else recommendation.model),
        "dtype": _text(config.dtype if config else None),
        "quantization": _text(config.quantization if config else None),
        "max_model_len": config.max_context_tokens if config else None,
        "gpu_memory_utilization": config.gpu_memory_utilization if config else None,
        "max_num_seqs": config.max_batch_size if config else None,
        "tensor_parallel_size": config.tensor_parallelism if config else None,
        "benchmark_concurrency": _optional_int(extra.get("workload_concurrency")) if config else None,
    }
    if config is not None:
        if config.backend != "vllm":
            selected.pop("gpu_memory_utilization", None)
        profile = extra.get("workload_profile")
        if isinstance(profile, dict) and profile.get("profile_name"):
            selected["workload_profile"] = profile.get("profile_name")
        _add_if_present(selected, "block_size", config.block_size)
        _add_if_present(selected, "kv_cache_dtype", config.kv_cache_dtype)
        _add_if_present(selected, "enforce_eager", config.enforce_eager)
        _add_if_present(selected, "max_num_batched_tokens", config.max_num_batched_tokens)
        _add_if_present(selected, "enable_chunked_prefill", config.enable_chunked_prefill)
        _add_if_present(selected, "max_cudagraph_capture_size", config.max_cudagraph_capture_size)
        _add_if_present(selected, "enable_prefix_caching", config.enable_prefix_caching)
    return selected


def metrics_summary(recommendation: RecommendationResult) -> dict[str, Any]:
    measured = recommendation.measured_metrics or {}
    telemetry = recommendation.telemetry_metrics or {}
    return {
        "throughput_tokens_per_sec": _optional_float(measured.get("total_tokens_s")),
        "p95_latency_ms": _seconds_to_ms(measured.get("p95_latency_s")),
        "average_power_w": _optional_float(telemetry.get("average_power_watts")),
        "joules_per_token": _optional_float(telemetry.get("joules_per_token")),
        "tokens_per_watt": _optional_float(telemetry.get("tokens_per_second_per_watt")),
        "failed_requests": _optional_int(measured.get("failed_requests")),
    }


def fidelity_summary(recommendation: RecommendationResult) -> dict[str, Any]:
    fidelity = dict(recommendation.evaluated_set_fidelity or {})
    if not fidelity:
        return {}
    return {
        "scope": fidelity.get("scope"),
        "selected_rank": fidelity.get("selected_rank"),
        "selected_score_over_best_score": fidelity.get("selected_score_over_best_score"),
        "gap_to_best_score": fidelity.get("gap_to_best_score"),
        "selected_is_best_evaluated": fidelity.get("selected_is_best_evaluated"),
        "selected_is_pareto_optimal": fidelity.get("selected_is_pareto_optimal"),
        "valid_candidate_count": fidelity.get("valid_candidate_count"),
        "pareto_candidate_count": fidelity.get("pareto_candidate_count"),
        "note": "Evaluated-set fidelity only; not an exhaustive search claim.",
    }


def recommendation_type(recommendation: RecommendationResult, selected_source: str | None) -> str:
    if recommendation.recommended_candidate_id is None:
        return "unavailable"
    source = selected_source or ""
    if source == "managed_evidence_hit":
        return "exact fresh measured evidence recommendation"
    if recommendation.was_comparative:
        return "comparative measured recommendation"
    return "single-candidate measured validation"


def why_summary(recommendation: RecommendationResult, selected: dict[str, Any]) -> list[str]:
    reasons = [str(reason) for reason in recommendation.selection_reasons if reason]
    if not reasons and recommendation.recommended_candidate_id:
        reasons.append("Selected by measured recommendation score.")
    failed = selected.get("candidate_id")
    if failed and (recommendation.measured_metrics or {}).get("failed_requests") == 0:
        zero_failed = "It completed with 0 failed requests."
        if zero_failed not in reasons:
            reasons.append(zero_failed)
    return reasons[:5] if reasons else ["n/a"]


def format_recommendation_summary_text(payload: dict[str, Any]) -> str:
    selected = _dict(payload.get("selected"))
    metrics = _dict(payload.get("metrics"))
    fidelity = _dict(payload.get("evaluated_set_fidelity"))
    lines = [
        "=" * 60,
        "Serve Optimize Recommendation",
        "=" * 60,
        "",
        f"Goal: {_display(payload.get('goal'))}",
        f"Confidence: {_display(payload.get('confidence')).upper()}",
        f"Recommendation type: {_display(payload.get('recommendation_type'))}",
        "",
    ]
    if payload.get("status") != "success":
        lines.extend(
            [
                "Recommendation: unavailable",
                f"Reason: {_display(payload.get('reason'))}",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Recommended serve command:",
                "",
                _display(payload.get("recommended_command")),
                "",
                "Selected configuration:",
                f"  backend: {_display(selected.get('backend'))}",
                f"  dtype: {_display(selected.get('dtype'))}",
                f"  quantization: {_display(selected.get('quantization'))}",
                f"  max_model_len: {_display(selected.get('max_model_len'))}",
                f"  gpu_memory_utilization: {_display(selected.get('gpu_memory_utilization'))}",
                f"  max_num_seqs: {_display(selected.get('max_num_seqs'))}",
                f"  tensor_parallel_size: {_display(selected.get('tensor_parallel_size'))}",
                f"  benchmark_concurrency: {_display(selected.get('benchmark_concurrency'))}",
            ]
        )
        lines.extend(_selected_optional_lines(selected))
        lines.extend(
            [
                "",
                "Measured performance:",
                f"  throughput: {_metric(metrics.get('throughput_tokens_per_sec'), ' tokens/sec')}",
                f"  p95_latency: {_metric(metrics.get('p95_latency_ms'), ' ms')}",
                f"  average_power: {_metric(metrics.get('average_power_w'), ' W')}",
                f"  joules_per_token: {_metric(metrics.get('joules_per_token'), '')}",
                f"  tokens_per_watt: {_metric(metrics.get('tokens_per_watt'), '')}",
                "",
                "Evaluated-set fidelity:",
                f"  selected_rank: {_display(fidelity.get('selected_rank'))}",
                f"  selected_is_best_evaluated: {_display(fidelity.get('selected_is_best_evaluated'))}",
                f"  selected_is_pareto_optimal: {_display(fidelity.get('selected_is_pareto_optimal'))}",
                f"  selected_score_over_best_score: {_metric(fidelity.get('selected_score_over_best_score'), '')}",
                "  note: best among evaluated candidates only",
                "",
                "Why this won:",
            ]
        )
        lines.extend(f"  - {reason}" for reason in payload.get("why", ["n/a"]))
        lines.append("")
    lines.extend(
        [
            "Full artifacts:",
            f"  {_display(_dict(payload.get('artifacts')).get('full_recommendation'))}",
            f"  {_display(_dict(payload.get('artifacts')).get('pareto_frontier'))}",
            f"  {_display(_dict(payload.get('artifacts')).get('report'))}",
            f"  {_display(_dict(payload.get('artifacts')).get('run'))}",
            "",
        ]
    )
    return "\n".join(lines)


def _artifact_name(value: str | None, default: str) -> str:
    if not value:
        return default
    return Path(value).name


def _add_if_present(row: dict[str, Any], key: str, value: object) -> None:
    if value is not None:
        row[key] = value


def _selected_optional_lines(selected: dict[str, Any]) -> list[str]:
    keys = [
        "block_size",
        "kv_cache_dtype",
        "enforce_eager",
        "max_num_batched_tokens",
        "enable_chunked_prefill",
        "max_cudagraph_capture_size",
        "enable_prefix_caching",
        "workload_profile",
    ]
    return [f"  {key}: {_display(selected.get(key))}" for key in keys if key in selected]


def _text(value: object) -> str:
    if value is None or value == "":
        return "n/a"
    return str(value)


def _display(value: object) -> str:
    if value is None or value == "":
        return "n/a"
    return str(value)


def _metric(value: object, suffix: str) -> str:
    number = _optional_float(value)
    if number is None:
        return "n/a"
    if suffix in {" tokens/sec", " ms", " W"}:
        return f"{number:,.2f}{suffix}"
    return f"{number:,.6f}{suffix}"


def _number_text(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text or "0"


def _seconds_to_ms(value: object) -> float | None:
    seconds = _optional_float(value)
    return seconds * 1000.0 if seconds is not None else None


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
