"""Workload profile presets, manifests, and SLO helpers."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

from .schemas import RecommendationInput, WorkloadProfile, to_dict

SLO_CONSTRAINT_FIELDS = {
    "ttft_ms",
    "tpot_ms",
    "p95_latency_ms",
    "min_throughput_tokens_per_sec",
    "max_failed_request_rate",
}

WORKLOAD_PROFILE_PRESETS: dict[str, WorkloadProfile] = {
    "default": WorkloadProfile(),
    "short": WorkloadProfile(
        profile_name="short",
        input_tokens=128,
        output_tokens=64,
        max_new_tokens=64,
        concurrency=4,
        num_requests=64,
        dataset="synthetic-short",
        token_distribution={
            "input_tokens": {"p50": 128, "p95": 256},
            "output_tokens": {"p50": 64, "p95": 128},
        },
        notes=["Short synthetic prompts for quick managed evaluation."],
    ),
    "medium": WorkloadProfile(
        profile_name="medium",
        input_tokens=1024,
        output_tokens=256,
        max_new_tokens=256,
        concurrency=8,
        num_requests=96,
        dataset="synthetic-medium",
        token_distribution={
            "input_tokens": {"p50": 1024, "p95": 2048},
            "output_tokens": {"p50": 256, "p95": 512},
        },
        notes=["Medium synthetic prompt and decode mix."],
    ),
    "long": WorkloadProfile(
        profile_name="long",
        input_tokens=4096,
        output_tokens=512,
        max_new_tokens=512,
        concurrency=4,
        num_requests=48,
        dataset="synthetic-long",
        token_distribution={
            "input_tokens": {"p50": 4096, "p95": 8192},
            "output_tokens": {"p50": 512, "p95": 1024},
        },
        notes=["Long context synthetic workload."],
    ),
    "long-prefill": WorkloadProfile(
        profile_name="long-prefill",
        input_tokens=4096,
        output_tokens=128,
        max_new_tokens=128,
        concurrency=4,
        num_requests=48,
        dataset="synthetic-long-prefill",
        token_distribution={
            "input_tokens": {"p50": 4096, "p95": 8192},
            "output_tokens": {"p50": 128, "p95": 256},
        },
        notes=["Long prompt and short output synthetic workload."],
    ),
    "decode-heavy": WorkloadProfile(
        profile_name="decode-heavy",
        input_tokens=512,
        output_tokens=1024,
        max_new_tokens=1024,
        concurrency=8,
        num_requests=64,
        dataset="synthetic-decode-heavy",
        token_distribution={
            "input_tokens": {"p50": 512, "p95": 1024},
            "output_tokens": {"p50": 1024, "p95": 1536},
        },
        notes=["Decode heavy synthetic workload."],
    ),
    "repeated-prefix": WorkloadProfile(
        profile_name="repeated-prefix",
        input_tokens=1024,
        output_tokens=128,
        max_new_tokens=128,
        concurrency=8,
        num_requests=96,
        dataset="synthetic-repeated-prefix",
        token_distribution={
            "input_tokens": {"p50": 1024, "p95": 2048},
            "output_tokens": {"p50": 128, "p95": 256},
            "repeated_prefix_ratio": 0.75,
        },
        prefix_reuse_expected=True,
        repeated_prefix_ratio=0.75,
        notes=["Repeated prefix synthetic workload for prefix cache behavior."],
    ),
    "code-generation": WorkloadProfile(
        profile_name="code-generation",
        input_tokens=1024,
        output_tokens=512,
        max_new_tokens=512,
        concurrency=8,
        num_requests=96,
        dataset="synthetic-code-generation",
        token_distribution={
            "input_tokens": {"p50": 1024, "p95": 2048},
            "output_tokens": {"p50": 512, "p95": 1024},
        },
        notes=["Code generation style synthetic prompts."],
    ),
    "mixed": WorkloadProfile(
        profile_name="mixed",
        input_tokens=768,
        output_tokens=256,
        max_new_tokens=256,
        concurrency=8,
        num_requests=128,
        dataset="synthetic-mixed",
        token_distribution={
            "input_tokens": {"p50": 768, "p95": 4096},
            "output_tokens": {"p50": 256, "p95": 1024},
            "mix": {"short": 0.35, "medium": 0.45, "long": 0.20},
        },
        notes=["Mixed synthetic short, medium, and long workload."],
    ),
}


def workload_profile_choices() -> list[str]:
    return sorted(WORKLOAD_PROFILE_PRESETS)


def load_workload_profile(
    *,
    profile_name: str | None = None,
    manifest_path: Path | None = None,
    slo_constraints: dict[str, Any] | None = None,
) -> WorkloadProfile:
    profile = WORKLOAD_PROFILE_PRESETS.get(profile_name or "default")
    if profile is None:
        raise ValueError(f"Unsupported workload profile: {profile_name}")
    if manifest_path is not None:
        profile = _profile_from_manifest(manifest_path)
    constraints = _clean_slo_constraints(slo_constraints or {})
    if constraints:
        profile = replace(profile, slo_constraints={**profile.slo_constraints, **constraints})
    return profile


def workload_profile_to_payload(profile: WorkloadProfile | dict[str, Any] | None) -> dict[str, Any]:
    if profile is None:
        return {}
    payload = to_dict(profile)
    if isinstance(payload, dict):
        return {key: value for key, value in payload.items() if value not in (None, {}, [])}
    return {}


def workload_profile_summary(profile: WorkloadProfile | dict[str, Any] | None) -> dict[str, Any]:
    payload = workload_profile_to_payload(profile)
    return {
        "profile_name": payload.get("profile_name", "default"),
        "dataset": payload.get("dataset"),
        "token_distribution": payload.get("token_distribution", {}),
        "slo_constraints": payload.get("slo_constraints", {}),
    }


def slo_disqualifiers(item: RecommendationInput) -> list[str]:
    constraints = _candidate_slo_constraints(item)
    if not constraints:
        return []
    disqualifiers: list[str] = []
    p95_limit = _optional_float(constraints.get("p95_latency_ms"))
    p95_latency_s = _optional_float(item.measured_metrics.get("p95_latency_s"))
    if p95_limit is not None:
        if p95_latency_s is None:
            disqualifiers.append("slo_p95_latency_ms_missing")
        elif p95_latency_s * 1000.0 > p95_limit:
            disqualifiers.append("slo_p95_latency_ms_exceeded")
    throughput_floor = _optional_float(constraints.get("min_throughput_tokens_per_sec"))
    throughput = _optional_float(item.measured_metrics.get("total_tokens_s"))
    if throughput_floor is not None:
        if throughput is None:
            disqualifiers.append("slo_min_throughput_tokens_per_sec_missing")
        elif throughput < throughput_floor:
            disqualifiers.append("slo_min_throughput_tokens_per_sec_not_met")
    max_failed_rate = _optional_float(constraints.get("max_failed_request_rate"))
    if max_failed_rate is not None:
        total = _optional_int(item.measured_metrics.get("total_requests"))
        failed = _optional_int(item.measured_metrics.get("failed_requests"))
        if not total or failed is None:
            disqualifiers.append("slo_max_failed_request_rate_missing")
        elif failed / total > max_failed_rate:
            disqualifiers.append("slo_max_failed_request_rate_exceeded")
    ttft_limit = _optional_float(constraints.get("ttft_ms"))
    ttft = _first_float(item.measured_metrics, ("ttft_ms", "time_to_first_token_ms"))
    if ttft_limit is not None:
        if ttft is None:
            disqualifiers.append("slo_ttft_ms_missing")
        elif ttft > ttft_limit:
            disqualifiers.append("slo_ttft_ms_exceeded")
    tpot_limit = _optional_float(constraints.get("tpot_ms"))
    tpot = _first_float(item.measured_metrics, ("tpot_ms", "time_per_output_token_ms"))
    if tpot_limit is not None:
        if tpot is None:
            disqualifiers.append("slo_tpot_ms_missing")
        elif tpot > tpot_limit:
            disqualifiers.append("slo_tpot_ms_exceeded")
    return disqualifiers


def slo_note(profile: WorkloadProfile | dict[str, Any] | None) -> str | None:
    constraints = workload_profile_to_payload(profile).get("slo_constraints")
    if not isinstance(constraints, dict) or not constraints:
        return None
    fields = ", ".join(sorted(str(field) for field in constraints))
    return f"SLO constraints are eligibility guards for recommendation scoring: {fields}."


def _profile_from_manifest(path: Path) -> WorkloadProfile:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Workload manifest must contain a JSON object.")
    base_name = str(payload.get("profile_name") or "default")
    base = WORKLOAD_PROFILE_PRESETS.get(base_name, WorkloadProfile(profile_name=base_name))
    values = to_dict(base)
    values.update(payload)
    values["slo_constraints"] = _clean_slo_constraints(values.get("slo_constraints", {}))
    return WorkloadProfile(**{key: value for key, value in values.items() if key in WorkloadProfile.__dataclass_fields__})


def _clean_slo_constraints(payload: dict[str, Any]) -> dict[str, Any]:
    constraints = {}
    for key, value in payload.items():
        if key not in SLO_CONSTRAINT_FIELDS:
            raise ValueError(f"Unsupported SLO constraint: {key}")
        if value is None:
            continue
        number = _optional_float(value)
        if number is None or not math.isfinite(number):
            raise ValueError(f"SLO constraint {key} must be a finite number.")
        if key == "max_failed_request_rate":
            if not 0.0 <= number <= 1.0:
                raise ValueError("SLO constraint max_failed_request_rate must be between 0 and 1.")
        elif number < 0:
            raise ValueError(f"SLO constraint {key} must be nonnegative.")
        constraints[key] = number
    return constraints


def _candidate_slo_constraints(item: RecommendationInput) -> dict[str, Any]:
    raw = item.candidate.raw or {}
    direct = raw.get("slo_constraints")
    if isinstance(direct, dict):
        return direct
    profile = raw.get("workload_profile")
    if isinstance(profile, dict) and isinstance(profile.get("slo_constraints"), dict):
        return dict(profile["slo_constraints"])
    return {}


def _first_float(payload: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = _optional_float(payload.get(key))
        if value is not None:
            return value
    return None


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
