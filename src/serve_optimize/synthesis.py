"""Bounded AIConfigurator candidate synthesis for Managed Evaluation Mode."""

from __future__ import annotations

import hashlib
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol

from .aiconfig_parser import parse_aiconfigurator_best_configs, parse_aiconfigurator_text_candidates
from .aiconfigurator_bridge import AIConfiguratorRun, run_aiconfigurator
from .backends.vllm import VLLMArgumentCapabilities
from .schemas import Goal, GpuDevice, HardwareSnapshot, ModelCapabilityMetadata, ServeCandidate, ServingConfig, to_dict
from .validation import normalize_quantization

AIConfiguratorRunner = Callable[..., AIConfiguratorRun]
SYNTHESIS_SOURCE = "ai_configurator_synthesized"
SYNTHESIS_SCHEMA_VERSION = "candidate-synthesis/v1"
AICONFIGURATOR_SYSTEM_MAP = (
    ("gb300", "gb300"),
    ("gb200", "gb200"),
    ("b300", "b300_sxm"),
    ("b200", "b200_sxm"),
    ("h200", "h200_sxm"),
    ("h100 pcie", "h100_pcie"),
    ("h100", "h100_sxm"),
    ("a100 pcie", "a100_pcie"),
    ("a100", "a100_sxm"),
    ("l40s", "l40s"),
    ("l4", "l4"),
    ("a30", "a30"),
)


class CandidateSynthesisProvider(Protocol):
    source: str

    def synthesize(
        self,
        *,
        context: CandidateSynthesisContext,
        out_dir: Path,
    ) -> CandidateSynthesisResult:
        ...


@dataclass(frozen=True)
class CandidateSynthesisContext:
    hardware: HardwareSnapshot
    backend: str
    backend_argument_capabilities: VLLMArgumentCapabilities | None
    model: str
    model_metadata: ModelCapabilityMetadata
    goal: Goal
    evidence_summary: dict[str, object]
    safe_baseline: ServingConfig | None
    existing_candidates: list[ServingConfig]
    workload_profile: dict[str, object] = field(default_factory=dict)
    max_candidates: int = 3


@dataclass(frozen=True)
class CandidateSynthesisResult:
    source: str
    available: bool
    used: bool
    candidates: list[ServingConfig] = field(default_factory=list)
    rationale: str | None = None
    confidence: float | None = None
    constraints_used: dict[str, object] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    skipped_reason: str | None = None
    raw_metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class AICSystemResolution:
    system_key: str | None
    confidence: float
    source: str
    reason: str
    parent_gpu_model: str | None = None
    is_mig: bool = False
    warnings: list[str] = field(default_factory=list)


class NoopCandidateSynthesisProvider:
    source = "unavailable"

    def synthesize(self, *, context: CandidateSynthesisContext, out_dir: Path) -> CandidateSynthesisResult:
        del context, out_dir
        return CandidateSynthesisResult(
            source=self.source,
            available=False,
            used=False,
            skipped_reason="Candidate synthesis provider is unavailable.",
        )


class AIConfiguratorSynthesisProvider:
    source = SYNTHESIS_SOURCE

    def __init__(
        self,
        *,
        runner: AIConfiguratorRunner = run_aiconfigurator,
        system: str | None = None,
        total_gpus: int = 1,
        isl: int | None = None,
        osl: int | None = None,
        top_k: int = 5,
        max_candidates: int = 3,
    ) -> None:
        self.runner = runner
        self.system = system
        self.total_gpus = total_gpus
        self.isl = isl
        self.osl = osl
        self.top_k = top_k
        self.max_candidates = max(0, min(5, max_candidates))

    def synthesize(self, *, context: CandidateSynthesisContext, out_dir: Path) -> CandidateSynthesisResult:
        if context.backend != "vllm":
            return CandidateSynthesisResult(
                source=self.source,
                available=False,
                used=False,
                skipped_reason="AIConfigurator synthesis currently has only vLLM candidate mapping.",
            )
        if context.safe_baseline is None:
            return CandidateSynthesisResult(
                source=self.source,
                available=False,
                used=False,
                skipped_reason="Safe baseline is unavailable.",
            )
        if self.max_candidates <= 0 or context.max_candidates <= 0:
            return CandidateSynthesisResult(
                source=self.source,
                available=True,
                used=False,
                skipped_reason="Synthesis budget is zero.",
            )
        if self.runner is run_aiconfigurator and not _aiconfigurator_cli_available():
            return CandidateSynthesisResult(
                source=self.source,
                available=False,
                used=False,
                warnings=["AIConfigurator CLI is not installed."],
                skipped_reason="AIConfigurator CLI is not installed.",
            )

        system_resolution = resolve_aiconfigurator_system_key(context.hardware)
        if self.system:
            system_resolution = AICSystemResolution(
                system_key=self.system,
                confidence=1.0,
                source="provider_override",
                reason="AIConfigurator system key was provided explicitly.",
                parent_gpu_model=system_resolution.parent_gpu_model,
                is_mig=system_resolution.is_mig,
                warnings=system_resolution.warnings,
            )
        if not system_resolution.system_key:
            return CandidateSynthesisResult(
                source=self.source,
                available=False,
                used=False,
                warnings=system_resolution.warnings,
                skipped_reason="unable_to_resolve_aiconfigurator_system",
                raw_metadata={"system_resolution": to_dict(system_resolution)},
            )

        synthesis_dir = out_dir / "aiconfigurator_synthesis"
        isl, osl = _resolve_workload_tokens(context, self.isl, self.osl)
        try:
            run = self.runner(
                mode="default",
                model=context.model,
                system=system_resolution.system_key,
                backend=context.backend,
                output_dir=synthesis_dir,
                isl=isl,
                osl=osl,
                total_gpus=self.total_gpus,
            )
        except Exception as exc:
            return CandidateSynthesisResult(
                source=self.source,
                available=False,
                used=False,
                warnings=[f"AIConfigurator synthesis failed: {exc.__class__.__name__}: {exc}"],
                skipped_reason="AIConfigurator synthesis failed.",
                raw_metadata={
                    "system_resolution": to_dict(system_resolution),
                    "requested_workload": {"isl": isl, "osl": osl, "total_gpus": self.total_gpus},
                },
            )

        metadata = {
            "command": run.command,
            "returncode": run.returncode,
            "output_path": run.output_path,
            "system_resolution": to_dict(system_resolution),
            "requested_workload": {"isl": isl, "osl": osl, "total_gpus": self.total_gpus},
        }
        if run.returncode != 0:
            return CandidateSynthesisResult(
                source=self.source,
                available=True,
                used=False,
                warnings=[f"AIConfigurator exited with status {run.returncode} during synthesis."],
                skipped_reason="AIConfigurator exited unsuccessfully.",
                raw_metadata=metadata,
            )

        csv_path = _find_best_config_csv(synthesis_dir)
        try:
            if csv_path is not None:
                parsed = parse_aiconfigurator_best_configs(str(csv_path), top_k=self.top_k)
                parser_source = str(csv_path)
            else:
                output_text = _run_output_text(run)
                parsed = parse_aiconfigurator_text_candidates(output_text, source=run.output_path or "aiconfigurator-output", top_k=self.top_k)
                parser_source = run.output_path or "stdout_stderr"
        except Exception as exc:
            return CandidateSynthesisResult(
                source=self.source,
                available=True,
                used=False,
                warnings=[f"AIConfigurator synthesis parse failed: {exc.__class__.__name__}: {exc}"],
                skipped_reason="AIConfigurator synthesis output could not be parsed.",
                raw_metadata={**metadata, "csv_path": str(csv_path) if csv_path else None},
            )
        if not parsed:
            return CandidateSynthesisResult(
                source=self.source,
                available=True,
                used=False,
                warnings=["AIConfigurator synthesis produced no parseable candidates."],
                skipped_reason="AIConfigurator synthesis output was missing.",
                raw_metadata={**metadata, "csv_path": str(csv_path) if csv_path else None},
            )

        max_count = max(0, min(self.max_candidates, context.max_candidates, 5))
        candidates = [
            config
            for candidate in parsed
            if (config := _config_from_aic_candidate(candidate, context=context, system_resolution=system_resolution)) is not None
        ][:max_count]
        return CandidateSynthesisResult(
            source=self.source,
            available=True,
            used=bool(candidates),
            candidates=candidates,
            rationale="AIConfigurator suggested operating points were converted into bounded managed candidates.",
            confidence=_aggregate_confidence(candidates),
            constraints_used={
                "max_synthesized_candidates": max_count,
                "backend": context.backend,
                "aiconfigurator_system_key": system_resolution.system_key,
                "aiconfigurator_system_confidence": system_resolution.confidence,
                "aiconfigurator_system_source": system_resolution.source,
                "safe_baseline_first": True,
                "evidence_scope": _evidence_scope(context.evidence_summary),
                "allowed_fields": [
                    "dtype",
                    "quantization",
                    "max_model_len",
                    "gpu_memory_utilization",
                    "max_num_seqs",
                    "tensor_parallel_size",
                    "block_size",
                    "kv_cache_dtype",
                    "enforce_eager",
                    "max_num_batched_tokens",
                    "enable_chunked_prefill",
                    "max_cudagraph_capture_size",
                    "enable_prefix_caching",
                    "workload_concurrency",
                ],
            },
            raw_metadata={
                **metadata,
                "csv_path": str(csv_path) if csv_path else None,
                "parser_source": parser_source,
                "parsed_candidate_count": len(parsed),
                "injected_candidate_count": len(candidates),
            },
        )


def synthesis_result_to_artifact(result: CandidateSynthesisResult) -> dict[str, object]:
    return {
        "source": result.source,
        "available": result.available,
        "used": result.used,
        "candidate_count": len(result.candidates),
        "candidate_ids": [config.id for config in result.candidates],
        "rationale": result.rationale,
        "confidence": result.confidence,
        "constraints_used": result.constraints_used,
        "warnings": result.warnings,
        "skipped_reason": result.skipped_reason,
        "raw_metadata": result.raw_metadata,
    }


def resolve_aiconfigurator_system_key(hardware: HardwareSnapshot | dict[str, object] | None) -> AICSystemResolution:
    gpus = _hardware_gpus(hardware)
    if not gpus:
        return AICSystemResolution(
            system_key=None,
            confidence=0.0,
            source="hardware",
            reason="No GPU metadata was available.",
            warnings=["Unable to resolve AIConfigurator system key because no GPU was detected."],
        )
    gpu = max(gpus, key=lambda item: item.total_memory_mb or 0)
    model_text = _gpu_parent_model_text(gpu)
    normalized = _normalize_gpu_text(model_text)
    for marker, system_key in AICONFIGURATOR_SYSTEM_MAP:
        if marker in normalized:
            return AICSystemResolution(
                system_key=system_key,
                confidence=0.9 if gpu.is_mig else 0.95,
                source="hardware_gpu_name",
                reason=f"Matched GPU model '{model_text}' to AIConfigurator system '{system_key}'.",
                parent_gpu_model=model_text,
                is_mig=gpu.is_mig,
                warnings=[],
            )
    return AICSystemResolution(
        system_key=None,
        confidence=0.0,
        source="hardware_gpu_name",
        reason=f"GPU model '{model_text}' is not mapped to an AIConfigurator system key.",
        parent_gpu_model=model_text,
        is_mig=gpu.is_mig,
        warnings=[f"Unable to resolve AIConfigurator system key for GPU model '{model_text}'."],
    )


def _config_from_aic_candidate(
    candidate: ServeCandidate,
    *,
    context: CandidateSynthesisContext,
    system_resolution: AICSystemResolution | None = None,
) -> ServingConfig | None:
    baseline = context.safe_baseline
    if baseline is None:
        return None
    if candidate.backend and candidate.backend.lower() not in {context.backend.lower(), "vllm"}:
        return None
    existing = context.existing_candidates or [baseline]
    max_existing_context = max(config.max_context_tokens for config in existing)
    max_existing_batch = max(config.max_batch_size for config in existing)
    max_existing_concurrency = max(_positive_int((config.extra or {}).get("workload_concurrency"), default=config.max_batch_size) for config in existing)
    requested_context = _positive_int(candidate.isl, default=baseline.max_context_tokens) + _positive_int(candidate.osl, default=0)
    max_context_tokens = _nearest_existing_context(max(baseline.max_context_tokens, requested_context), existing, max_existing_context)
    max_batch_size = max(1, min(max_existing_batch, _positive_int(candidate.batch_size or candidate.concurrency, default=baseline.max_batch_size)))
    workload_concurrency = max(1, min(max_existing_concurrency, _positive_int(candidate.concurrency or candidate.batch_size, default=max_batch_size)))
    neighbor = _nearest_neighbor(existing, max_context_tokens=max_context_tokens, max_batch_size=max_batch_size)
    engine_options = _supported_engine_options(neighbor, context.backend_argument_capabilities)
    confidence = max(0.2, round(1.0 - 0.08 * max(0, candidate.rank - 1), 3))
    extra = dict(baseline.extra or {})
    extra.update(
        {
            "candidate_source": SYNTHESIS_SOURCE,
            "model_native": normalize_quantization(baseline.quantization) == "none",
            "workload_concurrency": workload_concurrency,
            "max_new_tokens": _positive_int(candidate.osl, default=_positive_int(extra.get("max_new_tokens"), default=128)),
            "input_length": _positive_int(candidate.isl, default=0) or None,
            "output_length": _positive_int(candidate.osl, default=0) or None,
            "request_rate": candidate.request_rate,
            "synthesis_rationale": "AIConfigurator predicted this bounded managed operating point.",
            "synthesis_confidence": confidence,
            "aiconfigurator_system_key": system_resolution.system_key if system_resolution else candidate.system,
            "aiconfigurator_rank": candidate.rank,
            "aiconfigurator_predicted_metrics": _predicted_metrics(candidate),
            "synthesis_constraints": {
                "max_existing_context": max_existing_context,
                "max_existing_batch": max_existing_batch,
                "max_existing_concurrency": max_existing_concurrency,
                "no_new_engine_fields": True,
                "evidence_scope": _evidence_scope(context.evidence_summary),
                "aiconfigurator_system_resolution": to_dict(system_resolution) if system_resolution else {},
            },
            "synthesis_status": "proposed",
            "raw_aiconfigurator_candidate": to_dict(candidate),
        }
    )
    return replace(
        baseline,
        id=_synthesized_config_id(context, candidate, max_context_tokens, max_batch_size, workload_concurrency, engine_options),
        max_context_tokens=max_context_tokens,
        max_batch_size=max_batch_size,
        tensor_parallelism=1,
        gpu_memory_utilization=baseline.gpu_memory_utilization,
        block_size=_optional_int(engine_options.get("block_size")),
        kv_cache_dtype=_optional_str(engine_options.get("kv_cache_dtype")),
        enforce_eager=_optional_bool(engine_options.get("enforce_eager")),
        max_num_batched_tokens=_optional_int(engine_options.get("max_num_batched_tokens")),
        enable_chunked_prefill=_optional_bool(engine_options.get("enable_chunked_prefill")),
        max_cudagraph_capture_size=_optional_int(engine_options.get("max_cudagraph_capture_size")),
        enable_prefix_caching=_optional_bool(engine_options.get("enable_prefix_caching")),
        notes=["AIConfigurator synthesized candidate. Requires validation, canonicalization, and measurement or exact evidence."],
        extra=extra,
    )


def _supported_engine_options(config: ServingConfig, capabilities: VLLMArgumentCapabilities | None) -> dict[str, object]:
    options = {
        "block_size": config.block_size,
        "kv_cache_dtype": config.kv_cache_dtype,
        "enforce_eager": config.enforce_eager,
        "max_num_batched_tokens": config.max_num_batched_tokens,
        "enable_chunked_prefill": config.enable_chunked_prefill,
        "max_cudagraph_capture_size": config.max_cudagraph_capture_size,
        "enable_prefix_caching": config.enable_prefix_caching,
    }
    if capabilities is None or capabilities.detection_status != "success":
        return {key: value for key, value in options.items() if value is not None}
    filtered: dict[str, object] = {}
    if config.block_size is not None and capabilities.supports("--block-size"):
        filtered["block_size"] = config.block_size
    if config.kv_cache_dtype is not None and capabilities.supports("--kv-cache-dtype"):
        choices = capabilities.choices_for("--kv-cache-dtype")
        if not choices or config.kv_cache_dtype in choices:
            filtered["kv_cache_dtype"] = config.kv_cache_dtype
    if config.enforce_eager is True and capabilities.supports("--enforce-eager"):
        filtered["enforce_eager"] = True
    if config.max_num_batched_tokens is not None and capabilities.supports("--max-num-batched-tokens"):
        filtered["max_num_batched_tokens"] = config.max_num_batched_tokens
    if config.enable_chunked_prefill is True and capabilities.supports("--enable-chunked-prefill"):
        filtered["enable_chunked_prefill"] = True
    if config.enable_chunked_prefill is False and capabilities.supports("--no-enable-chunked-prefill"):
        filtered["enable_chunked_prefill"] = False
    if config.max_cudagraph_capture_size is not None and capabilities.cudagraph_capture_flag() is not None:
        filtered["max_cudagraph_capture_size"] = config.max_cudagraph_capture_size
    return filtered


def _predicted_metrics(candidate: ServeCandidate) -> dict[str, object]:
    return {
        key: value
        for key, value in {
            "ttft_ms": candidate.predicted_ttft_ms,
            "tpot_ms": candidate.predicted_tpot_ms,
            "request_latency_ms": candidate.predicted_request_latency_ms,
            "tokens_per_sec": candidate.predicted_tokens_s,
            "tokens_per_sec_per_gpu": candidate.predicted_tokens_s_per_gpu,
            "tokens_per_sec_per_user": candidate.predicted_tokens_s_per_user,
            "memory_gb": candidate.predicted_memory_gb,
            "power_w": candidate.predicted_power_w,
        }.items()
        if value is not None
    }


def _resolve_workload_tokens(context: CandidateSynthesisContext, isl: int | None, osl: int | None) -> tuple[int, int]:
    profile = dict(context.workload_profile or {})
    resolved_isl = _positive_int(isl, default=0) or _positive_int(profile.get("input_tokens") or profile.get("max_existing_context"), default=512)
    resolved_osl = _positive_int(osl, default=0) or _positive_int(profile.get("output_tokens") or profile.get("max_new_tokens"), default=128)
    return resolved_isl, resolved_osl


def _run_output_text(run: AIConfiguratorRun) -> str:
    chunks = [run.stdout, run.stderr]
    if run.output_path:
        try:
            chunks.append(Path(run.output_path).read_text(encoding="utf-8"))
        except OSError:
            pass
    return "\n".join(chunk for chunk in chunks if chunk)


def _hardware_gpus(hardware: HardwareSnapshot | dict[str, object] | None) -> list[GpuDevice]:
    if isinstance(hardware, HardwareSnapshot):
        return list(hardware.gpus)
    if not isinstance(hardware, dict):
        return []
    gpus = []
    for index, row in enumerate(hardware.get("gpus", []) if isinstance(hardware.get("gpus"), list) else []):
        if isinstance(row, dict):
            gpus.append(
                GpuDevice(
                    index=int(row.get("index") or index),
                    name=str(row.get("name") or ""),
                    uuid=str(row.get("uuid")) if row.get("uuid") is not None else None,
                    total_memory_mb=_optional_int(row.get("total_memory_mb")),
                    mig_profile=str(row.get("mig_profile")) if row.get("mig_profile") is not None else None,
                    mig_mode=str(row.get("mig_mode")) if row.get("mig_mode") is not None else None,
                    raw=dict(row),
                )
            )
    return gpus


def _gpu_parent_model_text(gpu: GpuDevice) -> str:
    values = [gpu.name, gpu.raw.get("parent_name") if isinstance(gpu.raw, dict) else None, gpu.raw.get("gpu_name") if isinstance(gpu.raw, dict) else None]
    return next((str(value) for value in values if value), gpu.name)


def _normalize_gpu_text(text: str) -> str:
    return " ".join(text.lower().replace("_", " ").replace("-", " ").split())


def _nearest_neighbor(candidates: list[ServingConfig], *, max_context_tokens: int, max_batch_size: int) -> ServingConfig:
    return min(
        candidates,
        key=lambda config: (
            abs(config.max_context_tokens - max_context_tokens),
            abs(config.max_batch_size - max_batch_size),
            config.id,
        ),
    )


def _nearest_existing_context(requested: int, candidates: list[ServingConfig], fallback: int) -> int:
    contexts = sorted({config.max_context_tokens for config in candidates if config.max_context_tokens > 0})
    if not contexts:
        return fallback
    eligible = [value for value in contexts if value >= requested]
    return eligible[0] if eligible else contexts[-1]


def _aggregate_confidence(candidates: list[ServingConfig]) -> float | None:
    values = []
    for config in candidates:
        value = (config.extra or {}).get("synthesis_confidence")
        if isinstance(value, (int, float)):
            values.append(float(value))
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def _synthesized_config_id(
    context: CandidateSynthesisContext,
    candidate: ServeCandidate,
    max_context_tokens: int,
    max_batch_size: int,
    workload_concurrency: int,
    engine_options: dict[str, object],
) -> str:
    payload = {
        "source": SYNTHESIS_SOURCE,
        "model": context.model,
        "rank": candidate.rank,
        "candidate": to_dict(candidate),
        "max_context_tokens": max_context_tokens,
        "max_batch_size": max_batch_size,
        "workload_concurrency": workload_concurrency,
        "engine_options": engine_options,
    }
    digest = hashlib.sha1(repr(sorted(payload.items())).encode("utf-8"), usedforsecurity=False).hexdigest()[:10]
    return f"cfg-aic-synth-{digest}"


def _evidence_scope(evidence_summary: dict[str, object]) -> str:
    exact = evidence_summary.get("exact_fresh_candidate_ids")
    if isinstance(exact, list) and exact:
        return "exact_fresh_existing_candidates"
    priors = evidence_summary.get("evidence_prior_count")
    if isinstance(priors, int) and priors > 0:
        return "near_or_stale_existing_evidence"
    return "missing_or_unavailable_evidence"


def _find_best_config_csv(output_dir: Path) -> Path | None:
    matches = sorted(output_dir.rglob("best_config_topn.csv"))
    return matches[0] if matches else None


def _aiconfigurator_cli_available() -> bool:
    if shutil.which("aiconfigurator"):
        return True
    return Path(sys.executable).with_name("aiconfigurator").exists()


def _positive_int(value: object, *, default: int) -> int:
    try:
        parsed = int(value) if value is not None else default
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None
