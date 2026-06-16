"""Prior providers and pruning policy for Managed Evaluation Mode."""

from __future__ import annotations

import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .aiconfig_parser import parse_aiconfigurator_best_configs
from .aiconfigurator_bridge import AIConfiguratorRun, run_aiconfigurator
from .evidence import EvidenceHitType, EvidenceLookupResult, launch_config_hash
from .schemas import Goal, PriorCandidate, PriorResult, PriorSource, ServeCandidate, ServingConfig, to_dict


class PriorProvider(Protocol):
    source: str

    def collect_priors(
        self,
        *,
        model: str,
        backend: str,
        goal: Goal,
        candidates: list[ServingConfig],
        out_dir: Path,
    ) -> PriorResult:
        ...


AIConfiguratorRunner = Callable[..., AIConfiguratorRun]


@dataclass(frozen=True)
class ManagedPriorPolicy:
    max_prior_candidates: int = 8
    preserve_backend_default: bool = True
    preserve_low_memory_candidate: bool = True
    preserve_diversity: bool = True
    preserve_evidence_near_hits: bool = True
    require_measurement_for_final_recommendation: bool = True


@dataclass(frozen=True)
class PriorPruningResult:
    candidates: list[ServingConfig]
    prior_results: list[PriorResult] = field(default_factory=list)
    prior_by_config_id: dict[str, PriorCandidate] = field(default_factory=dict)
    prior_sources_used: list[str] = field(default_factory=list)
    prior_candidate_count: int = 0
    candidates_pruned_by_prior: int = 0
    ai_configurator_available: bool = False
    ai_configurator_used: bool = False
    summary: dict[str, object] = field(default_factory=dict)


class AIConfiguratorPriorProvider:
    source = PriorSource.AICONFIGURATOR.value

    def __init__(
        self,
        *,
        runner: AIConfiguratorRunner = run_aiconfigurator,
        system: str = "local_gpu",
        total_gpus: int = 1,
        isl: int = 512,
        osl: int = 128,
        top_k: int = 8,
    ) -> None:
        self.runner = runner
        self.system = system
        self.total_gpus = total_gpus
        self.isl = isl
        self.osl = osl
        self.top_k = top_k

    def collect_priors(
        self,
        *,
        model: str,
        backend: str,
        goal: Goal,
        candidates: list[ServingConfig],
        out_dir: Path,
    ) -> PriorResult:
        del goal, candidates
        if self.runner is run_aiconfigurator and not _aiconfigurator_cli_available():
            return PriorResult(
                source=self.source,
                available=False,
                used=False,
                warnings=["AIConfigurator CLI is not installed."],
            )
        prior_dir = out_dir / "aiconfigurator_prior"
        try:
            run = self.runner(
                mode="default",
                model=model,
                system=self.system,
                backend=backend,
                output_dir=prior_dir,
                isl=self.isl,
                osl=self.osl,
                total_gpus=self.total_gpus,
            )
        except Exception as exc:
            return PriorResult(
                source=self.source,
                available=False,
                used=False,
                warnings=[f"AIConfigurator prior collection failed: {exc.__class__.__name__}: {exc}"],
            )
        metadata = {
            "command": run.command,
            "returncode": run.returncode,
            "output_path": run.output_path,
        }
        if run.returncode != 0:
            return PriorResult(
                source=self.source,
                available=True,
                used=False,
                warnings=[f"AIConfigurator exited with status {run.returncode}."],
                raw_metadata=metadata,
            )
        csv_path = _find_best_config_csv(prior_dir)
        if csv_path is None:
            return PriorResult(
                source=self.source,
                available=True,
                used=False,
                warnings=["AIConfigurator did not produce best_config_topn.csv."],
                raw_metadata=metadata,
            )
        try:
            parsed = parse_aiconfigurator_best_configs(str(csv_path), top_k=self.top_k)
        except Exception as exc:
            return PriorResult(
                source=self.source,
                available=True,
                used=False,
                warnings=[f"AIConfigurator prior parse failed: {exc.__class__.__name__}: {exc}"],
                raw_metadata={**metadata, "csv_path": str(csv_path)},
            )
        prior_candidates = [_prior_from_aic_candidate(candidate) for candidate in parsed]
        return PriorResult(
            source=self.source,
            available=True,
            used=bool(prior_candidates),
            candidates=prior_candidates,
            notes=["AIConfigurator estimates are prior-only and require measured validation."],
            raw_metadata={**metadata, "csv_path": str(csv_path)},
        )


def evidence_lookup_to_prior(config: ServingConfig, lookup: EvidenceLookupResult) -> PriorCandidate | None:
    if lookup.hit_type == EvidenceHitType.EXACT_STALE_HIT:
        measurement = lookup.measurement or {}
        return PriorCandidate(
            source=PriorSource.EVIDENCE_STALE_HIT.value,
            candidate_id=config.id,
            config_id=config.id,
            support_status="measured_stale",
            confidence=0.65,
            predicted_throughput_tokens_per_sec=_optional_float(measurement.get("throughput_tokens_per_sec")),
            predicted_latency_ms=_optional_float(measurement.get("p95_latency_ms")),
            notes=[lookup.reason, "Stale measured evidence is prior metadata only."],
            raw_prior_payload={"measurement_id": measurement.get("measurement_id")},
        )
    if lookup.hit_type == EvidenceHitType.NEAR_COMPATIBLE_HIT:
        near_match = lookup.near_matches[0] if lookup.near_matches else {}
        return PriorCandidate(
            source=PriorSource.EVIDENCE_NEAR_HIT.value,
            candidate_id=config.id,
            config_id=config.id,
            support_status="measured_near",
            confidence=0.5,
            predicted_throughput_tokens_per_sec=_optional_float(near_match.get("throughput_tokens_per_sec")),
            predicted_latency_ms=_optional_float(near_match.get("p95_latency_ms")),
            notes=[lookup.reason, "Near-compatible measured evidence is prior metadata only."],
            raw_prior_payload={"measurement_id": near_match.get("measurement_id")},
        )
    return None


def apply_managed_prior_policy(
    candidates: list[ServingConfig],
    *,
    prior_results: list[PriorResult],
    evidence_priors: list[PriorCandidate],
    exact_fresh_candidate_ids: set[str] | None = None,
    policy: ManagedPriorPolicy | None = None,
) -> PriorPruningResult:
    policy = policy or ManagedPriorPolicy()
    exact_fresh_candidate_ids = set(exact_fresh_candidate_ids or set())
    if not candidates:
        return PriorPruningResult(candidates=[], prior_results=prior_results, summary={"reason": "no_candidates"})

    prior_by_config_id = _prior_by_config_id(prior_results, evidence_priors)
    selected_ids: list[str] = []
    selected_set: set[str] = set()

    def add(config_id: str | None) -> None:
        if config_id and config_id not in selected_set and any(config.id == config_id for config in candidates):
            selected_set.add(config_id)
            selected_ids.append(config_id)

    for config in candidates:
        if config.id in exact_fresh_candidate_ids:
            add(config.id)

    baseline = _safe_baseline_candidate(candidates)
    if baseline is not None:
        add(baseline.id)

    if policy.preserve_backend_default:
        add(candidates[0].id)

    if policy.preserve_low_memory_candidate:
        low_memory = min(
            candidates,
            key=lambda config: (
                config.estimated_vram_mb if config.estimated_vram_mb is not None else 10**12,
                config.max_batch_size,
                config.id,
            ),
        )
        add(low_memory.id)

    if policy.preserve_evidence_near_hits:
        for prior in evidence_priors:
            add(prior.config_id or prior.candidate_id)

    matched_ai_priors = [
        prior
        for result in prior_results
        for prior in result.candidates
        if result.source == PriorSource.AICONFIGURATOR.value and (prior.config_id or prior.candidate_id) in {config.id for config in candidates}
    ]
    matched_ai_priors.sort(key=lambda prior: prior.confidence if prior.confidence is not None else 0.0, reverse=True)
    for prior in matched_ai_priors:
        add(prior.config_id or prior.candidate_id)

    if policy.preserve_diversity:
        seen_launch_hashes: set[str] = set()
        for config in candidates:
            launch_hash = launch_config_hash(config)
            if launch_hash in seen_launch_hashes:
                continue
            seen_launch_hashes.add(launch_hash)
            add(config.id)
            if len(selected_set) >= policy.max_prior_candidates and not exact_fresh_candidate_ids:
                break

    max_candidates = max(1, policy.max_prior_candidates)
    for config in candidates:
        if len([candidate_id for candidate_id in selected_ids if candidate_id not in exact_fresh_candidate_ids]) >= max_candidates:
            break
        add(config.id)

    if not selected_ids:
        add(candidates[0].id)

    candidates_by_id = {config.id: config for config in candidates}
    selected_candidates = [_attach_prior_metadata(candidates_by_id[config_id], prior_by_config_id.get(config_id)) for config_id in selected_ids]
    source_set = {prior.source for prior in evidence_priors}
    for result in prior_results:
        if result.used:
            source_set.add(result.source)
    ai_result = next((result for result in prior_results if result.source == PriorSource.AICONFIGURATOR.value), None)
    return PriorPruningResult(
        candidates=selected_candidates,
        prior_results=prior_results,
        prior_by_config_id=prior_by_config_id,
        prior_sources_used=sorted(source_set),
        prior_candidate_count=sum(len(result.candidates) for result in prior_results) + len(evidence_priors),
        candidates_pruned_by_prior=max(0, len(candidates) - len(selected_candidates)),
        ai_configurator_available=bool(ai_result and ai_result.available),
        ai_configurator_used=bool(ai_result and ai_result.used),
        summary={
            "input_candidate_count": len(candidates),
            "selected_candidate_count": len(selected_candidates),
            "pruned_candidate_ids": [config.id for config in candidates if config.id not in selected_set],
            "require_measurement_for_final_recommendation": policy.require_measurement_for_final_recommendation,
        },
    )


def attach_prior_metadata(config: ServingConfig, prior: PriorCandidate | None) -> ServingConfig:
    return _attach_prior_metadata(config, prior)


def _prior_from_aic_candidate(candidate: ServeCandidate) -> PriorCandidate:
    confidence = max(0.2, round(1.0 - 0.08 * max(0, candidate.rank - 1), 3))
    notes = ["AIConfigurator estimate is prior-only and requires measured validation."]
    if candidate.predicted_power_w is not None:
        notes.append("Predicted power is retained only in the raw prior payload.")
    return PriorCandidate(
        source=PriorSource.AICONFIGURATOR.value,
        candidate_id=candidate.candidate_id,
        config_id=candidate.candidate_id,
        support_status="estimated",
        confidence=confidence,
        predicted_throughput_tokens_per_sec=candidate.predicted_tokens_s,
        predicted_ttft_ms=candidate.predicted_ttft_ms,
        predicted_tpot_ms=candidate.predicted_tpot_ms,
        predicted_latency_ms=candidate.predicted_request_latency_ms,
        predicted_memory_gb=candidate.predicted_memory_gb,
        notes=notes,
        raw_prior_payload=to_dict(candidate),
    )


def _prior_by_config_id(prior_results: list[PriorResult], evidence_priors: list[PriorCandidate]) -> dict[str, PriorCandidate]:
    prior_by_id: dict[str, PriorCandidate] = {}
    for prior in evidence_priors:
        config_id = prior.config_id or prior.candidate_id
        prior_by_id[config_id] = prior
    for result in prior_results:
        for prior in result.candidates:
            config_id = prior.config_id or prior.candidate_id
            prior_by_id.setdefault(config_id, prior)
    return prior_by_id


def _attach_prior_metadata(config: ServingConfig, prior: PriorCandidate | None) -> ServingConfig:
    if prior is None:
        return config
    extra = dict(config.extra or {})
    extra["prior_source"] = prior.source
    extra["prior_confidence"] = prior.confidence
    extra["prior_notes"] = list(prior.notes)
    return ServingConfig(
        id=config.id,
        backend=config.backend,
        model_id=config.model_id,
        dtype=config.dtype,
        quantization=config.quantization,
        max_batch_size=config.max_batch_size,
        max_context_tokens=config.max_context_tokens,
        kv_cache_policy=config.kv_cache_policy,
        scheduler=config.scheduler,
        tensor_parallelism=config.tensor_parallelism,
        gpu_memory_utilization=config.gpu_memory_utilization,
        block_size=config.block_size,
        kv_cache_dtype=config.kv_cache_dtype,
        enforce_eager=config.enforce_eager,
        max_num_batched_tokens=config.max_num_batched_tokens,
        enable_chunked_prefill=config.enable_chunked_prefill,
        max_cudagraph_capture_size=config.max_cudagraph_capture_size,
        enable_prefix_caching=config.enable_prefix_caching,
        power_limit_watts=config.power_limit_watts,
        estimated_vram_mb=config.estimated_vram_mb,
        notes=list(config.notes),
        extra=extra,
    )


def _safe_baseline_candidate(candidates: list[ServingConfig]) -> ServingConfig | None:
    for config in candidates:
        extra = config.extra or {}
        if config.quantization == "none" and extra.get("baseline") is True and extra.get("model_native") is True:
            return config
    return None


def _find_best_config_csv(output_dir: Path) -> Path | None:
    matches = sorted(output_dir.rglob("best_config_topn.csv"))
    return matches[0] if matches else None


def _aiconfigurator_cli_available() -> bool:
    if shutil.which("aiconfigurator"):
        return True
    return Path(sys.executable).with_name("aiconfigurator").exists()


def _optional_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
