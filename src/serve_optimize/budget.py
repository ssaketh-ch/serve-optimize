"""Budget-aware staged evaluation policy for Managed Evaluation Mode."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .evidence import launch_config_hash
from .schemas import (
    CandidatePromotionSummary,
    EvaluationRung,
    Goal,
    PriorCandidate,
    PromotionDecision,
    RungResult,
    ServingConfig,
)


@dataclass(frozen=True)
class ManagedBudgetPolicy:
    name: str = "pareto_successive_halving"
    rungs: list[EvaluationRung] = field(default_factory=list)
    min_candidates_for_staging: int = 3
    preserve_baseline: bool = True
    preserve_launch_diversity: bool = True
    strong_prior_confidence: float = 0.75

    @classmethod
    def default(cls) -> ManagedBudgetPolicy:
        return cls(
            rungs=[
                EvaluationRung(
                    index=0,
                    name="probe",
                    purpose="Cheap probe to reject clearly weak or unstable points.",
                    num_requests_scale=0.25,
                    min_num_requests=8,
                    max_num_requests=32,
                    trials=1,
                    promotion_fraction=0.5,
                ),
                EvaluationRung(
                    index=1,
                    name="measure",
                    purpose="Regular measurement for promoted candidates.",
                    num_requests_scale=1.0,
                    min_num_requests=1,
                    trials=None,
                    promotion_fraction=0.5,
                ),
                EvaluationRung(
                    index=2,
                    name="validate",
                    purpose="Validation pass for likely recommendation candidates.",
                    num_requests_scale=1.0,
                    min_num_requests=1,
                    trials=2,
                    promotion_fraction=0.25,
                    max_promotions=2,
                ),
            ]
        )

    @property
    def enabled(self) -> bool:
        return len(self.rungs) >= 2

    def should_stage(self, candidate_count: int) -> bool:
        return self.enabled and candidate_count >= self.min_candidates_for_staging


def select_promotion_decisions(
    *,
    candidates: list[ServingConfig],
    results: list[RungResult],
    prior_by_config_id: dict[str, PriorCandidate],
    goal: Goal,
    from_rung: EvaluationRung,
    to_rung: EvaluationRung | None,
    policy: ManagedBudgetPolicy,
) -> list[PromotionDecision]:
    candidate_ids = [candidate.id for candidate in candidates]
    results_by_candidate = {
        result.candidate_id: result
        for result in results
        if result.candidate_id in candidate_ids and result.status in {"completed", "evidence_hit"}
    }
    if not results_by_candidate:
        return [
            PromotionDecision(
                candidate_id=candidate.id,
                from_rung=from_rung.name,
                to_rung=to_rung.name if to_rung else None,
                promoted=False,
                reason="no measured probe result",
                prior_source=_prior_source(prior_by_config_id.get(candidate.id)),
                prior_confidence=_prior_confidence(prior_by_config_id.get(candidate.id)),
            )
            for candidate in candidates
        ]

    target_count = _target_promotion_count(len(results_by_candidate), from_rung)
    promoted_ids: set[str] = set()
    reasons: dict[str, list[str]] = {candidate_id: [] for candidate_id in candidate_ids}

    if policy.preserve_baseline and candidate_ids:
        _promote(candidate_ids[0], promoted_ids, reasons, "baseline")

    for candidate_id in _nondominated_candidate_ids(results_by_candidate.values()):
        _promote(candidate_id, promoted_ids, reasons, "nondominated")

    for candidate_id in _top_goal_candidate_ids(results_by_candidate.values(), goal, limit=target_count):
        _promote(candidate_id, promoted_ids, reasons, "top_goal_score")

    for candidate in candidates:
        prior = prior_by_config_id.get(candidate.id)
        if prior and prior.confidence is not None and prior.confidence >= policy.strong_prior_confidence and candidate.id in results_by_candidate:
            _promote(candidate.id, promoted_ids, reasons, "strong_prior")

    if policy.preserve_launch_diversity:
        for candidate_id in _launch_diversity_candidate_ids(candidates, results_by_candidate):
            if len(promoted_ids) >= target_count:
                break
            _promote(candidate_id, promoted_ids, reasons, "launch_diversity")

    if not promoted_ids:
        for candidate_id in _top_goal_candidate_ids(results_by_candidate.values(), goal, limit=1):
            _promote(candidate_id, promoted_ids, reasons, "fallback_top_goal_score")

    decisions: list[PromotionDecision] = []
    for candidate in candidates:
        result = results_by_candidate.get(candidate.id)
        prior = prior_by_config_id.get(candidate.id)
        promoted = candidate.id in promoted_ids
        decisions.append(
            PromotionDecision(
                candidate_id=candidate.id,
                from_rung=from_rung.name,
                to_rung=to_rung.name if promoted and to_rung else None,
                promoted=promoted,
                reason=", ".join(reasons.get(candidate.id) or ["not promoted"]),
                metrics=result.metrics if result else {},
                prior_source=_prior_source(prior),
                prior_confidence=_prior_confidence(prior),
            )
        )
    return decisions


def summarize_promotions(
    *,
    policy_name: str,
    candidate_count: int,
    rung_count: int,
    probe_candidate_count: int,
    decisions_by_rung: dict[str, list[PromotionDecision]],
) -> CandidatePromotionSummary:
    first_decisions = next(iter(decisions_by_rung.values()), [])
    promoted_after_probe = sum(1 for decision in first_decisions if decision.promoted)
    validation_candidates = 0
    if decisions_by_rung:
        last_decisions = list(decisions_by_rung.values())[-1]
        validation_candidates = sum(1 for decision in last_decisions if decision.promoted)
    return CandidatePromotionSummary(
        policy_name=policy_name,
        candidate_count=candidate_count,
        rung_count=rung_count,
        probe_candidate_count=probe_candidate_count,
        promoted_candidate_count=promoted_after_probe,
        validation_candidate_count=validation_candidates,
        pruned_after_probe_count=max(0, probe_candidate_count - promoted_after_probe),
        decisions_by_rung={
            rung: {
                "promoted": sum(1 for decision in decisions if decision.promoted),
                "not_promoted": sum(1 for decision in decisions if not decision.promoted),
            }
            for rung, decisions in decisions_by_rung.items()
        },
    )


def _target_promotion_count(result_count: int, rung: EvaluationRung) -> int:
    if rung.max_promotions is not None:
        return max(1, min(result_count, rung.max_promotions))
    return max(1, min(result_count, math.ceil(result_count * rung.promotion_fraction)))


def _promote(candidate_id: str, promoted_ids: set[str], reasons: dict[str, list[str]], reason: str) -> None:
    promoted_ids.add(candidate_id)
    reasons.setdefault(candidate_id, [])
    if reason not in reasons[candidate_id]:
        reasons[candidate_id].append(reason)


def _nondominated_candidate_ids(results: list[RungResult] | Any) -> list[str]:
    rows = [result for result in results if _metric(result, "throughput_tokens_per_sec") is not None and _metric(result, "joules_per_token") is not None]
    selected: list[str] = []
    for candidate in rows:
        dominated = any(_dominates(other, candidate) for other in rows if other is not candidate)
        if not dominated:
            selected.append(candidate.candidate_id)
    return selected


def _dominates(left: RungResult, right: RungResult) -> bool:
    left_throughput = _metric(left, "throughput_tokens_per_sec")
    right_throughput = _metric(right, "throughput_tokens_per_sec")
    left_energy = _metric(left, "joules_per_token")
    right_energy = _metric(right, "joules_per_token")
    if left_throughput is None or right_throughput is None or left_energy is None or right_energy is None:
        return False
    return (
        left_throughput >= right_throughput
        and left_energy <= right_energy
        and (left_throughput > right_throughput or left_energy < right_energy)
    )


def _top_goal_candidate_ids(results: list[RungResult] | Any, goal: Goal, *, limit: int) -> list[str]:
    ranked = sorted(results, key=lambda result: _goal_score(result, goal), reverse=True)
    return [result.candidate_id for result in ranked[:limit]]


def _goal_score(result: RungResult, goal: Goal) -> float:
    throughput = _metric(result, "throughput_tokens_per_sec") or 0.0
    latency = _metric(result, "p95_latency_ms")
    latency_score = 1.0 / latency if latency and latency > 0 else 0.0
    energy = _metric(result, "joules_per_token")
    efficiency_score = 1.0 / energy if energy and energy > 0 else 0.0
    if goal == Goal.PERFORMANCE:
        return throughput
    if goal == Goal.EFFICIENT:
        return efficiency_score
    return throughput + latency_score + efficiency_score


def _launch_diversity_candidate_ids(candidates: list[ServingConfig], results_by_candidate: dict[str, RungResult]) -> list[str]:
    selected: list[str] = []
    seen_hashes: set[str] = set()
    for candidate in candidates:
        if candidate.id not in results_by_candidate:
            continue
        launch_hash = launch_config_hash(candidate)
        if launch_hash in seen_hashes:
            continue
        seen_hashes.add(launch_hash)
        selected.append(candidate.id)
    return selected


def _metric(result: RungResult, key: str) -> float | None:
    value = result.metrics.get(key)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _prior_source(prior: PriorCandidate | None) -> str | None:
    return prior.source if prior else None


def _prior_confidence(prior: PriorCandidate | None) -> float | None:
    return prior.confidence if prior else None
