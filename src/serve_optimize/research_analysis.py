"""Deterministic finite pool analyses for the research artifact."""

from __future__ import annotations

import json
import math
import random
import statistics
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

SEARCH_FEATURES = (
    "benchmark_concurrency",
    "gpu_memory_utilization",
    "max_model_len",
    "max_num_seqs",
    "tensor_parallel_size",
    "estimated_vram_mb",
    "dtype",
    "quantization",
    "candidate_source",
)


@dataclass(frozen=True)
class SearchCandidate:
    candidate_id: str
    features: dict[str, object]
    baseline: bool = False


@dataclass(frozen=True)
class SearchResult:
    method: str
    seed: int | None
    budget: int
    evaluation_order: tuple[str, ...]
    selected_candidate_id: str
    selected_score: float


@dataclass(frozen=True)
class MeasuredPool:
    run_dir: Path
    identity: dict[str, object]
    goal: str
    candidates: tuple[SearchCandidate, ...]
    scores: dict[str, float]
    rows: dict[str, dict[str, object]]
    selected_candidate_id: str
    probe_order: tuple[str, ...]
    pareto_ids: frozenset[str]
    generation: dict[str, object]
    pruning: dict[str, object]
    score_weights: dict[str, float]


def load_measured_pool(run_dir: Path) -> MeasuredPool:
    recommendation = _load_json(run_dir / "managed_recommendation.json")
    managed = _load_json(run_dir / "managed_run.json")
    generation = _load_json(run_dir / "candidate_generation_report.json")
    pruning = _load_json(run_dir / "candidate_pruning_report.json")
    payload = recommendation.get("recommendation") or {}
    candidate_rows = payload.get("candidate_table") or []
    launch_configs = _launch_configs(run_dir / "rendered_launch_configs.jsonl")
    rows: dict[str, dict[str, object]] = {}
    candidates: list[SearchCandidate] = []
    scores: dict[str, float] = {}
    for raw_row in candidate_rows:
        row = dict(raw_row)
        candidate_id = str(row["candidate_id"])
        launch = launch_configs.get(candidate_id, {})
        canonical = launch.get("canonical_config") if isinstance(launch, dict) else {}
        if isinstance(canonical, dict):
            row.setdefault("estimated_vram_mb", canonical.get("estimated_vram_mb"))
        score = _finite_float(row.get("score"))
        if score is None:
            score = -1.0
        rows[candidate_id] = row
        scores[candidate_id] = score
        source = str(row.get("candidate_source") or "")
        candidates.append(
            SearchCandidate(
                candidate_id=candidate_id,
                features={name: row.get(name) for name in SEARCH_FEATURES},
                baseline=source == "safe_baseline",
            )
        )
    if not candidates:
        raise ValueError(f"No measured candidates in {run_dir}")
    if not any(candidate.baseline for candidate in candidates):
        raise ValueError(f"No backend default control in {run_dir}")
    selected = str(payload.get("recommended_candidate_id") or "")
    if selected not in rows:
        raise ValueError(f"Recommended candidate is absent from measured pool in {run_dir}")
    measured_ids = set(scores)
    promotion_rows = _load_jsonl(run_dir / "promotion_decisions.jsonl")
    probe_scores = _common_probe_scores(
        promotion_rows,
        candidate_ids=measured_ids,
        goal=str(managed.get("goal") or payload.get("goal") or "balanced"),
        weights={str(key): float(value) for key, value in (payload.get("score_weights") or {}).items()},
    )
    candidate_order_score_scope = "final_common_measure_rung"
    if probe_scores is not None:
        scores = probe_scores
        candidate_order_score_scope = "common_probe_rung"
    probe_order = _unique(
        str(row.get("candidate_id"))
        for row in promotion_rows
        if row.get("from_rung") == "probe" and str(row.get("candidate_id")) in measured_ids
    )
    probe_order = _unique(
        (
            *probe_order,
            *(candidate_id for candidate_id in launch_configs if candidate_id in measured_ids),
            *(candidate.candidate_id for candidate in candidates),
        )
    )
    model_summary = generation.get("model_summary") or {}
    model_metadata = model_summary.get("metadata") if isinstance(model_summary, dict) else {}
    runtime = managed.get("runtime_environment") or {}
    profile = managed.get("workload_profile") or {}
    identity = {
        "backend": managed.get("backend"),
        "backend_version": managed.get("backend_version") or runtime.get("backend_version"),
        "model": managed.get("model"),
        "model_revision": model_metadata.get("revision") if isinstance(model_metadata, dict) else None,
        "goal": managed.get("goal"),
        "workload_profile": profile.get("profile_name") if isinstance(profile, dict) else None,
        "runtime_environment_fingerprint": runtime.get("environment_fingerprint"),
        "candidate_order_score_scope": candidate_order_score_scope,
    }
    return MeasuredPool(
        run_dir=run_dir,
        identity=identity,
        goal=str(managed.get("goal") or payload.get("goal") or "balanced"),
        candidates=tuple(candidates),
        scores=scores,
        rows=rows,
        selected_candidate_id=selected,
        probe_order=tuple(probe_order),
        pareto_ids=frozenset(str(row.get("candidate_id")) for row in payload.get("pareto_frontier") or []),
        generation=generation,
        pruning=pruning,
        score_weights={str(key): float(value) for key, value in (payload.get("score_weights") or {}).items()},
    )


def run_search(
    candidates: Iterable[SearchCandidate],
    evaluate: Callable[[str], float],
    *,
    method: str,
    budget: int,
    seed: int = 0,
) -> SearchResult:
    pool = tuple(candidates)
    if not pool:
        raise ValueError("Search requires at least one candidate.")
    if budget < 1 or budget > len(pool):
        raise ValueError("Search budget must be between one and the candidate count.")
    baseline = next((candidate for candidate in pool if candidate.baseline), pool[0])
    if method == "random":
        order = [baseline, *_random_order(pool, baseline, seed)]
    elif method == "grid":
        order = [baseline, *_grid_order(pool, baseline)]
    elif method == "bayesian":
        observations: dict[str, float] = {}

        def cached_evaluate(candidate_id: str) -> float:
            if candidate_id not in observations:
                observations[candidate_id] = evaluate(candidate_id)
            return observations[candidate_id]

        order = _bayesian_order(pool, baseline, cached_evaluate, budget=budget, seed=seed)
        selected = max(order, key=lambda candidate: (observations[candidate.candidate_id], candidate.candidate_id))
        return SearchResult(
            method=method,
            seed=seed,
            budget=budget,
            evaluation_order=tuple(candidate.candidate_id for candidate in order),
            selected_candidate_id=selected.candidate_id,
            selected_score=observations[selected.candidate_id],
        )
    else:
        raise ValueError(f"Unsupported search method: {method}")
    observed = order[:budget]
    observations = {candidate.candidate_id: evaluate(candidate.candidate_id) for candidate in observed}
    selected = max(observed, key=lambda candidate: (observations[candidate.candidate_id], candidate.candidate_id))
    return SearchResult(
        method=method,
        seed=seed if method == "random" else None,
        budget=budget,
        evaluation_order=tuple(candidate.candidate_id for candidate in observed),
        selected_candidate_id=selected.candidate_id,
        selected_score=observations[selected.candidate_id],
    )


def compare_equal_budget(pool: MeasuredPool, *, seeds: Iterable[int] = range(20)) -> list[dict[str, object]]:
    method_selections: list[tuple[str, int | None, int, str, float, tuple[str, ...]]] = []
    seed_values = tuple(seeds)
    baseline_id = next(candidate.candidate_id for candidate in pool.candidates if candidate.baseline)
    method_selections.append(("backend_default", None, 1, baseline_id, pool.scores[baseline_id], (baseline_id,)))
    for budget in range(1, len(pool.candidates) + 1):
        observed = pool.probe_order[:budget]
        selected_id = max(observed, key=lambda item: (pool.scores[item], item))
        method_selections.append(
            ("serve_optimize_candidate_order", None, budget, selected_id, pool.scores[selected_id], observed)
        )
        for method in ("random", "grid", "bayesian"):
            method_seeds = seed_values if method in {"random", "bayesian"} else (0,)
            for seed in method_seeds:
                result = run_search(
                    pool.candidates,
                    pool.scores.__getitem__,
                    method=method,
                    budget=budget,
                    seed=seed,
                )
                method_selections.append(
                    (
                        method,
                        result.seed,
                        budget,
                        result.selected_candidate_id,
                        result.selected_score,
                        result.evaluation_order,
                    )
                )
    oracle_id, oracle_score = oracle_post_hoc(pool.scores)
    rank = _rank_map(pool.scores)
    return [
        _comparison_row(
            pool,
            method=method,
            seed=seed,
            budget=method_budget,
            selected_id=selected_id,
            selected_score=selected_score,
            evaluation_order=evaluation_order,
            oracle_id=oracle_id,
            oracle_score=oracle_score,
            rank=rank,
        )
        for method, seed, method_budget, selected_id, selected_score, evaluation_order in method_selections
    ]


def replay_ablations(pool: MeasuredPool, *, seeds: Iterable[int] = range(20)) -> list[dict[str, object]]:
    original = pool.selected_candidate_id
    rows: list[dict[str, object]] = []
    exact_ids = set((pool.generation.get("prior_and_synthesis") or {}).get("exact_fresh_candidate_ids") or [])
    prior_count = int((pool.generation.get("prior_and_synthesis") or {}).get("evidence_prior_count") or 0)
    if prior_count == 0 and not exact_ids:
        rows.append(_ablation_row(pool, "no_evidence_reuse", "supported_no_op", original, original, "No evidence influenced this measured pool."))
    else:
        rows.append(
            _ablation_row(
                pool,
                "no_evidence_reuse",
                "unsupported_missing_counterfactual_measurements",
                original,
                None,
                "Evidence influenced candidate generation or pruning, so removing it can change unmeasured candidates.",
            )
        )
    rows.append(
        _ablation_row(
            pool,
            "no_hardware_awareness",
            "unsupported_missing_counterfactual_measurements",
            original,
            None,
            "Hardware awareness shapes candidate generation; the alternative candidate pool was not measured.",
        )
    )
    rows.append(
        _ablation_row(
            pool,
            "no_backend_capability_registry",
            "unsupported_missing_counterfactual_measurements",
            original,
            None,
            "Capability filtering removed unsupported combinations before measurement.",
        )
    )
    failure_policy = pool.pruning.get("failure_memory_policy") or {}
    if not failure_policy.get("external_failure_cache_loaded") and int(pool.pruning.get("repeat_failed_exact_configuration_count") or 0) == 0:
        rows.append(_ablation_row(pool, "no_failure_memory", "supported_no_op", original, original, "No external failure memory entry applied."))
    else:
        rows.append(
            _ablation_row(
                pool,
                "no_failure_memory",
                "unsupported_missing_counterfactual_measurements",
                original,
                None,
                "Previously failed configurations were not remeasured without failure memory.",
            )
        )
    random_trials = []
    for seed in seeds:
        order = [next(candidate.candidate_id for candidate in pool.candidates if candidate.baseline)]
        remaining = [candidate.candidate_id for candidate in pool.candidates if candidate.candidate_id not in order]
        random.Random(seed).shuffle(remaining)
        order.extend(remaining)
        random_trials.append(order.index(original) + 1)
    rows.append(
        _ablation_row(
            pool,
            "random_candidate_order",
            "supported_replay_search_cost_only",
            original,
            original,
            "All candidates remain measured; order changes discovery cost but not full pool scoring.",
            extra={"mean_candidate_evaluations_to_original": statistics.fmean(random_trials), "seed_count": len(random_trials)},
        )
    )
    rows.append(_scoring_ablation(pool, "energy_term_removed", remove_energy=True, remove_latency=False))
    rows.append(_scoring_ablation(pool, "latency_guardrail_removed", remove_energy=False, remove_latency=True))
    baseline_id = next(candidate.candidate_id for candidate in pool.candidates if candidate.baseline)
    rows.append(_ablation_row(pool, "only_backend_defaults", "supported_replay", original, baseline_id, "The measured backend default control is retained alone."))
    return rows


def oracle_post_hoc(scores: dict[str, float]) -> tuple[str, float]:
    if not scores:
        raise ValueError("Oracle requires at least one measured score.")
    candidate_id = max(scores, key=lambda item: (scores[item], item))
    return candidate_id, scores[candidate_id]


def confidence_interval_95(values: Iterable[float]) -> dict[str, float | int | None]:
    samples = [float(value) for value in values]
    if not samples:
        return {"n": 0, "mean": None, "stddev": None, "ci95_low": None, "ci95_high": None}
    mean = statistics.fmean(samples)
    if len(samples) == 1:
        return {"n": 1, "mean": mean, "stddev": None, "ci95_low": None, "ci95_high": None}
    stddev = statistics.stdev(samples)
    critical = _t_critical_95(len(samples) - 1)
    margin = critical * stddev / math.sqrt(len(samples))
    return {"n": len(samples), "mean": mean, "stddev": stddev, "ci95_low": mean - margin, "ci95_high": mean + margin}


def _bayesian_order(
    pool: tuple[SearchCandidate, ...],
    baseline: SearchCandidate,
    evaluate: Callable[[str], float],
    *,
    budget: int,
    seed: int,
) -> list[SearchCandidate]:
    vectors = _feature_vectors(pool)
    by_id = {candidate.candidate_id: candidate for candidate in pool}
    rng = random.Random(seed)
    selected = [baseline.candidate_id]
    observed = {baseline.candidate_id: evaluate(baseline.candidate_id)}
    remaining = sorted(candidate.candidate_id for candidate in pool if candidate.candidate_id != baseline.candidate_id)
    if len(selected) < budget and remaining:
        initial = remaining[rng.randrange(len(remaining))]
        selected.append(initial)
        observed[initial] = evaluate(initial)
    while len(selected) < budget:
        best_seen = max(observed.values())
        scored = []
        for candidate_id in remaining:
            if candidate_id in observed:
                continue
            mean, variance = _gp_predict(
                [vectors[item] for item in observed],
                [observed[item] for item in observed],
                vectors[candidate_id],
            )
            stddev = math.sqrt(max(variance, 0.0))
            acquisition = _expected_improvement(mean, stddev, best_seen)
            scored.append((acquisition, candidate_id))
        _, next_id = max(scored, key=lambda item: (item[0], item[1]))
        selected.append(next_id)
        observed[next_id] = evaluate(next_id)
    return [by_id[candidate_id] for candidate_id in selected]


def _gp_predict(x_train: list[list[float]], y_train: list[float], x_value: list[float]) -> tuple[float, float]:
    if not x_train:
        return 0.0, 1.0
    y_mean = statistics.fmean(y_train)
    centered = [value - y_mean for value in y_train]
    kernel = [[_rbf(left, right) + (1e-6 if i == j else 0.0) for j, right in enumerate(x_train)] for i, left in enumerate(x_train)]
    lower = _cholesky(kernel)
    alpha = _cholesky_solve(lower, centered)
    cross = [_rbf(row, x_value) for row in x_train]
    mean = y_mean + sum(value * weight for value, weight in zip(cross, alpha, strict=True))
    solved = _forward_substitute(lower, cross)
    variance = max(1.0 - sum(value * value for value in solved), 1e-12)
    return mean, variance


def _feature_vectors(candidates: tuple[SearchCandidate, ...]) -> dict[str, list[float]]:
    numeric_fields = [name for name in SEARCH_FEATURES if all(_finite_float(candidate.features.get(name)) is not None for candidate in candidates)]
    categorical_fields = [name for name in SEARCH_FEATURES if name not in numeric_fields]
    ranges: dict[str, tuple[float, float]] = {}
    for name in numeric_fields:
        values = [float(candidate.features[name]) for candidate in candidates]
        ranges[name] = (min(values), max(values))
    categories = {
        name: sorted({str(candidate.features.get(name)) for candidate in candidates})
        for name in categorical_fields
    }
    result = {}
    for candidate in candidates:
        vector = []
        for name in numeric_fields:
            low, high = ranges[name]
            value = float(candidate.features[name])
            vector.append(0.0 if high == low else (value - low) / (high - low))
        for name in categorical_fields:
            value = str(candidate.features.get(name))
            vector.extend(1.0 if value == category else 0.0 for category in categories[name])
        result[candidate.candidate_id] = vector or [0.0]
    return result


def _grid_order(pool: tuple[SearchCandidate, ...], baseline: SearchCandidate) -> list[SearchCandidate]:
    return sorted(
        (candidate for candidate in pool if candidate.candidate_id != baseline.candidate_id),
        key=lambda candidate: tuple(_sort_value(candidate.features.get(name)) for name in SEARCH_FEATURES) + (candidate.candidate_id,),
    )


def _random_order(pool: tuple[SearchCandidate, ...], baseline: SearchCandidate, seed: int) -> list[SearchCandidate]:
    remaining = [candidate for candidate in pool if candidate.candidate_id != baseline.candidate_id]
    random.Random(seed).shuffle(remaining)
    return remaining


def _scoring_ablation(pool: MeasuredPool, name: str, *, remove_energy: bool, remove_latency: bool) -> dict[str, object]:
    scores = {
        candidate_id: _component_score(row, pool.score_weights, remove_energy=remove_energy, remove_latency=remove_latency)
        for candidate_id, row in pool.rows.items()
    }
    supported = {candidate_id: value for candidate_id, value in scores.items() if value is not None}
    if not supported:
        return _ablation_row(pool, name, "unsupported_missing_scoring_inputs", pool.selected_candidate_id, None, "Required score components are absent.")
    selected = max(supported, key=lambda item: (supported[item], item))
    removed = "energy" if remove_energy else "weighted latency"
    return _ablation_row(
        pool,
        name,
        "supported_replay",
        pool.selected_candidate_id,
        selected,
        f"Recomputed the fixed pool score without the {removed} term; eligibility guards remain enforced.",
    )


def _component_score(
    row: dict[str, object],
    weights: dict[str, float],
    *,
    remove_energy: bool,
    remove_latency: bool,
) -> float | None:
    if row.get("status") != "eligible":
        return None
    terms = {
        "throughput": _finite_float(row.get("throughput_score")),
        "latency": None if remove_latency else _finite_float(row.get("latency_score")),
        "reliability": _finite_float(row.get("reliability_score")),
        "power": None if remove_energy else _finite_float(row.get("power_score")),
    }
    used = [(terms[name], weight) for name, weight in weights.items() if name in terms and weight > 0]
    total_weight = sum(weight for _, weight in used)
    if not used or total_weight <= 0 or not any(value is not None for value, _ in used):
        return None
    return sum(float(value or 0.0) * weight for value, weight in used) / total_weight


def _common_probe_scores(
    decisions: list[dict[str, object]],
    *,
    candidate_ids: set[str],
    goal: str,
    weights: dict[str, float],
) -> dict[str, float] | None:
    metrics = {
        str(row.get("candidate_id")): row.get("metrics")
        for row in decisions
        if row.get("from_rung") == "probe"
        and str(row.get("candidate_id")) in candidate_ids
        and isinstance(row.get("metrics"), dict)
    }
    if set(metrics) != candidate_ids:
        return None
    failure_intolerant = goal in {"performance", "throughput", "efficient", "efficiency"}
    eligible = {
        candidate_id
        for candidate_id, row in metrics.items()
        if (_finite_float(row.get("successful_requests")) or 0) > 0
        and (_finite_float(row.get("output_tokens_per_sec")) or _finite_float(row.get("throughput_tokens_per_sec")) or 0) > 0
        and (not failure_intolerant or (_finite_float(row.get("failed_requests")) or 0) == 0)
    }
    throughput = _normalize_values(
        {
            candidate_id: _finite_float(row.get("output_tokens_per_sec"))
            or _finite_float(row.get("throughput_tokens_per_sec"))
            for candidate_id, row in metrics.items()
            if candidate_id in eligible
        },
        higher=True,
    )
    latency = _normalize_values(
        {
            candidate_id: _finite_float(row.get("p95_latency_ms"))
            for candidate_id, row in metrics.items()
            if candidate_id in eligible
        },
        higher=False,
    )
    efficiency = _normalize_values(
        {
            candidate_id: _finite_float(row.get("tokens_per_joule"))
            or _finite_float(row.get("active_tokens_per_watt"))
            or _finite_float(row.get("tokens_per_watt"))
            for candidate_id, row in metrics.items()
            if candidate_id in eligible
        },
        higher=True,
    )
    energy_cost = _normalize_values(
        {
            candidate_id: _finite_float(row.get("joules_per_generated_token"))
            or _finite_float(row.get("active_joules_per_token"))
            or _finite_float(row.get("joules_per_token"))
            for candidate_id, row in metrics.items()
            if candidate_id in eligible
        },
        higher=False,
    )
    scores = {}
    for candidate_id, row in metrics.items():
        if candidate_id not in eligible:
            scores[candidate_id] = -1.0
            continue
        total = _finite_float(row.get("total_requests")) or 0.0
        successful = _finite_float(row.get("successful_requests")) or 0.0
        power = _weighted_value(((efficiency.get(candidate_id), 0.6), (energy_cost.get(candidate_id), 0.4)))
        value = _weighted_value(
            (
                (throughput.get(candidate_id), weights.get("throughput", 0.0)),
                (latency.get(candidate_id), weights.get("latency", 0.0)),
                (power, weights.get("power", 0.0)),
                (successful / total if total > 0 else None, weights.get("reliability", 0.0)),
            )
        )
        scores[candidate_id] = value if value is not None else -1.0
    return scores


def _normalize_values(values: dict[str, float | None], *, higher: bool) -> dict[str, float | None]:
    valid = [value for value in values.values() if value is not None]
    if not valid:
        return {candidate_id: None for candidate_id in values}
    low, high = min(valid), max(valid)
    if high == low:
        return {candidate_id: 1.0 if value is not None else None for candidate_id, value in values.items()}
    if higher:
        return {
            candidate_id: value / high if value is not None and low >= 0 and high > 0 else (value - low) / (high - low) if value is not None else None
            for candidate_id, value in values.items()
        }
    if low >= 0:
        return {
            candidate_id: low / value if value not in {None, 0} and low > 0 else (high - value) / high if value is not None and high > 0 else None
            for candidate_id, value in values.items()
        }
    return {
        candidate_id: (high - value) / (high - low) if value is not None else None
        for candidate_id, value in values.items()
    }


def _weighted_value(parts: Iterable[tuple[float | None, float]]) -> float | None:
    values = [(value, weight) for value, weight in parts if weight > 0]
    total_weight = sum(weight for _, weight in values)
    if total_weight <= 0 or not any(value is not None for value, _ in values):
        return None
    return sum(float(value or 0.0) * weight for value, weight in values) / total_weight


def _comparison_row(
    pool: MeasuredPool,
    *,
    method: str,
    seed: int | None,
    budget: int,
    selected_id: str,
    selected_score: float,
    evaluation_order: Iterable[str],
    oracle_id: str,
    oracle_score: float,
    rank: dict[str, int],
) -> dict[str, object]:
    order = tuple(evaluation_order)
    return {
        **pool.identity,
        "run_dir": str(pool.run_dir),
        "method": method,
        "seed": seed,
        "candidate_evaluation_budget": budget,
        "budget_policy": "anytime_unique_candidate_observations",
        "candidate_pool_size": len(pool.candidates),
        "selected_candidate_id": selected_id,
        "selected_score": selected_score,
        "selection_score_scope": pool.identity.get("candidate_order_score_scope", "provided_score_map"),
        "selected_rank": rank[selected_id],
        "selected_on_pareto_frontier": selected_id in pool.pareto_ids,
        "pareto_scope": "final_recommendation_candidate_table",
        "oracle_candidate_id": oracle_id,
        "oracle_score": oracle_score,
        "score_regret": oracle_score - selected_score,
        "relative_score_regret": 0.0 if oracle_score == 0 else (oracle_score - selected_score) / abs(oracle_score),
        "candidate_evaluations_to_oracle": order.index(oracle_id) + 1 if oracle_id in order else None,
        "evaluation_order": ";".join(order),
        "oracle_used_during_selection": False,
        "replay_scope": "fixed_measured_candidate_pool",
    }


def _ablation_row(
    pool: MeasuredPool,
    name: str,
    status: str,
    original: str,
    selected: str | None,
    reason: str,
    *,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    row = {
        **pool.identity,
        "run_dir": str(pool.run_dir),
        "ablation": name,
        "status": status,
        "original_candidate_id": original,
        "ablated_candidate_id": selected,
        "recommendation_changed": selected is not None and selected != original,
        "ablation_evidence_scope": "final_recommendation_candidate_table",
        "reason": reason,
    }
    row.update(extra or {})
    return row


def _rank_map(scores: dict[str, float]) -> dict[str, int]:
    ordered = sorted(scores, key=lambda item: (-scores[item], item))
    return {candidate_id: index for index, candidate_id in enumerate(ordered, start=1)}


def _rbf(left: list[float], right: list[float], length_scale: float = 0.7) -> float:
    distance = sum((a - b) ** 2 for a, b in zip(left, right, strict=True))
    return math.exp(-0.5 * distance / (length_scale * length_scale))


def _expected_improvement(mean: float, stddev: float, best: float, exploration: float = 0.01) -> float:
    if stddev <= 1e-12:
        return 0.0
    improvement = mean - best - exploration
    z = improvement / stddev
    pdf = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return improvement * cdf + stddev * pdf


def _cholesky(matrix: list[list[float]]) -> list[list[float]]:
    size = len(matrix)
    lower = [[0.0] * size for _ in range(size)]
    for row in range(size):
        for column in range(row + 1):
            value = matrix[row][column] - sum(lower[row][k] * lower[column][k] for k in range(column))
            if row == column:
                lower[row][column] = math.sqrt(max(value, 1e-12))
            else:
                lower[row][column] = value / lower[column][column]
    return lower


def _forward_substitute(lower: list[list[float]], values: list[float]) -> list[float]:
    result = []
    for row, value in enumerate(values):
        result.append((value - sum(lower[row][column] * result[column] for column in range(row))) / lower[row][row])
    return result


def _cholesky_solve(lower: list[list[float]], values: list[float]) -> list[float]:
    forward = _forward_substitute(lower, values)
    result = [0.0] * len(values)
    for row in range(len(values) - 1, -1, -1):
        result[row] = (
            forward[row] - sum(lower[column][row] * result[column] for column in range(row + 1, len(values)))
        ) / lower[row][row]
    return result


def _t_critical_95(degrees_of_freedom: int) -> float:
    values = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
    if degrees_of_freedom in values:
        return values[degrees_of_freedom]
    if degrees_of_freedom <= 20:
        return 2.086
    if degrees_of_freedom <= 30:
        return 2.042
    return 1.96


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _launch_configs(path: Path) -> dict[str, dict[str, object]]:
    result = {}
    for row in _load_jsonl(path):
        candidate_id = row.get("canonical_config_id") or row.get("logical_config_id")
        if candidate_id:
            result[str(candidate_id)] = row
    return result


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _sort_value(value: object) -> tuple[int, float | str]:
    number = _finite_float(value)
    return (0, number) if number is not None else (1, str(value))


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    result = []
    for value in values:
        if value not in result:
            result.append(value)
    return tuple(result)
