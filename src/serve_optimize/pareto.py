"""Pareto frontier construction and goal-aware recommendations."""

from __future__ import annotations

from .schemas import BenchmarkResult, Goal


def pareto_frontier(results: list[BenchmarkResult]) -> list[BenchmarkResult]:
    feasible = [result for result in results if result.feasible and result.throughput_tok_s > 0 and result.joules_per_token > 0]
    frontier: list[BenchmarkResult] = []
    for candidate in feasible:
        dominated = any(_dominates(other, candidate) for other in feasible if other is not candidate)
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda result: (-result.throughput_tok_s, result.joules_per_token))


def select_recommendation(results: list[BenchmarkResult], goal: Goal) -> BenchmarkResult | None:
    frontier = pareto_frontier(results)
    if not frontier:
        return None
    if goal == Goal.PERFORMANCE:
        return max(frontier, key=lambda result: (result.throughput_tok_s, -result.joules_per_token))
    if goal == Goal.EFFICIENT:
        return min(frontier, key=lambda result: (result.joules_per_token, -result.throughput_tok_s))
    return max(frontier, key=lambda result: (_balanced_score(result, frontier), result.throughput_tok_s))


def _dominates(left: BenchmarkResult, right: BenchmarkResult) -> bool:
    throughput_ok = left.throughput_tok_s >= right.throughput_tok_s
    energy_ok = left.joules_per_token <= right.joules_per_token
    strictly_better = left.throughput_tok_s > right.throughput_tok_s or left.joules_per_token < right.joules_per_token
    return throughput_ok and energy_ok and strictly_better


def _balanced_score(result: BenchmarkResult, frontier: list[BenchmarkResult]) -> float:
    throughput_values = [item.throughput_tok_s for item in frontier]
    efficiency_values = [1.0 / item.joules_per_token for item in frontier]
    throughput = _normalize(result.throughput_tok_s, min(throughput_values), max(throughput_values))
    efficiency = _normalize(1.0 / result.joules_per_token, min(efficiency_values), max(efficiency_values))
    # Harmonic-style score favors configs that keep both dimensions healthy.
    if throughput + efficiency == 0:
        return 0.0
    return 2.0 * throughput * efficiency / (throughput + efficiency)


def _normalize(value: float, low: float, high: float) -> float:
    if high <= low:
        return 1.0
    if low >= 0 and high > 0:
        return value / high
    return (value - low) / (high - low)
