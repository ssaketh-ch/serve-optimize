from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from serve_optimize.research_analysis import (
    MeasuredPool,
    SearchCandidate,
    compare_equal_budget,
    confidence_interval_95,
    load_measured_pool,
    oracle_post_hoc,
    run_search,
)


def _candidates() -> tuple[SearchCandidate, ...]:
    return (
        SearchCandidate("default", {"benchmark_concurrency": 1, "dtype": "bf16"}, baseline=True),
        SearchCandidate("a", {"benchmark_concurrency": 2, "dtype": "bf16"}),
        SearchCandidate("b", {"benchmark_concurrency": 4, "dtype": "fp16"}),
        SearchCandidate("c", {"benchmark_concurrency": 8, "dtype": "bf16"}),
    )


@pytest.mark.parametrize("method", ["random", "grid", "bayesian"])
def test_search_is_deterministic_and_respects_budget(method) -> None:
    scores = {"default": 0.1, "a": 0.8, "b": 0.5, "c": 1.0}
    calls: list[str] = []

    def evaluate(candidate_id: str) -> float:
        calls.append(candidate_id)
        return scores[candidate_id]

    first = run_search(_candidates(), evaluate, method=method, budget=3, seed=7)
    first_calls = list(calls)
    calls.clear()
    second = run_search(_candidates(), evaluate, method=method, budget=3, seed=7)

    assert first == second
    assert first.evaluation_order[0] == "default"
    assert len(first.evaluation_order) == 3
    assert set(first_calls) == set(first.evaluation_order)
    assert len(first_calls) == len(first.evaluation_order)
    assert set(calls) == set(second.evaluation_order)
    assert len(calls) == len(second.evaluation_order)
    assert "c" not in calls or "c" in second.evaluation_order


def test_bayesian_search_does_not_evaluate_unselected_candidates() -> None:
    candidates = _candidates()
    scores = {"default": 0.1, "a": 0.8, "b": 0.5, "c": 1.0}
    calls: list[str] = []

    result = run_search(candidates, lambda item: calls.append(item) or scores[item], method="bayesian", budget=2, seed=3)

    assert set(calls) == set(result.evaluation_order)
    assert len(set(calls)) == 2


def test_oracle_is_a_separate_post_hoc_operation() -> None:
    assert oracle_post_hoc({"a": 0.5, "b": 0.9}) == ("b", 0.9)


def test_confidence_interval_uses_student_t_for_three_repetitions() -> None:
    result = confidence_interval_95([10.0, 12.0, 14.0])

    assert result["n"] == 3
    assert result["mean"] == 12.0
    assert result["stddev"] == 2.0
    expected_margin = 4.303 * 2.0 / math.sqrt(3)
    assert result["ci95_low"] == pytest.approx(12.0 - expected_margin)
    assert result["ci95_high"] == pytest.approx(12.0 + expected_margin)


def test_confidence_interval_refuses_to_invent_single_sample_uncertainty() -> None:
    result = confidence_interval_95([5.0])

    assert result == {"n": 1, "mean": 5.0, "stddev": None, "ci95_low": None, "ci95_high": None}


def test_equal_budget_builds_anytime_curves_without_future_information(tmp_path) -> None:
    pool = MeasuredPool(
        run_dir=tmp_path,
        identity={"backend": "vllm"},
        goal="balanced",
        candidates=_candidates(),
        scores={"default": 0.1, "a": 0.8, "b": 0.5, "c": 1.0},
        rows={},
        selected_candidate_id="a",
        probe_order=("default", "a", "b", "c"),
        pareto_ids=frozenset({"a", "c"}),
        generation={},
        pruning={},
        score_weights={},
    )

    rows = compare_equal_budget(pool, seeds=iter([3]))

    equal_budget = [row for row in rows if row["method"] != "backend_default"]
    assert {row["candidate_evaluation_budget"] for row in equal_budget} == {1, 2, 3, 4}
    assert all(
        len(str(row["evaluation_order"]).split(";")) == row["candidate_evaluation_budget"]
        for row in equal_budget
    )
    assert all(row["oracle_used_during_selection"] is False for row in rows)
    assert {row["oracle_candidate_id"] for row in rows} == {"c"}
    serve_rows = [row for row in rows if row["method"] == "serve_optimize_candidate_order"]
    assert [row["selected_candidate_id"] for row in serve_rows] == ["default", "a", "a", "c"]
    assert len([row for row in rows if row["method"] == "bayesian"]) == 4


def test_measured_pool_preserves_launch_order_without_oracle_leakage(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    payloads = {
        "managed_recommendation.json": {
            "recommendation": {
                "recommended_candidate_id": "fast",
                "score_weights": {"throughput": 0.8, "latency": 0.1, "reliability": 0.1},
                "candidate_table": [
                    {"candidate_id": "fast", "candidate_source": "generated", "score": 1.0},
                    {"candidate_id": "default", "candidate_source": "safe_baseline", "score": 0.2},
                ],
            }
        },
        "managed_run.json": {"backend": "vllm", "model": "model", "goal": "balanced"},
        "candidate_generation_report.json": {},
        "candidate_pruning_report.json": {},
    }
    for name, payload in payloads.items():
        (run_dir / name).write_text(json.dumps(payload), encoding="utf-8")
    probe_rows = [
        {"candidate_id": "not_measured", "from_rung": "probe", "metrics": {}},
        {
            "candidate_id": "default",
            "from_rung": "probe",
            "metrics": {
                "output_tokens_per_sec": 10.0,
                "p95_latency_ms": 10.0,
                "successful_requests": 8,
                "failed_requests": 0,
                "total_requests": 8,
            },
        },
        {
            "candidate_id": "fast",
            "from_rung": "probe",
            "metrics": {
                "output_tokens_per_sec": 5.0,
                "p95_latency_ms": 20.0,
                "successful_requests": 8,
                "failed_requests": 0,
                "total_requests": 8,
            },
        },
    ]
    (run_dir / "promotion_decisions.jsonl").write_text(
        "\n".join(json.dumps(row) for row in probe_rows) + "\n", encoding="utf-8"
    )
    (run_dir / "rendered_launch_configs.jsonl").write_text(
        '\n'.join(
            [
                '{"canonical_config_id":"default","canonical_config":{}}',
                '{"canonical_config_id":"fast","canonical_config":{}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    pool = load_measured_pool(run_dir)

    assert pool.probe_order == ("default", "fast")
    assert pool.identity["candidate_order_score_scope"] == "common_probe_rung"
    assert pool.scores["default"] > pool.scores["fast"]


def test_research_runner_postprocesses_only_recorded_successful_runs() -> None:
    runner = (
        Path(__file__).resolve().parents[1] / "scripts" / "slurm" / "run_research_closure_h200.sbatch"
    ).read_text(encoding="utf-8")

    assert 'printf \'%s\\n\' "${run_path}" >"${MARKERS}/${step}.managed-run"' in runner
    assert 'find "${MARKERS}" -name \'*.managed-run\'' in runner
    assert 'find "${CLOSURE_ROOT}/managed" -name managed_run.json' not in runner
    assert 'rm -f "${MARKERS}/${step}.ok"' in runner
    assert '--goal throughput' in runner
