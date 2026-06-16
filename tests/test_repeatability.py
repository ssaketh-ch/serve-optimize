import json

import pytest

from serve_optimize.cli import main
from serve_optimize.repeatability import analyze_repeatability, write_repeatability_artifacts


def test_repeatability_one_run_is_insufficient(tmp_path) -> None:
    run = _write_run(tmp_path / "run1", selected_candidate_id="a")

    payload = analyze_repeatability([run])

    assert payload["run_count"] == 1
    assert payload["usable_run_count"] == 1
    assert payload["stability_classification"] == "insufficient_runs"


def test_repeatability_same_selected_config_is_stable(tmp_path) -> None:
    run1 = _write_run(tmp_path / "run1", selected_candidate_id="a")
    run2 = _write_run(tmp_path / "run2", selected_candidate_id="b")

    payload = analyze_repeatability([run1, run2])

    assert payload["stability_classification"] == "stable"
    assert payload["selected_canonical_config_stability"]["stable"] is True
    assert payload["selected_candidate_id_stability"]["stable"] is False
    assert payload["pareto_frontier_overlap"]["pair_count"] == 1
    assert payload["pareto_frontier_overlap"]["mean"] == 1.0


def test_repeatability_high_top3_overlap_is_mostly_stable(tmp_path) -> None:
    run1 = _write_run(
        tmp_path / "run1",
        selected_candidate_id="a",
        selected_overrides={"max_model_len": 2048},
        candidate_table=[
            _candidate("a", max_model_len=2048),
            _candidate("shared1", max_model_len=4096),
            _candidate("shared2", max_model_len=8192),
        ],
    )
    run2 = _write_run(
        tmp_path / "run2",
        selected_candidate_id="b",
        selected_overrides={"max_model_len": 4096},
        candidate_table=[
            _candidate("b", max_model_len=4096),
            _candidate("shared1", max_model_len=4096),
            _candidate("shared2", max_model_len=8192),
        ],
    )

    payload = analyze_repeatability([run1, run2])

    assert payload["stability_classification"] == "mostly_stable"
    assert payload["top3_overlap"]["mean"] >= 0.5


def test_repeatability_low_overlap_is_unstable(tmp_path) -> None:
    run1 = _write_run(
        tmp_path / "run1",
        selected_candidate_id="a",
        selected_overrides={"max_model_len": 2048},
        candidate_table=[_candidate("a", max_model_len=2048), _candidate("x1", max_model_len=3072)],
    )
    run2 = _write_run(
        tmp_path / "run2",
        selected_candidate_id="b",
        selected_overrides={"max_model_len": 4096},
        candidate_table=[_candidate("b", max_model_len=4096), _candidate("x2", max_model_len=6144)],
    )

    payload = analyze_repeatability([run1, run2])

    assert payload["stability_classification"] == "unstable"


def test_repeatability_metric_variation_math(tmp_path) -> None:
    run1 = _write_run(tmp_path / "run1", selected_candidate_id="a", metrics={"throughput_tokens_per_sec": 100.0})
    run2 = _write_run(tmp_path / "run2", selected_candidate_id="a", metrics={"throughput_tokens_per_sec": 110.0})

    payload = analyze_repeatability([run1, run2])
    variation = payload["selected_metric_variation"]["throughput_tokens_per_sec"]

    assert variation["min"] == 100.0
    assert variation["max"] == 110.0
    assert variation["absolute_delta"] == 10.0
    assert variation["relative_delta"] == pytest.approx(0.1)


def test_repeatability_missing_files_produce_warnings(tmp_path) -> None:
    run = tmp_path / "missing"
    run.mkdir()

    payload = analyze_repeatability([run])

    assert payload["usable_run_count"] == 0
    assert payload["skipped_run_count"] == 1
    assert payload["warnings"]


def test_repeatability_evidence_reuse_classification(tmp_path) -> None:
    run1 = _write_run(tmp_path / "run1", selected_candidate_id="a", cold=0, measurements=0, hits=2)
    run2 = _write_run(tmp_path / "run2", selected_candidate_id="a", cold=0, measurements=0, hits=3)

    payload = analyze_repeatability([run1, run2])

    assert payload["evidence_reuse"]["reuse_classification"] == "strong_reuse"
    assert payload["evidence_reuse"]["evidence_hits"] == 5


def test_repeatability_reads_counter_aliases_from_managed_run(tmp_path) -> None:
    run1 = _write_run(tmp_path / "run1", selected_candidate_id="a", use_counter_aliases=True, cold=0, measurements=0, hits=2)
    run2 = _write_run(tmp_path / "run2", selected_candidate_id="a", use_counter_aliases=True, cold=0, measurements=0, hits=3)

    payload = analyze_repeatability([run1, run2])

    assert payload["evidence_reuse"]["reuse_classification"] == "strong_reuse"
    assert payload["evidence_reuse"]["cold_launches"] == 0
    assert payload["evidence_reuse"]["workload_measurements"] == 0
    assert payload["evidence_reuse"]["evidence_hits"] == 5


def test_repeatability_cli_writes_artifacts(tmp_path, capsys) -> None:
    run1 = _write_run(tmp_path / "run1", selected_candidate_id="a")
    run2 = _write_run(tmp_path / "run2", selected_candidate_id="a")

    main(["repeatability", str(run1), str(run2)])
    output = capsys.readouterr().out

    assert "Recommendation repeatability" in output
    assert (tmp_path / "recommendation_repeatability.json").exists()
    assert (tmp_path / "recommendation_repeatability.txt").exists()


def test_repeatability_write_artifacts_payload_includes_paths(tmp_path) -> None:
    run1 = _write_run(tmp_path / "run1", selected_candidate_id="a")
    run2 = _write_run(tmp_path / "run2", selected_candidate_id="a")

    payload = write_repeatability_artifacts([run1, run2], output_dir=tmp_path / "out")

    assert payload["artifacts"]["recommendation_repeatability_json"].endswith("recommendation_repeatability.json")
    assert payload["artifacts"]["recommendation_repeatability_txt"].endswith("recommendation_repeatability.txt")
    saved = json.loads((tmp_path / "out" / "recommendation_repeatability.json").read_text(encoding="utf-8"))
    assert saved["artifacts"] == payload["artifacts"]


def _write_run(
    run_dir,
    *,
    selected_candidate_id: str,
    selected_overrides: dict[str, object] | None = None,
    candidate_table: list[dict[str, object]] | None = None,
    metrics: dict[str, object] | None = None,
    cold: int = 1,
    measurements: int = 1,
    hits: int = 0,
    use_counter_aliases: bool = False,
):
    run_dir.mkdir()
    selected = _candidate(selected_candidate_id)
    selected.update(selected_overrides or {})
    metric_payload = {
        "throughput_tokens_per_sec": 100.0,
        "p95_latency_ms": 10.0,
        "average_power_w": 200.0,
        "joules_per_token": 0.1,
        "tokens_per_watt": 5.0,
    }
    metric_payload.update(metrics or {})
    table = candidate_table or [
        dict(selected),
        _candidate("runner", max_model_len=4096),
        _candidate("third", max_model_len=8192),
    ]
    _write_json(
        run_dir / "recommendation_summary.json",
        {
            "recommended_command": f"vllm serve model-path --candidate {selected_candidate_id}",
            "selected": selected,
            "metrics": metric_payload,
        },
    )
    _write_json(
        run_dir / "managed_recommendation.json",
        {
            "recommendation": {
                "recommended_candidate_id": selected_candidate_id,
                "selected_serve_command": f"vllm serve model-path --candidate {selected_candidate_id}",
                "candidate_table": table,
            }
        },
    )
    managed_run = (
        {
            "cold_launches": cold,
            "workload_measurements": measurements,
            "evidence_hits": hits,
        }
        if use_counter_aliases
        else {
            "cold_launch_count": cold,
            "workload_measurement_count": measurements,
            "evidence_hit_candidate_count": hits,
        }
    )
    _write_json(run_dir / "managed_run.json", managed_run)
    _write_json(run_dir / "managed_pareto_frontier.json", table[:2])
    return run_dir


def _candidate(candidate_id: str, *, max_model_len: int = 2048) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "backend": "vllm",
        "model": "model-path",
        "dtype": "bfloat16",
        "quantization": "none",
        "max_model_len": max_model_len,
        "gpu_memory_utilization": 0.9,
        "max_num_seqs": 1,
        "tensor_parallel_size": 1,
        "benchmark_concurrency": 1,
        "score": 1.0,
    }


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
