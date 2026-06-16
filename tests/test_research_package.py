import json
from pathlib import Path

from serve_optimize.research_package import build_research_package, write_research_package_artifacts


def test_research_package_writes_manifest_and_tables(tmp_path) -> None:
    run_dir = _write_run(tmp_path / "run-a", backend="vllm", workload_profile="mixed")

    payload = write_research_package_artifacts([run_dir], output_dir=tmp_path / "package")

    assert payload["schema_version"] == "research-package/v1"
    assert payload["summary"]["usable_run_count"] == 1
    assert payload["coverage"]["backends"] == ["vllm"]
    assert payload["coverage"]["workload_profiles"] == ["mixed"]
    assert (tmp_path / "package" / "research_package.json").exists()
    assert (tmp_path / "package" / "methodology.md").exists()
    assert (tmp_path / "package" / "runs.csv").exists()
    assert (tmp_path / "package" / "coverage.csv").exists()


def test_research_package_reports_supplied_coverage_only(tmp_path) -> None:
    runs = [
        _write_run(tmp_path / "run-vllm", backend="vllm", workload_profile="short"),
        _write_run(tmp_path / "run-sglang", backend="sglang", workload_profile="long"),
    ]

    payload = build_research_package(runs)

    assert payload["coverage"]["backends"] == ["sglang", "vllm"]
    assert payload["coverage"]["workload_profiles"] == ["long", "short"]
    assert payload["summary"]["backend_count"] == 2
    assert payload["summary"]["workload_profile_count"] == 2
    assert "Do not infer coverage" in payload["methodology"][-1]


def _write_run(run_dir: Path, *, backend: str, workload_profile: str) -> Path:
    run_dir.mkdir(parents=True)
    candidate_id = f"cfg-{backend}"
    selected = {
        "candidate_id": candidate_id,
        "backend": backend,
        "model": "model-path",
        "dtype": "bf16",
        "quantization": "none",
        "max_model_len": 2048,
        "gpu_memory_utilization": 0.9,
        "max_num_seqs": 1,
        "tensor_parallel_size": 1,
        "benchmark_concurrency": 1,
    }
    candidate = {
        **selected,
        "source": "managed_measured",
        "status": "eligible",
        "score": 1.0,
        "pareto_optimal": True,
        "total_tokens_s": 100.0,
        "p95_latency_s": 0.01,
        "average_power_watts": 200.0,
        "joules_per_token": 0.1,
        "tokens_per_second_per_watt": 5.0,
        "failed_requests": 0,
        "telemetry_quality": "good",
    }
    _write_json(
        run_dir / "managed_run.json",
        {
            "status": "completed",
            "backend": backend,
            "goal": "balanced",
            "completed_candidate_count": 1,
            "candidate_source_counts": {"safe_baseline": 1},
            "workload_profile": {"profile_name": workload_profile},
            "backend_metadata": {"argument_capabilities_help_hash": f"hash-{backend}"},
        },
    )
    _write_json(
        run_dir / "managed_recommendation.json",
        {
            "status": "success",
            "selected_source": "managed_measured",
            "recommendation_quality_audit": {
                "selected_candidate_id": candidate_id,
                "selected_score": 1.0,
                "selected_rank": 1,
                "selected_is_best_evaluated": True,
                "selected_is_pareto_optimal": True,
                "valid_candidate_count": 1,
                "pareto_candidate_count": 1,
            },
            "recommendation": {
                "recommended_candidate_id": candidate_id,
                "goal": "balanced",
                "selected_serve_command": f"{backend} serve model-path",
                "candidate_table": [candidate],
                "evaluated_set_fidelity": {
                    "selected_rank": 1,
                    "selected_is_best_evaluated": True,
                    "selected_is_pareto_optimal": True,
                    "valid_candidate_count": 1,
                    "pareto_candidate_count": 1,
                },
            },
        },
    )
    _write_json(
        run_dir / "recommendation_summary.json",
        {
            "status": "success",
            "goal": "balanced",
            "selected": selected,
            "recommended_command": f"{backend} serve model-path",
            "metrics": {
                "throughput_tokens_per_sec": 100.0,
                "p95_latency_ms": 10.0,
                "average_power_w": 200.0,
                "joules_per_token": 0.1,
                "tokens_per_watt": 5.0,
                "failed_requests": 0,
            },
            "evaluated_set_fidelity": {
                "selected_rank": 1,
                "selected_is_best_evaluated": True,
                "selected_is_pareto_optimal": True,
                "valid_candidate_count": 1,
                "pareto_candidate_count": 1,
            },
        },
    )
    _write_json(run_dir / "managed_pareto_frontier.json", [candidate])
    _write_json(run_dir / "candidate_synthesis.json", {"summary": {"candidate_count": 0}})
    _write_json(run_dir / f"{backend}_argument_capabilities.json", {"help_hash": f"hash-{backend}"})
    _write_jsonl(run_dir / "evidence_decisions.jsonl", [{"candidate_id": candidate_id, "classification": "exact_fresh"}])
    _write_jsonl(run_dir / "rendered_launch_configs.jsonl", [{"canonical_config_id": candidate_id, "canonical_config": selected}])
    return run_dir


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[object]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
