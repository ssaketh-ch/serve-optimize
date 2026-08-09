import csv
import json

from serve_optimize.cli import main
from serve_optimize.validation_campaign import analyze_validation_campaign, write_validation_campaign_artifacts


def test_validation_campaign_parses_valid_dirs_and_passes_quality(tmp_path) -> None:
    run1 = _write_run(tmp_path / "run1", selected_candidate_id="cfg-a")
    run2 = _write_run(tmp_path / "run2", selected_candidate_id="cfg-b")

    payload = analyze_validation_campaign([run1, run2])

    assert payload["schema_version"] == "validation-campaign/v1"
    assert payload["usable_run_count"] == 2
    assert payload["recommendation_quality"]["classification"] == "pass"
    assert payload["repeatability"]["stability_classification"] == "stable"
    assert payload["backend_coverage"]["classification"] == "vllm_only"


def test_validation_campaign_missing_files_warns(tmp_path) -> None:
    run = tmp_path / "missing"
    run.mkdir()

    payload = analyze_validation_campaign([run])

    assert payload["usable_run_count"] == 0
    assert payload["skipped_run_count"] == 1
    assert payload["recommendation_quality"]["classification"] == "fail"
    assert payload["warnings"]


def test_validation_campaign_writes_json_text_and_csv(tmp_path) -> None:
    run = _write_run(tmp_path / "run1", selected_candidate_id="cfg-a")

    payload = write_validation_campaign_artifacts([run], output_dir=tmp_path / "campaign")

    assert (tmp_path / "campaign" / "validation_campaign.json").exists()
    assert (tmp_path / "campaign" / "validation_campaign.txt").exists()
    assert (tmp_path / "campaign" / "validation_campaign_runs.csv").exists()
    saved = json.loads((tmp_path / "campaign" / "validation_campaign.json").read_text(encoding="utf-8"))
    assert saved["artifacts"] == payload["artifacts"]
    rows = list(csv.DictReader((tmp_path / "campaign" / "validation_campaign_runs.csv").open(encoding="utf-8")))
    assert rows[0]["selected_candidate_id"] == "cfg-a"
    assert rows[0]["backend_version"] == "0.24.0"
    assert rows[0]["output_tokens_per_sec"] == "80.0"
    assert rows[0]["joules_per_generated_token"] == "0.125"
    assert rows[0]["load_sufficiency_classification"] == "not_evaluated_non_throughput_goal"


def test_validation_campaign_does_not_alias_legacy_total_throughput_to_output(tmp_path) -> None:
    run = _write_run(tmp_path / "run1", selected_candidate_id="cfg-a")
    summary_path = run / "recommendation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["metrics"].pop("output_tokens_per_sec")
    _write_json(summary_path, summary)
    recommendation_path = run / "managed_recommendation.json"
    recommendation = json.loads(recommendation_path.read_text(encoding="utf-8"))
    recommendation["recommendation"]["candidate_table"][0].pop("output_tokens_s")
    _write_json(recommendation_path, recommendation)

    payload = analyze_validation_campaign([run])

    assert payload["runs"][0]["output_tokens_per_sec"] is None
    assert payload["runs"][0]["total_tokens_per_sec"] == 100.0


def test_validation_campaign_cli_help_and_command(tmp_path, capsys) -> None:
    run = _write_run(tmp_path / "run1", selected_candidate_id="cfg-a")

    main(["validate-campaign", str(run), "--out", str(tmp_path / "campaign")])
    output = capsys.readouterr().out

    assert "Validation campaign" in output
    assert (tmp_path / "campaign" / "validation_campaign.json").exists()


def test_validation_campaign_prior_only_selected_fails(tmp_path) -> None:
    run = _write_run(tmp_path / "run1", selected_candidate_id="cfg-a", selected_source="aiconfigurator_prior")

    payload = analyze_validation_campaign([run])

    assert payload["recommendation_quality"]["classification"] == "fail"
    assert any("not measured or exact" in warning for warning in payload["warnings"])


def test_validation_campaign_stale_or_near_exact_evidence_fails(tmp_path) -> None:
    run = _write_run(
        tmp_path / "run1",
        selected_candidate_id="cfg-a",
        evidence_decisions=[
            {
                "candidate_id": "cfg-a",
                "classification": "near_compatible",
                "used_as_exact": True,
                "used_as_prior": False,
            }
        ],
    )

    payload = analyze_validation_campaign([run])

    assert payload["evidence_reuse"]["classification"] == "fail"
    assert payload["evidence_reuse"]["invalid_exact_decisions"]


def test_validation_campaign_aiconfigurator_local_gpu_fails(tmp_path) -> None:
    run = _write_run(
        tmp_path / "run1",
        selected_candidate_id="cfg-a",
        synthesis_records=[
            {
                "candidate_id": "cfg-synth",
                "status": "selected_for_evaluation",
                "aiconfigurator_system_key": "local_gpu",
            }
        ],
    )

    payload = analyze_validation_campaign([run])

    assert payload["aiconfigurator_synthesis"]["classification"] == "fail"
    assert payload["aiconfigurator_synthesis"]["local_gpu_runs"] == [str(run)]


def test_validation_campaign_safe_baseline_missing_needs_review(tmp_path) -> None:
    run = _write_run(tmp_path / "run1", selected_candidate_id="cfg-a", candidate_source_counts={"capability_aware": 1})

    payload = analyze_validation_campaign([run])

    assert payload["candidate_sources"]["classification"] == "needs_review"
    assert str(run) in payload["candidate_sources"]["safe_baseline_missing_runs"]


def test_validation_campaign_telemetry_classifications(tmp_path) -> None:
    good = _write_run(tmp_path / "good", selected_candidate_id="cfg-a", telemetry_quality="good")
    poor = _write_run(tmp_path / "poor", selected_candidate_id="cfg-b", telemetry_quality="poor", average_power_w=None)

    payload = analyze_validation_campaign([good, poor])

    assert payload["telemetry_quality"]["classification"] == "mixed"
    assert payload["telemetry_quality"]["counts"]["poor"] == 1


def test_validation_campaign_repeatability_unstable(tmp_path) -> None:
    run1 = _write_run(tmp_path / "run1", selected_candidate_id="cfg-a", max_model_len=2048)
    run2 = _write_run(tmp_path / "run2", selected_candidate_id="cfg-b", max_model_len=4096)

    payload = analyze_validation_campaign([run1, run2])

    assert payload["repeatability"]["stability_classification"] == "unstable"


def test_validation_campaign_score_ratio_uses_best_candidate(tmp_path) -> None:
    run = _write_run(tmp_path / "run1", selected_candidate_id="cfg-a", selected_best=False)

    payload = analyze_validation_campaign([run])

    assert payload["recommendation_quality"]["classification"] == "needs_review"
    assert payload["recommendation_quality"]["selected_score_ratio_to_best"]["min"] == 0.5


def test_validation_campaign_workload_prefix_caching_default_flagged(tmp_path) -> None:
    run = _write_run(
        tmp_path / "run1",
        selected_candidate_id="cfg-a",
        enable_prefix_caching=True,
        workload_profile={"profile_name": "default", "prefix_reuse_expected": False},
    )

    payload = analyze_validation_campaign([run])

    assert payload["workload_coverage"]["classification"] == "needs_review"
    assert str(run) in payload["workload_coverage"]["prefix_caching_without_reuse_runs"]


def test_validation_campaign_backend_detects_sglang_success(tmp_path) -> None:
    run = _write_run(tmp_path / "run1", selected_candidate_id="cfg-a", backend="sglang")

    payload = analyze_validation_campaign([run])

    assert payload["backend_coverage"]["classification"] == "sglang_present"
    assert payload["backend_coverage"]["successful_sglang_run_count"] == 1
    assert not payload["backend_coverage"]["failures"]


def _write_run(
    run_dir,
    *,
    selected_candidate_id: str,
    backend: str = "vllm",
    selected_source: str = "managed_measured",
    selected_status: str = "eligible",
    selected_best: bool = True,
    selected_pareto: bool = True,
    max_model_len: int = 2048,
    telemetry_quality: str = "good",
    average_power_w: float | None = 200.0,
    evidence_decisions: list[dict[str, object]] | None = None,
    synthesis_records: list[dict[str, object]] | None = None,
    candidate_source_counts: dict[str, int] | None = None,
    enable_prefix_caching: bool | None = None,
    workload_profile: dict[str, object] | None = None,
):
    run_dir.mkdir()
    candidate = _candidate(
        selected_candidate_id,
        backend=backend,
        source=selected_source,
        status=selected_status,
        pareto=selected_pareto,
        max_model_len=max_model_len,
        telemetry_quality=telemetry_quality,
        average_power_w=average_power_w,
        enable_prefix_caching=enable_prefix_caching,
    )
    other = _candidate(
        "cfg-other",
        backend=backend,
        source="managed_measured",
        score=0.5 if selected_best else 2.0,
        max_model_len=max_model_len + 512,
    )
    candidate_table = [candidate, other]
    profile = workload_profile or {"profile_name": "default", "prefix_reuse_expected": False}
    selected = {
        "candidate_id": selected_candidate_id,
        "backend": backend,
        "model": "model-path",
        "dtype": "bf16",
        "quantization": "none",
        "max_model_len": max_model_len,
        "gpu_memory_utilization": 0.9,
        "max_num_seqs": 1,
        "tensor_parallel_size": 1,
        "benchmark_concurrency": 1,
    }
    if enable_prefix_caching is not None:
        selected["enable_prefix_caching"] = enable_prefix_caching
    _write_json(
        run_dir / "recommendation_summary.json",
        {
            "status": "success",
            "goal": "balanced",
            "recommended_command": _command_for_backend(backend, selected_candidate_id),
            "selected": selected,
            "metrics": {
                "throughput_tokens_per_sec": candidate["total_tokens_s"],
                "output_tokens_per_sec": candidate["output_tokens_s"],
                "total_tokens_per_sec": candidate["total_tokens_s"],
                "request_rate_req_s": candidate["request_rate_req_s"],
                "p95_latency_ms": candidate["p95_latency_s"] * 1000.0,
                "p95_ttft_ms": candidate["p95_ttft_ms"],
                "p95_tpot_ms": candidate["p95_tpot_ms"],
                "average_power_w": average_power_w,
                "joules_per_token": candidate["joules_per_token"],
                "joules_per_generated_token": candidate["joules_per_generated_token"],
                "tokens_per_joule": candidate["tokens_per_joule"],
                "tokens_per_watt": candidate["tokens_per_second_per_watt"],
                "energy_accounting": "idle_subtracted",
                "failed_requests": 0,
            },
            "evaluated_set_fidelity": {
                "scope": "evaluated_candidates_only",
                "selected_rank": 1 if selected_best else 2,
                "selected_score_over_best_score": 1.0 if selected_best else 0.5,
                "selected_is_best_evaluated": selected_best,
                "selected_is_pareto_optimal": selected_pareto,
                "valid_candidate_count": len(candidate_table),
                "pareto_candidate_count": 1,
            },
        },
    )
    _write_json(
        run_dir / "managed_recommendation.json",
        {
            "schema_version": "managed-recommendation/v1",
            "status": "success",
            "selected_source": selected_source,
            "recommendation_quality_audit": {
                "scope": "evaluated_candidates_only",
                "selected_candidate_id": selected_candidate_id,
                "selected_score": candidate["score"],
                "selected_rank": 1 if selected_best else 2,
                "selected_is_best_evaluated": selected_best,
                "selected_is_pareto_optimal": selected_pareto,
                "valid_candidate_count": len(candidate_table),
                "pareto_candidate_count": 1,
            },
            "recommendation": {
                "recommended_candidate_id": selected_candidate_id,
                "selected_serve_command": _command_for_backend(backend, selected_candidate_id),
                "candidate_table": candidate_table,
                "evaluated_set_fidelity": {
                    "scope": "evaluated_candidates_only",
                    "selected_rank": 1 if selected_best else 2,
                    "selected_is_best_evaluated": selected_best,
                    "selected_is_pareto_optimal": selected_pareto,
                    "valid_candidate_count": len(candidate_table),
                    "pareto_candidate_count": 1,
                },
            },
        },
    )
    _write_json(
        run_dir / "managed_run.json",
        {
            "status": "success",
            "backend": backend,
            "goal": "balanced",
            "completed_candidate_count": len(candidate_table),
            "candidate_source_counts": candidate_source_counts or {"safe_baseline": 1, "capability_aware": 1},
            "workload_profile": profile,
            "runtime_environment": {
                "backend_version": "0.24.0" if backend == "vllm" else "0.5.13.post1",
                "python_version": "3.10.12",
                "torch_version": "2.11.0+cu130",
                "cuda_runtime_version": "13.0",
                "gpu_driver_version": "580.65.06",
                "environment_fingerprint": f"runtime-{backend}",
            },
            "client_saturation": {"classification": "not_client_limited"},
            "load_sufficiency": {"classification": "not_evaluated_non_throughput_goal"},
        },
    )
    _write_json(run_dir / "managed_pareto_frontier.json", [candidate])
    _write_json(
        run_dir / "candidate_synthesis.json",
        {
            "schema_version": "candidate-synthesis/v1",
            "candidate_records": synthesis_records or [],
            "provider_results": [],
        },
    )
    if backend == "sglang":
        _write_json(
            run_dir / "sglang_argument_capabilities.json",
            {"schema_version": "sglang-argument-capabilities/v1", "backend": "sglang", "help_hash": "hash-sglang", "detection_status": "success"},
        )
    else:
        _write_json(
            run_dir / "vllm_argument_capabilities.json",
            {"schema_version": "vllm-argument-capabilities/v1", "help_hash": "hash1"},
        )
    _write_jsonl(
        run_dir / "evidence_decisions.jsonl",
        evidence_decisions
        or [
            {
                "candidate_id": selected_candidate_id,
                "classification": "exact_fresh",
                "used_as_exact": selected_source == "managed_evidence_hit",
                "used_as_prior": False,
            }
        ],
    )
    rendered_config = dict(selected)
    rendered_config["id"] = selected_candidate_id
    rendered_config["model_id"] = "model-path"
    rendered_config["max_context_tokens"] = max_model_len
    rendered_config["max_batch_size"] = 1
    if enable_prefix_caching is not None:
        rendered_config["enable_prefix_caching"] = enable_prefix_caching
    _write_jsonl(
        run_dir / "rendered_launch_configs.jsonl",
        [{"canonical_config_id": selected_candidate_id, "canonical_config": rendered_config}],
    )
    return run_dir


def _candidate(
    candidate_id: str,
    *,
    backend: str,
    source: str,
    status: str = "eligible",
    score: float = 1.0,
    pareto: bool = True,
    max_model_len: int = 2048,
    telemetry_quality: str = "good",
    average_power_w: float | None = 200.0,
    enable_prefix_caching: bool | None = None,
) -> dict[str, object]:
    row = {
        "candidate_id": candidate_id,
        "backend": backend,
        "model": "model-path",
        "dtype": "bf16",
        "quantization": "none",
        "max_model_len": max_model_len,
        "gpu_memory_utilization": 0.9,
        "max_num_seqs": 1,
        "tensor_parallel_size": 1,
        "benchmark_concurrency": 1,
        "source": source,
        "status": status,
        "score": score,
        "pareto_optimal": pareto,
        "total_tokens_s": 100.0,
        "output_tokens_s": 80.0,
        "request_rate_req_s": 2.0,
        "p95_latency_s": 0.01,
        "p95_ttft_ms": 4.0,
        "p95_tpot_ms": 1.5,
        "average_power_watts": average_power_w,
        "joules_per_token": 0.1,
        "joules_per_generated_token": 0.125,
        "tokens_per_joule": 8.0,
        "tokens_per_second_per_watt": 5.0,
        "energy_accounting": "idle_subtracted",
        "failed_requests": 0,
        "telemetry_quality": telemetry_quality,
    }
    if enable_prefix_caching is not None:
        row["enable_prefix_caching"] = enable_prefix_caching
    return row


def _command_for_backend(backend: str, candidate_id: str) -> str:
    if backend == "sglang":
        return f"python -m sglang.launch_server --model-path model-path --candidate {candidate_id}"
    return f"vllm serve model-path --candidate {candidate_id}"


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path, rows) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
