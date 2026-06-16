import json
from pathlib import Path

import pytest

from serve_optimize.aiconfig_parser import parse_aiconfigurator_best_configs
from serve_optimize.aiconfig_plans import candidate_to_endpoint_benchmark_plan, candidate_to_vllm_serve_plan
from serve_optimize.cli import main

REAL_AIC_CSV = Path(
    "/home/saketh-msc/h200-aiconfig-test/results/aiconfigurator/default_run/"
    "TinyLlama-1.1B-Chat_h200_sxm_vllm_isl512_osl128_ttft2000_tpot30_485541/agg/best_config_topn.csv"
)


def test_parse_real_aiconfigurator_csv() -> None:
    if not REAL_AIC_CSV.exists():
        pytest.skip("Real AIConfigurator CSV is not available in this workspace.")

    candidates = parse_aiconfigurator_best_configs(str(REAL_AIC_CSV), top_k=1)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.rank == 1
    assert candidate.model == "/home/saketh-msc/h200-aiconfig-test/models/TinyLlama-1.1B-Chat"
    assert candidate.backend == "vllm"
    assert candidate.backend_version == "0.19.0"
    assert candidate.system == "h200_sxm"
    assert candidate.isl == 512
    assert candidate.osl == 128
    assert candidate.concurrency == 512
    assert candidate.batch_size == 512
    assert candidate.global_batch_size == 512
    assert candidate.tp == 1
    assert candidate.pp == 1
    assert candidate.dp == 1
    assert candidate.gemm == "bfloat16"
    assert candidate.kvcache == "bfloat16"
    assert candidate.fmha == "bfloat16"
    assert candidate.predicted_request_latency_ms == pytest.approx(1496.106)
    assert candidate.predicted_tokens_s == pytest.approx(44568.232)
    assert candidate.predicted_tokens_s_per_gpu == pytest.approx(44568.232)
    assert candidate.predicted_tokens_s_per_user == pytest.approx(88.306)
    assert candidate.predicted_memory_gb == pytest.approx(12.574)
    assert candidate.predicted_power_w == pytest.approx(0.0)
    assert candidate.raw["tokens/s"] == "44568.232"


def test_top_k_and_numeric_parsing(tmp_path) -> None:
    path = tmp_path / "best_config_topn.csv"
    path.write_text(
        "model,isl,osl,prefix,concurrency,request_rate,bs,global_bs,ttft,tpot,request_latency,"
        "seq/s,tokens/s,tokens/s/gpu,tokens/s/user,tp,pp,dp,moe_tp,moe_ep,backend,version,system,memory,power_w\n"
        "m1,512,128,0,64,10.5,64,64,1.0,2.0,3.0,4.0,5.0,6.0,7.0,1,1,1,1,1,vllm,0.1,sys,8.5,9.5\n"
        "m2,256,32,0,8,2.5,8,8,1.0,2.0,3.0,4.0,5.0,6.0,7.0,1,1,1,1,1,vllm,0.1,sys,8.5,9.5\n",
        encoding="utf-8",
    )

    candidates = parse_aiconfigurator_best_configs(str(path), top_k=1)

    assert len(candidates) == 1
    assert candidates[0].rank == 1
    assert candidates[0].model == "m1"
    assert candidates[0].concurrency == 64
    assert candidates[0].request_rate == pytest.approx(10.5)
    assert candidates[0].predicted_tokens_s == pytest.approx(5.0)


def test_malformed_numeric_row_is_preserved(tmp_path) -> None:
    path = tmp_path / "best_config_topn.csv"
    path.write_text(
        "model,isl,osl,concurrency,tokens/s,backend\n"
        "m,not-an-int,128,also-bad,not-a-float,vllm\n",
        encoding="utf-8",
    )

    candidate = parse_aiconfigurator_best_configs(str(path))[0]

    assert candidate.model == "m"
    assert candidate.isl is None
    assert candidate.concurrency is None
    assert candidate.predicted_tokens_s is None
    assert candidate.raw["isl"] == "not-an-int"


def test_vllm_serve_plan_generation(tmp_path) -> None:
    candidate = parse_aiconfigurator_best_configs(str(_write_one_candidate(tmp_path)))[0]

    plan = candidate_to_vllm_serve_plan(candidate, host="0.0.0.0", port=9000, gpu_memory_utilization=0.5)

    assert plan.model == "model-path"
    assert plan.dtype == "bfloat16"
    assert plan.tensor_parallel_size == 2
    assert plan.pipeline_parallel_size == 1
    assert plan.max_model_len == 2048
    assert plan.command == [
        "vllm",
        "serve",
        "model-path",
        "--host",
        "0.0.0.0",
        "--port",
        "9000",
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        "2",
        "--max-model-len",
        "2048",
        "--gpu-memory-utilization",
        "0.50",
    ]
    assert "vllm serve model-path" in plan.shell_command


def test_endpoint_benchmark_plan_generation(tmp_path) -> None:
    candidate = parse_aiconfigurator_best_configs(str(_write_one_candidate(tmp_path)))[0]

    plan = candidate_to_endpoint_benchmark_plan(candidate, base_url="http://127.0.0.1:8080/v1")

    assert plan.model == "model-path"
    assert plan.concurrency == 16
    assert plan.num_requests == 128
    assert plan.max_tokens == 128
    assert plan.expected_input_tokens == 512
    assert plan.expected_output_tokens == 128


def test_plan_from_aic_cli_writes_artifacts(tmp_path) -> None:
    csv_path = _write_one_candidate(tmp_path)
    out_dir = tmp_path / "plans"

    main(
        [
            "plan-from-aic",
            "--best-config-csv",
            str(csv_path),
            "--top-k",
            "1",
            "--host",
            "127.0.0.1",
            "--port",
            "8080",
            "--base-url",
            "http://127.0.0.1:8080/v1",
            "--gpu-memory-utilization",
            "0.90",
            "--out",
            str(out_dir),
        ]
    )

    run_dirs = list(out_dir.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "candidates.jsonl").exists()
    assert (run_dir / "serve_plans.jsonl").exists()
    assert (run_dir / "benchmark_plans.jsonl").exists()
    assert (run_dir / "evaluation_plans.jsonl").exists()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["candidate_count"] == 1
    serve_row = json.loads((run_dir / "serve_plans.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert serve_row["command"][0:3] == ["vllm", "serve", "model-path"]


def test_console_script_entrypoint_is_packaged() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert '[project.scripts]' in pyproject
    assert 'serve-optimize = "serve_optimize.cli:main"' in pyproject


def _write_one_candidate(tmp_path: Path) -> Path:
    path = tmp_path / "best_config_topn.csv"
    path.write_text(
        "model,isl,osl,prefix,concurrency,request_rate,bs,global_bs,ttft,tpot,request_latency,"
        "seq/s,tokens/s,tokens/s/gpu,tokens/s/user,tp,pp,dp,moe_tp,moe_ep,parallel,gemm,kvcache,fmha,"
        "moe,comm,backend,version,system,memory,power_w\n"
        "model-path,512,128,0,16,4.0,16,16,57.0,11.0,1496.0,4.0,1024.0,1024.0,64.0,"
        "2,1,1,1,1,tp2pp1dp1,bfloat16,bfloat16,bfloat16,bfloat16,half,vllm,0.1,sys,12.5,0.0\n",
        encoding="utf-8",
    )
    return path
