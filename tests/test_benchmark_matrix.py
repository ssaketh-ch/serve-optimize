import csv
import json
import os
import subprocess

from serve_optimize.benchmark_matrix import (
    BenchmarkMatrixRequest,
    build_benchmark_matrix_plan,
    write_benchmark_matrix_artifacts,
)
from serve_optimize.cli import main
from serve_optimize.workloads import load_workload_profile, workload_profile_choices


def test_stage1_benchmark_matrix_matches_journal_shape() -> None:
    payload = build_benchmark_matrix_plan(BenchmarkMatrixRequest(stages=["stage1"], telemetry="none"))

    assert payload["schema_version"] == "benchmark-matrix-plan/v1"
    assert payload["summary"]["cell_count"] == 36
    assert payload["summary"]["runnable_cell_count"] == 36
    assert payload["stages"][0]["stage_id"] == "stage_1_sanity"
    assert "All summary fields are populated." in payload["stages"][0]["success_criteria"]

    models = {cell["model_class"] for cell in payload["cells"]}
    backends = {cell["backend"] for cell in payload["cells"]}
    workloads = {cell["workload_profile"] for cell in payload["cells"]}
    objective_labels = {cell["objective_label"] for cell in payload["cells"]}

    assert models == {"small_open_under_1b", "medium_open_7_to_8b"}
    assert backends == {"vllm", "sglang"}
    assert workloads == {"short", "medium", "long-prefill"}
    assert objective_labels == {"balanced", "throughput", "efficient"}
    assert "--idle-baseline-seconds" in payload["cells"][0]["command"]
    assert "--steady-state-seconds" in payload["cells"][0]["command"]
    assert "15" in payload["cells"][0]["command"]


def test_stage2_tracks_optional_and_prerequisite_cells() -> None:
    payload = build_benchmark_matrix_plan(
        BenchmarkMatrixRequest(stages=["stage2"], include_gated=False, include_optional_large=False)
    )

    families = {cell["model_family"] for cell in payload["cells"] if cell.get("model_family")}
    blocked = [cell for cell in payload["cells"] if cell["runnable"] is False]

    assert {"Qwen", "Mistral", "DeepSeek", "Granite"}.issubset(families)
    assert "Llama" not in families
    assert any(cell["scenario"] == "real_chat_trace_permitted_dataset" for cell in blocked)
    assert all("real-chat-manifest" in cell["prerequisite"] for cell in blocked)


def test_stage4_includes_both_modes_and_both_backends() -> None:
    payload = build_benchmark_matrix_plan(
        BenchmarkMatrixRequest(stages=["stage4"], attach_base_url="http://127.0.0.1:8000/v1")
    )

    modes = {cell["mode"] for cell in payload["cells"]}
    backends = {cell["backend"] for cell in payload["cells"] if cell.get("backend")}
    scenarios = {cell["scenario"] for cell in payload["cells"]}

    assert modes == {"attach", "managed", "manual"}
    assert {"vllm", "sglang"}.issubset(backends)
    assert "streaming_requests" in scenarios
    assert "slo_constrained_serving" in scenarios
    assert "backend_crash_or_out_of_memory_recovery" in scenarios
    attach_cells = [cell for cell in payload["cells"] if cell["mode"] == "attach"]
    assert all(cell["runnable"] is True for cell in attach_cells)


def test_benchmark_matrix_writes_artifacts_and_runner_continues(tmp_path) -> None:
    payload = write_benchmark_matrix_artifacts(
        BenchmarkMatrixRequest(stages=["stage1"], telemetry="none", output_root=str(tmp_path / "runs")),
        output_dir=tmp_path / "plan",
    )

    assert (tmp_path / "plan" / "benchmark_matrix_plan.json").exists()
    assert (tmp_path / "plan" / "benchmark_matrix_plan.md").exists()
    assert (tmp_path / "plan" / "benchmark_matrix.csv").exists()
    assert (tmp_path / "plan" / "benchmark_matrix_commands.sh").exists()
    assert (tmp_path / "plan" / "benchmark_matrix_stage_1_sanity_vllm.sh").exists()
    assert (tmp_path / "plan" / "benchmark_matrix_commands.sh").stat().st_mode & 0o111
    rows = list(csv.DictReader((tmp_path / "plan" / "benchmark_matrix.csv").open(encoding="utf-8")))
    assert len(rows) == payload["summary"]["cell_count"]
    saved = json.loads((tmp_path / "plan" / "benchmark_matrix_plan.json").read_text(encoding="utf-8"))
    assert saved["artifacts"]["benchmark_matrix_csv"].endswith("benchmark_matrix.csv")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    count_path = tmp_path / "count.txt"
    fake_command = fake_bin / "serve-optimize"
    fake_command.write_text(
        "#!/usr/bin/env bash\n"
        "count=0\n"
        "if [[ -f \"$COUNT_FILE\" ]]; then count=$(cat \"$COUNT_FILE\"); fi\n"
        "count=$((count + 1))\n"
        "printf '%s' \"$count\" >\"$COUNT_FILE\"\n"
        "if [[ $count -eq 1 ]]; then exit 1; fi\n",
        encoding="utf-8",
    )
    fake_command.chmod(0o755)
    env = dict(os.environ)
    env["COUNT_FILE"] = str(count_path)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    completed = subprocess.run(
        [str(tmp_path / "plan" / "benchmark_matrix_stage_1_sanity_vllm.sh")],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 1
    assert count_path.read_text(encoding="utf-8") == "18"
    assert "1 stage_1_sanity vllm benchmark cell(s) failed" in completed.stderr


def test_benchmark_matrix_cli_writes_plan(tmp_path, capsys) -> None:
    main(
        [
            "benchmark-matrix-plan",
            "--stage",
            "stage1",
            "--telemetry",
            "none",
            "--out",
            str(tmp_path / "plan"),
        ]
    )
    output = capsys.readouterr().out

    assert "Benchmark matrix plan" in output
    assert (tmp_path / "plan" / "benchmark_matrix_plan.json").exists()


def test_benchmark_workload_profiles_are_available() -> None:
    choices = workload_profile_choices()
    long_prefill = load_workload_profile(profile_name="long-prefill")
    code_generation = load_workload_profile(profile_name="code-generation")

    assert "long-prefill" in choices
    assert "code-generation" in choices
    assert long_prefill.input_tokens > long_prefill.output_tokens
    assert code_generation.dataset == "synthetic-code-generation"
