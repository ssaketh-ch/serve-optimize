import csv
import json
import os
import subprocess

import pytest

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
    performance_cells = [cell for cell in payload["cells"] if cell["goal"] == "performance"]
    balanced_cells = [cell for cell in payload["cells"] if cell["goal"] == "balanced"]
    assert all(cell["command"][cell["command"].index("--limit") + 1] == "8" for cell in performance_cells)
    assert all(cell["command"][cell["command"].index("--limit") + 1] == "5" for cell in balanced_cells)


def test_benchmark_matrix_preserves_zero_warmup_and_idle_settings() -> None:
    payload = build_benchmark_matrix_plan(
        BenchmarkMatrixRequest(
            stages=["stage1"],
            warmup_requests=0,
            idle_baseline_seconds=0,
            steady_state_seconds=None,
        )
    )
    command = payload["cells"][0]["command"]

    assert command[command.index("--warmup-requests") + 1] == "0"
    assert command[command.index("--idle-baseline-seconds") + 1] == "0"


def test_benchmark_matrix_rejects_invalid_runtime_settings() -> None:
    with pytest.raises(ValueError, match="warmup requests"):
        build_benchmark_matrix_plan(BenchmarkMatrixRequest(stages=["stage1"], warmup_requests=-1))


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


def test_stage2_deduplicates_models_and_runs_real_chat_on_both_backends() -> None:
    payload = build_benchmark_matrix_plan(
        BenchmarkMatrixRequest(
            stages=["stage2"],
            real_chat_manifest="configs/workloads/oasst1-root-en-64.json",
            repeats=2,
        )
    )
    synthetic_cells = [cell for cell in payload["cells"] if cell["scenario"] == "single_gpu_generality"]
    synthetic_keys = {
        (cell["model"], cell["backend"], cell["goal"], cell["workload_profile"], cell["repeat"])
        for cell in synthetic_cells
    }
    real_chat_cells = [cell for cell in payload["cells"] if cell["scenario"] == "real_chat_trace_permitted_dataset"]

    assert len(synthetic_keys) == len(synthetic_cells)
    assert {cell["backend"] for cell in real_chat_cells} == {"vllm", "sglang"}
    assert {cell["model"] for cell in real_chat_cells} == {"Qwen/Qwen3-0.6B", "Qwen/Qwen2.5-7B-Instruct"}
    assert {cell["repeat"] for cell in real_chat_cells} == {1, 2}
    assert len(real_chat_cells) == 8
    repeat_one = next(cell for cell in real_chat_cells if cell["repeat"] == 1)
    repeat_two = next(cell for cell in real_chat_cells if cell["repeat"] == 2)
    assert repeat_one["command"][repeat_one["command"].index("--evidence-db") + 1].endswith("evidence.sqlite")
    assert repeat_two["command"][repeat_two["command"].index("--evidence-db") + 1].endswith("evidence-repeat-02.sqlite")


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
    assert all("--stream" in cell["command"] for cell in attach_cells)


def test_stage4_runnable_cells_honor_repeat_count() -> None:
    payload = build_benchmark_matrix_plan(
        BenchmarkMatrixRequest(
            stages=["stage4"],
            attach_base_url="http://127.0.0.1:8000/v1",
            repeats=2,
        )
    )
    runnable_cells = [cell for cell in payload["cells"] if cell["runnable"] is True]
    scenario_backends = {(cell["scenario"], cell["backend"]) for cell in runnable_cells}

    for scenario, backend in scenario_backends:
        repeats = {
            cell["repeat"]
            for cell in runnable_cells
            if cell["scenario"] == scenario and cell["backend"] == backend
        }
        assert repeats == {1, 2}


def test_stage4_attach_cells_use_distinct_outputs_and_matrix_controls() -> None:
    payload = build_benchmark_matrix_plan(
        BenchmarkMatrixRequest(
            stages=["stage4"],
            attach_base_url="http://127.0.0.1:8000/v1",
            warmup_requests=7,
            steady_state_seconds=23.0,
            idle_baseline_seconds=11.0,
            soak_seconds=19.0,
            stream=True,
        )
    )

    attach_cells = [cell for cell in payload["cells"] if cell["mode"] == "attach"]
    assert len({cell["out_dir"] for cell in attach_cells}) == 2
    for cell in attach_cells:
        command = cell["command"]
        assert command[command.index("--out") + 1] == cell["out_dir"]
        assert command[command.index("--top-k") + 1] == "5"
        assert command[command.index("--warmup-requests") + 1] == "7"
        assert command[command.index("--steady-state-seconds") + 1] == "23"
        assert command[command.index("--idle-baseline-seconds") + 1] == "11"
        assert command[command.index("--soak-seconds") + 1] == "19"
        assert "--stream" in command


def test_stage4_high_concurrency_records_actual_managed_sweep() -> None:
    payload = build_benchmark_matrix_plan(BenchmarkMatrixRequest(stages=["stage4"]))

    cell = next(cell for cell in payload["cells"] if cell["scenario"] == "high_concurrency_saturation")
    evidence = cell["load_sufficiency"]

    assert cell["runnable"] is True
    assert cell["command"][cell["command"].index("--limit") + 1] == "8"
    assert evidence["method"] == "managed_performance_candidate_sweep"
    assert evidence["concurrency_levels"] == [16, 32, 64]
    assert evidence["minimum_concurrency_levels"] == 3
    assert "candidate generation" in evidence["source"]
    assert "not claim" in evidence["not_claimed"]


def test_matrix_uses_backend_specific_cli_paths() -> None:
    payload = build_benchmark_matrix_plan(
        BenchmarkMatrixRequest(
            stages=["stage1"],
            vllm_cli="/envs/vllm/bin/serve-optimize",
            sglang_cli="/envs/sglang/bin/serve-optimize",
        )
    )

    for cell in payload["cells"]:
        expected = f"/envs/{cell['backend']}/bin/serve-optimize"
        assert cell["command"][0] == expected
        assert cell["shell_command"].startswith(expected)


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
