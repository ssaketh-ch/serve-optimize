import csv
import json
import os
import subprocess

from serve_optimize.campaign_plan import CampaignPlanRequest, build_campaign_plan, write_campaign_plan_artifacts
from serve_optimize.cli import main


def test_campaign_plan_builds_matrix_and_commands() -> None:
    payload = build_campaign_plan(
        CampaignPlanRequest(
            models=["model-a", "model-b"],
            backends=["vllm"],
            goals=["balanced", "efficient"],
            workload_profiles=["short"],
            repeats=2,
            output_root="results/campaign",
            evidence_db="results/evidence.sqlite",
            soak_seconds=120,
            stream=True,
        )
    )

    assert payload["schema_version"] == "campaign-plan/v1"
    assert payload["summary"]["planned_run_count"] == 8
    first = payload["runs"][0]
    assert first["run_id"] == "001-vllm-balanced-short-r01"
    assert first["command"][:6] == ["serve-optimize", "managed-evaluate", "--backend", "vllm", "--model", "model-a"]
    assert "--evidence-db" in first["command"]
    assert "--soak-seconds" in first["command"]
    assert "--stream" in first["command"]
    assert "validate-campaign" in payload["post_commands"]["validate_campaign"]
    assert "results/campaign/*/*" in payload["post_commands"]["validate_campaign"]


def test_campaign_plan_writes_artifacts(tmp_path) -> None:
    payload = write_campaign_plan_artifacts(
        CampaignPlanRequest(
            models=["model-a"],
            backends=["vllm", "sglang"],
            goals=["balanced"],
            workload_profiles=["short", "mixed"],
            output_root=str(tmp_path / "managed"),
        ),
        output_dir=tmp_path / "plan",
    )

    assert payload["summary"]["planned_run_count"] == 4
    assert (tmp_path / "plan" / "campaign_plan.json").exists()
    assert (tmp_path / "plan" / "campaign_plan.txt").exists()
    assert (tmp_path / "plan" / "campaign_commands.sh").exists()
    assert (tmp_path / "plan" / "campaign_commands_vllm.sh").exists()
    assert (tmp_path / "plan" / "campaign_commands_sglang.sh").exists()
    assert (tmp_path / "plan" / "campaign_postprocess.sh").exists()
    assert (tmp_path / "plan" / "campaign_commands.sh").stat().st_mode & 0o111
    assert (tmp_path / "plan" / "campaign_commands_vllm.sh").stat().st_mode & 0o111
    dispatcher = (tmp_path / "plan" / "campaign_commands.sh").read_text(encoding="utf-8")
    assert "campaign_commands_vllm.sh" in dispatcher
    assert "campaign_commands_sglang.sh" in dispatcher
    vllm_commands = (tmp_path / "plan" / "campaign_commands_vllm.sh").read_text(encoding="utf-8")
    assert "--backend vllm" in vllm_commands
    assert "--backend sglang" not in vllm_commands
    assert "failures=$((failures + 1))" in vllm_commands
    postprocess = (tmp_path / "plan" / "campaign_postprocess.sh").read_text(encoding="utf-8")
    assert f"{tmp_path}/managed/*/*" in postprocess
    rows = list(csv.DictReader((tmp_path / "plan" / "campaign_matrix.csv").open(encoding="utf-8")))
    assert len(rows) == 4
    assert rows[0]["backend"] == "vllm"
    saved = json.loads((tmp_path / "plan" / "campaign_plan.json").read_text(encoding="utf-8"))
    assert saved["artifacts"]["campaign_matrix_csv"].endswith("campaign_matrix.csv")


def test_campaign_plan_cli_writes_plan(tmp_path, capsys) -> None:
    main(
        [
            "campaign-plan",
            "--model",
            "model-a",
            "--backend",
            "vllm",
            "--workload-profile",
            "short",
            "--out",
            str(tmp_path / "plan"),
        ]
    )
    output = capsys.readouterr().out

    assert "Campaign plan" in output
    assert (tmp_path / "plan" / "campaign_plan.json").exists()


def test_backend_campaign_runner_continues_after_failed_cell(tmp_path) -> None:
    write_campaign_plan_artifacts(
        CampaignPlanRequest(
            models=["model-a"],
            backends=["vllm"],
            goals=["balanced"],
            workload_profiles=["short", "mixed"],
            output_root=str(tmp_path / "managed"),
        ),
        output_dir=tmp_path / "plan",
    )
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
        [str(tmp_path / "plan" / "campaign_commands_vllm.sh")],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 1
    assert count_path.read_text(encoding="utf-8") == "2"
    assert "1 vllm campaign run(s) failed" in completed.stderr
