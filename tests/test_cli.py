import json
from types import SimpleNamespace

import pytest

from serve_optimize.aiconfigurator_bridge import AIConfiguratorRun
from serve_optimize.cli import DEFAULT_MODEL, main
from serve_optimize.endpoint_benchmark import summarize_requests
from serve_optimize.schemas import RequestRecord


def test_optimize_help_hides_advanced_flags_by_default(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["optimize", "--help"])

    output = capsys.readouterr().out

    assert "--backend" in output
    assert "--workload-profile" in output
    assert "--startup-timeout" not in output
    assert "--evidence-db" not in output


def test_verbose_help_shows_advanced_flags(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["optimize", "--verbose-help"])

    output = capsys.readouterr().out

    assert "--startup-timeout" in output
    assert "--cooldown-seconds" in output
    assert "--evidence-db" in output
    assert "--cooldown " not in output


def test_campaign_plan_defaults_model(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    main(
        [
            "campaign-plan",
            "--backend",
            "vllm",
            "--workload-profile",
            "short",
            "--out",
            str(tmp_path / "plan"),
        ]
    )
    capsys.readouterr()
    payload = json.loads((tmp_path / "plan" / "campaign_plan.json").read_text(encoding="utf-8"))

    assert payload["request"]["models"] == [DEFAULT_MODEL]


def test_aiconfig_subprocess_failure_returns_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "serve_optimize.cli.run_aiconfigurator",
        lambda **_kwargs: AIConfiguratorRun(
            command=["aiconfigurator"],
            returncode=7,
            stdout="",
            stderr="invalid configuration",
        ),
    )

    with pytest.raises(SystemExit) as exc:
        main(["aiconfig"])

    assert exc.value.code == 7


def test_endpoint_bench_all_failed_requests_returns_nonzero(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    summary = summarize_requests(
        "failed-endpoint",
        [RequestRecord(0, 0.0, 0.1, 0.1, "error", error="unavailable")],
        wall_time_s=0.1,
    )
    monkeypatch.setattr("serve_optimize.cli.detect_hardware", lambda: None)
    monkeypatch.setattr(
        "serve_optimize.cli.run_endpoint_benchmark",
        lambda **_kwargs: SimpleNamespace(run_dir=tmp_path, summary=summary, comparison=None),
    )

    with pytest.raises(SystemExit) as exc:
        main(["endpoint-bench", "model-path", "--out", str(tmp_path)])

    assert exc.value.code == 1


def test_telemetry_check_unavailable_returns_nonzero(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "serve_optimize.cli.run_telemetry_check",
        lambda **_kwargs: SimpleNamespace(
            run_dir=tmp_path,
            summary=SimpleNamespace(telemetry_available=False),
        ),
    )
    monkeypatch.setattr("serve_optimize.cli.RichTelemetryCheckReporter.render", lambda self, **kwargs: None)

    with pytest.raises(SystemExit) as exc:
        main(["telemetry-check", "--duration", "0.01", "--out", str(tmp_path)])

    assert exc.value.code == 1


def test_missing_workload_manifest_returns_clean_error(tmp_path) -> None:
    missing = tmp_path / "missing.json"

    with pytest.raises(SystemExit) as exc:
        main(["optimize", "model_path", "--dry-run", "--workload-manifest", str(missing)])

    assert str(exc.value.code).startswith("Could not read workload manifest")


def test_missing_prompt_file_returns_clean_error(tmp_path) -> None:
    missing = tmp_path / "missing.txt"

    with pytest.raises(SystemExit) as exc:
        main(["endpoint-bench", "model_path", "--prompt-file", str(missing)])

    assert str(exc.value.code).startswith("Could not read prompt file")


def test_missing_evaluation_plan_returns_clean_error(tmp_path) -> None:
    missing = tmp_path / "missing_plan"

    with pytest.raises(SystemExit) as exc:
        main(["run-evaluation-plan", "--plan-dir", str(missing)])

    assert str(exc.value.code).startswith("Could not load evaluation plan")
