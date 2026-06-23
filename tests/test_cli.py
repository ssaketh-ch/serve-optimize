import json

import pytest

from serve_optimize.cli import DEFAULT_MODEL, main


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
