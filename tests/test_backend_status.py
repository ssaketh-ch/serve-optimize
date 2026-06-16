from importlib import metadata

from serve_optimize.backend_status import (
    INSTALLATION_PROFILES,
    check_installation_profile,
)
from serve_optimize.cli import main


def test_installation_profiles_are_explicit() -> None:
    assert INSTALLATION_PROFILES == ("core", "telemetry", "vllm", "sglang")


def test_vllm_profile_reports_version_drift(monkeypatch) -> None:
    versions = {
        "serve-optimize": "0.1.0",
        "vllm": "0.10.0",
        "torch": "2.7.1",
        "transformers": "5.3.0",
        "huggingface-hub": "0.36.2",
        "nvidia-ml-py": "13.610.43",
    }
    monkeypatch.setattr(metadata, "version", lambda name: versions[name])
    monkeypatch.setattr("serve_optimize.backend_status.shutil.which", lambda command: f"/env/bin/{command}")

    statuses = check_installation_profile("vllm")
    transformers = next(status for status in statuses if status.name == "transformers")

    assert transformers.available is False
    assert transformers.version == "5.3.0"
    assert transformers.reason == "expected 4.57.6"


def test_vllm_profile_does_not_accept_command_from_ambient_path(monkeypatch, tmp_path) -> None:
    versions = {
        "serve-optimize": "0.1.0",
        "vllm": "0.10.0",
        "torch": "2.7.1",
        "transformers": "4.57.6",
        "huggingface-hub": "0.36.2",
        "nvidia-ml-py": "13.610.43",
    }
    python = tmp_path / "profile" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.touch()
    monkeypatch.setattr(metadata, "version", lambda name: versions[name])
    monkeypatch.setattr("serve_optimize.backend_status.sys.executable", str(python))
    monkeypatch.setattr(
        "serve_optimize.backend_status.shutil.which",
        lambda command: f"/ambient/bin/{command}",
    )

    statuses = check_installation_profile("vllm")
    command = next(status for status in statuses if status.name == "cmd:vllm")

    assert command.available is False
    assert command.reason == f"not installed beside {python}"


def test_sglang_profile_requires_toolset_and_cuda_helper(monkeypatch) -> None:
    versions = {
        "serve-optimize": "0.1.0",
        "sglang": "0.5.10.post1",
        "sglang-kernel": "0.4.1",
        "torch": "2.9.1",
        "transformers": "5.3.0",
        "huggingface-hub": "1.19.0",
        "nvidia-ml-py": "13.610.43",
    }
    monkeypatch.setattr(metadata, "version", lambda name: versions[name])
    monkeypatch.setattr("serve_optimize.backend_status._sglang_runtime_status", lambda: _ok_status("sglang-runtime"))
    monkeypatch.setattr("serve_optimize.backend_status.shutil.which", lambda command: f"/usr/bin/{command}")
    monkeypatch.delenv("CUDA_HOME", raising=False)

    statuses = check_installation_profile("sglang")

    assert next(status for status in statuses if status.name == "compiler:gcc").available is False
    assert next(status for status in statuses if status.name == "cuda-toolkit").available is False


def test_doctor_profile_exits_when_requirement_is_missing(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "serve_optimize.cli.check_installation_profile",
        lambda _profile: [_missing_status("vllm")],
    )

    try:
        main(["doctor", "--profile", "vllm"])
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("doctor profile check should fail")

    assert "missing" in capsys.readouterr().out


def _ok_status(name: str):
    from serve_optimize.schemas import BackendStatus

    return BackendStatus(name=name, available=True)


def _missing_status(name: str):
    from serve_optimize.schemas import BackendStatus

    return BackendStatus(name=name, available=False, reason="missing")
