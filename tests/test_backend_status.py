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
        "vllm": "0.23.0",
        "torch": "2.11.0",
        "transformers": "5.12.1",
        "huggingface-hub": "1.17.0",
        "nvidia-ml-py": "13.610.43",
    }
    monkeypatch.setattr(metadata, "version", lambda name: versions[name])
    monkeypatch.setattr("serve_optimize.backend_status.shutil.which", lambda command: f"/env/bin/{command}")

    statuses = check_installation_profile("vllm")
    transformers = next(status for status in statuses if status.name == "transformers")

    assert transformers.available is False
    assert transformers.version == "5.12.1"
    assert transformers.reason == "expected 5.9.0"


def test_vllm_profile_accepts_torch_cuda_local_version(monkeypatch) -> None:
    versions = {
        "serve-optimize": "0.1.0",
        "vllm": "0.23.0",
        "torch": "2.11.0+cu129",
        "transformers": "5.9.0",
        "huggingface-hub": "1.17.0",
        "nvidia-ml-py": "13.610.43",
    }
    monkeypatch.setattr(metadata, "version", lambda name: versions[name])
    monkeypatch.setattr("serve_optimize.backend_status.shutil.which", lambda command: f"/env/bin/{command}")
    monkeypatch.setattr("serve_optimize.backend_status._python_headers_status", lambda: _ok_status("python-headers"))

    statuses = check_installation_profile("vllm")

    assert all(status.available for status in statuses)


def test_vllm_profile_accepts_command_from_active_path(monkeypatch, tmp_path) -> None:
    versions = {
        "serve-optimize": "0.1.0",
        "vllm": "0.23.0",
        "torch": "2.11.0",
        "transformers": "5.9.0",
        "huggingface-hub": "1.17.0",
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

    assert command.available is True
    assert command.command == "/ambient/bin/vllm"


def test_sglang_profile_checks_current_runtime_without_host_specific_toolchain(monkeypatch) -> None:
    versions = {
        "serve-optimize": "0.1.0",
        "sglang": "0.5.13.post1",
        "flash-attn-4": "4.0.0b18",
        "sglang-kernel": "0.4.3",
        "torch": "2.11.0",
        "transformers": "5.8.1",
        "huggingface-hub": "1.17.0",
        "nvidia-ml-py": "13.610.43",
    }
    monkeypatch.setattr(metadata, "version", lambda name: versions[name])
    monkeypatch.setattr("serve_optimize.backend_status._sglang_runtime_status", lambda: _ok_status("sglang-runtime"))
    monkeypatch.setattr("serve_optimize.backend_status._python_headers_status", lambda: _ok_status("python-headers"))
    monkeypatch.setattr("serve_optimize.backend_status.shutil.which", lambda command: f"/usr/bin/{command}")
    statuses = check_installation_profile("sglang")

    assert all(status.available for status in statuses)
    assert next(status for status in statuses if status.name == "compiler:gcc").available is True
    assert not any(status.name == "cuda-toolkit" for status in statuses)


def test_backend_profile_reports_missing_python_headers(monkeypatch) -> None:
    monkeypatch.setattr("serve_optimize.backend_status.sysconfig.get_path", lambda _name: "/missing/include")

    from serve_optimize.backend_status import _python_headers_status

    status = _python_headers_status()

    assert status.available is False
    assert "Python.h" in str(status.reason)


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
