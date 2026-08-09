from serve_optimize import runtime_environment


def test_virtual_environment_uses_interpreter_prefix_without_activation(monkeypatch) -> None:
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setattr(runtime_environment.sys, "prefix", "/tmp/backend-env")
    monkeypatch.setattr(runtime_environment.sys, "base_prefix", "/usr")

    assert runtime_environment._virtual_environment() == "/tmp/backend-env"
