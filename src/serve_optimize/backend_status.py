"""Backend and dependency availability checks."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .schemas import BackendStatus

BACKEND_MODULES = {
    "transformers": "transformers",
    "torch": "torch",
    "huggingface_hub": "huggingface_hub",
    "pynvml": "pynvml",
    "vllm": "vllm",
    "sglang": "sglang",
    "aiconfigurator": "aiconfigurator",
    "tokenpowerbench": "tokenpowerbench",
}

INSTALLATION_PROFILES = ("core", "telemetry", "vllm", "sglang")

PROFILE_DISTRIBUTIONS = {
    "core": (
        ("serve-optimize", "0.1.0"),
        ("rich", None),
    ),
    "telemetry": (
        ("serve-optimize", "0.1.0"),
        ("nvidia-ml-py", "13.610.43"),
    ),
    "vllm": (
        ("serve-optimize", "0.1.0"),
        ("vllm", "0.10.0"),
        ("torch", "2.7.1"),
        ("transformers", "4.57.6"),
        ("huggingface-hub", "0.36.2"),
        ("nvidia-ml-py", "13.610.43"),
    ),
    "sglang": (
        ("serve-optimize", "0.1.0"),
        ("sglang", "0.5.10.post1"),
        ("sglang-kernel", "0.4.1"),
        ("torch", "2.9.1"),
        ("transformers", "5.3.0"),
        ("huggingface-hub", "1.19.0"),
        ("nvidia-ml-py", "13.610.43"),
    ),
}


def check_backend_status() -> list[BackendStatus]:
    statuses = [
        _dependency_status(name, module)
        for name, module in BACKEND_MODULES.items()
    ]
    statuses.extend(
        [
            _command_status("nvidia-smi"),
            _command_status("vllm"),
            _command_status("sglang"),
            _sglang_runtime_status(),
        ]
    )
    return statuses


def check_installation_profile(profile: str) -> list[BackendStatus]:
    if profile not in INSTALLATION_PROFILES:
        raise ValueError(f"Unknown installation profile '{profile}'.")

    statuses = [
        BackendStatus(
            name="python",
            available=sys.version_info >= (3, 10),
            version=".".join(str(item) for item in sys.version_info[:3]),
            command=sys.executable,
            reason=None if sys.version_info >= (3, 10) else "Python 3.10 or newer is required.",
        )
    ]
    statuses.extend(
        _distribution_status(name, expected)
        for name, expected in PROFILE_DISTRIBUTIONS[profile]
    )

    if profile == "vllm":
        statuses.append(_environment_command_status("vllm"))
    elif profile == "sglang":
        statuses.extend(
            [
                _environment_command_status("sglang"),
                _sglang_runtime_status(),
                _compiler_status("gcc", required_path="/opt/rh/gcc-toolset-12/"),
                _compiler_status("g++", required_path="/opt/rh/gcc-toolset-12/"),
                _cuda_status(),
            ]
        )
    return statuses


def _distribution_status(name: str, expected_version: str | None) -> BackendStatus:
    try:
        version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return BackendStatus(name=name, available=False, reason="distribution is not installed")
    if expected_version is not None and version != expected_version:
        return BackendStatus(
            name=name,
            available=False,
            version=version,
            reason=f"expected {expected_version}",
        )
    return BackendStatus(name=name, available=True, version=version)


def _module_status(name: str, module_name: str) -> BackendStatus:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return BackendStatus(name=name, available=False, reason=f"{exc.__class__.__name__}: {exc}")
    return BackendStatus(name=name, available=True, version=getattr(module, "__version__", None))


def _dependency_status(name: str, module_name: str) -> BackendStatus:
    if name == "sglang":
        return _sglang_status()
    if name == "aiconfigurator":
        return _module_or_cli_status(name, module_name, [name, "--help"])
    return _module_status(name, module_name)


def _module_or_cli_status(name: str, module_name: str, command: list[str]) -> BackendStatus:
    module_status = _module_status(name, module_name)
    if module_status.available:
        return module_status
    executable = shutil.which(command[0])
    if not executable:
        return module_status
    try:
        completed = subprocess.run(
            [executable, *command[1:]],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception as exc:
        return BackendStatus(
            name=name,
            available=False,
            command=executable,
            reason=f"{module_status.reason}; CLI check failed: {exc.__class__.__name__}: {exc}",
        )
    if completed.returncode == 0:
        return BackendStatus(name=name, available=True, command=executable)
    reason = _last_nonempty_line(completed.stderr) or _last_nonempty_line(completed.stdout) or "CLI check failed"
    return BackendStatus(
        name=name,
        available=False,
        command=executable,
        reason=f"{module_status.reason}; {reason}",
    )


def _sglang_status() -> BackendStatus:
    module_status = _module_status("sglang", "sglang")
    if module_status.available:
        return module_status
    command = shutil.which("sglang")
    if not command:
        return module_status
    try:
        completed = subprocess.run(
            [command, "version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception as exc:
        return BackendStatus(
            name="sglang",
            available=False,
            command=command,
            reason=f"{module_status.reason}; CLI check failed: {exc.__class__.__name__}: {exc}",
        )
    if completed.returncode == 0:
        version = _sglang_version_from_output(f"{completed.stdout}\n{completed.stderr}")
        return BackendStatus(name="sglang", available=True, command=command, version=version)
    reason = _last_nonempty_line(completed.stderr) or _last_nonempty_line(completed.stdout) or "CLI check failed"
    return BackendStatus(
        name="sglang",
        available=False,
        command=command,
        reason=f"{module_status.reason}; {reason}",
    )


def _command_status(command: str) -> BackendStatus:
    path = shutil.which(command)
    if path is None:
        sibling = Path(sys.executable).with_name(command)
        if sibling.exists():
            path = str(sibling)
    if not path:
        return BackendStatus(name=f"cmd:{command}", available=False, reason="not found on PATH")
    return BackendStatus(name=f"cmd:{command}", available=True, command=path)


def _environment_command_status(command: str) -> BackendStatus:
    path = Path(sys.executable).with_name(command)
    if not path.is_file() or not os.access(path, os.X_OK):
        return BackendStatus(
            name=f"cmd:{command}",
            available=False,
            reason=f"not installed beside {sys.executable}",
        )
    return BackendStatus(name=f"cmd:{command}", available=True, command=str(path))


def _compiler_status(command: str, *, required_path: str) -> BackendStatus:
    path = shutil.which(command)
    if path is None:
        return BackendStatus(
            name=f"compiler:{command}",
            available=False,
            reason=f"{command} is not available; source scripts/env_base_runtime.sh",
        )
    if required_path not in path:
        return BackendStatus(
            name=f"compiler:{command}",
            available=False,
            command=path,
            reason="GCC Toolset 12 must lead PATH; source scripts/env_base_runtime.sh",
        )
    return BackendStatus(name=f"compiler:{command}", available=True, command=path)


def _cuda_status() -> BackendStatus:
    cuda_home = os.environ.get("CUDA_HOME")
    nvcc = shutil.which("nvcc")
    if not cuda_home:
        return BackendStatus(
            name="cuda-toolkit",
            available=False,
            command=nvcc,
            reason="CUDA_HOME is not set; source scripts/env_base_runtime.sh",
        )
    if not Path(cuda_home).exists():
        return BackendStatus(
            name="cuda-toolkit",
            available=False,
            command=cuda_home,
            reason="CUDA_HOME does not exist",
        )
    if not nvcc:
        return BackendStatus(
            name="cuda-toolkit",
            available=False,
            command=cuda_home,
            reason="nvcc is not available on PATH",
        )
    return BackendStatus(name="cuda-toolkit", available=True, command=f"{cuda_home} ({nvcc})")


def _sglang_runtime_status() -> BackendStatus:
    try:
        env = os.environ.copy()
        nvrtc = Path(sys.executable).parents[1] / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "nvidia" / "cuda_nvrtc" / "lib"
        if nvrtc.exists():
            env["LD_LIBRARY_PATH"] = f"{nvrtc}:{env.get('LD_LIBRARY_PATH', '')}"
        commands = [
            ([sys.executable, "-m", "sglang.launch_server", "--help"], f"{sys.executable} -m sglang.launch_server"),
        ]
        sglang = shutil.which("sglang")
        if sglang:
            commands.extend(
                [
                    ([sglang, "serve", "--help"], f"{sglang} serve"),
                    ([sglang, "launch_server", "--help"], f"{sglang} launch_server"),
                ]
            )
        errors = []
        for command, display in commands:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
                check=False,
            )
            if completed.returncode == 0 and "--model-path" in f"{completed.stdout}\n{completed.stderr}":
                return BackendStatus(name="sglang-runtime", available=True, command=display)
            reason = _last_nonempty_line(completed.stderr) or _last_nonempty_line(completed.stdout) or "runtime check failed"
            errors.append(f"{' '.join(command)}: {reason}")
    except Exception as exc:
        return BackendStatus(name="sglang-runtime", available=False, reason=f"{exc.__class__.__name__}: {exc}")
    return BackendStatus(name="sglang-runtime", available=False, reason="; ".join(errors))


def _last_nonempty_line(text: str) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else None


def _sglang_version_from_output(text: str) -> str | None:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("sglang version:"):
            return line.partition(":")[2].strip() or None
    return _last_nonempty_line(text)
