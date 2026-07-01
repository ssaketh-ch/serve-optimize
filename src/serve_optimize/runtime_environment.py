"""Runtime environment identity for managed evidence compatibility."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import socket
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class RuntimeEnvironmentFingerprint:
    backend_name: str
    backend_version: str
    torch_version: str
    cuda_runtime_version: str
    python_version: str
    hostname: str = UNAVAILABLE
    operating_system: str = UNAVAILABLE
    virtual_environment: str = UNAVAILABLE
    gpu_driver_version: str = UNAVAILABLE
    compiler_toolchain: dict[str, str] = field(default_factory=dict)
    compiler_toolchain_fingerprint: str = ""
    serve_optimize_git_commit: str = UNAVAILABLE
    dirty_tree: bool | None = None
    environment_fingerprint: str = ""
    schema_version: str = "runtime-environment/v1"

    def to_artifact(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "backend_name": self.backend_name,
            "backend_version": self.backend_version,
            "torch_version": self.torch_version,
            "cuda_runtime_version": self.cuda_runtime_version,
            "python_version": self.python_version,
            "hostname": self.hostname,
            "operating_system": self.operating_system,
            "virtual_environment": self.virtual_environment,
            "gpu_driver_version": self.gpu_driver_version,
            "compiler_toolchain": self.compiler_toolchain,
            "compiler_toolchain_fingerprint": self.compiler_toolchain_fingerprint,
            "serve_optimize_git_commit": self.serve_optimize_git_commit,
            "dirty_tree": self.dirty_tree,
            "environment_fingerprint": self.environment_fingerprint,
        }


def collect_runtime_environment(
    *,
    backend_name: str,
    backend_version: str | None,
    repo_root: Path | None = None,
) -> RuntimeEnvironmentFingerprint:
    root = repo_root or Path(__file__).resolve().parents[2]
    process = _process_runtime_metadata(str(root))
    payload = {
        "schema_version": "runtime-environment/v1",
        "backend_name": backend_name,
        "backend_version": backend_version or UNAVAILABLE,
        "torch_version": process["torch_version"],
        "cuda_runtime_version": process["cuda_runtime_version"],
        "python_version": process["python_version"],
        "hostname": process["hostname"],
        "operating_system": process["operating_system"],
        "virtual_environment": process["virtual_environment"],
        "gpu_driver_version": process["gpu_driver_version"],
        "compiler_toolchain": process["compiler_toolchain"],
        "compiler_toolchain_fingerprint": process["compiler_toolchain_fingerprint"],
        "serve_optimize_git_commit": process["serve_optimize_git_commit"],
        "dirty_tree": process["dirty_tree"],
    }
    return RuntimeEnvironmentFingerprint(
        backend_name=backend_name,
        backend_version=str(payload["backend_version"]),
        torch_version=str(payload["torch_version"]),
        cuda_runtime_version=str(payload["cuda_runtime_version"]),
        python_version=str(payload["python_version"]),
        hostname=str(payload["hostname"]),
        operating_system=str(payload["operating_system"]),
        virtual_environment=str(payload["virtual_environment"]),
        gpu_driver_version=str(payload["gpu_driver_version"]),
        compiler_toolchain=dict(process["compiler_toolchain"]),
        compiler_toolchain_fingerprint=str(payload["compiler_toolchain_fingerprint"]),
        serve_optimize_git_commit=str(payload["serve_optimize_git_commit"]),
        dirty_tree=bool(payload["dirty_tree"]) if payload["dirty_tree"] is not None else None,
        environment_fingerprint=stable_payload_hash(payload),
    )


def build_runtime_evidence_fingerprint(
    runtime_environment: RuntimeEnvironmentFingerprint | dict[str, Any],
    *,
    rendered_launch_command: list[str],
    backend_capability_help_hash: str | None,
    canonical_launch_config_identity: str,
    model_identity: str,
    workload_identity: str,
) -> dict[str, Any]:
    environment = (
        runtime_environment.to_artifact()
        if isinstance(runtime_environment, RuntimeEnvironmentFingerprint)
        else dict(runtime_environment)
    )
    payload = {
        "schema_version": "runtime-evidence-fingerprint/v1",
        "runtime_environment": environment,
        "rendered_launch_command_hash": stable_payload_hash(
            {"command": rendered_launch_command}
        ),
        "backend_capability_help_hash": backend_capability_help_hash or UNAVAILABLE,
        "canonical_launch_config_identity": canonical_launch_config_identity,
        "model_identity": model_identity,
        "workload_identity": workload_identity,
    }
    return {
        **payload,
        "fingerprint": stable_payload_hash(payload),
    }


def stable_payload_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@lru_cache(maxsize=4)
def _process_runtime_metadata(repo_root: str) -> dict[str, Any]:
    compiler_toolchain = {
        "gcc_version": _command_version("gcc"),
        "gxx_version": _command_version("g++"),
        "nvcc_version": _command_version("nvcc"),
        "cc": os.environ.get("CC") or UNAVAILABLE,
        "cxx": os.environ.get("CXX") or UNAVAILABLE,
        "cuda_home": os.environ.get("CUDA_HOME") or UNAVAILABLE,
        "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST") or UNAVAILABLE,
    }
    return {
        "torch_version": _package_version("torch"),
        "cuda_runtime_version": _torch_cuda_version(),
        "python_version": platform.python_version(),
        "hostname": socket.gethostname(),
        "operating_system": platform.platform(),
        "virtual_environment": os.environ.get("VIRTUAL_ENV") or UNAVAILABLE,
        "gpu_driver_version": _gpu_driver_version(),
        "compiler_toolchain": compiler_toolchain,
        "compiler_toolchain_fingerprint": stable_payload_hash(compiler_toolchain),
        "serve_optimize_git_commit": _git_commit(Path(repo_root)),
        "dirty_tree": _git_dirty(Path(repo_root)),
    }


def _package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return UNAVAILABLE


def _torch_cuda_version() -> str:
    try:
        import torch
    except (ImportError, OSError):
        return UNAVAILABLE
    version = getattr(getattr(torch, "version", None), "cuda", None)
    return str(version) if version else UNAVAILABLE


def _command_version(executable: str) -> str:
    resolved = shutil.which(executable)
    if not resolved:
        return UNAVAILABLE
    try:
        completed = subprocess.run(
            [resolved, "--version"],
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return UNAVAILABLE
    output = completed.stdout or completed.stderr
    first_line = output.splitlines()[0].strip() if output else ""
    return first_line or UNAVAILABLE


def _git_commit(repo_root: Path) -> str:
    if not (repo_root / ".git").exists():
        return UNAVAILABLE
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return UNAVAILABLE
    commit = completed.stdout.strip()
    return commit if completed.returncode == 0 and commit else UNAVAILABLE


def _git_dirty(repo_root: Path) -> bool | None:
    if not (repo_root / ".git").exists():
        return None
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--short"],
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return bool(completed.stdout.strip())


def _gpu_driver_version() -> str:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return UNAVAILABLE
    line = completed.stdout.splitlines()[0].strip() if completed.stdout.splitlines() else ""
    return line if completed.returncode == 0 and line else UNAVAILABLE
