"""Bridge to the installed AIConfigurator CLI."""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AIConfiguratorRun:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    output_path: str | None = None


def run_aiconfigurator(
    mode: str,
    model: str,
    system: str,
    backend: str = "vllm",
    output_dir: Path | None = None,
    isl: int = 1024,
    osl: int = 128,
    batch_size: int = 16,
    total_gpus: int = 1,
    ttft: float | None = None,
    tpot: float | None = None,
    prefix: int | None = None,
    request_latency: float | None = None,
) -> AIConfiguratorRun:
    binary = shutil.which("aiconfigurator")
    if binary is None:
        sibling = Path(sys.executable).with_name("aiconfigurator")
        if sibling.exists():
            binary = str(sibling)
    if binary is None:
        raise RuntimeError("AIConfigurator CLI is not installed. Install with: pip install -e '.[aiconfigurator]'")
    if mode in {"default", "generate", "estimate"} and backend == "all":
        raise ValueError("AIConfigurator backend='all' is only valid for support mode.")

    if mode == "support":
        command = [
            binary,
            "cli",
            "support",
            "--model-path",
            model,
            "--system",
            system,
            "--backend",
            backend,
        ]
    elif mode == "default":
        command = [
            binary,
            "cli",
            "default",
            "--model-path",
            model,
            "--system",
            system,
            "--backend",
            backend,
            "--total-gpus",
            str(total_gpus),
            "--isl",
            str(isl),
            "--osl",
            str(osl),
        ]
        if prefix is not None:
            command.extend(["--prefix", str(prefix)])
        if request_latency is not None:
            command.extend(["--request-latency", str(request_latency)])
        else:
            if ttft is not None:
                command.extend(["--ttft", str(ttft)])
            if tpot is not None:
                command.extend(["--tpot", str(tpot)])
    elif mode == "generate":
        command = [
            binary,
            "cli",
            "generate",
            "--model-path",
            model,
            "--system",
            system,
            "--backend",
            backend,
            "--total-gpus",
            str(total_gpus),
        ]
    elif mode == "estimate":
        command = [
            binary,
            "cli",
            "estimate",
            "--model-path",
            model,
            "--system",
            system,
            "--backend",
            backend,
            "--isl",
            str(isl),
            "--osl",
            str(osl),
            "--batch-size",
            str(batch_size),
        ]
    else:
        raise ValueError(f"Unsupported AIConfigurator mode: {mode}")

    if output_dir and mode != "support":
        output_dir.mkdir(parents=True, exist_ok=True)
        command.extend(["--save-dir", str(output_dir)])

    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    output_path = None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"aiconfigurator-{mode}.txt")
        Path(output_path).write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")

    return AIConfiguratorRun(
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        output_path=output_path,
    )
