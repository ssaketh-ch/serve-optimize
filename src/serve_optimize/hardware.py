"""Hardware detection with NVML, nvidia-smi, and no-GPU fallbacks."""

from __future__ import annotations

import csv
import io
import os
import platform
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

from .schemas import GpuDevice, HardwareSnapshot


def detect_hardware() -> HardwareSnapshot:
    notes: list[str] = []
    gpus = _detect_with_nvml(notes)
    if not gpus:
        gpus = _detect_with_nvidia_smi(notes)
    if not gpus:
        notes.append("No NVIDIA GPU telemetry detected. Using CPU/no-GPU smoke-test profile.")

    return HardwareSnapshot(
        hostname=socket.gethostname(),
        platform=platform.platform(),
        python_version=sys.version.split()[0],
        detected_at=datetime.now(timezone.utc).isoformat(),
        gpus=gpus,
        notes=notes,
        gpu_count=len(gpus),
        interconnect=_detect_interconnect(len(gpus), notes),
        cpu_model=_detect_cpu_model(),
        cpu_core_count=os.cpu_count(),
        system_memory_mb=_detect_system_memory_mb(),
        storage_type=_detect_storage_type(notes),
        operating_system=platform.platform(),
        container_or_virtual_environment=_environment_details(),
    )


def _detect_with_nvml(notes: list[str]) -> list[GpuDevice]:
    try:
        import pynvml  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        notes.append(f"pynvml unavailable: {exc.__class__.__name__}")
        return []

    try:
        pynvml.nvmlInit()
    except Exception as exc:  # pragma: no cover - environment dependent
        notes.append(f"NVML init failed: {exc}")
        return []

    gpus: list[GpuDevice] = []
    try:
        driver = _decode(pynvml.nvmlSystemGetDriverVersion())
        count = pynvml.nvmlDeviceGetCount()
        for index in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            name = _decode(pynvml.nvmlDeviceGetName(handle))
            uuid = _decode(pynvml.nvmlDeviceGetUUID(handle))
            memory = _safe_call(lambda handle=handle: pynvml.nvmlDeviceGetMemoryInfo(handle))
            power_draw = _safe_call(lambda handle=handle: pynvml.nvmlDeviceGetPowerUsage(handle))
            power_limit = _safe_call(lambda handle=handle: pynvml.nvmlDeviceGetEnforcedPowerLimit(handle))
            sm_clock = _safe_call(lambda handle=handle: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))
            mem_clock = _safe_call(lambda handle=handle: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM))
            compute_capability = _safe_call(lambda handle=handle: pynvml.nvmlDeviceGetCudaComputeCapability(handle))
            mig_mode = _safe_call(lambda handle=handle: pynvml.nvmlDeviceGetMigMode(handle))

            gpus.append(
                GpuDevice(
                    index=index,
                    name=name,
                    uuid=uuid,
                    total_memory_mb=int(memory.total / (1024 * 1024)) if memory else None,
                    free_memory_mb=int(memory.free / (1024 * 1024)) if memory else None,
                    compute_capability=_format_compute_capability(compute_capability),
                    mig_mode=_format_mig_mode(mig_mode),
                    power_limit_watts=(power_limit / 1000.0) if power_limit else None,
                    current_power_watts=(power_draw / 1000.0) if power_draw else None,
                    sm_clock_mhz=int(sm_clock) if sm_clock else None,
                    mem_clock_mhz=int(mem_clock) if mem_clock else None,
                    driver_version=driver,
                    source="pynvml",
                    raw={"nvml_index": index},
                )
            )
    except Exception as exc:  # pragma: no cover - environment dependent
        notes.append(f"NVML detection failed after init: {exc}")
        return []
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    if gpus:
        notes.append("Detected NVIDIA GPU telemetry through pynvml.")
    return gpus


def _detect_with_nvidia_smi(notes: list[str]) -> list[GpuDevice]:
    query = [
        "index",
        "name",
        "uuid",
        "memory.total",
        "memory.free",
        "power.draw",
        "power.limit",
        "clocks.sm",
        "clocks.mem",
        "compute_cap",
        "driver_version",
    ]
    command = [
        "nvidia-smi",
        f"--query-gpu={','.join(query)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=8)
    except Exception as exc:
        notes.append(f"nvidia-smi query unavailable: {exc.__class__.__name__}")
        return []

    mig_modes = _query_mig_modes()
    mig_listing = _query_mig_listing()
    gpus: list[GpuDevice] = []
    reader = csv.reader(io.StringIO(completed.stdout))
    for row in reader:
        values = [item.strip() for item in row]
        if len(values) != len(query):
            continue
        raw = dict(zip(query, values, strict=True))
        index = _to_int(raw["index"], default=len(gpus))
        gpus.append(
            GpuDevice(
                index=index,
                name=raw["name"],
                uuid=_none_if_missing(raw["uuid"]),
                total_memory_mb=_to_int(raw["memory.total"]),
                free_memory_mb=_to_int(raw["memory.free"]),
                compute_capability=_none_if_missing(raw["compute_cap"]),
                mig_mode=mig_modes.get(index),
                mig_profile=mig_listing.get(raw["uuid"]),
                power_limit_watts=_to_float(raw["power.limit"]),
                current_power_watts=_to_float(raw["power.draw"]),
                sm_clock_mhz=_to_int(raw["clocks.sm"]),
                mem_clock_mhz=_to_int(raw["clocks.mem"]),
                driver_version=_none_if_missing(raw["driver_version"]),
                source="nvidia-smi",
                raw=raw,
            )
        )

    if gpus:
        notes.append("Detected NVIDIA GPU telemetry through nvidia-smi.")
    return gpus


def _query_mig_modes() -> dict[int, str]:
    command = ["nvidia-smi", "--query-gpu=index,mig.mode.current", "--format=csv,noheader,nounits"]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=8)
    except Exception:
        return {}
    modes: dict[int, str] = {}
    reader = csv.reader(io.StringIO(completed.stdout))
    for row in reader:
        values = [item.strip() for item in row]
        if len(values) == 2:
            mode = _none_if_missing(values[1])
            if mode:
                modes[_to_int(values[0], default=len(modes))] = mode
    return modes


def _query_mig_listing() -> dict[str, str]:
    try:
        completed = subprocess.run(["nvidia-smi", "-L"], check=True, capture_output=True, text=True, timeout=8)
    except Exception:
        return {}

    profiles: dict[str, str] = {}
    current_gpu_uuid: str | None = None
    gpu_matcher = re.compile(r"GPU \d+: .* \(UUID: (?P<uuid>GPU-[^)]+)\)")
    mig_matcher = re.compile(r"MIG (?P<profile>[^:]+) Device \d+: \(UUID: (?P<uuid>MIG-[^)]+)\)")
    for line in completed.stdout.splitlines():
        gpu_match = gpu_matcher.search(line)
        if gpu_match:
            current_gpu_uuid = gpu_match.group("uuid")
            continue
        mig_match = mig_matcher.search(line)
        if mig_match:
            profiles[mig_match.group("uuid")] = mig_match.group("profile").strip()
            if current_gpu_uuid:
                profiles[current_gpu_uuid] = profiles.get(current_gpu_uuid, "MIG enabled")
    return profiles


def _safe_call(fn: Any) -> Any:
    try:
        return fn()
    except Exception:
        return None


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _detect_cpu_model() -> str | None:
    cpuinfo = "/proc/cpuinfo"
    try:
        with open(cpuinfo, encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith(("model name", "hardware", "processor")):
                    _, _, value = line.partition(":")
                    value = value.strip()
                    if value:
                        return value
    except OSError:
        pass
    processor = platform.processor()
    return processor or None


def _detect_system_memory_mb() -> int | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(int(parts[1]) / 1024)
    except (OSError, ValueError):
        return None
    return None


def _detect_storage_type(notes: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["findmnt", "-no", "FSTYPE,SOURCE", "/"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception as exc:
        notes.append(f"storage detection unavailable: {exc.__class__.__name__}")
        return None
    value = completed.stdout.strip()
    return value or None


def _detect_interconnect(gpu_count: int, notes: list[str]) -> str | None:
    if gpu_count <= 1:
        return "single_gpu"
    try:
        completed = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except Exception as exc:
        notes.append(f"interconnect detection unavailable: {exc.__class__.__name__}")
        return None
    output = completed.stdout.strip()
    return output or None


def _environment_details() -> dict[str, Any]:
    container_markers = []
    for path in ("/.dockerenv", "/run/.containerenv"):
        if os.path.exists(path):
            container_markers.append(path)
    virtual_env = os.environ.get("VIRTUAL_ENV")
    return {
        "virtual_env": virtual_env,
        "python_prefix": sys.prefix,
        "python_base_prefix": getattr(sys, "base_prefix", sys.prefix),
        "in_virtual_env": bool(virtual_env) or sys.prefix != getattr(sys, "base_prefix", sys.prefix),
        "container": os.environ.get("container"),
        "container_markers": container_markers,
    }


def _format_compute_capability(value: Any) -> str | None:
    if isinstance(value, tuple) and len(value) == 2:
        return f"{value[0]}.{value[1]}"
    return None


def _format_mig_mode(value: Any) -> str | None:
    if isinstance(value, tuple) and value:
        current = value[0]
        return "enabled" if current == 1 else "disabled"
    return None


def _none_if_missing(value: str) -> str | None:
    if value in {"", "N/A", "[N/A]", "[Not Supported]"}:
        return None
    return value


def _to_int(value: str, default: int | None = None) -> int | None:
    value = value.strip()
    if value in {"", "N/A", "[N/A]", "[Not Supported]"}:
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def _to_float(value: str, default: float | None = None) -> float | None:
    value = value.strip()
    if value in {"", "N/A", "[N/A]", "[Not Supported]"}:
        return default
    try:
        return float(value)
    except ValueError:
        return default
