"""vLLM launch and managed lifecycle adapter."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from types import TracebackType
from typing import Any

from serve_optimize.backends.base import LaunchPlan
from serve_optimize.endpoint_benchmark import RequestFn, send_chat_completion_request
from serve_optimize.schemas import (
    EndpointBenchmarkConfig,
    HealthCheckResult,
    ManagedLifecycleRecord,
    RequestRecord,
    ServerHandle,
    ServerLaunchSpec,
    ServingConfig,
)

PopenFactory = Callable[..., subprocess.Popen[Any]]
KillpgFn = Callable[[int, int], None]
RELEVANT_VLLM_FLAGS = (
    "--block-size",
    "--kv-cache-dtype",
    "--enforce-eager",
    "--max-num-batched-tokens",
    "--enable-chunked-prefill",
    "--no-enable-chunked-prefill",
    "--max-cudagraph-capture-size",
    "--cuda-graph-sizes",
    "--enable-prefix-caching",
    "--no-enable-prefix-caching",
)


@dataclass(frozen=True)
class VLLMArgumentCapabilities:
    executable: str
    version: str | None
    supported_flags: frozenset[str] = field(default_factory=frozenset)
    option_choices: dict[str, frozenset[str]] = field(default_factory=dict)
    help_hash: str | None = None
    detection_status: str = "unavailable"
    detection_error: str | None = None

    def supports(self, flag: str) -> bool:
        return flag in self.supported_flags

    def choices_for(self, flag: str) -> frozenset[str]:
        return self.option_choices.get(flag, frozenset())

    def cudagraph_capture_flag(self) -> str | None:
        if self.supports("--max-cudagraph-capture-size"):
            return "--max-cudagraph-capture-size"
        if self.supports("--cuda-graph-sizes"):
            return "--cuda-graph-sizes"
        return None

    def to_artifact(self) -> dict[str, object]:
        return {
            "schema_version": "vllm-argument-capabilities/v1",
            "executable": self.executable,
            "version": self.version,
            "detection_status": self.detection_status,
            "detection_error": self.detection_error,
            "help_hash": self.help_hash,
            "supported_flags": {flag: self.supports(flag) for flag in RELEVANT_VLLM_FLAGS},
            "option_choices": {flag: sorted(values) for flag, values in sorted(self.option_choices.items())},
        }


@dataclass(frozen=True)
class VLLMRenderedLaunch:
    command: list[str]
    canonical_config: ServingConfig
    rendered_fields: dict[str, object] = field(default_factory=dict)
    omitted_fields: dict[str, str] = field(default_factory=dict)
    unsupported_fields: dict[str, str] = field(default_factory=dict)
    flag_aliases: dict[str, str] = field(default_factory=dict)
    capabilities_help_hash: str | None = None

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema_version": "vllm-rendered-launch/v1",
            "canonical_config": self.canonical_config,
            "rendered_fields": self.rendered_fields,
            "omitted_fields": self.omitted_fields,
            "unsupported_fields": self.unsupported_fields,
            "flag_aliases": self.flag_aliases,
            "capabilities_help_hash": self.capabilities_help_hash,
        }


class _ManagedLogHandles:
    def __init__(self, stdout_path: str | None, stderr_path: str | None):
        self.stdout_path = stdout_path
        self.stderr_path = stderr_path
        self.stdout = None
        self.stderr = None

    def __enter__(self) -> _ManagedLogHandles:
        if self.stdout_path is not None:
            path = Path(self.stdout_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.stdout = path.open("ab")
        if self.stderr_path is not None:
            path = Path(self.stderr_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.stderr = path.open("ab")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        if self.stdout is not None:
            self.stdout.close()
        if self.stderr is not None:
            self.stderr.close()


class VllmAdapter:
    name = "vllm"

    def __init__(
        self,
        *,
        popen_factory: PopenFactory | None = None,
        killpg_fn: KillpgFn | None = None,
        argument_capabilities: VLLMArgumentCapabilities | None = None,
    ) -> None:
        self._popen_factory = popen_factory or subprocess.Popen
        self._killpg_fn = killpg_fn or os.killpg
        self._argument_capabilities = argument_capabilities
        self._processes: dict[int, subprocess.Popen[Any]] = {}
        self._launched_pgids: set[int] = set()

    def is_available(self) -> bool:
        return _vllm_sibling_executable() is not None or shutil.which("vllm") is not None

    def build_launch_plan(self, config: ServingConfig) -> LaunchPlan:
        command = render_vllm_launch(config, capabilities=self.argument_capabilities()).command
        return LaunchPlan(command=command, environment={}, notes=["Launch plan only; process management lands in the benchmark runner."])

    def build_launch_spec(
        self,
        config: ServingConfig,
        *,
        host: str,
        port: int | None,
        log_dir: Path,
    ) -> ServerLaunchSpec:
        if config.backend != self.name:
            raise ValueError(f"vLLM managed adapter cannot launch backend '{config.backend}'.")
        resolved_port = port if port is not None else allocate_port(host)
        validate_port_available(host, resolved_port)
        rendered = render_vllm_launch(config, host=host, port=resolved_port, capabilities=self.argument_capabilities())
        candidate_log_dir = log_dir / config.id
        base_url = f"http://{_health_host(host)}:{resolved_port}/v1"
        metadata = self.backend_metadata()
        metadata["rendered_launch"] = rendered.to_metadata()
        return ServerLaunchSpec(
            config_id=config.id,
            backend=self.name,
            model_id=config.model_id,
            host=host,
            port=resolved_port,
            base_url=base_url,
            command=rendered.command,
            environment={},
            stdout_log_path=str(candidate_log_dir / "stdout.log"),
            stderr_log_path=str(candidate_log_dir / "stderr.log"),
            metadata=metadata,
        )

    def launch_server(self, spec: ServerLaunchSpec) -> ServerHandle:
        env = os.environ.copy()
        env.update(spec.environment)
        with _ManagedLogHandles(spec.stdout_log_path, spec.stderr_log_path) as logs:
            process = self._popen_factory(
                spec.command,
                stdout=logs.stdout,
                stderr=logs.stderr,
                env=env,
                start_new_session=True,
            )
        pgid = os.getpgid(process.pid)
        self._processes[process.pid] = process
        self._launched_pgids.add(pgid)
        return ServerHandle(
            config_id=spec.config_id,
            backend=spec.backend,
            pid=process.pid,
            pgid=pgid,
            host=spec.host,
            port=spec.port,
            base_url=spec.base_url,
            started_at=datetime.now(timezone.utc).isoformat(),
            stdout_log_path=spec.stdout_log_path,
            stderr_log_path=spec.stderr_log_path,
            metadata=spec.metadata,
        )

    def wait_for_health(
        self,
        handle: ServerHandle,
        *,
        model: str,
        timeout_s: float,
        request_fn: RequestFn | None = None,
    ) -> HealthCheckResult:
        request_fn = request_fn or send_chat_completion_request
        started_at = datetime.now(timezone.utc)
        deadline = time.monotonic() + timeout_s
        attempts = 0
        last_error = None
        while time.monotonic() <= deadline:
            attempts += 1
            process = self._processes.get(handle.pid)
            if process is not None and process.poll() is not None:
                last_error = f"server process exited with return code {process.returncode}"
                break
            request_start = time.perf_counter()
            record = _health_request(handle, model, request_fn)
            latency_s = time.perf_counter() - request_start
            if record.status == "ok":
                ended_at = datetime.now(timezone.utc)
                return HealthCheckResult(
                    config_id=handle.config_id,
                    backend=handle.backend,
                    base_url=handle.base_url,
                    healthy=True,
                    status="ok",
                    attempts=attempts,
                    started_at=started_at.isoformat(),
                    ended_at=ended_at.isoformat(),
                    latency_s=latency_s,
                )
            last_error = record.error or record.status
            time.sleep(0.5)
        ended_at = datetime.now(timezone.utc)
        return HealthCheckResult(
            config_id=handle.config_id,
            backend=handle.backend,
            base_url=handle.base_url,
            healthy=False,
            status="timeout",
            attempts=attempts,
            started_at=started_at.isoformat(),
            ended_at=ended_at.isoformat(),
            error=last_error or "health check timed out",
        )

    def stop_server(self, handle: ServerHandle, *, timeout_s: float = 30.0) -> ManagedLifecycleRecord:
        process = self._processes.get(handle.pid)
        if handle.pgid not in self._launched_pgids:
            return _lifecycle_record(
                handle,
                event="stop",
                status="skipped",
                message="Process group was not launched by this adapter.",
                returncode=process.returncode if process is not None else None,
            )
        if process is not None and process.poll() is not None:
            self._launched_pgids.discard(handle.pgid)
            self._processes.pop(handle.pid, None)
            return _lifecycle_record(
                handle,
                event="stop",
                status="already-exited",
                message="Server process had already exited.",
                returncode=process.returncode,
            )
        self._killpg_fn(handle.pgid, signal.SIGTERM)
        deadline = time.monotonic() + timeout_s
        while process is not None and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        if process is not None and process.poll() is None:
            self._killpg_fn(handle.pgid, signal.SIGKILL)
        returncode = process.returncode if process is not None else None
        self._launched_pgids.discard(handle.pgid)
        self._processes.pop(handle.pid, None)
        return _lifecycle_record(
            handle,
            event="stop",
            status="stopped",
            message="Stopped launched vLLM process group.",
            returncode=returncode,
        )

    def backend_metadata(self) -> dict[str, str | None]:
        capabilities = self.argument_capabilities()
        return {
            "adapter": self.name,
            "executable": _vllm_sibling_executable() or shutil.which("vllm"),
            "version": _installed_version("vllm"),
            "argument_detection_status": capabilities.detection_status,
            "argument_capabilities_help_hash": capabilities.help_hash,
        }

    def argument_capabilities(self) -> VLLMArgumentCapabilities:
        if self._argument_capabilities is not None:
            return self._argument_capabilities
        return detect_vllm_argument_capabilities()


def allocate_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def validate_port_available(host: str, port: int) -> None:
    if port < 1 or port > 65535:
        raise ValueError("port must be between 1 and 65535.")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError as exc:
            raise ValueError(f"port {port} is not available on {host}: {exc}") from exc


def detect_vllm_argument_capabilities(*, executable: str | None = None, timeout_s: float = 30.0) -> VLLMArgumentCapabilities:
    resolved = executable or _vllm_sibling_executable() or shutil.which("vllm")
    cache_key = resolved or "vllm"
    return _detect_vllm_argument_capabilities_cached(cache_key, timeout_s)


@lru_cache(maxsize=8)
def _detect_vllm_argument_capabilities_cached(executable: str, timeout_s: float) -> VLLMArgumentCapabilities:
    version = _installed_version("vllm")
    commands: list[list[str]] = []
    if executable and shutil.which(executable):
        commands.append([executable, "serve", "--help"])
    elif executable and Path(executable).exists():
        commands.append([executable, "serve", "--help"])
    else:
        resolved = shutil.which("vllm")
        if resolved:
            commands.append([resolved, "serve", "--help"])
    commands.append([sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--help"])

    errors: list[str] = []
    for command in commands:
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(f"{' '.join(command)}: {exc.__class__.__name__}: {exc}")
            continue
        help_text = f"{completed.stdout}\n{completed.stderr}"
        if completed.returncode != 0:
            errors.append(f"{' '.join(command)} exited {completed.returncode}")
            continue
        return parse_vllm_argument_capabilities(
            help_text,
            executable=command[0],
            version=version,
        )

    return VLLMArgumentCapabilities(
        executable=executable,
        version=version,
        detection_status="unavailable" if not errors else "failed",
        detection_error="; ".join(errors) if errors else "vLLM executable was not found.",
    )


def parse_vllm_argument_capabilities(
    help_text: str,
    *,
    executable: str = "vllm",
    version: str | None = None,
) -> VLLMArgumentCapabilities:
    flags = frozenset(re.findall(r"--[A-Za-z0-9][A-Za-z0-9_-]*", help_text))
    choices = _parse_option_choices(help_text, "--kv-cache-dtype")
    option_choices = {"--kv-cache-dtype": frozenset(choices)} if choices else {}
    help_hash = _vllm_capability_hash(flags, option_choices)
    return VLLMArgumentCapabilities(
        executable=executable,
        version=version,
        supported_flags=flags,
        option_choices=option_choices,
        help_hash=help_hash,
        detection_status="success",
    )


def _vllm_capability_hash(
    flags: frozenset[str],
    option_choices: dict[str, frozenset[str]],
) -> str:
    payload = {
        "flags": sorted(flags),
        "option_choices": {
            flag: sorted(values) for flag, values in sorted(option_choices.items())
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_option_choices(help_text: str, flag: str) -> set[str]:
    lines = help_text.splitlines()
    for index, line in enumerate(lines):
        if flag not in line:
            continue
        block = " ".join(lines[index : index + 3])
        brace_match = re.search(r"\{([^}]+)\}", block)
        if brace_match:
            return {item.strip() for item in brace_match.group(1).split(",") if item.strip()}
        choices_match = re.search(r"choices?:\s*([^.;\n]+)", block, flags=re.IGNORECASE)
        if choices_match:
            raw = re.split(r"[, ]+", choices_match.group(1))
            return {item.strip("{}[](),") for item in raw if item.strip("{}[](),")}
    return set()


def render_vllm_launch(
    config: ServingConfig,
    *,
    host: str | None = None,
    port: int | None = None,
    capabilities: VLLMArgumentCapabilities | None = None,
) -> VLLMRenderedLaunch:
    if (config.extra or {}).get("backend_defaults") is True:
        command = [_vllm_sibling_executable() or "vllm", "serve", config.model_id]
        if host is not None:
            command.extend(["--host", host])
        if port is not None:
            command.extend(["--port", str(port)])
        return VLLMRenderedLaunch(
            command=command,
            canonical_config=replace(
                config,
                block_size=None,
                kv_cache_dtype=None,
                enforce_eager=None,
                max_num_batched_tokens=None,
                enable_chunked_prefill=None,
                max_cudagraph_capture_size=None,
                enable_prefix_caching=None,
            ),
            rendered_fields={"backend_defaults": True},
            omitted_fields={
                "dtype": "backend default",
                "quantization": "backend default",
                "max_model_len": "backend default",
                "gpu_memory_utilization": "backend default",
                "max_num_seqs": "backend default",
                "tensor_parallel_size": "backend default",
            },
            capabilities_help_hash=capabilities.help_hash if capabilities is not None else None,
        )
    rendered_fields: dict[str, object] = {
        "dtype": config.dtype,
        "quantization": config.quantization,
        "max_model_len": config.max_context_tokens,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "max_num_seqs": config.max_batch_size,
        "tensor_parallel_size": config.tensor_parallelism,
    }
    omitted_fields: dict[str, str] = {}
    unsupported_fields: dict[str, str] = {}
    flag_aliases: dict[str, str] = {}
    canonical_values: dict[str, object] = {
        "block_size": None,
        "kv_cache_dtype": None,
        "enforce_eager": None,
        "max_num_batched_tokens": None,
        "enable_chunked_prefill": None,
        "max_cudagraph_capture_size": None,
        "enable_prefix_caching": None,
    }
    command = [
        _vllm_sibling_executable() or "vllm",
        "serve",
        config.model_id,
        "--dtype",
        _vllm_dtype(config.dtype),
        "--max-model-len",
        str(config.max_context_tokens),
        "--gpu-memory-utilization",
        str(config.gpu_memory_utilization),
        "--max-num-seqs",
        str(config.max_batch_size),
        "--tensor-parallel-size",
        str(config.tensor_parallelism),
    ]

    if config.block_size is not None:
        if _supports_flag(capabilities, "--block-size"):
            command.extend(["--block-size", str(config.block_size)])
            rendered_fields["block_size"] = config.block_size
            canonical_values["block_size"] = config.block_size
        else:
            unsupported_fields["block_size"] = "Installed vLLM does not support --block-size."

    if config.kv_cache_dtype is not None:
        rendered_kv_cache_dtype = _vllm_kv_cache_dtype(config.kv_cache_dtype)
        if _supports_kv_cache_dtype(capabilities, config.kv_cache_dtype):
            command.extend(["--kv-cache-dtype", rendered_kv_cache_dtype])
            rendered_fields["kv_cache_dtype"] = rendered_kv_cache_dtype
            canonical_values["kv_cache_dtype"] = rendered_kv_cache_dtype
        else:
            unsupported_fields["kv_cache_dtype"] = "Installed vLLM does not support the requested --kv-cache-dtype value."

    if config.enforce_eager is True:
        if _supports_flag(capabilities, "--enforce-eager"):
            command.append("--enforce-eager")
            rendered_fields["enforce_eager"] = True
            canonical_values["enforce_eager"] = True
        else:
            unsupported_fields["enforce_eager"] = "Installed vLLM does not support --enforce-eager."

    if config.max_num_batched_tokens is not None:
        if _supports_flag(capabilities, "--max-num-batched-tokens"):
            command.extend(["--max-num-batched-tokens", str(config.max_num_batched_tokens)])
            rendered_fields["max_num_batched_tokens"] = config.max_num_batched_tokens
            canonical_values["max_num_batched_tokens"] = config.max_num_batched_tokens
        else:
            unsupported_fields["max_num_batched_tokens"] = "Installed vLLM does not support --max-num-batched-tokens."

    if config.enable_chunked_prefill is True:
        if _supports_flag(capabilities, "--enable-chunked-prefill"):
            command.append("--enable-chunked-prefill")
            rendered_fields["enable_chunked_prefill"] = True
            canonical_values["enable_chunked_prefill"] = True
        else:
            unsupported_fields["enable_chunked_prefill"] = "Installed vLLM does not support --enable-chunked-prefill."
    elif config.enable_chunked_prefill is False:
        if _supports_flag(capabilities, "--no-enable-chunked-prefill"):
            command.append("--no-enable-chunked-prefill")
            rendered_fields["enable_chunked_prefill"] = False
            canonical_values["enable_chunked_prefill"] = False
        else:
            unsupported_fields["enable_chunked_prefill"] = "Installed vLLM does not support --no-enable-chunked-prefill."

    if config.max_cudagraph_capture_size is not None:
        flag = capabilities.cudagraph_capture_flag() if capabilities is not None and capabilities.detection_status == "success" else "--max-cudagraph-capture-size"
        if flag is not None:
            command.extend([flag, str(config.max_cudagraph_capture_size)])
            rendered_fields["max_cudagraph_capture_size"] = config.max_cudagraph_capture_size
            canonical_values["max_cudagraph_capture_size"] = config.max_cudagraph_capture_size
            if flag != "--max-cudagraph-capture-size":
                flag_aliases["max_cudagraph_capture_size"] = flag
        else:
            unsupported_fields["max_cudagraph_capture_size"] = "Installed vLLM supports no CUDA graph capture size flag."

    if config.enable_prefix_caching is True:
        if _supports_flag(capabilities, "--enable-prefix-caching"):
            command.append("--enable-prefix-caching")
            rendered_fields["enable_prefix_caching"] = True
            canonical_values["enable_prefix_caching"] = True
        else:
            unsupported_fields["enable_prefix_caching"] = "Installed vLLM does not support --enable-prefix-caching."
    elif config.enable_prefix_caching is False:
        if _supports_flag(capabilities, "--no-enable-prefix-caching"):
            command.append("--no-enable-prefix-caching")
            rendered_fields["enable_prefix_caching"] = False
            canonical_values["enable_prefix_caching"] = False
        else:
            unsupported_fields["enable_prefix_caching"] = "Installed vLLM does not support --no-enable-prefix-caching."

    for field_name, value in (
        ("block_size", config.block_size),
        ("kv_cache_dtype", config.kv_cache_dtype),
        ("enforce_eager", config.enforce_eager),
        ("max_num_batched_tokens", config.max_num_batched_tokens),
        ("enable_chunked_prefill", config.enable_chunked_prefill),
        ("max_cudagraph_capture_size", config.max_cudagraph_capture_size),
        ("enable_prefix_caching", config.enable_prefix_caching),
    ):
        if value is None:
            omitted_fields[field_name] = "not set"
        elif field_name in unsupported_fields:
            canonical_values[field_name] = None

    if host is not None:
        command.extend(["--host", host])
    if port is not None:
        command.extend(["--port", str(port)])
    if config.quantization != "none":
        command.extend(["--quantization", _vllm_quantization(config.quantization)])
    canonical_config = replace(config, **canonical_values)
    return VLLMRenderedLaunch(
        command=command,
        canonical_config=canonical_config,
        rendered_fields=rendered_fields,
        omitted_fields=omitted_fields,
        unsupported_fields=unsupported_fields,
        flag_aliases=flag_aliases,
        capabilities_help_hash=capabilities.help_hash if capabilities is not None else None,
    )


def vllm_command(
    config: ServingConfig,
    *,
    host: str | None = None,
    port: int | None = None,
    capabilities: VLLMArgumentCapabilities | None = None,
) -> list[str]:
    return render_vllm_launch(config, host=host, port=port, capabilities=capabilities).command


def _vllm_sibling_executable() -> str | None:
    path = Path(sys.executable).with_name("vllm")
    return str(path) if path.is_file() and os.access(path, os.X_OK) else None


def _supports_flag(capabilities: VLLMArgumentCapabilities | None, flag: str) -> bool:
    if capabilities is None or capabilities.detection_status != "success":
        return True
    return capabilities.supports(flag)


def _supports_kv_cache_dtype(capabilities: VLLMArgumentCapabilities | None, value: str) -> bool:
    if not _supports_flag(capabilities, "--kv-cache-dtype"):
        return False
    if capabilities is None or capabilities.detection_status != "success":
        return True
    choices = capabilities.choices_for("--kv-cache-dtype")
    return not choices or _vllm_kv_cache_dtype(value) in choices or value in choices


def _health_request(handle: ServerHandle, model: str, request_fn: RequestFn) -> RequestRecord:
    config = EndpointBenchmarkConfig(
        run_id=f"{handle.config_id}-health",
        base_url=handle.base_url,
        model=model,
        concurrency=1,
        num_requests=1,
        max_tokens=1,
        prompt="health check",
        timeout_s=5.0,
    )
    return request_fn(config, 0)


def _lifecycle_record(
    handle: ServerHandle,
    *,
    event: str,
    status: str,
    message: str,
    returncode: int | None,
) -> ManagedLifecycleRecord:
    return ManagedLifecycleRecord(
        run_id="",
        config_id=handle.config_id,
        backend=handle.backend,
        event=event,
        status=status,
        timestamp=datetime.now(timezone.utc).isoformat(),
        message=message,
        pid=handle.pid,
        pgid=handle.pgid,
        returncode=returncode,
    )


def _installed_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _vllm_dtype(dtype: str) -> str:
    return {
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
    }.get(dtype, dtype)


def _vllm_quantization(quantization: str) -> str:
    return {
        "awq-int4": "awq",
        "gptq-int4": "gptq",
        "bnb-int8": "bitsandbytes",
    }.get(quantization, quantization)


def _vllm_kv_cache_dtype(dtype: str) -> str:
    return {
        "bf16": "bfloat16",
        "fp16": "float16",
    }.get(dtype, dtype)


def _health_host(host: str) -> str:
    # Convert wildcard bind addresses to a local health check host.
    if host in {"0.0.0.0", "::"}:  # nosec B104
        return "127.0.0.1"
    return host
