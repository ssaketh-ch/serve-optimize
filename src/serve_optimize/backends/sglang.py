"""SGLang launch and managed lifecycle adapter."""

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
RELEVANT_SGLANG_FLAGS = (
    "--model-path",
    "--host",
    "--port",
    "--dtype",
    "--context-length",
    "--max-total-tokens",
    "--tp-size",
    "--tensor-parallel-size",
    "--mem-fraction-static",
    "--max-running-requests",
    "--quantization",
    "--chunked-prefill-size",
    "--disable-radix-cache",
    "--disable-cuda-graph",
    "--cuda-graph-max-bs",
    "--served-model-name",
    "--trust-remote-code",
    "--disable-piecewise-cuda-graph",
)
SGLANG_UNSUPPORTED_VLLM_FIELDS = (
    "block_size",
    "kv_cache_dtype",
    "enforce_eager",
    "max_num_batched_tokens",
    "enable_chunked_prefill",
    "max_cudagraph_capture_size",
    "enable_prefix_caching",
)


@dataclass(frozen=True)
class SGLangArgumentCapabilities:
    executable: str
    launch_command: tuple[str, ...] = field(default_factory=tuple)
    version: str | None = None
    supported_flags: frozenset[str] = field(default_factory=frozenset)
    option_choices: dict[str, frozenset[str]] = field(default_factory=dict)
    help_hash: str | None = None
    detection_status: str = "unavailable"
    detection_error: str | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def supports(self, flag: str) -> bool:
        return flag in self.supported_flags

    def choices_for(self, flag: str) -> frozenset[str]:
        return self.option_choices.get(flag, frozenset())

    def context_length_flag(self) -> str | None:
        if self.supports("--context-length"):
            return "--context-length"
        if self.supports("--max-total-tokens"):
            return "--max-total-tokens"
        return None

    def tensor_parallel_flag(self) -> str | None:
        if self.supports("--tp-size"):
            return "--tp-size"
        if self.supports("--tensor-parallel-size"):
            return "--tensor-parallel-size"
        return None

    def to_artifact(self) -> dict[str, object]:
        return {
            "schema_version": "sglang-argument-capabilities/v1",
            "backend": "sglang",
            "executable": self.executable,
            "launch_command": list(self.launch_command),
            "version": self.version,
            "detection_status": self.detection_status,
            "detection_error": self.detection_error,
            "help_hash": self.help_hash,
            "supported_flags": {flag: self.supports(flag) for flag in RELEVANT_SGLANG_FLAGS},
            "option_choices": {flag: sorted(values) for flag, values in sorted(self.option_choices.items())},
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class SGLangRenderedLaunch:
    command: list[str]
    canonical_config: ServingConfig
    rendered_fields: dict[str, object] = field(default_factory=dict)
    omitted_fields: dict[str, str] = field(default_factory=dict)
    unsupported_fields: dict[str, str] = field(default_factory=dict)
    unavailable_fields: dict[str, str] = field(default_factory=dict)
    flag_aliases: dict[str, str] = field(default_factory=dict)
    capabilities_help_hash: str | None = None
    backend_metadata: dict[str, object] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema_version": "sglang-rendered-launch/v1",
            "backend": "sglang",
            "command": self.command,
            "canonical_config": self.canonical_config,
            "rendered_fields": self.rendered_fields,
            "omitted_fields": self.omitted_fields,
            "unsupported_fields": self.unsupported_fields,
            "unavailable_fields": self.unavailable_fields,
            "flag_aliases": self.flag_aliases,
            "capabilities_help_hash": self.capabilities_help_hash,
            "backend_metadata": self.backend_metadata,
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


class SglangAdapter:
    name = "sglang"

    def __init__(
        self,
        *,
        popen_factory: PopenFactory | None = None,
        killpg_fn: KillpgFn | None = None,
        argument_capabilities: SGLangArgumentCapabilities | None = None,
    ) -> None:
        self._popen_factory = popen_factory or subprocess.Popen
        self._killpg_fn = killpg_fn or os.killpg
        self._argument_capabilities = argument_capabilities
        self._processes: dict[int, subprocess.Popen[Any]] = {}
        self._launched_pgids: set[int] = set()

    def is_available(self) -> bool:
        capabilities = self.argument_capabilities()
        return (
            capabilities.detection_status == "success"
            and bool(capabilities.launch_command)
            and capabilities.supports("--model-path")
        )

    def build_launch_plan(self, config: ServingConfig) -> LaunchPlan:
        rendered = render_sglang_launch(config, capabilities=self.argument_capabilities())
        notes = ["Launch plan only; process management lands in the managed runner."]
        if rendered.unsupported_fields:
            notes.append("Some fields are unsupported by the detected SGLang launch surface.")
        return LaunchPlan(command=rendered.command, environment={}, notes=notes)

    def build_launch_spec(
        self,
        config: ServingConfig,
        *,
        host: str,
        port: int | None,
        log_dir: Path,
    ) -> ServerLaunchSpec:
        if config.backend != self.name:
            raise ValueError(f"SGLang managed adapter cannot launch backend '{config.backend}'.")
        capabilities = self.argument_capabilities()
        if not self.is_available():
            reason = capabilities.detection_error or "SGLang launch surface is unavailable."
            raise RuntimeError(f"Managed backend 'sglang' is unavailable: {reason}")
        resolved_port = port if port is not None else allocate_port(host)
        validate_port_available(host, resolved_port)
        rendered = render_sglang_launch(config, host=host, port=resolved_port, capabilities=capabilities)
        if rendered.unsupported_fields:
            fields = ", ".join(sorted(rendered.unsupported_fields))
            raise ValueError(f"SGLang launch config has unsupported fields: {fields}.")
        grpc_port = _allocate_distinct_port(host, resolved_port)
        candidate_log_dir = log_dir / config.id
        base_url = f"http://{_health_host(host)}:{resolved_port}/v1"
        metadata = self.backend_metadata()
        metadata["rendered_launch"] = rendered.to_metadata()
        metadata["allocated_grpc_port"] = grpc_port
        return ServerLaunchSpec(
            config_id=config.id,
            backend=self.name,
            model_id=config.model_id,
            host=host,
            port=resolved_port,
            base_url=base_url,
            command=rendered.command,
            environment={"SGLANG_GRPC_PORT": str(grpc_port)},
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
            message="Stopped launched SGLang process group.",
            returncode=returncode,
        )

    def backend_metadata(self) -> dict[str, object]:
        capabilities = self.argument_capabilities()
        return {
            "adapter": self.name,
            "backend": self.name,
            "executable": capabilities.executable,
            "version": capabilities.version,
            "argument_detection_status": capabilities.detection_status,
            "argument_capabilities_help_hash": capabilities.help_hash,
            "capability_artifact": "sglang_argument_capabilities.json",
        }

    def argument_capabilities(self) -> SGLangArgumentCapabilities:
        if self._argument_capabilities is not None:
            return self._argument_capabilities
        return detect_sglang_argument_capabilities()


def allocate_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _allocate_distinct_port(host: str, port: int) -> int:
    for _ in range(16):
        candidate = allocate_port(host)
        if candidate != port:
            return candidate
    raise RuntimeError("could not allocate a distinct SGLang gRPC port")


def validate_port_available(host: str, port: int) -> None:
    if port < 1 or port > 65535:
        raise ValueError("port must be between 1 and 65535.")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError as exc:
            raise ValueError(f"port {port} is not available on {host}: {exc}") from exc


def detect_sglang_argument_capabilities(*, timeout_s: float = 30.0) -> SGLangArgumentCapabilities:
    return _detect_sglang_argument_capabilities_cached(sys.executable, shutil.which("sglang.launch_server") or "", shutil.which("sglang") or "", timeout_s)


@lru_cache(maxsize=8)
def _detect_sglang_argument_capabilities_cached(
    python_executable: str,
    launch_server_executable: str,
    sglang_executable: str,
    timeout_s: float,
) -> SGLangArgumentCapabilities:
    version = _installed_version("sglang") or _sglang_cli_version(sglang_executable, timeout_s=timeout_s)
    candidates: list[tuple[list[str], tuple[str, ...]]] = [
        (
            [python_executable, "-m", "sglang.launch_server", "--help"],
            (python_executable, "-m", "sglang.launch_server"),
        )
    ]
    if launch_server_executable:
        candidates.append(([launch_server_executable, "--help"], (launch_server_executable,)))
    if sglang_executable:
        candidates.append(([sglang_executable, "serve", "--help"], (sglang_executable, "serve")))
        candidates.append(([sglang_executable, "launch_server", "--help"], (sglang_executable, "launch_server")))
        candidates.append(([sglang_executable, "--help"], (sglang_executable, "launch_server")))

    errors: list[str] = []
    timed_out = False
    for command, launch_prefix in candidates:
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            errors.append(f"{' '.join(command)}: TimeoutExpired: {exc}")
            continue
        except OSError as exc:
            errors.append(f"{' '.join(command)}: {exc.__class__.__name__}: {exc}")
            continue
        help_text = _select_sglang_help_text(completed.stdout, completed.stderr)
        if completed.returncode != 0:
            errors.append(f"{' '.join(command)} exited {completed.returncode}")
            continue
        if "--model-path" not in help_text:
            errors.append(f"{' '.join(command)} did not expose --model-path")
            continue
        return parse_sglang_argument_capabilities(
            help_text,
            executable=command[0],
            launch_command=launch_prefix,
            version=version,
        )

    return SGLangArgumentCapabilities(
        executable=launch_server_executable or sglang_executable or python_executable,
        version=version,
        detection_status="timeout" if timed_out else "unavailable" if not (launch_server_executable or sglang_executable) else "error",
        detection_error="; ".join(errors) if errors else "SGLang executable or module was not found.",
        warnings=tuple(errors),
    )


def parse_sglang_argument_capabilities(
    help_text: str,
    *,
    executable: str = "sglang",
    launch_command: tuple[str, ...] = ("sglang", "launch_server"),
    version: str | None = None,
) -> SGLangArgumentCapabilities:
    flags = frozenset(re.findall(r"--[A-Za-z0-9][A-Za-z0-9_-]*", help_text))
    option_choices: dict[str, frozenset[str]] = {}
    dtype_choices = _parse_option_choices(help_text, "--dtype")
    if dtype_choices:
        option_choices["--dtype"] = frozenset(dtype_choices)
    quantization_choices = _parse_option_choices(help_text, "--quantization")
    if quantization_choices:
        option_choices["--quantization"] = frozenset(quantization_choices)
    help_hash = _sglang_capability_hash(flags, option_choices)
    warnings = []
    if "--model-path" not in flags:
        warnings.append("Detected SGLang help did not expose --model-path.")
    return SGLangArgumentCapabilities(
        executable=executable,
        launch_command=launch_command,
        version=version,
        supported_flags=flags,
        option_choices=option_choices,
        help_hash=help_hash,
        detection_status="success",
        warnings=tuple(warnings),
    )


def _select_sglang_help_text(stdout: str, stderr: str) -> str:
    for stream in (stdout, stderr):
        if "usage:" in stream.lower() and "--model-path" in stream:
            return stream
    return f"{stdout}\n{stderr}"


def _sglang_cli_version(sglang_executable: str, *, timeout_s: float) -> str | None:
    if not sglang_executable:
        return None
    try:
        completed = subprocess.run(
            [sglang_executable, "version"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    text = f"{completed.stdout}\n{completed.stderr}"
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("sglang version:"):
            return line.partition(":")[2].strip() or None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else None


def _sglang_capability_hash(
    flags: frozenset[str],
    option_choices: dict[str, frozenset[str]],
) -> str:
    payload = {
        "flags": sorted(flags),
        "option_choices": {
            flag: sorted(values)
            for flag, values in sorted(option_choices.items())
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def render_sglang_launch(
    config: ServingConfig,
    *,
    host: str | None = None,
    port: int | None = None,
    capabilities: SGLangArgumentCapabilities | None = None,
) -> SGLangRenderedLaunch:
    rendered_fields: dict[str, object] = {}
    omitted_fields: dict[str, str] = {}
    unsupported_fields: dict[str, str] = {}
    unavailable_fields: dict[str, str] = {}
    flag_aliases: dict[str, str] = {}
    capabilities = capabilities or SGLangArgumentCapabilities(executable="sglang", detection_status="unavailable")
    command = list(capabilities.launch_command or (sys.executable, "-m", "sglang.launch_server"))
    capabilities_known = capabilities.detection_status == "success"

    if _supports_flag(capabilities, "--model-path"):
        command.extend(["--model-path", config.model_id])
        rendered_fields["model"] = config.model_id
    elif not capabilities_known:
        unavailable_fields["model"] = _capability_unavailable_reason(capabilities, "--model-path")
    else:
        unsupported_fields["model"] = "Detected SGLang launch surface does not support --model-path."

    if host is not None:
        if _supports_flag(capabilities, "--host"):
            command.extend(["--host", host])
            rendered_fields["host"] = host
        elif not capabilities_known:
            unavailable_fields["host"] = _capability_unavailable_reason(capabilities, "--host")
        else:
            unsupported_fields["host"] = "Detected SGLang launch surface does not support --host."
    if port is not None:
        if _supports_flag(capabilities, "--port"):
            command.extend(["--port", str(port)])
            rendered_fields["port"] = port
        elif not capabilities_known:
            unavailable_fields["port"] = _capability_unavailable_reason(capabilities, "--port")
        else:
            unsupported_fields["port"] = "Detected SGLang launch surface does not support --port."

    if (config.extra or {}).get("backend_defaults") is True:
        canonical_config = replace(
            config,
            block_size=None,
            kv_cache_dtype=None,
            enforce_eager=None,
            max_num_batched_tokens=None,
            enable_chunked_prefill=None,
            max_cudagraph_capture_size=None,
            enable_prefix_caching=None,
            kv_cache_policy="backend-default",
            scheduler="backend-default",
        )
        rendered_fields["backend_defaults"] = True
        omitted_fields.update(
            {
                "dtype": "backend default",
                "quantization": "backend default",
                "max_model_len": "backend default",
                "gpu_memory_utilization": "backend default",
                "max_num_seqs": "backend default",
                "tensor_parallel_size": "backend default",
            }
        )
        return SGLangRenderedLaunch(
            command=command,
            canonical_config=canonical_config,
            rendered_fields=rendered_fields,
            omitted_fields=omitted_fields,
            unsupported_fields=unsupported_fields,
            unavailable_fields=unavailable_fields,
            flag_aliases=flag_aliases,
            capabilities_help_hash=capabilities.help_hash,
            backend_metadata={
                "backend": "sglang",
                "capability_detection_status": capabilities.detection_status,
                "capability_help_hash": capabilities.help_hash,
            },
        )

    if config.dtype:
        rendered_dtype = _sglang_dtype(config.dtype)
        if _supports_dtype(capabilities, rendered_dtype):
            command.extend(["--dtype", rendered_dtype])
            rendered_fields["dtype"] = rendered_dtype
        elif _supports_flag(capabilities, "--dtype"):
            unsupported_fields["dtype"] = f"Detected SGLang launch surface does not list dtype '{rendered_dtype}'."
        elif not capabilities_known:
            unavailable_fields["dtype"] = _capability_unavailable_reason(capabilities, "--dtype")
        else:
            omitted_fields["dtype"] = "SGLang help did not expose --dtype."

    context_flag = capabilities.context_length_flag()
    if context_flag is not None:
        command.extend([context_flag, str(config.max_context_tokens)])
        rendered_fields["max_model_len"] = config.max_context_tokens
        if context_flag != "--context-length":
            flag_aliases["max_model_len"] = context_flag
    elif not capabilities_known:
        unavailable_fields["max_model_len"] = _capability_unavailable_reason(
            capabilities,
            "--context-length or --max-total-tokens",
        )
    else:
        omitted_fields["max_model_len"] = "SGLang help did not expose a context length flag."

    tp_flag = capabilities.tensor_parallel_flag()
    if config.tensor_parallelism > 1:
        if tp_flag is not None:
            command.extend([tp_flag, str(config.tensor_parallelism)])
            rendered_fields["tensor_parallel_size"] = config.tensor_parallelism
            if tp_flag != "--tp-size":
                flag_aliases["tensor_parallel_size"] = tp_flag
        elif not capabilities_known:
            unavailable_fields["tensor_parallel_size"] = _capability_unavailable_reason(
                capabilities,
                "--tp-size or --tensor-parallel-size",
            )
        else:
            unsupported_fields["tensor_parallel_size"] = "Detected SGLang launch surface does not support tensor parallel size."
    elif tp_flag is not None:
        omitted_fields["tensor_parallel_size"] = "default tensor parallel size is 1."

    served_model_name = _served_model_name(config)
    if served_model_name:
        if _supports_flag(capabilities, "--served-model-name"):
            command.extend(["--served-model-name", served_model_name])
            rendered_fields["served_model_name"] = served_model_name
        elif not capabilities_known:
            unavailable_fields["served_model_name"] = _capability_unavailable_reason(
                capabilities,
                "--served-model-name",
            )
        else:
            omitted_fields["served_model_name"] = "SGLang help did not expose --served-model-name."

    if _trust_remote_code(config):
        if _supports_flag(capabilities, "--trust-remote-code"):
            command.append("--trust-remote-code")
            rendered_fields["trust_remote_code"] = True
        elif not capabilities_known:
            unavailable_fields["trust_remote_code"] = _capability_unavailable_reason(
                capabilities,
                "--trust-remote-code",
            )
        else:
            unsupported_fields["trust_remote_code"] = "Detected SGLang launch surface does not support --trust-remote-code."

    if _disable_piecewise_cuda_graph(config):
        if _supports_flag(capabilities, "--disable-piecewise-cuda-graph"):
            command.append("--disable-piecewise-cuda-graph")
            rendered_fields["disable_piecewise_cuda_graph"] = True
        elif not capabilities_known:
            unavailable_fields["disable_piecewise_cuda_graph"] = _capability_unavailable_reason(
                capabilities,
                "--disable-piecewise-cuda-graph",
            )
        else:
            unsupported_fields["disable_piecewise_cuda_graph"] = (
                "Detected SGLang launch surface does not support --disable-piecewise-cuda-graph."
            )

    if config.max_batch_size > 1:
        if _supports_flag(capabilities, "--max-running-requests"):
            command.extend(["--max-running-requests", str(config.max_batch_size)])
            rendered_fields["max_running_requests"] = config.max_batch_size
        elif not capabilities_known:
            unavailable_fields["max_num_seqs"] = _capability_unavailable_reason(
                capabilities,
                "--max-running-requests",
            )
        else:
            unsupported_fields["max_num_seqs"] = "Detected SGLang launch surface does not support --max-running-requests."
    else:
        omitted_fields["max_num_seqs"] = "default SGLang scheduler behavior."

    if config.gpu_memory_utilization > 0:
        if _supports_flag(capabilities, "--mem-fraction-static"):
            command.extend(["--mem-fraction-static", str(config.gpu_memory_utilization)])
            rendered_fields["gpu_memory_utilization"] = config.gpu_memory_utilization
        elif not capabilities_known:
            unavailable_fields["gpu_memory_utilization"] = _capability_unavailable_reason(
                capabilities,
                "--mem-fraction-static",
            )
        else:
            unsupported_fields["gpu_memory_utilization"] = (
                "Detected SGLang launch surface does not support --mem-fraction-static."
            )
    else:
        omitted_fields["gpu_memory_utilization"] = "backend default memory fraction."

    quantization = _sglang_quantization(config.quantization)
    if quantization != "none":
        if _supports_choice(capabilities, "--quantization", quantization):
            command.extend(["--quantization", quantization])
            rendered_fields["quantization"] = quantization
        elif not capabilities_known:
            unavailable_fields["quantization"] = _capability_unavailable_reason(
                capabilities,
                "--quantization",
            )
        elif capabilities.supports("--quantization"):
            unsupported_fields["quantization"] = (
                f"Detected SGLang launch surface does not list quantization '{quantization}'."
            )
        else:
            unsupported_fields["quantization"] = "Detected SGLang launch surface does not support --quantization."

    extra = config.extra or {}
    _render_sglang_integer_option(
        command,
        capabilities,
        capabilities_known=capabilities_known,
        field_name="chunked_prefill_size",
        flag="--chunked-prefill-size",
        value=extra.get("chunked_prefill_size"),
        rendered_fields=rendered_fields,
        unsupported_fields=unsupported_fields,
        unavailable_fields=unavailable_fields,
    )
    _render_sglang_boolean_option(
        command,
        capabilities,
        capabilities_known=capabilities_known,
        field_name="disable_radix_cache",
        flag="--disable-radix-cache",
        value=extra.get("disable_radix_cache"),
        rendered_fields=rendered_fields,
        unsupported_fields=unsupported_fields,
        unavailable_fields=unavailable_fields,
    )
    _render_sglang_boolean_option(
        command,
        capabilities,
        capabilities_known=capabilities_known,
        field_name="disable_cuda_graph",
        flag="--disable-cuda-graph",
        value=extra.get("disable_cuda_graph"),
        rendered_fields=rendered_fields,
        unsupported_fields=unsupported_fields,
        unavailable_fields=unavailable_fields,
    )
    _render_sglang_integer_option(
        command,
        capabilities,
        capabilities_known=capabilities_known,
        field_name="cuda_graph_max_bs",
        flag="--cuda-graph-max-bs",
        value=extra.get("cuda_graph_max_bs"),
        rendered_fields=rendered_fields,
        unsupported_fields=unsupported_fields,
        unavailable_fields=unavailable_fields,
    )

    for field_name in SGLANG_UNSUPPORTED_VLLM_FIELDS:
        if getattr(config, field_name) is not None:
            unsupported_fields[field_name] = f"SGLang does not translate vLLM field {field_name}."
        else:
            omitted_fields[field_name] = "not set"

    canonical_extra = dict(extra)
    for field_name in (
        "disable_piecewise_cuda_graph",
        "disable_radix_cache",
        "disable_cuda_graph",
        "trust_remote_code",
    ):
        if canonical_extra.get(field_name) is not True:
            canonical_extra.pop(field_name, None)
    canonical_config = replace(
        config,
        quantization=quantization,
        block_size=None,
        kv_cache_dtype=None,
        enforce_eager=None,
        max_num_batched_tokens=None,
        enable_chunked_prefill=None,
        max_cudagraph_capture_size=None,
        enable_prefix_caching=None,
        kv_cache_policy="backend-default",
        scheduler="backend-default",
        extra=canonical_extra,
    )
    return SGLangRenderedLaunch(
        command=command,
        canonical_config=canonical_config,
        rendered_fields=rendered_fields,
        omitted_fields=omitted_fields,
        unsupported_fields=unsupported_fields,
        unavailable_fields=unavailable_fields,
        flag_aliases=flag_aliases,
        capabilities_help_hash=capabilities.help_hash,
        backend_metadata={
            "backend": "sglang",
            "capability_detection_status": capabilities.detection_status,
            "capability_help_hash": capabilities.help_hash,
        },
    )


def sglang_command(
    config: ServingConfig,
    *,
    host: str | None = None,
    port: int | None = None,
    capabilities: SGLangArgumentCapabilities | None = None,
) -> list[str]:
    return render_sglang_launch(config, host=host, port=port, capabilities=capabilities).command


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


def _supports_flag(capabilities: SGLangArgumentCapabilities, flag: str) -> bool:
    return capabilities.detection_status == "success" and capabilities.supports(flag)


def _supports_dtype(capabilities: SGLangArgumentCapabilities, dtype: str) -> bool:
    if not _supports_flag(capabilities, "--dtype"):
        return False
    choices = capabilities.choices_for("--dtype")
    return not choices or dtype in choices or _compact_dtype(dtype) in choices


def _supports_choice(capabilities: SGLangArgumentCapabilities, flag: str, value: str) -> bool:
    if not _supports_flag(capabilities, flag):
        return False
    choices = capabilities.choices_for(flag)
    return not choices or value in choices


def _capability_unavailable_reason(capabilities: SGLangArgumentCapabilities, flag: str) -> str:
    detail = capabilities.detection_error or f"capability detection status is {capabilities.detection_status}"
    return f"Could not verify {flag}: {detail}"


def _render_sglang_integer_option(
    command: list[str],
    capabilities: SGLangArgumentCapabilities,
    *,
    capabilities_known: bool,
    field_name: str,
    flag: str,
    value: object,
    rendered_fields: dict[str, object],
    unsupported_fields: dict[str, str],
    unavailable_fields: dict[str, str],
) -> None:
    if value is None:
        return
    if _supports_flag(capabilities, flag):
        command.extend([flag, str(value)])
        rendered_fields[field_name] = value
    elif not capabilities_known:
        unavailable_fields[field_name] = _capability_unavailable_reason(capabilities, flag)
    else:
        unsupported_fields[field_name] = f"Detected SGLang launch surface does not support {flag}."


def _render_sglang_boolean_option(
    command: list[str],
    capabilities: SGLangArgumentCapabilities,
    *,
    capabilities_known: bool,
    field_name: str,
    flag: str,
    value: object,
    rendered_fields: dict[str, object],
    unsupported_fields: dict[str, str],
    unavailable_fields: dict[str, str],
) -> None:
    if value is not True:
        return
    if _supports_flag(capabilities, flag):
        command.append(flag)
        rendered_fields[field_name] = True
    elif not capabilities_known:
        unavailable_fields[field_name] = _capability_unavailable_reason(capabilities, flag)
    else:
        unsupported_fields[field_name] = f"Detected SGLang launch surface does not support {flag}."


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


def _sglang_dtype(dtype: str) -> str:
    return {
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
    }.get(dtype, dtype)


def _sglang_quantization(quantization: str) -> str:
    return {
        "awq-int4": "awq",
        "gptq-int4": "gptq",
    }.get(quantization.strip().lower(), quantization.strip().lower())


def _compact_dtype(dtype: str) -> str:
    return {
        "bfloat16": "bf16",
        "float16": "fp16",
        "float32": "fp32",
    }.get(dtype, dtype)


def _served_model_name(config: ServingConfig) -> str | None:
    value = (config.extra or {}).get("served_model_name")
    return str(value) if value else None


def _trust_remote_code(config: ServingConfig) -> bool:
    return bool((config.extra or {}).get("trust_remote_code"))


def _disable_piecewise_cuda_graph(config: ServingConfig) -> bool:
    return (config.extra or {}).get("disable_piecewise_cuda_graph") is True


def _health_host(host: str) -> str:
    # Convert wildcard bind addresses to a local health check host.
    if host in {"0.0.0.0", "::"}:  # nosec B104
        return "127.0.0.1"
    return host
