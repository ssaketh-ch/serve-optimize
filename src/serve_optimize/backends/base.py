"""Base interface for serving backend adapters."""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from serve_optimize.endpoint_benchmark import RequestFn
from serve_optimize.schemas import (
    HealthCheckResult,
    ManagedLifecycleRecord,
    ServerHandle,
    ServerLaunchSpec,
    ServingConfig,
)


@dataclass(frozen=True)
class LaunchPlan:
    command: list[str]
    environment: dict[str, str]
    notes: list[str]


def environment_with_command_dir(
    command: Sequence[str],
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    result = dict(environment or {})
    if not command:
        return result
    executable = _resolve_command_executable(command[0])
    if executable is None:
        return result
    bin_dir = str(executable.parent)
    existing_path = result.get("PATH") or os.environ.get("PATH", "")
    path_parts = [part for part in existing_path.split(os.pathsep) if part]
    if bin_dir in path_parts:
        if "PATH" not in result and existing_path:
            result["PATH"] = existing_path
        return result
    result["PATH"] = os.pathsep.join([bin_dir, *path_parts]) if path_parts else bin_dir
    return result


def _resolve_command_executable(command_name: str) -> Path | None:
    candidate = Path(command_name).expanduser()
    if candidate.is_absolute() or candidate.parent != Path("."):
        return candidate.resolve()
    resolved = shutil.which(command_name)
    return Path(resolved).resolve() if resolved else None


@dataclass(frozen=True)
class BackendArgumentCapabilities:
    backend: str
    detection_status: str = "unavailable"
    supported_arguments: frozenset[str] = field(default_factory=frozenset)
    metadata: dict[str, Any] = field(default_factory=dict)

    def supports_argument(self, argument: str) -> bool:
        return argument in self.supported_arguments


class BackendAdapter(Protocol):
    name: str

    def is_available(self) -> bool:
        ...

    def build_launch_plan(self, config: ServingConfig) -> LaunchPlan:
        ...


class ManagedBackendAdapter(Protocol):
    name: str

    def is_available(self) -> bool:
        ...

    def build_launch_spec(
        self,
        config: ServingConfig,
        *,
        host: str,
        port: int | None,
        log_dir: Path,
    ) -> ServerLaunchSpec:
        ...

    def launch_server(self, spec: ServerLaunchSpec) -> ServerHandle:
        ...

    def wait_for_health(
        self,
        handle: ServerHandle,
        *,
        model: str,
        timeout_s: float,
        request_fn: RequestFn | None = None,
    ) -> HealthCheckResult:
        ...

    def stop_server(self, handle: ServerHandle, *, timeout_s: float = 30.0) -> ManagedLifecycleRecord:
        ...


class BackendLaunchRenderer(Protocol):
    def render_launch(self, config: ServingConfig) -> object:
        ...


class BackendCandidatePolicy(Protocol):
    def generate_candidates(self, context: object, *, limit: int) -> object:
        ...


class BackendEvidenceAdapter(Protocol):
    def canonical_launch_config(self, config: ServingConfig) -> ServingConfig:
        ...
