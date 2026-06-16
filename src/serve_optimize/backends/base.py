"""Base interface for serving backend adapters."""

from __future__ import annotations

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
