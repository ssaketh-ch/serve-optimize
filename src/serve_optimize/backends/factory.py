"""Managed backend adapter selection."""

from __future__ import annotations

from serve_optimize.backends.base import ManagedBackendAdapter
from serve_optimize.backends.sglang import SglangAdapter
from serve_optimize.backends.vllm import VllmAdapter

SUPPORTED_MANAGED_BACKENDS = ("vllm", "sglang")
SCAFFOLDED_MANAGED_BACKENDS: tuple[str, ...] = ()
PLANNED_MANAGED_BACKENDS = ("trt-llm", "tensorrt-llm", "tensorrt_llm")
ATTACH_ONLY_BACKENDS = ("tgi", "lmdeploy", "llama.cpp", "llama-cpp", "llama_cpp", "nim")
MANAGED_BACKEND_CHOICES = SUPPORTED_MANAGED_BACKENDS


class UnsupportedManagedBackendError(ValueError):
    """Raised when a managed backend is registered but not launchable."""


def normalize_managed_backend_name(backend: str | None) -> str:
    name = "" if backend is None else str(backend).strip().lower()
    if not name:
        raise UnsupportedManagedBackendError(_unsupported_message(""))
    return name


def validate_managed_backend_supported(backend: str | None) -> str:
    name = normalize_managed_backend_name(backend)
    if name in SUPPORTED_MANAGED_BACKENDS:
        return name
    if name in SCAFFOLDED_MANAGED_BACKENDS:
        raise UnsupportedManagedBackendError(
            f"Managed backend '{name}' is registered but not enabled. Currently supported: {_supported_text()}."
        )
    if name in PLANNED_MANAGED_BACKENDS:
        raise UnsupportedManagedBackendError(
            "TensorRT LLM is planned only and is not in the current Managed Mode scope. "
            "An engine build lifecycle must be designed and approved before an adapter is added."
        )
    if name in ATTACH_ONLY_BACKENDS:
        raise UnsupportedManagedBackendError(
            f"Backend '{name}' is Attach Mode only. Serve Optimize does not own its Managed Mode lifecycle."
        )
    raise UnsupportedManagedBackendError(_unsupported_message(name))


def create_managed_backend_adapter(backend: str | None) -> ManagedBackendAdapter:
    name = validate_managed_backend_supported(backend)
    if name == "vllm":
        return VllmAdapter()
    if name == "sglang":
        return SglangAdapter()
    raise UnsupportedManagedBackendError(_unsupported_message(name))


def _unsupported_message(name: str) -> str:
    return f"Unsupported managed backend '{name}'. Currently supported: {_supported_text()}."


def _supported_text() -> str:
    return ", ".join(SUPPORTED_MANAGED_BACKENDS)
