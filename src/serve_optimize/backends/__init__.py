"""Backend adapters for real serving engines."""

from serve_optimize.backends.factory import (
    ATTACH_ONLY_BACKENDS,
    MANAGED_BACKEND_CHOICES,
    PLANNED_MANAGED_BACKENDS,
    SCAFFOLDED_MANAGED_BACKENDS,
    SUPPORTED_MANAGED_BACKENDS,
    UnsupportedManagedBackendError,
    create_managed_backend_adapter,
    normalize_managed_backend_name,
    validate_managed_backend_supported,
)

__all__ = [
    "ATTACH_ONLY_BACKENDS",
    "MANAGED_BACKEND_CHOICES",
    "PLANNED_MANAGED_BACKENDS",
    "SCAFFOLDED_MANAGED_BACKENDS",
    "SUPPORTED_MANAGED_BACKENDS",
    "UnsupportedManagedBackendError",
    "create_managed_backend_adapter",
    "normalize_managed_backend_name",
    "validate_managed_backend_supported",
]
