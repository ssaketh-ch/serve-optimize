"""Backend adapters for real serving engines."""

from serve_optimize.backends.factory import (
    MANAGED_BACKEND_CHOICES,
    SCAFFOLDED_MANAGED_BACKENDS,
    SUPPORTED_MANAGED_BACKENDS,
    UnsupportedManagedBackendError,
    create_managed_backend_adapter,
    normalize_managed_backend_name,
    validate_managed_backend_supported,
)

__all__ = [
    "MANAGED_BACKEND_CHOICES",
    "SCAFFOLDED_MANAGED_BACKENDS",
    "SUPPORTED_MANAGED_BACKENDS",
    "UnsupportedManagedBackendError",
    "create_managed_backend_adapter",
    "normalize_managed_backend_name",
    "validate_managed_backend_supported",
]
