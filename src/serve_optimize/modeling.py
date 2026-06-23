"""Model metadata inference used before heavyweight model inspection exists."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .schemas import ModelCapabilityMetadata, ModelSpec

KNOWN_MODELS: dict[str, tuple[float, int, str]] = {
    "tiny-random-gpt2": (0.0001, 1024, "gpt2"),
    "tiny-random-llamaforcausallm": (0.0001, 1024, "llama"),
    "tinyllama": (1.1, 2048, "llama"),
    "llama-3.1-8b": (8.0, 131072, "llama"),
    "llama-3-8b": (8.0, 8192, "llama"),
    "mistral-7b": (7.3, 32768, "mistral"),
    "mixtral-8x7b": (46.7, 32768, "mixtral"),
    "qwen2.5-7b": (7.6, 32768, "qwen"),
    "qwen2.5-14b": (14.7, 32768, "qwen"),
    "qwen3-32b": (32.0, 32768, "qwen"),
    "falcon-7b": (7.0, 2048, "falcon"),
}


def infer_model_spec(model_id: str, max_context_tokens: int | None = None) -> ModelSpec:
    normalized = model_id.lower()
    for key, (params_b, context, family) in KNOWN_MODELS.items():
        if key in normalized:
            return ModelSpec(
                model_id=model_id,
                parameter_count_b=params_b,
                max_context_tokens=max_context_tokens or context,
                family=family,
            )

    params_b = _parse_parameter_count(normalized)
    family = _infer_family(normalized)
    return ModelSpec(
        model_id=model_id,
        parameter_count_b=params_b,
        max_context_tokens=max_context_tokens or 4096,
        family=family,
    )


def infer_model_capability_metadata(
    model_id: str,
    *,
    allow_remote_download: bool = False,
) -> ModelCapabilityMetadata:
    model_path = Path(model_id).expanduser()
    if model_path.exists():
        config_path = model_path / "config.json" if model_path.is_dir() else model_path
        return _metadata_from_config(
            model_id,
            config_path,
            is_local_path=True,
            source_label="Local",
        )
    config_path, notes, warnings = _resolve_remote_config_path(
        model_id,
        allow_download=allow_remote_download,
    )
    if config_path is None:
        return ModelCapabilityMetadata(
            model_id=model_id,
            metadata_known=False,
            is_local_path=False,
            notes=notes or ["Remote model metadata is unavailable."],
            warnings=warnings,
        )
    return _metadata_from_config(
        model_id,
        config_path,
        is_local_path=False,
        source_label="Remote",
        notes=notes,
    )


def _metadata_from_config(
    model_id: str,
    config_path: Path,
    *,
    is_local_path: bool,
    source_label: str,
    notes: list[str] | None = None,
) -> ModelCapabilityMetadata:
    if config_path.name != "config.json" or not config_path.exists():
        return ModelCapabilityMetadata(
            model_id=model_id,
            metadata_known=False,
            is_local_path=is_local_path,
            config_path=str(config_path),
            notes=notes or [],
            warnings=[f"{source_label} model config.json was not found."],
        )
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return ModelCapabilityMetadata(
            model_id=model_id,
            metadata_known=False,
            is_local_path=is_local_path,
            config_path=str(config_path),
            notes=notes or [],
            warnings=[f"{source_label} model config.json could not be read: {exc.__class__.__name__}: {exc}"],
        )
    quantization_config = payload.get("quantization_config")
    if not isinstance(quantization_config, dict):
        quantization_config = {}
    quant_method = quantization_config.get("quant_method")
    torch_dtype = payload.get("torch_dtype", payload.get("dtype"))
    return ModelCapabilityMetadata(
        model_id=model_id,
        metadata_known=True,
        is_local_path=is_local_path,
        config_path=str(config_path),
        torch_dtype=str(torch_dtype) if torch_dtype is not None else None,
        quantization_method=str(quant_method).lower() if quant_method is not None else None,
        quantization_config=quantization_config,
        notes=notes or [],
    )


def _resolve_remote_config_path(
    model_id: str,
    *,
    allow_download: bool,
) -> tuple[Path | None, list[str], list[str]]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None, ["Remote model config lookup requires huggingface_hub."], []

    modes = (True, False) if allow_download else (True,)
    warnings: list[str] = []
    for local_files_only in modes:
        try:
            path = hf_hub_download(
                repo_id=model_id,
                filename="config.json",
                local_files_only=local_files_only,
            )
        except Exception as exc:
            if local_files_only and allow_download:
                continue
            mode = "cached" if local_files_only else "downloaded"
            warnings.append(f"Remote model {mode} config lookup failed: {exc.__class__.__name__}: {exc}")
            continue
        note = "Remote model config was read from the local Hub cache."
        if not local_files_only:
            note = "Remote model config was downloaded before candidate generation."
        return Path(path), [note], warnings
    return None, ["Remote model config was not found in the local Hub cache."], warnings


def _parse_parameter_count(model_id: str) -> float:
    match = re.search(r"(?P<count>\d+(?:\.\d+)?)\s*b(?:\b|-|_)", model_id)
    if match:
        return float(match.group("count"))
    match = re.search(r"(?P<count>\d+(?:\.\d+)?)\s*m(?:\b|-|_)", model_id)
    if match:
        return float(match.group("count")) / 1000.0
    return 7.0


def _infer_family(model_id: str) -> str:
    for family in ("llama", "mistral", "mixtral", "qwen", "falcon", "gemma", "phi"):
        if family in model_id:
            return family
    return "unknown"
