"""Model download and local cache helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

TINY_MODEL_IDS = [
    "hf-internal-testing/tiny-random-gpt2",
    "hf-internal-testing/tiny-random-LlamaForCausalLM",
]


@dataclass(frozen=True)
class LocalModel:
    model_id: str
    path: str
    revision: str | None = None


def download_model(model_id: str, cache_dir: Path | None = None, revision: str | None = None) -> LocalModel:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("huggingface_hub is required. Install with: pip install -e '.[runtime]'") from exc

    path = snapshot_download(
        repo_id=model_id,
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir else None,
        local_files_only=False,
    )
    return LocalModel(model_id=model_id, path=path, revision=revision)


def download_tiny_models(cache_dir: Path | None = None) -> list[LocalModel]:
    return [download_model(model_id, cache_dir=cache_dir) for model_id in TINY_MODEL_IDS]
