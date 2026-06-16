"""Artifact serialization helpers."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from .schemas import BenchmarkResult, to_dict


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_dict(row), sort_keys=True) + "\n")


def load_result_jsonl(path: Path) -> list[BenchmarkResult]:
    # Full dataclass rehydration is intentionally deferred until artifact schemas settle.
    raise NotImplementedError("Result JSONL rehydration will be implemented with the stable artifact schema.")

