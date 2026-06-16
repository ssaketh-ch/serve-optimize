"""Parse AIConfigurator predictions for Serve Optimize planning."""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path

from .schemas import AICPrediction, ServeCandidate

INT_FIELDS = {"isl", "osl", "prefix", "concurrency", "bs", "global_bs", "tp", "pp", "dp", "moe_tp", "moe_ep"}
FLOAT_FIELDS = {"request_rate", "ttft", "tpot", "request_latency", "seq/s", "tokens/s", "tokens/s/gpu", "tokens/s/user", "memory", "power_w"}
KEY_ALIASES = {
    "batch": "bs",
    "batch_size": "bs",
    "batchsize": "bs",
    "global_batch_size": "global_bs",
    "input_length": "isl",
    "input_tokens": "isl",
    "latency": "request_latency",
    "max_new_tokens": "osl",
    "output_length": "osl",
    "output_tokens": "osl",
    "request_latency_ms": "request_latency",
    "throughput": "tokens/s",
    "throughput_tokens_per_sec": "tokens/s",
    "tokens_per_second": "tokens/s",
    "ttft_ms": "ttft",
}


def parse_aiconfigurator_best_configs(path: str, top_k: int | None = None) -> list[ServeCandidate]:
    """Parse ranked AIConfigurator best_config_topn.csv rows into candidates."""

    csv_path = Path(path)
    candidates: list[ServeCandidate] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            if top_k is not None and len(candidates) >= top_k:
                break
            candidates.append(candidate_from_row(row, rank=index, source=str(csv_path)))
    return candidates


def parse_aiconfigurator_text_candidates(text: str, *, source: str = "aiconfigurator-output", top_k: int | None = None) -> list[ServeCandidate]:
    """Parse simple AIConfigurator text tables when CSV output is unavailable."""

    rows = _table_rows_from_text(text)
    candidates: list[ServeCandidate] = []
    for index, row in enumerate(rows, start=1):
        if top_k is not None and len(candidates) >= top_k:
            break
        rank = int(row.pop("rank", "") or index) if str(row.get("rank", "")).strip().isdigit() else index
        candidates.append(candidate_from_row(row, rank=rank, source=source))
    return candidates


def candidate_from_row(row: dict[str, str | None], rank: int, source: str) -> ServeCandidate:
    values = {_normalize_key(key): value for key, value in row.items() if key is not None}
    raw = {str(key): value for key, value in row.items()}
    if None in row:
        raw["parse_error"] = "CSV row has more values than headers."
    return ServeCandidate(
        candidate_id=f"aic-rank-{rank:04d}",
        rank=rank,
        source=source,
        model=_text(values, "model"),
        backend=_text(values, "backend"),
        backend_version=_text(values, "version"),
        system=_text(values, "system"),
        isl=_int(values, "isl"),
        osl=_int(values, "osl"),
        prefix=_int(values, "prefix"),
        concurrency=_int(values, "concurrency"),
        request_rate=_float(values, "request_rate"),
        batch_size=_int(values, "bs"),
        global_batch_size=_int(values, "global_bs"),
        tp=_int(values, "tp"),
        pp=_int(values, "pp"),
        dp=_int(values, "dp"),
        moe_tp=_int(values, "moe_tp"),
        moe_ep=_int(values, "moe_ep"),
        parallel=_text(values, "parallel"),
        gemm=_text(values, "gemm"),
        kvcache=_text(values, "kvcache"),
        fmha=_text(values, "fmha"),
        moe=_text(values, "moe"),
        comm=_text(values, "comm"),
        predicted_ttft_ms=_float(values, "ttft"),
        predicted_tpot_ms=_float(values, "tpot"),
        predicted_request_latency_ms=_float(values, "request_latency"),
        predicted_seq_s=_float(values, "seq/s"),
        predicted_tokens_s=_float(values, "tokens/s"),
        predicted_tokens_s_per_gpu=_float(values, "tokens/s/gpu"),
        predicted_tokens_s_per_user=_float(values, "tokens/s/user"),
        predicted_memory_gb=_float(values, "memory"),
        predicted_power_w=_float(values, "power_w"),
        raw=raw,
    )


def parse_aiconfig_prediction_csv(path: Path) -> AICPrediction:
    """Parse the first row from an AIConfigurator result CSV."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        row = next(reader, None)
    if row is None:
        raise ValueError(f"AIConfigurator CSV has no rows: {path}")
    return prediction_from_row(row, source_path=path)


def prediction_from_row(row: dict[str, str | None], source_path: Path | None = None) -> AICPrediction:
    values = {_normalize_key(key): value for key, value in row.items()}
    raw = {key: value for key, value in row.items()}
    return AICPrediction(
        backend=_text(values, "backend"),
        version=_text(values, "version"),
        system=_text(values, "system"),
        model=_text(values, "model"),
        isl=_int(values, "isl"),
        osl=_int(values, "osl"),
        concurrency=_int(values, "concurrency"),
        request_rate=_float(values, "request_rate"),
        bs=_int(values, "bs"),
        ttft=_float(values, "ttft"),
        tpot=_float(values, "tpot"),
        request_latency=_float(values, "request_latency"),
        tokens_s=_float(values, "tokens/s"),
        tokens_s_gpu=_float(values, "tokens/s/gpu"),
        tokens_s_user=_float(values, "tokens/s/user"),
        tp=_int(values, "tp"),
        pp=_int(values, "pp"),
        dp=_int(values, "dp"),
        parallel=_text(values, "parallel"),
        memory=_float(values, "memory"),
        power_w=_float(values, "power_w"),
        source_path=str(source_path) if source_path else None,
        raw=raw,
    )


def _normalize_key(key: str | None) -> str:
    if key is None:
        return ""
    normalized = key.strip().lower().replace(" ", "_").replace("-", "_")
    return KEY_ALIASES.get(normalized, normalized)


def _table_rows_from_text(text: str) -> list[dict[str, str | None]]:
    rows: list[dict[str, str | None]] = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        columns = _split_table_line(line)
        if not _looks_like_header(columns):
            continue
        headers = [_normalize_key(column) for column in columns]
        for candidate_line in lines[index + 1 :]:
            if _is_separator(candidate_line):
                continue
            values = _split_table_line(candidate_line)
            if len(values) < 2:
                continue
            if len(values) != len(headers):
                if rows:
                    break
                continue
            row = {header: value for header, value in zip(headers, values, strict=False)}
            if _looks_like_data_row(row):
                rows.append(row)
        if rows:
            break
    return rows


def _split_table_line(line: str) -> list[str]:
    if "|" in line:
        return [part.strip() for part in line.strip("|").split("|") if part.strip()]
    return [part.strip() for part in line.split() if part.strip()]


def _looks_like_header(columns: list[str]) -> bool:
    normalized = {_normalize_key(column) for column in columns}
    return bool(normalized & {"backend", "tokens/s", "ttft", "request_latency", "concurrency", "bs"})


def _looks_like_data_row(row: dict[str, str | None]) -> bool:
    if row.get("backend") or row.get("tokens/s") or row.get("ttft"):
        return True
    return any(row.get(key) for key in ("concurrency", "bs", "tp", "pp", "dp"))


def _is_separator(line: str) -> bool:
    stripped = line.replace("|", "").replace(":", "").strip()
    return bool(stripped) and set(stripped) <= {"-", "="}


def _text(values: dict[str, str | None], key: str) -> str | None:
    value = values.get(key)
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _int(values: dict[str, str | None], key: str) -> int | None:
    return _parse(values, key, lambda value: int(float(value)))


def _float(values: dict[str, str | None], key: str) -> float | None:
    return _parse(values, key, float)


def _parse(values: dict[str, str | None], key: str, parser: Callable[[str], int | float]) -> int | float | None:
    value = values.get(key)
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return parser(text)
    except ValueError:
        return None
