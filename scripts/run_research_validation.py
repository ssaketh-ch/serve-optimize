"""Run bounded, fresh validation experiments for the paper evidence package."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import replace
from pathlib import Path

from serve_optimize.hardware import detect_hardware
from serve_optimize.managed import (
    _adapter_for_backend,
    _backend_argument_capabilities,
    _backend_metadata,
    _backend_sglang_argument_capabilities,
    _generate_managed_candidate_generation,
    run_managed_evaluation,
)
from serve_optimize.modeling import infer_model_capability_metadata
from serve_optimize.research_analysis import SearchCandidate, run_search
from serve_optimize.schemas import Goal, ServingConfig, WorkloadProfile, to_dict
from serve_optimize.workloads import load_workload_profile

MODEL_RECORDS = {
    "qwen3-0.6b": ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"),
    "mistral-7b": ("mistralai/Mistral-7B-Instruct-v0.3", "c170c708c41dac9275d15a8fff4eca08d52bab71"),
    "qwen38-27b": ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"),
}

WORKLOADS = ("short", "long-prefill", "real-chat", "real-chat-holdout")
SEARCH_METHODS = ("backend-default", "serve-optimize", "random", "grid", "bayesian")
RANDOM_SEEDS = (0, 1, 2)
ALL_METHOD_SEEDS = {
    "backend-default": (None,),
    "serve-optimize": (None,),
    "random": RANDOM_SEEDS,
    "grid": (None,),
    "bayesian": RANDOM_SEEDS,
}


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.]+", "_", value).strip("_") or "value"


def _model_path(hf_home: Path, model: str, revision: str) -> Path:
    path = hf_home / "hub" / f"models--{model.replace('/', '--')}" / "snapshots" / revision
    if not (path / "config.json").is_file():
        raise SystemExit(f"Pinned model snapshot is missing: {path}")
    return path


def _profile(root: Path, name: str) -> WorkloadProfile:
    if name in {"short", "long-prefill"}:
        return load_workload_profile(profile_name=name)
    manifest = root / "configs" / "workloads" / "oasst1-root-en-64.json"
    profile = load_workload_profile(manifest_path=manifest)
    if name == "real-chat":
        prompts = profile.prompts[:48]
        return replace(
            profile,
            profile_name="real-chat-oasst1-root-en-train-v1",
            prompts=prompts,
            num_requests=max(96, len(prompts)),
            notes=[*profile.notes, "Deterministic training split: prompt indices 0 through 47."],
        )
    if name == "real-chat-holdout":
        prompts = profile.prompts[48:]
        if not prompts:
            raise SystemExit("The real chat manifest does not contain a holdout split.")
        return replace(
            profile,
            profile_name="real-chat-oasst1-root-en-holdout-v1",
            prompts=prompts,
            num_requests=max(64, len(prompts)),
            notes=[*profile.notes, "Deterministic holdout: prompt indices 48 through 63."],
        )
    raise SystemExit(f"Unsupported validation workload: {name}")


def _candidate_features(config: ServingConfig) -> dict[str, object]:
    extra = config.extra or {}
    return {
        "max_batch_size": config.max_batch_size,
        "max_context_tokens": config.max_context_tokens,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "workload_concurrency": extra.get("workload_concurrency"),
        "dtype": config.dtype,
        "quantization": config.quantization,
    }


def _search_pool(candidates: list[ServingConfig]) -> tuple[SearchCandidate, ...]:
    return tuple(
        SearchCandidate(
            candidate_id=config.id,
            features=_candidate_features(config),
            baseline=bool((config.extra or {}).get("baseline"))
            or (config.extra or {}).get("candidate_source") == "safe_baseline",
        )
        for config in candidates
    )


def _baseline_first(candidates: list[ServingConfig]) -> list[ServingConfig]:
    baseline = next((config for config in candidates if _search_pool([config])[0].baseline), None)
    rest = [config for config in candidates if config is not baseline]
    return ([baseline] if baseline is not None else []) + rest


def _metric_from_run(summary: object) -> tuple[float, dict[str, object]]:
    candidates = getattr(summary, "candidates", [])
    if not candidates:
        return float("-inf"), {"status": "missing_candidate_result"}
    paths = getattr(candidates[0], "summary_paths", [])
    if not paths:
        return float("-inf"), {"status": "missing_summary_path"}
    payload = json.loads(Path(paths[-1]).read_text(encoding="utf-8"))
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else payload
    throughput = metrics.get("output_tokens_s", metrics.get("output_tokens_per_sec"))
    failed = metrics.get("failed_requests", 0)
    try:
        score = float(throughput)
    except (TypeError, ValueError):
        score = float("-inf")
    if failed not in (None, 0, 0.0):
        score = float("-inf")
    return score, {
        "output_tokens_per_sec": throughput,
        "total_tokens_per_sec": metrics.get("total_tokens_s", metrics.get("throughput_tokens_per_sec")),
        "p95_latency_ms": metrics.get("p95_latency_s"),
        "failed_requests": failed,
        "summary_path": str(paths[-1]),
        "run_status": getattr(summary, "status", None),
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


class CellRunner:
    def __init__(
        self,
        *,
        cell_root: Path,
        backend: str,
        model_path: Path,
        model_id: str,
        profile: WorkloadProfile,
        trials: int,
    ) -> None:
        self.cell_root = cell_root
        self.backend = backend
        self.model_path = model_path
        self.model_id = model_id
        self.profile = profile
        self.trials = trials
        self.rows: list[dict[str, object]] = []

    def evaluate(self, config: ServingConfig, *, method: str, seed: int | None, ordinal: int) -> float:
        seed_label = "none" if seed is None else str(seed)
        run_root = self.cell_root / "measurements" / _safe_name(method) / f"seed-{seed_label}" / f"{ordinal:03d}-{_safe_name(config.id)}"
        summary = run_managed_evaluation(
            backend=self.backend,
            model=str(self.model_path),
            goal=Goal.PERFORMANCE,
            limit=1,
            trials=self.trials,
            startup_timeout_s=600.0,
            cooldown_s=10.0,
            host="127.0.0.1",
            port=None,
            out_dir=run_root,
            telemetry="auto",
            evidence_db_path=None,
            evidence_write=False,
            workload_profile=self.profile,
            warmup_requests=16,
            steady_state_duration_s=60.0,
            idle_baseline_duration_s=30.0,
            stream=True,
            allow_remote_model_config_download=False,
        )
        score, metrics = _metric_from_run(summary)
        row = {
            "method": method,
            "seed": seed,
            "ordinal": ordinal,
            "candidate_id": config.id,
            "score_output_tokens_per_sec": None if score == float("-inf") else score,
            "config": _candidate_features(config),
            "artifacts": metrics,
        }
        self.rows.append(row)
        _write_json(run_root / "validation_measurement.json", row)
        return score


def _run_search_method(
    runner: CellRunner,
    candidates: list[ServingConfig],
    *,
    method: str,
    seed: int | None,
    budget: int,
) -> dict[str, object]:
    ordered = _baseline_first(candidates)
    by_id = {config.id: config for config in ordered}
    search_pool = _search_pool(ordered)
    ordinal = 0

    def evaluate(candidate_id: str) -> float:
        nonlocal ordinal
        config = by_id[candidate_id]
        score = runner.evaluate(config, method=method, seed=seed, ordinal=ordinal)
        ordinal += 1
        return score

    if method == "backend-default":
        baseline = next(candidate for candidate in search_pool if candidate.baseline)
        order = [baseline.candidate_id]
        scores = {baseline.candidate_id: evaluate(baseline.candidate_id)}
        selected_id = baseline.candidate_id
        selected_score = scores[selected_id]
    elif method == "serve-optimize":
        order = [candidate.candidate_id for candidate in search_pool[:budget]]
        scores = {candidate_id: evaluate(candidate_id) for candidate_id in order}
        selected_id = max(order, key=lambda item: (scores[item], item))
        selected_score = scores[selected_id]
    else:
        result = run_search(search_pool, evaluate, method=method, budget=budget, seed=seed or 0)
        order = list(result.evaluation_order)
        selected_id = result.selected_candidate_id
        selected_score = result.selected_score
    return {
        "method": method,
        "seed": seed,
        "budget": budget,
        "evaluation_order": order,
        "selected_candidate_id": selected_id,
        "selected_score_output_tokens_per_sec": None if selected_score == float("-inf") else selected_score,
        "measurement_mode": "fresh_independent_candidate_runs",
    }


def run_cell(
    *,
    out: Path,
    root: Path,
    label: str,
    model: str,
    revision: str,
    backend: str,
    workload: str,
    hf_home: Path,
    pool_limit: int,
    budget: int,
    trials: int,
) -> None:
    profile = _profile(root, workload)
    model_path = _model_path(hf_home, model, revision)
    hardware = detect_hardware()
    adapter = _adapter_for_backend(backend)
    metadata = infer_model_capability_metadata(str(model_path), allow_remote_download=False)
    generation = _generate_managed_candidate_generation(
        backend=backend,
        model=str(model_path),
        goal=Goal.PERFORMANCE,
        limit=pool_limit,
        hardware=hardware,
        model_metadata=metadata,
        backend_metadata=_backend_metadata(adapter, backend),
        vllm_argument_capabilities=_backend_argument_capabilities(adapter, backend),
        sglang_argument_capabilities=_backend_sglang_argument_capabilities(adapter, backend),
        workload_profile=profile,
    )
    candidates = _baseline_first(generation.candidates)
    if len(candidates) < 2:
        raise SystemExit(f"Candidate pool for {label} {backend} {workload} has fewer than two candidates.")
    actual_budget = min(budget, len(candidates))
    cell_root = out / f"{_safe_name(label)}-{_safe_name(backend)}-{_safe_name(workload)}"
    cell_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        cell_root / "cell_manifest.json",
        {
            "schema_version": "research-validation-cell/v1",
            "model_label": label,
            "model": model,
            "model_revision": revision,
            "model_snapshot": str(model_path),
            "backend": backend,
            "workload": workload,
            "workload_profile": profile,
            "goal": Goal.PERFORMANCE.value,
            "candidate_pool_limit": pool_limit,
            "candidate_pool_count": len(candidates),
            "budget": actual_budget,
            "trials": trials,
            "fresh_measurements": True,
            "evidence_reuse": False,
            "candidate_ids": [config.id for config in candidates],
            "candidate_source_counts": generation.candidate_source_counts,
            "hardware": hardware,
            "backend_metadata": _backend_metadata(adapter, backend),
        },
    )
    runner = CellRunner(
        cell_root=cell_root,
        backend=backend,
        model_path=model_path,
        model_id=model,
        profile=profile,
        trials=trials,
    )
    results: list[dict[str, object]] = []
    for method in SEARCH_METHODS:
        for seed in ALL_METHOD_SEEDS[method]:
            results.append(_run_search_method(runner, candidates, method=method, seed=seed, budget=actual_budget))
    oracle_scores: dict[str, float] = {}
    for ordinal, config in enumerate(candidates):
        oracle_scores[config.id] = runner.evaluate(config, method="measured-oracle", seed=0, ordinal=ordinal)
    oracle_id = max(oracle_scores, key=lambda item: (oracle_scores[item], item))
    oracle_score = oracle_scores[oracle_id]
    for row in results:
        selected = row["selected_candidate_id"]
        selected_score = row.get("selected_score_output_tokens_per_sec")
        row["oracle_candidate_id"] = oracle_id
        row["oracle_score_output_tokens_per_sec"] = None if oracle_score == float("-inf") else oracle_score
        row["regret_output_tokens_per_sec"] = (
            None
            if selected_score is None or oracle_score == float("-inf")
            else max(0.0, oracle_score - float(selected_score))
        )
        row["selected_is_oracle"] = selected == oracle_id
    _write_json(cell_root / "search_results.json", results)
    _write_json(cell_root / "measured_oracle.json", {"candidate_id": oracle_id, "score_output_tokens_per_sec": None if oracle_score == float("-inf") else oracle_score, "scores": oracle_scores})
    _write_jsonl(cell_root / "measurement_index.jsonl", runner.rows)


def run_holdout_cell(
    *,
    out: Path,
    root: Path,
    label: str,
    model: str,
    revision: str,
    backend: str,
    hf_home: Path,
    pool_limit: int,
    trials: int,
) -> None:
    train_profile = _profile(root, "real-chat")
    holdout_profile = _profile(root, "real-chat-holdout")
    train_root = out / f"{_safe_name(label)}-{_safe_name(backend)}-real-chat"
    search_results_path = train_root / "search_results.json"
    oracle_path = train_root / "measured_oracle.json"
    if not search_results_path.is_file() or not oracle_path.is_file():
        raise SystemExit(f"Training search artifacts are missing for holdout cell: {train_root}")
    search_results = json.loads(search_results_path.read_text(encoding="utf-8"))
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    selected_by_method: dict[str, list[object]] = {}
    selected_ids: list[str] = []
    for row in search_results:
        method = str(row.get("method"))
        candidate_id = str(row.get("selected_candidate_id"))
        selected_by_method.setdefault(method, []).append(candidate_id)
        if candidate_id not in selected_ids:
            selected_ids.append(candidate_id)
    oracle_id = str(oracle["candidate_id"])
    if oracle_id not in selected_ids:
        selected_ids.append(oracle_id)
    model_path = _model_path(hf_home, model, revision)
    hardware = detect_hardware()
    adapter = _adapter_for_backend(backend)
    metadata = infer_model_capability_metadata(str(model_path), allow_remote_download=False)
    generation = _generate_managed_candidate_generation(
        backend=backend,
        model=str(model_path),
        goal=Goal.PERFORMANCE,
        limit=pool_limit,
        hardware=hardware,
        model_metadata=metadata,
        backend_metadata=_backend_metadata(adapter, backend),
        vllm_argument_capabilities=_backend_argument_capabilities(adapter, backend),
        sglang_argument_capabilities=_backend_sglang_argument_capabilities(adapter, backend),
        workload_profile=train_profile,
    )
    candidates = {config.id: config for config in _baseline_first(generation.candidates)}
    missing = [candidate_id for candidate_id in selected_ids if candidate_id not in candidates]
    if missing:
        raise SystemExit(f"Training selected candidates are absent from holdout pool: {missing}")
    cell_root = out / f"{_safe_name(label)}-{_safe_name(backend)}-real-chat-holdout"
    cell_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        cell_root / "cell_manifest.json",
        {
            "schema_version": "research-validation-holdout-cell/v1",
            "model_label": label,
            "model": model,
            "model_revision": revision,
            "model_snapshot": str(model_path),
            "backend": backend,
            "workload": "real-chat-holdout",
            "selection_source": str(train_root),
            "selection_is_holdout_free": True,
            "holdout_prompt_indices": "48 through 63",
            "workload_profile": holdout_profile,
            "candidate_pool_limit": pool_limit,
            "candidate_pool_count": len(candidates),
            "selected_candidate_ids": selected_ids,
            "training_oracle_candidate_id": oracle_id,
            "trials": trials,
            "fresh_measurements": True,
            "evidence_reuse": False,
            "hardware": hardware,
            "backend_metadata": _backend_metadata(adapter, backend),
        },
    )
    runner = CellRunner(
        cell_root=cell_root,
        backend=backend,
        model_path=model_path,
        model_id=model,
        profile=holdout_profile,
        trials=trials,
    )
    holdout_scores: dict[str, float] = {}
    for ordinal, candidate_id in enumerate(selected_ids):
        holdout_scores[candidate_id] = runner.evaluate(
            candidates[candidate_id],
            method="holdout-evaluation",
            seed=0,
            ordinal=ordinal,
        )
    _write_json(
        cell_root / "holdout_evaluations.json",
        {
            "selection_source": str(train_root),
            "selection_is_holdout_free": True,
            "selected_by_method": selected_by_method,
            "training_oracle_candidate_id": oracle_id,
            "scores_output_tokens_per_sec": {
                candidate_id: None if score == float("-inf") else score
                for candidate_id, score in holdout_scores.items()
            },
        },
    )
    _write_jsonl(cell_root / "measurement_index.jsonl", runner.rows)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Checked out Serve Optimize source root.")
    parser.add_argument("--out", type=Path, required=True, help="Validation artifact root.")
    parser.add_argument("--hf-home", type=Path, required=True)
    parser.add_argument("--model-label", action="append", required=True, choices=sorted(MODEL_RECORDS))
    parser.add_argument("--backend", action="append", choices=("vllm", "sglang"), required=True)
    parser.add_argument("--workload", action="append", choices=WORKLOADS, required=True)
    parser.add_argument("--pool-limit", type=int, default=8)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    if args.pool_limit < 2 or args.budget < 2 or args.trials < 1:
        raise SystemExit("pool limit and budget must be at least two, and trials must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "research-validation/v1",
        "search_methods": SEARCH_METHODS,
        "random_and_bayesian_seeds": RANDOM_SEEDS,
        "models": args.model_label,
        "backends": args.backend,
        "workloads": args.workload,
        "pool_limit": args.pool_limit,
        "budget": args.budget,
        "trials": args.trials,
        "unsupported_ablation_status": {
            "no_hardware_awareness": "not_executed_requires_counterfactual_candidate_generation",
            "no_backend_capability_registry": "not_executed_requires_unsafe_launch_candidates",
            "no_failure_memory": "covered_by_fresh_runs_without_resume",
            "energy_term_removed": "not_executed_in_performance_only_search",
            "latency_guardrail_removed": "not_executed_without_an_explicit_slo",
        },
        "executed_controls": {
            "backend_default": "fresh_three_trial_measurement",
            "no_evidence_reuse": "all_validation_measurements_disable_evidence_reuse",
            "no_failure_memory": "all_validation_measurements_run_without_resume",
        },
    }
    _write_json(args.out / "validation_manifest.json", manifest)
    for label in args.model_label:
        model, revision = MODEL_RECORDS[label]
        for backend in args.backend:
            for workload in args.workload:
                if workload == "real-chat-holdout":
                    run_holdout_cell(
                        out=args.out,
                        root=args.root,
                        label=label,
                        model=model,
                        revision=revision,
                        backend=backend,
                        hf_home=args.hf_home,
                        pool_limit=args.pool_limit,
                        trials=args.trials,
                    )
                else:
                    run_cell(
                        out=args.out,
                        root=args.root,
                        label=label,
                        model=model,
                        revision=revision,
                        backend=backend,
                        workload=workload,
                        hf_home=args.hf_home,
                        pool_limit=args.pool_limit,
                        budget=args.budget,
                        trials=args.trials,
                    )


if __name__ == "__main__":
    main()
