"""Staged benchmark matrix planning for journal level evaluation."""

from __future__ import annotations

import csv
import shlex
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import write_json
from .schemas import Goal

BENCHMARK_MATRIX_SCHEMA_VERSION = "benchmark-matrix-plan/v1"
DEFAULT_STAGES = ("stage1", "stage2", "stage3", "stage4")
MANAGED_BACKENDS = ("vllm", "sglang")
JOURNAL_GOALS = (Goal.BALANCED.value, Goal.PERFORMANCE.value, Goal.EFFICIENT.value)
STAGE1_WORKLOADS = ("short", "medium", "long-prefill")
STAGE2_WORKLOADS = ("short", "medium", "long", "mixed", "code-generation", "repeated-prefix")


@dataclass(frozen=True)
class MatrixModel:
    model: str
    model_class: str
    family: str
    access: str = "public"
    optional: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BenchmarkMatrixRequest:
    stages: list[str] = field(default_factory=lambda: list(DEFAULT_STAGES))
    output_root: str = "results/benchmark-matrix/runs"
    evidence_db: str | None = "results/benchmark-matrix/evidence.sqlite"
    repeats: int = 1
    limit: int = 5
    trials: int = 1
    telemetry: str = "auto"
    startup_timeout_s: float = 300.0
    cooldown_s: float = 5.0
    warmup_requests: int = 4
    steady_state_seconds: float | None = None
    idle_baseline_seconds: float = 15.0
    idle_power_watts: float | None = None
    soak_seconds: float | None = None
    include_optional_large: bool = False
    include_gated: bool = False
    real_chat_manifest: str | None = None
    attach_base_url: str | None = None
    small_model: str = "Qwen/Qwen3-0.6B"
    medium_model: str = "Qwen/Qwen2.5-7B-Instruct"
    notes: list[str] = field(default_factory=list)


def build_benchmark_matrix_plan(request: BenchmarkMatrixRequest) -> dict[str, Any]:
    _validate_request(request)
    stages = [_canonical_stage(stage) for stage in request.stages]
    cells: list[dict[str, Any]] = []
    for stage in stages:
        if stage == "stage1":
            cells.extend(_stage1_cells(request, start_index=len(cells) + 1))
        elif stage == "stage2":
            cells.extend(_stage2_cells(request, start_index=len(cells) + 1))
        elif stage == "stage3":
            cells.extend(_stage3_cells(request, start_index=len(cells) + 1))
        elif stage == "stage4":
            cells.extend(_stage4_cells(request, start_index=len(cells) + 1))
    stage_payloads = [_stage_payload(stage, cells) for stage in stages]
    runnable_cells = [cell for cell in cells if cell.get("runnable") is True]
    blocked_cells = [cell for cell in cells if cell.get("runnable") is not True]
    return {
        "schema_version": BENCHMARK_MATRIX_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "stage_count": len(stages),
            "cell_count": len(cells),
            "runnable_cell_count": len(runnable_cells),
            "blocked_or_manual_cell_count": len(blocked_cells),
            "managed_backend_count": len({cell.get("backend") for cell in runnable_cells if cell.get("backend")}),
            "model_count": len({cell.get("model") for cell in cells if cell.get("model")}),
            "workload_profile_count": len({cell.get("workload_profile") for cell in cells if cell.get("workload_profile")}),
        },
        "request": asdict(request),
        "stages": stage_payloads,
        "cells": cells,
        "notes": [
            "This plan does not launch servers or create measured evidence.",
            "Stage 1 should pass before later stages are executed.",
            "Cells with runnable=false require the listed prerequisite before execution.",
            "The local GPU can be left idle by generating this plan and running scripts only on a selected execution host.",
            *request.notes,
        ],
    }


def write_benchmark_matrix_artifacts(request: BenchmarkMatrixRequest, *, output_dir: Path) -> dict[str, Any]:
    payload = build_benchmark_matrix_plan(request)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "benchmark_matrix_plan.json"
    text_path = output_dir / "benchmark_matrix_plan.md"
    csv_path = output_dir / "benchmark_matrix.csv"
    dispatcher_path = output_dir / "benchmark_matrix_commands.sh"
    script_paths = _script_paths(output_dir, payload)
    payload["artifacts"] = {
        "benchmark_matrix_plan_json": str(json_path),
        "benchmark_matrix_plan_md": str(text_path),
        "benchmark_matrix_csv": str(csv_path),
        "benchmark_matrix_commands_sh": str(dispatcher_path),
        **{f"{key}_sh": str(path) for key, path in script_paths.items()},
    }
    write_json(json_path, payload)
    text_path.write_text(format_benchmark_matrix_markdown(payload), encoding="utf-8")
    _write_matrix_csv(csv_path, payload["cells"])
    dispatcher_path.write_text(format_benchmark_matrix_dispatcher(payload), encoding="utf-8")
    dispatcher_path.chmod(0o755)
    for key, path in script_paths.items():
        stage_id, backend = key.rsplit("_", 1)
        path.write_text(format_benchmark_matrix_runner(payload, stage_id=stage_id, backend=backend), encoding="utf-8")
        path.chmod(0o755)
    return payload


def format_benchmark_matrix_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    lines = [
        "# Benchmark Matrix Plan",
        "",
        f"planned cells: {summary.get('cell_count')}",
        f"runnable cells: {summary.get('runnable_cell_count')}",
        f"manual or blocked cells: {summary.get('blocked_or_manual_cell_count')}",
        "",
    ]
    for stage in payload.get("stages", []):
        if not isinstance(stage, dict):
            continue
        lines.extend(
            [
                f"## {stage.get('name')}",
                "",
                str(stage.get("purpose") or ""),
                "",
                f"cells: {stage.get('cell_count')}",
                f"runnable cells: {stage.get('runnable_cell_count')}",
                "",
                "Success criteria:",
            ]
        )
        for criterion in stage.get("success_criteria", []):
            lines.append(f"1. {criterion}")
        lines.append("")
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    if artifacts:
        lines.extend(["## Artifacts", ""])
        for key, value in artifacts.items():
            lines.append(f"1. `{key}`: `{value}`")
        lines.append("")
    return "\n".join(lines)


def format_benchmark_matrix_dispatcher(payload: dict[str, Any]) -> str:
    script_keys = sorted(_script_keys(payload))
    choices = "|".join(script_keys)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"',
        'target="${1:-}"',
        'if [[ -z "$target" ]]; then',
        f'  echo "usage: $0 {{{choices}}}" >&2',
        "  exit 2",
        "fi",
        'exec "$SCRIPT_DIR/benchmark_matrix_${target}.sh"',
    ]
    return "\n".join(lines) + "\n"


def format_benchmark_matrix_runner(payload: dict[str, Any], *, stage_id: str, backend: str) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -uo pipefail",
        "failures=0",
        "",
        f"# Benchmark matrix runner for {stage_id} on {backend}.",
    ]
    for cell in payload.get("cells", []):
        if not isinstance(cell, dict):
            continue
        if cell.get("stage_id") != stage_id or cell.get("backend") != backend or cell.get("runnable") is not True:
            continue
        lines.extend(
            [
                "",
                f"# {cell.get('cell_id')}: {cell.get('scenario')}",
                f"if ! {cell.get('shell_command')}; then",
                '  failures=$((failures + 1))',
                "fi",
            ]
        )
    lines.extend(
        [
            "",
            "if (( failures > 0 )); then",
            f'  echo "$failures {stage_id} {backend} benchmark cell(s) failed" >&2',
            "  exit 1",
            "fi",
        ]
    )
    return "\n".join(lines) + "\n"


def _stage1_cells(request: BenchmarkMatrixRequest, *, start_index: int) -> list[dict[str, Any]]:
    models = [
        MatrixModel(request.small_model, "small_open_under_1b", "Qwen"),
        MatrixModel(request.medium_model, "medium_open_7_to_8b", "Qwen"),
    ]
    return _managed_matrix_cells(
        request,
        stage_id="stage_1_sanity",
        start_index=start_index,
        models=models,
        backends=MANAGED_BACKENDS,
        goals=JOURNAL_GOALS,
        workloads=STAGE1_WORKLOADS,
        scenario="sanity_matrix",
    )


def _stage2_cells(request: BenchmarkMatrixRequest, *, start_index: int) -> list[dict[str, Any]]:
    models = [
        MatrixModel("Qwen/Qwen3-0.6B", "tiny_fast_validation", "Qwen"),
        MatrixModel("Qwen/Qwen2.5-1.5B-Instruct", "small_instruction_1_to_3b", "Qwen"),
        MatrixModel("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "small_instruction_1_to_3b", "DeepSeek"),
        MatrixModel("ibm-granite/granite-3.3-2b-instruct", "small_instruction_1_to_3b", "Granite"),
        MatrixModel("Qwen/Qwen2.5-7B-Instruct", "medium_instruction_7_to_8b", "Qwen"),
        MatrixModel("mistralai/Mistral-7B-Instruct-v0.3", "medium_instruction_7_to_8b", "Mistral"),
        MatrixModel("ibm-granite/granite-3.3-8b-instruct", "medium_instruction_7_to_8b", "Granite"),
        MatrixModel("Qwen/Qwen2.5-14B-Instruct", "larger_around_14b", "Qwen", optional=True),
        MatrixModel("Qwen/Qwen2.5-7B-Instruct", "long_context_model", "Qwen"),
        MatrixModel("meta-llama/Llama-3.1-8B-Instruct", "gated_if_access_approved", "Llama", access="gated", optional=True),
    ]
    selected_models = [
        model
        for model in models
        if (request.include_optional_large or model.model_class != "larger_around_14b")
        and (request.include_gated or model.access != "gated")
    ]
    cells = _managed_matrix_cells(
        request,
        stage_id="stage_2_broad_single_gpu",
        start_index=start_index,
        models=selected_models,
        backends=MANAGED_BACKENDS,
        goals=(Goal.BALANCED.value,),
        workloads=STAGE2_WORKLOADS,
        scenario="single_gpu_generality",
    )
    real_chat_index = start_index + len(cells)
    cells.extend(_real_chat_cells(request, start_index=real_chat_index, models=selected_models[:2]))
    return cells


def _stage3_cells(request: BenchmarkMatrixRequest, *, start_index: int) -> list[dict[str, Any]]:
    hardware_classes = [
        ("current_server", "Current server. If it is the only measured hardware, report device agnostic support as an artifact claim."),
        ("substantially_different_gpu", "Minimum additional hardware class required for measured device generality."),
        ("consumer_high_memory_gpu", "Ideal consumer high memory GPU class."),
        ("datacenter_a100_or_h100", "Ideal A100 or H100 family datacenter GPU class."),
        ("lower_power_l4_class", "Ideal lower power inference GPU class."),
        ("multi_gpu_tensor_parallel", "Required only if tensor parallel evaluation is claimed."),
    ]
    cells: list[dict[str, Any]] = []
    for offset, (hardware_class, note) in enumerate(hardware_classes):
        cells.append(
            _manual_cell(
                request,
                stage_id="stage_3_multi_hardware",
                index=start_index + offset,
                scenario="multi_hardware_replay",
                prerequisite=note,
                hardware_class=hardware_class,
                success_criteria=[
                    "At least current server plus one substantially different GPU class is measured for device generality claims.",
                    "If only one hardware class is available, paper wording frames device agnostic support as artifact coverage.",
                ],
            )
        )
    return cells


def _stage4_cells(request: BenchmarkMatrixRequest, *, start_index: int) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    base_model = MatrixModel(request.small_model, "small_open_under_1b", "Qwen")
    index = start_index
    for backend in MANAGED_BACKENDS:
        cells.append(
            _attach_cell(
                request,
                stage_id="stage_4_production_realism",
                index=index,
                model=base_model,
                backend=backend,
                workload_profile="mixed",
                scenario="attached_mode_observing_manual_service",
            )
        )
        index += 1
    managed_specs = [
        ("managed_mode_launching_services", "medium", Goal.BALANCED.value, {}),
        ("mixed_prompt_lengths", "mixed", Goal.BALANCED.value, {}),
        ("streaming_requests", "medium", Goal.BALANCED.value, {"stream": True}),
        ("high_concurrency_saturation", "medium", Goal.PERFORMANCE.value, {"limit": max(request.limit, 8)}),
        ("slo_constrained_serving", "medium", Goal.BALANCED.value, {"slo_p95_latency_ms": 2500.0, "slo_max_failed_request_rate": 0.01}),
        ("repeated_prefix_workload", "repeated-prefix", Goal.BALANCED.value, {}),
    ]
    for backend in MANAGED_BACKENDS:
        for scenario, workload, goal, overrides in managed_specs:
            cells.append(
                _managed_cell(
                    request,
                    stage_id="stage_4_production_realism",
                    index=index,
                    model=base_model,
                    backend=backend,
                    goal=goal,
                    workload_profile=workload,
                    scenario=scenario,
                    overrides=overrides,
                )
            )
            index += 1
    cells.append(
        _manual_cell(
            request,
            stage_id="stage_4_production_realism",
            index=index,
            scenario="model_nearly_fills_gpu_memory",
            prerequisite="Select the largest model that fits the execution GPU with a conservative memory margin.",
            model=request.medium_model,
            success_criteria=["Run completes or fails with precise out_of_memory or invalid_config reason."],
        )
    )
    index += 1
    cells.append(
        _manual_cell(
            request,
            stage_id="stage_4_production_realism",
            index=index,
            scenario="backend_crash_or_out_of_memory_recovery",
            prerequisite="Run an intentional failure injection cell only on an isolated execution host.",
            model=request.medium_model,
            success_criteria=["Failure taxonomy separates backend_failed_to_start, backend_crashed_during_load, and out_of_memory."],
        )
    )
    return cells


def _managed_matrix_cells(
    request: BenchmarkMatrixRequest,
    *,
    stage_id: str,
    start_index: int,
    models: list[MatrixModel],
    backends: tuple[str, ...],
    goals: tuple[str, ...],
    workloads: tuple[str, ...],
    scenario: str,
) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    index = start_index
    for model in models:
        for backend in backends:
            for goal in goals:
                for workload in workloads:
                    for repeat in range(1, request.repeats + 1):
                        cells.append(
                            _managed_cell(
                                request,
                                stage_id=stage_id,
                                index=index,
                                model=model,
                                backend=backend,
                                goal=goal,
                                workload_profile=workload,
                                scenario=scenario,
                                repeat=repeat,
                            )
                        )
                        index += 1
    return cells


def _managed_cell(
    request: BenchmarkMatrixRequest,
    *,
    stage_id: str,
    index: int,
    model: MatrixModel,
    backend: str,
    goal: str,
    workload_profile: str,
    scenario: str,
    repeat: int = 1,
    workload_manifest: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    overrides = dict(overrides or {})
    cell_id = _cell_id(index, stage_id, backend, goal, workload_profile, model.model)
    out_dir = str(Path(request.output_root) / stage_id / cell_id)
    command = _managed_command(
        request,
        model=model.model,
        backend=backend,
        goal=goal,
        workload_profile=workload_profile,
        out_dir=out_dir,
        workload_manifest=workload_manifest,
        overrides=overrides,
    )
    return {
        "index": index,
        "stage_id": stage_id,
        "cell_id": cell_id,
        "mode": "managed",
        "scenario": scenario,
        "model": model.model,
        "model_class": model.model_class,
        "model_family": model.family,
        "model_access": model.access,
        "backend": backend,
        "goal": goal,
        "objective_label": "throughput" if goal == Goal.PERFORMANCE.value else goal,
        "workload_profile": workload_profile,
        "repeat": repeat,
        "runnable": True,
        "prerequisite": None,
        "out_dir": out_dir,
        "command": command,
        "shell_command": " ".join(shlex.quote(part) for part in command),
        "success_criteria": _cell_success_criteria(stage_id),
    }


def _attach_cell(
    request: BenchmarkMatrixRequest,
    *,
    stage_id: str,
    index: int,
    model: MatrixModel,
    backend: str,
    workload_profile: str,
    scenario: str,
) -> dict[str, Any]:
    cell_id = _cell_id(index, stage_id, backend, "balanced", workload_profile, model.model)
    runnable = bool(request.attach_base_url)
    command = _attach_command(request, model=model.model, backend=backend, workload_profile=workload_profile) if runnable else []
    return {
        "index": index,
        "stage_id": stage_id,
        "cell_id": cell_id,
        "mode": "attach",
        "scenario": scenario,
        "model": model.model,
        "model_class": model.model_class,
        "model_family": model.family,
        "model_access": model.access,
        "backend": backend,
        "goal": Goal.BALANCED.value,
        "objective_label": Goal.BALANCED.value,
        "workload_profile": workload_profile,
        "repeat": 1,
        "runnable": runnable,
        "prerequisite": None if runnable else "Provide --attach-base-url for a manually launched OpenAI compatible service.",
        "out_dir": str(Path(request.output_root) / stage_id / cell_id),
        "command": command,
        "shell_command": " ".join(shlex.quote(part) for part in command),
        "success_criteria": _cell_success_criteria(stage_id),
    }


def _real_chat_cells(
    request: BenchmarkMatrixRequest,
    *,
    start_index: int,
    models: list[MatrixModel],
) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for offset, model in enumerate(models):
        if request.real_chat_manifest:
            cells.append(
                _managed_cell(
                    request,
                    stage_id="stage_2_broad_single_gpu",
                    index=start_index + offset,
                    model=model,
                    backend="vllm",
                    goal=Goal.BALANCED.value,
                    workload_profile="medium",
                    workload_manifest=request.real_chat_manifest,
                    scenario="real_chat_trace_permitted_dataset",
                )
            )
            continue
        cells.append(
            _manual_cell(
                request,
                stage_id="stage_2_broad_single_gpu",
                index=start_index + offset,
                scenario="real_chat_trace_permitted_dataset",
                prerequisite="Provide --real-chat-manifest for a permitted dataset before running this cell.",
                model=model.model,
                backend="vllm",
                workload_profile="medium",
                success_criteria=["Dataset source and license status are recorded in workload metadata."],
            )
        )
    return cells


def _manual_cell(
    request: BenchmarkMatrixRequest,
    *,
    stage_id: str,
    index: int,
    scenario: str,
    prerequisite: str,
    model: str | None = None,
    backend: str | None = None,
    workload_profile: str | None = None,
    hardware_class: str | None = None,
    success_criteria: list[str] | None = None,
) -> dict[str, Any]:
    cell_id = _cell_id(index, stage_id, backend or "manual", "manual", workload_profile or scenario, model or hardware_class or "manual")
    return {
        "index": index,
        "stage_id": stage_id,
        "cell_id": cell_id,
        "mode": "manual",
        "scenario": scenario,
        "model": model,
        "model_class": None,
        "model_family": None,
        "model_access": None,
        "backend": backend,
        "goal": None,
        "objective_label": None,
        "workload_profile": workload_profile,
        "hardware_class": hardware_class,
        "repeat": 1,
        "runnable": False,
        "prerequisite": prerequisite,
        "out_dir": str(Path(request.output_root) / stage_id / cell_id),
        "command": [],
        "shell_command": "",
        "success_criteria": success_criteria or _cell_success_criteria(stage_id),
    }


def _managed_command(
    request: BenchmarkMatrixRequest,
    *,
    model: str,
    backend: str,
    goal: str,
    workload_profile: str,
    out_dir: str,
    workload_manifest: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> list[str]:
    overrides = dict(overrides or {})
    command = [
        "serve-optimize",
        "managed-evaluate",
        "--backend",
        backend,
        "--model",
        model,
        "--goal",
        goal,
        "--limit",
        str(int(overrides.get("limit", request.limit))),
        "--trials",
        str(request.trials),
        "--telemetry",
        request.telemetry,
        "--workload-profile",
        workload_profile,
        "--startup-timeout",
        _number(request.startup_timeout_s),
        "--cooldown-seconds",
        _number(request.cooldown_s),
        "--out",
        out_dir,
    ]
    if request.evidence_db:
        command.extend(["--evidence-db", request.evidence_db])
    if workload_manifest:
        command.extend(["--workload-manifest", workload_manifest])
    if request.warmup_requests:
        command.extend(["--warmup-requests", str(request.warmup_requests)])
    if request.steady_state_seconds is not None:
        command.extend(["--steady-state-seconds", _number(request.steady_state_seconds)])
    if request.idle_baseline_seconds:
        command.extend(["--idle-baseline-seconds", _number(request.idle_baseline_seconds)])
    if request.idle_power_watts is not None:
        command.extend(["--idle-power-watts", _number(request.idle_power_watts)])
    if request.soak_seconds is not None:
        command.extend(["--soak-seconds", _number(request.soak_seconds)])
    if overrides.get("stream") is True:
        command.append("--stream")
    if overrides.get("slo_p95_latency_ms") is not None:
        command.extend(["--slo-p95-latency-ms", _number(float(overrides["slo_p95_latency_ms"]))])
    if overrides.get("slo_max_failed_request_rate") is not None:
        command.extend(["--slo-max-failed-request-rate", _number(float(overrides["slo_max_failed_request_rate"]))])
    return command


def _attach_command(
    request: BenchmarkMatrixRequest,
    *,
    model: str,
    backend: str,
    workload_profile: str,
) -> list[str]:
    return [
        "serve-optimize",
        "recommend",
        model,
        "--base-url",
        request.attach_base_url or "",
        "--backend",
        backend,
        "--goal",
        Goal.BALANCED.value,
        "--workload-profile",
        workload_profile,
        "--telemetry",
        request.telemetry,
        "--out",
        str(Path(request.output_root) / "stage_4_production_realism" / "attach"),
    ]


def _stage_payload(stage: str, cells: list[dict[str, Any]]) -> dict[str, Any]:
    stage_id = _stage_id(stage)
    stage_cells = [cell for cell in cells if cell.get("stage_id") == stage_id]
    return {
        "stage": stage,
        "stage_id": stage_id,
        "name": _stage_name(stage),
        "purpose": _stage_purpose(stage),
        "cell_count": len(stage_cells),
        "runnable_cell_count": sum(1 for cell in stage_cells if cell.get("runnable") is True),
        "success_criteria": _stage_success_criteria(stage),
    }


def _cell_success_criteria(stage_id: str) -> list[str]:
    if stage_id == "stage_1_sanity":
        return _stage_success_criteria("stage1")
    if stage_id == "stage_2_broad_single_gpu":
        return _stage_success_criteria("stage2")
    if stage_id == "stage_3_multi_hardware":
        return _stage_success_criteria("stage3")
    return _stage_success_criteria("stage4")


def _stage_success_criteria(stage: str) -> list[str]:
    stage = _canonical_stage(stage)
    if stage == "stage1":
        return [
            "All summary fields are populated.",
            "Both backends either complete or fail with a precise reason.",
            "At least one configuration changes throughput or latency by more than noise.",
            "Energy values are nonzero and plausible when telemetry is available.",
        ]
    if stage == "stage2":
        return [
            "One machine covers tiny, small, medium, long context, optional larger, and approved gated model classes as available.",
            "Suggested Qwen, Mistral, Llama, DeepSeek distilled, and Granite families are represented when access allows.",
            "Synthetic, code generation, repeated prefix, and permitted real chat workloads are represented.",
        ]
    if stage == "stage3":
        return [
            "Current server plus one substantially different GPU class is measured for device generality claims.",
            "If only one hardware class is available, device agnostic behavior is framed as artifact support rather than a measured claim.",
        ]
    return [
        "Attach Mode and Managed Mode are both represented.",
        "Mixed lengths, streaming, high concurrency, SLO constraints, prefix reuse, near memory capacity, and failure recovery are represented.",
        "Backend crash and out of memory recovery cells preserve precise failure reasons.",
    ]


def _stage_purpose(stage: str) -> str:
    stage = _canonical_stage(stage)
    return {
        "stage1": "Prove the harness measures real differences before longer campaigns.",
        "stage2": "Prove model and workload generality on one machine.",
        "stage3": "Prove or honestly bound device agnostic behavior.",
        "stage4": "Exercise production realism beyond basic synthetic managed runs.",
    }[stage]


def _stage_name(stage: str) -> str:
    stage = _canonical_stage(stage)
    return {
        "stage1": "Stage 1 sanity matrix",
        "stage2": "Stage 2 broad single GPU matrix",
        "stage3": "Stage 3 multi hardware matrix",
        "stage4": "Stage 4 production realism matrix",
    }[stage]


def _stage_id(stage: str) -> str:
    stage = _canonical_stage(stage)
    return {
        "stage1": "stage_1_sanity",
        "stage2": "stage_2_broad_single_gpu",
        "stage3": "stage_3_multi_hardware",
        "stage4": "stage_4_production_realism",
    }[stage]


def _canonical_stage(stage: str) -> str:
    normalized = str(stage).strip().lower().replace("_", "").replace("-", "")
    aliases = {
        "1": "stage1",
        "stage1": "stage1",
        "sanity": "stage1",
        "2": "stage2",
        "stage2": "stage2",
        "broad": "stage2",
        "singlegpu": "stage2",
        "3": "stage3",
        "stage3": "stage3",
        "multihardware": "stage3",
        "4": "stage4",
        "stage4": "stage4",
        "production": "stage4",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported benchmark matrix stage: {stage}") from exc


def _script_paths(output_dir: Path, payload: dict[str, Any]) -> dict[str, Path]:
    return {
        key: output_dir / f"benchmark_matrix_{key}.sh"
        for key in _script_keys(payload)
    }


def _script_keys(payload: dict[str, Any]) -> set[str]:
    keys = set()
    for cell in payload.get("cells", []):
        if not isinstance(cell, dict) or cell.get("runnable") is not True:
            continue
        backend = cell.get("backend")
        stage_id = cell.get("stage_id")
        if backend and stage_id:
            keys.add(f"{stage_id}_{backend}")
    return keys


def _write_matrix_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "index",
        "stage_id",
        "cell_id",
        "mode",
        "scenario",
        "model",
        "model_class",
        "model_family",
        "model_access",
        "backend",
        "goal",
        "objective_label",
        "workload_profile",
        "hardware_class",
        "repeat",
        "runnable",
        "prerequisite",
        "out_dir",
        "shell_command",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def _cell_id(index: int, stage_id: str, backend: str, goal: str, workload: str, model: str) -> str:
    return f"{index:04d}-{_slug(stage_id)}-{_slug(backend)}-{_slug(goal)}-{_slug(workload)}-{_slug(model)}"


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in str(value).lower()).strip("_") or "item"


def _number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _validate_request(request: BenchmarkMatrixRequest) -> None:
    if not request.stages:
        raise ValueError("at least one stage is required.")
    for stage in request.stages:
        _canonical_stage(stage)
    if request.repeats < 1:
        raise ValueError("repeats must be at least 1.")
    if request.limit < 1:
        raise ValueError("limit must be at least 1.")
    if request.trials < 1:
        raise ValueError("trials must be at least 1.")
    if request.idle_baseline_seconds < 0:
        raise ValueError("idle baseline seconds must be nonnegative.")
    if request.soak_seconds is not None and request.soak_seconds <= 0:
        raise ValueError("soak seconds must be greater than 0 when provided.")
