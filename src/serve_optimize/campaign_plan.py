"""Managed validation campaign planning artifacts."""

from __future__ import annotations

import csv
import shlex
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import write_json

CAMPAIGN_PLAN_SCHEMA_VERSION = "campaign-plan/v1"


@dataclass(frozen=True)
class CampaignPlanRequest:
    models: list[str]
    backends: list[str]
    goals: list[str]
    workload_profiles: list[str]
    repeats: int = 1
    limit: int = 4
    trials: int = 1
    telemetry: str = "auto"
    evidence_db: str | None = None
    output_root: str = "results/managed-campaign"
    startup_timeout_s: float = 300.0
    cooldown_s: float = 5.0
    warmup_requests: int = 0
    steady_state_seconds: float | None = None
    idle_baseline_seconds: float = 0.0
    idle_power_watts: float | None = None
    soak_seconds: float | None = None
    stream: bool = False
    notes: list[str] = field(default_factory=list)


def build_campaign_plan(request: CampaignPlanRequest) -> dict[str, Any]:
    _validate_request(request)
    runs = []
    index = 0
    for model in request.models:
        for backend in request.backends:
            for goal in request.goals:
                for workload_profile in request.workload_profiles:
                    for repeat in range(1, request.repeats + 1):
                        index += 1
                        run_id = _planned_run_id(index, backend, goal, workload_profile, repeat)
                        out_dir = str(Path(request.output_root) / run_id)
                        command = _managed_command(
                            request,
                            model=model,
                            backend=backend,
                            goal=goal,
                            workload_profile=workload_profile,
                            out_dir=out_dir,
                        )
                        runs.append(
                            {
                                "index": index,
                                "run_id": run_id,
                                "model": model,
                                "backend": backend,
                                "goal": goal,
                                "workload_profile": workload_profile,
                                "repeat": repeat,
                                "out_dir": out_dir,
                                "command": command,
                                "shell_command": " ".join(shlex.quote(part) for part in command),
                            }
                        )
    return {
        "schema_version": CAMPAIGN_PLAN_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "planned_run_count": len(runs),
            "model_count": len(request.models),
            "backend_count": len(request.backends),
            "goal_count": len(request.goals),
            "workload_profile_count": len(request.workload_profiles),
            "repeats": request.repeats,
        },
        "request": asdict(request),
        "runs": runs,
        "post_commands": _post_commands(request),
        "notes": [
            "Campaign plans do not launch servers or create measured evidence.",
            "Each listed command must be run in the correct backend environment.",
            "Research and validation claims remain scoped to completed managed run artifacts.",
            *request.notes,
        ],
    }


def write_campaign_plan_artifacts(request: CampaignPlanRequest, *, output_dir: Path) -> dict[str, Any]:
    payload = build_campaign_plan(request)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "campaign_plan.json"
    text_path = output_dir / "campaign_plan.txt"
    csv_path = output_dir / "campaign_matrix.csv"
    commands_path = output_dir / "campaign_commands.sh"
    postprocess_path = output_dir / "campaign_postprocess.sh"
    backend_command_paths = {
        backend: output_dir / f"campaign_commands_{_slug(backend)}.sh"
        for backend in request.backends
    }
    payload["artifacts"] = {
        "campaign_plan_json": str(json_path),
        "campaign_plan_txt": str(text_path),
        "campaign_matrix_csv": str(csv_path),
        "campaign_commands_sh": str(commands_path),
        "campaign_postprocess_sh": str(postprocess_path),
        **{
            f"campaign_commands_{_slug(backend)}_sh": str(path)
            for backend, path in backend_command_paths.items()
        },
    }
    write_json(json_path, payload)
    text_path.write_text(format_campaign_plan_text(payload), encoding="utf-8")
    _write_campaign_csv(csv_path, payload["runs"])
    commands_path.write_text(format_campaign_dispatcher(payload), encoding="utf-8")
    commands_path.chmod(0o755)
    postprocess_path.write_text(format_campaign_postprocess(payload), encoding="utf-8")
    postprocess_path.chmod(0o755)
    for backend, path in backend_command_paths.items():
        path.write_text(format_campaign_commands(payload, backend=backend), encoding="utf-8")
        path.chmod(0o755)
    return payload


def format_campaign_plan_text(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    request = payload.get("request") if isinstance(payload.get("request"), dict) else {}
    lines = [
        "Serve Optimize Campaign Plan",
        "",
        f"planned runs: {summary.get('planned_run_count')}",
        f"models: {', '.join(request.get('models', []))}",
        f"backends: {', '.join(request.get('backends', []))}",
        f"goals: {', '.join(request.get('goals', []))}",
        f"workload profiles: {', '.join(request.get('workload_profiles', []))}",
        f"repeats: {summary.get('repeats')}",
        "",
        "Next steps:",
        "  Run campaign_commands.sh BACKEND in the matching backend environment.",
        "  Run campaign_postprocess.sh after all backend scripts finish.",
    ]
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    if artifacts:
        lines.extend(["", "Artifacts:"])
        for key, value in artifacts.items():
            lines.append(f"  {key}: {value}")
    return "\n".join(lines) + "\n"


def format_campaign_dispatcher(payload: dict[str, Any]) -> str:
    request = payload.get("request") if isinstance(payload.get("request"), dict) else {}
    backends = [str(backend) for backend in request.get("backends", [])]
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"',
        'backend="${1:-}"',
        "case \"$backend\" in",
    ]
    for backend in backends:
        lines.append(f'  {_slug(backend)}) exec "$SCRIPT_DIR/campaign_commands_{_slug(backend)}.sh" ;;')
    choices = "|".join(_slug(backend) for backend in backends)
    lines.extend(
        [
            "  *)",
            f'    echo "usage: $0 {{{choices}}}" >&2',
            "    exit 2",
            "    ;;",
            "esac",
        ]
    )
    return "\n".join(lines) + "\n"


def format_campaign_commands(payload: dict[str, Any], *, backend: str) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -uo pipefail",
        "failures=0",
        "",
        f"# Run in the {backend} backend environment.",
    ]
    for row in payload.get("runs", []):
        if not isinstance(row, dict) or row.get("backend") != backend:
            continue
        lines.extend(
            [
                "",
                f"# {row.get('run_id')}",
                f"if ! {row.get('shell_command')}; then",
                '  failures=$((failures + 1))',
                "fi",
            ]
        )
    lines.extend(
        [
            "",
            "if (( failures > 0 )); then",
            f'  echo "$failures {backend} campaign run(s) failed" >&2',
            "  exit 1",
            "fi",
        ]
    )
    return "\n".join(lines) + "\n"


def format_campaign_postprocess(payload: dict[str, Any]) -> str:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
    post_commands = payload.get("post_commands") if isinstance(payload.get("post_commands"), dict) else {}
    if post_commands:
        lines.extend(["", "# Run after all backend campaign scripts complete."])
        for command in post_commands.values():
            if isinstance(command, str):
                lines.append(command)
    return "\n".join(lines) + "\n"


def _write_campaign_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = ["index", "run_id", "model", "backend", "goal", "workload_profile", "repeat", "out_dir", "shell_command"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def _managed_command(
    request: CampaignPlanRequest,
    *,
    model: str,
    backend: str,
    goal: str,
    workload_profile: str,
    out_dir: str,
) -> list[str]:
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
        str(request.limit),
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
    if request.stream:
        command.append("--stream")
    return command


def _post_commands(request: CampaignPlanRequest) -> dict[str, str]:
    pattern = str(Path(request.output_root) / "*" / "*")
    return {
        "validate_campaign": f"serve-optimize validate-campaign {pattern} --out results/validation-campaign",
        "research_package": f"serve-optimize research-package {pattern} --out results/research-package",
    }


def _planned_run_id(index: int, backend: str, goal: str, workload_profile: str, repeat: int) -> str:
    return f"{index:03d}-{_slug(backend)}-{_slug(goal)}-{_slug(workload_profile)}-r{repeat:02d}"


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value.lower()).strip("_") or "item"


def _number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _validate_request(request: CampaignPlanRequest) -> None:
    if not request.models:
        raise ValueError("at least one model is required.")
    if not request.backends:
        raise ValueError("at least one backend is required.")
    if not request.goals:
        raise ValueError("at least one goal is required.")
    if not request.workload_profiles:
        raise ValueError("at least one workload profile is required.")
    if request.repeats < 1:
        raise ValueError("repeats must be at least 1.")
    if request.limit < 1:
        raise ValueError("limit must be at least 1.")
    if request.trials < 1:
        raise ValueError("trials must be at least 1.")
    if request.soak_seconds is not None and request.soak_seconds <= 0:
        raise ValueError("soak seconds must be greater than 0 when provided.")
