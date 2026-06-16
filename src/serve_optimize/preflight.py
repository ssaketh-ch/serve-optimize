"""Shared preflight artifacts for user facing workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import write_json


@dataclass(frozen=True)
class PreflightRun:
    run_dir: Path
    payload: dict[str, Any]
    artifacts: dict[str, str]


def write_preflight_artifacts(run_dir: Path, payload: dict[str, Any]) -> PreflightRun:
    artifacts = {
        "run_dir": str(run_dir),
        "preflight_json": str(run_dir / "preflight.json"),
        "preflight_txt": str(run_dir / "preflight.txt"),
    }
    payload = {**payload, "artifacts": {**artifacts, **dict(payload.get("artifacts", {}))}}
    write_json(run_dir / "preflight.json", payload)
    (run_dir / "preflight.txt").write_text(format_preflight_text(payload), encoding="utf-8")
    return PreflightRun(run_dir=run_dir, payload=payload, artifacts=payload["artifacts"])


def format_preflight_text(payload: dict[str, Any]) -> str:
    mode = _text(payload.get("mode"))
    safety = _dict(payload.get("safety"))
    candidates = _dict(payload.get("candidates"))
    budget = _dict(payload.get("budget"))
    evidence = _dict(payload.get("evidence"))
    guidance = _dict(payload.get("guidance"))
    outputs = _dict(payload.get("outputs"))
    artifacts = _dict(payload.get("artifacts"))
    lines = [
        "Serve Optimize preflight",
        f"  mode: {mode}",
        f"  status: {_text(payload.get('status'))}",
        f"  dry run: {_yes_no(payload.get('dry_run'))}",
        f"  will call endpoint: {_yes_no(safety.get('will_call_endpoint'))}",
        f"  will launch servers: {_yes_no(safety.get('will_launch_servers'))}",
        f"  will write measured evidence: {_yes_no(safety.get('will_write_measured_evidence'))}",
        "",
        "Plan",
        f"  backend: {_text(payload.get('backend'))}",
        f"  model: {_text(payload.get('model'))}",
        f"  goal: {_text(payload.get('goal'))}",
        f"  telemetry: {_text(payload.get('telemetry'))}",
        f"  candidates: {_text(candidates.get('valid_count'))} valid of {_text(candidates.get('generated_count'))} generated",
        f"  rejected candidates: {_text(candidates.get('rejected_count'))}",
        f"  launch groups: {_text(budget.get('launch_group_count'))}",
        f"  planned workload measurements: {_text(budget.get('planned_workload_measurements'))}",
        "",
        "Evidence",
        f"  db: {_text(evidence.get('db_path'))}",
        f"  writes enabled: {_yes_no(evidence.get('write_enabled'))}",
        f"  exact reuse: {_text(evidence.get('exact_reuse'))}",
        "",
        "Outputs",
        f"  run dir: {_text(artifacts.get('run_dir'))}",
        f"  preflight json: {_text(artifacts.get('preflight_json'))}",
        f"  preflight text: {_text(artifacts.get('preflight_txt'))}",
    ]
    if outputs:
        lines.extend([f"  {key}: {_text(value)}" for key, value in sorted(outputs.items())])
    lines.extend(
        [
            "",
            "Guidance",
            f"  execute: {_text(guidance.get('execute'))}",
            f"  repeat: {_text(guidance.get('repeat'))}",
            f"  resume: {_text(guidance.get('resume'))}",
            "",
        ]
    )
    return "\n".join(lines)


def _dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text(value: object) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _yes_no(value: object) -> str:
    return "yes" if bool(value) else "no"
