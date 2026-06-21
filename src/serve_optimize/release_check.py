"""Release readiness checks for local product packaging."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import write_json

RELEASE_CHECK_SCHEMA_VERSION = "release-check/v1"

REQUIRED_FILES = (
    ".github/workflows/ci.yml",
    "CONTRIBUTING.md",
    "LICENSE",
    "README.md",
    "SECURITY.md",
    "configs/backends.yaml",
    "configs/serving_engines.yaml",
    "docs/compatibility.md",
    "docs/decisions/0003-backend-expansion-scope.md",
    "docs/design.md",
    "docs/installation.md",
    "docs/product_readiness.md",
    "docs/release.md",
    "docs/support_matrix.md",
    "docs/verification.md",
    "feature_list.json",
    "pyproject.toml",
    "requirements/README.md",
    "scripts/verify_fast.sh",
    "scripts/verify_full.sh",
)

FAST_COMMANDS = (
    "python -m compileall -q src tests",
    "pytest -q",
    "ruff check .",
    "python -m json.tool feature_list.json",
    "serve-optimize --help",
    "serve-optimize managed-evaluate --help",
    "serve-optimize validate-campaign --help",
    "serve-optimize campaign-plan --help",
)

FULL_COMMANDS = (
    *FAST_COMMANDS,
    "serve-optimize release-check --help",
    "serve-optimize research-package --help",
    "python -m build",
)

REQUIRED_EXTRA_GROUPS = ("telemetry", "vllm", "sglang", "dev")


def run_release_check(*, root: Path | None = None) -> dict[str, Any]:
    root = (root or Path.cwd()).resolve()
    checks: list[dict[str, Any]] = []
    checks.extend(_required_file_checks(root))
    checks.extend(_pyproject_checks(root))
    checks.extend(_verification_script_checks(root))
    checks.extend(_ci_checks(root))
    checks.extend(_schema_checks(root))
    checks.extend(_support_document_checks(root))
    failed = [check for check in checks if check["status"] == "fail"]
    warnings = [check for check in checks if check["status"] == "warn"]
    return {
        "schema_version": RELEASE_CHECK_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "status": "fail" if failed else "pass",
        "summary": {
            "check_count": len(checks),
            "failed_count": len(failed),
            "warning_count": len(warnings),
        },
        "checks": checks,
        "notes": [
            "Release check inspects local packaging and documentation readiness.",
            "It does not run backend measurements or broad benchmark campaigns.",
        ],
    }


def write_release_check_artifacts(*, out_dir: Path, root: Path | None = None) -> dict[str, Any]:
    payload = run_release_check(root=root)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "release_check.json"
    text_path = out_dir / "release_check.txt"
    payload["artifacts"] = {
        "release_check_json": str(json_path),
        "release_check_txt": str(text_path),
    }
    write_json(json_path, payload)
    text_path.write_text(format_release_check_text(payload), encoding="utf-8")
    return payload


def format_release_check_text(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    lines = [
        "Serve Optimize release check",
        "",
        f"status: {payload.get('status')}",
        f"checks: {summary.get('check_count')}",
        f"failed: {summary.get('failed_count')}",
        f"warnings: {summary.get('warning_count')}",
        "",
        "Checks:",
    ]
    for check in payload.get("checks", []):
        if not isinstance(check, dict):
            continue
        lines.append(f"  {check.get('status')}: {check.get('name')} {check.get('message')}")
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    if artifacts:
        lines.extend(["", "Artifacts:"])
        for key, value in artifacts.items():
            lines.append(f"  {key}: {value}")
    return "\n".join(lines) + "\n"


def _required_file_checks(root: Path) -> list[dict[str, Any]]:
    return [
        _check(
            name=f"required_file:{relative_path}",
            status="pass" if (root / relative_path).is_file() else "fail",
            message="present" if (root / relative_path).is_file() else "missing",
        )
        for relative_path in REQUIRED_FILES
    ]


def _pyproject_checks(root: Path) -> list[dict[str, Any]]:
    path = root / "pyproject.toml"
    if not path.is_file():
        return [_check(name="pyproject", status="fail", message="pyproject.toml missing")]
    text = path.read_text(encoding="utf-8")
    checks = [
        _check("pyproject:name", "pass" if 'name = "serve-optimize"' in text else "fail", "project name"),
        _check("pyproject:version", "pass" if "version =" in text else "fail", "project version"),
        _check("pyproject:license", "pass" if 'license = "Apache-2.0"' in text else "fail", "project license"),
        _check(
            "pyproject:repository_url",
            "pass" if 'Homepage = "https://github.com/ssaketh-ch/serve-optimize"' in text else "fail",
            "repository URL",
        ),
        _check(
            "pyproject:cli",
            "pass" if 'serve-optimize = "serve_optimize.cli:main"' in text else "fail",
            "console script",
        ),
    ]
    for group in REQUIRED_EXTRA_GROUPS:
        checks.append(
            _check(
                f"pyproject:extra:{group}",
                "pass" if f"{group} = [" in text else "fail",
                "extra present",
            )
        )
    checks.append(
        _check(
            "pyproject:build_dependency",
            "pass" if '"build>=1.2,<2"' in text and '"hatchling>=1.24"' in text else "fail",
            "build frontend and backend available in development profile",
        )
    )
    return checks


def _verification_script_checks(root: Path) -> list[dict[str, Any]]:
    checks = []
    fast_text = _read_text(root / "scripts/verify_fast.sh")
    full_text = _read_text(root / "scripts/verify_full.sh")
    for command in FAST_COMMANDS:
        checks.append(
            _check(
                f"verify_fast:{command}",
                "pass" if command in fast_text else "fail",
                "standard command present",
            )
        )
    for command in FULL_COMMANDS:
        checks.append(
            _check(
                f"verify_full:{command}",
                "pass" if command in full_text else "fail",
                "full command present",
            )
        )
    return checks


def _schema_checks(root: Path) -> list[dict[str, Any]]:
    source_files = [
        root / "src/serve_optimize/validation_campaign.py",
        root / "src/serve_optimize/release_check.py",
        root / "src/serve_optimize/research_package.py",
    ]
    checks = []
    for path in source_files:
        text = _read_text(path)
        checks.append(
            _check(
                f"schema_version:{path.name}",
                "pass" if "SCHEMA_VERSION" in text or "schema_version" in text else "fail",
                "schema marker present",
            )
        )
    return checks


def _ci_checks(root: Path) -> list[dict[str, Any]]:
    text = _read_text(root / ".github/workflows/ci.yml")
    return [
        _check(
            "ci:python_matrix",
            "pass" if all(version in text for version in ('"3.10"', '"3.11"', '"3.12"')) else "fail",
            "supported Python versions covered",
        ),
        _check(
            "ci:package_build",
            "pass" if "python -m build" in text and "serve-optimize --version" in text else "fail",
            "wheel build and installed CLI smoke covered",
        ),
    ]


def _support_document_checks(root: Path) -> list[dict[str, Any]]:
    backend_config_text = _read_text(root / "configs/backends.yaml")
    serving_engine_text = _read_text(root / "configs/serving_engines.yaml")
    release_text = _read_text(root / "docs/release.md")
    requirements_text = _read_text(root / "requirements/README.md")
    support_text = _read_text(root / "docs/support_matrix.md")
    compatibility_text = _read_text(root / "docs/compatibility.md")
    backend_factory_text = _read_text(root / "src/serve_optimize/backends/factory.py")
    backend_scope_decision_text = _read_text(root / "docs/decisions/0003-backend-expansion-scope.md")
    trt_adapter_exists = (root / "src/serve_optimize/backends/trt_llm.py").exists()
    return [
        _check("docs:release:phase8", "pass" if "Phase Eight" in release_text else "fail", "release phase recorded"),
        _check(
            "docs:release:git_wording",
            "pass" if "not a Git repository" not in release_text else "fail",
            "workspace wording",
        ),
        _check("docs:support:vllm", "pass" if "vLLM" in support_text else "fail", "vLLM support recorded"),
        _check("docs:support:sglang", "pass" if "SGLang" in support_text else "fail", "SGLang support recorded"),
        _check("docs:compat:tensorrt", "pass" if "TensorRT LLM | Planned only" in compatibility_text else "fail", "TensorRT LLM exclusion recorded"),
        _check(
            "docs:compat:attach_only_external",
            "pass" if "TGI, LMDeploy, llama.cpp, NIM | Attach only" in compatibility_text else "fail",
            "external engine Attach Mode boundary recorded",
        ),
        _check(
            "docs:decision:backend_expansion",
            "pass" if "TensorRT LLM is planned only" in backend_scope_decision_text and "engine build lifecycle" in backend_scope_decision_text else "fail",
            "backend expansion decision and admission gate recorded",
        ),
        _check(
            "configs:backends:tensorrt",
            "pass" if "trt-llm:\n    status: planned_only" in backend_config_text else "fail",
            "TensorRT LLM planned only status recorded",
        ),
        _check(
            "configs:serving_engines:tensorrt",
            "pass" if "tensorrt_llm:\n    priority: 3\n    status: planned_only" in serving_engine_text else "fail",
            "TensorRT LLM serving engine scope recorded",
        ),
        _check(
            "managed_backends:registered_scope",
            "pass" if 'SUPPORTED_MANAGED_BACKENDS = ("vllm", "sglang")' in backend_factory_text else "fail",
            "managed backend registration limited to vLLM and SGLang",
        ),
        _check(
            "managed_backends:tensorrt_adapter_absent",
            "pass" if not trt_adapter_exists else "fail",
            "TensorRT LLM adapter remains absent until engine lifecycle design is accepted",
        ),
        _check(
            "configs:backends:vllm",
            "pass" if "vllm:\n    status: first_class_managed" in backend_config_text else "fail",
            "vLLM managed status recorded",
        ),
        _check(
            "configs:backends:sglang",
            "pass" if "sglang:\n    status: first_class_managed" in backend_config_text else "fail",
            "SGLang managed status recorded",
        ),
        _check(
            "configs:serving_engines:vllm",
            "pass" if "vllm:\n    priority: 1\n    status: first_class_managed" in serving_engine_text else "fail",
            "vLLM serving engine status recorded",
        ),
        _check(
            "configs:serving_engines:sglang",
            "pass" if "sglang:\n    priority: 2\n    status: first_class_managed" in serving_engine_text else "fail",
            "SGLang serving engine status recorded",
        ),
        _check(
            "configs:backends:attach_only_external",
            "pass"
            if all(
                marker in backend_config_text
                for marker in (
                    "tgi:\n    status: attach_only_external",
                    "lmdeploy:\n    status: attach_only_external",
                    "llama-cpp:\n    status: attach_only_external",
                )
            )
            else "fail",
            "external backend Attach Mode statuses recorded",
        ),
        _check(
            "configs:serving_engines:attach_only_external",
            "pass"
            if all(
                marker in serving_engine_text
                for marker in (
                    "tgi:\n    priority: 4\n    status: attach_only_external",
                    "lmdeploy:\n    priority: 5\n    status: attach_only_external",
                    "llama_cpp:\n    priority: 6\n    status: attach_only_external",
                )
            )
            else "fail",
            "external serving engine Attach Mode statuses recorded",
        ),
        _check(
            "requirements:core_scope",
            "pass" if "synthetic paths, endpoint client" in requirements_text and "tests, docs" not in requirements_text else "fail",
            "core profile scope recorded",
        ),
    ]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def _check(name: str, status: str, message: str) -> dict[str, Any]:
    return {"name": name, "status": status, "message": message}
