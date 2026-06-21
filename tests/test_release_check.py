from pathlib import Path

from serve_optimize.release_check import run_release_check, write_release_check_artifacts


def test_release_check_passes_for_repository_root(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]

    payload = write_release_check_artifacts(out_dir=tmp_path / "release", root=root)

    assert payload["schema_version"] == "release-check/v1"
    assert payload["status"] == "pass"
    assert payload["summary"]["failed_count"] == 0
    checks = {check["name"]: check for check in payload["checks"]}
    assert checks["pyproject:repository_url"]["status"] == "pass"
    assert checks["pyproject:build_dependency"]["status"] == "pass"
    assert checks["ci:python_matrix"]["status"] == "pass"
    assert checks["ci:package_build"]["status"] == "pass"
    assert checks["required_file:SECURITY.md"]["status"] == "pass"
    assert (tmp_path / "release" / "release_check.json").exists()
    assert (tmp_path / "release" / "release_check.txt").exists()


def test_release_check_fails_for_missing_required_files(tmp_path) -> None:
    payload = run_release_check(root=tmp_path)

    assert payload["status"] == "fail"
    assert payload["summary"]["failed_count"] > 0


def test_release_check_fails_for_stale_backend_support_config(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    mirror = tmp_path / "repo"
    for relative_path in (
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
        "src/serve_optimize/validation_campaign.py",
        "src/serve_optimize/backends/factory.py",
        "src/serve_optimize/release_check.py",
        "src/serve_optimize/research_package.py",
    ):
        source = root / relative_path
        target = mirror / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    (mirror / "configs/backends.yaml").write_text(
        (mirror / "configs/backends.yaml").read_text(encoding="utf-8").replace(
            "vllm:\n    status: first_class_managed",
            "vllm:\n    status: planned",
        ),
        encoding="utf-8",
    )

    payload = run_release_check(root=mirror)

    assert payload["status"] == "fail"
    checks = {check["name"]: check for check in payload["checks"]}
    assert checks["configs:backends:vllm"]["status"] == "fail"
    assert checks["docs:decision:backend_expansion"]["status"] == "pass"
    assert checks["managed_backends:tensorrt_adapter_absent"]["status"] == "pass"

    trt_adapter = mirror / "src/serve_optimize/backends/trt_llm.py"
    trt_adapter.parent.mkdir(parents=True, exist_ok=True)
    trt_adapter.write_text("class TrtLlmAdapter: pass\n", encoding="utf-8")
    payload = run_release_check(root=mirror)
    checks = {check["name"]: check for check in payload["checks"]}
    assert checks["managed_backends:tensorrt_adapter_absent"]["status"] == "fail"
