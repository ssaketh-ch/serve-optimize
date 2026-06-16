from pathlib import Path

from serve_optimize.release_check import run_release_check, write_release_check_artifacts


def test_release_check_passes_for_repository_root(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]

    payload = write_release_check_artifacts(out_dir=tmp_path / "release", root=root)

    assert payload["schema_version"] == "release-check/v1"
    assert payload["status"] == "pass"
    assert payload["summary"]["failed_count"] == 0
    assert (tmp_path / "release" / "release_check.json").exists()
    assert (tmp_path / "release" / "release_check.txt").exists()


def test_release_check_fails_for_missing_required_files(tmp_path) -> None:
    payload = run_release_check(root=tmp_path)

    assert payload["status"] == "fail"
    assert payload["summary"]["failed_count"] > 0
