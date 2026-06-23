import json

from serve_optimize.overnight import summarize_campaign, write_outputs


def test_summarize_overnight_baseline_comparison(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "model" / "managed-run"
    run_dir.mkdir(parents=True)
    (run_dir / "managed_run.json").write_text(
        json.dumps({"model": "org/model", "backend": "vllm"}),
        encoding="utf-8",
    )
    (run_dir / "recommendation_summary.json").write_text(
        json.dumps(
            {
                "status": "success",
                "confidence": "high",
                "goal": "balanced",
                "selected": {},
                "baseline_comparison": {
                    "available": True,
                    "selected_candidate_id": "optimized",
                    "baseline_candidate_id": "baseline",
                    "selected_is_baseline": False,
                    "metrics": {
                        "throughput_tokens_per_sec": {
                            "baseline": 100.0,
                            "selected": 120.0,
                            "improvement_percent": 20.0,
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    rows = summarize_campaign(tmp_path)
    write_outputs(tmp_path, rows)

    assert rows[0]["model"] == "org/model"
    assert rows[0]["goal"] == "balanced"
    assert rows[0]["throughput_tokens_per_sec_improvement_percent"] == 20.0
    assert (tmp_path / "overnight_summary.csv").exists()
    assert (tmp_path / "overnight_summary.md").exists()
    summary = json.loads((tmp_path / "overnight_summary.json").read_text())
    assert summary["run_count"] == 1
    assert summary["failure_count"] == 0
    markdown = (tmp_path / "overnight_summary.md").read_text(encoding="utf-8")
    assert "100.0 tok/s to 120.0 tok/s (+20.0%)" in markdown


def test_summarize_overnight_includes_skipped_cells(tmp_path) -> None:
    (tmp_path / "failures.tsv").write_text(
        "timestamp\tbackend\tgoal\tmodel\tstatus\treason\tlog_path\n"
        "2026-06-23T00:00:00Z\tsglang\tenergy_efficient\torg/model\t137\toom\tlogs/model.log\n",
        encoding="utf-8",
    )

    write_outputs(tmp_path, [])

    summary = json.loads((tmp_path / "overnight_summary.json").read_text(encoding="utf-8"))
    markdown = (tmp_path / "overnight_summary.md").read_text(encoding="utf-8")

    assert summary["failure_count"] == 1
    assert summary["failures"][0]["reason"] == "oom"
    assert "Skipped Cells" in markdown
    assert "energy_efficient" in markdown
