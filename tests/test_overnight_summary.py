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
    assert rows[0]["throughput_tokens_per_sec_improvement_percent"] == 20.0
    assert (tmp_path / "overnight_summary.csv").exists()
    assert json.loads((tmp_path / "overnight_summary.json").read_text())["run_count"] == 1
