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
    assert "Skipped Or Unavailable Cells" in markdown
    assert "energy_efficient" in markdown


def test_summarize_overnight_keeps_latest_attempt_per_cell(tmp_path) -> None:
    _write_run(
        tmp_path,
        "runs/vllm/balanced/org--model/managed-20260623T010000Z-old",
        throughput=100.0,
        run_id="managed-20260623T010000Z-old",
    )
    _write_run(
        tmp_path,
        "runs/vllm/balanced/org--model/managed-20260623T020000Z-new",
        throughput=130.0,
        run_id="managed-20260623T020000Z-new",
    )

    rows = summarize_campaign(tmp_path)
    write_outputs(tmp_path, rows)

    summary = json.loads((tmp_path / "overnight_summary.json").read_text(encoding="utf-8"))

    assert len(rows) == 1
    assert rows[0]["attempt_count"] == 2
    assert rows[0]["run_id"] == "managed-20260623T020000Z-new"
    assert rows[0]["throughput_tokens_per_sec_selected"] == 130.0
    assert summary["run_count"] == 1
    assert summary["attempt_count"] == 2


def test_summarize_overnight_filters_stale_failures(tmp_path) -> None:
    (tmp_path / "failures.tsv").write_text(
        "timestamp\tbackend\tgoal\tmodel\tstatus\treason\tlog_path\n"
        "2026-06-23T01:00:00Z\tvllm\tenergy_efficient\torg/model\t1\tbackend_launch\tlogs/old.log\n",
        encoding="utf-8",
    )
    _write_run(
        tmp_path,
        "runs/vllm/energy_efficient/org--model/managed-20260623T020000Z-new",
        throughput=130.0,
        run_id="managed-20260623T020000Z-new",
        goal="energy_efficient",
        summary_goal="efficiency",
    )

    rows = summarize_campaign(tmp_path)
    write_outputs(tmp_path, rows)

    summary = json.loads((tmp_path / "overnight_summary.json").read_text(encoding="utf-8"))

    assert rows[0]["goal"] == "energy_efficient"
    assert summary["run_count"] == 1
    assert summary["failure_count"] == 0
    assert summary["failure_attempt_count"] == 1


def test_summarize_overnight_filters_superseded_success(tmp_path) -> None:
    _write_run(
        tmp_path,
        "runs/vllm/balanced/org--model/managed-20260623T010000Z-old",
        throughput=130.0,
        run_id="managed-20260623T010000Z-old",
    )
    (tmp_path / "failures.tsv").write_text(
        "timestamp\tbackend\tgoal\tmodel\tstatus\treason\tlog_path\n"
        "2026-06-23T02:00:00Z\tvllm\tbalanced\torg/model\t1\tbackend_launch\tlogs/new.log\n",
        encoding="utf-8",
    )

    rows = summarize_campaign(tmp_path)
    write_outputs(tmp_path, rows)

    summary = json.loads((tmp_path / "overnight_summary.json").read_text(encoding="utf-8"))

    assert summary["run_count"] == 0
    assert summary["failure_count"] == 1
    assert summary["failures"][0]["reason"] == "backend_launch"


def test_summarize_overnight_moves_unavailable_rows_to_skipped(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "vllm" / "balanced" / "org--model" / "managed-20260623T010000Z-run"
    run_dir.mkdir(parents=True)
    (run_dir / "managed_run.json").write_text(
        json.dumps(
            {
                "model": "org/model",
                "backend": "vllm",
                "goal": "balanced",
                "run_id": "managed-20260623T010000Z-run",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "recommendation_summary.json").write_text(
        json.dumps(
            {
                "status": "unavailable",
                "goal": "balanced",
                "selected": {"model": "org/model", "backend": "vllm"},
                "baseline_comparison": {"available": False, "reason": "no viable candidates"},
            }
        ),
        encoding="utf-8",
    )

    rows = summarize_campaign(tmp_path)
    write_outputs(tmp_path, rows)

    summary = json.loads((tmp_path / "overnight_summary.json").read_text(encoding="utf-8"))

    assert summary["run_count"] == 0
    assert summary["failure_count"] == 1
    assert summary["failures"][0]["reason"] == "no viable candidates"


def test_summarize_overnight_enriches_generic_failure_reason(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "vllm" / "balanced" / "org_model" / "managed_run"
    log_dir = tmp_path / "logs"
    run_dir.mkdir(parents=True)
    log_dir.mkdir()
    (log_dir / "model.log").write_text(
        "Recommendation: unavailable\n"
        f"  artifacts: {run_dir}\n"
        "  reason: No candidates were available for scoring.\n",
        encoding="utf-8",
    )
    (run_dir / "candidate_failures.jsonl").write_text(
        json.dumps({"stage": "health", "error": "URLError: Connection refused", "details": {"status": "timeout"}})
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "failures.tsv").write_text(
        "timestamp\tbackend\tgoal\tmodel\tstatus\treason\tlog_path\n"
        "2026-06-23T00:00:00Z\tvllm\tbalanced\torg/model\t1\tcommand_failed\tlogs/model.log\n",
        encoding="utf-8",
    )

    write_outputs(tmp_path, [])

    summary = json.loads((tmp_path / "overnight_summary.json").read_text(encoding="utf-8"))

    assert summary["failure_count"] == 1
    assert summary["failures"][0]["reason"] == "startup_timeout"


def _write_run(
    tmp_path,
    relative_path: str,
    *,
    throughput: float,
    run_id: str,
    goal: str = "balanced",
    summary_goal: str | None = None,
) -> None:
    run_dir = tmp_path / relative_path
    run_dir.mkdir(parents=True)
    (run_dir / "managed_run.json").write_text(
        json.dumps({"model": "org/model", "backend": "vllm", "goal": goal, "run_id": run_id}),
        encoding="utf-8",
    )
    (run_dir / "recommendation_summary.json").write_text(
        json.dumps(
            {
                "status": "success",
                "confidence": "high",
                "goal": summary_goal or goal,
                "selected": {"model": "org/model", "backend": "vllm"},
                "baseline_comparison": {
                    "available": True,
                    "selected_candidate_id": "optimized",
                    "baseline_candidate_id": "baseline",
                    "selected_is_baseline": False,
                    "metrics": {
                        "throughput_tokens_per_sec": {
                            "baseline": 100.0,
                            "selected": throughput,
                            "improvement_percent": throughput - 100.0,
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
