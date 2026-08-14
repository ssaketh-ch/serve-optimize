from __future__ import annotations

import json
import runpy
from pathlib import Path

audit_root = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "audit_guidellm_results.py")
)["audit_root"]


def _distribution(total: float, *, p50: float = 1.0) -> dict[str, object]:
    return {"successful": {"mean": p50, "total_sum": total, "percentiles": {"p50": p50, "p95": p50, "p99": p50}}}


def test_audit_guidellm_result_checks_counts_tokens_and_metrics(tmp_path) -> None:
    run_dir = tmp_path / "guidellm" / "vllm-tiny-short"
    run_dir.mkdir(parents=True)
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "backend": "vllm",
                "backend_version": "0.24.0",
                "backend_command": "vllm serve model",
                "backend_target": "http://127.0.0.1:8000",
                "guidellm_command": "guidellm run",
                "guidellm_version": "0.7.3",
                "model": "model",
                "model_revision": "abc",
                "serve_optimize_command": "serve-optimize recommend model",
                "workload_profile": "short",
            }
        ),
        encoding="utf-8",
    )
    metrics = {
        "request_totals": {"successful": 2, "errored": 0, "incomplete": 2, "total": 4},
        "prompt_token_count": _distribution(20),
        "output_token_count": _distribution(8),
        "total_token_count": _distribution(28),
        "requests_per_second": _distribution(2, p50=2),
        "output_tokens_per_second": _distribution(8, p50=8),
        "request_latency": _distribution(1, p50=0.25),
        "time_to_first_token_ms": _distribution(1, p50=20),
        "time_per_output_token_ms": _distribution(1, p50=5),
    }
    report = {
        "metadata": {"guidellm_version": "0.7.3"},
        "benchmarks": [
            {
                "config": {"strategy": {"streams": streams}},
                "duration": 30,
                "metrics": metrics,
                "requests": {"successful": [{}, {}], "errored": [], "incomplete": [{}, {}]},
                "scheduler_metrics": {
                    "requests_made": {"successful": 2, "errored": 0, "incomplete": 2, "total": 4},
                    "queued_time_avg": 0.01,
                },
            }
            for streams in (16, 32, 64, 128, 256)
        ],
    }
    (run_dir / "benchmarks.json").write_text(json.dumps(report), encoding="utf-8")

    payload = audit_root(tmp_path)

    assert payload["report_count"] == 1
    assert payload["benchmark_count"] == 5
    assert payload["failed_benchmark_count"] == 0
    assert payload["rows"][0]["p50_latency_ms"] == 250
    assert payload["rows"][0]["scheduler_requests"] == 4


def test_audit_guidellm_result_preserves_failures(tmp_path) -> None:
    run_dir = tmp_path / "guidellm" / "failed"
    run_dir.mkdir(parents=True)
    (run_dir / "run_metadata.json").write_text("{}", encoding="utf-8")
    (run_dir / "benchmarks.json").write_text(
        json.dumps(
            {
                "metadata": {"guidellm_version": "0.7.3"},
                "benchmarks": [
                    {
                        "config": {"strategy": {"streams": 1}},
                        "metrics": {"request_totals": {"successful": 0, "errored": 1, "incomplete": 0}},
                        "requests": {"successful": [], "errored": [{}], "incomplete": []},
                        "scheduler_metrics": {
                            "requests_made": {"successful": 0, "errored": 1, "incomplete": 0, "total": 1}
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = audit_root(tmp_path)

    assert payload["failed_benchmark_count"] == 1
    assert "no_measured_errors" in payload["rows"][0]["errors"]
