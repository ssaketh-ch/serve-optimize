import json

import pytest

from serve_optimize.workloads import load_workload_profile, workload_profile_to_payload


def test_load_builtin_workload_profile_includes_token_distribution() -> None:
    profile = load_workload_profile(profile_name="decode-heavy")
    payload = workload_profile_to_payload(profile)

    assert payload["profile_name"] == "decode-heavy"
    assert payload["dataset"] == "synthetic-decode-heavy"
    assert payload["token_distribution"]["output_tokens"]["p95"] == 1536


def test_workload_manifest_overrides_preset_and_slos(tmp_path) -> None:
    manifest = tmp_path / "workload.json"
    manifest.write_text(
        json.dumps(
            {
                "profile_name": "short",
                "concurrency": 3,
                "dataset": "fixture",
                "token_distribution": {"input_tokens": {"p50": 33}},
                "slo_constraints": {"p95_latency_ms": 900},
            }
        ),
        encoding="utf-8",
    )

    profile = load_workload_profile(
        profile_name="medium",
        manifest_path=manifest,
        slo_constraints={"min_throughput_tokens_per_sec": 100},
    )
    payload = workload_profile_to_payload(profile)

    assert payload["profile_name"] == "short"
    assert payload["concurrency"] == 3
    assert payload["dataset"] == "fixture"
    assert payload["token_distribution"]["input_tokens"]["p50"] == 33
    assert payload["slo_constraints"]["p95_latency_ms"] == 900
    assert payload["slo_constraints"]["min_throughput_tokens_per_sec"] == 100


def test_unknown_slo_constraint_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported SLO constraint"):
        load_workload_profile(slo_constraints={"unknown": 1})


@pytest.mark.parametrize(
    ("constraints", "message"),
    [
        ({"ttft_ms": -1}, "must be nonnegative"),
        ({"max_failed_request_rate": 1.1}, "must be between 0 and 1"),
        ({"p95_latency_ms": float("nan")}, "must be a finite number"),
    ],
)
def test_invalid_slo_values_are_rejected(constraints: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_workload_profile(slo_constraints=constraints)
