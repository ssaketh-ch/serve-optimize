import json

import pytest

from serve_optimize.workloads import load_workload_profile, workload_profile_to_payload, workload_prompts


def test_load_builtin_workload_profile_includes_token_distribution() -> None:
    profile = load_workload_profile(profile_name="decode-heavy")
    payload = workload_profile_to_payload(profile)

    assert payload["profile_name"] == "decode-heavy"
    assert payload["dataset"] == "synthetic-decode-heavy"
    assert payload["token_distribution"]["output_tokens"]["p95"] == 1024


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


def test_synthetic_profiles_generate_deterministic_materially_different_prompts() -> None:
    short = workload_prompts(load_workload_profile(profile_name="short"), count=2)
    medium = workload_prompts(load_workload_profile(profile_name="medium"), count=2)
    long = workload_prompts(load_workload_profile(profile_name="long"), count=2)

    assert short == workload_prompts(load_workload_profile(profile_name="short"), count=2)
    assert len(short[0]) < len(medium[0]) < len(long[0])


def test_mixed_code_and_repeated_prefix_profiles_shape_payloads() -> None:
    mixed = workload_prompts(load_workload_profile(profile_name="mixed"), count=8)
    code = workload_prompts(load_workload_profile(profile_name="code-generation"), count=2)
    repeated = workload_prompts(load_workload_profile(profile_name="repeated-prefix"), count=4)

    assert len({len(prompt) for prompt in mixed}) >= 3
    assert all("Python" in prompt and "code" in prompt for prompt in code)
    shared_prefix = repeated[0].split(" request_")[0]
    assert shared_prefix
    assert all(prompt.startswith(shared_prefix) for prompt in repeated)
    assert len({prompt[len(shared_prefix):] for prompt in repeated}) == len(repeated)


def test_workload_manifest_preserves_ordered_real_prompts_and_metadata(tmp_path) -> None:
    manifest = tmp_path / "real-workload.json"
    manifest.write_text(
        json.dumps(
            {
                "profile_name": "real-chat",
                "prompts": ["first prompt", "second prompt"],
                "dataset_source": "permitted-chat-set",
                "dataset_license": "research-only",
                "synthetic_or_real": "real",
            }
        ),
        encoding="utf-8",
    )

    profile = load_workload_profile(manifest_path=manifest)

    assert profile.prompts == ["first prompt", "second prompt"]
    assert profile.dataset_source == "permitted-chat-set"
    assert profile.dataset_license == "research-only"
    assert profile.synthetic_or_real == "real"
    assert workload_prompts(profile, count=4) == [
        "first prompt",
        "second prompt",
        "first prompt",
        "second prompt",
    ]


@pytest.mark.parametrize(
    "prompts",
    [[], "not a list", [""], ["valid", 2]],
)
def test_workload_manifest_rejects_malformed_prompt_lists(tmp_path, prompts) -> None:
    manifest = tmp_path / "invalid-workload.json"
    manifest.write_text(json.dumps({"prompts": prompts}), encoding="utf-8")

    with pytest.raises(ValueError, match="nonempty list of strings|nonempty string"):
        load_workload_profile(manifest_path=manifest)
