from serve_optimize.budget import ManagedBudgetPolicy, select_promotion_decisions
from serve_optimize.schemas import EvaluationRung, Goal, PriorCandidate, PriorSource, RungResult, ServingConfig


def test_default_budget_policy_has_probe_measure_and_validate_rungs() -> None:
    policy = ManagedBudgetPolicy.default()

    assert [rung.name for rung in policy.rungs] == ["probe", "measure", "validate"]
    assert policy.rungs[0].num_requests_scale < policy.rungs[1].num_requests_scale
    assert policy.rungs[2].trials == 2


def test_nondominated_low_energy_candidate_is_promoted() -> None:
    candidates = [_config("fast"), _config("efficient"), _config("weak")]
    decisions = select_promotion_decisions(
        candidates=candidates,
        results=[
            _result("fast", throughput=200.0, energy=2.0),
            _result("efficient", throughput=120.0, energy=0.5),
            _result("weak", throughput=100.0, energy=3.0),
        ],
        prior_by_config_id={},
        goal=Goal.PERFORMANCE,
        from_rung=_rung("probe"),
        to_rung=_rung("measure", index=1),
        policy=ManagedBudgetPolicy.default(),
    )

    promoted = {decision.candidate_id for decision in decisions if decision.promoted}
    assert "efficient" in promoted
    efficient = next(decision for decision in decisions if decision.candidate_id == "efficient")
    assert "nondominated" in efficient.reason


def test_top_goal_score_candidate_is_promoted() -> None:
    candidates = [_config("cfg-a"), _config("cfg-b"), _config("cfg-c")]
    decisions = select_promotion_decisions(
        candidates=candidates,
        results=[
            _result("cfg-a", throughput=100.0, energy=None),
            _result("cfg-b", throughput=250.0, energy=None),
            _result("cfg-c", throughput=90.0, energy=None),
        ],
        prior_by_config_id={},
        goal=Goal.PERFORMANCE,
        from_rung=_rung("probe"),
        to_rung=_rung("measure", index=1),
        policy=ManagedBudgetPolicy.default(),
    )

    cfg_b = next(decision for decision in decisions if decision.candidate_id == "cfg-b")
    assert cfg_b.promoted is True
    assert "top_goal_score" in cfg_b.reason


def test_backend_default_baseline_is_preserved() -> None:
    candidates = [_config("baseline"), _config("better"), _config("other")]
    decisions = select_promotion_decisions(
        candidates=candidates,
        results=[
            _result("baseline", throughput=10.0, energy=None),
            _result("better", throughput=100.0, energy=None),
            _result("other", throughput=90.0, energy=None),
        ],
        prior_by_config_id={},
        goal=Goal.PERFORMANCE,
        from_rung=_rung("probe"),
        to_rung=_rung("measure", index=1),
        policy=ManagedBudgetPolicy.default(),
    )

    baseline = next(decision for decision in decisions if decision.candidate_id == "baseline")
    assert baseline.promoted is True
    assert "baseline" in baseline.reason


def test_prior_only_candidate_does_not_promote_without_measured_result() -> None:
    candidates = [_config("cfg-a"), _config("cfg-b")]
    decisions = select_promotion_decisions(
        candidates=candidates,
        results=[],
        prior_by_config_id={
            "cfg-b": PriorCandidate(
                source=PriorSource.AICONFIGURATOR.value,
                candidate_id="cfg-b",
                config_id="cfg-b",
                confidence=0.95,
            )
        },
        goal=Goal.BALANCED,
        from_rung=_rung("probe"),
        to_rung=_rung("measure", index=1),
        policy=ManagedBudgetPolicy.default(),
    )

    assert not any(decision.promoted for decision in decisions)


def _rung(name: str, index: int = 0) -> EvaluationRung:
    return EvaluationRung(index=index, name=name, purpose=name)


def _result(candidate_id: str, throughput: float, energy: float | None) -> RungResult:
    return RungResult(
        candidate_id=candidate_id,
        workload_id=f"{candidate_id}-probe",
        rung="probe",
        rung_index=0,
        status="completed",
        measured_or_evidence_source="measured",
        metrics={
            "throughput_tokens_per_sec": throughput,
            "joules_per_token": energy,
        },
    )


def _config(config_id: str) -> ServingConfig:
    return ServingConfig(
        id=config_id,
        backend="vllm",
        model_id="model-path",
        dtype="fp16",
        quantization="none",
        max_batch_size=2,
        max_context_tokens=2048,
        kv_cache_policy="paged",
        scheduler="continuous-batching",
    )
