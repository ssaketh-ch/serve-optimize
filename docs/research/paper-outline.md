# Paper Outline

This outline is a claim contract for a planned paper. A statement moves into the paper only after an audited artifact supports it. Product support is defined in [Compatibility](../compatibility.md), and the evaluation protocol is defined in [Experimental Methodology](../experiments.md).

Working title:

> Serve Optimize: Measurement Grounded Configuration Search for LLM Serving

## Evidence State

* RTX PRO 6000: completed tiny model managed evidence for vLLM and SGLang, plus a vLLM power gate.
* H200: closure job 319 completed 46 managed runs across vLLM and SGLang. The raw audit passed 397 trial summaries and 139352 request records with zero failed checks. Twelve performance cells did not establish server saturation and are excluded from throughput claims.
* Cross platform claim: two hardware classes are tested. Direct performance or energy deltas require matched software and workload cells.
* Search quality claim: pending equal budget baselines and ablations.
* Repeatability claim: pending matched repeated cells. The 46 H200 cells are heterogeneous and cannot be treated as replications.
* RTX PRO 6000 publication claim: historical recommendation artifacts remain operational evidence, but the trial records predate six current acceptance fields and are excluded from current inferential tables.

The scheduled validation phase is intentionally bounded. It executes fresh random, grid, Bayesian, and Serve Optimize candidate order searches against a common candidate pool, measures an independent oracle, repeats each candidate three times, and evaluates both the pinned OASST1 root English prompts and a deterministic holdout split. Hardware awareness, capability registry removal, energy scoring removal, and latency guardrail removal remain unsupported unless a later run produces their counterfactual candidate pools and fresh measurements.

## Abstract

State the serving configuration problem, the measured evidence workflow, the version aware recommendation method, the evaluated platforms, and only the results supported by the final audited matrix. Include recommendation regret and trial cost, not only end to end speedup.

## 1. Introduction

* LLM serving behavior depends on interacting backend, model, workload, and hardware choices.
* Backend defaults are necessary controls but are not an oracle for every objective.
* Tuning claims are unreliable when load, failures, warmup, token counts, or applied flags are not verified.
* Serve Optimize combines guarded candidate generation, controlled measurement, failure memory, and bounded recommendation scoring.
* Contributions should be limited to the implemented system, the measurement integrity protocol, the recommendation quality study, and the released evidence artifact.

## 2. Background And Related Work

* Continuous batching, KV cache pressure, prefill, and decode behavior.
* vLLM and SGLang configuration surfaces and version churn.
* Configuration search systems, including AIConfigurator.
* Inference energy measurement, including TokenPowerBench and ML energy benchmarks.
* Generic random, grid, and Bayesian optimization.

The novelty claim must be written as a measured distinction from prior work. Energy awareness, Pareto output, or backend launch automation alone is not sufficient.

## 3. System Design

* Attach Mode and Managed Mode lifecycle.
* Hardware, model, workload, and backend discovery.
* Versioned capability detection and canonical launch configuration.
* Candidate generation and explicit pruning records.
* Request measurement, telemetry, failure taxonomy, and evidence identity.
* Objective scoring, SLO guards, Pareto frontier, and recommendation report.

## 4. Recommendation Method

Define each candidate source, pruning rule, objective formula, failure penalty, load sufficiency rule, and evidence reuse decision. Explain why each recommended candidate was eligible and why pruned candidates were removed.

For every eligible matrix cell, report:

* oracle best measured candidate
* recommended candidate and rank
* regret relative to the oracle
* Pareto membership
* trials required
* failed trials avoided by pruning
* evidence reuse effect
* recommendation change after new evidence

## 5. Evaluation

### Research Questions

1. Do the measurement gates prevent misleading throughput, latency, energy, and failure summaries?
2. How close does the recommender come to the measured oracle under a fixed trial budget?
3. How do recommendation quality and selected configurations vary across backend, model, workload, objective, and platform strata?
4. When telemetry is defensible, how much energy efficiency changes while useful throughput and tail latency guardrails are preserved?
5. Which recommendation components matter under controlled ablation?

### Platforms

Report RTX PRO 6000 and H200 as distinct platform strata. Include exact GPU identity, memory, driver, CUDA runtime, Python, backend version, model revision, and command provenance. Do not infer per MIG instance energy from board scoped power.

### Baselines

* backend default
* human reasonable preset
* measured oracle on a bounded subset
* random search with equal trial budget
* grid search with equal trial budget
* generic Bayesian tuning with equal trial budget
* native backend benchmark cross check on a representative subset
* GuideLLM OpenAI HTTP cross check on a representative subset

### Ablations

* no evidence reuse
* no hardware awareness
* no capability registry
* no failure memory
* random candidate order
* no energy term
* no latency guardrail
* backend defaults only

### Statistical Protocol

Use repeated trials, confidence intervals, paired prompt sets where possible, explicit warmup and measurement windows, saturation evidence, and complete failure counts. Keep exploratory or contaminated runs outside inferential tables.

## 6. Results

Required figures:

* recommendation regret and rank by matrix cell
* trials to reach the recommendation
* throughput and latency relative to backend default
* joules per generated token where power evidence passes
* Pareto frontiers for representative cells
* failure taxonomy and failures avoided by pruning
* ablation results
* separate RTX PRO 6000 and H200 coverage summaries

Required tables:

* platform and software environments
* model and workload matrix
* backend configuration spaces
* objective formulas and guardrails
* main recommendation quality results
* baseline comparisons
* artifact completeness and exclusions

## 7. Discussion And Threats To Validity

* Results are bounded by measured candidate sets.
* Backend version differences can confound cross platform comparisons.
* Board scoped power cannot establish per instance energy.
* Short runs may not establish thermal steady state.
* Tiny models can be dominated by client or framework overhead.
* JSON prompt manifests do not reproduce timestamped production arrivals.
* Evidence reuse is valid only under exact runtime and workload identity.

## 8. Artifact And Reproducibility

Release commands, environment manifests, raw request records, raw telemetry, backend logs, failure records, validation outputs, analysis tables, checksums, and plotting code. Provide a small reviewer rerun path for both supported backends where hardware permits.

## 9. Conclusion

Summarize only audited results. Separate product capability, measured evidence, and future work.
