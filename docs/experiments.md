# Planned Experimental Methodology

This document describes future evaluation work. It is not a statement that every metric, workload, or control is already implemented.

## Current Measured Baseline

Serve Optimize currently measures:

* end to end throughput
* request rate
* p50, p95, and p99 request latency
* request failures
* average and peak power when telemetry is available
* gross joules per token
* idle subtracted active energy when an idle baseline is available
* tokens per watt
* confidence intervals and stability classification across managed trials
* bounded evaluated candidate regret

Current recommendations are best among evaluated candidates.

## Planned Baselines

* backend default configuration
* highest throughput evaluated configuration
* best efficiency evaluated configuration
* Serve Optimize balanced recommendation
* bounded exhaustive baseline for optimizer regret studies

## Planned Workloads

* short chat
* medium assistant
* long context
* decode heavy
* repeated prefix
* mixed production trace

Workload manifests and token distributions must participate in evidence fingerprints before these are release supported.

## Planned Metrics

* prefill and decode energy
* larger cross hardware regret studies

These metrics must not appear as implemented until measurement boundaries and tests exist.

TTFT, stream chunk TPOT, and thermal trend reporting are implemented for endpoint benchmarks when the required measurement boundaries exist. Non streaming endpoint responses do not expose TTFT or TPOT. Short thermal windows are reported as limited evidence rather than stable soak results.

## Planned Controls

* fixed backend and model revisions
* fixed driver and CUDA environment per comparison
* explicit warmup policy
* steady state measurement window
* multiple trials
* idle power sampling
* no competing GPU workloads
* identical prompt sets across candidates

## Planned Comparisons

* vLLM versus SGLang
* model families and sizes
* BF16, FP16, AWQ, and GPTQ where valid
* concurrency and context length
* full GPU versus MIG where telemetry scope is defensible
* bounded exhaustive versus guided search

## Campaign Planning

Use `serve-optimize campaign-plan` to generate an explicit matrix of managed run commands before collecting evidence.

The planner records:

* models
* managed backends
* recommendation goals
* workload profiles
* repeat count
* candidate limits
* measurement quality controls
* validation and research package follow up commands

The planner does not run benchmarks, launch servers, or write measured evidence. Claims remain scoped to managed run artifacts that are later collected and validated.

Campaign artifacts include executable per backend runners because the validated vLLM and SGLang environments are mutually exclusive. The dispatcher selects one backend runner at a time. After all runners complete, `campaign_postprocess.sh` discovers the nested timestamped managed run directories for validation and research packaging.

## Reporting Rules

* Publish raw artifacts and environment metadata.
* Distinguish board level and instance level power.
* Report unsupported and failed candidates.
* Report confidence and telemetry limitations.
* Use best among evaluated candidates wording.
* Do not claim prefill or decode energy before implementation.
