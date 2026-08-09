# Experimental Methodology

This document separates the implemented measurement protocol from the evidence still required for a paper. Product support is defined in [Compatibility](compatibility.md).

## Implemented Measurement Protocol

Managed trials record:

* raw request records, including success or failure and token counts
* output token throughput, total token throughput, and request rate
* p50, p95, and p99 request latency
* streaming TTFT and TPOT when response chunks expose those boundaries
* client CPU utilization, issue rate, and queueing indicators when available
* GPU utilization, memory use, request backlog, and token backlog when available
* idle, warmup, and measurement power samples as separate records
* measurement window energy, joules per generated token, and tokens per joule
* raw or idle subtracted energy accounting
* backend version, exact launch command, applied configuration, and runtime fingerprint
* candidate failures using the documented failure taxonomy
* confidence intervals, stability classification, recommendation rank, Pareto status, and bounded regret when enough candidates are measured

Recommendations remain scoped to the evaluated candidate set.

## Measurement Acceptance Rules

A run is publishable only when its raw artifacts establish all applicable checks:

1. Client request records and summary counts agree.
2. Input, output, and total token counts are nonzero.
3. Failed requests are excluded from successful throughput.
4. Latency, TTFT, and TPOT percentiles are derived from request records.
5. Warmup requests and warmup power samples are excluded from measurement summaries.
6. Backend version, launch command, applied configuration, model identity, and runtime fingerprint are present.
7. Throughput conclusions include load sufficiency and client saturation evidence.
8. Energy conclusions use the measurement window and disclose telemetry quality and accounting mode.
9. Failed candidates retain a precise failure reason.
10. No competing GPU workload is present during a controlled comparison.

## Hardware Evidence Strata

Serve Optimize has measured both RTX PRO 6000 and H200 systems. They are separate evidence strata, not a single pooled platform result.

The completed RTX PRO 6000 evidence includes tiny model managed runs for vLLM and SGLang and a vLLM power gate. The H200 campaign expands model, workload, objective, and backend coverage.

Results across the two platforms may be compared directly only when model revision, backend version, workload, candidate policy, repetition policy, and measurement controls match. Otherwise, report each platform independently as evidence that the workflow operates across distinct hardware classes. MIG power remains board scoped unless the telemetry provider establishes instance attribution.

## Workloads

Implemented workload profiles include:

* short chat
* medium assistant
* long context
* long prefill
* decode heavy
* code generation
* repeated prefix
* mixed lengths
* JSON prompt manifests, including permitted real prompt datasets

JSON manifests preserve prompt content and workload settings. They do not replay original production arrival timestamps.

## Publication Baselines

Every reported matrix cell should include a backend default control and a conservative reliability candidate. The paper also requires:

* a human reasonable preset
* the oracle best measured candidate for each objective
* random search with the same trial budget
* grid search with the same trial budget
* a generic Bayesian tuner with the same trial budget
* native vLLM or SGLang benchmark results on a representative subset

The current artifacts do not yet establish superiority over all of these baselines. That claim remains pending.

## Recommendation Ablations

Run the following ablations on a bounded subset with a known measured oracle:

* no evidence reuse
* no hardware awareness
* no backend capability registry
* no failure memory
* random candidate order
* energy term removed
* latency guardrail removed
* backend defaults only

Report regret, selected rank, Pareto membership, trials to recommendation, failed trials avoided, and recommendation changes after new evidence.

## Matrix Planning

Use `serve-optimize benchmark-matrix-plan` for the staged paper matrix and `serve-optimize campaign-plan` for a smaller custom campaign.

Both planners write commands and provenance without launching a backend or creating measured evidence. Backend specific runners keep the validated vLLM and SGLang environments isolated. Postprocessing discovers completed managed runs, validates them, and builds analysis tables.

## Reporting Rules

* Publish raw request, telemetry, lifecycle, capability, launch, and summary artifacts.
* Archive raw run directories with checksums. A research package alone is an index, not a raw artifact archive.
* Distinguish RTX PRO 6000 and H200 evidence unless the comparison is matched.
* Distinguish board level and instance level power.
* Report unsupported, failed, incomplete, and provisional candidates.
* Report confidence intervals, trial counts, and telemetry limitations.
* Use `best among evaluated candidates` wording.
* Do not claim prefill or decode energy, power cap optimization, exhaustive coverage, or search superiority before the required evidence exists.
