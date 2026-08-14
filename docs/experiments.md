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

The completed RTX PRO 6000 evidence includes tiny model managed runs for vLLM and SGLang and a vLLM power gate. These historical records predate the current prompt distribution, output distribution, measurement duration, warmup ordering, p99 TTFT, and p99 TPOT acceptance fields, so they demonstrate the workflow but are not part of current inferential tables.

H200 closure job 319 completed 46 managed runs across vLLM and SGLang. Its raw audit covers 397 trial summaries and 139352 request records with zero failed checks. Twelve performance cells are usable artifacts but are excluded from throughput conclusions because their load sufficiency classification is `not_saturated`.

Results across the two platforms may be compared directly only when model revision, backend version, workload, candidate policy, repetition policy, and measurement controls match. Otherwise, report each platform independently as evidence that the workflow operates across distinct hardware classes. MIG power remains board scoped unless the telemetry provider establishes instance attribution.

GuideLLM 0.7.3 is the independent OpenAI HTTP benchmark client for the final cross checks. Use the same model revision, synthetic token shape, static seed, concurrency sweep, and measurement duration as the Serve Optimize client. Each concurrency point uses GuideLLM's native 90 second warmup followed by a 60 second measured window. Preserve its raw request records and complete p50, p95, and p99 latency, TTFT, and TPOT metrics. GuideLLM validates the measurement client; it is not a configuration search baseline.

GuideLLM duration bounded runs cancel the final in flight wave at the measurement boundary. Treat at most one terminal incomplete request per configured stream as boundary accounting and report it separately. A zero success point is overload evidence only when lower concurrency points succeeded, no higher point succeeded, the saturation monitor was enabled, and no request errored. Continue to reject isolated failures, any errored request, or a larger incomplete count.

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

Equal budget replay must use the common probe rung measured for every candidate. Do not mix probe, measure, and validation rung scores. Label the Serve Optimize series as a candidate order replay, keep the oracle post hoc, and retain the fixed measured pool limitation in every output.

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
* Compute repeatability only across runs with matching backend version, model, goal, and workload profile.
* Do not claim prefill or decode energy, power cap optimization, exhaustive coverage, or search superiority before the required evidence exists.
