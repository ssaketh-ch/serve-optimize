# System Design

Serve Optimize recommends LLM serving configurations from measured runtime behavior.

The product owns candidate generation, validation, benchmark execution, telemetry collection, evidence compatibility, metric computation, scoring, and recommendation artifacts.

AIConfigurator is an optional candidate source and prior. It is not the executor, measurement system, or source of truth.

For the complete flow with diagrams, see [Architecture](architecture.md). This page is the shorter design contract.

## Product Workflows

Serve Optimize supports two production style workflows:

* Attach Mode for measuring already running OpenAI compatible endpoints.
* Managed Mode for launching, measuring, and recommending configurations for vLLM and the detected supported SGLang surface.

Recommendations remain scoped to measured candidates, resumed completed workloads, or exact fresh measured evidence.

## Product Modes

### Attach Mode

Attach Mode targets an already running OpenAI compatible endpoint.

It:

* generates or imports candidate metadata and load shapes
* benchmarks the live endpoint
* collects optional host telemetry
* compares predicted and measured behavior
* scores measured operating points
* writes recommendation and Pareto artifacts

It does not:

* launch or stop the server
* prove that the endpoint was launched with a proposed serve command
* measure alternate launch configurations unless the operator changes the server

### Managed Mode

Managed Mode owns server lifecycle through backend adapters.

It:

* detects hardware, model metadata, backend availability, version, and capabilities
* generates bounded candidates
* rejects invalid or unsupported candidates before launch
* renders canonical launch commands
* checks runtime fingerprinted evidence
* launches vLLM or SGLang when measurement is required
* checks health through the OpenAI compatible endpoint
* runs the shared endpoint benchmark
* collects optional telemetry
* records evidence
* stops the launched process group
* writes lifecycle, failure, recommendation, and Pareto artifacts

## Architecture

```text
CLI
 |
 +-- hardware and model discovery
 |
 +-- candidate generation and optional AIConfigurator input
 |
 +-- validation and backend capability checks
 |
 +-- canonical launch and workload configuration
 |
 +-- runtime fingerprint and evidence compatibility
 |
 +-- Attach execution or Managed backend lifecycle
 |
 +-- shared OpenAI compatible endpoint benchmark
 |
 +-- optional telemetry
 |
 +-- measured evidence, scoring, Pareto analysis, and reports
```

## Layer Responsibilities

### Hardware And Model Discovery

Hardware discovery records NVIDIA GPU and MIG metadata when available. Model discovery reads local metadata conservatively and does not invent quantization compatibility.

### Candidate Generation

Candidate sources include:

* safe model native baselines
* heuristic candidates
* attach mode concurrency sweeps
* installed backend capability aware candidates
* exact or near evidence metadata
* bounded AIConfigurator proposals

Candidate sources remain advisory until validation and measurement.

### Validation

Validation rejects:

* incompatible quantization
* invalid values
* unsupported installed backend flags
* ambiguous cross backend field translations
* incompatible graph settings

Rejected candidates are recorded at stage `validation` and are not launched.

### Backend Adapters

Backend adapters own:

* availability and version detection
* help and capability metadata
* canonical capability hashes
* launch command rendering
* canonical launch metadata
* process start and stop
* health checks
* stdout and stderr log paths

vLLM and SGLang are first class Managed Mode backends for their documented surfaces.

### Canonical Launch Configuration

Logical candidate fields are canonicalized after backend rendering. Evidence, launch grouping, measurement records, recommendations, and summaries use active rendered behavior rather than unsupported or omitted fields.

`rendered_launch_configs.jsonl` records:

* command
* command hash
* canonical configuration
* canonical configuration hash
* rendered fields
* omitted fields
* unsupported fields
* unavailable fields
* flag aliases
* backend capability hash
* runtime environment

### Runtime Fingerprinting

Runtime identity includes backend, framework, compiler, command, capability, model, canonical configuration, and workload identities.

This prevents exact evidence reuse after meaningful runtime drift.

### Evidence

Evidence classification distinguishes:

* exact fresh
* exact stale
* near compatible
* incompatible
* unsupported under current backend
* missing runtime fingerprint
* runtime drift
* missing

Only exact fresh measured evidence can skip work.

### Benchmark Execution

Attach Mode and Managed Mode share the OpenAI compatible endpoint benchmark path. This keeps request behavior, metrics, telemetry integration, and benchmark artifacts consistent across backends.

### Telemetry

Telemetry providers emit generic optional fields. Missing values are unavailable, not zero.

Telemetry failure does not invalidate a benchmark by default. It may reduce confidence or make evidence unsuitable for an efficiency goal.

### Recommendation

Final recommendations use measured results or exact fresh measured evidence hits.

Recommendation artifacts expose:

* score weights and components
* selected candidate
* objective alternatives
* Pareto membership
* evaluated set fidelity
* telemetry confidence
* selected command
* evidence and runtime identity

The recommendation scope is always the best among evaluated candidates.

## Workloads And SLOs

Attach Mode and Managed Mode support workload profile presets and JSON workload manifests. Built in profiles cover short, medium, long, decode heavy, repeated prefix, and mixed synthetic workloads.

Workload identity includes profile name, dataset, token distribution, and SLO constraints. This identity participates in evidence fingerprints, so changing token distribution or constraints blocks exact evidence reuse.

SLO constraints are recommendation eligibility guards. TTFT, TPOT, p95 latency, throughput, and failed request rate constraints can make a candidate ineligible instead of merely lowering its score. Recommendation artifacts expose these guards through candidate status and workload profile metadata.

## Measurement Quality

Endpoint benchmarks support warmup request exclusion, steady state windows, soak duration targets, sampled idle baselines, supplied idle power baselines, and streaming response timing.

Benchmark summaries record both gross energy and idle subtracted active energy when an idle baseline is available. Managed multi trial runs aggregate trial summaries before recommendation and evidence writes, and aggregate summaries include confidence intervals plus a stability classification.

TTFT and TPOT are recorded only when streaming responses expose timed output chunks. TPOT is a stream chunk cadence metric unless the endpoint provides stronger token timing semantics. Evidence rows preserve idle subtracted power measurement type, timing source, and stability metadata through raw summaries. Changing measurement quality policy changes workload identity for exact evidence reuse.

Prefill and decode energy attribution is unavailable in the current measurement path because host telemetry and OpenAI compatible responses do not expose defensible phase boundary markers.

## Optimizer Quality

Recommendation artifacts include optimizer quality metadata scoped to evaluated candidates and exact fresh measured evidence hits. The payload records bounded evaluated candidate baselines, search regret, metric regret, and candidate source coverage.

Managed Mode also writes standalone `optimizer_quality.json` and `optimizer_failure_cache.json` artifacts. Failure cache entries use stable keys for equivalent failed serving configs so validation or runtime failures can be inspected and recognized without claiming unexplored configurations were evaluated.

## Release And Research Artifacts

`serve-optimize release-check` inspects local release readiness. It checks required files, package metadata, verification scripts, schema markers, and support documents, then writes `release_check.json` and `release_check.txt`.

`serve-optimize research-package` packages existing managed run artifacts. It reuses validation campaign analysis and writes a research manifest, methodology, run table, coverage table, and validation campaign summary. It does not launch servers or create measured evidence.

## Managed Lifecycle Failure Model

Managed Mode records:

* availability failures before launch
* launch exceptions
* health failures
* benchmark exceptions
* evidence read or write warnings
* stop failures
* operator interruption

If a server handle exists, cleanup runs through the stop path even after health, benchmark, or interruption failures.

Managed Mode can resume from a previous run directory with `--resume-from`. Resume reuses only completed measured workloads when candidate id, workload id, launch configuration hash, and workload configuration hash still match the current plan. Reused workloads are recorded as `resume_skip` lifecycle events and enter recommendations as measured resume results. Failed, unavailable, incomplete, or drifted workloads are not reused.

## Preflight UX

Attach Mode and Managed Mode both support `--dry-run`.

Attach dry run writes a candidate and benchmark plan without endpoint health checks or benchmark requests.

Managed dry run writes rendered launch configs, workload configs, launch groups, candidate validation failures, and shared preflight summaries without backend launches, health checks, benchmark requests, or measured evidence writes.

The shared preflight artifacts are:

* `preflight.json`
* `preflight.txt`

These artifacts explain planned backend, workload, budget, evidence, output, repeat, and resume behavior before execution.

## Artifact Principles

* JSON and JSONL artifacts are the automation source of truth.
* Human reports summarize, but do not replace, machine readable records.
* Measured, predicted, prior, and synthetic values remain distinguishable.
* Failed and unavailable runs keep diagnostics.
* Artifact changes should remain backward compatible where practical.

## Limits

* vLLM and SGLang require separate installation profiles because their validated runtime stacks conflict.
* Candidate generation is bounded rather than exhaustive.
* Workload profiles are not yet complete production trace manifests.
* Prefill and decode energy attribution is not implemented.
* TensorRT LLM is planned only and is not in the current Managed Mode scope.
* TGI, LMDeploy, llama.cpp, and NIM remain external Attach Mode targets unless lifecycle ownership is separately scoped.
* Managed candidates are evaluated sequentially.

See [Compatibility](compatibility.md) for the precise support contract.
