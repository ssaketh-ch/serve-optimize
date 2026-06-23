# Compatibility Contract

This document defines the current supported product surface. It is the authority for backend, evidence, artifact, installation, and exclusion claims.

## Support Levels

* First class: implemented, tested, and validated with real runtime artifacts on the current validation host.
* Supported when available: implemented and tested, but dependent on local runtime or hardware availability.
* Planned: not implemented and must not be presented as available.

## Operating Modes

| Mode | Support | Contract |
|---|---|---|
| Attach Mode | First class | Benchmarks an already running OpenAI compatible endpoint. It does not launch or stop the server and cannot prove the endpoint launch command. |
| Managed Mode | First class | Generates and validates candidates, checks evidence, launches supported backends, benchmarks OpenAI compatible endpoints, captures logs, stops process groups, and writes recommendations. |
| Synthetic and local smoke paths | Supported when available | Used for schema, optimizer, artifact, and functional validation. Synthetic values are never measured evidence. |
| Release check | First class | Inspects local release readiness without backend measurements. |
| Research package | First class | Packages existing managed run artifacts for analysis without launching servers. |

## Managed Backends

| Backend | Support | Validated runtime | Notes |
|---|---|---|---|
| vLLM | First class | vLLM `0.23.0`, Torch `2.11.0+cu130`, Python `3.12.3` | Installed capability detection, canonical rendering, lifecycle, evidence, and recommendation paths are validated on the current Blackwell host. |
| SGLang | First class for the detected supported surface | SGLang `0.5.13.post1`, Torch `2.11.0`, Transformers `5.8.1` | The clean install profile resolves on Python 3.12. Runtime support is bounded by installed capability detection and is validated by a local profile doctor and Qwen smoke run. |
| TensorRT LLM | Planned only | none | Not in current Managed Mode scope. No adapter, engine build lifecycle, evidence, or recommendation support exists. |
| TGI, LMDeploy, llama.cpp, NIM | Attach only | none | They may be measured through Attach Mode when they expose a compatible endpoint. Serve Optimize does not own their Managed Mode lifecycle. |

First class SGLang support does not mean universal parity with every SGLang option. It means parity for the capability detected Managed Mode surface.

## Backend Option Contract

Common managed fields include model, dtype, quantization, context length, memory policy, request capacity, tensor parallelism, and workload settings.

vLLM directly supports additional detected engine fields such as:

* block size
* KV cache dtype
* eager execution
* maximum batched tokens
* chunked prefill
* CUDA graph capture sizing
* prefix caching

SGLang directly supports detected fields such as:

* memory fraction
* maximum running requests
* model compatible AWQ or GPTQ
* chunked prefill size
* radix cache disable
* CUDA graph disable and maximum batch size
* served model name
* remote code trust
* piecewise CUDA graph disable

Fields without direct semantics are rejected before launch or recorded as unsupported or unavailable. They are not silently approximated.

## Availability And Capability Detection

Each managed adapter owns:

* runtime availability detection
* installed backend version detection
* help and capability parsing
* canonical capability hashing
* command rendering
* canonical launch metadata
* process lifecycle
* OpenAI compatible health checks
* stdout and stderr log paths

Capability hashes are derived from normalized parsed capabilities, not raw timestamped help output.

## Runtime Evidence Contract

Measured evidence records include:

* hardware fingerprint
* backend fingerprint
* model fingerprint
* telemetry fingerprint
* canonical launch configuration hash
* workload configuration hash
* runtime fingerprint
* measured throughput, latency, stream timing, power, energy, thermal, and efficiency fields when available

The runtime fingerprint includes:

* backend name and version
* Torch version
* CUDA runtime version
* Python version
* compiler toolchain identity
* Serve Optimize revision when available
* rendered launch command hash
* backend capability hash
* canonical launch identity
* model identity
* workload identity

## Exact Evidence Reuse

Exact reuse is allowed only for fresh measured evidence with compatible hardware, backend, runtime, command, capabilities, model, canonical configuration, workload, and telemetry requirements.

The following cannot be exact hits:

* stale evidence
* near compatible evidence
* prior predictions
* synthetic measurements
* legacy rows without runtime fingerprints
* runtime drift
* backend capability drift
* rendered command drift
* canonical configuration drift
* required telemetry incompatibility

Runtime drift is recorded explicitly in `evidence_decisions.jsonl`.

TTFT and TPOT evidence is available only when the benchmark uses streaming and observes response chunks. Non streaming responses do not provide these metrics. Current energy evidence covers gross and idle subtracted active windows only. Prefill and decode energy attribution is unavailable without backend phase markers or equivalent request trace events.

## Managed Resume

Managed Mode supports `--resume-from RUN_DIR` for prior managed run directories.

Resume may reuse only completed measured workload artifacts when all of these match the current plan:

* candidate id
* workload id
* canonical launch configuration hash
* workload configuration hash

Resume does not reuse failed, unavailable, incomplete, stale, or drifted workloads. Reused workloads are recorded in `server_lifecycle.jsonl` with `resume_skip` events and are counted separately in `managed_run.json`.

## Artifact Contract

Machine readable JSON and JSONL artifacts are the source of truth.

Every managed run initializes:

* `managed_run.json`
* `runtime_environment.json`
* backend capability artifact
* `rendered_launch_configs.jsonl`
* `launch_specs.jsonl`
* `launch_groups.json`
* `workload_configs.jsonl`
* `server_lifecycle.jsonl`
* `candidate_failures.jsonl`
* `evidence_decisions.jsonl`
* prior, synthesis, rung, and promotion artifacts
* optimizer quality and failure cache artifacts
* recommendation, Pareto, report, and summary artifacts

Launched backends also receive stdout and stderr log paths.

Unavailable or failed runs preserve diagnostics. They must not create a misleading successful recommendation.

## Recommendation Contract

Final Managed Mode recommendations use:

* measured results
* resumed completed measured workload results
* exact fresh measured evidence hits

Predictions, stale evidence, and near compatible evidence may influence pruning or promotion but cannot become final measured truth.

Recommendation wording is limited to the evaluated set. Use `best among evaluated candidates`. Do not claim all possible configuration coverage.

## Release And Research Contract

`serve-optimize release-check` writes release readiness artifacts and checks required files, package metadata, verification scripts, schema markers, and support documents.

`serve-optimize research-package` writes research manifests, methodology, run tables, coverage tables, and validation campaign summaries from existing managed run artifacts. It must not launch servers or create measured evidence.

## Telemetry Contract

Telemetry is optional. Providers emit generic fields and capability metadata.

Missing fields mean unavailable unless a provider explicitly measured zero.

Telemetry failure does not fail a benchmark by default. It lowers confidence, prevents poor power samples from affecting efficiency scoring, and may prevent exact reuse for a power aware goal.

Energy metrics include gross active window estimates. When an idle baseline is supplied or sampled, summaries also include idle subtracted active energy and efficiency. Recommendation scoring prefers idle subtracted metrics when they are available. Prefill and decode attribution remains unavailable because the endpoint path has no defensible phase boundary markers.

## Installation Contract

Current core development install:

```bash
uv pip install -e ".[dev]"
```

Validated backend measurements use isolated environments installed from:

* `requirements/profiles/vllm.txt`
* `requirements/profiles/sglang.txt`

Backend extras and requirement profiles are pinned to the validated vLLM and SGLang stacks. The backend profiles are mutually exclusive because their Torch and Transformers requirements conflict.

See `docs/installation.md` for clean uv installation and profile verification commands.

Unvalidated backend versions are not part of the support contract. SSL verification must remain enabled for dependency installation.

## Explicit Exclusions

The current product does not provide:

* TensorRT LLM Managed Mode
* universal backend option parity
* Kubernetes or cluster orchestration
* multi node candidate execution
* parallel managed candidate launches
* power limit or clock control
* prefill and decode energy attribution
* production workload trace manifests
* all possible configuration coverage guarantees

## Preflight And Dry Run

Attach Mode supports `--dry-run` to write preflight artifacts and candidate plans without endpoint health checks or benchmark requests.

Managed Mode supports `--dry-run` to write preflight artifacts, rendered launch configs, workload configs, launch groups, and validation failures without backend launches, health checks, benchmark requests, or measured evidence writes.

Dry run artifacts are planning artifacts. They are not measured evidence and cannot justify a final recommendation.

## Workload Profiles And SLOs

Supported built in workload profiles are `default`, `short`, `medium`, `long`, `decode-heavy`, `repeated-prefix`, and `mixed`.

Attach Mode and Managed Mode accept JSON workload manifests. Workload fingerprints include profile name, token distribution, and SLO constraints.

Supported SLO guards are TTFT, TPOT, p95 latency, minimum token throughput, and maximum failed request rate. A candidate that violates one of these constraints is ineligible for recommendation.

## Validation Baseline

Phase One release evidence is stored under:

* `results/phase1-runtime-evidence-v2`
* `results/phase1-failure-lifecycle`

The validation campaign passed with four usable vLLM and SGLang fresh and repeat runs. Each backend produced one fresh measurement and one identical exact evidence reuse with zero launches and zero workload measurements on the repeat.
