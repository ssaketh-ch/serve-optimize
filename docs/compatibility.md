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
| vLLM | First class | vLLM `0.10.0`, Torch `2.7.1+cu126`, CUDA `12.6`, Python `3.10.20` | Installed capability detection, canonical rendering, lifecycle, evidence, and recommendation paths are validated. |
| SGLang | First class for the detected supported surface | SGLang `0.5.10.post1`, Torch `2.9.1+cu128`, CUDA `12.8`, Python `3.10.20`, GCC Toolset `12.2.1` | Requires `source scripts/env_base_runtime.sh` on the validation host. The validated command preserves `--disable-piecewise-cuda-graph`. |
| TensorRT LLM | Planned | none | No adapter, lifecycle, candidate, evidence, or recommendation support exists. |
| TGI, LMDeploy, llama.cpp, NIM | Planned | none | They may be used manually through Attach Mode only when they expose a compatible endpoint. They are not Managed Mode backends. |

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
* measured throughput, latency, power, energy, and efficiency fields when available

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
* exact fresh measured evidence hits

Predictions, stale evidence, and near compatible evidence may influence pruning or promotion but cannot become final measured truth.

Recommendation wording is limited to the evaluated set. Use `best among evaluated candidates`. Do not claim all possible configuration coverage.

## Release And Research Contract

`serve-optimize release-check` writes release readiness artifacts and checks required files, package metadata, verification scripts, schema markers, and support documents.

`serve-optimize research-package` writes research manifests, methodology, run tables, coverage tables, and validation campaign summaries from existing managed run artifacts. It must not launch servers or create measured evidence.

## Telemetry Contract

Telemetry is optional. Providers emit generic fields and capability metadata.

Missing fields mean unavailable unless a provider explicitly measured zero.

Telemetry failure does not fail a benchmark by default. It may lower confidence or prevent exact reuse for a power aware goal.

Energy metrics include gross active window estimates. When an idle baseline is supplied or sampled, summaries also include idle subtracted active energy. Prefill and decode attribution is planned.

## Installation Contract

Current core development install:

```bash
pip install -e ".[dev]"
```

Validated backend measurements use isolated environments installed from:

* `requirements/profiles/vllm.txt`
* `requirements/profiles/sglang.txt`

Backend extras and requirement profiles are pinned to the validated vLLM and SGLang stacks. The backend profiles are mutually exclusive because their Torch and Transformers requirements conflict.

See `docs/installation.md` for clean pip installation and profile verification commands.

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
