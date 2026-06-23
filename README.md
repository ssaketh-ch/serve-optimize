# Serve Optimize

Serve Optimize measures LLM serving behavior and recommends the best configuration among evaluated candidates for a selected goal.

It supports two product paths:

* Attach Mode benchmarks an already running OpenAI compatible endpoint.
* Managed Mode generates candidates, validates them, launches supported backends, benchmarks their OpenAI compatible endpoints, collects optional telemetry, stops the launched process group, and writes recommendation artifacts.

Measured runtime evidence is the source of truth. AIConfigurator predictions may propose or prioritize candidates, but they never replace measured results or exact fresh measured evidence.

## Current Status

Serve Optimize is a first public release for measured LLM serving configuration optimization.

Verified capabilities include:

* NVIDIA GPU and MIG hardware detection
* synthetic and local functional benchmark paths
* OpenAI compatible endpoint benchmarking
* optional NVML and `nvidia-smi` telemetry
* throughput, latency, power, energy, and efficiency metrics
* Attach Mode recommendations
* first class Managed Mode support for vLLM and SGLang
* capability aware candidate generation and prelaunch validation
* canonical rendered launch configurations
* runtime fingerprinted evidence and exact reuse
* measured recommendation, Pareto, repeatability, and campaign artifacts
* lifecycle diagnostics for availability, launch, health, benchmark, stop, and interruption failures
* release readiness checks and research package artifacts

Validated backend stacks:

| Backend | Validated version | Status |
|---|---|---|
| vLLM | `0.23.0` | First class Managed Mode, validated on the current Blackwell host |
| SGLang | `0.5.13.post1` | First class for the supported detected surface; clean profile resolution verified |
| TensorRT LLM | none | Planned only; Managed Mode is not in current scope |

External TGI, LMDeploy, llama.cpp, NIM, and TensorRT LLM endpoints may be measured through Attach Mode when they expose an OpenAI compatible API. Serve Optimize does not own their lifecycle.

See [Compatibility](docs/compatibility.md) for the exact support, evidence, artifact, installation, and exclusion contracts.

## Install

Core development installation:

```bash
uv venv --python python3 .venv
uv pip install --python .venv/bin/python -e ".[dev,telemetry]"
```

The backend profiles are pinned to the validated split stacks and must be installed in separate environments.

Do not disable SSL verification to install backend dependencies.

Reproducible uv profiles are documented in [Installation](docs/installation.md).
Dependency audit commands and current backend advisory boundaries are documented in [Security Notes](docs/security.md).

## Basic Checks

```bash
serve-optimize detect
serve-optimize doctor
serve-optimize telemetry-check --telemetry auto --duration 5 --out results/telemetry-check
```

`telemetry-check` samples telemetry without running inference. It writes raw samples, a summary, capability metadata, and a report.

## Attach Mode

Attach Mode measures the endpoint that is already running:

```bash
serve-optimize recommend /path/to/model \
  --base-url http://127.0.0.1:8000/v1
```

For an authenticated endpoint, keep the secret out of artifacts and pass only its environment variable name:

```bash
export SERVE_OPTIMIZE_API_KEY="..."
serve-optimize endpoint-bench served-model \
  --base-url https://example.com/v1 \
  --api-key-env SERVE_OPTIMIZE_API_KEY
```

The environment variable name is recorded for reproducibility. Its value is read only when requests are sent and is not written to benchmark artifacts.

Preview the same Attach Mode work without touching the endpoint:

```bash
serve-optimize recommend /path/to/model \
  --dry-run \
  --base-url http://127.0.0.1:8000/v1
```

Attach Mode can compare candidate metadata and load shapes, but it cannot prove that the running endpoint was started with a proposed serve command.

Important artifacts include:

* `recommendation.json`
* `scores.jsonl`
* `pareto_frontier.json`
* `pareto_frontier.csv`
* `summary.json`
* `metadata.json`
* `report.txt`
* per candidate benchmark and telemetry artifacts

## Managed Mode

Managed Mode owns process lifecycle through backend adapters.

### vLLM

```bash
# Activate an environment installed from requirements/profiles/vllm.txt

serve-optimize optimize /path/to/model \
  --out results/managed-vllm
```

Add `--dry-run` to write `preflight.json`, `preflight.txt`, rendered launch configs, workload configs, launch groups, and validation failures without launching servers or writing measured evidence.

Managed Mode also accepts workload identity and SLO guards:

```bash
serve-optimize optimize /path/to/model \
  --workload-profile mixed \
  --slo-p95-latency-ms 900 \
  --slo-min-throughput-tokens-per-sec 100 \
  --dry-run
```

Use `--workload-manifest path/to/workload.json` for a JSON workload manifest. Workload fingerprints include the profile name, token distribution, and SLO constraints.

Measurement quality controls can be added to endpoint or managed benchmarks:

```bash
serve-optimize optimize /path/to/model \
  --warmup-requests 8 \
  --steady-state-seconds 30 \
  --soak-seconds 120 \
  --stream \
  --idle-baseline-seconds 5
```

Summaries include gross energy and idle subtracted active energy when an idle baseline is available. Streaming runs also report TTFT and stream chunk TPOT when the endpoint emits timed response chunks. Soak runs extend the active request window and thermal summaries report observed temperature rise, slope, and whether the window is long enough to support a stability claim.

Managed recommendations also write optimizer quality artifacts. `optimizer_quality.json` reports bounded evaluated candidate baselines, search regret, metric regret, and candidate policy coverage. `optimizer_failure_cache.json` records artifact backed failure cache entries for failed managed candidates. These artifacts remain scoped to evaluated candidates.

`recommendation_summary.json` and `recommendation_summary.txt` compare the selected configuration with the safe default baseline. Positive percentages mean the selected configuration improved that metric. Power and energy comparisons use idle subtracted values when a valid idle baseline is available.

Resume an interrupted or partial managed campaign from completed matching workloads:

```bash
serve-optimize optimize /path/to/model \
  --goal balanced \
  --resume-from results/managed-vllm/managed-run-id \
  --out results/managed-vllm-resumed
```

Resume reuses only prior completed measured workloads whose candidate id, workload id, rendered launch identity, and workload identity still match. Drifted, failed, unavailable, or incomplete workloads are measured normally.

### SGLang

Use the isolated SGLang profile:

```bash
# Activate an environment installed from requirements/profiles/sglang.txt

serve-optimize optimize /path/to/model \
  --backend sglang \
  --out results/managed-sglang
```

SGLang candidates preserve and fingerprint:

```text
--disable-piecewise-cuda-graph
```

The option is emitted only when the installed SGLang command surface reports support.

## Managed Lifecycle

A managed evaluation performs:

1. Hardware, model, backend, and capability detection.
2. Candidate generation and prelaunch validation.
3. Canonical command rendering.
4. Runtime fingerprint and evidence compatibility checks.
5. Exact fresh evidence reuse where allowed.
6. Completed workload resume where `--resume-from` matches exact launch and workload identity.
7. Server launch for remaining workloads.
8. OpenAI compatible health checks and benchmarks.
9. Optional telemetry collection.
10. Evidence writes for measured workloads.
11. Process group shutdown and lifecycle recording.
12. Recommendation and Pareto artifact generation.

Unsupported backend options are rejected or marked unavailable before launch. They are not silently translated.

## Evidence

Managed evidence is stored in SQLite and remains backend separated.

Exact reuse requires a match across:

* hardware identity
* backend name and version
* Torch, CUDA runtime, and Python versions
* compiler toolchain identity
* backend capability metadata
* rendered launch command
* canonical launch configuration
* model identity
* workload identity
* telemetry compatibility
* freshness policy

Only exact fresh measured evidence may skip a measurement. Stale, near compatible, legacy, or runtime drifted evidence may be advisory but cannot be used as exact truth.

Inspect recent evidence:

```bash
serve-optimize evidence list \
  --db results/serve_optimize_evidence.sqlite \
  --limit 20
```

## Managed Artifacts

Every managed run writes inspectable artifacts, including:

* `managed_run.json`
* `runtime_environment.json`
* `vllm_argument_capabilities.json` or `sglang_argument_capabilities.json`
* `rendered_launch_configs.jsonl`
* `launch_specs.jsonl`
* `launch_groups.json`
* `workload_configs.jsonl`
* `server_lifecycle.jsonl`
* `candidate_failures.jsonl`
* `evidence_decisions.jsonl`
* `candidate_synthesis.json`
* `prior_candidates.json`
* `prior_summary.json`
* `evaluation_rungs.json`
* `promotion_decisions.jsonl`
* `managed_recommendation.json`
* `managed_pareto_frontier.json`
* `managed_pareto_frontier.csv`
* `managed_report.txt`
* `recommendation_summary.json`
* `recommendation_summary.txt`
* backend stdout and stderr logs for launched servers

`recommendation_summary.txt` is the primary human facing deployment answer. `managed_recommendation.json` is the detailed automation and audit artifact.

## Recommendation Scope

Recommendations are always scoped to the evaluated set.

The product may state:

```text
best among evaluated candidates
```

It must not claim exhaustive coverage unless a separate exhaustive experiment proves that claim.

## Repeatability And Campaign Validation

Compare existing managed runs:

```bash
serve-optimize repeatability RUN_DIR_1 RUN_DIR_2
```

Validate a collection of existing managed artifacts:

```bash
serve-optimize validate-campaign \
  RUN_DIR_1 RUN_DIR_2 \
  --out results/validation-campaign
```

Campaign validation reads existing artifacts. It does not launch servers or create new measured evidence.

Plan a broader managed validation campaign before collecting evidence:

```bash
serve-optimize campaign-plan \
  --model /path/to/model-a \
  --model /path/to/model-b \
  --backend vllm \
  --backend sglang \
  --workload-profile short \
  --workload-profile mixed \
  --repeats 2 \
  --out results/campaign-plan
```

Campaign plans write a managed run matrix, executable per backend command scripts, a backend dispatcher, and a postprocessing script. Run each backend script in its matching isolated environment. The runners continue after individual failures so later matrix cells still execute. Campaign planning itself does not launch servers or create measured evidence.

For a ready to run multi family model suite with direct default versus optimized comparisons:

```bash
output=results/overnight-campaign
scripts/run_overnight_campaign.sh standard vllm "$output"
scripts/run_overnight_campaign.sh standard sglang "$output"
```

See [Overnight Model Campaign](docs/overnight_campaign.md) for the vLLM and SGLang matrix, gated model setup, model tiers, measurement defaults, output tables, and environment overrides.

## Release And Research Artifacts

Check release readiness:

```bash
serve-optimize release-check --out results/release-check
```

Package existing managed run artifacts for research analysis:

```bash
serve-optimize research-package \
  RUN_DIR_1 RUN_DIR_2 \
  --out results/research-package
```

Research packages write a manifest, methodology, run table, coverage table, and validation campaign summary. They do not launch servers or create new measured evidence.

## Telemetry

Telemetry providers are optional and emit generic capability fields. Missing counters are represented as unavailable, not as zero.

Current measured fields may include:

* power
* temperature
* memory usage
* GPU and memory utilization
* clocks
* power limit
* throttle reasons

Current energy values include gross active window estimates and idle subtracted active energy when an idle baseline is available. Prefill and decode phase attribution is not implemented.

Prefill and decode energy attribution is not claimed because the current endpoint and telemetry path does not observe phase boundary markers. The raw power window is real, but assigning that energy to prefill versus decode would require backend phase events or a request trace with defensible phase timestamps.

## Verification

```bash
python -m compileall -q src tests
pytest -q
ruff check .
python -m json.tool feature_list.json
serve-optimize --help
serve-optimize optimize --help
serve-optimize optimize --verbose-help
serve-optimize validate-campaign --help
serve-optimize release-check --help
serve-optimize research-package --help
```

The latest recorded result is maintained in [Verification](docs/verification.md).

## Current Limitations

* Managed candidate policies are bounded rather than exhaustive.
* Workload profiles are not yet complete production trace manifests.
* Prefill and decode phase attribution is not implemented.
* Thermal stability claims require a sufficiently long active window. Short runs are reported as limited thermal evidence.
* TensorRT LLM is planned only and outside current Managed Mode scope. Kubernetes, power limit control, and parallel candidate execution are not implemented.
* SGLang runtime validation must be repeated on each target GPU environment because its supported command surface and kernels are capability dependent.

## Documentation

* [System design](docs/design.md)
* [Compatibility contract](docs/compatibility.md)
* [Installation profiles](docs/installation.md)
* [Architecture rules](docs/architecture_rules.md)
* [Quickstart](docs/quickstart.md)
* [Verification](docs/verification.md)
* [Release engineering](docs/release.md)
* [Product readiness](docs/product_readiness.md)
* [Contributing](CONTRIBUTING.md)
* [Security policy](SECURITY.md)
* [Support matrix](docs/support_matrix.md)
* [Research package](docs/research_package.md)
* [Overnight model campaign](docs/overnight_campaign.md)
* [Planned experimental methodology](docs/experiments.md)
* [Literature and landscape](docs/literature-and-landscape.md)

## License

Apache 2.0
