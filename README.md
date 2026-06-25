# Serve Optimize

Serve Optimize measures LLM serving behavior and recommends the best configuration among evaluated candidates for a selected goal.

```text
  ____                             ___        _   _           _
 / ___|  ___ _ ____   _____       / _ \ _ __ | |_(_)_ __ ___ (_)_______
 \___ \ / _ \ '__\ \ / / _ \_____| | | | '_ \| __| | '_ ` _ \| |_  / _ \
  ___) |  __/ |   \ V /  __/_____| |_| | |_) | |_| | | | | | | |/ /  __/
 |____/ \___|_|    \_/ \___|      \___/| .__/ \__|_|_| |_| |_|_/___\___|
                                        |_|

Serve Optimize
Energy aware LLM serving optimization
```

Serve Optimize has two operating modes:

* Attach Mode benchmarks an already running OpenAI compatible endpoint.
* Managed Mode generates candidates, validates them, launches supported backends, benchmarks their OpenAI compatible endpoints, collects optional telemetry, stops the launched process group, and writes recommendation artifacts.

Measured runtime evidence is the source of truth. AIConfigurator predictions can propose or prioritize candidates, but they never replace measured results or exact fresh measured evidence.

Recommendations are always scoped to the evaluated set. The correct claim is:

```text
best among evaluated candidates
```

## Quick Start

Start with the core development and Attach Mode environment:

```bash
uv venv --python python3 .venv
uv pip install --python .venv/bin/python -e ".[dev,telemetry]"
.venv/bin/serve-optimize --help
.venv/bin/serve-optimize detect
.venv/bin/serve-optimize doctor
```

Run the quick workflow guide from [QUICKSTART.md](QUICKSTART.md).

## Installation

Managed vLLM and Managed SGLang use separate Python environments because their validated backend stacks pin different runtime packages. Install those environments from [INSTALL.md](INSTALL.md) before running Managed Mode measurements.

The short rule:

* Use `.venv-vllm` for vLLM Managed Mode.
* Use `.venv-sglang` for SGLang Managed Mode.
* Keep SSL verification enabled when installing dependencies.

## Attach Mode

Attach Mode measures an endpoint you already started:

```bash
serve-optimize recommend /path/to/model \
  --base-url http://127.0.0.1:8000/v1
```

For authenticated endpoints, keep the secret out of artifacts and pass only its environment variable name:

```bash
export SERVE_OPTIMIZE_API_KEY="..."
serve-optimize endpoint-bench served-model \
  --base-url https://example.com/v1 \
  --api-key-env SERVE_OPTIMIZE_API_KEY
```

The environment variable name is recorded for reproducibility. Its value is read only when requests are sent and is not written to benchmark artifacts.

Preview Attach Mode without sending benchmark requests:

```bash
serve-optimize recommend /path/to/model \
  --dry-run \
  --base-url http://127.0.0.1:8000/v1
```

Attach Mode can compare candidate metadata and load shapes, but it cannot prove that the running endpoint was started with a proposed serve command.

## Managed Mode

Managed Mode owns process lifecycle through backend adapters.

### vLLM

```bash
source .venv-vllm/bin/activate
serve-optimize doctor --profile vllm

serve-optimize optimize /path/to/model \
  --backend vllm \
  --out results/managed-vllm
```

### SGLang

```bash
source .venv-sglang/bin/activate
serve-optimize doctor --profile sglang

serve-optimize optimize /path/to/model \
  --backend sglang \
  --out results/managed-sglang
```

Add `--dry-run` to write preflight artifacts without launching servers, health checking endpoints, sending benchmark requests, or writing measured evidence.

Managed Mode also accepts workload identity and SLO guards:

```bash
serve-optimize optimize /path/to/model \
  --workload-profile mixed \
  --slo-p95-latency-ms 900 \
  --slo-min-throughput-tokens-per-sec 100 \
  --dry-run
```

Measurement quality controls can be added to endpoint or managed benchmarks:

```bash
serve-optimize optimize /path/to/model \
  --warmup-requests 8 \
  --steady-state-seconds 30 \
  --soak-seconds 120 \
  --stream \
  --idle-baseline-seconds 5
```

Summaries include gross energy and idle subtracted active energy when an idle baseline is available. Streaming runs also report TTFT and stream chunk TPOT when the endpoint emits timed response chunks.

## Evidence And Resume

Managed evidence is stored in SQLite and remains backend separated. Exact reuse requires matching hardware, backend, runtime, capabilities, rendered command, model identity, workload identity, telemetry requirements, and freshness policy.

Inspect recent evidence:

```bash
serve-optimize evidence list \
  --db results/serve_optimize_evidence.sqlite \
  --limit 20
```

Resume an interrupted or partial managed run from completed matching workloads:

```bash
serve-optimize optimize /path/to/model \
  --goal balanced \
  --resume-from results/managed-vllm/managed-run-id \
  --out results/managed-vllm-resumed
```

Resume reuses only prior completed measured workloads whose candidate id, workload id, rendered launch identity, and workload identity still match.

## Campaign Planning

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

Campaign planning writes a managed run matrix, executable per backend scripts, a dispatcher, and a postprocessing script. Run each backend script in the matching isolated environment. Campaign planning itself does not launch servers or create measured evidence.

Validate existing managed run artifacts:

```bash
serve-optimize validate-campaign \
  RUN_DIR_1 RUN_DIR_2 \
  --out results/validation-campaign
```

## Artifacts

Managed runs write inspectable artifacts, including:

* `managed_run.json`
* `runtime_environment.json`
* backend capability metadata
* `rendered_launch_configs.jsonl`
* `launch_specs.jsonl`
* `launch_groups.json`
* `workload_configs.jsonl`
* `server_lifecycle.jsonl`
* `candidate_failures.jsonl`
* `evidence_decisions.jsonl`
* `managed_recommendation.json`
* `managed_pareto_frontier.json`
* `managed_pareto_frontier.csv`
* `managed_report.txt`
* `recommendation_summary.json`
* `recommendation_summary.txt`
* backend stdout and stderr logs for launched servers

`recommendation_summary.txt` is the primary human facing answer. `managed_recommendation.json` is the detailed automation and audit artifact.

## Supported Surface

| Surface | Support |
|---|---|
| Attach Mode | First class for OpenAI compatible endpoints |
| Managed vLLM | First class for the validated vLLM profile |
| Managed SGLang | First class for the detected supported SGLang surface |
| TensorRT LLM | Planned only for Managed Mode |
| TGI, LMDeploy, llama.cpp, NIM | Attach Mode only when they expose an OpenAI compatible API |

See [docs/compatibility.md](docs/compatibility.md) for exact support, evidence, artifact, installation, and exclusion contracts.

## Verification

```bash
python -m compileall -q src tests
pytest -q
ruff check .
python -m json.tool feature_list.json
serve-optimize --help
serve-optimize optimize --help
serve-optimize validate-campaign --help
serve-optimize campaign-plan --help
```

The latest recorded verification notes are maintained in [docs/development/verification.md](docs/development/verification.md).

## Documentation

* [Quick start](QUICKSTART.md)
* [Installation](INSTALL.md)
* [Documentation index](docs/README.md)
* [Architecture](docs/architecture.md)
* [System design](docs/design.md)
* [Compatibility contract](docs/compatibility.md)
* [Support matrix](docs/support_matrix.md)
* [Security notes](docs/security.md)
* [Research package](docs/research_package.md)
* [Release engineering](docs/development/release.md)

## License

Apache 2.0
