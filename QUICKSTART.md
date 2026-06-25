# Quickstart

## Core Setup

```bash
uv venv --python python3 .venv
uv pip install --python .venv/bin/python -e ".[dev,telemetry]"
.venv/bin/serve-optimize --help
.venv/bin/serve-optimize detect
.venv/bin/serve-optimize doctor
```

Use the mutually exclusive backend profiles in `requirements/profiles` for managed vLLM and managed SGLang. See [Installation](installation.md) for the full setup and update flow.

For a full explanation of how Attach Mode, Managed Mode, evidence, pruning, and recommendations fit together, see [Architecture](architecture.md).

Normal help shows the common path:

```bash
serve-optimize optimize --help
```

Use verbose help when you need lifecycle, evidence, SLO, or measurement tuning flags:

```bash
serve-optimize optimize --verbose-help
```

## Telemetry Check

```bash
serve-optimize telemetry-check \
  --telemetry auto \
  --duration 5 \
  --interval 0.2 \
  --out results/telemetry-check
```

This does not run inference.

## Attach Mode

Start an OpenAI compatible server separately, then run:

```bash
serve-optimize recommend /path/to/model \
  --base-url http://127.0.0.1:8000/v1
```

Attach Mode measures the running endpoint. It does not verify its launch command.

Add `--dry-run` first to write a preflight plan without endpoint health checks or benchmark requests.

## Managed vLLM

```bash
source .venv-vllm/bin/activate
serve-optimize doctor --profile vllm

serve-optimize optimize /path/to/model \
  --backend vllm \
  --out results/managed-vllm
```

## Managed SGLang

```bash
source .venv-sglang/bin/activate
serve-optimize doctor --profile sglang

serve-optimize optimize /path/to/model \
  --backend sglang \
  --out results/managed-sglang
```

The validated SGLang path preserves `--disable-piecewise-cuda-graph`.

Add `--dry-run` to either Managed command to write a preflight plan without launching a backend server, health checking, benchmarking, or writing measured evidence.

Add a workload profile and SLO guards when you want recommendation eligibility to reflect a specific workload:

```bash
serve-optimize optimize /path/to/model \
  --workload-profile repeated-prefix \
  --slo-p95-latency-ms 900 \
  --slo-max-failed-request-rate 0.02 \
  --dry-run
```

Built in workload profiles are `default`, `short`, `medium`, `long`, `decode-heavy`, `repeated-prefix`, and `mixed`. JSON manifests can be passed with `--workload-manifest`.

For measurement quality, add warmup, steady state, and idle baseline controls:

```bash
serve-optimize optimize /path/to/model \
  --warmup-requests 8 \
  --steady-state-seconds 30 \
  --soak-seconds 120 \
  --stream \
  --idle-baseline-seconds 5 \
  --dry-run
```

Use `--stream` only when the endpoint supports OpenAI compatible streaming. TTFT and TPOT are then derived from observed stream chunks. Serve Optimize requests response usage for token counts and does not treat chunk counts as token counts. Without streaming chunks or response usage, the affected timing or token metrics remain unavailable rather than estimated.

Authenticated endpoints use `--api-key-env NAME`. Serve Optimize records the environment variable name and reads its value only while sending requests, so the API key is not serialized into run artifacts.

## Evidence Reuse

Repeat the same managed command with the same evidence database. An exact fresh hit can produce:

```text
cold launches: 0
workload measurements: 0
evidence hits: 1
```

Runtime or command drift blocks exact reuse.

Resume a managed run from completed matching workload artifacts:

```bash
serve-optimize optimize /path/to/model \
  --resume-from results/managed-vllm/managed-run-id
```

Resume reuses only completed workloads whose candidate, launch, and workload identities still match. Drifted or incomplete workloads are measured normally.

## Inspect Results

Start with:

* `recommendation_summary.txt`
* `managed_run.json`
* `rendered_launch_configs.jsonl`
* `runtime_environment.json`
* `evidence_decisions.jsonl`
* `server_lifecycle.jsonl`
* `candidate_failures.jsonl`

Recommendations mean best among evaluated candidates.

## Campaign Planning

Write a broader managed campaign plan without launching servers:

```bash
serve-optimize campaign-plan \
  --model /path/to/model \
  --backend vllm \
  --backend sglang \
  --workload-profile short \
  --workload-profile mixed \
  --out results/campaign-plan
```

The plan contains `campaign_plan.json`, `campaign_matrix.csv`, `campaign_commands.sh`, one executable script per backend, `campaign_postprocess.sh`, and a readable summary. Run `campaign_commands.sh vllm` in the vLLM environment and `campaign_commands.sh sglang` in the SGLang environment. Each backend runner continues through failed matrix cells. After all backend scripts finish, run `campaign_postprocess.sh` to analyze the timestamped managed run directories.

For the ready made overnight model suite with baseline comparisons:

```bash
output=results/overnight-campaign

source .venv-vllm/bin/activate
scripts/run_overnight_campaign.sh standard vllm "$output"
deactivate

source .venv-sglang/bin/activate
scripts/run_overnight_campaign.sh standard sglang "$output"
deactivate
```

The overnight runner uses the active shell environment. See [Overnight Model Campaign](overnight_campaign.md) before using gated models or running both backends into one report.

## Verify The Repository

```bash
python -m compileall -q src tests
pytest -q
ruff check .
python -m json.tool feature_list.json
```

See [Compatibility](compatibility.md) before treating an optional backend or metric as supported.
