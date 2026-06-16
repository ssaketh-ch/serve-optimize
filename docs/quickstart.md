# Quickstart

## Core Setup

```bash
pip install -e ".[dev]"
serve-optimize --help
serve-optimize detect
serve-optimize doctor
```

Use the mutually exclusive pip profiles in `requirements/profiles` for new environments. See [Installation](installation.md).

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
serve-optimize recommend \
  --base-url http://127.0.0.1:8080/v1 \
  --model /path/to/model \
  --backend vllm \
  --system local_gpu \
  --total-gpus 1 \
  --isl 512 \
  --osl 128 \
  --goal balanced
```

Attach Mode measures the running endpoint. It does not verify its launch command.

Add `--dry-run` first to write a preflight plan without endpoint health checks or benchmark requests.

## Managed vLLM

```bash
# Activate an environment installed from requirements/profiles/vllm.txt

serve-optimize managed-evaluate \
  --backend vllm \
  --model /path/to/model \
  --goal balanced \
  --limit 1 \
  --trials 1 \
  --telemetry auto \
  --evidence-db results/evidence.sqlite \
  --out results/managed-vllm
```

## Managed SGLang

```bash
# Activate an environment installed from requirements/profiles/sglang.txt
source scripts/env_base_runtime.sh

serve-optimize managed-evaluate \
  --backend sglang \
  --model /path/to/model \
  --goal balanced \
  --limit 1 \
  --trials 1 \
  --telemetry auto \
  --evidence-db results/evidence.sqlite \
  --out results/managed-sglang
```

The validated SGLang path preserves `--disable-piecewise-cuda-graph`.

Add `--dry-run` to either Managed command to write a preflight plan without launching a backend server, health checking, benchmarking, or writing measured evidence.

Add a workload profile and SLO guards when you want recommendation eligibility to reflect a specific workload:

```bash
serve-optimize managed-evaluate \
  --backend vllm \
  --model /path/to/model \
  --workload-profile repeated-prefix \
  --slo-p95-latency-ms 900 \
  --slo-max-failed-request-rate 0.02 \
  --dry-run
```

Built in workload profiles are `default`, `short`, `medium`, `long`, `decode-heavy`, `repeated-prefix`, and `mixed`. JSON manifests can be passed with `--workload-manifest`.

For measurement quality, add warmup, steady state, and idle baseline controls:

```bash
serve-optimize managed-evaluate \
  --backend vllm \
  --model /path/to/model \
  --warmup-requests 8 \
  --steady-state-seconds 30 \
  --idle-baseline-seconds 5 \
  --dry-run
```

## Evidence Reuse

Repeat the same managed command with the same evidence database. An exact fresh hit can produce:

```text
cold launches: 0
workload measurements: 0
evidence hits: 1
```

Runtime or command drift blocks exact reuse.

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

## Verify The Repository

```bash
python -m compileall -q src tests
pytest -q
ruff check .
python -m json.tool feature_list.json
```

See [Compatibility](compatibility.md) before treating an optional backend or metric as supported.
