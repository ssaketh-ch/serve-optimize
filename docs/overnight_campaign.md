# Overnight Model Campaign

The overnight runner measures each model with the safe backend baseline and generated candidates. Every recommendation summary reports the selected configuration and its percentage change from the baseline for throughput, p95 latency, average power, joules per token, and tokens per watt.

## Before You Run

Install the backend environments first. See [Installation](installation.md) for the full from scratch setup.

The important point is that the overnight runner uses the active shell environment. It does not activate a backend environment internally.

Use this layout on a fresh server:

1. `.venv-vllm` contains Serve Optimize plus vLLM.
2. `.venv-sglang` contains Serve Optimize plus SGLang.
3. Both backend runs write into the same campaign output directory.

Check each environment before a long run:

```bash
source .venv-vllm/bin/activate
serve-optimize doctor --profile vllm
deactivate

source .venv-sglang/bin/activate
serve-optimize doctor --profile sglang
deactivate
```

If vLLM or SGLang was already installed, update the environment from the pinned profile first:

```bash
uv pip install \
  --python .venv-vllm/bin/python \
  --upgrade \
  --torch-backend=auto \
  -r requirements/profiles/vllm.txt

uv pip install \
  --python .venv-sglang/bin/python \
  --upgrade \
  -r requirements/profiles/sglang.txt
```

Then rerun the two doctor commands above.

## Run With Separate Backend Environments

Run vLLM and SGLang separately, using one shared output directory:

```bash
output=results/overnight-campaign

source .venv-vllm/bin/activate
scripts/run_overnight_campaign.sh standard vllm "$output"
deactivate

source .venv-sglang/bin/activate
scripts/run_overnight_campaign.sh standard sglang "$output"
deactivate
```

Use `all` only in an environment where both backend commands are installed and verified:

```bash
scripts/run_overnight_campaign.sh standard all
```

The split environment flow is the safer default because the validated vLLM and SGLang package stacks are intentionally separate.

## Arguments

Arguments are:

```text
scripts/run_overnight_campaign.sh SUITE BACKENDS OUTPUT_DIR
```

Defaults are `standard`, `all`, and a timestamped directory under `results`. `BACKENDS` can be `vllm`, `sglang`, `all`, or a comma separated list such as `vllm,sglang`.

The default goal matrix is:

1. `balanced`
2. `throughput`, mapped to the Managed Mode `performance` goal
3. `energy_efficient`, mapped to the Managed Mode `efficient` goal

Override it with:

```bash
GOALS=balanced,throughput scripts/run_overnight_campaign.sh standard vllm
```

The runner continues after failures and records skipped cells in `failures.tsv` with a reason such as `oom`, `model_access`, `backend_launch`, `startup_timeout`, or `command_failed`. Access and memory failures skip the remaining cells for that model. Backend launch and startup timeout failures skip the remaining goals for that model and backend. It writes one log per matrix cell, all managed run artifacts, a campaign evidence database, `overnight_summary.json`, `overnight_summary.csv`, `overnight_summary.md`, and validation campaign artifacts.

## Suites

The model manifest is `configs/overnight_models.tsv`.

| Suite | Models | Purpose |
|---|---:|---|
| quick | 2 | Validate the workflow with 0.6B and 3.8B dense models. |
| standard | 7 | Cover Qwen, Phi, Mistral, Granite, and DeepSeek from 0.6B through 30B, including hybrid, reasoning distill, and mixture of experts models. |
| extended | 9 | Add Falcon 10B and Qwen 32B for broader family and upper memory coverage. |

All selected default repositories were public and ungated when checked on 2026-06-23. A model host can still change availability, revisions, or remote code after that date.

## Gated Models

Optional gated models live in `configs/overnight_gated_models.tsv`. With `INCLUDE_GATED=auto`, the runner appends that manifest only when `HF_TOKEN` is set or a saved Hugging Face token exists. Force the behavior with:

```bash
INCLUDE_GATED=1 scripts/run_overnight_campaign.sh quick vllm
INCLUDE_GATED=0 scripts/run_overnight_campaign.sh standard vllm
```

Set up a read token once on this server. Do it inside each backend environment if you use saved local authentication:

```bash
source .venv-vllm/bin/activate
hf auth login
hf auth whoami
deactivate

source .venv-sglang/bin/activate
hf auth login
hf auth whoami
deactivate
```

For noninteractive shells, export a token before running each backend campaign command:

```bash
export HF_TOKEN=hf_your_read_token
```

Create tokens at <https://huggingface.co/settings/tokens>. Hugging Face documents `hf auth login` for saved local authentication and `HF_TOKEN` for environment based authentication.

Request gated model access from each model page in a browser while logged in:

* <https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct>
* <https://huggingface.co/google/gemma-3-4b-it>
* <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct>
* <https://huggingface.co/google/gemma-3-12b-it>
* <https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct>

Approval is controlled by the model owner. The runner treats missing approval as a skipped `model_access` cell and continues.

## Measurement Defaults

The script uses:

1. six evaluated candidates, including the safe default baseline
2. two trials per candidate
3. medium workload profile
4. eight warmup requests
5. five seconds of idle baseline sampling
6. sixty seconds of active soak time
7. fifteen minute backend startup timeout
8. fresh measurements by default, so new campaigns build evidence instead of reusing old rows
9. Hugging Face model prefetch before backend launches when the `hf` CLI is available
10. streaming measurements for TTFT and TPOT

Override these without editing the script:

```bash
LIMIT=8 \
TRIALS=3 \
WARMUP_REQUESTS=12 \
IDLE_BASELINE_SECONDS=8 \
SOAK_SECONDS=120 \
STARTUP_TIMEOUT=1200 \
EVIDENCE_FRESHNESS_HOURS=168 \
PREFETCH_MODELS=0 \
scripts/run_overnight_campaign.sh standard vllm results/my-campaign
```

Use `MODEL_FILE=/path/to/models.tsv` for a custom manifest with the same tab separated columns.

## Interpret Results

A positive `improvement_percent` always means the selected configuration is better for that metric. The calculation increases with throughput and tokens per watt, and decreases with latency, power, and joules per token.

The quickest human view is `overnight_summary.md`. It lists the latest measured comparison for each backend, goal, and model, then lists the latest skipped or unavailable cells. If you reuse an output directory for retries, older attempts are kept in the run artifacts but no longer double count in the headline tables.

The comparison remains scoped to evaluated candidates. It does not claim a globally optimal serving configuration.
