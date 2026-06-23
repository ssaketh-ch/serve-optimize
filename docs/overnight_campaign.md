# Overnight Model Campaign

The overnight runner measures each model with the safe backend baseline and generated candidates. Every recommendation summary reports the selected configuration and its percentage change from the baseline for throughput, p95 latency, average power, joules per token, and tokens per watt.

## Run

Activate each matching backend profile, then run the backend you want from that environment.

```bash
output=results/overnight-campaign
scripts/run_overnight_campaign.sh standard vllm "$output"
scripts/run_overnight_campaign.sh standard sglang "$output"
```

Use `all` only in an environment where both backend commands are installed:

```bash
scripts/run_overnight_campaign.sh standard all
```

Arguments are:

```text
scripts/run_overnight_campaign.sh SUITE BACKENDS OUTPUT_DIR
```

Defaults are `standard`, `all`, and a timestamped directory under `results`. `BACKENDS` can be `vllm`, `sglang`, `all`, or a comma separated list such as `vllm,sglang`.

The default goal matrix is:

* `balanced`
* `throughput`, mapped to the Managed Mode `performance` goal
* `energy_efficient`, mapped to the Managed Mode `efficient` goal

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
INCLUDE_GATED=0 scripts/run_overnight_campaign.sh standard all
```

Set up a read token on this server:

```bash
uv pip install huggingface-hub
hf auth login
hf auth whoami
```

For noninteractive shells, export a token before running the campaign:

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

* four evaluated candidates, including the safe default baseline
* two trials per candidate
* medium workload profile
* eight warmup requests
* five seconds of idle baseline sampling
* sixty seconds of active soak time
* fifteen minute backend startup timeout
* streaming measurements for TTFT and TPOT

Override these without editing the script:

```bash
LIMIT=6 \
TRIALS=3 \
WARMUP_REQUESTS=12 \
IDLE_BASELINE_SECONDS=8 \
SOAK_SECONDS=120 \
STARTUP_TIMEOUT=1200 \
scripts/run_overnight_campaign.sh standard vllm results/my-campaign
```

Use `MODEL_FILE=/path/to/models.tsv` for a custom manifest with the same tab separated columns.

## Interpret Results

A positive `improvement_percent` always means the selected configuration is better for that metric. The calculation increases with throughput and tokens per watt, and decreases with latency, power, and joules per token.

The quickest human view is `overnight_summary.md`. It lists backend, goal, model, selected versus baseline values, percentage changes, and skipped cells.

The comparison remains scoped to evaluated candidates. It does not claim a globally optimal serving configuration.
