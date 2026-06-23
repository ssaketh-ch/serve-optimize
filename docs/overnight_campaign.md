# Overnight Model Campaign

The overnight runner measures each model with the safe backend baseline and generated candidates. Every recommendation summary reports the selected configuration and its percentage change from the baseline for throughput, p95 latency, average power, joules per token, and tokens per watt.

## Run

Activate the matching backend profile, then run:

```bash
scripts/run_overnight_campaign.sh standard vllm
```

Arguments are:

```text
scripts/run_overnight_campaign.sh SUITE BACKEND OUTPUT_DIR
```

Defaults are `standard`, `vllm`, and a timestamped directory under `results`.

The runner continues after individual model failures. It writes one log per model, all managed run artifacts, a campaign evidence database, `overnight_summary.json`, `overnight_summary.csv`, and validation campaign artifacts.

## Suites

The model manifest is `configs/overnight_models.tsv`.

| Suite | Models | Purpose |
|---|---:|---|
| quick | 2 | Validate the workflow with 0.6B and 3.8B dense models. |
| standard | 7 | Cover Qwen, Phi, Mistral, Granite, and DeepSeek from 0.6B through 30B, including hybrid, reasoning distill, and mixture of experts models. |
| extended | 9 | Add Falcon 10B and Qwen 32B for broader family and upper memory coverage. |

All selected repositories were public and ungated when checked on 2026-06-23. A model host can still change availability, revisions, or remote code after that date.

## Measurement Defaults

The script uses:

* four evaluated candidates, including the safe default baseline
* two trials per candidate
* medium workload profile
* eight warmup requests
* five seconds of idle baseline sampling
* sixty seconds of active soak time
* streaming measurements for TTFT and TPOT

Override these without editing the script:

```bash
LIMIT=6 \
TRIALS=3 \
WARMUP_REQUESTS=12 \
IDLE_BASELINE_SECONDS=8 \
SOAK_SECONDS=120 \
scripts/run_overnight_campaign.sh standard vllm results/my-campaign
```

Use `MODEL_FILE=/path/to/models.tsv` for a custom manifest with the same tab separated columns.

## Interpret Results

A positive `improvement_percent` always means the selected configuration is better for that metric. The calculation increases with throughput and tokens per watt, and decreases with latency, power, and joules per token.

The comparison remains scoped to evaluated candidates. It does not claim a globally optimal serving configuration.
