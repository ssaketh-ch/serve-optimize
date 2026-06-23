#!/usr/bin/env bash

set -uo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
suite=${1:-standard}
backend=${2:-vllm}
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
output_root=${3:-$repo_root/results/overnight-$timestamp}
model_file=${MODEL_FILE:-$repo_root/configs/overnight_models.tsv}
limit=${LIMIT:-4}
trials=${TRIALS:-2}
warmup_requests=${WARMUP_REQUESTS:-8}
idle_baseline_seconds=${IDLE_BASELINE_SECONDS:-5}
soak_seconds=${SOAK_SECONDS:-60}

case "$suite" in
  quick) max_tier=1 ;;
  standard) max_tier=2 ;;
  extended) max_tier=3 ;;
  *) echo "suite must be quick, standard, or extended" >&2; exit 2 ;;
esac

if [[ "$backend" != "vllm" && "$backend" != "sglang" ]]; then
  echo "backend must be vllm or sglang" >&2
  exit 2
fi
if [[ ! -f "$model_file" ]]; then
  echo "model file not found: $model_file" >&2
  exit 2
fi
if ! command -v serve-optimize >/dev/null 2>&1; then
  echo "serve-optimize is not available in the active environment" >&2
  exit 2
fi

mkdir -p "$output_root/logs" "$output_root/runs"
cp "$model_file" "$output_root/models.tsv"
printf 'suite=%s\nbackend=%s\nlimit=%s\ntrials=%s\nwarmup_requests=%s\nidle_baseline_seconds=%s\nsoak_seconds=%s\n' \
  "$suite" "$backend" "$limit" "$trials" "$warmup_requests" "$idle_baseline_seconds" "$soak_seconds" \
  >"$output_root/campaign.env"

failures=0
while IFS=$'\t' read -r tier model family size_class architecture; do
  [[ -z "$tier" || "$tier" == \#* ]] && continue
  (( tier > max_tier )) && continue
  slug=${model//\//--}
  log_path="$output_root/logs/$slug.log"
  echo "[$(date -u +%FT%TZ)] starting $model ($family, $size_class, $architecture)" | tee -a "$log_path"
  serve-optimize optimize "$model" \
    --backend "$backend" \
    --goal balanced \
    --limit "$limit" \
    --trials "$trials" \
    --workload-profile medium \
    --warmup-requests "$warmup_requests" \
    --idle-baseline-seconds "$idle_baseline_seconds" \
    --soak-seconds "$soak_seconds" \
    --evidence-db "$output_root/evidence.sqlite" \
    --out "$output_root/runs/$slug" \
    2>&1 | tee -a "$log_path"
  status=${PIPESTATUS[0]}
  if (( status != 0 )); then
    failures=$((failures + 1))
    echo "[$(date -u +%FT%TZ)] failed $model with status $status" | tee -a "$log_path"
  else
    echo "[$(date -u +%FT%TZ)] completed $model" | tee -a "$log_path"
  fi
done <"$model_file"

python "$repo_root/scripts/summarize_overnight.py" "$output_root"
mapfile -t run_dirs < <(find "$output_root/runs" -type f -name managed_run.json -printf '%h\n' | sort)
if (( ${#run_dirs[@]} > 0 )); then
  serve-optimize validate-campaign "${run_dirs[@]}" --out "$output_root/validation" || failures=$((failures + 1))
fi

echo "Campaign artifacts: $output_root"
echo "Failed models or validation steps: $failures"
(( failures == 0 ))
