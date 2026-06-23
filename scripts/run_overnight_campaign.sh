#!/usr/bin/env bash

set -uo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
suite=${1:-standard}
backend_arg=${2:-all}
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
output_root=${3:-$repo_root/results/overnight-$timestamp}
model_file=${MODEL_FILE:-$repo_root/configs/overnight_models.tsv}
gated_model_file=${GATED_MODEL_FILE:-$repo_root/configs/overnight_gated_models.tsv}
include_gated=${INCLUDE_GATED:-auto}
goals_csv=${GOALS:-balanced,throughput,energy_efficient}
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

if [[ ! -f "$model_file" ]]; then
  echo "model file not found: $model_file" >&2
  exit 2
fi
if ! command -v serve-optimize >/dev/null 2>&1; then
  echo "serve-optimize is not available in the active environment" >&2
  exit 2
fi

split_csv() {
  local csv=$1
  local -n values_ref=$2
  IFS=',' read -r -a values_ref <<< "$csv"
}

normalize_backends() {
  local value=$1
  if [[ "$value" == "all" ]]; then
    backends=(vllm sglang)
  else
    split_csv "$value" backends
  fi
  for backend in "${backends[@]}"; do
    if [[ "$backend" != "vllm" && "$backend" != "sglang" ]]; then
      echo "backend must be vllm, sglang, or all" >&2
      exit 2
    fi
  done
}

goal_value() {
  case "$1" in
    throughput) echo "performance" ;;
    energy_efficient|energy-efficient) echo "efficient" ;;
    balanced|performance|efficient) echo "$1" ;;
    *) echo "goal must be balanced, throughput, energy_efficient, performance, or efficient" >&2; exit 2 ;;
  esac
}

goal_label() {
  case "$1" in
    performance) echo "throughput" ;;
    efficient) echo "energy_efficient" ;;
    *) echo "$1" ;;
  esac
}

token_path() {
  if [[ -n "${HF_TOKEN_PATH:-}" ]]; then
    echo "$HF_TOKEN_PATH"
  else
    echo "${HF_HOME:-$HOME/.cache/huggingface}/token"
  fi
}

has_hf_token() {
  [[ -n "${HF_TOKEN:-}" || -f "$(token_path)" ]]
}

should_include_gated() {
  case "$include_gated" in
    1|true|yes) return 0 ;;
    0|false|no) return 1 ;;
    auto) has_hf_token ;;
    *) echo "INCLUDE_GATED must be auto, 1, or 0" >&2; exit 2 ;;
  esac
}

classify_failure() {
  local status=$1
  local log_path=$2
  if (( status == 137 || status == 143 )); then
    echo "process_killed_possible_oom"
    return
  fi
  if grep -Eiq 'out of memory|cuda out of memory|oom|cannot allocate memory|killed' "$log_path"; then
    echo "oom"
    return
  fi
  if grep -Eiq 'gated repo|restricted|not in the authorized list|401 client error|403 client error|authentication' "$log_path"; then
    echo "model_access"
    return
  fi
  echo "command_failed"
}

normalize_backends "$backend_arg"
split_csv "$goals_csv" requested_goals
goals=()
for goal in "${requested_goals[@]}"; do
  goals+=("$(goal_value "$goal")")
done

mkdir -p "$output_root/logs" "$output_root/runs"
combined_model_file="$output_root/models.tsv"
cp "$model_file" "$combined_model_file"
if should_include_gated; then
  if [[ -f "$gated_model_file" ]]; then
    {
      echo ""
      grep -Ev '^# tier[[:space:]]' "$gated_model_file"
    } >> "$combined_model_file"
  else
    echo "gated model file not found: $gated_model_file" >&2
  fi
fi
printf 'suite=%s\nbackends=%s\ngoals=%s\nlimit=%s\ntrials=%s\nwarmup_requests=%s\nidle_baseline_seconds=%s\nsoak_seconds=%s\ninclude_gated=%s\n' \
  "$suite" "${backends[*]}" "${goals[*]}" "$limit" "$trials" "$warmup_requests" "$idle_baseline_seconds" "$soak_seconds" "$include_gated" \
  >"$output_root/campaign.env"
failures_path="$output_root/failures.tsv"
if [[ ! -f "$failures_path" ]]; then
  printf 'timestamp\tbackend\tgoal\tmodel\tstatus\treason\tlog_path\n' >"$failures_path"
fi

failures=0
while IFS=$'\t' read -r tier model family size_class architecture; do
  [[ -z "$tier" || "$tier" == \#* ]] && continue
  (( tier > max_tier )) && continue
  slug=${model//\//--}
  for backend in "${backends[@]}"; do
    for goal in "${goals[@]}"; do
      label=$(goal_label "$goal")
      log_dir="$output_root/logs/$backend/$label"
      run_dir="$output_root/runs/$backend/$label/$slug"
      mkdir -p "$log_dir"
      log_path="$log_dir/$slug.log"
      echo "[$(date -u +%FT%TZ)] starting $model ($family, $size_class, $architecture) backend=$backend goal=$label" | tee -a "$log_path"
      serve-optimize optimize "$model" \
        --backend "$backend" \
        --goal "$goal" \
        --limit "$limit" \
        --trials "$trials" \
        --workload-profile medium \
        --warmup-requests "$warmup_requests" \
        --idle-baseline-seconds "$idle_baseline_seconds" \
        --soak-seconds "$soak_seconds" \
        --evidence-db "$output_root/evidence.sqlite" \
        --out "$run_dir" \
        2>&1 | tee -a "$log_path"
      status=${PIPESTATUS[0]}
      if (( status != 0 )); then
        failures=$((failures + 1))
        reason=$(classify_failure "$status" "$log_path")
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$backend" "$label" "$model" "$status" "$reason" "$log_path" >>"$failures_path"
        echo "[$(date -u +%FT%TZ)] skipped $model backend=$backend goal=$label status=$status reason=$reason" | tee -a "$log_path"
      else
        echo "[$(date -u +%FT%TZ)] completed $model backend=$backend goal=$label" | tee -a "$log_path"
      fi
    done
  done
done <"$combined_model_file"

python "$repo_root/scripts/summarize_overnight.py" "$output_root"
mapfile -t run_dirs < <(find "$output_root/runs" -type f -name managed_run.json -printf '%h\n' | sort)
if (( ${#run_dirs[@]} > 0 )); then
  serve-optimize validate-campaign "${run_dirs[@]}" --out "$output_root/validation" || failures=$((failures + 1))
fi

echo "Campaign artifacts: $output_root"
echo "Failed model, backend, goal cells or validation steps: $failures"
(( failures == 0 ))
