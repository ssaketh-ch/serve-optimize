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
startup_timeout=${STARTUP_TIMEOUT:-900}

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

failure_matches() {
  local pattern=$1
  local log_path=$2
  local run_dir=$3
  if grep -Eiq "$pattern" "$log_path"; then
    return 0
  fi
  if [[ -d "$run_dir" ]] && grep -R -Eiq --include='candidate_failures.jsonl' "$pattern" "$run_dir"; then
    return 0
  fi
  return 1
}

classify_failure() {
  local status=$1
  local log_path=$2
  local run_dir=$3
  if (( status == 137 || status == 143 )); then
    echo "process_killed_possible_oom"
    return
  fi
  if failure_matches 'out of memory|cuda out of memory|oom|cannot allocate memory|killed' "$log_path" "$run_dir"; then
    echo "oom"
    return
  fi
  if failure_matches 'gated repo|restricted|not in the authorized list|401 client error|403 client error|authentication' "$log_path" "$run_dir"; then
    echo "model_access"
    return
  fi
  if failure_matches 'SGLANG_GRPC_PORT|server process exited with return code' "$log_path" "$run_dir"; then
    echo "backend_launch"
    return
  fi
  if failure_matches 'Connection refused|health check timed out|No candidates were available for scoring' "$log_path" "$run_dir"; then
    echo "startup_timeout"
    return
  fi
  echo "command_failed"
}

skip_entire_model_reason() {
  case "$1" in
    oom|model_access|process_killed_possible_oom) return 0 ;;
    *) return 1 ;;
  esac
}

skip_backend_reason() {
  case "$1" in
    backend_launch|startup_timeout) return 0 ;;
    *) return 1 ;;
  esac
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
printf 'suite=%s\nbackends=%s\ngoals=%s\nlimit=%s\ntrials=%s\nwarmup_requests=%s\nidle_baseline_seconds=%s\nsoak_seconds=%s\nstartup_timeout=%s\ninclude_gated=%s\n' \
  "$suite" "${backends[*]}" "${goals[*]}" "$limit" "$trials" "$warmup_requests" "$idle_baseline_seconds" "$soak_seconds" "$startup_timeout" "$include_gated" \
  >"$output_root/campaign.env"
failures_path="$output_root/failures.tsv"
if [[ ! -f "$failures_path" ]]; then
  printf 'timestamp\tbackend\tgoal\tmodel\tstatus\treason\tlog_path\n' >"$failures_path"
fi

failures=0
declare -A skipped_models=()
declare -A skipped_model_backends=()
while IFS=$'\t' read -r tier model family size_class architecture; do
  [[ -z "$tier" || "$tier" == \#* ]] && continue
  (( tier > max_tier )) && continue
  slug=${model//\//--}
  for backend in "${backends[@]}"; do
    if [[ -n "${skipped_models[$model]:-}" ]]; then
      echo "[$(date -u +%FT%TZ)] skipping $model for remaining cells after ${skipped_models[$model]}"
      continue
    fi
    backend_key="$model|$backend"
    if [[ -n "${skipped_model_backends[$backend_key]:-}" ]]; then
      echo "[$(date -u +%FT%TZ)] skipping $model backend=$backend after ${skipped_model_backends[$backend_key]}"
      continue
    fi
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
        --startup-timeout "$startup_timeout" \
        --evidence-db "$output_root/evidence.sqlite" \
        --out "$run_dir" \
        2>&1 | tee -a "$log_path"
      status=${PIPESTATUS[0]}
      if (( status != 0 )); then
        failures=$((failures + 1))
        reason=$(classify_failure "$status" "$log_path" "$run_dir")
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$backend" "$label" "$model" "$status" "$reason" "$log_path" >>"$failures_path"
        echo "[$(date -u +%FT%TZ)] skipped $model backend=$backend goal=$label status=$status reason=$reason" | tee -a "$log_path"
        if skip_entire_model_reason "$reason"; then
          skipped_models[$model]=$reason
          break 2
        fi
        if skip_backend_reason "$reason"; then
          skipped_model_backends[$backend_key]=$reason
          break
        fi
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
