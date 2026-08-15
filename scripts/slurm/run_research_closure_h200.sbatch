#!/usr/bin/env bash
#SBATCH --job-name=serveopt-research
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_h200_nvl_3g.71gb:1
#SBATCH --time=36:00:00
#SBATCH --output=serveopt-research-%j.out
#SBATCH --error=serveopt-research-%j.err

set -uo pipefail

CLOSURE_ROOT="${SERVE_OPTIMIZE_CLOSURE_ROOT:?SERVE_OPTIMIZE_CLOSURE_ROOT is required}"
SOURCE_TARBALL="${SERVE_OPTIMIZE_SOURCE_TARBALL:?SERVE_OPTIMIZE_SOURCE_TARBALL is required}"
SOURCE_SHA256="${SERVE_OPTIMIZE_SOURCE_SHA256:?SERVE_OPTIMIZE_SOURCE_SHA256 is required}"
VLLM_VERSION="0.27.1"
SGLANG_VERSION="0.5.17"
BACKEND_PYTHON_VERSION="3.12.13"
VLLM_ENV="${SERVE_OPTIMIZE_VLLM_ENV:-/home/saketh-msc/serve-optimize-envs/vllm-${VLLM_VERSION}-py312}"
SGLANG_ENV="${SERVE_OPTIMIZE_SGLANG_ENV:-/home/saketh-msc/serve-optimize-envs/sglang-${SGLANG_VERSION}-py312}"
GUIDELLM_VERSION="0.7.3"
GUIDELLM_PYTHON_VERSION="3.10.12"
GUIDELLM_ENV="${SERVE_OPTIMIZE_GUIDELLM_ENV:-/home/saketh-msc/serve-optimize-envs/guidellm-${GUIDELLM_VERSION}-paper}"
HISTORICAL_ROOT="${SERVE_OPTIMIZE_HISTORICAL_ROOT:-/home/saketh-msc/serve-optimize/results/benchmark-matrix/closure-20260809T041109}"
MODEL_SCOPE="${SERVE_OPTIMIZE_MODEL_SCOPE:-all}"
BACKEND_SCOPE="${SERVE_OPTIMIZE_BACKEND_SCOPE:-all}"
TELEMETRY_SCOPE="${SERVE_OPTIMIZE_TELEMETRY_SCOPE:-physical_gpu_board_for_mig_allocation}"
ENERGY_LIMIT="${SERVE_OPTIMIZE_ENERGY_LIMIT:-Power and energy are board scoped because the allocation is a MIG instance.}"
DRIVER_LIB_ROOT="${SERVE_OPTIMIZE_DRIVER_LIB_ROOT:-}"
UV_BIN="${SERVE_OPTIMIZE_UV:-/home/saketh-msc/.local/bin/uv}"
JOB_ID="${SLURM_JOB_ID:-manual}"
STATUS="${CLOSURE_ROOT}/status-${JOB_ID}.env"
MARKERS="${CLOSURE_ROOT}/markers"
LOGS="${CLOSURE_ROOT}/logs"
SOURCE_ROOT="${CLOSURE_ROOT}/source/worktree"
POSTPROCESS="${CLOSURE_ROOT}/postprocess"
HF_HOME="${HF_HOME:-/home/saketh-msc/quantization/models}"
TINY_MODEL="Qwen/Qwen3-0.6B"
TINY_REVISION="c1899de289a04d12100db370d81485cdf75e47ca"
SEVEN_B_MODEL="Qwen/Qwen2.5-7B-Instruct"
SEVEN_B_REVISION="a09a35458c702b33eeacc393d103063234e8bc28"
# Wide scope is the publication coverage matrix. Keep revisions immutable so
# every host records the same model bytes.
WIDE_MODEL_RECORDS=(
    "qwen3-0.6b|Qwen/Qwen3-0.6B|c1899de289a04d12100db370d81485cdf75e47ca|128|64"
    "qwen25-1.5b|Qwen/Qwen2.5-1.5B-Instruct|989aa7980e4cf806f80c7fef2b1adb7bc71aa306|1024|256"
    "deepseek-r1-qwen-1.5b|deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B|ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562|1024|256"
    "granite-3.3-2b|ibm-granite/granite-3.3-2b-instruct|707f574c62054322f6b5b04b6d075f0a8f05e0f0|1024|256"
    "qwen25-7b|Qwen/Qwen2.5-7B-Instruct|a09a35458c702b33eeacc393d103063234e8bc28|1024|256"
    "mistral-7b|mistralai/Mistral-7B-Instruct-v0.3|c170c708c41dac9275d15a8fff4eca08d52bab71|1024|256"
    "qwen3-8b|Qwen/Qwen3-8B|b968826d9c46dd6066d109eabc6255188de91218|1024|256"
    "granite-3.3-8b|ibm-granite/granite-3.3-8b-instruct|51dd4bc2ade4059a6bd87649d68aa11e4fb2529b|1024|256"
    "qwen35-9b|Qwen/Qwen3.5-9B|c202236235762e1c871ad0ccb60c8ee5ba337b9a|1024|256"
    "qwen36-27b|Qwen/Qwen3.6-27B|6a9e13bd6fc8f0983b9b99948120bc37f49c13e9|1024|256"
    "qwen38-27b|Qwen/Qwen3.8-27B|1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0|1024|256"
    "qwen38-27b-fp8|Qwen/Qwen3.8-27B-FP8|017b9c7af6b5689d5dd426a76e0bc077eb5ca20a|1024|256"
    "llama31-8b|meta-llama/Llama-3.1-8B-Instruct|0e9e39f249a16976918f6564b8830bc894c89659|1024|256"
)
WARMUP_REQUESTS=16
STEADY_STATE_SECONDS=60
IDLE_BASELINE_SECONDS=30
GUIDELLM_DURATION_SECONDS=60
GUIDELLM_WARMUP_SECONDS=90
GUIDELLM_TOTAL_SECONDS=$((GUIDELLM_WARMUP_SECONDS + GUIDELLM_DURATION_SECONDS))

case "${MODEL_SCOPE}" in
    all)
        MANAGED_RUNS_PER_BACKEND=12
        GUIDELLM_REPORTS_PER_BACKEND=3
        ;;
    tiny)
        MANAGED_RUNS_PER_BACKEND=6
        GUIDELLM_REPORTS_PER_BACKEND=1
        ;;
    seven_b)
        MANAGED_RUNS_PER_BACKEND=6
        GUIDELLM_REPORTS_PER_BACKEND=2
        ;;
    wide)
        MANAGED_RUNS_PER_BACKEND=$((${#WIDE_MODEL_RECORDS[@]} * 10))
        GUIDELLM_REPORTS_PER_BACKEND=$((${#WIDE_MODEL_RECORDS[@]} * 3))
        ;;
    *)
        printf 'Unsupported model scope: %s\n' "${MODEL_SCOPE}" >&2
        exit 2
        ;;
esac

case "${BACKEND_SCOPE}" in
    all)
        BACKENDS=(vllm sglang)
        ;;
    vllm|sglang)
        BACKENDS=("${BACKEND_SCOPE}")
        ;;
    *)
        printf 'Unsupported backend scope: %s\n' "${BACKEND_SCOPE}" >&2
        exit 2
        ;;
esac
EXPECTED_MANAGED_RUNS=$((MANAGED_RUNS_PER_BACKEND * ${#BACKENDS[@]}))
EXPECTED_GUIDELLM_REPORTS=$((GUIDELLM_REPORTS_PER_BACKEND * ${#BACKENDS[@]}))

backend_enabled() {
    local backend="$1"
    [[ " ${BACKENDS[*]} " == *" ${backend} "* ]]
}

mkdir -p "${CLOSURE_ROOT}" "${MARKERS}" "${LOGS}" "${POSTPROCESS}" \
    "${CLOSURE_ROOT}/environment" "${CLOSURE_ROOT}/telemetry"

mark() {
    printf '%s=%s\n' "$1" "$2" >>"${STATUS}"
}

finish_step() {
    local step="$1"
    local status="$2"
    mark "${step}_finished_at" "$(date --iso-8601=seconds)"
    mark "${step}_exit_code" "${status}"
    if (( status == 0 )); then
        touch "${MARKERS}/${step}.ok"
        rm -f "${MARKERS}/${step}.failed"
    else
        rm -f "${MARKERS}/${step}.ok"
        printf '%s\n' "${status}" >"${MARKERS}/${step}.failed"
    fi
}

command_file() {
    local path="$1"
    shift
    printf '%q ' "$@" >"${path}"
    printf '\n' >>"${path}"
}

start_telemetry() {
    local path="$1"
    (
        printf 'timestamp,index,utilization_gpu_percent,memory_used_mib,power_draw_watts\n'
        while true; do
            nvidia-smi \
                --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw \
                --format=csv,noheader,nounits || true
            sleep 1
        done
    ) >"${path}" 2>&1 &
    TELEMETRY_PID=$!
}

stop_telemetry() {
    if [[ -n "${TELEMETRY_PID:-}" ]]; then
        kill "${TELEMETRY_PID}" 2>/dev/null || true
        wait "${TELEMETRY_PID}" 2>/dev/null || true
        TELEMETRY_PID=""
    fi
}

stop_server() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill -- "-${SERVER_PID}" 2>/dev/null || kill "${SERVER_PID}" 2>/dev/null || true
        for _ in $(seq 1 30); do
            kill -0 "${SERVER_PID}" 2>/dev/null || break
            sleep 1
        done
        kill -KILL -- "-${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
        SERVER_PID=""
    fi
}

cleanup() {
    stop_telemetry
    stop_server
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

actual_source_sha="$(sha256sum "${SOURCE_TARBALL}" | awk '{print $1}')"
if [[ "${actual_source_sha}" != "${SOURCE_SHA256}" ]]; then
    printf 'Source snapshot checksum mismatch: expected %s, got %s\n' "${SOURCE_SHA256}" "${actual_source_sha}" >&2
    exit 2
fi
if [[ ! -f "${MARKERS}/source-extracted.ok" ]]; then
    if [[ -e "${SOURCE_ROOT}" ]]; then
        printf 'Incomplete source extraction already exists: %s\n' "${SOURCE_ROOT}" >&2
        exit 2
    fi
    mkdir -p "${SOURCE_ROOT}"
    tar -xzf "${SOURCE_TARBALL}" -C "${SOURCE_ROOT}"
    touch "${MARKERS}/source-extracted.ok"
fi
cd "${SOURCE_ROOT}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.2}"
export CUDA_PATH="${CUDA_HOME}"
export PATH="${SGLANG_ENV}/bin:${PATH}"
export PYTHONPATH="${SOURCE_ROOT}/src"
export PYTHONUNBUFFERED=1
export HF_HOME
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_XET_HIGH_PERFORMANCE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export TOKENIZERS_PARALLELISM=false
unset HF_HUB_ENABLE_HF_TRANSFER
SLURM_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-unset}"
export CUDA_VISIBLE_DEVICES=0

printf 'job_id=%s\n' "${JOB_ID}" >"${STATUS}"
mark started_at "$(date --iso-8601=seconds)"
mark closure_root "${CLOSURE_ROOT}"
mark source_sha256 "${SOURCE_SHA256}"
mark historical_root "${HISTORICAL_ROOT}"
mark model_scope "${MODEL_SCOPE}"
mark backend_scope "${BACKEND_SCOPE}"
mark telemetry_scope "${TELEMETRY_SCOPE}"

{
    echo "started_at=$(date --iso-8601=seconds)"
    echo "hostname=$(hostname)"
    echo "slurm_job_id=${JOB_ID}"
    echo "slurm_cuda_visible_devices=${SLURM_CUDA_VISIBLE_DEVICES}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
    echo "source_sha256=${SOURCE_SHA256}"
    echo "cuda_home=${CUDA_HOME}"
    "${CUDA_HOME}/bin/nvcc" --version
    sha256sum "${CUDA_HOME}/bin/nvcc"
    echo "driver_lib_root=${DRIVER_LIB_ROOT:-system_default}"
    echo "ld_library_path=${LD_LIBRARY_PATH:-unset}"
    "${UV_BIN}" --version
    echo "hf_home=${HF_HOME}"
    echo "model_scope=${MODEL_SCOPE}"
    echo "backend_scope=${BACKEND_SCOPE}"
    echo "malloc_arena_max=${MALLOC_ARENA_MAX}"
    echo "omp_num_threads=${OMP_NUM_THREADS}"
    echo "tokenizers_parallelism=${TOKENIZERS_PARALLELISM}"
    echo "warmup_requests=${WARMUP_REQUESTS}"
    echo "steady_state_seconds=${STEADY_STATE_SECONDS}"
    echo "idle_baseline_seconds=${IDLE_BASELINE_SECONDS}"
    echo "guidellm_duration_seconds=${GUIDELLM_DURATION_SECONDS}"
    echo "guidellm_warmup_seconds=${GUIDELLM_WARMUP_SECONDS}"
    echo "guidellm_total_seconds=${GUIDELLM_TOTAL_SECONDS}"
    echo "gpu_telemetry_scope=${TELEMETRY_SCOPE}"
    echo "energy_limit=${ENERGY_LIMIT}"
    nvidia-smi
    [[ -n "${SLURM_JOB_ID:-}" ]] && scontrol show job "${JOB_ID}"
    "${VLLM_ENV}/bin/python" --version
    "${VLLM_ENV}/bin/python" -c 'import torch, vllm; print(f"vllm={vllm.__version__} torch={torch.__version__} cuda={torch.version.cuda}")'
    backend_enabled sglang && "${SGLANG_ENV}/bin/python" --version
    backend_enabled sglang && "${SGLANG_ENV}/bin/python" -c 'import sglang, torch; print(f"sglang={sglang.__version__} torch={torch.__version__} cuda={torch.version.cuda}")'
    "${GUIDELLM_ENV}/bin/python" --version
    "${GUIDELLM_ENV}/bin/guidellm" --version
} >"${CLOSURE_ROOT}/environment/platform.txt" 2>&1
"${UV_BIN}" pip freeze --python "${VLLM_ENV}/bin/python" >"${CLOSURE_ROOT}/environment/vllm-freeze.txt"
backend_enabled sglang && "${UV_BIN}" pip freeze --python "${SGLANG_ENV}/bin/python" >"${CLOSURE_ROOT}/environment/sglang-freeze.txt"
"${UV_BIN}" pip freeze --python "${GUIDELLM_ENV}/bin/python" >"${CLOSURE_ROOT}/environment/guidellm-${GUIDELLM_VERSION}-freeze.txt"
"${GUIDELLM_ENV}/bin/guidellm" --help >"${CLOSURE_ROOT}/environment/guidellm-help.txt" 2>&1
"${GUIDELLM_ENV}/bin/guidellm" run --help >"${CLOSURE_ROOT}/environment/guidellm-run-help.txt" 2>&1
if [[ -n "${DRIVER_LIB_ROOT}" ]]; then
    find "${DRIVER_LIB_ROOT}" -maxdepth 1 -type f \( -name 'libcuda.so.*' -o -name 'libnvidia-*.so.*' \) -print0 \
        | sort -z \
        | xargs -0 sha256sum >"${CLOSURE_ROOT}/environment/driver-libraries.sha256"
fi

preflight_status=0
[[ -z "${DRIVER_LIB_ROOT}" || -s "${CLOSURE_ROOT}/environment/driver-libraries.sha256" ]] || preflight_status=1
expected_models=()
model_ref_path() {
    local model="$1"
    printf '%s/hub/models--%s/refs/main\n' "${HF_HOME}" "${model//\//--}"
}
if [[ "${MODEL_SCOPE}" == "wide" ]]; then
    for record in "${WIDE_MODEL_RECORDS[@]}"; do
        IFS='|' read -r _label model revision _prompt_tokens _output_tokens <<<"${record}"
        expected_models+=("$(model_ref_path "${model}"):${revision}")
    done
else
    [[ "${MODEL_SCOPE}" == "all" || "${MODEL_SCOPE}" == "tiny" ]] \
        && expected_models+=("$(model_ref_path "${TINY_MODEL}"):${TINY_REVISION}")
    [[ "${MODEL_SCOPE}" == "all" || "${MODEL_SCOPE}" == "seven_b" ]] \
        && expected_models+=("$(model_ref_path "${SEVEN_B_MODEL}"):${SEVEN_B_REVISION}")
fi
for expected in "${expected_models[@]}"; do
    ref_path="${expected%%:*}"
    ref_sha="${expected##*:}"
    snapshot_path="${ref_path%/refs/main}/snapshots/${ref_sha}"
    [[ -f "${ref_path}" && "$(cat "${ref_path}")" == "${ref_sha}" ]] || preflight_status=1
    [[ -f "${snapshot_path}/config.json" ]] || preflight_status=1
    [[ -n "$(find -L "${snapshot_path}" -maxdepth 1 -name '*.safetensors' -type f -print -quit 2>/dev/null)" ]] || preflight_status=1
done
nvidia-smi >/dev/null || preflight_status=1
[[ -x "${CUDA_HOME}/bin/nvcc" ]] || preflight_status=1
[[ "$("${VLLM_ENV}/bin/python" -c 'import platform; print(platform.python_version())')" == "${BACKEND_PYTHON_VERSION}" ]] || preflight_status=1
[[ "$("${GUIDELLM_ENV}/bin/python" -c 'import platform; print(platform.python_version())')" == "${GUIDELLM_PYTHON_VERSION}" ]] || preflight_status=1
[[ "$("${GUIDELLM_ENV}/bin/guidellm" --version 2>/dev/null | tail -1)" == "guidellm version: ${GUIDELLM_VERSION}" ]] || preflight_status=1
[[ "$("${VLLM_ENV}/bin/python" -c 'import vllm; print(vllm.__version__)')" == "${VLLM_VERSION}" ]] || preflight_status=1
"${VLLM_ENV}/bin/serve-optimize" doctor --profile vllm >"${CLOSURE_ROOT}/environment/vllm-doctor.txt" || preflight_status=1
if backend_enabled sglang; then
    [[ "$("${SGLANG_ENV}/bin/python" -c 'import platform; print(platform.python_version())')" == "${BACKEND_PYTHON_VERSION}" ]] || preflight_status=1
    [[ "$("${SGLANG_ENV}/bin/python" -c 'import sglang; print(sglang.__version__)' 2>/dev/null | tail -1)" == "${SGLANG_VERSION}" ]] || preflight_status=1
    "${SGLANG_ENV}/bin/serve-optimize" doctor --profile sglang >"${CLOSURE_ROOT}/environment/sglang-doctor.txt" || preflight_status=1
fi
"${UV_BIN}" pip check --python "${VLLM_ENV}/bin/python" >/dev/null || preflight_status=1
if backend_enabled sglang; then
    "${UV_BIN}" pip check --python "${SGLANG_ENV}/bin/python" >/dev/null || preflight_status=1
fi
"${UV_BIN}" pip check --python "${GUIDELLM_ENV}/bin/python" >/dev/null || preflight_status=1
if (( preflight_status == 0 )); then
    for expected in "${expected_models[@]}"; do
        ref_path="${expected%%:*}"
        ref_sha="${expected##*:}"
        snapshot_path="${ref_path%/refs/main}/snapshots/${ref_sha}"
        find -L "${snapshot_path}" -maxdepth 1 -type f -print0 \
            | sort -z \
            | xargs -0 sha256sum
    done >"${CLOSURE_ROOT}/environment/model-files.sha256"
fi
finish_step preflight "${preflight_status}"
if (( preflight_status != 0 )); then
    exit "${preflight_status}"
fi

overall_status=0

run_managed() {
    local phase="$1"
    local label="$2"
    local backend="$3"
    local model="$4"
    local goal="$5"
    local profile="$6"
    local manifest="$7"
    local repetition="$8"
    local limit="$9"
    local step="${phase}_${label}_${backend}_r${repetition}"
    local output="${CLOSURE_ROOT}/managed/${phase}/${label}-${backend}-r${repetition}"
    local log="${LOGS}/${step}.log"
    local telemetry="${CLOSURE_ROOT}/telemetry/${step}.csv"
    local python="${SGLANG_ENV}/bin/python"
    local evidence="${CLOSURE_ROOT}/evidence/${step}.sqlite"
    local manifest_args=()
    local status
    local command
    local run_path

    if [[ -f "${MARKERS}/${step}.ok" ]]; then
        if [[ -s "${MARKERS}/${step}.managed-run" ]]; then
            return 0
        fi
        finish_step "${step}" 1
        overall_status=1
        return 1
    fi
    [[ "${backend}" == "vllm" ]] && python="${VLLM_ENV}/bin/python"
    [[ -n "${manifest}" ]] && manifest_args=(--workload-manifest "${SOURCE_ROOT}/${manifest}")
    mkdir -p "${output}" "$(dirname "${evidence}")"
    command=(
        "${python}" -m serve_optimize managed-evaluate
        --backend "${backend}"
        --model "${model}"
        --goal "${goal}"
        --limit "${limit}"
        --trials 1
        --telemetry auto
        --workload-profile "${profile}"
        --startup-timeout 600
        --cooldown-seconds 5
        --out "${output}"
        --evidence-db "${evidence}"
        --warmup-requests "${WARMUP_REQUESTS}"
        --steady-state-seconds "${STEADY_STATE_SECONDS}"
        --idle-baseline-seconds "${IDLE_BASELINE_SECONDS}"
        "${manifest_args[@]}"
    )
    command_file "${output}/command.txt" "${command[@]}"
    mark "${step}_started_at" "$(date --iso-8601=seconds)"
    start_telemetry "${telemetry}"
    timeout --signal=TERM --kill-after=2m 2h "${command[@]}" >"${log}" 2>&1
    status=$?
    stop_telemetry
    if (( status == 0 )); then
        run_path="$(find "${output}" -mindepth 2 -maxdepth 2 -name managed_run.json -type f -printf '%h\n' | sort | tail -1)"
        if [[ -n "${run_path}" && -f "${run_path}/managed_run.json" ]]; then
            printf '%s\n' "${run_path}" >"${MARKERS}/${step}.managed-run"
        else
            status=1
        fi
    fi
    finish_step "${step}" "${status}"
    (( status == 0 )) || overall_status=1
    return "${status}"
}

server_command_for() {
    local backend="$1"
    local model="$2"
    local revision="$3"
    local port="$4"
    if [[ "${backend}" == "vllm" ]]; then
        SERVER_COMMAND=(
            "${VLLM_ENV}/bin/vllm" serve "${model}"
            --revision "${revision}"
            --served-model-name "${model}"
            --host 127.0.0.1
            --port "${port}"
        )
    else
        SERVER_COMMAND=(
            "${SGLANG_ENV}/bin/python" -m sglang.launch_server
            --model-path "${model}"
            --revision "${revision}"
            --served-model-name "${model}"
            --host 127.0.0.1
            --port "${port}"
        )
    fi
}

start_server() {
    local backend="$1"
    local model="$2"
    local revision="$3"
    local port="$4"
    local label="$5"
    local log="${LOGS}/server-${label}-${backend}.log"
    local command_path="${CLOSURE_ROOT}/guidellm/${label}-${backend}/backend-command.txt"

    mkdir -p "$(dirname "${command_path}")"
    server_command_for "${backend}" "${model}" "${revision}" "${port}"
    command_file "${command_path}" "${SERVER_COMMAND[@]}"
    setsid "${SERVER_COMMAND[@]}" >"${log}" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 600); do
        if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
            SERVER_COMMAND_FILE="${command_path}"
            return 0
        fi
        kill -0 "${SERVER_PID}" 2>/dev/null || return 1
        sleep 1
    done
    return 1
}

write_guidellm_metadata() {
    local run_dir="$1"
    local backend="$2"
    local version="$3"
    local model="$4"
    local revision="$5"
    local profile="$6"
    "${GUIDELLM_ENV}/bin/python" - "${run_dir}" "${backend}" "${version}" "${model}" "${revision}" "${profile}" "${GUIDELLM_VERSION}" "${TELEMETRY_SCOPE}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

run_dir = Path(sys.argv[1])
payload = {
    "schema_version": "guidellm-run-metadata/v1",
    "guidellm_version": sys.argv[7],
    "backend": sys.argv[2],
    "backend_version": sys.argv[3],
    "model": sys.argv[4],
    "model_revision": sys.argv[5],
    "workload_profile": sys.argv[6],
    "seed": {"kind": "static", "value": 7},
    "backend_target": (run_dir / "backend-target.txt").read_text().strip(),
    "backend_command": (run_dir / "backend-command.txt").read_text().strip(),
    "serve_optimize_command": (run_dir / "serve-optimize-command.txt").read_text().strip(),
    "guidellm_command": (run_dir / "guidellm-command.txt").read_text().strip(),
    "telemetry_scope": sys.argv[8],
    "created_at": datetime.now(timezone.utc).isoformat(),
}
(run_dir / "run_metadata.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

run_crosscheck() {
    local label="$1"
    local backend="$2"
    local version="$3"
    local model="$4"
    local revision="$5"
    local port="$6"
    local profile="$7"
    local prompt_tokens="$8"
    local output_tokens="$9"
    local step="guidellm_${label}_${backend}_${profile}"
    local run_dir="${CLOSURE_ROOT}/guidellm/${label}-${backend}/${profile}"
    local tokenizer="${HF_HOME}/hub/models--${model//\//--}/snapshots/${revision}"
    local serve_python="${SGLANG_ENV}/bin/python"
    local telemetry="${CLOSURE_ROOT}/telemetry/${step}.csv"
    local serve_status
    local guide_status
    local status=0
    local serve_command
    local guide_command

    [[ -f "${MARKERS}/${step}.ok" ]] && return 0
    [[ "${backend}" == "vllm" ]] && serve_python="${VLLM_ENV}/bin/python"
    mkdir -p "${run_dir}"
    cp "${SERVER_COMMAND_FILE}" "${run_dir}/backend-command.txt"
    printf 'http://127.0.0.1:%s\n' "${port}" >"${run_dir}/backend-target.txt"
    serve_command=(
        "${serve_python}" -m serve_optimize recommend "${model}"
        --base-url "http://127.0.0.1:${port}/v1"
        --backend "${backend}"
        --candidate-source sweep
        --concurrency-sweep 16,32,64,128,256
        --top-k 5
        --goal throughput
        --telemetry auto
        --workload-profile "${profile}"
        --override-num-requests 272
        --warmup-requests "${WARMUP_REQUESTS}"
        --steady-state-seconds "${STEADY_STATE_SECONDS}"
        --idle-baseline-seconds "${IDLE_BASELINE_SECONDS}"
        --stream
        --timeout 180
        --out "${run_dir}/serve-optimize"
    )
    guide_command=(
        "${GUIDELLM_ENV}/bin/guidellm" run
        --backend "kind=openai_http,target=http://127.0.0.1:${port},request_format=/v1/chat/completions,model=${model}"
        --profile "kind=concurrent,warmup=${GUIDELLM_WARMUP_SECONDS}"
        --override profile.streams 16,32,64,128,256
        --constraint "kind=max_duration,seconds=${GUIDELLM_TOTAL_SECONDS}"
        --constraint kind=max_errors,count=5
        --constraint kind=over_saturation,mode=monitor,min_seconds=30
        --tokenizer "kind=huggingface_auto,model=${tokenizer}"
        --data "kind=synthetic_text,prompt_tokens=${prompt_tokens},output_tokens=${output_tokens}"
        --seed kind=static,value=7
        --output "kind=json,path=${run_dir}/benchmarks.json"
        --output "kind=csv,path=${run_dir}/benchmarks.csv"
        --disable-console-interactive
    )
    command_file "${run_dir}/serve-optimize-command.txt" "${serve_command[@]}"
    command_file "${run_dir}/guidellm-command.txt" "${guide_command[@]}"
    write_guidellm_metadata "${run_dir}" "${backend}" "${version}" "${model}" "${revision}" "${profile}"
    mark "${step}_started_at" "$(date --iso-8601=seconds)"
    start_telemetry "${telemetry}"
    timeout --signal=TERM --kill-after=1m 30m "${serve_command[@]}" >"${run_dir}/serve-optimize.log" 2>&1
    serve_status=$?
    timeout --signal=TERM --kill-after=1m 30m "${guide_command[@]}" >"${run_dir}/guidellm.log" 2>&1
    guide_status=$?
    stop_telemetry
    printf 'serve_optimize_exit_code=%s\nguidellm_exit_code=%s\n' "${serve_status}" "${guide_status}" >"${run_dir}/exit-codes.env"
    (( serve_status == 0 )) || status=1
    (( guide_status == 0 )) || status=1
    finish_step "${step}" "${status}"
    (( status == 0 )) || overall_status=1
    return "${status}"
}

run_server_group() {
    local label="$1"
    local backend="$2"
    local version="$3"
    local model="$4"
    local revision="$5"
    local port="$6"
    shift 6
    local group="server_${label}_${backend}"
    local status=0

    mark "${group}_started_at" "$(date --iso-8601=seconds)"
    if ! start_server "${backend}" "${model}" "${revision}" "${port}" "${label}"; then
        status=1
    else
        while (( $# >= 3 )); do
            run_crosscheck "${label}" "${backend}" "${version}" "${model}" "${revision}" "${port}" "$1" "$2" "$3" || status=1
            shift 3
        done
    fi
    stop_server
    finish_step "${group}" "${status}"
    (( status == 0 )) || overall_status=1
    return "${status}"
}

run_wide_model() {
    local record="$1"
    local label model revision prompt_tokens output_tokens
    local backend_version
    IFS='|' read -r label model revision prompt_tokens output_tokens <<<"${record}"
    for backend in "${BACKENDS[@]}"; do
        backend_version="${VLLM_VERSION}"
        [[ "${backend}" == "sglang" ]] && backend_version="${SGLANG_VERSION}"
        run_server_group "${label}" "${backend}" "${backend_version}" "${model}" "${revision}" \
            18101 short "${prompt_tokens}" "${output_tokens}" medium 1024 256 long-prefill 4096 128 || true
        run_managed saturation "${label}-short" "${backend}" "${model}" performance short \
            configs/workloads/saturation-short.json 1 8 || true
        run_managed saturation "${label}-medium" "${backend}" "${model}" performance medium \
            configs/workloads/saturation-medium.json 1 8 || true
        run_managed saturation "${label}-long-prefill" "${backend}" "${model}" performance long-prefill \
            configs/workloads/saturation-long-prefill.json 1 8 || true
        run_managed workloads "${label}-mixed" "${backend}" "${model}" balanced mixed "" 1 5 || true
        run_managed workloads "${label}-code-generation" "${backend}" "${model}" balanced code-generation "" 1 5 || true
        run_managed workloads "${label}-repeated-prefix" "${backend}" "${model}" balanced repeated-prefix "" 1 5 || true
        run_managed workloads "${label}-real-chat" "${backend}" "${model}" balanced medium \
            configs/workloads/oasst1-root-en-64.json 1 5 || true
        for repetition in 1 2 3; do
            run_managed repetitions "${label}-short" "${backend}" "${model}" balanced short "" "${repetition}" 5 || true
        done
    done
}

if [[ "${MODEL_SCOPE}" == "wide" ]]; then
    for record in "${WIDE_MODEL_RECORDS[@]}"; do
        run_wide_model "${record}"
    done
elif [[ "${MODEL_SCOPE}" != "seven_b" ]]; then
    backend_enabled vllm && run_server_group Qwen3-0.6B vllm "${VLLM_VERSION}" "${TINY_MODEL}" "${TINY_REVISION}" 18101 short 128 64 || true
    backend_enabled sglang && run_server_group Qwen3-0.6B sglang "${SGLANG_VERSION}" "${TINY_MODEL}" "${TINY_REVISION}" 18102 short 128 64 || true
    for backend in "${BACKENDS[@]}"; do
        run_managed saturation qwen3-short "${backend}" "${TINY_MODEL}" performance short \
            configs/workloads/saturation-short.json 1 8 || true
        run_managed saturation qwen3-medium "${backend}" "${TINY_MODEL}" performance medium \
            configs/workloads/saturation-medium.json 1 8 || true
        run_managed saturation qwen3-long-prefill "${backend}" "${TINY_MODEL}" performance long-prefill \
            configs/workloads/saturation-long-prefill.json 1 8 || true
    done
    for repetition in 1 2 3; do
        for backend in "${BACKENDS[@]}"; do
            run_managed repetitions qwen3-short "${backend}" "${TINY_MODEL}" balanced short "" "${repetition}" 5 || true
        done
    done
fi

if [[ "${MODEL_SCOPE}" != "tiny" && "${MODEL_SCOPE}" != "wide" ]]; then
    backend_enabled vllm && run_server_group Qwen2.5-7B-Instruct vllm "${VLLM_VERSION}" "${SEVEN_B_MODEL}" "${SEVEN_B_REVISION}" 18101 \
        medium 1024 256 long-prefill 4096 128 || true
    backend_enabled sglang && run_server_group Qwen2.5-7B-Instruct sglang "${SGLANG_VERSION}" "${SEVEN_B_MODEL}" "${SEVEN_B_REVISION}" 18102 \
        medium 1024 256 long-prefill 4096 128 || true
    for backend in "${BACKENDS[@]}"; do
        run_managed saturation qwen25-7b-short "${backend}" "${SEVEN_B_MODEL}" performance short \
            configs/workloads/saturation-short.json 1 8 || true
        run_managed saturation qwen25-7b-medium "${backend}" "${SEVEN_B_MODEL}" performance medium \
            configs/workloads/saturation-medium.json 1 8 || true
        run_managed saturation qwen25-7b-long-prefill "${backend}" "${SEVEN_B_MODEL}" performance long-prefill \
            configs/workloads/saturation-long-prefill.json 1 8 || true
    done
    for repetition in 1 2 3; do
        for backend in "${BACKENDS[@]}"; do
            run_managed repetitions qwen25-7b-long-prefill "${backend}" "${SEVEN_B_MODEL}" balanced long-prefill "" "${repetition}" 5 || true
        done
    done
fi

postprocess_status=0
mapfile -t managed_runs < <(find "${MARKERS}" -name '*.managed-run' -type f -exec cat {} + | sort -u)
for run_path in "${managed_runs[@]}"; do
    [[ "${run_path}" == "${CLOSURE_ROOT}/managed/"* && -f "${run_path}/managed_run.json" ]] || postprocess_status=1
done
if (( ${#managed_runs[@]} > 0 )); then
    "${VLLM_ENV}/bin/python" "${SOURCE_ROOT}/scripts/audit_benchmark_matrix_records.py" \
        "${managed_runs[@]}" --out "${POSTPROCESS}/managed-raw-audit.json" \
        >"${LOGS}/managed-raw-audit.log" 2>&1 || postprocess_status=1
else
    postprocess_status=1
fi
"${VLLM_ENV}/bin/python" "${SOURCE_ROOT}/scripts/audit_guidellm_results.py" \
    "${CLOSURE_ROOT}/guidellm" \
    --json-out "${POSTPROCESS}/guidellm-audit.json" \
    --csv-out "${POSTPROCESS}/guidellm-summary.csv" \
    >"${LOGS}/guidellm-audit.log" 2>&1 || postprocess_status=1
if (( ${#managed_runs[@]} > 0 )); then
    "${VLLM_ENV}/bin/python" -m serve_optimize validate-campaign "${managed_runs[@]}" \
        --out "${POSTPROCESS}/validation-campaign" >"${LOGS}/validation-campaign.log" 2>&1 || postprocess_status=1
    "${VLLM_ENV}/bin/python" -m serve_optimize research-package "${managed_runs[@]}" \
        --out "${POSTPROCESS}/research-package" --json >"${LOGS}/research-package.log" 2>&1 || postprocess_status=1
    "${VLLM_ENV}/bin/python" "${SOURCE_ROOT}/scripts/analyze_recommendation_search.py" \
        "${managed_runs[@]}" --out "${POSTPROCESS}/recommendation-search" --seed-count 20 \
        >"${LOGS}/recommendation-search.log" 2>&1 || postprocess_status=1
else
    postprocess_status=1
fi

managed_run_count="${#managed_runs[@]}"
guidellm_report_count="$(find "${CLOSURE_ROOT}/guidellm" -name benchmarks.json -type f | wc -l)"
mark managed_run_count "${managed_run_count}"
mark expected_managed_run_count "${EXPECTED_MANAGED_RUNS}"
mark guidellm_report_count "${guidellm_report_count}"
mark expected_guidellm_report_count "${EXPECTED_GUIDELLM_REPORTS}"
(( managed_run_count == EXPECTED_MANAGED_RUNS )) || postprocess_status=1
(( guidellm_report_count == EXPECTED_GUIDELLM_REPORTS )) || postprocess_status=1

(
    cd "${CLOSURE_ROOT}"
    find source managed guidellm telemetry environment logs evidence markers -type f -print0 \
        | sort -z \
        | xargs -0 sha256sum >"${POSTPROCESS}/RAW_MANIFEST.sha256"
    sha256sum -c "${POSTPROCESS}/RAW_MANIFEST.sha256" >"${POSTPROCESS}/raw-manifest-verification.txt"
) || postprocess_status=1
finish_step postprocess "${postprocess_status}"
(( postprocess_status == 0 )) || overall_status=1

mark finished_at "$(date --iso-8601=seconds)"
mark overall_exit_code "${overall_status}"
if (( overall_status == 0 )); then
    touch "${CLOSURE_ROOT}/COMPLETE.ok"
else
    touch "${CLOSURE_ROOT}/INCOMPLETE.failed"
fi
exit "${overall_status}"
