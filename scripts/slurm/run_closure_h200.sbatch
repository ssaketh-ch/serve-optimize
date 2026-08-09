#!/usr/bin/env bash
#SBATCH --job-name=serveopt-closure
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_h200_nvl_3g.71gb:1
#SBATCH --time=36:00:00
#SBATCH --output=serveopt-closure-%j.out
#SBATCH --error=serveopt-closure-%j.err

set -uo pipefail

ROOT="${SERVE_OPTIMIZE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
CLOSURE_ROOT="${SERVE_OPTIMIZE_CLOSURE_ROOT:?SERVE_OPTIMIZE_CLOSURE_ROOT is required}"
PLAN="${CLOSURE_ROOT}/plan-vllm024-py310"
RUN_ROOT="${CLOSURE_ROOT}/runs"
SLURM_ROOT="${ROOT}/results/benchmark-matrix/slurm"
VLLM_ENV="${SERVE_OPTIMIZE_VLLM_ENV:-${ROOT}/.venv-vllm024-py310}"
SGLANG_ENV="${SERVE_OPTIMIZE_SGLANG_ENV:-${ROOT}/.venv-sglang}"
JOB_TAG="${SLURM_JOB_ID:-manual}"
STATUS_PATH="${CLOSURE_ROOT}/closure-${JOB_TAG}.status"
AUDIT_PATH="${CLOSURE_ROOT}/audit-${JOB_TAG}.json"
ENVIRONMENT_PATH="${CLOSURE_ROOT}/environment-${JOB_TAG}.txt"

case "${CLOSURE_ROOT}" in
    "${ROOT}"/results/benchmark-matrix/closure-*) ;;
    *)
        echo "Closure root must be a new directory under ${ROOT}/results/benchmark-matrix/closure-*" >&2
        exit 2
        ;;
esac

mkdir -p "${CLOSURE_ROOT}" "${RUN_ROOT}" "${SLURM_ROOT}" "${CLOSURE_ROOT}/logs"
cd "${ROOT}"

export CUDA_HOME="/usr/local/cuda"
export CUDA_PATH="${CUDA_HOME}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export HF_XET_HIGH_PERFORMANCE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
unset HF_HUB_ENABLE_HF_TRANSFER
SLURM_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-unset}"
export CUDA_VISIBLE_DEVICES=0

git status --short >"${CLOSURE_ROOT}/workspace-${JOB_TAG}.status"
git diff --binary >"${CLOSURE_ROOT}/workspace-${JOB_TAG}.patch"
git ls-files --others --exclude-standard -z >"${CLOSURE_ROOT}/workspace-${JOB_TAG}.untracked"
if [[ -s "${CLOSURE_ROOT}/workspace-${JOB_TAG}.untracked" ]]; then
    tar --null --files-from="${CLOSURE_ROOT}/workspace-${JOB_TAG}.untracked" \
        -czf "${CLOSURE_ROOT}/workspace-${JOB_TAG}.untracked.tgz"
fi

{
    echo "started_at=$(date --iso-8601=seconds)"
    echo "hostname=$(hostname)"
    echo "slurm_job_id=${SLURM_JOB_ID:-manual}"
    echo "slurm_cuda_visible_devices=${SLURM_CUDA_VISIBLE_DEVICES}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
    echo "repository_commit=$(git rev-parse HEAD)"
    echo "dirty_tree=$(test -n "$(git status --porcelain)" && echo true || echo false)"
    echo "workspace_patch_sha256=$(sha256sum "${CLOSURE_ROOT}/workspace-${JOB_TAG}.patch" | cut -d' ' -f1)"
    echo "cuda_home=${CUDA_HOME}"
    echo "gpu_telemetry_scope=physical_gpu_board_for_mig_allocation"
    echo "energy_limit=Power and energy are board scoped because NVML reports the parent H200 for this MIG allocation."
    nvidia-smi
    scontrol show job "${SLURM_JOB_ID}" || true
    "${VLLM_ENV}/bin/python" -c 'import torch, vllm; print(f"vllm={vllm.__version__} torch={torch.__version__} cuda={torch.version.cuda}")'
    "${SGLANG_ENV}/bin/python" -c 'import sglang, torch; print(f"sglang={sglang.__version__} torch={torch.__version__} cuda={torch.version.cuda}")'
} >"${ENVIRONMENT_PATH}" 2>&1

overall_status=0
completed_cells=0
failed_cells=0

run_stage1_backend() {
    local backend="$1"
    local runner="${PLAN}/benchmark_matrix_stage_1_sanity_${backend}.sh"
    local log="${CLOSURE_ROOT}/logs/stage1-${backend}.log"
    local status

    echo "stage1_${backend}_started_at=$(date --iso-8601=seconds)" >>"${STATUS_PATH}"
    bash "${runner}" >"${log}" 2>&1
    status=$?
    echo "stage1_${backend}_finished_at=$(date --iso-8601=seconds)" >>"${STATUS_PATH}"
    echo "stage1_${backend}_exit_code=${status}" >>"${STATUS_PATH}"
    if (( status != 0 )); then
        overall_status=1
    fi
}

run_stage2_cell() {
    local backend="$1"
    local slug="$2"
    local model="$3"
    local profile="$4"
    local manifest="${5:-}"
    local cli="${SGLANG_ENV}/bin/serve-optimize"
    if [[ "${backend}" == "vllm" ]]; then
        cli="${VLLM_ENV}/bin/serve-optimize"
    fi
    local out="${RUN_ROOT}/stage_2_closure/${slug}-${backend}"
    local log="${CLOSURE_ROOT}/logs/stage2-${slug}-${backend}.log"
    local status
    local manifest_args=()

    if [[ -n "${manifest}" ]]; then
        manifest_args=(--workload-manifest "${manifest}")
    fi
    echo "stage2_${slug}_${backend}_started_at=$(date --iso-8601=seconds)" >>"${STATUS_PATH}"
    "${cli}" managed-evaluate \
        --backend "${backend}" \
        --model "${model}" \
        --goal balanced \
        --limit 5 \
        --trials 1 \
        --telemetry auto \
        --workload-profile "${profile}" \
        --startup-timeout 300 \
        --cooldown-seconds 5 \
        --out "${out}" \
        --evidence-db "${CLOSURE_ROOT}/stage2-evidence.sqlite" \
        --warmup-requests 4 \
        --steady-state-seconds 15 \
        --idle-baseline-seconds 15 \
        "${manifest_args[@]}" >"${log}" 2>&1
    status=$?
    echo "stage2_${slug}_${backend}_finished_at=$(date --iso-8601=seconds)" >>"${STATUS_PATH}"
    echo "stage2_${slug}_${backend}_exit_code=${status}" >>"${STATUS_PATH}"
    if (( status == 0 )); then
        completed_cells=$((completed_cells + 1))
    else
        failed_cells=$((failed_cells + 1))
        overall_status=1
    fi
}

echo "job_id=${JOB_TAG}" >"${STATUS_PATH}"
echo "started_at=$(date --iso-8601=seconds)" >>"${STATUS_PATH}"
echo "closure_root=${CLOSURE_ROOT}" >>"${STATUS_PATH}"

run_stage1_backend vllm
run_stage1_backend sglang

for backend in vllm sglang; do
    run_stage2_cell "${backend}" qwen25-1p5b-medium "Qwen/Qwen2.5-1.5B-Instruct" medium
    run_stage2_cell "${backend}" deepseek-1p5b-code "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" code-generation
    run_stage2_cell "${backend}" granite-2b-prefix "ibm-granite/granite-3.3-2b-instruct" repeated-prefix
    run_stage2_cell "${backend}" mistral-7b-mixed "mistralai/Mistral-7B-Instruct-v0.3" mixed
    run_stage2_cell "${backend}" qwen25-7b-real-chat "Qwen/Qwen2.5-7B-Instruct" medium \
        "${ROOT}/configs/workloads/oasst1-root-en-64.json"
done

"${VLLM_ENV}/bin/python" "${ROOT}/scripts/audit_benchmark_matrix_records.py" \
    "${RUN_ROOT}" --out "${AUDIT_PATH}" >"${CLOSURE_ROOT}/logs/audit.log" 2>&1
audit_status=$?
managed_run_count=$(find "${RUN_ROOT}" -name managed_run.json -type f | wc -l)

echo "stage2_completed_cell_count=${completed_cells}" >>"${STATUS_PATH}"
echo "stage2_failed_cell_count=${failed_cells}" >>"${STATUS_PATH}"
echo "managed_run_count=${managed_run_count}" >>"${STATUS_PATH}"
echo "expected_managed_run_count=46" >>"${STATUS_PATH}"
echo "audit_exit_code=${audit_status}" >>"${STATUS_PATH}"
echo "audit_path=${AUDIT_PATH}" >>"${STATUS_PATH}"
if (( audit_status != 0 || managed_run_count != 46 )); then
    overall_status=1
fi

echo "finished_at=$(date --iso-8601=seconds)" >>"${STATUS_PATH}"
echo "overall_exit_code=${overall_status}" >>"${STATUS_PATH}"
exit "${overall_status}"
