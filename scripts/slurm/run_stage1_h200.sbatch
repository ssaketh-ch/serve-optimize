#!/usr/bin/env bash
#SBATCH --job-name=serveopt-stage1
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_h200_nvl_3g.71gb:1
#SBATCH --time=36:00:00
#SBATCH --output=serveopt-stage1-%j.out
#SBATCH --error=serveopt-stage1-%j.err

set -uo pipefail

ROOT="${SERVE_OPTIMIZE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PLAN="${SERVE_OPTIMIZE_PLAN_ROOT:?SERVE_OPTIMIZE_PLAN_ROOT is required}"
RUN_ROOT="${SERVE_OPTIMIZE_RUN_ROOT:?SERVE_OPTIMIZE_RUN_ROOT is required}"
SLURM_ROOT="${ROOT}/results/benchmark-matrix/slurm"
VLLM_ENV="${SERVE_OPTIMIZE_VLLM_ENV:-${ROOT}/.venv-vllm024-py310}"
SGLANG_ENV="${SERVE_OPTIMIZE_SGLANG_ENV:-${ROOT}/.venv-sglang}"
JOB_TAG="${SLURM_JOB_ID:-manual}"
STATUS_PATH="${SLURM_ROOT}/stage1-${JOB_TAG}.status"
AUDIT_PATH="${ROOT}/results/benchmark-matrix/corrected-stage1-audit-${JOB_TAG}.json"

mkdir -p "${SLURM_ROOT}"
cd "${ROOT}"

export CUDA_HOME="/usr/local/cuda"
export PYTHONUNBUFFERED=1
export HF_XET_HIGH_PERFORMANCE=1
unset HF_HUB_ENABLE_HF_TRANSFER
SLURM_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-unset}"
export CUDA_VISIBLE_DEVICES=0

{
    echo "started_at=$(date --iso-8601=seconds)"
    echo "hostname=$(hostname)"
    echo "slurm_job_id=${SLURM_JOB_ID:-manual}"
    echo "slurm_cuda_visible_devices=${SLURM_CUDA_VISIBLE_DEVICES}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
    echo "repository_commit=$(git rev-parse HEAD)"
    echo "dirty_tree=$(test -n "$(git status --porcelain)" && echo true || echo false)"
    echo "cuda_home=${CUDA_HOME}"
    nvidia-smi
    "${VLLM_ENV}/bin/python" -c 'import torch, vllm; print(f"vllm={vllm.__version__} torch={torch.__version__} cuda={torch.version.cuda}")'
    "${SGLANG_ENV}/bin/python" -c 'import sglang, torch; print(f"sglang={sglang.__version__} torch={torch.__version__} cuda={torch.version.cuda}")'
} >"${SLURM_ROOT}/stage1-${JOB_TAG}.environment.txt" 2>&1

overall_status=0

run_backend() {
    local backend="$1"
    local runner="${PLAN}/benchmark_matrix_stage_1_sanity_${backend}.sh"
    local log="${SLURM_ROOT}/stage1-${JOB_TAG}-${backend}.log"
    local started
    local finished
    local status

    started="$(date --iso-8601=seconds)"
    echo "${backend}_started_at=${started}" >>"${STATUS_PATH}"
    bash "${runner}" >"${log}" 2>&1
    status=$?
    finished="$(date --iso-8601=seconds)"
    echo "${backend}_finished_at=${finished}" >>"${STATUS_PATH}"
    echo "${backend}_exit_code=${status}" >>"${STATUS_PATH}"
    return "${status}"
}

echo "job_id=${JOB_TAG}" >"${STATUS_PATH}"
echo "started_at=$(date --iso-8601=seconds)" >>"${STATUS_PATH}"

if ! run_backend vllm; then
    overall_status=1
fi
if ! run_backend sglang; then
    overall_status=1
fi

"${VLLM_ENV}/bin/python" "${ROOT}/scripts/audit_benchmark_matrix_records.py" \
    "${RUN_ROOT}" \
    --out "${AUDIT_PATH}" \
    >"${SLURM_ROOT}/stage1-${JOB_TAG}-audit.log" 2>&1
audit_status=$?
echo "audit_exit_code=${audit_status}" >>"${STATUS_PATH}"
echo "audit_path=${AUDIT_PATH}" >>"${STATUS_PATH}"
if (( audit_status != 0 )); then
    overall_status=1
fi

echo "finished_at=$(date --iso-8601=seconds)" >>"${STATUS_PATH}"
echo "overall_exit_code=${overall_status}" >>"${STATUS_PATH}"
exit "${overall_status}"
