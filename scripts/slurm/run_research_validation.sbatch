#!/usr/bin/env bash
#SBATCH --job-name=serveopt-validation
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --output=serveopt-validation-%j.out
#SBATCH --error=serveopt-validation-%j.err

set -euo pipefail

VALIDATION_ROOT="${SERVE_OPTIMIZE_VALIDATION_ROOT:?SERVE_OPTIMIZE_VALIDATION_ROOT is required}"
SOURCE_TARBALL="${SERVE_OPTIMIZE_SOURCE_TARBALL:?SERVE_OPTIMIZE_SOURCE_TARBALL is required}"
SOURCE_SHA256="${SERVE_OPTIMIZE_SOURCE_SHA256:?SERVE_OPTIMIZE_SOURCE_SHA256 is required}"
SOURCE_ROOT="${VALIDATION_ROOT}/source/worktree"
HF_HOME="${HF_HOME:-/home/saketh-msc/quantization/models}"
VLLM_ENV="${SERVE_OPTIMIZE_VLLM_ENV:-/home/saketh-msc/serve-optimize-envs/vllm-0.27.1-py312}"
SGLANG_ENV="${SERVE_OPTIMIZE_SGLANG_ENV:-/home/saketh-msc/serve-optimize-envs/sglang-0.5.17-py312}"
GUIDELLM_ENV="${SERVE_OPTIMIZE_GUIDELLM_ENV:-/home/saketh-msc/serve-optimize-envs/guidellm-0.7.3-paper}"
TELEMETRY_SCOPE="${SERVE_OPTIMIZE_TELEMETRY_SCOPE:-physical_gpu_board_for_mig_allocation}"
POSTPROCESS="${VALIDATION_ROOT}/postprocess"

mkdir -p "${VALIDATION_ROOT}" "${POSTPROCESS}"
actual_source_sha="$(sha256sum "${SOURCE_TARBALL}" | awk '{print $1}')"
[[ "${actual_source_sha}" == "${SOURCE_SHA256}" ]]
if [[ ! -f "${VALIDATION_ROOT}/source-extracted.ok" ]]; then
    [[ ! -e "${SOURCE_ROOT}" ]]
    mkdir -p "${SOURCE_ROOT}"
    tar -xzf "${SOURCE_TARBALL}" -C "${SOURCE_ROOT}"
    touch "${VALIDATION_ROOT}/source-extracted.ok"
fi
cd "${SOURCE_ROOT}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.2}"
export CUDA_PATH="${CUDA_HOME}"
export PATH="${SGLANG_ENV}/bin:${VLLM_ENV}/bin:${PATH}"
export PYTHONPATH="${SOURCE_ROOT}/src"
export PYTHONUNBUFFERED=1
export HF_HOME
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_XET_HIGH_PERFORMANCE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-24}}"
export SERVE_OPTIMIZE_TELEMETRY_SCOPE="${TELEMETRY_SCOPE}"
if [[ -n "${SERVE_OPTIMIZE_DRIVER_LIB_ROOT:-}" ]]; then
    export LD_LIBRARY_PATH="${SERVE_OPTIMIZE_DRIVER_LIB_ROOT}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

PYTHON="${VLLM_ENV}/bin/python"
"${PYTHON}" scripts/run_research_validation.py \
    --root "${SOURCE_ROOT}" \
    --out "${VALIDATION_ROOT}/results" \
    --hf-home "${HF_HOME}" \
    --model-label qwen3-0.6b \
    --model-label mistral-7b \
    --model-label qwen38-27b \
    --backend vllm \
    --backend sglang \
    --workload short \
    --workload long-prefill \
    --workload real-chat \
    --workload real-chat-holdout \
    --pool-limit 8 \
    --budget 4 \
    --trials 3 \
    >"${VALIDATION_ROOT}/validation.log" 2>&1

mapfile -t managed_runs < <(find "${VALIDATION_ROOT}/results" -type f -name managed_run.json -exec dirname {} \; | sort -u)
(( ${#managed_runs[@]} > 0 ))
(( "$(find "${VALIDATION_ROOT}/results" -type f -name cell_manifest.json | wc -l)" == 24 ))
(( "$(find "${VALIDATION_ROOT}/results" -type f -name search_results.json | wc -l)" == 18 ))
(( "$(find "${VALIDATION_ROOT}/results" -type f -name measured_oracle.json | wc -l)" == 18 ))
(( "$(find "${VALIDATION_ROOT}/results" -type f -name holdout_evaluations.json | wc -l)" == 6 ))

"${PYTHON}" scripts/audit_benchmark_matrix_records.py \
    "${managed_runs[@]}" \
    --out "${POSTPROCESS}/managed-raw-audit.json" \
    >"${POSTPROCESS}/managed-raw-audit.log" 2>&1
"${PYTHON}" -m serve_optimize validate-campaign "${managed_runs[@]}" \
    --out "${POSTPROCESS}/validation-campaign" \
    >"${POSTPROCESS}/validation-campaign.log" 2>&1
"${PYTHON}" -m serve_optimize research-package "${managed_runs[@]}" \
    --out "${POSTPROCESS}/research-package" --json \
    >"${POSTPROCESS}/research-package.log" 2>&1

python_version="$(${PYTHON} -c 'import platform; print(platform.python_version())')"
vllm_version="$(${VLLM_ENV}/bin/python -c 'import vllm; print(vllm.__version__)')"
sglang_version="$(${SGLANG_ENV}/bin/python -c 'import sglang; print(sglang.__version__)')"
guidellm_version="$("${GUIDELLM_ENV}/bin/guidellm" --version 2>/dev/null | tail -1 || true)"
cat >"${POSTPROCESS}/validation-summary.env" <<EOF
source_sha256=${SOURCE_SHA256}
python_version=${python_version}
vllm_version=${vllm_version}
sglang_version=${sglang_version}
guidellm_version=${guidellm_version}
cell_count=24
search_cell_count=18
holdout_cell_count=6
managed_run_count=${#managed_runs[@]}
search_method_set=backend-default,serve-optimize,random,grid,bayesian
random_and_bayesian_seed_count=3
candidate_budget=4
candidate_pool_limit=8
trials_per_candidate=3
evidence_reuse=false
EOF

(
    cd "${VALIDATION_ROOT}"
    find source results postprocess -type f -print0 | sort -z | xargs -0 sha256sum >"${POSTPROCESS}/RAW_MANIFEST.sha256"
    sha256sum -c "${POSTPROCESS}/RAW_MANIFEST.sha256" >"${POSTPROCESS}/raw-manifest-verification.txt"
)
touch "${VALIDATION_ROOT}/COMPLETE.ok"
