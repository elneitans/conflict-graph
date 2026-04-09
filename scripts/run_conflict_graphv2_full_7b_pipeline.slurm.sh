#!/bin/bash
#SBATCH --job-name=CGv2_full_7b_pipeline
#SBATCH -p ialab
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH -w antuco
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=natan.brugueras@cenia.cl

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

mkdir -p logs
mkdir -p data/conflict_graphv2_provisional/collections/runs

if [[ ! -f "scripts/collect_olmo_responses.py" || ! -f "data/conflict_graphv2_provisional/prompt_table.jsonl" ]]; then
  echo "This job must be submitted from the conflict_graph repo root so relative dataset and script paths resolve correctly." >&2
  exit 1
fi

if [[ ! -f ".env" ]]; then
  echo "Missing .env in repo root. Add DEEPSEEK_API_KEY there before submitting this pipeline." >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1

export HF_HOME="${HF_HOME:-/mnt/dccnas/archive/gilureta/natan.brugueras/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
unset TRANSFORMERS_CACHE
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-garbage_collection_threshold:0.8,max_split_size_mb:512}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"

unset CUDA_VISIBLE_DEVICES
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"

CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-antuco_torch310}"

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "Conda activation script not found at ${CONDA_SH}" >&2
  exit 1
fi

source "${CONDA_SH}"
conda activate "${CONDA_ENV_NAME}"

set -a
source .env
set +a

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "DEEPSEEK_API_KEY is not set after sourcing .env" >&2
  exit 1
fi

RUN_NAME="${RUN_NAME:-Olmo-3-1025-7B_full}"
RUN_DIR="data/conflict_graphv2_provisional/collections/runs/${RUN_NAME}"
JUDGE_DIR="${RUN_DIR}/judging/deepseek_action_judge"
SELF_METRICS_DIR="${RUN_DIR}/metrics_self_report"
RESOLVED_METRICS_DIR="${RUN_DIR}/metrics_resolved_singleton"

echo "=== Environment preflight ==="
echo "PWD=$PWD"
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}"
echo "CONDA_ENV_NAME=${CONDA_ENV_NAME}"
echo "RUN_NAME=${RUN_NAME}"
echo "RUN_DIR=${RUN_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-UNSET}"
echo "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES-UNSET}"
echo "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES-UNSET}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS-UNSET}"
echo "which python: $(which python)"
python --version

python - <<'PY'
import torch
info = {
    "torch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    "hip_version": getattr(torch.version, "hip", None),
}
if torch.cuda.is_available():
    info["device_name_0"] = torch.cuda.get_device_name(0)
print(info)
if not torch.cuda.is_available():
    raise SystemExit("No ROCm/CUDA device visible to PyTorch. Check the antuco environment and Slurm GPU allocation.")
PY

echo "=== Step 1/4: full 7B collection ==="
python scripts/collect_olmo_responses.py \
  --dataset-root data/conflict_graphv2_provisional \
  --selection all \
  --run-name "${RUN_NAME}" \
  --model-id allenai/Olmo-3-1025-7B \
  --dtype float16 \
  --quantization none \
  --device-map auto \
  --max-memory-per-gpu 46GiB \
  --attn-implementation eager \
  --max-new-tokens 512

echo "=== Step 2/4: DeepSeek judge ==="
python scripts/judge_conflict_graphv2_actions.py \
  --run-dir "${RUN_DIR}"

echo "=== Step 3/4: self-report metrics ==="
python scripts/compute_conflict_graphv2_metrics.py \
  --run-dir "${RUN_DIR}" \
  --action-source self_report \
  --output-dir "${SELF_METRICS_DIR}"

echo "=== Step 4/4: resolved-singleton metrics ==="
python scripts/compute_conflict_graphv2_metrics.py \
  --run-dir "${RUN_DIR}" \
  --action-source resolved_actions \
  --resolved-actions-jsonl "${JUDGE_DIR}/resolved_actions.jsonl" \
  --output-dir "${RESOLVED_METRICS_DIR}"

echo "=== Pipeline complete ==="
echo "Run dir: ${RUN_DIR}"
echo "Judge dir: ${JUDGE_DIR}"
echo "Self metrics: ${SELF_METRICS_DIR}"
echo "Resolved metrics: ${RESOLVED_METRICS_DIR}"
