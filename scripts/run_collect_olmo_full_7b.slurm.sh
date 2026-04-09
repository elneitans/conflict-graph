#!/bin/bash
#SBATCH --job-name=OLMO_collect_full_7b
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

echo "=== Environment preflight ==="
echo "PWD=$PWD"
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}"
echo "CONDA_ENV_NAME=${CONDA_ENV_NAME}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-UNSET}"
echo "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES-UNSET}"
echo "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES-UNSET}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS-UNSET}"
echo "which python: $(which python)"
python --version

python scripts/collect_olmo_responses.py \
  --dataset-root data/conflict_graphv2_provisional \
  --selection all \
  --run-name Olmo-3-1025-7B_full \
  --model-id allenai/Olmo-3-1025-7B \
  --dtype float16 \
  --quantization none \
  --device-map auto \
  --max-memory-per-gpu 46GiB \
  --attn-implementation eager \
  --max-new-tokens 512
