#!/bin/bash
#SBATCH --job-name=simple_inf
#SBATCH -p ialab
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH -w antuco
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=natan.brugueras@cenia.cl

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs outputs


# Activate your prepared environment.
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate antuco_torch310

# Recommended caches on local/scratch storage.
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export PYTORCH_HIP_ALLOC_CONF="garbage_collection_threshold:0.8,max_split_size_mb:512"

# Important for ROCm on this cluster:
unset CUDA_VISIBLE_DEVICES
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-UNSET}"
echo "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES-UNSET}"
echo "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES-UNSET}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS-UNSET}"

python - <<'PY'
import torch
print("torch.__version__ =", torch.__version__)
print("torch.version.hip =", torch.version.hip)
print("torch.cuda.is_available() =", torch.cuda.is_available())
print("torch.cuda.device_count() =", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print("device", i, torch.cuda.get_device_name(i))
PY

# Number of dataset prompts to process (override at submit time):
# sbatch --export=ALL,N_PROMPTS=50 slurm_qwen3_14b_transformers.sbatch
N_PROMPTS="${N_PROMPTS:-1}"
START_INDEX="${START_INDEX:-0}"
OUTPUT_JSONL="${OUTPUT_JSONL:-outputs/safetyconflicts_qwen3_14b_${SLURM_JOB_ID}.jsonl}"

python run_qwen3_transformers.py \
  --model-id "Qwen/Qwen3-14B" \
  --dtype bfloat16 \
  --device-map auto \
  --max-memory-per-gpu 46GiB \
  --max-new-tokens 256 \
  --temperature 0.7 \
  --top-p 0.9 \
  --dataset-id "hadikhalaf/safetyconflicts" \
  --dataset-split "train" \
  --prompt-column "prompt" \
  --num-prompts "${N_PROMPTS}" \
  --start-index "${START_INDEX}" \
  --output-jsonl "${OUTPUT_JSONL}"
