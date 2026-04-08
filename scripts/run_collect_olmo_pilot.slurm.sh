#!/bin/bash
#SBATCH --job-name=OLMO_collect
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
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-garbage_collection_threshold:0.8,max_split_size_mb:512}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"

# Important for ROCm visibility on antuco.
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

if [[ "${INSTALL_REQUIREMENTS:-0}" == "1" ]]; then
  python -m pip install -r .requirements.txt
fi

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
python -m pip show torch || true
python -m pip show bitsandbytes || true
if command -v module >/dev/null 2>&1; then
  echo "=== module list ==="
  module list 2>&1 || true
fi
if command -v rocminfo >/dev/null 2>&1; then
  echo "=== rocminfo (first 40 lines) ==="
  rocminfo 2>/dev/null | head -n 40 || true
fi
if command -v rocm-smi >/dev/null 2>&1; then
  echo "=== rocm-smi ==="
  rocm-smi || true
fi

python - <<'PY'
import torch
try:
    import bitsandbytes as bnb
    bnb_version = getattr(bnb, "__version__", "unknown")
except Exception as exc:
    bnb_version = f"IMPORT_FAILED: {exc!r}"
info = {
    "torch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    "hip_version": getattr(torch.version, "hip", None),
    "cuda_version": getattr(torch.version, "cuda", None),
    "bitsandbytes": bnb_version,
}
if torch.cuda.is_available():
    info["device_name_0"] = torch.cuda.get_device_name(0)
print(info)
if not torch.cuda.is_available():
    raise SystemExit(
        "torch.cuda.is_available() is false. The active conda environment likely does not expose a ROCm-enabled PyTorch build, or the GPU is not visible in the job."
    )
PY

python scripts/collect_olmo_responses.py \
  --dataset-root data/conflict_graphv2_provisional \
  --run-name olmo3_1125_32b_pilot_8bit \
  --model-id allenai/Olmo-3-1125-32B \
  --dtype float16 \
  --quantization 8bit \
  --device-map auto \
  --max-memory-per-gpu 46GiB \
  --attn-implementation eager \
  --max-new-tokens 256
