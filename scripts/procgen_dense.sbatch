#!/usr/bin/env bash
#SBATCH --job-name=procgen-dense
#SBATCH --partition=digitallab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-%x_%j.out
#SBATCH --error=slurm-%x_%j.err

set -euo pipefail

# ---- Repo root auto-detect (assumes this file lives in scripts/) ----
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---- User-configurable knobs (override via env vars) ----
VENV="${CORA_VENV:-/scratch/$USER/venvs/cora}"
EXPERIMENT="${EXPERIMENT:-procgen_6_tasks_5_cycles}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
NUM_STEPS="${NUM_STEPS:-256}"

echo "HOSTNAME: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}"
echo "REPO: $REPO"
echo "VENV: $VENV"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"

cd "$REPO"

if [[ ! -f "$VENV/bin/activate" ]]; then
  echo "ERROR: venv not found at: $VENV"
  echo "Set it with: export CORA_VENV=/path/to/venv"
  exit 1
fi
source "$VENV/bin/activate"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python - <<'PY'
import sys, torch
print("python:", sys.executable)
print("torch:", torch.__version__, "torch.cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device_count:", torch.cuda.device_count())
    print("device0:", torch.cuda.get_device_name(0))
PY

RUN_DIR="runs/${EXPERIMENT}/dense/ppo_dense_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
echo "RUN_DIR: $RUN_DIR"

PYTHONWARNINGS=ignore python main.py \
  --policy ppo \
  --experiment "$EXPERIMENT" \
  --output-dir "$RUN_DIR" \
  --num_processes "$NUM_PROCESSES" \
  --num_steps "$NUM_STEPS" \
  --cuda True
