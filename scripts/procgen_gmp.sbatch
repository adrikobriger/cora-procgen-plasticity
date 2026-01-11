#!/usr/bin/env bash
#SBATCH --job-name=procgen-gmp
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

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"

VENV="${CORA_VENV:-/scratch/$USER/venvs/cora}"
EXPERIMENT="${EXPERIMENT:-procgen_6_tasks_5_cycles}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
NUM_STEPS="${NUM_STEPS:-256}"

# GMP params (override via env var if you want)
GMP_PARAMS="${GMP_PARAMS:-{"final_sparsity":0.8,"tasks_per_cycle":6,"prune_cycle":0}}"

echo "HOSTNAME: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}"
echo "REPO: $REPO"
echo "VENV: $VENV"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"
echo "GMP_PARAMS: $GMP_PARAMS"

cd "$REPO"
[[ -f "$VENV/bin/activate" ]] || { echo "ERROR: venv not found at $VENV"; exit 1; }
source "$VENV/bin/activate"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

RUN_DIR="runs/${EXPERIMENT}/gmp/ppo_gmp_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
echo "RUN_DIR: $RUN_DIR"

PYTHONWARNINGS=ignore python main.py \
  --policy ppo \
  --experiment "$EXPERIMENT" \
  --output-dir "$RUN_DIR" \
  --num_processes "$NUM_PROCESSES" \
  --num_steps "$NUM_STEPS" \
  --cuda True \
  --intervention_type gmp \
  --intervention_params "$(python - <<PY
import json
print(json.dumps($GMP_PARAMS))
PY
)"
