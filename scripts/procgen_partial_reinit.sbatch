#!/usr/bin/env bash
#SBATCH --job-name=fruitbot-tiny-partialreinit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --chdir=/scratch/%u/cora-procgen-plasticity
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO="/scratch/$USER/cora-procgen-plasticity"
VENV="/scratch/$USER/venvs/cora"

echo "HOSTNAME: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}"
echo "REPO: $REPO"

cd "$REPO"
source "$VENV/bin/activate"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

python - <<'PY'
import sys, torch
print("python:", sys.executable)
print("torch:", torch.__version__, "torch.cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device_count:", torch.cuda.device_count())
    print("device0:", torch.cuda.get_device_name(0))
PY

RUN_DIR="runs/procgen_fruitbot_tiny/partial_reinit/ppo_partial_reinit_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

PYTHONWARNINGS=ignore python main.py \
  --policy ppo \
  --experiment procgen_fruitbot_tiny \
  --output-dir "$RUN_DIR" \
  --num_processes 1 \
  --num_steps 64 \
  --cuda True \
  --intervention_type partial_reinit \
  --intervention_params "{}"
