#!/usr/bin/env bash
#SBATCH --job-name=tune-ppo-debug
#SBATCH --partition=digitallab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO="/scratch/s5579783/cora-procgen-plasticity"
VENV="/scratch/s5579783/venvs/cora"

EXPERIMENT="${EXPERIMENT:-procgen_3_tasks_1_cycle_10k_debug}"
TRIALS="${TRIALS:-3}"
SEARCH="${SEARCH:-random}"
BUDGET_OVERRIDE="${BUDGET_OVERRIDE:-5000}"
EVAL_MODE="${EVAL_MODE:-final_only}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-5}"
OBJECTIVE="${OBJECTIVE:-mean}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
SEED="${SEED:-0}"

# Setup
mkdir -p "$REPO/logs"
cd "$REPO"
source "$VENV/bin/activate"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "PPO Baseline Hyperparameter Tuning (DEBUG!):"
echo "HOSTNAME: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-local}"
echo "REPO: $REPO"
echo "------------------------------------------------------------"
echo "EXPERIMENT: $EXPERIMENT"
echo "TRIALS: $TRIALS"
echo "SEARCH: $SEARCH"
echo "BUDGET_OVERRIDE: $BUDGET_OVERRIDE"
echo "EVAL_MODE: $EVAL_MODE"
echo "EPISODES_PER_TASK: $EPISODES_PER_TASK"
echo "OBJECTIVE: $OBJECTIVE"
echo "NUM_PROCESSES: $NUM_PROCESSES"
echo "SEED: $SEED"
echo "------------------------------------------------------------"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"

# Check Python/CUDA setup
python - <<'PY'
import sys, torch
print("python:", sys.executable)
print("torch:", torch.__version__, "torch.cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device_count:", torch.cuda.device_count())
    print("device0:", torch.cuda.get_device_name(0))
PY

echo "Starting PPO tuning (DEBUG)..."

PYTHONWARNINGS=ignore python tools/tune_ppo.py \
    --experiment "$EXPERIMENT" \
    --policy ppo \
    --trials "$TRIALS" \
    --search "$SEARCH" \
    --budget_override "$BUDGET_OVERRIDE" \
    --eval_mode "$EVAL_MODE" \
    --episodes_per_task "$EPISODES_PER_TASK" \
    --objective "$OBJECTIVE" \
    --num_processes "$NUM_PROCESSES" \
    --seed "$SEED" \
    --strict_verify \
    --output_root "runs/tuning"

echo "PPO tuning (DEBUG) complete!"
echo "Check runs/tuning/$EXPERIMENT/ppo_tune/ for results"
