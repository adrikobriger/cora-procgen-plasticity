#!/usr/bin/env bash
#SBATCH --job-name=tune-ppo-debug
#SBATCH --partition=digitallab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=0-02:00:00
#SBATCH --output=/scratch/s5579783/cora-procgen-plasticity/logs/%x_%j.out
#SBATCH --error=/scratch/s5579783/cora-procgen-plasticity/logs/%x_%j.err

set -euo pipefail

REPO="/scratch/s5579783/cora-procgen-plasticity"
VENV="/scratch/s5579783/venvs/cora"

EXPERIMENT="${EXPERIMENT:-procgen_3_tasks_1_cycle_10k_debug}"
TRIALS="${TRIALS:-2}"
SEARCH="${SEARCH:-random}"
BUDGET_OVERRIDE="${BUDGET_OVERRIDE:-5000}"
EVAL_MODE="${EVAL_MODE:-final_only}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-5}"
OBJECTIVE="${OBJECTIVE:-iqm}"
NUM_PROCESSES="${NUM_PROCESSES:-16}"
SEED="${SEED:-0}"
SEEDS="${SEEDS:-0,1}"

mkdir -p "$REPO/logs"
cd "$REPO"
source "$VENV/bin/activate"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "PPO Baseline Hyperparameter Tuning (DEBUG!)"
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
echo "SEED (candidate gen): $SEED"
echo "SEEDS (per trial): $SEEDS"
echo "------------------------------------------------------------"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"

python - <<'PY'
import sys, torch
print("python:", sys.executable)
print("torch:", torch.__version__, "torch.cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is not available. Refusing to run on CPU.")
print("device_count:", torch.cuda.device_count())
print("device0:", torch.cuda.get_device_name(0))
PY

echo "Starting PPO tuning (DEBUG)..."

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" python tools/tune_ppo.py \
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
  --seeds "$SEEDS" \
  --strict_verify \
  --output_root "runs/tuning"

echo "PPO tuning (DEBUG) complete!"
echo "Check runs/tuning/$EXPERIMENT/ppo_tune/ for results"
