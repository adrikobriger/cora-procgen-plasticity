#!/usr/bin/env bash
#SBATCH --job-name=grid-search-debug-5k
#SBATCH --partition=digitallab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Usage:
#   sbatch scripts/grid_search_debug_5k.sbatch
#   EXPERIMENT=procgen_3_tasks_1_cycle_10k_debug
#   METHOD=gmp|redo|set|reset|partial_reinit|dense
#   TRIALS=5
#   PARALLEL_TRIALS=1
#   BUDGET_OVERRIDE=5000
#   TRIAL_SEEDS=0
#   NUM_PROCESSES=16
#   PPO_CONFIG=configs/procgen/tuned_ppo_hyperparams.json
#   OUTPUT_ROOT=runs/grid_debug

REPO="${REPO:-/scratch/s5579783/cora-procgen-plasticity}"
VENV="${VENV:-/scratch/s5579783/venvs/cora}"

EXPERIMENT="${EXPERIMENT:-procgen_3_tasks_1_cycle_10k_debug}"
METHOD="${METHOD:-gmp}"
TRIALS="${TRIALS:-5}"
PARALLEL_TRIALS="${PARALLEL_TRIALS:-1}"
BUDGET_OVERRIDE="${BUDGET_OVERRIDE:-5000}"
EVAL_MODE="${EVAL_MODE:-final_only}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-5}"
OBJECTIVE="${OBJECTIVE:-iqm}"
NUM_PROCESSES="${NUM_PROCESSES:-16}"
SEED="${SEED:-0}"
TRIAL_SEEDS="${TRIAL_SEEDS:-0}"
PPO_CONFIG="${PPO_CONFIG:-configs/procgen/tuned_ppo_hyperparams.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/grid_debug}"

mkdir -p "$REPO/logs"
cd "$REPO"
source "$VENV/bin/activate"

export PYTHONUNBUFFERED=1
# Fix thread oversubscription: 1 thread per process for RL env stepping
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

echo "Grid Search DEBUG (5k steps/task)"
echo "HOSTNAME: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-local}"
echo "------------------------------------------------------------"
echo "EXPERIMENT: $EXPERIMENT"
echo "METHOD: $METHOD"
echo "TRIALS: $TRIALS"
echo "PARALLEL_TRIALS: $PARALLEL_TRIALS"
echo "BUDGET_OVERRIDE: $BUDGET_OVERRIDE"
echo "EVAL_MODE: $EVAL_MODE"
echo "EPISODES_PER_TASK: $EPISODES_PER_TASK"
echo "OBJECTIVE: $OBJECTIVE"
echo "NUM_PROCESSES: $NUM_PROCESSES"
echo "SEED (candidate gen): $SEED"
echo "TRIAL_SEEDS (per candidate): $TRIAL_SEEDS"
echo "PPO_CONFIG: $PPO_CONFIG"
echo "OUTPUT_ROOT: $OUTPUT_ROOT"
echo "------------------------------------------------------------"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"

srun --cpu-bind=cores --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" \
  python tools/tune_interventions.py \
    --experiment "$EXPERIMENT" \
    --method "$METHOD" \
    --ppo_config "$PPO_CONFIG" \
    --search grid \
    --trials "$TRIALS" \
    --parallel_trials "$PARALLEL_TRIALS" \
    --budget_override "$BUDGET_OVERRIDE" \
    --eval_mode "$EVAL_MODE" \
    --episodes_per_task "$EPISODES_PER_TASK" \
    --objective "$OBJECTIVE" \
    --num_processes "$NUM_PROCESSES" \
    --seed "$SEED" \
    --trial_seeds "$TRIAL_SEEDS" \
    --output_root "$OUTPUT_ROOT"

echo "Done. Results in $OUTPUT_ROOT/$EXPERIMENT/$METHOD/"