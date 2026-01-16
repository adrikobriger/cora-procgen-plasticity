#!/usr/bin/env bash
#SBATCH --job-name=tune-interventions-calibrate
#SBATCH --partition=digitallab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=0-00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# ------------------------------------------------------------
# Minimal calibration job to:
#  - sanity-check multi-trial + multi-seed wiring
#  - get a walltime-per-(trial,seed,budget) estimate
#  - optionally exercise both SET and GMP in one job
#
# Override via sbatch --export=ALL,VAR=...,
# e.g.:
#   sbatch --export=ALL,METHODS=set,TRIALS=2,TRIAL_SEEDS=0,1,BUDGET_OVERRIDE=2000 tune_interventions_calibrate.sbatch
# ------------------------------------------------------------

REPO="${REPO:-/scratch/s5579783/cora-procgen-plasticity}"
VENV="${VENV:-/scratch/s5579783/venvs/cora}"

EXPERIMENT="${EXPERIMENT:-procgen_3_tasks_1_cycle_10k_debug}"
METHODS="${METHODS:-set,gmp}"

TRIALS="${TRIALS:-2}"
TRIAL_SEEDS="${TRIAL_SEEDS:-0,1}"
SEARCH="${SEARCH:-random}"
BUDGET_OVERRIDE="${BUDGET_OVERRIDE:-2000}"

# Make this as fast as possible while still exercising the full evaluation/forgetting pipeline.
EVAL_MODE="${EVAL_MODE:-none}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-1}"
SNAPSHOT_EPISODES_PER_TASK="${SNAPSHOT_EPISODES_PER_TASK:-1}"
FORGETTING_EPISODES_PER_TASK="${FORGETTING_EPISODES_PER_TASK:-1}"

# Retention-aligned defaults
OBJECTIVE="${OBJECTIVE:-composite}"
PRIMARY_METRIC="${PRIMARY_METRIC:-iqm}"
FINAL_EVAL_SET="${FINAL_EVAL_SET:-train}"

NUM_PROCESSES="${NUM_PROCESSES:-16}"
CANDIDATE_SEED="${CANDIDATE_SEED:-0}"
PPO_CONFIG="${PPO_CONFIG:-configs/procgen/tuned_ppo_hyperparams.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/tuning_calibrate}"

# Optional: if set, we print a simple linear extrapolation estimate.
TARGET_BUDGET="${TARGET_BUDGET:-100000}"
TARGET_TRIALS="${TARGET_TRIALS:-20}"
TARGET_SEEDS="${TARGET_SEEDS:-3}"

mkdir -p "$REPO/logs"
cd "$REPO"
source "$VENV/bin/activate"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Intervention Hyperparameter Tuning (CALIBRATION)"
echo "HOSTNAME: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-local}"
echo "REPO: $REPO"
echo "------------------------------------------------------------"
echo "EXPERIMENT: $EXPERIMENT"
echo "METHODS: $METHODS"
echo "TRIALS: $TRIALS"
echo "TRIAL_SEEDS: $TRIAL_SEEDS"
echo "SEARCH: $SEARCH"
echo "BUDGET_OVERRIDE: $BUDGET_OVERRIDE"
echo "EVAL_MODE: $EVAL_MODE"
echo "EPISODES_PER_TASK: $EPISODES_PER_TASK"
echo "SNAPSHOT_EPISODES_PER_TASK: $SNAPSHOT_EPISODES_PER_TASK"
echo "FORGETTING_EPISODES_PER_TASK: $FORGETTING_EPISODES_PER_TASK"
echo "OBJECTIVE: $OBJECTIVE | PRIMARY_METRIC: $PRIMARY_METRIC | FINAL_EVAL_SET: $FINAL_EVAL_SET"
echo "NUM_PROCESSES: $NUM_PROCESSES"
echo "CANDIDATE_SEED: $CANDIDATE_SEED"
echo "PPO_CONFIG: $PPO_CONFIG"
echo "OUTPUT_ROOT: $OUTPUT_ROOT"
echo "------------------------------------------------------------"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"

python - <<'PY'
import os, sys
import torch
print("python:", sys.executable)
print("torch:", torch.__version__, "torch.cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device_count:", torch.cuda.device_count())
    print("device0:", torch.cuda.get_device_name(0))
PY

TOTAL_START=$(date +%s)
TOTAL_ELAPSED=0

# Turn comma-separated methods into a space-separated list
METHOD_LIST="${METHODS//,/ }"

for METHOD in $METHOD_LIST; do
  echo "------------------------------------------------------------"
  echo "Starting CALIBRATION run for METHOD=$METHOD ..."
  START=$(date +%s)

  PYTHONWARNINGS=ignore python tools/tune_interventions.py \
      --experiment "$EXPERIMENT" \
      --method "$METHOD" \
      --ppo_config "$PPO_CONFIG" \
      --trials "$TRIALS" \
      --trial_seeds "$TRIAL_SEEDS" \
      --search "$SEARCH" \
      --budget_override "$BUDGET_OVERRIDE" \
      --eval_mode "$EVAL_MODE" \
      --episodes_per_task "$EPISODES_PER_TASK" \
      --snapshot_episodes_per_task "$SNAPSHOT_EPISODES_PER_TASK" \
      --forgetting_episodes_per_task "$FORGETTING_EPISODES_PER_TASK" \
      --final_eval_set "$FINAL_EVAL_SET" \
      --objective "$OBJECTIVE" \
      --primary_metric "$PRIMARY_METRIC" \
      --num_processes "$NUM_PROCESSES" \
      --seed "$CANDIDATE_SEED" \
      --save_best_k 1 \
      --output_root "$OUTPUT_ROOT"

  END=$(date +%s)
  ELAPSED=$((END-START))
  TOTAL_ELAPSED=$((TOTAL_ELAPSED+ELAPSED))

  echo "METHOD=$METHOD elapsed_seconds=$ELAPSED"
  echo "Results: $OUTPUT_ROOT/$EXPERIMENT/$METHOD/"
done

TOTAL_END=$(date +%s)
TOTAL_WALL=$((TOTAL_END-TOTAL_START))

echo "------------------------------------------------------------"
echo "CALIBRATION COMPLETE"
echo "Total elapsed_seconds=$TOTAL_WALL (sum_of_method_seconds=$TOTAL_ELAPSED)"


export CALIBRATED_WALL_SECONDS="$TOTAL_WALL"

python - <<'PY'
import os
import math


def parse_int_list(s: str):
    s = (s or "").strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]

trials = int(os.environ.get("TRIALS", "1"))
seeds = parse_int_list(os.environ.get("TRIAL_SEEDS", "0"))
seed_count = max(1, len(seeds))
budget = int(os.environ.get("BUDGET_OVERRIDE", "1"))
methods = [m for m in (os.environ.get("METHODS", "").replace(",", " ").split()) if m]
method_count = max(1, len(methods))

try:
    measured_wall = float(os.environ.get("CALIBRATED_WALL_SECONDS", "nan"))
except Exception:
    measured_wall = float("nan")

try:
    target_budget = int(os.environ.get("TARGET_BUDGET", "0"))
    target_trials = int(os.environ.get("TARGET_TRIALS", "0"))
    target_seeds = int(os.environ.get("TARGET_SEEDS", "0"))
except Exception:
    target_budget = target_trials = target_seeds = 0

print("\nScaling notes (linear approximation):")
print(f"  measured: methods={method_count}, trials={trials}, seeds={seed_count}, budget_override={budget}")
print("  estimate ≈ measured_walltime * (target_trials/trials) * (target_seeds/seeds) * (target_budget/budget_override)")
print("  (Assumes similar throughput and identical PPO settings; good enough for planning.)")

if (not math.isnan(measured_wall)) and measured_wall > 0 and target_budget > 0 and target_trials > 0 and target_seeds > 0:
    est = measured_wall
    est *= (target_trials / max(1, trials))
    est *= (target_seeds / max(1, seed_count))
    est *= (target_budget / max(1, budget))
    est_hours = est / 3600.0
    print(f"\nEstimated walltime for target: budget={target_budget}, trials={target_trials}, seeds={target_seeds} (methods={method_count})")
    print(f"  ~{est_hours:.2f} hours")
else:
    print("\nTip: this job prints METHOD elapsed_seconds and Total elapsed_seconds in the log.")
    print("Use those values with the scaling formula above.")
PY

echo "------------------------------------------------------------"
echo "Done." 
