#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/scratch/s5579783/cora-procgen-plasticity}"
VENV="${VENV:-/scratch/s5579783/venvs/cora}"
EXPERIMENT="${EXPERIMENT:-procgen_3_tasks_1_cycle_500k_starpilot}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
TRIAL_SEEDS="${TRIAL_SEEDS:-0,1,2}"
if [[ "$TRIAL_SEEDS" == "0" ]]; then
  TRIAL_SEEDS="0,1,2"
fi
PPO_CONFIG="${PPO_CONFIG:-configs/procgen/tuned_ppo_hyperparams.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/grid}"

if [[ -z "${METHOD:-}" ]]; then
  echo "METHOD is required"
  exit 1
fi
if [[ -z "${PARAMS:-}" ]]; then
  echo "PARAMS is required"
  exit 1
fi

mkdir -p "$REPO/logs"
cd "$REPO"
source "$VENV/bin/activate"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

echo "Grid intervention run"
echo "EXPERIMENT: $EXPERIMENT"
echo "METHOD: $METHOD"
echo "PARAMS: $PARAMS"
echo "NUM_PROCESSES: $NUM_PROCESSES"
echo "TRIAL_SEEDS: $TRIAL_SEEDS"

echo "Starting grid run..."

srun --cpu-bind=cores --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" \
  python tools/tune_interventions.py \
    --experiment "$EXPERIMENT" \
    --method "$METHOD" \
    --ppo_config "$PPO_CONFIG" \
    --search grid \
    --trials 1 \
    --num_processes "$NUM_PROCESSES" \
    --trial_seeds "$TRIAL_SEEDS" \
    --objective iqm \
    --params_inline "$PARAMS" \
    --output_root "$OUTPUT_ROOT"

echo "Grid run complete!"