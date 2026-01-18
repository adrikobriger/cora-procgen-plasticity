#!/usr/bin/env bash
#SBATCH --job-name=procgen-set-3t1c-1m
#SBATCH --partition=digitallab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO="/scratch/s5579783/cora-procgen-plasticity"
VENV="/scratch/s5579783/venvs/cora"

EXPERIMENT="procgen_3_tasks_1_cycle_1M"
NUM_PROCESSES=16
NUM_STEPS=256

INTERVENTION_CONFIG="configs/procgen/tuned_set_hyperparams.json"
SET_PARAMS="$(cat "$INTERVENTION_CONFIG")"

mkdir -p "$REPO/logs"
cd "$REPO"
source "$VENV/bin/activate"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "HOSTNAME: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}"
echo "REPO: $REPO"
echo "EXPERIMENT: $EXPERIMENT"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"
echo "INTERVENTION_CONFIG: $INTERVENTION_CONFIG"

RUN_DIR="runs/${EXPERIMENT}/set/ppo_set_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
echo "RUN_DIR: $RUN_DIR"

PYTHONWARNINGS=ignore python main.py \
  --policy ppo \
  --experiment "$EXPERIMENT" \
  --output-dir "$RUN_DIR" \
  --load-config "configs/procgen/tuned_ppo_hyperparams.json" \
  --num_processes "$NUM_PROCESSES" \
  --num_steps "$NUM_STEPS" \
  --cuda True \
  --intervention_type set \
  --intervention_params "$SET_PARAMS"