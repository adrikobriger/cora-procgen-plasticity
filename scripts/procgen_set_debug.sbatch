#!/usr/bin/env bash
#SBATCH --job-name=procgen-set-3tasks-10k-debug
#SBATCH --partition=digitallab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO="/scratch/$USER/cora-procgen-plasticity"
VENV="/scratch/$USER/venvs/cora"

EXPERIMENT="procgen_3_tasks_1_cycle_10k_debug"
NUM_PROCESSES=1
NUM_STEPS=256

SET_PARAMS='{"target_sparsity":0.8,"update_interval":50,"prune_fraction":0.1,"warmup_steps":0,"seed":0}'

echo "HOSTNAME: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}"
echo "REPO: $REPO"
echo "EXPERIMENT: $EXPERIMENT"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"

cd "$REPO"
mkdir -p logs
source "$VENV/bin/activate"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

RUN_DIR="runs/${EXPERIMENT}/set/ppo_set_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
echo "RUN_DIR: $RUN_DIR"

PYTHONWARNINGS=ignore python main.py \
  --policy ppo \
  --experiment "$EXPERIMENT" \
  --output-dir "$RUN_DIR" \
  --num_processes "$NUM_PROCESSES" \
  --num_steps "$NUM_STEPS" \
  --cuda True \
  --intervention_type set \
  --intervention_params "$SET_PARAMS"
z