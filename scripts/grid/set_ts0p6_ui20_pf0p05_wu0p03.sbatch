#!/usr/bin/env bash
#SBATCH --job-name=grid-set-ts0p6-ui20-pf0p05-wu0p03
#SBATCH --partition=digitallab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

METHOD="set"
PARAMS='{"target_sparsity":0.6,"update_interval":20,"prune_fraction":0.05,"warmup_steps":0.03}'
NUM_PROCESSES="${NUM_PROCESSES:-8}"
TRIAL_SEEDS="${TRIAL_SEEDS:-0}"

source scripts/grid/_grid_runner.sh
