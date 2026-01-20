#!/usr/bin/env bash
#SBATCH --job-name=grid-gmp-fs0p9
#SBATCH --partition=digitallab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

METHOD="gmp"
PARAMS='{"final_sparsity":0.9}'
NUM_PROCESSES="${NUM_PROCESSES:-6}"
TRIAL_SEEDS="${TRIAL_SEEDS:-0}"

source scripts/grid/_grid_runner.sh
