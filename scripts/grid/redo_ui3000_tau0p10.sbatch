#!/usr/bin/env bash
#SBATCH --job-name=grid-redo-ui3000-tau0p10
#SBATCH --partition=digitallab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

METHOD="redo"
PARAMS='{"update_interval":3000,"tau":0.10}'
NUM_PROCESSES="${NUM_PROCESSES:-8}"
TRIAL_SEEDS="${TRIAL_SEEDS:-0,1,2}"

source scripts/grid/_grid_runner.sh
