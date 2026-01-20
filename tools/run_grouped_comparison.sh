#!/bin/bash
# Quick script to generate grouped comparison plots from final run data

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=================================="
echo "Grouped Comparison Plots"
echo "=================================="
echo ""

# Configuration
RUNS_DIR="${ROOT_DIR}/runs/procgen_3_tasks_1_cycle_500k_starpilot"
OUT_DIR="${ROOT_DIR}/results/final_results"
TAG_PREFIX="eval_reward_iqm/"
NUM_TASKS=3
TASK_LENGTH=500000
MIN_POINTS=1

# Check if runs directory exists
if [ ! -d "$RUNS_DIR" ]; then
    echo "ERROR: Runs directory not found: $RUNS_DIR"
    exit 1
fi

echo "Configuration:"
echo "  Runs directory: $RUNS_DIR"
echo "  Output directory: $OUT_DIR"
echo "  Tag prefix: $TAG_PREFIX"
echo "  Number of tasks: $NUM_TASKS"
echo "  Task length: $TASK_LENGTH steps"
echo ""

# Run the plotting script
python3 "${SCRIPT_DIR}/plot_iqm_return.py" \
    --runs_dir "$RUNS_DIR" \
    --out_dir "$OUT_DIR" \
    --legacy_average \
    --grouped_comparisons \
    --tag_prefix "$TAG_PREFIX" \
    --num_tasks "$NUM_TASKS" \
    --task_length "$TASK_LENGTH" \
    --min_points "$MIN_POINTS" \
    --formats both

echo ""
echo "=================================="
echo "Grouped comparison plots created!"
echo ""
echo "Results organized in: ${OUT_DIR}/"
echo ""
echo "Directory structure:"
echo "  individual_interventions/"
echo "    Dense PPO/, GMP/, SET/, etc."
echo "      - iqm_return_ci.png/.pdf (with shared y-axis)"
echo "  grouped_comparisons/"
echo "    - sparse_methods.png/.pdf"
echo "    - reset_based_methods.png/.pdf"
echo "    - baselines.png/.pdf"
echo "=================================="
