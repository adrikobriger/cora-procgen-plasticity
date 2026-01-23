#!/bin/bash
# Run the final run analysis with appropriate environment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=================================="
echo "Final Run Analysis + Plots"
echo "=================================="
echo ""

# Resolve Python (prefer local venv if present)
PYTHON=""
if [ -x "${ROOT_DIR}/.venv/bin/python" ]; then
    PYTHON="${ROOT_DIR}/.venv/bin/python"
elif [ -x "${ROOT_DIR}/.venv/Scripts/python.exe" ]; then
    PYTHON="${ROOT_DIR}/.venv/Scripts/python.exe"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON="python"
else
    echo "ERROR: Python not found. Install Python or create .venv in the repo root."
    exit 1
fi

# Default arguments
RUNS_DIR="${ROOT_DIR}/runs/procgen_3_tasks_1_cycle_500k_starpilot"
OUT_DIR="${ROOT_DIR}/results/final_results"
BOOTSTRAP=10000
STATISTIC="median"
VERBOSE=""
GROUPED="false"
TAG_PREFIX="train_reward_iqm/"
NUM_TASKS=3
TASK_LENGTH=500000
MIN_POINTS=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose|-v)
            VERBOSE="--verbose"
            shift
            ;;
        --quick)
            BOOTSTRAP=1000
            echo "Quick mode: using 1000 bootstrap samples"
            shift
            ;;
        --runs-dir)
            RUNS_DIR="$2"
            shift 2
            ;;
        --grouped)
            GROUPED="true"
            shift
            ;;
        --tag-prefix)
            TAG_PREFIX="$2"
            shift 2
            ;;
        --num-tasks)
            NUM_TASKS="$2"
            shift 2
            ;;
        --task-length)
            TASK_LENGTH="$2"
            shift 2
            ;;
        --min-points)
            MIN_POINTS="$2"
            shift 2
            ;;
        --out-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --verbose, -v       Enable verbose output"
            echo "  --quick             Use fewer bootstrap samples (faster)"
            echo "  --runs-dir DIR      Specify runs directory"
            echo "  --out-dir DIR       Specify output directory"
            echo "  --grouped           Also generate grouped comparison plots"
            echo "  --tag-prefix STR    Tag prefix for grouped plots (default: train_reward_iqm/)"
            echo "  --num-tasks N       Number of tasks for grouped plots"
            echo "  --task-length N     Task length in steps for grouped plots"
            echo "  --min-points N      Min points for grouped plots"
            echo "  --help, -h          Show this help message"
            echo ""
            echo "Default runs dir: ${RUNS_DIR}"
            echo "Default output dir: ${OUT_DIR}"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if runs directory exists
if [ ! -d "$RUNS_DIR" ]; then
    echo "ERROR: Runs directory not found: $RUNS_DIR"
    exit 1
fi

echo "Configuration:"
echo "  Runs directory: $RUNS_DIR"
echo "  Output directory: $OUT_DIR"
echo "  Bootstrap samples: $BOOTSTRAP"
echo "  Statistic: $STATISTIC"
echo "  Grouped plots: $GROUPED"
if [ "$GROUPED" = "true" ]; then
    echo "    Tag prefix: $TAG_PREFIX"
    echo "    Num tasks: $NUM_TASKS"
    echo "    Task length: $TASK_LENGTH"
    echo "    Min points: $MIN_POINTS"
fi
echo ""

# Run the analysis
"$PYTHON" "${SCRIPT_DIR}/analyze_final_run.py" \
    --runs-dir "$RUNS_DIR" \
    --out-dir "$OUT_DIR" \
    --bootstrap "$BOOTSTRAP" \
    --statistic "$STATISTIC" \
    $VERBOSE

# Optional grouped comparison plots
if [ "$GROUPED" = "true" ]; then
    "$PYTHON" "${SCRIPT_DIR}/plot_iqm_return.py" \
        --runs_dir "$RUNS_DIR" \
        --out_dir "$OUT_DIR" \
        --legacy_average \
        --grouped_comparisons \
        --tag_prefix "$TAG_PREFIX" \
        --num_tasks "$NUM_TASKS" \
        --task_length "$TASK_LENGTH" \
        --min_points "$MIN_POINTS" \
        --formats both
fi

echo ""
echo "=================================="
echo "Analysis complete!"
echo "Results saved to: $OUT_DIR"
echo "=================================="
