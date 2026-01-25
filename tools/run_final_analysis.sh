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
OUT_DIR_BASE="${ROOT_DIR}/results/final_results"
BOOTSTRAP=10000
STATISTIC="median"
VERBOSE=""
TAG_PREFIX="train_reward_iqm/"
NUM_TASKS=3
TASK_LENGTH=500000
MIN_POINTS=1
RUNS_DIR_OVERRIDE=""
RUNS_DIRS_OVERRIDE=""

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
            RUNS_DIR_OVERRIDE="$2"
            shift 2
            ;;
        --runs-dirs)
            RUNS_DIRS_OVERRIDE="$2"
            shift 2
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
            echo "  --runs-dirs DIRS    Comma-separated runs directories"
            echo "  --out-dir DIR       Base output directory"
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

declare -a RUNS_DIRS=()
if [ -n "$RUNS_DIR_OVERRIDE" ]; then
    RUNS_DIRS=("$RUNS_DIR_OVERRIDE")
elif [ -n "$RUNS_DIRS_OVERRIDE" ]; then
    IFS=',' read -ra RUNS_DIRS <<< "$RUNS_DIRS_OVERRIDE"
else
    RUNS_DIRS=(
        "${ROOT_DIR}/runs/procgen_3_tasks_1_cycle_500k_starpilot"
        "${ROOT_DIR}/runs/whole_vs_last_layer"
    )
fi

for RUNS_DIR in "${RUNS_DIRS[@]}"; do
    if [ ! -d "$RUNS_DIR" ]; then
        echo "WARNING: Runs directory not found: $RUNS_DIR"
        continue
    fi

    RUNS_NAME="$(basename "$RUNS_DIR")"
    OUT_DIR="${OUT_DIR_BASE}/${RUNS_NAME}"

    echo "Configuration:"
    echo "  Runs directory: $RUNS_DIR"
    echo "  Output directory: $OUT_DIR"
    echo "  Bootstrap samples: $BOOTSTRAP"
    echo "  Statistic: $STATISTIC"
    echo "  Grouped plots: yes"
    echo "    Tag prefix: $TAG_PREFIX"
    echo "    Num tasks: $NUM_TASKS"
    echo "    Task length: $TASK_LENGTH"
    echo "    Min points: $MIN_POINTS"
    echo ""

    "$PYTHON" "${SCRIPT_DIR}/analyze_final_run.py" \
        --runs-dir "$RUNS_DIR" \
        --out-dir "$OUT_DIR" \
        --bootstrap "$BOOTSTRAP" \
        --statistic "$STATISTIC" \
        $VERBOSE

    # Prefer grid-search trials for ablations (they contain trial_summary.json)
    GRID_ABLATION_DIR="${ROOT_DIR}/runs/grid/${RUNS_NAME}"
    if [ -d "$GRID_ABLATION_DIR" ]; then
        ABLATION_INPUT="$GRID_ABLATION_DIR"
    else
        ABLATION_INPUT="$RUNS_DIR"
    fi
    echo ""
    echo "Generating ablation study plots... (source: $ABLATION_INPUT)"
    "$PYTHON" "${SCRIPT_DIR}/ablation_plots.py" \
        --runs-dir "$ABLATION_INPUT" \
        --out-dir "${OUT_DIR_BASE}/ablations" \
        --task-length 500000 \
        --num-tasks 3 \
        --grid-step 50000 \
        --min-points "$MIN_POINTS" \
        --bootstrap "$BOOTSTRAP" \
        --statistic "$STATISTIC" \
        --formats png \
        $VERBOSE
    
    # (config comparison plots removed)
done

echo ""
echo "=================================="
echo "Analysis complete!"
echo "Results saved under: $OUT_DIR_BASE"
echo "=================================="
