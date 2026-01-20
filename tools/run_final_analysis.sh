#!/bin/bash
# Run the final run analysis with appropriate environment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=================================="
echo "Final Run Analysis"
echo "=================================="
echo ""

# Check if we're in a conda/virtual environment
if [ -z "$CONDA_DEFAULT_ENV" ] && [ -z "$VIRTUAL_ENV" ]; then
    echo "WARNING: No conda or virtual environment detected."
    echo "Please activate your environment first:"
    echo "  conda activate venv_continual_rl"
    echo "or"
    echo "  source venv/bin/activate"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Default arguments
RUNS_DIR="${ROOT_DIR}/runs/procgen_3_tasks_1_cycle_500k_starpilot"
OUT_DIR="${ROOT_DIR}/results/final_results"
BOOTSTRAP=10000
STATISTIC="median"
VERBOSE=""

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
echo ""

# Run the analysis
python3 "${SCRIPT_DIR}/analyze_final_run.py" \
    --runs-dir "$RUNS_DIR" \
    --out-dir "$OUT_DIR" \
    --bootstrap "$BOOTSTRAP" \
    --statistic "$STATISTIC" \
    $VERBOSE

echo ""
echo "=================================="
echo "Analysis complete!"
echo "Results saved to: $OUT_DIR"
echo "=================================="
