# Analysis Guide: Computing IQM and 95% CI Tables

This guide explains how to generate result tables with IQM and 95% confidence intervals for your continual RL experiments.

## Quick Start

### 1. Run Multiple Seeds

First, run your experiments with different random seeds. For example:

```bash
# Run Dense baseline with 5 different seeds
for seed in {0..4}; do
    python main.py --experiment procgen_lifelong --policy ppo \
        --intervention_type dense \
        --output_dir runs/my_experiment/Dense/seed_$seed \
        --seed $seed
done

# Run ReDo intervention with 5 different seeds
for seed in {0..4}; do
    python main.py --experiment procgen_lifelong --policy ppo \
        --intervention_type redo \
        --intervention_params '{"tau": 0.10, "update_interval": 5000}' \
        --output_dir runs/my_experiment/ReDo/seed_$seed \
        --seed $seed
done

# Repeat for other interventions (SET, GMP, Reset, etc.)
```

### 2. Analyze Results

Once you have multiple runs, use the analysis script:

```bash
python tools/analyze_results.py \
    --exp_dir runs/my_experiment \
    --output results/my_results.csv \
    --latex
```

This will:
- Read TensorBoard event files from all runs
- Compute IQM per task across seeds
- Calculate 95% confidence intervals using bootstrap
- Generate CSV and LaTeX tables

## Expected Directory Structure

```
runs/my_experiment/
├── Dense/
│   ├── seed_0/
│   │   └── events.out.tfevents.*
│   ├── seed_1/
│   │   └── events.out.tfevents.*
│   └── seed_2/
│       └── events.out.tfevents.*
├── ReDo/
│   ├── seed_0/
│   │   └── events.out.tfevents.*
│   ├── seed_1/
│   │   └── events.out.tfevents.*
│   └── seed_2/
│       └── events.out.tfevents.*
└── SET/
    ├── seed_0/
    │   └── events.out.tfevents.*
    └── ...
```

## Command Options

```bash
python tools/analyze_results.py --help
```

Key options:

- `--exp_dir`: Base directory containing intervention subdirectories
- `--output`: Output CSV file path (default: `results.csv`)
- `--interventions`: Specific interventions to analyze (default: all subdirectories)
- `--use_stored_iqm`: Use IQM values from TensorBoard (if already logged)
- `--ci_method`: `bootstrap` (default) or `parametric`
- `--task_order`: Specify order of tasks in table (e.g., `task_0 task_1 task_2`)
- `--latex`: Also generate LaTeX formatted table

## Example Output

### CSV Format
```csv
Agent Treatment,Task 1 IQM,Task 1 95% CI,Task 2 IQM,Task 2 95% CI,Task 3 IQM,Task 3 95% CI
Dense,0.70,"(0.60, 0.76)",0.65,"(0.61, 0.71)",0.72,"(0.69, 0.75)"
ReDo,0.74,"(0.68, 0.77)",0.83,"(0.78, 0.84)",0.80,"(0.76, 0.83)"
SET,0.76,"(0.74, 0.77)",0.80,"(0.75, 0.84)",0.80,"(0.77, 0.84)"
```

### LaTeX Format
Automatically generates publication-ready LaTeX tables.

## What Gets Logged to TensorBoard

The codebase automatically logs these metrics to TensorBoard for each task:

### During Training
- `train_reward/{task_id}`: Mean episode return
- `train_reward_iqm/{task_id}`: IQM of episode returns

### During Evaluation
- `eval_reward/{task_id}`: Mean episode return  
- `eval_reward_iqm/{task_id}`: IQM of episode returns

The analysis script reads the **final value** of `eval_reward_iqm/{task_id}` from each run.

## Confidence Interval Methods

### Bootstrap (Recommended)
```bash
python tools/analyze_results.py --exp_dir runs/my_exp --ci_method bootstrap
```

- Non-parametric, makes no distributional assumptions
- Uses 10,000 bootstrap samples by default
- More robust for small sample sizes

### Parametric (T-distribution)
```bash
python tools/analyze_results.py --exp_dir runs/my_exp --ci_method parametric
```

- Assumes normal distribution of IQMs across runs
- Uses t-distribution with n-1 degrees of freedom
- Faster but less robust

## Customization

### Specify Task Order
If tasks have specific names or ordering:

```bash
python tools/analyze_results.py \
    --exp_dir runs/my_exp \
    --task_order bigfish bossfight caveflyer coinrun
```

### Analyze Specific Interventions
```bash
python tools/analyze_results.py \
    --exp_dir runs/my_exp \
    --interventions Dense ReDo SET GMP
```

## Troubleshooting

### "No results found!"
- Check that `--exp_dir` points to correct location
- Verify event files exist: `ls runs/my_exp/*/*/events.out.tfevents.*`
- Ensure runs completed successfully (check TensorBoard)

### "Warning: Directory not found"
- Make sure intervention names match directory names exactly
- Use `--interventions` to specify exact names

### Missing Tasks
- Some runs may not have completed all tasks
- Analysis script will show N/A for missing data
- Check logs to see which runs failed

### Different Number of Runs per Intervention
- Script handles uneven sample sizes
- CI will be wider for interventions with fewer runs
- Aim for at least 3-5 runs per intervention for reliable CIs

## Integration with Hyperparameter Tuning

If using the tuning scripts (`tune_ppo.py` or `tune_interventions.py`):

1. After tuning, select best hyperparameters
2. Run multiple confirmation trials with different seeds
3. Use the confirmation run directories as input to `analyze_results.py`

Example:
```bash
# Step 1: Tune (finds best params)
python tools/tune_interventions.py --experiment procgen_lifelong --method redo --trials 50

# Step 2: Confirm with multiple seeds (put in dedicated directory)
for seed in {0..4}; do
    python main.py --experiment procgen_lifelong --policy ppo \
        --intervention_type redo \
        --intervention_params '{"tau": 0.08, "update_interval": 4000}' \
        --output_dir runs/final_results/ReDo/seed_$seed \
        --seed $seed
done

# Step 3: Analyze
python tools/analyze_results.py --exp_dir runs/final_results --output final_table.csv
```

## Advanced: Adding More Metrics

To add additional metrics (e.g., max return, standard deviation), modify:
1. `tools/analyze_results.py` - Add computation logic
2. Update `format_table()` to include new columns
3. Optionally modify `task_base.py` to log additional values to TensorBoard
