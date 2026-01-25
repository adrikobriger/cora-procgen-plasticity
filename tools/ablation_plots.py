#!/usr/bin/env python3
"""
Generate ablation study plots with consistent styling.

Scans for trial_summary.json files and extracts hyperparameter variations to create
publication-quality ablation plots showing sensitivity to key parameters.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Add tools to path
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from result_utils import (
    find_event_files,
    load_scalars,
    get_curve,
    extract_task_avg_eval_iqm_curve,
    extract_task_avg_train_iqm_curve,
    bootstrap_ci,
    aggregate_curves_across_seeds,
)


# Publication style settings
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["figure.dpi"] = 100


def format_steps_ticker(x, pos):
    """Format environment step labels (e.g., 800k, 900k, 1M, 1.1M)."""
    if x >= 1_000_000:
        return f"{x / 1e6:.1f}M"
    elif x >= 1_000:
        return f"{x / 1e3:.0f}k"
    return f"{int(x)}"


def load_json(path: Path) -> Optional[Dict]:
    """Safely load JSON file."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _task_avg_from_matching_substrs(
    scalars: Dict[str, List[tuple]],
    include_substrs: List[str],
    required_substrs: Optional[List[str]] = None,
) -> Optional[tuple]:
    """Average across tags that match substrings (used for loose tag formats)."""
    required_substrs = required_substrs or []
    candidate_tags = [
        t for t in scalars.keys()
        if any(s in t.lower() for s in include_substrs)
        and all(r in t.lower() for r in required_substrs)
    ]
    if not candidate_tags:
        return None

    step_vals = defaultdict(list)
    for tag in candidate_tags:
        for step, val in scalars[tag]:
            step_vals[int(step)].append(float(val))

    if not step_vals:
        return None

    steps = sorted(step_vals.keys())
    vals = [float(np.mean(step_vals[s])) for s in steps]
    return steps, vals


def _extract_curve_with_fallbacks(
    scalars: Dict[str, List[tuple]],
    primary: Optional[tuple],
    fallback_tags: List[str],
    fallback_substrs: List[str],
    min_points: int,
) -> Optional[tuple]:
    """Return primary curve if valid, else try exact tags, then substring matches."""
    if primary and len(primary[0]) >= min_points:
        return primary
    for tag in fallback_tags:
        curve = get_curve(scalars, tag)
        if curve and len(curve[0]) >= min_points:
            return curve
    curve = _task_avg_from_matching_substrs(
        scalars,
        fallback_substrs,
        required_substrs=["iqm"],
    )
    if curve and len(curve[0]) >= min_points:
        return curve
    return None


def collect_trial_summaries(
    runs_dir: Path,
    min_points: int = 5
) -> Dict[str, Dict[str, Dict]]:
    """
    Scan for trial_summary.json and collect configs by method.
    
    Returns:
        method -> parameter_value -> {
            'params': dict,
            'seeds': dict (seed_id -> (steps, values))
        }
    """
    results = defaultdict(lambda: defaultdict(lambda: {"params": {}, "seeds": {}}))
    
    for summary_path in runs_dir.rglob("trial_summary.json"):
        data = load_json(summary_path)
        if not data:
            continue
        
        method = data.get("method", "").lower()
        params = data.get("params", {})
        
        if not method:
            # Try to infer from path
            path_str = str(summary_path).lower()
            if "gmp" in path_str:
                method = "gmp"
            elif "set" in path_str:
                method = "set"
            elif "redo" in path_str:
                method = "redo"
            else:
                continue
        
        # Normalize method name
        method_display = {
            "gmp": "GMP",
            "set": "SET",
            "redo": "ReDo",
        }.get(method, method.upper())
        
        trial_dir = summary_path.parent
        
        # Use trial directory as config key
        config_key = str(trial_dir.relative_to(runs_dir)) if runs_dir in trial_dir.parents else str(trial_dir)
        config_entry = results[method_display][config_key]
        config_entry["params"] = params
        config_entry.setdefault("seeds_eval", {})
        config_entry.setdefault("seeds_train", {})
        
        # Collect seed data
        seed_results = data.get("seed_results", [])
        seed_dirs = []
        
        if seed_results:
            for s in seed_results:
                seed_path = s.get("seed_dir")
                if seed_path:
                    seed_dirs.append(Path(seed_path))
        else:
            # Auto-find seed directories
            seed_dirs = [
                d for d in trial_dir.iterdir()
                if d.is_dir() and "seed" in d.name
            ]
        
        for seed_path in seed_dirs:
            # Resolve path if relative
            if not seed_path.is_absolute():
                candidate_paths = [
                    runs_dir / seed_path,
                    runs_dir.parent / seed_path,
                    runs_dir.parent.parent / seed_path,
                    Path.cwd() / seed_path,
                ]
                resolved = next((p for p in candidate_paths if p.exists()), None)
                if resolved is not None:
                    seed_path = resolved
            
            if not seed_path.exists():
                continue
            
            # Find event files
            event_files = find_event_files(str(seed_path))
            if not event_files:
                continue
            
            # Load scalars
            seed_scalars = {}
            for event_file in event_files:
                try:
                    scalars = load_scalars(event_file)
                    seed_scalars.update(scalars)
                except Exception:
                    continue
            
            # Extract eval/train curves (task-averaged, fallback to aggregate tags)
            eval_curve = extract_task_avg_eval_iqm_curve(seed_scalars)
            eval_curve = _extract_curve_with_fallbacks(
                seed_scalars,
                eval_curve,
                ["eval_reward_iqm", "eval_iqm", "eval/iqm"],
                ["eval_reward_iqm", "eval_iqm", "eval/iqm", "eval"],
                min_points,
            )
            if eval_curve:
                steps, values = eval_curve
                seed_id = seed_path.name
                config_entry["seeds_eval"][seed_id] = (np.array(steps), np.array(values))

            train_curve = extract_task_avg_train_iqm_curve(seed_scalars)
            train_curve = _extract_curve_with_fallbacks(
                seed_scalars,
                train_curve,
                ["train_reward_iqm", "train_iqm", "train/iqm"],
                ["train_reward_iqm", "train_iqm", "train/iqm", "train"],
                min_points,
            )
            if train_curve:
                steps, values = train_curve
                seed_id = seed_path.name
                config_entry["seeds_train"][seed_id] = (np.array(steps), np.array(values))
    
    return dict(results)


def extract_varying_param(
    configs: Dict[str, Dict],
    param_name: str,
    seed_key: str = "seeds_eval",
    allowed_values: Optional[List[object]] = None,
) -> Dict[str, Dict]:
    """
    Extract a subset of configs that vary a single parameter.
    
    Returns:
        param_value_str -> {'params': dict, 'seeds': dict, 'param_value': float}
    """
    result = {}
    for config_key, config in configs.items():
        params = config.get("params", {})
        if param_name not in params:
            continue
        
        param_val = params[param_name]
        if allowed_values is not None and not _value_in_allowed(param_val, allowed_values):
            continue
        label = _format_param_label(param_name, param_val)
        
        result[label] = {
            "params": params,
            "seeds": config.get(seed_key, {}),
            "param_value": param_val,
        }
    
    # Sort by numeric value
    try:
        result = dict(sorted(
            result.items(),
            key=lambda x: float(x[1]["param_value"]) if isinstance(x[1]["param_value"], (int, float)) else 0
        ))
    except Exception:
        pass
    
    return result


def extract_varying_param_aliases(
    configs: Dict[str, Dict],
    param_names: List[str],
    seed_key: str = "seeds_eval",
    allowed_values: Optional[List[object]] = None,
) -> Dict[str, Dict]:
    """
    Try multiple parameter names and return the first non-empty result.
    """
    for name in param_names:
        subset = extract_varying_param(configs, name, seed_key=seed_key, allowed_values=allowed_values)
        if subset:
            return subset
    return {}


def _value_in_allowed(value, allowed_values: List[object]) -> bool:
    """Check if a value is in allowed_values with numeric tolerance when possible."""
    try:
        value_num = float(value)
    except Exception:
        value_num = None

    for allowed in allowed_values:
        if value_num is not None:
            try:
                allowed_num = float(allowed)
                if abs(value_num - allowed_num) < 1e-9:
                    return True
                continue
            except Exception:
                pass
        if str(value) == str(allowed):
            return True
    return False


def _format_param_label(param_name: str, param_val) -> str:
    """Format a single parameter as a label."""
    if isinstance(param_val, float):
        if param_val == int(param_val):
            return f"{param_name.replace('_', ' ').title()}: {int(param_val)}"
        else:
            formatted = f"{param_val:.3f}".rstrip("0").rstrip(".")
            return f"{param_name.replace('_', ' ').title()}: {formatted}"
    else:
        return f"{param_name.replace('_', ' ').title()}: {param_val}"


def plot_ablation(
    title: str,
    filename: str,
    configs: Dict[str, Dict],
    out_dir: Path,
    highlight_label: Optional[str] = None,
    task_length: int = 500_000,
    num_tasks: int = 3,
    grid_step: int = 50_000,
    formats: List[str] = ["png"],
    bootstrap: int = 2000,
    alpha: float = 0.05,
    statistic: str = "median",
) -> None:
    """
    Plot ablation study for varying configurations.
    
    Args:
        title: Plot title
        filename: Output filename (without extension)
        configs: param_value_str -> {'params': dict, 'seeds': dict}
        out_dir: Output directory
        highlight_label: Label to highlight (usually best config)
        task_length: Steps per task for x-axis divisions
        num_tasks: Number of tasks
        grid_step: Step size for interpolation
        formats: Output formats
    """
    if not configs:
        print(f"  Skipping {filename} (no data)")
        return
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(11, 6))
    
    labels = sorted(configs.keys())
    
    # Move highlight to end if present
    if highlight_label and highlight_label in labels:
        labels.remove(highlight_label)
        labels.append(highlight_label)
    
    # Color mapping
    n_configs = len(labels)
    if highlight_label:
        # Use purples for normal, green for best
        base_colors = plt.cm.Purples(np.linspace(0.4, 0.8, n_configs - 1))
        colors = {labels[i]: base_colors[i] for i in range(n_configs - 1)}
        colors[highlight_label] = "#2CA02C"  # Green
    else:
        colors = {label: c for label, c in zip(labels, plt.cm.tab10(np.linspace(0, 1, n_configs)))}
    
    # Initialize grid to handle case where all curves fail to plot
    grid = np.array([])
    
    plotted_any = False
    for label in labels:
        config = configs[label]
        seeds = config.get("seeds", {})
        
        if not seeds:
            continue
        
        # Prepare seed data (list of (steps, values))
        seed_curves = []
        for _seed_id, (steps, values) in seeds.items():
            seed_curves.append((steps, values))

        try:
            aggregated = aggregate_curves_across_seeds(
                seed_curves,
                n_bootstrap=bootstrap,
                alpha=alpha,
                seed=hash(label) % 10000,
                statistic=statistic,
            )
        except Exception as e:
            print(f"  Warning: Could not plot {label}: {e}")
            continue

        grid = np.array(aggregated["steps"], dtype=np.int64)
        mean = np.array(aggregated["central"], dtype=np.float64)
        lower = np.array(aggregated["ci_low"], dtype=np.float64)
        upper = np.array(aggregated["ci_high"], dtype=np.float64)
        
        # Plot styling
        is_best = label == highlight_label
        linewidth = 3.0 if is_best else 2.0
        zorder = 10 if is_best else 2
        alpha_fill = 0.2 if is_best else 0.15
        
        display_label = f"{label} (Best)" if is_best else label
        color = colors[label]
        
        ax.plot(grid, mean, label=display_label, color=color, linewidth=linewidth, zorder=zorder)
        ax.fill_between(grid, lower, upper, color=color, alpha=alpha_fill, zorder=zorder - 1)
        plotted_any = True
    
    # Add task divisions
    if not plotted_any:
        print(f"  Skipping {filename} (no valid curves)")
        plt.close(fig)
        return

    # Add task boundaries: vertical dashed lines at every task_length steps
    task_boundaries = [task_length * k for k in range(1, num_tasks)]
    for boundary in task_boundaries:
        ax.axvline(boundary, linestyle='--', color='black', alpha=0.5,
                  linewidth=1.5, zorder=1)
    
    # Add task labels: "Task 1", "Task 2", "Task 3"
    y_min, y_max = ax.get_ylim()
    label_y = y_min + 0.96 * (y_max - y_min)
    
    for k in range(num_tasks):
        task_center = (k + 0.5) * task_length
        ax.text(task_center, label_y, f'Task {k+1}',
               horizontalalignment='center', verticalalignment='top',
               fontsize=14, fontweight='bold', alpha=0.85,
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                        edgecolor='gray', alpha=0.85, linewidth=0.5),
               zorder=4)
    
    # Formatting
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("Environment Steps", fontsize=14)
    ax.set_ylabel("IQM Return", fontsize=14)
    
    ax.xaxis.set_major_formatter(FuncFormatter(format_steps_ticker))
    ax.legend(loc="best", fontsize=12, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    # Save
    for fmt in formats:
        out_path = out_dir / f"{filename}.{fmt}"
        plt.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {out_path}")
    
    plt.close(fig)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate ablation study plots")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Root runs directory"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/ablation_plots"),
        help="Output directory"
    )
    parser.add_argument(
        "--task-length",
        type=int,
        default=500_000,
        help="Steps per task"
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=3,
        help="Number of tasks"
    )
    parser.add_argument(
        "--grid-step",
        type=int,
        default=50_000,
        help="Interpolation grid step size"
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=2000,
        help="Bootstrap samples for CI"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="CI alpha level (0.05 => 95%% CI)"
    )
    parser.add_argument(
        "--statistic",
        type=str,
        default="median",
        choices=["mean", "median", "iqm"],
        help="Central estimate for curves"
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=1,
        help="Minimum points required in a curve"
    )
    parser.add_argument(
        "--formats",
        type=str,
        default="png",
        help="Comma-separated output formats"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    formats = [f.strip() for f in args.formats.split(",")]
    
    if args.verbose:
        print(f"Scanning {args.runs_dir} for ablation studies...")
    
    # Collect all trials
    all_configs = collect_trial_summaries(args.runs_dir, min_points=args.min_points)
    
    if not all_configs:
        print("No trial summaries found.")
        return
    
    if args.verbose:
        for method, configs in all_configs.items():
            print(f"  {method}: {len(configs)} trials")
    
    # Generate ablations per method
    for method in sorted(all_configs.keys()):
        configs = all_configs[method]
        
        if args.verbose:
            print(f"\nGenerating ablations for {method}...")
        
        # Define ablations per method
        if method == "GMP":
            # Sparsity ablation
            sparsity_values = [0.75, 0.85, 0.95]
            sparsity_train = extract_varying_param(
                configs, "final_sparsity", seed_key="seeds_train", allowed_values=sparsity_values
            )
            if sparsity_train:
                plot_ablation(
                    f"{method}: Sensitivity to Final Sparsity",
                    f"{method.lower()}_ablation_sparsity",
                    sparsity_train,
                    args.out_dir,
                    highlight_label="Final Sparsity: 0.95",
                    task_length=args.task_length,
                    num_tasks=args.num_tasks,
                    grid_step=args.grid_step,
                    formats=formats,
                    bootstrap=args.bootstrap,
                    alpha=args.alpha,
                    statistic=args.statistic,
                )
            
            # Frequency / rewire ablation
            freq_values = [10, 20, 40]
            freq_train = extract_varying_param_aliases(
                configs,
                ["prune_frequency", "prune_cycle", "update_frequency", "update_interval"],
                seed_key="seeds_train",
                allowed_values=freq_values,
            )
            if freq_train:
                plot_ablation(
                    f"{method}: Sensitivity to Rewire Frequency",
                    f"{method.lower()}_ablation_frequency",
                    freq_train,
                    args.out_dir,
                    task_length=args.task_length,
                    num_tasks=args.num_tasks,
                    grid_step=args.grid_step,
                    formats=formats,
                    bootstrap=args.bootstrap,
                    alpha=args.alpha,
                    statistic=args.statistic,
                )
        
        elif method == "SET":
            # Update interval ablation
            ui_values = [10, 20, 40]
            ui_train = extract_varying_param_aliases(
                configs,
                ["update_interval", "update_frequency"],
                seed_key="seeds_train",
                allowed_values=ui_values,
            )
            if ui_train:
                plot_ablation(
                    f"{method}: Sensitivity to Update Interval",
                    f"{method.lower()}_ablation_update_interval",
                    ui_train,
                    args.out_dir,
                    highlight_label="Update Interval: 10",
                    task_length=args.task_length,
                    num_tasks=args.num_tasks,
                    grid_step=args.grid_step,
                    formats=formats,
                    bootstrap=args.bootstrap,
                    alpha=args.alpha,
                    statistic=args.statistic,
                )
            
            # Connection percentage ablation
            conn_train = extract_varying_param(
                configs, "connection_percentage", seed_key="seeds_train"
            )
            if conn_train:
                plot_ablation(
                    f"{method}: Sensitivity to Connection Percentage",
                    f"{method.lower()}_ablation_connection",
                    conn_train,
                    args.out_dir,
                    task_length=args.task_length,
                    num_tasks=args.num_tasks,
                    grid_step=args.grid_step,
                    formats=formats,
                    bootstrap=args.bootstrap,
                    alpha=args.alpha,
                    statistic=args.statistic,
                )
        
        elif method == "ReDo":
            # Update interval ablation
            ui_values = [3000, 5000]
            ui_train = extract_varying_param_aliases(
                configs,
                ["update_interval", "update_frequency"],
                seed_key="seeds_train",
                allowed_values=ui_values,
            )
            if ui_train:
                plot_ablation(
                    f"{method}: Sensitivity to Update Interval",
                    f"{method.lower()}_ablation_update_interval",
                    ui_train,
                    args.out_dir,
                    highlight_label="Update Interval: 5000",
                    task_length=args.task_length,
                    num_tasks=args.num_tasks,
                    grid_step=args.grid_step,
                    formats=formats,
                    bootstrap=args.bootstrap,
                    alpha=args.alpha,
                    statistic=args.statistic,
                )
    
    print(f"\nAblation plots saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
