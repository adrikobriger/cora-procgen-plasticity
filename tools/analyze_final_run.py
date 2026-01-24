#!/usr/bin/env python3
"""
Analyze final run results for procgen_3_tasks_1_cycle_500k_starpilot.

Creates:
  1. Individual plots per intervention (IQM return with 95% CI)
  2. Combined plot with all interventions on one graph
  3. CSV table with metrics for all interventions

Outputs saved to results/final_results/
"""

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Add tools directory to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from result_utils import (
    find_event_files,
    infer_group_and_seed,
    load_scalars,
    get_curve,
    extract_task_avg_eval_iqm_curve,
    extract_task_avg_dormant_frac_curve,
    bootstrap_ci,
    aggregate_curves_across_seeds,
)


# Intervention name mapping to clean display names
INTERVENTION_NAMES = {
    "dense": "Dense PPO",
    "gmp": "GMP",
    "partial_reinit": "Partial Reinit",
    "redo": "ReDo",
    "reset": "Reset",
    "set": "SET",
}

# Color scheme for interventions
INTERVENTION_COLORS = {
    "dense": "#1f77b4",      # blue
    "gmp": "#ff7f0e",        # orange
    "partial_reinit": "#2ca02c",  # green
    "redo": "#d62728",       # red
    "reset": "#9467bd",      # purple
    "set": "#8c564b",        # brown
}


def choose_best_event_file(files: List[str]) -> str:
    """
    Pick the most reliable event file for a run:
    - prefer larger size (more data written)
    - then newer modified time
    """
    def score(fp: str):
        try:
            return (os.path.getsize(fp), os.path.getmtime(fp))
        except OSError:
            return (0, 0)

    return sorted(files, key=score, reverse=True)[0]


def extract_run_metrics(
    scalars: Dict[str, List[Tuple[int, float]]],
    last_k_rank: int = 10,
) -> Dict[str, Any]:
    """
    Extract per-seed metrics from one TensorBoard run (one event file).
    
    Returns dict with per-seed summary scalars and curves for aggregation.
    """

    # ---- task-avg eval IQM curve ----
    eval_curve = extract_task_avg_eval_iqm_curve(scalars)
    if eval_curve is not None:
        eval_steps, eval_vals = eval_curve
        final_iqm = float(eval_vals[-1])  # LAST value
        peak_iqm = float(np.nanmax(eval_vals))  # MAX value
    else:
        eval_steps, eval_vals = None, None
        final_iqm = float("nan")
        peak_iqm = float("nan")

    # ---- forgetting curve + MAX ----
    # Try both possible tag names for forgetting
    forget_curve = get_curve(scalars, "forgetting/isolated_avg_iqm")
    if forget_curve is None:
        forget_curve = get_curve(scalars, "forgetting/isolated_avg_mean")
    if forget_curve is None:
        forget_curve = get_curve(scalars, "forgetting/isolated_avg")
    
    if forget_curve is not None:
        f_steps, f_vals = forget_curve
        max_forgetting = float(np.nanmax(f_vals))  # MAX value
    else:
        f_steps, f_vals = None, None
        max_forgetting = float("nan")

    # ---- dormant frac (task-averaged) ----
    dorm_curve = extract_task_avg_dormant_frac_curve(scalars)
    if dorm_curve is not None:
        d_steps, d_vals = dorm_curve
        final_dormant = float(d_vals[-1])  # LAST value
        peak_dormant = float(np.nanmax(d_vals))  # MAX value
    else:
        d_steps, d_vals = None, None
        final_dormant = float("nan")
        peak_dormant = float("nan")

    # ---- effective rank: AVERAGE of last K values (robust) ----
    # Try both possible tag names for effective rank
    er_curve = get_curve(scalars, "effective_rank/across_tasks_avg")
    if er_curve is None:
        er_curve = get_curve(scalars, "effective_rank/avg")
    
    if er_curve is not None:
        er_steps, er_vals = er_curve
        # Take last K values (or all if fewer than K)
        tail = er_vals[-last_k_rank:] if len(er_vals) >= last_k_rank else er_vals
        final_er = float(np.nanmean(tail)) if len(tail) > 0 else float("nan")
    else:
        er_steps, er_vals = None, None
        final_er = float("nan")

    return {
        "final_iqm_return": final_iqm,
        "peak_iqm_return": peak_iqm,
        "max_isolated_forgetting": max_forgetting,
        "final_effective_rank": final_er,
        "final_dormant_frac": final_dormant,
        "peak_dormant_frac": peak_dormant,
        "curves": {
            "eval_iqm": (eval_steps, eval_vals),
            "forgetting": (f_steps, f_vals),
            "dormant_frac": (d_steps, d_vals),
            "effective_rank_avg": (er_steps, er_vals),
        },
    }


def plot_individual_intervention(
    intervention: str,
    aggregated_curve: Dict[str, List[float]],
    n_seeds: int,
    out_path: Path,
    global_y_min: Optional[float] = None,
    global_y_max: Optional[float] = None,
    task_length: int = 500000,
    num_tasks: int = 3,
):
    """
    Plot individual intervention IQM return curve with 95% CI.
    
    Args:
        intervention: Intervention name
        aggregated_curve: Aggregated curve data with steps, central, ci_low, ci_high
        n_seeds: Number of seeds
        out_path: Output file path
        global_y_min: Global minimum y-axis value (for consistent scaling)
        global_y_max: Global maximum y-axis value (for consistent scaling)
        task_length: Length of each task in steps (for task boundaries)
        num_tasks: Number of tasks (for task labels)
    """
    steps = np.array(aggregated_curve["steps"])
    central = np.array(aggregated_curve["central"])
    ci_low = np.array(aggregated_curve["ci_low"])
    ci_high = np.array(aggregated_curve["ci_high"])

    # Convert steps to thousands for readability
    steps_k = steps / 1000.0

    fig, ax = plt.subplots(figsize=(10, 6))
    
    color = INTERVENTION_COLORS.get(intervention, "#333333")
    display_name = INTERVENTION_NAMES.get(intervention, intervention.capitalize())
    
    # Plot mean line
    ax.plot(steps_k, central, label=display_name, color=color, linewidth=2.5)
    
    # Plot 95% CI
    if n_seeds > 1:
        ax.fill_between(steps_k, ci_low, ci_high, color=color, alpha=0.25, 
                        label='95% CI')
    
    # Apply global y-limits if provided (do this before adding task labels)
    if global_y_min is not None and global_y_max is not None:
        ax.set_ylim(global_y_min, global_y_max)
    
    # Add task boundaries: vertical dashed lines
    task_boundaries_k = [task_length * k / 1000.0 for k in range(1, num_tasks)]
    for boundary_k in task_boundaries_k:
        ax.axvline(boundary_k, linestyle='--', color='black', alpha=0.6, 
                  linewidth=1.5, zorder=1)
    
    # Add task labels: "Task 1", "Task 2", "Task 3"
    # Position at 95% of y-range
    if global_y_min is not None and global_y_max is not None:
        label_y = global_y_min + 0.95 * (global_y_max - global_y_min)
    else:
        # Fallback if no global limits
        y_min, y_max = ax.get_ylim()
        label_y = y_min + 0.95 * (y_max - y_min)
    
    for k in range(num_tasks):
        # Center of task k (0-indexed)
        task_center_k = (k + 0.5) * task_length / 1000.0
        ax.text(task_center_k, label_y, f'Task {k+1}', 
               horizontalalignment='center', verticalalignment='top',
               fontsize=12, fontweight='normal', alpha=0.8,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor='none', alpha=0.7),
               zorder=4)
    
    ax.set_xlabel('Environment Steps (thousands)', fontsize=12)
    ax.set_ylabel('IQM Return (avg across tasks)', fontsize=12)
    ax.set_title(f'{display_name} - IQM Return (n={n_seeds} seeds)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, zorder=0)
    ax.legend(fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    fig.savefig(out_path, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_combined_interventions(
    all_curves: Dict[str, Dict[str, Any]],
    out_path: Path,
    task_length: int = 500000,
    num_tasks: int = 3,
):
    """
    Plot all interventions on one graph for comparison.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Sort interventions for consistent ordering
    sorted_interventions = sorted(all_curves.keys())
    
    for intervention in sorted_interventions:
        data = all_curves[intervention]
        aggregated = data["aggregated_curve"]
        n_seeds = data["n_seeds"]
        
        steps = np.array(aggregated["steps"])
        central = np.array(aggregated["central"])
        ci_low = np.array(aggregated["ci_low"])
        ci_high = np.array(aggregated["ci_high"])
        
        # Convert steps to thousands for readability
        steps_k = steps / 1000.0
        
        color = INTERVENTION_COLORS.get(intervention, "#333333")
        display_name = INTERVENTION_NAMES.get(intervention, intervention.capitalize())
        
        # Plot mean line with legend showing seed count
        ax.plot(steps_k, central, label=f'{display_name} (n={n_seeds})', 
                color=color, linewidth=2.5)
        
        # Plot 95% CI
        if n_seeds > 1:
            ax.fill_between(steps_k, ci_low, ci_high, color=color, alpha=0.15)
    
    ax.set_xlabel('Environment Steps (thousands)', fontsize=12)
    ax.set_ylabel('IQM Return (avg across tasks)', fontsize=12)
    ax.set_title('All Interventions - IQM Return Comparison', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, framealpha=0.9, loc='best')
    
    # Add task boundaries
    task_boundaries_k = [task_length * k / 1000.0 for k in range(1, num_tasks)]
    for boundary_k in task_boundaries_k:
        ax.axvline(boundary_k, linestyle='--', color='black', alpha=0.6, linewidth=1.5, zorder=1)
    
    # Add task labels
    y_min, y_max = ax.get_ylim()
    label_y = y_min + 0.95 * (y_max - y_min)
    for k in range(num_tasks):
        task_center_k = (k + 0.5) * task_length / 1000.0
        ax.text(task_center_k, label_y, f'Task {k+1}', 
               horizontalalignment='center', verticalalignment='top',
               fontsize=12, fontweight='normal', alpha=0.8,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor='none', alpha=0.7),
               zorder=4)
    
    plt.tight_layout()
    fig.savefig(out_path, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_grouped_comparisons(
    all_curves: Dict[str, Dict[str, Any]],
    out_dir: Path,
    task_length: int = 500000,
    num_tasks: int = 3,
):
    """
    Create three grouped comparison plots: Sparse, Reset-based, Baselines.
    Each group shows individual interventions with shared y-axis.
    """
    # Define groups
    groups = [
        {
            'name': 'sparse_methods',
            'title': 'Sparse Methods – IQM Return',
            'interventions': ['gmp', 'set'],
            'colors': {'gmp': '#ff7f0e', 'set': '#8c564b'}  # orange, brown
        },
        {
            'name': 'reset_based_methods',
            'title': 'Reset-Based Methods – IQM Return',
            'interventions': ['redo', 'partial_reinit'],
            'colors': {'redo': '#d62728', 'partial_reinit': '#2ca02c'}  # red, green
        },
        {
            'name': 'baselines',
            'title': 'Baselines – IQM Return',
            'interventions': ['dense', 'reset'],
            'colors': {'dense': '#1f77b4', 'reset': '#9467bd'}  # blue, purple
        }
    ]
    
    # Step 1: Compute global y-limits
    global_y_min = float('inf')
    global_y_max = float('-inf')
    
    for group in groups:
        for intervention in group['interventions']:
            if intervention not in all_curves:
                continue
            data = all_curves[intervention]
            aggregated = data['aggregated_curve']
            ci_low = np.array(aggregated['ci_low'])
            ci_high = np.array(aggregated['ci_high'])
            global_y_min = min(global_y_min, float(np.nanmin(ci_low)))
            global_y_max = max(global_y_max, float(np.nanmax(ci_high)))
    
    # Add padding
    if global_y_min != float('inf'):
        y_range = global_y_max - global_y_min
        global_y_min -= 0.05 * y_range
        global_y_max += 0.05 * y_range
    
    # Step 2: Create one plot per group
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for group in groups:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for intervention in group['interventions']:
            if intervention not in all_curves:
                print(f"  ⚠ {intervention} not found, skipping")
                continue
            
            data = all_curves[intervention]
            aggregated = data['aggregated_curve']
            n_seeds = data['n_seeds']
            
            steps = np.array(aggregated['steps'])
            central = np.array(aggregated['central'])
            ci_low = np.array(aggregated['ci_low'])
            ci_high = np.array(aggregated['ci_high'])
            
            steps_k = steps / 1000.0
            
            color = group['colors'].get(intervention, '#333333')
            display_name = INTERVENTION_NAMES.get(intervention, intervention.capitalize())
            
            # Plot mean line
            ax.plot(steps_k, central, label=f'{display_name} (n={n_seeds})',
                   color=color, linewidth=3, zorder=3)
            
            # Plot CI band
            if n_seeds > 1:
                ax.fill_between(steps_k, ci_low, ci_high, color=color, alpha=0.25, zorder=2)
        
        # Apply global y-limits
        if global_y_min != float('inf'):
            ax.set_ylim(global_y_min, global_y_max)
        
        # Add task boundaries
        task_boundaries_k = [task_length * k / 1000.0 for k in range(1, num_tasks)]
        for boundary_k in task_boundaries_k:
            ax.axvline(boundary_k, linestyle='--', color='black', alpha=0.6, linewidth=1.5, zorder=1)
        
        # Add task labels
        y_min, y_max = ax.get_ylim()
        label_y = y_min + 0.95 * (y_max - y_min)
        for k in range(num_tasks):
            task_center_k = (k + 0.5) * task_length / 1000.0
            ax.text(task_center_k, label_y, f'Task {k+1}', 
                   horizontalalignment='center', verticalalignment='top',
                   fontsize=12, fontweight='normal', alpha=0.8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='none', alpha=0.7),
                   zorder=4)
        
        ax.set_xlabel('Environment Steps (thousands)', fontsize=12)
        ax.set_ylabel('IQM Return (avg across tasks)', fontsize=12)
        ax.set_title(group['title'], fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, zorder=0)
        ax.legend(fontsize=11, framealpha=0.9)
        
        plt.tight_layout()
        out_path = out_dir / f"{group['name']}.png"
        fig.savefig(out_path, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved {group['name']} plot to {out_path}")
    
    return len(groups)


def create_metrics_table(
    all_metrics: Dict[str, Dict[str, Any]],
    bootstrap_samples: int,
    statistic: str,
) -> pd.DataFrame:
    """
    Create a comprehensive metrics table for all interventions.
    """
    rows = []
    
    # Sort interventions for consistent ordering
    sorted_interventions = sorted(all_metrics.keys())
    
    for intervention in sorted_interventions:
        data = all_metrics[intervention]
        n_seeds = data["n_seeds"]
        display_name = INTERVENTION_NAMES.get(intervention, intervention.capitalize())
        
        row = {
            "Intervention": display_name,
            "Seeds": n_seeds,
        }
        
        # Add metrics with CI
        metrics = [
            ("Final IQM Return", "final_iqm_return"),
            ("Peak IQM Return", "peak_iqm_return"),
            ("Max Forgetting", "max_isolated_forgetting"),
            ("Final Effective Rank", "final_effective_rank"),
            ("Final Dormant Frac", "final_dormant_frac"),
            ("Peak Dormant Frac", "peak_dormant_frac"),
        ]
        
        for metric_name, metric_key in metrics:
            central = data[f"{metric_key}_{statistic}"]
            ci_low = data[f"{metric_key}_ci_low"]
            ci_high = data[f"{metric_key}_ci_high"]
            ci_width = ci_high - ci_low
            
            # Format values
            row[metric_name] = f"{central:.4f}"
            row[f"{metric_name} CI"] = f"[{ci_low:.4f}, {ci_high:.4f}]"
            row[f"{metric_name} CI Width"] = f"{ci_width:.4f}"
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze final run results and create plots + metrics table"
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="runs/procgen_3_tasks_1_cycle_500k_starpilot",
        help="Root directory containing intervention folders",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/final_results",
        help="Output directory for plots and tables",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=10000,
        help="Number of bootstrap resamples for CI (10000 recommended)",
    )
    parser.add_argument(
        "--task-length",
        type=int,
        default=500000,
        help="Length of each task in environment steps (for task boundaries)",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=3,
        help="Number of tasks in the continual learning setup",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="CI alpha level (0.05 => 95% CI)",
    )
    parser.add_argument(
        "--last-k-rank",
        type=int,
        default=10,
        help="Average last K effective rank points per seed",
    )
    parser.add_argument(
        "--statistic",
        type=str,
        default="median",
        choices=["mean", "median", "iqm"],
        help="Central estimate: median (robust, recommended), mean, or iqm",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information",
    )
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    runs_dir = Path(args.runs_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not runs_dir.exists():
        print(f"ERROR: Runs directory does not exist: {runs_dir}")
        sys.exit(1)
    
    print("=" * 80)
    print("ANALYZING FINAL RUN RESULTS")
    print("=" * 80)
    print(f"Runs directory: {runs_dir}")
    print(f"Output directory: {out_dir}")
    print(f"Bootstrap samples: {args.bootstrap}")
    print(f"Statistic: {args.statistic}")
    print()
    
    # 1) Find all event files
    if args.verbose:
        print("Searching for TensorBoard event files...")
    
    event_files = find_event_files(str(runs_dir))
    
    if not event_files:
        print(f"ERROR: No event files found in {runs_dir}")
        sys.exit(1)
    
    print(f"Found {len(event_files)} TensorBoard event files")
    
    # 2) Infer group+seed for each event file
    identities = [infer_group_and_seed(str(runs_dir), ef) for ef in event_files]
    
    # 3) Group into: intervention -> seed -> [event_files]
    # The directory structure is: runs_dir/intervention/seed_*/run_*/events...
    # We need to extract the intervention from the path
    intervention_seeds: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    
    for rid in identities:
        # Extract intervention from the event file path
        # Path is like: .../dense/seed_0_20260120_110836/.../events...
        event_path = Path(rid.event_file)
        rel_path = event_path.relative_to(runs_dir)
        
        # First part of relative path should be the intervention
        intervention = rel_path.parts[0] if len(rel_path.parts) > 0 else "unknown"
        
        # Extract seed from the seed_* directory
        seed = rid.seed
        
        intervention_seeds[intervention][seed].append(rid.event_file)
    
    if args.verbose:
        print("\nDiscovered interventions and seeds:")
        for intervention, seed_map in sorted(intervention_seeds.items()):
            print(f"  {intervention}: {len(seed_map)} seeds ({sorted(seed_map.keys())})")
    
    print()
    
    # 4) Extract per-seed metrics for each intervention
    all_intervention_data: Dict[str, Dict[str, Any]] = {}
    
    for intervention, seed_map in sorted(intervention_seeds.items()):
        print(f"Processing intervention: {intervention}")
        
        seed_metrics: Dict[str, Dict[str, Any]] = {}
        skipped = 0
        
        for seed, files in sorted(seed_map.items()):
            best = choose_best_event_file(files)
            
            try:
                scalars = load_scalars(best)
            except ValueError as e:
                skipped += 1
                if args.verbose:
                    print(f"  [SKIP] seed={seed}: {e}")
                continue
            
            metrics = extract_run_metrics(scalars, last_k_rank=args.last_k_rank)
            seed_metrics[seed] = metrics
        
        if skipped > 0:
            print(f"  Skipped {skipped} invalid/corrupt event files")
        
        seeds = sorted(seed_metrics.keys())
        n_seeds = len(seeds)
        
        if n_seeds == 0:
            print(f"  WARNING: No valid seeds found for {intervention}")
            continue
        
        print(f"  Valid seeds: {n_seeds}")
        
        # 5) Aggregate metrics across seeds
        # Collect per-seed summary values
        final_iqm_vals = np.array([seed_metrics[s]["final_iqm_return"] for s in seeds])
        peak_iqm_vals = np.array([seed_metrics[s]["peak_iqm_return"] for s in seeds])
        max_forget_vals = np.array([seed_metrics[s]["max_isolated_forgetting"] for s in seeds])
        final_er_vals = np.array([seed_metrics[s]["final_effective_rank"] for s in seeds])
        final_dorm_vals = np.array([seed_metrics[s]["final_dormant_frac"] for s in seeds])
        peak_dorm_vals = np.array([seed_metrics[s]["peak_dormant_frac"] for s in seeds])
        
        # Bootstrap CI across seeds
        final_iqm_stat, final_iqm_lo, final_iqm_hi = bootstrap_ci(
            final_iqm_vals, args.bootstrap, args.alpha, seed=0, statistic=args.statistic
        )
        peak_iqm_stat, peak_iqm_lo, peak_iqm_hi = bootstrap_ci(
            peak_iqm_vals, args.bootstrap, args.alpha, seed=1, statistic=args.statistic
        )
        max_forget_stat, max_forget_lo, max_forget_hi = bootstrap_ci(
            max_forget_vals, args.bootstrap, args.alpha, seed=2, statistic=args.statistic
        )
        final_er_stat, final_er_lo, final_er_hi = bootstrap_ci(
            final_er_vals, args.bootstrap, args.alpha, seed=3, statistic=args.statistic
        )
        final_dorm_stat, final_dorm_lo, final_dorm_hi = bootstrap_ci(
            final_dorm_vals, args.bootstrap, args.alpha, seed=4, statistic=args.statistic
        )
        peak_dorm_stat, peak_dorm_lo, peak_dorm_hi = bootstrap_ci(
            peak_dorm_vals, args.bootstrap, args.alpha, seed=5, statistic=args.statistic
        )
        
        # 6) Aggregate eval IQM curves across seeds
        eval_curves = []
        for s in seeds:
            curves = seed_metrics[s]["curves"]
            if curves["eval_iqm"][0] is not None:
                eval_curves.append(curves["eval_iqm"])
        
        if not eval_curves:
            print(f"  WARNING: No eval IQM curves found for {intervention}")
            continue
        
        aggregated_curve = aggregate_curves_across_seeds(
            eval_curves, args.bootstrap, args.alpha, seed=10, statistic=args.statistic
        )
        
        # Store all data
        all_intervention_data[intervention] = {
            "n_seeds": n_seeds,
            "aggregated_curve": aggregated_curve,
            f"final_iqm_return_{args.statistic}": final_iqm_stat,
            "final_iqm_return_ci_low": final_iqm_lo,
            "final_iqm_return_ci_high": final_iqm_hi,
            f"peak_iqm_return_{args.statistic}": peak_iqm_stat,
            "peak_iqm_return_ci_low": peak_iqm_lo,
            "peak_iqm_return_ci_high": peak_iqm_hi,
            f"max_isolated_forgetting_{args.statistic}": max_forget_stat,
            "max_isolated_forgetting_ci_low": max_forget_lo,
            "max_isolated_forgetting_ci_high": max_forget_hi,
            f"final_effective_rank_{args.statistic}": final_er_stat,
            "final_effective_rank_ci_low": final_er_lo,
            "final_effective_rank_ci_high": final_er_hi,
            f"final_dormant_frac_{args.statistic}": final_dorm_stat,
            "final_dormant_frac_ci_low": final_dorm_lo,
            "final_dormant_frac_ci_high": final_dorm_hi,
            f"peak_dormant_frac_{args.statistic}": peak_dorm_stat,
            "peak_dormant_frac_ci_low": peak_dorm_lo,
            "peak_dormant_frac_ci_high": peak_dorm_hi,
        }
        
        print(f"  ✓ Processed {intervention}")
    
    if not all_intervention_data:
        print("\nERROR: No valid intervention data found")
        sys.exit(1)
    
    print()
    print("=" * 80)
    print("GENERATING OUTPUTS")
    print("=" * 80)
    
    # 7) Compute global y-limits for individual plots
    global_y_min = float('inf')
    global_y_max = float('-inf')
    
    for intervention, data in all_intervention_data.items():
        aggregated = data["aggregated_curve"]
        ci_low = np.array(aggregated["ci_low"])
        ci_high = np.array(aggregated["ci_high"])
        
        global_y_min = min(global_y_min, float(np.nanmin(ci_low)))
        global_y_max = max(global_y_max, float(np.nanmax(ci_high)))
    
    # Add 5% padding
    y_range = global_y_max - global_y_min
    global_y_min -= 0.05 * y_range
    global_y_max += 0.05 * y_range
    
    print(f"\nGlobal y-axis limits for individual plots: [{global_y_min:.2f}, {global_y_max:.2f}]")
    
    # 8) Create individual plots for each intervention with shared y-limits
    print("\nCreating individual intervention plots...")
    individual_dir = out_dir / "individual_interventions"
    individual_dir.mkdir(parents=True, exist_ok=True)
    
    for intervention, data in sorted(all_intervention_data.items()):
        display_name = INTERVENTION_NAMES.get(intervention, intervention.capitalize())
        out_path = individual_dir / f"{intervention}_iqm_return.png"
        
        plot_individual_intervention(
            intervention,
            data["aggregated_curve"],
            data["n_seeds"],
            out_path,
            global_y_min=global_y_min,
            global_y_max=global_y_max,
            task_length=args.task_length,
            num_tasks=args.num_tasks,
        )
        print(f"  ✓ Saved {display_name} plot to {out_path}")
    
    # 9) Create combined plot with all interventions
    print("\nCreating combined plot...")
    combined_path = out_dir / "all_interventions_combined.png"
    plot_combined_interventions(
        all_intervention_data,
        combined_path,
        task_length=args.task_length,
        num_tasks=args.num_tasks
    )
    print(f"  ✓ Saved combined plot to {combined_path}")
    
    # 10) Create grouped comparison plots
    print("\nCreating grouped comparison plots...")
    grouped_dir = out_dir / "grouped_comparisons"
    plot_grouped_comparisons(
        all_intervention_data,
        grouped_dir,
        task_length=args.task_length,
        num_tasks=args.num_tasks
    )
    print(f"  ✓ Saved grouped comparison plots to {grouped_dir}")
    
    # 11) Create metrics table
    print("\nCreating metrics table...")
    metrics_df = create_metrics_table(
        all_intervention_data,
        args.bootstrap,
        args.statistic,
    )
    
    # Save as CSV
    csv_path = out_dir / "metrics_summary.csv"
    metrics_df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved metrics table to {csv_path}")
    
    # Also save as pretty formatted text
    txt_path = out_dir / "metrics_summary.txt"
    with open(txt_path, "w") as f:
        f.write("=" * 120 + "\n")
        f.write("FINAL RUN METRICS SUMMARY\n")
        f.write("=" * 120 + "\n")
        f.write(f"Bootstrap samples: {args.bootstrap}\n")
        f.write(f"Confidence level: 95%\n")
        f.write(f"Statistic: {args.statistic.upper()}\n")
        f.write("=" * 120 + "\n\n")
        f.write(metrics_df.to_string(index=False))
        f.write("\n\n" + "=" * 120 + "\n")
        f.write("Metrics explained:\n")
        f.write("  • Final IQM Return: Performance at END of training (last value)\n")
        f.write("  • Peak IQM Return: BEST performance achieved during training (max value)\n")
        f.write("  • Max Forgetting: WORST forgetting observed (max value)\n")
        f.write("  • Final Effective Rank: Diversity of learned features (avg of last 10 values)\n")
        f.write("  • Final Dormant Frac: Fraction of dormant neurons at END (lower is better)\n")
        f.write("  • Peak Dormant Frac: HIGHEST dormant fraction observed (lower is better)\n")
        f.write("=" * 120 + "\n")
    
    print(f"  ✓ Saved formatted metrics to {txt_path}")
    
    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nAll outputs saved to: {out_dir}")
    print(f"  • {len(all_intervention_data)} individual intervention plots")
    print(f"  • 1 combined comparison plot")
    print(f"  • 1 metrics CSV table")
    print(f"  • 1 formatted metrics text file")
    print()


if __name__ == "__main__":
    main()
