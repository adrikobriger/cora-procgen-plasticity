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
    extract_task_avg_train_iqm_curve,
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
    # Handle variants
    "gmp_one_layer": "GMP (One Layer)",
    "gmp_whole_network": "GMP (Whole Network)",
    "set_one_layer": "SET (One Layer)",
    "set_whole_network": "SET (Whole Network)",
}

def get_display_name(intervention: str) -> str:
    """Get display name for intervention, handling dynamic variants."""
    if intervention in INTERVENTION_NAMES:
        return INTERVENTION_NAMES[intervention]
    # Handle dynamic variants like method_variant
    parts = intervention.split('_', 1)
    if len(parts) == 2:
        method, variant = parts
        base_name = INTERVENTION_NAMES.get(method, method.capitalize())
        variant_name = variant.replace('_', ' ').title()
        return f"{base_name} ({variant_name})"
    return intervention.replace('_', ' ').title()

# Color scheme for interventions
INTERVENTION_COLORS = {
    "dense": "#1f77b4",      # blue
    "gmp": "#ff7f0e",        # orange
    "partial_reinit": "#2ca02c",  # green
    "redo": "#d62728",       # red
    "reset": "#9467bd",      # purple
    "set": "#8c564b",        # brown
    # Variants
    "gmp_one_layer": "#ff7f0e",
    "gmp_whole_network": "#ff9f4e",
    "set_one_layer": "#8c564b",
    "set_whole_network": "#bc766b",
}

def get_color(intervention: str) -> str:
    """Get color for intervention, with fallback for dynamic variants."""
    if intervention in INTERVENTION_COLORS:
        return INTERVENTION_COLORS[intervention]
    # Fallback: use base method color if available
    base = intervention.split('_')[0]
    return INTERVENTION_COLORS.get(base, "#333333")


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
    metric_type: str = "eval",
) -> Dict[str, Any]:
    """
    Extract per-seed metrics from one TensorBoard run (one event file).
    
    Args:
        scalars: Scalar data from TensorBoard
        last_k_rank: Number of last points to average for effective rank
        metric_type: "eval" for eval_reward_iqm or "train" for train_reward_iqm
    
    Returns dict with per-seed summary scalars and curves for aggregation.
    """

    # ---- task-avg IQM curve (eval or train) ----
    if metric_type == "train":
        iqm_curve = extract_task_avg_train_iqm_curve(scalars)
    else:
        iqm_curve = extract_task_avg_eval_iqm_curve(scalars)
    eval_curve = iqm_curve
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


def format_steps_label(steps_k):
    """Convert steps in thousands to clean labels (800k, 900k, 1M, 1.1M)."""
    if steps_k < 1000:
        return f"{steps_k:.0f}k"
    else:
        return f"{steps_k / 1000:.1f}M".rstrip('0').rstrip('.')


def plot_individual_intervention(
    intervention: str,
    aggregated_curve: Dict[str, List[float]],
    seed_curves: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None,
    n_seeds: int = 1,
    out_path: Path = None,
    global_y_min: Optional[float] = None,
    global_y_max: Optional[float] = None,
    task_length: int = 500000,
    num_tasks: int = 3,
    metric_type: str = "eval",
):
    """
    Plot individual intervention IQM return curve with 95% CI and individual seed traces.
    
    Args:
        intervention: Intervention name
        aggregated_curve: Aggregated curve data with steps, central, ci_low, ci_high
        seed_curves: Dict mapping seed -> (steps, values) for individual seed traces
        n_seeds: Number of seeds
        out_path: Output file path
        global_y_min: Global minimum y-axis value (for consistent scaling)
        global_y_max: Global maximum y-axis value (for consistent scaling)
        task_length: Length of each task in steps (for task boundaries)
        num_tasks: Number of tasks (for task labels)
        metric_type: "eval" or "train" for labeling
    """
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    
    steps = np.array(aggregated_curve["steps"])
    central = np.array(aggregated_curve["central"])
    ci_low = np.array(aggregated_curve["ci_low"])
    ci_high = np.array(aggregated_curve["ci_high"])

    # Convert steps to thousands for readability
    steps_k = steps / 1000.0
    
    # No smoothing - use raw data directly
    steps_plot = steps_k
    central_plot = central
    ci_low_plot = ci_low
    ci_high_plot = ci_high

    fig, ax = plt.subplots(figsize=(11, 7))
    
    color = get_color(intervention)
    display_name = get_display_name(intervention)
    
    # Plot individual seed curves as thin lines (if provided)
    if seed_curves:
        for seed_idx, (seed, (seed_steps, seed_vals)) in enumerate(sorted(seed_curves.items())):
            seed_steps_k = seed_steps / 1000.0
            # Only plot on first iteration for legend
            ax.plot(seed_steps_k, seed_vals, color=color, linewidth=0.8, alpha=0.25, 
                   zorder=1, label='Individual seeds' if seed_idx == 0 else None)
    
    # Plot 95% CI band
    if n_seeds > 1:
        ax.fill_between(steps_plot, ci_low_plot, ci_high_plot, color=color, alpha=0.35, 
                        label='95% Confidence Interval', zorder=2)
    
    # Plot mean line (thick, on top)
    ax.plot(steps_plot, central_plot, label=f'{display_name} (Mean, n={n_seeds} seeds)', 
            color=color, linewidth=3.5, zorder=3)

    # Apply global y-limits if provided
    if global_y_min is not None and global_y_max is not None:
        ax.set_ylim(global_y_min, global_y_max)
    
    # Add task boundaries: vertical dashed lines
    task_boundaries_k = [task_length * k / 1000.0 for k in range(1, num_tasks)]
    for boundary_k in task_boundaries_k:
        ax.axvline(boundary_k, linestyle='--', color='black', alpha=0.5, 
                  linewidth=1.5, zorder=1)
    
    # Add task labels: "Task 1", "Task 2", "Task 3"
    if global_y_min is not None and global_y_max is not None:
        label_y = global_y_min + 0.96 * (global_y_max - global_y_min)
    else:
        y_min, y_max = ax.get_ylim()
        label_y = y_min + 0.96 * (y_max - y_min)
    
    for k in range(num_tasks):
        task_center_k = (k + 0.5) * task_length / 1000.0
        ax.text(task_center_k, label_y, f'Task {k+1}', 
               horizontalalignment='center', verticalalignment='top',
               fontsize=16, fontweight='bold', alpha=0.85,
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                        edgecolor='gray', alpha=0.85, linewidth=0.5),
               zorder=4)
    
    # Format x-axis with custom tick labels
    from matplotlib.ticker import FuncFormatter
    def step_formatter(x, pos):
        return format_steps_label(x)
    ax.xaxis.set_major_formatter(FuncFormatter(step_formatter))
    
    ax.set_xlabel('Environment Steps', fontsize=17, fontweight='bold')
    ax.set_ylabel(f'{metric_type} IQM Return (averaged across tasks)', fontsize=17, fontweight='bold')
    ax.set_title(f'{display_name} – Continual Learning Performance', 
                 fontsize=19, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.25, zorder=0, linestyle='-', linewidth=0.5)
    ax.tick_params(labelsize=15)
    
    # Improved legend - smaller font, bottom right position
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=11, framealpha=0.95, loc='lower left', 
             edgecolor='black', fancybox=True, shadow=True)
    
    plt.tight_layout()
    fig.savefig(out_path, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)



def plot_combined_interventions(
    all_curves: Dict[str, Dict[str, Any]],
    out_path: Path,
    task_length: int = 500000,
    num_tasks: int = 3,
    metric_type: str = "eval",
):
    """
    Plot all interventions on one graph for comparison with 95% CI bands.
    """
    # Set professional font
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    
    fig, ax = plt.subplots(figsize=(13, 8))
    
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
        
        # No smoothing - use raw data directly
        steps_plot = steps_k
        central_plot = central
        ci_low_plot = ci_low
        ci_high_plot = ci_high
        
        color = get_color(intervention)
        display_name = get_display_name(intervention)
        
        # Plot 95% CI band first (behind mean)
        if n_seeds > 1:
            ax.fill_between(steps_plot, ci_low_plot, ci_high_plot, color=color, 
                           alpha=0.3, zorder=1, label=None)
        
        # Plot mean line with legend showing seed count and CI
        ax.plot(steps_plot, central_plot, label=f'{display_name} (n={n_seeds}, 95% CI)', 
                color=color, linewidth=3.5, zorder=3)
    
    # Format x-axis with custom tick labels
    from matplotlib.ticker import FuncFormatter
    def step_formatter(x, pos):
        return format_steps_label(x)
    ax.xaxis.set_major_formatter(FuncFormatter(step_formatter))
    
    ax.set_xlabel('Environment Steps', fontsize=17, fontweight='bold')
    ax.set_ylabel(f'{metric_type} IQM Return (averaged across tasks)', fontsize=17, fontweight='bold')
    ax.set_title('Continual Learning: Comparison of All Interventions', 
                 fontsize=19, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.25, zorder=0, linestyle='-', linewidth=0.5)
    ax.tick_params(labelsize=15)

    # Add task boundaries
    task_boundaries_k = [task_length * k / 1000.0 for k in range(1, num_tasks)]
    for boundary_k in task_boundaries_k:
        ax.axvline(boundary_k, linestyle='--', color='black', alpha=0.5, 
                  linewidth=1.5, zorder=1)

    # Add task labels higher to avoid covering plot
    y_min, y_max = ax.get_ylim()
    label_y = y_min + 0.92 * (y_max - y_min)
    for k in range(num_tasks):
        task_center_k = (k + 0.5) * task_length / 1000.0
        ax.text(task_center_k, label_y, f'Task {k+1}', 
               horizontalalignment='center', verticalalignment='bottom',
               fontsize=16, fontweight='bold', alpha=0.85,
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                        edgecolor='gray', alpha=0.85, linewidth=0.5),
               zorder=4)
    
    # Legend - placed outside the plot on the right
    ax.legend(fontsize=11, framealpha=0.95, loc='center left', 
             bbox_to_anchor=(1.02, 0.5), edgecolor='black', fancybox=True, shadow=True, 
             title='Method (Seeds, Confidence)', title_fontsize=11)
    
    plt.tight_layout()
    fig.savefig(out_path, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_additional_metrics(
    all_intervention_data: Dict[str, Dict[str, Any]],
    out_dir: Path,
    bootstrap: int,
    alpha: float,
    statistic: str,
    task_length: int = 500000,
    num_tasks: int = 3,
    use_optimizer_steps_for_dormant: bool = True,
):
    """
    Create individual per-method plots for forgetting, effective_rank, and dormant_frac.
    Uses the same aesthetic as the IQM plots with seed traces, task boundaries, etc.
    
    Args:
        all_intervention_data: Dict mapping intervention -> data (including seed_metrics)
        out_dir: Output directory for plots
        bootstrap: Number of bootstrap samples
        alpha: CI alpha level
        statistic: "mean", "median", or "iqm"
        task_length: Length of each task in steps
        num_tasks: Number of tasks
        use_optimizer_steps_for_dormant: If True, convert dormant_frac x-axis from opt steps to env steps
    """
    # Set professional font
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    
    # Metrics to plot: (metric_key, y_label, title_suffix, use_opt_steps_conversion)
    metrics = [
        ("forgetting", "Isolated Forgetting", "Isolated Forgetting", False),
        ("effective_rank_avg", "Effective Rank", "Effective Rank", False),
        ("dormant_frac", "Dormant Fraction", "Dormant Neuron Fraction", True),
    ]
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for intervention, data in sorted(all_intervention_data.items()):
        display_name = get_display_name(intervention)
        n_seeds = data["n_seeds"]
        seed_metrics = data.get("seed_metrics", {})
        
        for metric_key, y_label, title_suffix, needs_opt_conversion in metrics:
            # Collect curves from seeds
            seed_curves = []
            seed_curves_dict = {}
            
            for seed_id, seed_metric_data in seed_metrics.items():
                curves = seed_metric_data["curves"]
                if curves[metric_key][0] is not None:
                    seed_curves.append(curves[metric_key])
                    steps, values = curves[metric_key]
                    seed_curves_dict[seed_id] = (np.array(steps), np.array(values))
            
            if not seed_curves:
                continue  # Skip if no data for this metric
            
            # Aggregate curves across seeds
            aggregated_curve = aggregate_curves_across_seeds(
                seed_curves, bootstrap, alpha, seed=hash(metric_key) % 10000, statistic=statistic
            )
            
            steps = np.array(aggregated_curve["steps"])
            central = np.array(aggregated_curve["central"])
            ci_low = np.array(aggregated_curve["ci_low"])
            ci_high = np.array(aggregated_curve["ci_high"])
            
            # Handle optimizer steps conversion for dormant_frac
            use_optimizer_steps = needs_opt_conversion and use_optimizer_steps_for_dormant
            
            if use_optimizer_steps and len(steps) > 0:
                # Dormant fraction is logged in optimizer steps, keep it that way
                # Don't convert - just use the original optimizer steps
                steps_k = steps / 1000.0  # Just convert to thousands for display
                x_label = 'Optimizer Steps'
                
                # Calculate conversion ratio for task boundaries ONLY
                max_opt_steps = float(np.max(steps))
                total_env_steps = num_tasks * task_length
                conversion_ratio = total_env_steps / max_opt_steps if max_opt_steps > 0 else 1.0
            else:
                # Regular environment steps
                steps_k = steps / 1000.0
                x_label = 'Environment Steps'
                conversion_ratio = 1.0  # No conversion needed
            
            # Create figure
            fig, ax = plt.subplots(figsize=(11, 7))
            color = get_color(intervention)
            
            # Plot individual seed curves (thin, transparent)
            if seed_curves_dict:
                for seed_idx, (seed_id, (seed_steps, seed_vals)) in enumerate(sorted(seed_curves_dict.items())):
                    if use_optimizer_steps:
                        # Keep in optimizer steps (don't convert)
                        seed_steps_k = seed_steps / 1000.0
                    else:
                        seed_steps_k = seed_steps / 1000.0
                    
                    label = f'Individual seeds (n={len(seed_curves_dict)})' if seed_idx == 0 else None
                    ax.plot(seed_steps_k, seed_vals, color=color, linewidth=0.8, alpha=0.25,
                           zorder=1, label=label)
            
            # Plot 95% CI band
            if n_seeds > 1:
                ax.fill_between(steps_k, ci_low, ci_high, color=color, alpha=0.35,
                               label='95% Confidence Interval', zorder=2)
            
            # Plot central line (thick, on top)
            ax.plot(steps_k, central, label=f'{display_name} ({statistic.capitalize()}, n={n_seeds} seeds)',
                   color=color, linewidth=3.5, zorder=3)
            
            # Set consistent x-axis limits
            if use_optimizer_steps:
                # For optimizer steps: 0 to max observed * 1.02 for padding
                x_max = float(np.max(steps_k)) * 1.02
            else:
                # For env steps: 0 to total steps
                x_max = (num_tasks * task_length) / 1000.0
            
            ax.set_xlim(0, x_max)
            
            # Add task boundaries
            if not use_optimizer_steps:
                # Regular environment steps
                task_boundaries_k = [task_length * k / 1000.0 for k in range(1, num_tasks)]
            else:
                # For optimizer steps: convert env step boundaries to optimizer step scale
                # Boundary in env steps / conversion ratio = boundary in opt steps
                task_boundaries_k = [(task_length * k / conversion_ratio) / 1000.0 
                                    for k in range(1, num_tasks)]
            
            for boundary_k in task_boundaries_k:
                ax.axvline(boundary_k, linestyle='--', color='gray', alpha=0.5,
                          linewidth=1.5, zorder=0)
            
            # Add task labels
            y_min, y_max = ax.get_ylim()
            label_y = y_min + 0.96 * (y_max - y_min)
            
            for k in range(num_tasks):
                if not use_optimizer_steps:
                    # Regular environment steps
                    task_center_k = (k + 0.5) * task_length / 1000.0
                else:
                    # For optimizer steps: convert env step task center to optimizer step scale
                    task_center_k = ((k + 0.5) * task_length / conversion_ratio) / 1000.0
                
                ax.text(task_center_k, label_y, f'Task {k+1}',
                       horizontalalignment='center', verticalalignment='top',
                       fontsize=16, fontweight='bold', alpha=0.85,
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                edgecolor='gray', alpha=0.85, linewidth=0.5),
                       zorder=4)
            
            # Format x-axis with custom tick labels
            from matplotlib.ticker import FuncFormatter
            if use_optimizer_steps:
                # For optimizer steps: show raw values (0, 1000, 2000, 3000, 4000)
                def opt_step_formatter(x, pos):
                    return f"{int(x * 1000)}"
                ax.xaxis.set_major_formatter(FuncFormatter(opt_step_formatter))
                ax.set_xlabel(x_label + ' (×1000)', fontsize=17, fontweight='bold')
            else:
                # For environment steps: use standard formatter
                def step_formatter(x, pos):
                    return format_steps_label(x)
                ax.xaxis.set_major_formatter(FuncFormatter(step_formatter))
                ax.set_xlabel(x_label, fontsize=17, fontweight='bold')
            
            # Format y-axis for dormant fraction (show as percentage)
            if metric_key == "dormant_frac":
                from matplotlib.ticker import PercentFormatter
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
            
            # Labels and styling
            ax.set_ylabel(y_label, fontsize=17, fontweight='bold')
            ax.set_title(f'{display_name} – {title_suffix}',
                        fontsize=19, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.25, zorder=0, linestyle='-', linewidth=0.5)
            ax.tick_params(labelsize=15)
            
            # Legend
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, fontsize=11, framealpha=0.95, loc='best',
                     edgecolor='black', fancybox=True, shadow=True)
            
            plt.tight_layout()
            
            # Save plot
            out_path = out_dir / f"{intervention}_{metric_key}.png"
            fig.savefig(out_path, format='png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"  ✓ Saved {display_name} {title_suffix} plot to {out_path}")


def plot_grouped_comparisons(
    all_curves: Dict[str, Dict[str, Any]],
    out_dir: Path,
    task_length: int = 500000,
    num_tasks: int = 3,
    metric_type: str = "eval",
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
    
    # Set professional font
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    
    
    for group in groups:
        # Check if ANY intervention in this group exists
        group_has_data = any(intervention in all_curves for intervention in group['interventions'])
        if not group_has_data:
            print(f"  ⚠ Skipping group '{group['name']}' - no interventions found")
            continue
        
        fig, ax = plt.subplots(figsize=(11, 7))
        
        for intervention in group['interventions']:
            if intervention not in all_curves:
                print(f"  ⚠ {intervention} not found in this run, skipping")
                continue
            
            data = all_curves[intervention]
            aggregated = data['aggregated_curve']
            n_seeds = data['n_seeds']
            
            steps = np.array(aggregated['steps'])
            central = np.array(aggregated['central'])
            ci_low = np.array(aggregated['ci_low'])
            ci_high = np.array(aggregated['ci_high'])
            
            steps_k = steps / 1000.0
            
            # No smoothing - use raw data directly
            steps_plot = steps_k
            central_plot = central
            ci_low_plot = ci_low
            ci_high_plot = ci_high
            
            color = group['colors'].get(intervention, '#333333')
            display_name = INTERVENTION_NAMES.get(intervention, intervention.capitalize())
            
            # Plot CI band first (behind)
            if n_seeds > 1:
                ax.fill_between(steps_plot, ci_low_plot, ci_high_plot, color=color, 
                               alpha=0.35, zorder=1, label=None)
            
            # Plot mean line on top
            ax.plot(steps_plot, central_plot, label=f'{display_name} (n={n_seeds}, 95% CI)',
                   color=color, linewidth=3.5, zorder=3)
        
        # Apply global y-limits
        if global_y_min != float('inf'):
            ax.set_ylim(global_y_min, global_y_max)
        
        # Add task boundaries
        task_boundaries_k = [task_length * k / 1000.0 for k in range(1, num_tasks)]
        for boundary_k in task_boundaries_k:
            ax.axvline(boundary_k, linestyle='--', color='black', alpha=0.5, linewidth=1.5, zorder=1)
        
        # Add task labels
        y_min, y_max = ax.get_ylim()
        label_y = y_min + 0.95 * (y_max - y_min)
        for k in range(num_tasks):
            task_center_k = (k + 0.5) * task_length / 1000.0
            ax.text(task_center_k, label_y, f'Task {k+1}', 
                   horizontalalignment='center', verticalalignment='top',
                   fontsize=16, fontweight='bold', alpha=0.85,
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                            edgecolor='gray', alpha=0.85, linewidth=0.5),
                   zorder=4)
        
        # Format x-axis with custom tick labels
        from matplotlib.ticker import FuncFormatter
        def step_formatter(x, pos):
            return format_steps_label(x)
        ax.xaxis.set_major_formatter(FuncFormatter(step_formatter))
        
        ax.set_xlabel('Environment Steps', fontsize=17, fontweight='bold')
        ax.set_ylabel(f'{metric_type} IQM Return (averaged across tasks)', fontsize=17, fontweight='bold')
        ax.set_title(group['title'], fontsize=19, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.25, zorder=0, linestyle='-', linewidth=0.5)
        ax.tick_params(labelsize=15)
        ax.legend(fontsize=11, framealpha=0.95, loc='lower left', 
                 edgecolor='black', fancybox=True, shadow=True)
        
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
        display_name = get_display_name(intervention)
        
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
    parser.add_argument(
        "--metric-type",
        type=str,
        default="both",
        choices=["eval", "train", "both"],
        help="Metric type to analyze: 'eval' for eval_reward_iqm, 'train' for train_reward_iqm, or 'both' for separate analyses",
    )
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    runs_dir = Path(args.runs_dir).resolve()
    base_out_dir = Path(args.out_dir).resolve()
    
    if not runs_dir.exists():
        print(f"ERROR: Runs directory does not exist: {runs_dir}")
        sys.exit(1)
    
    # Determine which metric types to run
    metric_types = []
    if args.metric_type == "both":
        metric_types = ["eval", "train"]
    else:
        metric_types = [args.metric_type]
    
    for metric_type in metric_types:
        # Create separate output directory for each metric type
        out_dir = base_out_dir / metric_type
        out_dir.mkdir(parents=True, exist_ok=True)
        
        print("="* 80)
        print(f"ANALYZING FINAL RUN RESULTS - {metric_type.upper()} METRICS")
        print("=" * 80)
        print(f"Runs directory: {runs_dir}")
        print(f"Output directory: {out_dir}")
        print(f"Metric type: {metric_type}_reward_iqm")
        print(f"Bootstrap samples: {args.bootstrap}")
        print(f"Statistic: {args.statistic}")
        print()
        
        run_analysis(runs_dir, out_dir, args, metric_type)


def run_analysis(runs_dir: Path, out_dir: Path, args, metric_type: str):
    """Run the analysis for a specific metric type."""
    
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
        # OR: .../gmp/one_layer/seed_0_20260120_110836/.../events...
        event_path = Path(rid.event_file)
        rel_path = event_path.relative_to(runs_dir)
        
        # Detect intervention name, handling nested structures
        if len(rel_path.parts) > 1:
            # Check if second part is a subfolder like 'one_layer' or 'whole_network'
            first_part = rel_path.parts[0]
            second_part = rel_path.parts[1] if len(rel_path.parts) > 1 else ""
            
            # If second part looks like a variant (not a seed folder), combine them
            if second_part in ['one_layer', 'whole_network', 'last_layer', 'full_network']:
                intervention = f"{first_part}_{second_part}"
            else:
                intervention = first_part
        else:
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
            
            metrics = extract_run_metrics(scalars, last_k_rank=args.last_k_rank, metric_type=metric_type)
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
        seed_curves_dict = {}
        for s in seeds:
            curves = seed_metrics[s]["curves"]
            if curves["eval_iqm"][0] is not None:
                eval_curves.append(curves["eval_iqm"])
                # Store individual seed curve
                steps, values = curves["eval_iqm"]
                seed_curves_dict[s] = (np.array(steps), np.array(values))
        
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
            "seed_curves": seed_curves_dict,
            "seed_metrics": seed_metrics,  # Store seed metrics for additional plots
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
        display_name = get_display_name(intervention)
        out_path = individual_dir / f"{intervention}_iqm_return.png"
        
        plot_individual_intervention(
            intervention,
            data["aggregated_curve"],
            seed_curves=data.get("seed_curves"),
            n_seeds=data["n_seeds"],
            out_path=out_path,
            global_y_min=global_y_min,
            global_y_max=global_y_max,
            task_length=args.task_length,
            num_tasks=args.num_tasks,
            metric_type=metric_type,
        )
        print(f"  ✓ Saved {display_name} plot to {out_path}")
    
    # 9) Create combined plot with all interventions
    print("\nCreating combined plot...")
    combined_path = out_dir / "all_interventions_combined.png"
    plot_combined_interventions(
        all_intervention_data,
        combined_path,
        task_length=args.task_length,
        num_tasks=args.num_tasks,
        metric_type=metric_type,
    )
    print(f"  ✓ Saved combined plot to {combined_path}")
    
    # 10) Create grouped comparison plots
    print("\nCreating grouped comparison plots...")
    grouped_dir = out_dir / "grouped_comparisons"
    plot_grouped_comparisons(
        all_intervention_data,
        grouped_dir,
        task_length=args.task_length,
        num_tasks=args.num_tasks,
        metric_type=metric_type,
    )
    print(f"  ✓ Saved grouped comparison plots to {grouped_dir}")
    
    # 10.5) Create additional metric plots (forgetting, effective_rank, dormant_frac)
    # NOTE: These metrics are NOT separate for train vs eval - they represent global training properties:
    #   - Forgetting: Based on EVAL returns over time
    #   - Dormant fraction: Measured during TRAINING (optimizer steps)
    #   - Effective rank: Computed during EVAL phase
    # Therefore, we only plot these for "eval" mode to avoid duplication
    if metric_type == "eval":
        print("\nCreating additional metric plots (forgetting, effective_rank, dormant_frac)...")
        additional_metrics_dir = out_dir / "additional_metrics"
        plot_additional_metrics(
            all_intervention_data,
            additional_metrics_dir,
            bootstrap=args.bootstrap,
            alpha=args.alpha,
            statistic=args.statistic,
            task_length=args.task_length,
            num_tasks=args.num_tasks,
            use_optimizer_steps_for_dormant=True
        )
        print(f"  ✓ Saved additional metric plots to {additional_metrics_dir}")
    else:
        print("\n  ℹ Skipping additional metrics for train mode (they're the same as eval mode)")

    
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
        f.write(f"  • Final IQM Return: {metric_type.upper()} performance at END of training (last value)\n")
        f.write(f"  • Peak IQM Return: BEST {metric_type.upper()} performance achieved during training (max value)\n")
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
    print(f"  • {len(all_intervention_data)} individual intervention plots ({metric_type.upper()} IQM return)")
    if metric_type == "eval":
        print(f"  • {len(all_intervention_data) * 3} additional metric plots (forgetting, effective_rank, dormant_frac)")
    print(f"  • 1 combined comparison plot")
    print(f"  • 3 grouped comparison plots")
    print(f"  • 1 metrics CSV table")
    print(f"  • 1 formatted metrics text file")
    print()


if __name__ == "__main__":
    main()
