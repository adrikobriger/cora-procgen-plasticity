#!/usr/bin/env python3
"""
Plot continual evaluation IQM returns with 95% confidence intervals across seeds.

Reads TensorBoard event files from a directory tree (e.g., runs/) and produces
publication-quality matplotlib plots of IQM returns averaged across tasks,
with bootstrap confidence intervals across multiple seeds per method.
"""

import argparse
import re
import os
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def find_event_files(runs_dir: Path) -> Dict[str, List[Path]]:
    """
    Find all TensorBoard event files under runs_dir.
    
    Returns:
        Dict mapping run directory (str) -> list of event file paths
    """
    run_events = defaultdict(list)
    
    for root, dirs, files in os.walk(runs_dir):
        event_files = [f for f in files if f.startswith('events.out.tfevents')]
        if event_files:
            root_path = Path(root)
            for event_file in event_files:
                run_events[str(root_path)].append(root_path / event_file)
    
    return dict(run_events)


def extract_run_series(
    event_files: List[Path],
    tag_prefix: str,
    min_points: int
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Extract time series from TensorBoard event files for a single run.
    
    For each step, computes the mean IQM across all tasks logged at that step.
    
    Args:
        event_files: List of event file paths for this run
        tag_prefix: Scalar tag prefix (e.g., 'eval_reward_iqm/')
        min_points: Minimum number of data points required
        
    Returns:
        Tuple of (steps, values) arrays, or None if insufficient data
    """
    # Accumulator for all events
    ea = event_accumulator.EventAccumulator(str(event_files[0].parent))
    ea.Reload()
    
    # Find all scalar tags matching the prefix
    all_tags = ea.Tags().get('scalars', [])
    matching_tags = [tag for tag in all_tags if tag.startswith(tag_prefix)]
    
    if not matching_tags:
        return None
    
    # Build dict: step -> list of task IQM values at that step
    step_values = defaultdict(list)
    
    for tag in matching_tags:
        try:
            scalar_events = ea.Scalars(tag)
            for event in scalar_events:
                step_values[event.step].append(event.value)
        except KeyError:
            continue
    
    if not step_values or len(step_values) < min_points:
        return None
    
    # Compute mean IQM across tasks at each step
    steps = sorted(step_values.keys())
    values = [np.mean(step_values[step]) for step in steps]
    
    return np.array(steps), np.array(values)


def find_metadata_file(run_dir: Path) -> Optional[Path]:
    """
    Find metadata/config file in run directory.
    
    Looks for common metadata files:
    - run_metadata.json
    - config.json
    - args.json
    - hparams.yaml
    
    Args:
        run_dir: Path to run directory
        
    Returns:
        Path to metadata file or None if not found
    """
    candidates = [
        'run_metadata.json',
        'config.json',
        'args.json',
        'hparams.yaml',
        'params.json',
        'metadata.json'
    ]
    
    for candidate in candidates:
        path = run_dir / candidate
        if path.exists():
            return path
    
    return None


def parse_method_from_metadata(metadata_path: Path) -> Optional[str]:
    """
    Parse method name from metadata file.
    
    Maps intervention types to clean method names:
    - "none" or missing intervention → "Dense PPO"
    - "reset" → "Reset"
    - "redo" → "ReDo"
    - "gmp" → "GMP"
    - "set" → "SET"
    - "partial_reinit" → "Partial Reinit"
    
    Args:
        metadata_path: Path to metadata file (JSON or YAML)
        
    Returns:
        Method name or None if parsing fails
    """
    try:
        if metadata_path.suffix == '.json':
            with open(metadata_path, 'r') as f:
                data = json.load(f)
        elif metadata_path.suffix in ['.yaml', '.yml']:
            try:
                import yaml
                with open(metadata_path, 'r') as f:
                    data = yaml.safe_load(f)
            except ImportError:
                return None
        else:
            return None
        
        # Look for intervention type in various possible locations
        intervention_type = None
        
        # Try common key patterns
        if isinstance(data, dict):
            # Direct key
            if 'intervention' in data:
                if isinstance(data['intervention'], dict):
                    intervention_type = data['intervention'].get('type')
                else:
                    intervention_type = data['intervention']
            elif 'intervention_type' in data:
                intervention_type = data['intervention_type']
            # Nested in policy or experiment config
            elif 'policy' in data and isinstance(data['policy'], dict):
                if 'intervention' in data['policy']:
                    if isinstance(data['policy']['intervention'], dict):
                        intervention_type = data['policy']['intervention'].get('type')
                    else:
                        intervention_type = data['policy']['intervention']
            elif 'experiment' in data and isinstance(data['experiment'], dict):
                if 'intervention' in data['experiment']:
                    if isinstance(data['experiment']['intervention'], dict):
                        intervention_type = data['experiment']['intervention'].get('type')
                    else:
                        intervention_type = data['experiment']['intervention']
            # Check experiment name for clues
            elif 'experiment_name' in data:
                exp_name = data['experiment_name'].lower()
                if 'reset' in exp_name:
                    intervention_type = 'reset'
                elif 'redo' in exp_name:
                    intervention_type = 'redo'
                elif 'gmp' in exp_name:
                    intervention_type = 'gmp'
                elif 'set' in exp_name:
                    intervention_type = 'set'
                elif 'partial' in exp_name or 'reinit' in exp_name:
                    intervention_type = 'partial_reinit'
                elif 'dense' in exp_name:
                    intervention_type = 'none'
        
        if intervention_type:
            intervention_type = str(intervention_type).lower()
            
            # Map to clean method name
            mapping = {
                'none': 'Dense PPO',
                'reset': 'Reset',
                'redo': 'ReDo',
                'gmp': 'GMP',
                'set': 'SET',
                'partial_reinit': 'Partial Reinit',
                'partial': 'Partial Reinit',
            }
            
            return mapping.get(intervention_type)
        
    except Exception as e:
        # Silent failure - will fall back to folder name
        pass
    
    return None


def parse_method_from_path(run_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse method name from path components (NOT from final folder name).
    
    Looks for method tokens in path components:
    - "dense" → "Dense PPO"
    - "reset" → "Reset"
    - "redo" → "ReDo"
    - "gmp" → "GMP"
    - "set" → "SET"
    - "partial_reinit" or "partial" → "Partial Reinit"
    
    Args:
        run_path: Full path to run directory
        
    Returns:
        Tuple of (method_name, path_component) where path_component is the part that matched
    """
    path = Path(run_path)
    parts = path.parts
    
    # Method token mapping
    method_mapping = {
        'dense': 'Dense PPO',
        'partial_reinit': 'Partial Reinit',
        'partial': 'Partial Reinit',
        'reset': 'Reset',
        'redo': 'ReDo',
        'gmp': 'GMP',
        'set': 'SET',
    }
    
    # Search through path components (excluding the last one which is the run name)
    for part in parts[:-1]:
        part_lower = part.lower()
        for token, method_name in method_mapping.items():
            if token == part_lower or part_lower.startswith(token + '_') or part_lower.endswith('_' + token):
                return method_name, part
    
    return None, None


def parse_seed_from_path(run_path: str) -> Tuple[Optional[int], Optional[str]]:
    """
    Parse seed from path components looking for "seed_<k>" pattern.
    
    Args:
        run_path: Full path to run directory
        
    Returns:
        Tuple of (seed, path_component) where path_component is the part that matched
    """
    path = Path(run_path)
    parts = path.parts
    
    # Look for seed_<k> pattern in path components
    for part in parts:
        match = re.match(r'^seed[_-]?(\d+)$', part.lower())
        if match:
            return int(match.group(1)), part
    
    return None, None


def parse_seed_from_metadata(metadata_path: Path) -> Optional[int]:
    """
    Parse seed from metadata file.
    
    Args:
        metadata_path: Path to metadata file (JSON or YAML)
        
    Returns:
        Seed number or None if not found
    """
    try:
        if metadata_path.suffix == '.json':
            with open(metadata_path, 'r') as f:
                data = json.load(f)
        elif metadata_path.suffix in ['.yaml', '.yml']:
            try:
                import yaml
                with open(metadata_path, 'r') as f:
                    data = yaml.safe_load(f)
            except ImportError:
                return None
        else:
            return None
        
        # Look for seed in various possible locations
        if isinstance(data, dict):
            if 'seed' in data:
                return int(data['seed'])
            elif 'random_seed' in data:
                return int(data['random_seed'])
            elif 'rng_seed' in data:
                return int(data['rng_seed'])
            elif 'experiment' in data and isinstance(data['experiment'], dict):
                if 'seed' in data['experiment']:
                    return int(data['experiment']['seed'])
        
    except Exception:
        pass
    
    return None


def parse_metadata(
    run_path: str,
    method_regex: Optional[str],
    seed_regex: Optional[str],
    verbose: bool = False
) -> Tuple[Optional[str], int, Optional[str], str, str]:
    """
    Parse method name and seed from run path and metadata files.
    
    Args:
        run_path: Full path to run directory
        method_regex: Optional regex to extract method name
        seed_regex: Optional regex to extract seed
        verbose: If True, print debug info
        
    Returns:
        Tuple of (method_name, seed, metadata_file_path, method_source, seed_source).
        method_name can be None if no match.
    """
    run_dir = Path(run_path)
    
    # Find metadata file for debugging
    metadata_file = find_metadata_file(run_dir)
    
    # Parse method from path components
    method = None
    method_source = "none"
    
    if method_regex:
        # User provided explicit regex
        match = re.search(method_regex, run_path)
        if match:
            method = match.group(1)
            method_source = f"regex: {method_regex}"
    else:
        # Parse from path components
        method, method_component = parse_method_from_path(run_path)
        if method:
            method_source = f"path component: {method_component}"
    
    # Parse seed from path or metadata
    seed = 0
    seed_source = "default"
    
    if seed_regex:
        # User provided explicit regex
        match = re.search(seed_regex, run_path)
        if match:
            seed = int(match.group(1))
            seed_source = f"regex: {seed_regex}"
    else:
        # First try to parse from path (seed_<k> folder)
        seed_from_path, seed_component = parse_seed_from_path(run_path)
        if seed_from_path is not None:
            seed = seed_from_path
            seed_source = f"path component: {seed_component}"
        elif metadata_file:
            # Fallback to metadata file
            seed_from_metadata = parse_seed_from_metadata(metadata_file)
            if seed_from_metadata is not None:
                seed = seed_from_metadata
                seed_source = f"metadata: {metadata_file.name}"
    
    return method, seed, str(metadata_file) if metadata_file else None, method_source, seed_source


def group_runs(
    run_events: Dict[str, List[Path]],
    tag_prefix: str,
    min_points: int,
    method_regex: Optional[str],
    seed_regex: Optional[str],
    verbose: bool = True
) -> Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]]:
    """
    Group runs by method and seed, extract time series for each.
    
    Returns:
        Nested dict: method -> seed -> (steps, values)
    """
    methods = defaultdict(dict)
    skipped_no_data = 0
    skipped_no_method = 0
    
    if verbose:
        print("\nProcessing runs:")
        print("-" * 80)
    
    for run_path, event_files in sorted(run_events.items()):
        if verbose:
            print(f"\nRun: {run_path}")
        
        series = extract_run_series(event_files, tag_prefix, min_points)
        
        method, seed, metadata_file, method_source, seed_source = parse_metadata(
            run_path, method_regex, seed_regex, verbose
        )
        
        if verbose:
            print(f"  Metadata file: {metadata_file if metadata_file else 'none'}")
            print(f"  Method: {method if method else 'UNRECOGNIZED'} (from {method_source})")
            print(f"  Seed: {seed} (from {seed_source})")
        
        if series is None:
            if verbose:
                print(f"  IQM points: 0 (skipped - insufficient data)")
            skipped_no_data += 1
            continue
        
        steps, values = series
        if verbose:
            print(f"  IQM points: {len(steps)}")
        
        if method is None:
            if verbose:
                print(f"  → SKIPPED (no method recognized)")
            skipped_no_method += 1
            continue
        
        if verbose:
            print(f"  → Added to method '{method}', seed {seed}")
        
        methods[method][seed] = (steps, values)
    
    if verbose:
        print("\n" + "=" * 80)
        print(f"Summary: {len(methods)} methods, {skipped_no_data} runs with no data, "
              f"{skipped_no_method} runs with unrecognized methods")
        print("=" * 80)
    
    return dict(methods)


def align_and_interpolate(
    seed_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    grid_step: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align multiple seeds to a common grid via interpolation.
    
    Args:
        seed_data: Dict mapping seed -> (steps, values)
        grid_step: Spacing for common x-grid
        
    Returns:
        Tuple of (grid, aligned_values) where aligned_values has shape [num_seeds, num_grid_points]
    """
    if not seed_data:
        return np.array([]), np.array([])
    
    # Find overlap range across all seeds
    min_steps = []
    max_steps = []
    
    for steps, values in seed_data.values():
        if len(steps) > 0:
            min_steps.append(steps.min())
            max_steps.append(steps.max())
    
    if not min_steps:
        return np.array([]), np.array([])
    
    start = max(min_steps)
    end = min(max_steps)
    
    if start >= end:
        return np.array([]), np.array([])
    
    # Create common grid
    grid = np.arange(start, end + 1, grid_step)
    if len(grid) == 0:
        return np.array([]), np.array([])
    
    # Interpolate each seed to the grid
    aligned = []
    for seed in sorted(seed_data.keys()):
        steps, values = seed_data[seed]
        # Only interpolate within the valid range
        interp_values = np.interp(grid, steps, values)
        aligned.append(interp_values)
    
    return grid, np.array(aligned)


def bootstrap_ci(
    aligned_values: np.ndarray,
    n_bootstrap: int = 2000,
    ci_percentiles: Tuple[float, float] = (2.5, 97.5),
    rng_seed: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mean and bootstrap confidence intervals across seeds.
    
    Args:
        aligned_values: Array of shape [num_seeds, num_points]
        n_bootstrap: Number of bootstrap resamples
        ci_percentiles: Percentiles for CI bounds (default 95% CI)
        rng_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (mean, lower_ci, upper_ci) each of shape [num_points]
    """
    num_seeds, num_points = aligned_values.shape
    
    if num_seeds < 2:
        # No CI possible with single seed
        return aligned_values[0], aligned_values[0], aligned_values[0]
    
    # Compute mean
    mean = aligned_values.mean(axis=0)
    
    # Bootstrap
    rng = np.random.RandomState(rng_seed)
    bootstrap_means = np.zeros((n_bootstrap, num_points))
    
    for i in range(n_bootstrap):
        # Sample seeds with replacement
        indices = rng.choice(num_seeds, size=num_seeds, replace=True)
        bootstrap_sample = aligned_values[indices]
        bootstrap_means[i] = bootstrap_sample.mean(axis=0)
    
    # Compute percentiles
    lower_ci = np.percentile(bootstrap_means, ci_percentiles[0], axis=0)
    upper_ci = np.percentile(bootstrap_means, ci_percentiles[1], axis=0)
    
    return mean, lower_ci, upper_ci


def format_step_tick(x, pos):
    """Format tick labels as 100k, 200k, etc."""
    return f'{int(x/1000)}k'


def plot_method(
    method_name: str,
    seed_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    grid: np.ndarray,
    aligned_values: np.ndarray,
    mean: np.ndarray,
    lower_ci: np.ndarray,
    upper_ci: np.ndarray,
    out_path: Path,
    formats: List[str] = ['png'],
    task_length: int = 500_000,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
    save_data: bool = True,
    save_aligned: bool = False,
    rng_seed: int = 0
):
    """
    Create publication-quality plot for a single method showing seed curves and mean with CI.
    
    Args:
        method_name: Name of the method
        seed_data: Dict mapping seed -> (steps, values) - original data
        grid: Common x-grid (environment steps)
        aligned_values: Interpolated values [num_seeds, num_points]
        mean: Mean across seeds
        lower_ci: Lower confidence bound
        upper_ci: Upper confidence bound
        out_path: Output file path (extension will be replaced for each format)
        formats: List of output formats (e.g., ['png', 'pdf'])
        task_length: Length of each task in environment steps (default 500,000)
        ymin: Optional minimum y-axis value
        ymax: Optional maximum y-axis value
    """
    # Set publication-quality font
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
    plt.rcParams['mathtext.fontset'] = 'stix'  # For math symbols
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color for this method
    color = '#2E86AB'  # Professional blue
    
    # Plot individual seed curves (thin, transparent, no legend)
    for i, seed in enumerate(sorted(seed_data.keys())):
        ax.plot(grid, aligned_values[i], color=color, alpha=0.25, linewidth=1, 
                label='_nolegend_', zorder=1)
    
    # Plot mean curve (thick, labeled)
    ax.plot(grid, mean, label=method_name, color=color, linewidth=3, zorder=3)
    
    # Plot CI band if we have multiple seeds
    num_seeds = aligned_values.shape[0]
    if num_seeds >= 2 and not np.array_equal(lower_ci, upper_ci):
        ax.fill_between(grid, lower_ci, upper_ci, alpha=0.20, color=color, 
                       label='95% CI', zorder=2)
    
    # Add task boundaries and labels
    x_min, x_max = grid.min(), grid.max()
    y_min, y_max = ax.get_ylim()
    
    # Calculate number of tasks
    num_tasks = int(np.ceil(x_max / task_length))
    
    # Draw vertical lines at task boundaries
    for k in range(1, num_tasks):
        boundary = k * task_length
        if x_min < boundary < x_max:
            ax.axvline(boundary, linestyle='--', color='black', alpha=0.6, linewidth=1.5, zorder=4)
    
    # Add task labels
    # Position at 95% of y-range
    label_y = y_min + 0.95 * (y_max - y_min)
    
    for k in range(num_tasks):
        # Center of task k (0-indexed)
        task_center = (k + 0.5) * task_length
        
        # Only label if center is within visible range
        if x_min < task_center < x_max:
            ax.text(task_center, label_y, f'Task {k+1}', 
                   horizontalalignment='center', verticalalignment='top',
                   fontsize=12, fontweight='normal', alpha=0.8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='none', alpha=0.7))
    
    # Format x-axis with custom ticks every 100k
    from matplotlib.ticker import FuncFormatter, MultipleLocator
    ax.xaxis.set_major_locator(MultipleLocator(100000))
    ax.xaxis.set_major_formatter(FuncFormatter(format_step_tick))
    
    # Labels and title
    ax.set_xlabel('Environment steps', fontsize=12)
    ax.set_ylabel('Eval IQM return (avg across tasks)', fontsize=12)
    ax.set_title(f'{method_name} – Continual IQM Return', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Set x-limits to the computed overlap range
    ax.set_xlim(grid.min(), grid.max())
    
    # Set y-limits if specified
    if ymin is not None or ymax is not None:
        current_ymin, current_ymax = ax.get_ylim()
        ax.set_ylim(ymin if ymin is not None else current_ymin,
                    ymax if ymax is not None else current_ymax)
    
    # Tidy up
    plt.tight_layout()
    
    # Save in all requested formats
    out_path.parent.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    
    for fmt in formats:
        # Replace extension in output path
        fmt_path = out_path.with_suffix(f'.{fmt}')
        plt.savefig(fmt_path, format=fmt, dpi=300, bbox_inches='tight')
        saved_paths.append(str(fmt_path))
    
    plt.close()
    
    print(f"  ✓ Saved plot to {', '.join(saved_paths)}")

    # Save processed data so plots can be reconstructed without TensorBoard
    if save_data:
        data_json = {
            "method": method_name,
            "task_length": int(task_length),
            "grid_step": int(grid[1] - grid[0]) if len(grid) > 1 else None,
            "steps": grid.tolist(),
            "mean": mean.tolist(),
            "lower_ci": lower_ci.tolist(),
            "upper_ci": upper_ci.tolist(),
            "seeds": sorted(list(seed_data.keys())),
            "num_seeds": int(aligned_values.shape[0]),
            "rng_seed": int(rng_seed),
        }
        json_path = out_path.with_name("iqm_return_ci_data.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data_json, f, indent=2)

        # Optional: save aligned per-seed values for exact reconstruction
        if save_aligned:
            npz_path = out_path.with_name("iqm_return_ci_data.npz")
            np.savez_compressed(
                npz_path,
                steps=grid,
                mean=mean,
                lower_ci=lower_ci,
                upper_ci=upper_ci,
                aligned=aligned_values,
                seeds=np.array(sorted(list(seed_data.keys())), dtype=np.int64),
            )


def main():
    parser = argparse.ArgumentParser(
        description="Plot continual IQM return curves with 95% CI from TensorBoard logs"
    )
    parser.add_argument('--runs_dir', type=str, default='runs',
                        help='Directory containing run folders')
    parser.add_argument('--out_dir', type=str, default='plots',
                        help='Output directory for plots')
    parser.add_argument('--tag_prefix', type=str, default='eval_reward_iqm/',
                        help='TensorBoard scalar tag prefix')
    parser.add_argument('--grid_step', type=int, default=50000,
                        help='Common x-grid spacing in environment steps')
    parser.add_argument('--bootstrap', type=int, default=2000,
                        help='Number of bootstrap resamples for CI')
    parser.add_argument('--seed_regex', type=str, default=None,
                        help='Regex to extract seed from path (group 1)')
    parser.add_argument('--method_regex', type=str, default=None,
                        help='Regex to extract method name from path (group 1)')
    parser.add_argument('--min_points', type=int, default=5,
                        help='Minimum number of logged points required per run')
    parser.add_argument('--formats', type=str, default='png',
                        help='Output file format(s), comma-separated (e.g., "png,pdf" or "both" for both)')
    parser.add_argument('--rng_seed', type=int, default=0,
                        help='Random seed for bootstrap')
    parser.add_argument('--task_length', type=int, default=500000,
                        help='Length of each task in environment steps (for task boundary lines)')
    parser.add_argument('--ymin', type=float, default=None,
                        help='Minimum y-axis value (for consistent scaling across methods)')
    parser.add_argument('--ymax', type=float, default=None,
                        help='Maximum y-axis value (for consistent scaling across methods)')
    parser.add_argument('--save_data', action='store_true', default=True,
                        help='Save processed data (grid/mean/CI) as JSON for manual reconstruction')
    parser.add_argument('--no_save_data', action='store_true', default=False,
                        help='Disable saving processed data')
    parser.add_argument('--save_aligned', action='store_true', default=False,
                        help='Also save aligned per-seed values as NPZ (larger)')
    
    args = parser.parse_args()
    
    # Parse formats
    if args.formats.lower() == 'both':
        formats = ['png', 'pdf']
    else:
        formats = [fmt.strip() for fmt in args.formats.split(',')]
    
    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    
    if not runs_dir.exists():
        print(f"Error: runs_dir '{runs_dir}' does not exist")
        return 1
    
    print(f"Scanning {runs_dir} for TensorBoard event files...")
    run_events = find_event_files(runs_dir)
    print(f"Found {len(run_events)} run directories with event files")
    
    if not run_events:
        print("No event files found!")
        return 1
    
    print(f"\nExtracting time series (tag_prefix='{args.tag_prefix}')...")
    methods = group_runs(
        run_events,
        args.tag_prefix,
        args.min_points,
        args.method_regex,
        args.seed_regex,
        verbose=True
    )
    
    if not methods:
        print(f"No valid runs found with tag prefix '{args.tag_prefix}'")
        return 1
    
    print(f"\nFound {len(methods)} method(s):")
    for method, seeds in methods.items():
        print(f"  - {method}: {len(seeds)} seed(s) (seeds: {sorted(seeds.keys())})")
    
    print(f"\nAligning and computing bootstrap CI (grid_step={args.grid_step})...")
    print(f"Generating plots (one per method)...\n")
    
    plots_created = 0
    
    for method, seed_data in methods.items():
        print(f"Processing {method}...")
        
        grid, aligned = align_and_interpolate(seed_data, args.grid_step)
        
        if len(grid) == 0:
            print(f"  ⚠ Skipping {method}: no overlap in step ranges\n")
            continue
        
        mean, lower_ci, upper_ci = bootstrap_ci(
            aligned,
            n_bootstrap=args.bootstrap,
            rng_seed=args.rng_seed
        )
        
        print(f"  • {len(seed_data)} seeds used")
        print(f"  • {len(grid)} grid points")
        print(f"  • Step range: [{grid.min():,}, {grid.max():,}]")
        
        # Create output directory and path for this method
        method_dir = out_dir / method
        # Use first format for base filename (will be replaced per format)
        out_path = method_dir / f"iqm_return_ci.{formats[0]}"
        
        # Plot this method
        plot_method(
            method_name=method,
            seed_data=seed_data,
            grid=grid,
            aligned_values=aligned,
            mean=mean,
            lower_ci=lower_ci,
            upper_ci=upper_ci,
            out_path=out_path,
            formats=formats,
            task_length=args.task_length,
            ymin=args.ymin,
            ymax=args.ymax,
            save_data=(not args.no_save_data) and args.save_data,
            save_aligned=args.save_aligned,
            rng_seed=args.rng_seed
        )
        
        plots_created += 1
        print()
    
    if plots_created == 0:
        print("No plots created! Check your data.")
        return 1
    
    print(f"✓ Done! Created {plots_created} plot(s) in {out_dir}/")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
