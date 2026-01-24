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
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tensorboard.backend.event_processing import event_accumulator

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from select_best_config import select_best_configs, write_best_configs


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


def extract_run_series_with_mode(
    event_files: List[Path],
    tag_prefix: str,
    min_points: int,
    allow_constant_step_sequence: bool = False
) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
    """
    Like extract_run_series, but can convert constant-step eval logs into a sequence.
    Returns (x, y, x_kind) where x_kind is 'steps' or 'index'.
    """
    ea = event_accumulator.EventAccumulator(str(event_files[0].parent))
    ea.Reload()

    all_tags = ea.Tags().get('scalars', [])
    matching_tags = [tag for tag in all_tags if tag.startswith(tag_prefix)]
    if not matching_tags:
        return None

    # Build dict: step -> list of task IQM values at that step
    step_values = defaultdict(list)
    steps_seen = set()

    for tag in matching_tags:
        try:
            scalar_events = ea.Scalars(tag)
            for event in scalar_events:
                step_values[event.step].append(event.value)
                steps_seen.add(event.step)
        except KeyError:
            continue

    if not step_values or len(step_values) < min_points:
        # If all eval steps collapse to a single step, we may still want to recover a sequence.
        if not allow_constant_step_sequence:
            return None

    # If all steps are identical and requested, use index-based series across tags.
    if allow_constant_step_sequence and len(steps_seen) == 1:
        tag_series = []
        min_len = None
        for tag in matching_tags:
            try:
                scalar_events = ea.Scalars(tag)
            except KeyError:
                continue
            values = [ev.value for ev in scalar_events]
            if not values:
                continue
            tag_series.append(values)
            min_len = len(values) if min_len is None else min(min_len, len(values))

        if not tag_series or min_len is None or min_len < min_points:
            return None

        # Average across tasks at each index
        vals = []
        for i in range(min_len):
            vals.append(float(np.mean([series[i] for series in tag_series])))
        x = np.arange(min_len, dtype=np.float64)
        return x, np.array(vals, dtype=np.float64), 'index'

    # Default path (step-based)
    steps = sorted(step_values.keys())
    values = [np.mean(step_values[step]) for step in steps]
    return np.array(steps), np.array(values), 'steps'


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
    
    # Look for seed_<k> pattern in path components (e.g., seed_0, seed_1_timestamp, etc.)
    for part in parts:
        match = re.match(r'^seed[_-]?(\d+)', part.lower())
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


def _seed_step_bounds(seed_data: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> Tuple[Optional[int], Optional[int]]:
    min_steps = []
    max_steps = []
    for steps, _ in seed_data.values():
        if len(steps) == 0:
            continue
        min_steps.append(int(steps.min()))
        max_steps.append(int(steps.max()))
    return (max(min_steps) if min_steps else None, min(max_steps) if max_steps else None)


def _interpolate_seeds_to_grid(
    seed_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    grid: np.ndarray,
    start: float,
    end: float
) -> Optional[np.ndarray]:
    aligned = []
    for steps, values in seed_data.values():
        if len(steps) == 0:
            continue
        if steps.min() > start or steps.max() < end:
            continue
        aligned.append(np.interp(grid, steps, values))
    if not aligned:
        return None
    return np.vstack(aligned)


def _normalize_curve(mean: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    divisor = max(float(mean.max()), float(lower.max()), float(upper.max()), 1e-8)
    return mean / divisor, lower / divisor, upper / divisor


def _load_json_config(config_path: Path) -> Optional[Dict]:
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _compute_common_range(
    methods: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]],
    method_names: List[str]
) -> Tuple[Optional[int], Optional[int]]:
    starts = []
    ends = []
    for method in method_names:
        seed_data = methods.get(method)
        if not seed_data:
            continue
        lower, upper = _seed_step_bounds(seed_data)
        if lower is None or upper is None:
            continue
        starts.append(lower)
        ends.append(upper)
    if not starts or not ends:
        return None, None
    return max(starts), min(ends)


def _plot_ablation_grid(
    methods: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]],
    config: Dict,
    args,
    out_dir: Path,
    formats: List[str]
) -> int:
    panels = config.get('panels', [])
    if not panels:
        print("Ablation config has no panels defined")
        return 0

    unique_methods = []
    for panel in panels:
        for series in panel.get('series', []):
            method_name = series.get('method')
            if method_name and method_name not in unique_methods:
                unique_methods.append(method_name)

    start, end = _compute_common_range(methods, unique_methods)
    if start is None or end is None or start >= end:
        print("Unable to compute overlapping step range for ablation figure")
        return 0

    grid_step = args.grid_step
    grid = np.arange(start, end + grid_step, grid_step)
    if len(grid) == 0:
        print("Computed grid has zero points for ablation figure")
        return 0

    # Precompute mean/CI for each method in the config
    prepared = {}
    for method in unique_methods:
        data = methods.get(method)
        if not data:
            continue
        aligned = _interpolate_seeds_to_grid(data, grid, start, end)
        if aligned is None:
            continue
        mean, lower, upper = bootstrap_ci(aligned, n_bootstrap=args.bootstrap, rng_seed=args.rng_seed)
        if args.normalize == 'max':
            mean, lower, upper = _normalize_curve(mean, lower, upper)
        prepared[method] = {
            'grid': grid,
            'mean': mean,
            'lower': lower,
            'upper': upper,
            'num_seeds': aligned.shape[0]
        }

    if not prepared:
        print("No methods had enough data for the ablation figure")
        return 0

    num_panels = len(panels)
    cols = min(num_panels, config.get('cols', num_panels))
    rows = int(np.ceil(num_panels / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    axes_flat = axes.flatten()

    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    legend_handles = []
    legend_labels = []

    x_scale = args.steps_per_epoch if args.steps_per_epoch > 0 else 1
    common_xlabel = config.get('x_label', 'Epoch' if x_scale != 1 else 'Environment steps')
    common_ylabel = config.get('y_label', 'Normalized IQM' if args.normalize == 'max' else 'Eval IQM return (avg across tasks)')

    for panel_idx, panel in enumerate(panels):
        ax = axes_flat[panel_idx]
        panel_title = panel.get('title', f'Panel {panel_idx+1}')
        series_style_cycle = iter(default_colors)
        panel_x_values = None
        panel_x_min = float('inf')
        panel_x_max = float('-inf')
        
        for series in panel.get('series', []):
            method_name = series.get('method')
            label = series.get('label') or method_name
            color = series.get('color')
            if not color:
                color = next(series_style_cycle, None)
            prepared_series = prepared.get(method_name)
            if not prepared_series:
                print(f"  ⚠ Missing data for '{method_name}' (panel '{panel_title}')")
                continue
            x_values = prepared_series['grid'] / x_scale
            mean = prepared_series['mean']
            lower = prepared_series['lower']
            upper = prepared_series['upper']

            ax.plot(x_values, mean, label=label, color=color, linewidth=2.5)
            if args.bootstrap > 0 and prepared_series['num_seeds'] > 1:
                ax.fill_between(x_values, lower, upper, color=color, alpha=0.25)

            if panel_idx == 0 and label not in legend_labels:
                legend_handles.append(Line2D([], [], color=color, linewidth=2.5))
                legend_labels.append(label)
            panel_x_values = x_values
            panel_x_min = min(panel_x_min, x_values.min())
            panel_x_max = max(panel_x_max, x_values.max())

        ax.set_title(panel_title, fontsize=12, fontweight='semibold')
        ax.set_xlabel(common_xlabel)
        ax.set_ylabel(common_ylabel if panel_idx % cols == 0 else '')
        ax.grid(True, alpha=0.3)
        
        # Add task boundaries and labels
        if args.task_length > 0 and panel_x_min != float('inf'):
            num_tasks = int(args.num_tasks) if args.num_tasks is not None else int(np.ceil((panel_x_max * x_scale) / args.task_length))
            
            # Draw vertical lines at task boundaries
            for k in range(1, num_tasks + 1):
                boundary = (k * args.task_length) / x_scale
                if panel_x_min < boundary <= panel_x_max:
                    ax.axvline(boundary, linestyle='--', color='black', alpha=0.6, linewidth=1.5, zorder=1)
            
            # Add task labels at 95% of y-range
            y_min, y_max = ax.get_ylim()
            label_y = y_min + 0.95 * (y_max - y_min)
            for k in range(num_tasks):
                task_center = ((k + 0.5) * args.task_length) / x_scale
                if panel_x_min < task_center < panel_x_max:
                    ax.text(task_center, label_y, f'Task {k+1}', 
                           horizontalalignment='center', verticalalignment='top',
                           fontsize=11, fontweight='normal', alpha=0.8,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                    edgecolor='none', alpha=0.7),
                           zorder=4)
        
        if panel_x_values is not None:
            ax.set_xlim(panel_x_values.min(), panel_x_values.max())
        panel_ylim = panel.get('ylim')
        if panel_ylim and len(panel_ylim) == 2:
            ax.set_ylim(panel_ylim[0], panel_ylim[1])

    # Turn off unused subplots
    for unused in axes_flat[num_panels:]:
        unused.axis('off')

    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc='upper center', ncol=len(legend_handles), borderaxespad=0.5)
        fig.subplots_adjust(top=0.88)

    fig.suptitle(config.get('figure_title', 'IQM Return Ablation'), fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    out_path = out_dir / args.ablation_plot_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    saved = 0
    for fmt in formats:
        fmt_path = out_path.with_suffix(f'.{fmt}')
        fig.savefig(fmt_path, format=fmt, dpi=300, bbox_inches='tight')
        saved += 1

    plt.close(fig)
    print(f"  ✓ Saved ablation figure to {out_path.with_suffix('.' + formats[0])} (plus {saved-1} more)")
    return saved


def _plot_train_summary(
    methods: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]],
    config: Dict,
    args,
    out_dir: Path,
    formats: List[str]
) -> int:
    """
    Plot a single train IQM curve that overlays each intervention and highlights the best one.
    """
    series = config.get('series', [])
    if not series:
        print("Summary config is empty or missing the 'series' list")
        return 0

    method_names = [entry.get('method') for entry in series if entry.get('method')]
    if not method_names:
        print("Summary config does not specify any method names")
        return 0

    start, end = _compute_common_range(methods, method_names)
    if start is None or end is None or start >= end:
        print("Unable to compute overlapping step range for the train summary figure")
        return 0

    grid_step = args.grid_step
    grid = np.arange(start, end + grid_step, grid_step)
    if len(grid) == 0:
        print("Computed grid has zero points for the train summary figure")
        return 0

    prepared = {}
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_iter = iter(default_colors)

    for entry in series:
        method = entry.get('method')
        if not method:
            continue
        seed_data = methods.get(method)
        if not seed_data:
            continue
        aligned = _interpolate_seeds_to_grid(seed_data, grid, start, end)
        if aligned is None:
            continue
        mean = aligned.mean(axis=0)
        color = entry.get('color')
        prepared[method] = {
            'grid': grid,
            'mean': mean,
            'aligned': aligned,
            'label': entry.get('label', method),
            'details': entry.get('details', ''),
            'color': color,
            'line_style': entry.get('line_style', '-'),
            'intervention': entry.get('intervention') or method,
            'num_seeds': aligned.shape[0],
        }

    if not prepared:
        print("No methods had enough data for the train summary figure")
        return 0

    # Determine best-performing method (highest final value)
    best_method = max(prepared.items(), key=lambda kv: kv[1]['mean'][-1])[0]
    best_entry = prepared[best_method]
    if best_entry['num_seeds'] > 0:
        mean, lower_ci, upper_ci = bootstrap_ci(
            best_entry['aligned'],
            n_bootstrap=args.bootstrap,
            rng_seed=args.rng_seed
        )
        best_entry['mean'] = mean
        best_entry['lower'] = lower_ci
        best_entry['upper'] = upper_ci
    best_entry['is_best'] = True

    # Plotting
    figsize = tuple(config.get('figsize', (12, 6)))
    fig, ax = plt.subplots(figsize=figsize)
    global_x_min = float('inf')
    global_x_max = float('-inf')
    color_cycle = iter(default_colors)
    x_scale = args.steps_per_epoch if args.steps_per_epoch > 0 else 1
    x_label = config.get('x_label', 'Epoch' if x_scale != 1 else 'Environment steps')
    y_label = config.get('y_label', 'Eval IQM return (avg across tasks)')

    for entry in series:
        method = entry.get('method')
        plot_entry = prepared.get(method)
        if not plot_entry:
            continue
        color = plot_entry['color'] or next(color_cycle, '#333333')
        x_values = plot_entry['grid'] / x_scale
        label = plot_entry['label']
        if plot_entry['details']:
            label = f"{label} ({plot_entry['details']})"
        if plot_entry.get('is_best'):
            label = f"{label} (best)"
        ax.plot(x_values, plot_entry['mean'], label=label, color=color,
                linewidth=3 if plot_entry.get('is_best') else 2,
                linestyle=plot_entry.get('line_style', '-'), zorder=3)

        if plot_entry.get('is_best') and 'lower' in plot_entry and 'upper' in plot_entry:
            ax.fill_between(x_values, plot_entry['lower'], plot_entry['upper'],
                            color=color, alpha=0.25, zorder=2)

        global_x_min = min(global_x_min, float(x_values.min()))
        global_x_max = max(global_x_max, float(x_values.max()))

    if global_x_min == float('inf') or global_x_max == float('-inf'):
        print("Train summary figure has no plotted range")
        plt.close(fig)
        return 0

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(config.get('title', 'Train IQM Return Across Interventions'), fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    # If the user forces a known number of tasks, extend the visible x-range *before*
    # drawing task-region labels. Otherwise Task N won't be labeled if the data ends early.
    x_right = global_x_max
    if args.num_tasks is not None and args.task_length > 0:
        expected_xmax = (int(args.num_tasks) * args.task_length) / x_scale
        if expected_xmax > x_right:
            x_right = expected_xmax

    ax.set_xlim(global_x_min, x_right)
    ax.legend(fontsize=10, framealpha=0.9)

    # Annotate task boundaries and label each task region (Task 1..N)
    # Use the *visible* range for task annotations.
    _, visible_x_max = ax.get_xlim()
    x_max_steps = visible_x_max * x_scale
    if args.task_length > 0:
        num_tasks = int(args.num_tasks) if args.num_tasks is not None else int(np.ceil(x_max_steps / args.task_length))

        # Boundaries at task transitions (include end boundary if it lands on x_max)
        for k in range(1, num_tasks + 1):
            boundary_steps = k * args.task_length
            boundary = boundary_steps / x_scale
            if global_x_min < boundary <= global_x_max:
                ax.axvline(boundary, linestyle='--', color='black', alpha=0.5, linewidth=1.2, zorder=1)

        # Labels at task centers
        y_min, y_max = ax.get_ylim()
        label_y = y_min + 0.95 * (y_max - y_min)
        for k in range(num_tasks):
            center_steps = (k + 0.5) * args.task_length
            center = center_steps / x_scale
            if global_x_min < center < global_x_max:
                ax.text(center, label_y, f'Task {k+1}', ha='center', va='top', fontsize=11,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.7))

        # x-axis already extended above when forcing num_tasks

    plt.tight_layout()

    out_path = out_dir / args.summary_plot_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fmt_path = out_path.with_suffix(f'.{fmt}')
        fig.savefig(fmt_path, format=fmt, dpi=300, bbox_inches='tight')

    plt.close(fig)
    print(f"  ✓ Saved train summary figure to {out_path.with_suffix('.' + formats[0])} (plus {len(formats)-1} more)")
    return len(formats)


def format_step_tick(x, pos):
    """Format tick labels as 100k, 200k, 1.0M, 1.1M, etc."""
    try:
        x = float(x)
    except Exception:
        return ""
    if abs(x) >= 1_000_000:
        return f"{x/1_000_000:.1f}M"
    return f"{int(x/1000)}k"


def _format_params_label(params: dict, max_len: int = 80) -> str:
    if not isinstance(params, dict) or not params:
        return "(no params)"
    parts = [f"{k}={params[k]}" for k in sorted(params.keys())]
    label = ", ".join(parts)
    if len(label) > max_len:
        label = label[:max_len - 3] + "..."
    return label


def _load_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _collect_config_runs(
    runs_dir: Path,
    tag_prefix: str,
    min_points: int,
    allow_constant_step_sequence: bool = False
) -> Dict[str, Dict[str, dict]]:
    """
    Collect runs grouped by configuration (trial directory).
    Returns: method -> config_id(trial_dir) -> {params, runs:[{seed, steps, values}]}
    """
    results: Dict[str, Dict[str, dict]] = defaultdict(dict)

    for candidate_params in runs_dir.rglob("candidate_params.json"):
        trial_dir = candidate_params.parent
        params_blob = _load_json(candidate_params) or {}
        params = params_blob.get("params", params_blob)
        method, _ = parse_method_from_path(str(trial_dir))
        if method is None:
            continue
        config_id = str(trial_dir)
        entry = results[method].setdefault(config_id, {"params": params, "runs": []})

        for seed_dir in trial_dir.iterdir():
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue
            tb_dir = seed_dir / "tb"
            if not tb_dir.exists():
                continue
            event_files = list(tb_dir.glob("events.out.tfevents*"))
            if not event_files:
                continue
            series = extract_run_series_with_mode(
                event_files,
                tag_prefix,
                min_points,
                allow_constant_step_sequence=allow_constant_step_sequence,
            )
            if series is None:
                continue
            seed_match = re.match(r"^seed[_-]?(\d+)$", seed_dir.name)
            seed_val = int(seed_match.group(1)) if seed_match else 0
            steps, values, x_kind = series
            entry["runs"].append({
                "seed": seed_val,
                "steps": steps,
                "values": values,
                "x_kind": x_kind,
                "tb_dir": str(tb_dir),
            })

    return results


def _plot_configs_for_method(
    method: str,
    configs: Dict[str, dict],
    best_config_id: Optional[str],
    out_dir: Path,
    title_prefix: str,
    formats: List[str],
    task_length: int,
    num_tasks_override: Optional[int],
    ymin: Optional[float],
    ymax: Optional[float],
    best_only: bool = False,
):
    if not configs:
        return 0

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_iter = iter(colors)
    legend_handles = []
    legend_labels = []

    # Determine visible x-range
    all_steps = []
    for cfg in configs.values():
        for run in cfg.get("runs", []):
            all_steps.append(run["steps"])
    if not all_steps:
        plt.close(fig)
        return 0

    x_min = min([steps.min() for steps in all_steps])
    x_max = max([steps.max() for steps in all_steps])

    num_tasks = int(num_tasks_override) if num_tasks_override is not None else int(np.ceil(x_max / task_length))
    x_max_visible = x_max
    if num_tasks_override is not None and task_length > 0:
        expected_xmax = num_tasks * task_length
        if expected_xmax > x_max_visible:
            x_max_visible = expected_xmax

    x_kind = 'steps'
    for cfg in configs.values():
        for run in cfg.get("runs", []):
            if run.get("x_kind") == 'index':
                x_kind = 'index'
                break

    for config_id, cfg in configs.items():
        if best_only and config_id != best_config_id:
            continue
        color = next(color_iter, None) or "#333333"
        runs = cfg.get("runs", [])
        if not runs:
            continue

        is_best = (config_id == best_config_id)
        label = _format_params_label(cfg.get("params", {}))
        if is_best and not best_only:
            label = f"{label} (best)"

        for idx, run in enumerate(runs):
            line_alpha = 0.85 if is_best else 0.35
            line_width = 2.2 if is_best else 1.0
            ax.plot(run["steps"], run["values"], color=color, alpha=line_alpha, linewidth=line_width,
                    label=label if idx == 0 else "_nolegend_", zorder=3 if is_best else 2)

        if label not in legend_labels:
            legend_handles.append(Line2D([], [], color=color, linewidth=2.0))
            legend_labels.append(label)

    # Task boundaries/labels
    if task_length > 0:
        for k in range(1, num_tasks + 1):
            boundary = k * task_length
            if x_min < boundary <= x_max_visible:
                ax.axvline(boundary, linestyle='--', color='black', alpha=0.5, linewidth=1.2, zorder=1)

        y_min, y_max = ax.get_ylim()
        label_y = y_min + 0.95 * (y_max - y_min)
        for k in range(num_tasks):
            center = (k + 0.5) * task_length
            if x_min < center < x_max_visible:
                ax.text(center, label_y, f'Task {k+1}', ha='center', va='top', fontsize=11,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.7))

    from matplotlib.ticker import FuncFormatter, MultipleLocator
    if x_kind == 'steps':
        ax.xaxis.set_major_locator(MultipleLocator(100000))
        ax.xaxis.set_major_formatter(FuncFormatter(format_step_tick))

    ax.set_xlim(x_min, x_max_visible)
    ax.set_xlabel("Eval index" if x_kind == 'index' else "Environment steps", fontsize=12)
    ax.set_ylabel("IQM return (avg across tasks)", fontsize=12)
    ax.set_title(f"{method} – {title_prefix}", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if legend_handles:
        ax.legend(legend_handles, legend_labels, fontsize=9, framealpha=0.9)

    if ymin is not None or ymax is not None:
        current_ymin, current_ymax = ax.get_ylim()
        ax.set_ylim(ymin if ymin is not None else current_ymin,
                    ymax if ymax is not None else current_ymax)

    out_dir.mkdir(parents=True, exist_ok=True)
    safe_method = method.lower().replace(" ", "_")
    suffix = "best" if best_only else "all_configs"
    base_path = out_dir / f"{safe_method}_{suffix}"
    saved = 0
    for fmt in formats:
        fig.savefig(base_path.with_suffix(f".{fmt}"), format=fmt, dpi=300, bbox_inches="tight")
        saved += 1
    plt.close(fig)
    return saved


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
    num_tasks_override: Optional[int] = None,
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
    num_tasks = int(num_tasks_override) if num_tasks_override is not None else int(np.ceil(x_max / task_length))

    # If the user forces a known task count, extend the visible x-range *before*
    # drawing task-region labels. Otherwise Task N won't be labeled if the data ends early.
    x_max_visible = x_max
    if num_tasks_override is not None and task_length > 0:
        expected_xmax = num_tasks * task_length
        if expected_xmax > x_max_visible:
            x_max_visible = expected_xmax

    ax.set_xlim(x_min, x_max_visible)

    # Draw vertical lines at task boundaries (include end boundary if it lands on x_max)
    for k in range(1, num_tasks + 1):
        boundary = k * task_length
        if x_min < boundary <= x_max_visible:
            ax.axvline(boundary, linestyle='--', color='black', alpha=0.6, linewidth=1.5, zorder=4)
    
    # Add task labels
    # Position at 95% of y-range
    label_y = y_min + 0.95 * (y_max - y_min)
    
    for k in range(num_tasks):
        # Center of task k (0-indexed)
        task_center = (k + 0.5) * task_length
        
        # Only label if center is within visible range
        if x_min < task_center < x_max_visible:
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
    
    # x-limits already set above (and extended if num_tasks_override was provided)
    
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


def plot_grouped_comparisons(
    methods: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]],
    out_dir: Path,
    formats: List[str] = ['png'],
    task_length: int = 500_000,
    num_tasks: int = 3,
    grid_step: int = 50000,
) -> int:
    """
    Create three grouped comparison plots showing mean IQM return with individual seed traces.
    
    For each method, uses up to 5 seeds (selected by sorting seed numbers), plots individual
    seed curves as thin semi-transparent lines, and overlays the mean as a thick opaque line.
    All plots use the same y-axis limits for direct comparability.
    
    Groups:
    1. Sparse methods: GMP, SET
    2. Reset-based methods: ReDo, Partial Reinit
    3. Baselines: Dense PPO, Reset
    
    Args:
        methods: Dict mapping method name -> seed -> (steps, values)
        out_dir: Output directory for plots
        formats: List of output formats (e.g., ['png', 'pdf'])
        task_length: Length of each task in environment steps (default 500,000)
        num_tasks: Number of tasks (default 3)
        grid_step: Spacing for common x-grid
        
    Returns:
        Number of plots created
    """
    # Define the three groups
    groups = [
        {
            'name': 'sparse_methods',
            'title': 'Sparse Methods – IQM Return',
            'methods': ['GMP', 'SET'],
            'colors': {'GMP': '#ff7f0e', 'SET': '#8c564b'}  # orange, brown
        },
        {
            'name': 'reset_based_methods',
            'title': 'Reset-Based Methods – IQM Return',
            'methods': ['ReDo', 'Partial Reinit'],
            'colors': {'ReDo': '#d62728', 'Partial Reinit': '#2ca02c'}  # red, green
        },
        {
            'name': 'baselines',
            'title': 'Baselines – IQM Return',
            'methods': ['Dense PPO', 'Reset'],
            'colors': {'Dense PPO': '#1f77b4', 'Reset': '#9467bd'}  # blue, purple
        }
    ]
    
    # STEP 1: Compute mean curves for all methods that will be plotted
    method_means = {}
    global_y_min = float('inf')
    global_y_max = float('-inf')
    
    for group in groups:
        for method_name in group['methods']:
            if method_name not in methods:
                print(f"  ⚠ Method '{method_name}' not found in data, skipping")
                continue
            
            seed_data = methods[method_name]
            if not seed_data:
                print(f"  ⚠ Method '{method_name}' has no seed data, skipping")
                continue
            
            # Select up to 5 seeds (first 5 when sorted by seed number)
            selected_seeds = sorted(seed_data.keys())[:5]
            selected_seed_data = {seed: seed_data[seed] for seed in selected_seeds}
            
            # Align and interpolate to common grid
            grid, aligned = align_and_interpolate(selected_seed_data, grid_step)
            
            if len(grid) == 0:
                print(f"  ⚠ Method '{method_name}' has no overlapping data, skipping")
                continue
            
            # Compute mean across selected seeds
            mean = aligned.mean(axis=0)
            
            # Store for later (including individual seeds for plotting)
            method_means[method_name] = {
                'grid': grid,
                'mean': mean,
                'aligned': aligned,  # Individual seed curves for plotting
                'num_seeds': len(selected_seed_data),
                'total_seeds': len(seed_data)  # Track total available
            }
            
            # Update global y-limits
            global_y_min = min(global_y_min, float(np.nanmin(mean)))
            global_y_max = max(global_y_max, float(np.nanmax(mean)))
    
    if not method_means:
        print("  ⚠ No valid methods found for grouped comparison plots")
        return 0
    
    # Add some padding to y-limits (5% on each side)
    y_range = global_y_max - global_y_min
    global_y_min -= 0.05 * y_range
    global_y_max += 0.05 * y_range
    
    print(f"\nGlobal y-axis limits: [{global_y_min:.2f}, {global_y_max:.2f}]")
    
    # STEP 2: Create one plot per group with shared y-limits
    plots_created = 0
    
    for group in groups:
        print(f"\nCreating grouped plot: {group['name']}")
        
        # Filter to methods that have data
        available_methods = [m for m in group['methods'] if m in method_means]
        
        if not available_methods:
            print(f"  ⚠ No valid methods for group '{group['name']}', skipping")
            continue
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot each method in this group
        for method_name in available_methods:
            data = method_means[method_name]
            grid = data['grid']
            mean = data['mean']
            aligned = data['aligned']  # Individual seed curves
            color = group['colors'].get(method_name, '#333333')
            
            # Convert to thousands for x-axis
            grid_k = grid / 1000.0
            
            # First, plot individual seed curves as thin lines with low alpha
            for seed_idx in range(aligned.shape[0]):
                seed_curve = aligned[seed_idx]
                ax.plot(grid_k, seed_curve, color=color, linewidth=0.8, alpha=0.3, 
                       zorder=2, label=None)
            
            # Then plot mean line on top (thick, full opacity)
            num_total = data.get('total_seeds', data['num_seeds'])
            ax.plot(grid_k, mean, label=f"{method_name} (n={data['num_seeds']}/{num_total})", 
                   color=color, linewidth=3, zorder=3)
            
            print(f"  ✓ Added {method_name} (using {data['num_seeds']}/{num_total} seeds)")
        
        # CRITICAL: Apply global y-limits
        ax.set_ylim(global_y_min, global_y_max)
        
        # Task boundaries: vertical dashed lines at 500k and 1000k
        task_boundaries_k = [task_length * k / 1000.0 for k in range(1, num_tasks)]
        for boundary_k in task_boundaries_k:
            ax.axvline(boundary_k, linestyle='--', color='black', alpha=0.6, 
                      linewidth=1.5, zorder=1)
        
        # Task labels: "Task 1", "Task 2", "Task 3"
        # Position at 95% of y-range
        label_y = global_y_min + 0.95 * (global_y_max - global_y_min)
        
        for k in range(num_tasks):
            # Center of task k (0-indexed)
            task_center_k = (k + 0.5) * task_length / 1000.0
            ax.text(task_center_k, label_y, f'Task {k+1}', 
                   horizontalalignment='center', verticalalignment='top',
                   fontsize=12, fontweight='normal', alpha=0.8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='none', alpha=0.7),
                   zorder=4)
        
        # X-axis formatting: show in thousands (0, 250k, 500k, 750k, etc.)
        from matplotlib.ticker import FuncFormatter
        
        def format_thousands(x, pos):
            """Format x-axis labels as 0, 250k, 500k, etc."""
            if x == 0:
                return '0'
            elif x >= 1000:
                return f'{int(x)}k'
            else:
                return f'{int(x)}k'
        
        ax.xaxis.set_major_formatter(FuncFormatter(format_thousands))
        
        # Set x-limits to show full task range
        ax.set_xlim(0, num_tasks * task_length / 1000.0)
        
        # Labels and title
        ax.set_xlabel('Environment Steps (thousands)', fontsize=12)
        ax.set_ylabel('IQM Return (avg across tasks)', fontsize=12)
        ax.set_title(group['title'], fontsize=14, fontweight='bold')
        
        # Grid and legend
        ax.grid(True, alpha=0.3, zorder=0)
        ax.legend(fontsize=11, framealpha=0.9, loc='best')
        
        # Tight layout
        plt.tight_layout()
        
        # Save in all requested formats
        out_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = []
        
        for fmt in formats:
            out_path = out_dir / f"{group['name']}.{fmt}"
            fig.savefig(out_path, format=fmt, dpi=300, bbox_inches='tight')
            saved_paths.append(str(out_path))
        
        plt.close(fig)
        
        print(f"  ✓ Saved to {', '.join(saved_paths)}")
        plots_created += 1
    
    return plots_created


def main():
    parser = argparse.ArgumentParser(
        description="Plot continual IQM return curves with 95% CI from TensorBoard logs"
    )
    parser.add_argument('--runs_dir', type=str, default='runs',
                        help='Directory containing run folders')
    parser.add_argument('--out_dir', type=str, default='results/plots',
                        help='Output directory for legacy averaged plots')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Base results directory (plots and best configs)')
    parser.add_argument('--tag_prefix', type=str, default='train_reward_iqm/',
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
    parser.add_argument('--min_points_eval', type=int, default=1,
                        help='Minimum number of logged points required for eval plots')
    parser.add_argument('--formats', type=str, default='png',
                        help='Output file format(s), comma-separated (e.g., "png,pdf" or "both" for both)')
    parser.add_argument('--rng_seed', type=int, default=0,
                        help='Random seed for bootstrap')
    parser.add_argument('--task_length', type=int, default=500000,
                        help='Length of each task in environment steps (for task boundary lines)')
    parser.add_argument('--num_tasks', type=int, default=None,
                        help='Optional override for number of tasks (forces boundary/label placement to num_tasks * task_length)')
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
    parser.add_argument('--ablation_config', type=str, default=None,
                        help='JSON spec for a multi-panel ablation figure (optional)')
    parser.add_argument('--ablation_plot_name', type=str, default='iqm_ablation',
                        help='Base filename for the combined ablation figure')
    parser.add_argument('--summary_config', type=str, default=None,
                        help='JSON spec for the single-trace train IQM summary figure')
    parser.add_argument('--summary_plot_name', type=str, default='train_iqm_summary',
                        help='Base filename for the summary figure across all interventions')
    parser.add_argument('--normalize', choices=['none', 'max'], default='none',
                        help='Normalization mode for ablation lines (max scales each curve to [0,1])')
    parser.add_argument('--steps_per_epoch', type=int, default=1,
                        help='Divide x-axis steps by this value when drawing the ablation figure (use >1 to show epochs)')
    parser.add_argument('--legacy_average', action='store_true', default=False,
                        help='Use legacy averaged plotting (method-level CI)')
    parser.add_argument('--best_lambda', type=float, default=0.5,
                        help='Lambda for composite best-score: score - lambda * forgetting')
    parser.add_argument('--grouped_comparisons', action='store_true', default=False,
                        help='Create grouped comparison plots (Sparse, Reset-based, Baselines) with shared y-axis')
    
    args = parser.parse_args()
    
    # Parse formats
    if args.formats.lower() == 'both':
        formats = ['png', 'pdf']
    else:
        formats = [fmt.strip() for fmt in args.formats.split(',')]
    
    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    results_dir = Path(args.results_dir)
    
    if not runs_dir.exists():
        print(f"Error: runs_dir '{runs_dir}' does not exist")
        return 1

    # Legacy path for averaged plots / ablations
    if args.legacy_average or args.ablation_config or args.summary_config:
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
        
        # STEP 1: First pass - compute all means to determine global y-limits
        method_data = {}
        global_y_min = float('inf')
        global_y_max = float('-inf')
        
        for method, seed_data in methods.items():
            grid, aligned = align_and_interpolate(seed_data, args.grid_step)
            if len(grid) == 0:
                continue
                
            mean, lower_ci, upper_ci = bootstrap_ci(
                aligned,
                n_bootstrap=args.bootstrap,
                rng_seed=args.rng_seed
            )
            
            method_data[method] = {
                'grid': grid,
                'aligned': aligned,
                'mean': mean,
                'lower_ci': lower_ci,
                'upper_ci': upper_ci,
                'seed_data': seed_data
            }
            
            # Update global y-limits (use CI bounds for full range)
            global_y_min = min(global_y_min, float(np.nanmin(lower_ci)))
            global_y_max = max(global_y_max, float(np.nanmax(upper_ci)))
        
        if not method_data:
            print("No valid methods with overlapping data")
            return 1
        
        # Add padding (5%) if not manually specified
        if args.ymin is None or args.ymax is None:
            y_range = global_y_max - global_y_min
            computed_ymin = global_y_min - 0.05 * y_range
            computed_ymax = global_y_max + 0.05 * y_range
            final_ymin = args.ymin if args.ymin is not None else computed_ymin
            final_ymax = args.ymax if args.ymax is not None else computed_ymax
            print(f"\nGlobal y-axis limits for individual plots: [{final_ymin:.2f}, {final_ymax:.2f}]")
        else:
            final_ymin = args.ymin
            final_ymax = args.ymax
            print(f"\nUsing manual y-axis limits: [{final_ymin:.2f}, {final_ymax:.2f}]")
        
        # STEP 2: Second pass - create plots with shared y-limits
        for method, data in method_data.items():
            print(f"\nProcessing {method}...")
            
            print(f"  • {len(data['seed_data'])} seeds used")
            print(f"  • {len(data['grid'])} grid points")
            print(f"  • Step range: [{data['grid'].min():,}, {data['grid'].max():,}]")

            method_dir = out_dir / 'individual_interventions' / method
            out_path = method_dir / f"iqm_return_ci.{formats[0]}"

            plot_method(
                method_name=method,
                seed_data=data['seed_data'],
                grid=data['grid'],
                aligned_values=data['aligned'],
                mean=data['mean'],
                lower_ci=data['lower_ci'],
                upper_ci=data['upper_ci'],
                out_path=out_path,
                formats=formats,
                task_length=args.task_length,
                num_tasks_override=args.num_tasks,
                ymin=final_ymin,
                ymax=final_ymax,
                save_data=(not args.no_save_data) and args.save_data,
                save_aligned=args.save_aligned,
                rng_seed=args.rng_seed
            )

            plots_created += 1
    
        if args.ablation_config:
            config_path = Path(args.ablation_config)
            config = _load_json_config(config_path)
            if config is None:
                print(f"Error: cannot read ablation config '{config_path}'")
            else:
                saved = _plot_ablation_grid(methods, config, args, out_dir, formats)
                if saved:
                    plots_created += 1

        if args.summary_config:
            summary_path = Path(args.summary_config)
            summary_config = _load_json_config(summary_path)
            if summary_config is None:
                print(f"Error: cannot read summary config '{summary_path}'")
            else:
                saved = _plot_train_summary(methods, summary_config, args, out_dir, formats)
                if saved:
                    plots_created += saved
        
        # Grouped comparison plots (if requested)
        if args.grouped_comparisons:
            print("\nCreating grouped comparison plots...")
            saved = plot_grouped_comparisons(
                methods,
                out_dir / 'grouped_comparisons',
                formats=formats,
                task_length=args.task_length,
                num_tasks=args.num_tasks if args.num_tasks is not None else 3,
                grid_step=args.grid_step
            )
            if saved:
                plots_created += saved

        if plots_created == 0:
            print("No plots created! Check your data.")
            return 1

        print(f"✓ Done! Created {plots_created} plot(s) in {out_dir}/")
        return 0

    # Plot per-configuration curves (no averaging), train only
    print(f"Selecting best configs from {runs_dir}...")
    best_configs = select_best_configs(
        runs_dir,
        lambda_forgetting=args.best_lambda,
        prefer_eval=False,
    )
    write_best_configs(best_configs, results_dir)
    print(f"Saved best configs to {results_dir / 'best_configs'}")

    tag_prefix = 'train_reward_iqm/'
    print(f"\nCollecting per-config runs for train (tag_prefix='{tag_prefix}')...")
    config_runs = _collect_config_runs(
        runs_dir,
        tag_prefix,
        args.min_points,
        allow_constant_step_sequence=False,
    )
    if not config_runs:
        print(f"No valid runs found with tag prefix '{tag_prefix}'")
        return 1

    plots_out_dir = results_dir / 'plots'
    plots_created = 0
    for method, configs in config_runs.items():
        best_trial_dir = None
        if method in best_configs:
            best_trial_dir = best_configs[method].get('trial_dir')

        plots_created += _plot_configs_for_method(
            method=method,
            configs=configs,
            best_config_id=best_trial_dir,
            out_dir=plots_out_dir,
            title_prefix="Train IQM (all configs)",
            formats=formats,
            task_length=args.task_length,
            num_tasks_override=args.num_tasks,
            ymin=args.ymin,
            ymax=args.ymax,
            best_only=False,
        )
        if best_trial_dir is not None:
            plots_created += _plot_configs_for_method(
                method=method,
                configs=configs,
                best_config_id=best_trial_dir,
                out_dir=plots_out_dir,
                title_prefix="Train IQM (best config)",
                formats=formats,
                task_length=args.task_length,
                num_tasks_override=args.num_tasks,
                ymin=args.ymin,
                ymax=args.ymax,
                best_only=True,
            )

    if plots_created == 0:
        print("No plots created! Check your data.")
        return 1

    print(f"✓ Done! Created {plots_created} plot(s) in {results_dir / 'plots'}/")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
