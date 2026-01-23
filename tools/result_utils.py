# tools/result_utils.py
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# ---------------------------
# Event file discovery
# ---------------------------

EVENT_FILE_RE = re.compile(r"^events\.out\.tfevents\..+")


def find_event_files(root: str) -> List[str]:
    """
    Recursively find all TensorBoard event files under `root`.
    """
    out: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if EVENT_FILE_RE.match(fn):
                out.append(os.path.join(dirpath, fn))
    out.sort()
    return out


# ---------------------------
# Seed + group detection
# ---------------------------

SEED_DIR_RE = re.compile(r"seed[_-]?(\d+)", re.IGNORECASE)  # Match seed_N even with timestamp after


@dataclass(frozen=True)
class RunIdentity:
    group_key: str      # e.g. "procgen_3_tasks_1_cycle_500k/dense/ppo_dense_20260119_204309"
    seed: str           # e.g. "0"
    event_file: str     # full path to TB event file


def _split_relpath(relpath: str) -> List[str]:
    parts = []
    for p in relpath.replace("\\", "/").split("/"):
        if p.strip():
            parts.append(p)
    return parts


def infer_group_and_seed(runs_dir: str, event_file: str) -> RunIdentity:
    """
    Infer:
      - group_key: intervention/method name (e.g., "dense", "gmp")
      - seed: extracted from 'seed_<k>' directory

    Supports layouts like:
      runs/<intervention>/seed_0_timestamp/.../events...  → group=intervention, seed=0
      runs/<experiment>/<method>/seed_0/.../events...     → group=experiment/method, seed=0

    Strategy:
    1. Find any folder matching "seed_N" pattern (even with timestamp after)
    2. Extract seed number N
    3. Group key = everything before that seed folder
    """
    abs_runs = os.path.abspath(runs_dir)
    abs_event = os.path.abspath(event_file)

    if not abs_event.startswith(abs_runs):
        return RunIdentity(group_key="unknown", seed="0", event_file=event_file)

    rel = os.path.relpath(abs_event, abs_runs)
    parts = _split_relpath(rel)

    # Look for seed_<k> directory (may have timestamp like seed_0_20260120_112732)
    seed_idx = None
    seed_val = None
    for i, p in enumerate(parts):
        m = SEED_DIR_RE.search(p)  # Use search instead of match to find seed_N anywhere in string
        if m:
            seed_idx = i
            seed_val = m.group(1)
            break

    if seed_idx is not None and seed_idx > 0:
        # Has seed folder: group everything before seed folder
        group_parts = parts[:seed_idx]
        group_key = "/".join(group_parts) if group_parts else "root"
        seed = seed_val if seed_val is not None else "0"
        return RunIdentity(group_key=group_key, seed=seed, event_file=event_file)

    # Fallback: assume structure intervention/run_folder/...
    if len(parts) >= 2:
        group_key = parts[0]  # First folder = intervention name
        seed = "0"
        return RunIdentity(group_key=group_key, seed=seed, event_file=event_file)
    
    # Last resort fallback
    parent = os.path.dirname(rel).replace("\\", "/")
    group_key = parent if parent else "root"
    return RunIdentity(group_key=group_key, seed="0", event_file=event_file)


# ---------------------------
# TensorBoard scalar reading (ROBUST)
# ---------------------------

def load_scalars(event_file: str, size_guidance: int = 200000) -> Dict[str, List[Tuple[int, float]]]:
    """
    Load scalar time-series from a TensorBoard event file safely.

    Returns:
      dict[tag] = [(step, value), ...]

    Raises:
      ValueError if the event file is missing/corrupt/invalid.
    """
    # quick reject: empty / tiny files are almost always broken
    try:
        if not os.path.exists(event_file):
            raise ValueError(f"Missing event file: {event_file}")
        if os.path.getsize(event_file) < 200:  # bytes
            raise ValueError(f"Event file too small / likely invalid: {event_file}")
    except OSError:
        raise ValueError(f"Cannot stat event file: {event_file}")

    try:
        ea = EventAccumulator(
            event_file,
            size_guidance={
                "scalars": size_guidance,
                "images": 0,
                "histograms": 0,
                "tensors": 0,
                "audio": 0,
            },
        )
        ea.Reload()
    except Exception as e:
        raise ValueError(
            f"Invalid/corrupt TB event file: {event_file}\n{type(e).__name__}: {e}"
        )

    out: Dict[str, List[Tuple[int, float]]] = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        out[tag] = [(int(ev.step), float(ev.value)) for ev in events]
        out[tag].sort(key=lambda x: x[0])
    return out


# ---------------------------
# Curve utilities
# ---------------------------

def get_curve(scalars: Dict[str, List[Tuple[int, float]]], tag: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if tag not in scalars or len(scalars[tag]) == 0:
        return None
    steps = np.array([s for s, _ in scalars[tag]], dtype=np.int64)
    vals = np.array([v for _, v in scalars[tag]], dtype=np.float64)
    return steps, vals


def forward_fill_align(steps: np.ndarray, vals: np.ndarray, target_steps: np.ndarray) -> np.ndarray:
    """
    Forward-fill values from (steps, vals) onto target_steps.
    """
    out = np.empty_like(target_steps, dtype=np.float64)
    j = 0
    last = np.nan
    for i, ts in enumerate(target_steps):
        while j < len(steps) and steps[j] <= ts:
            last = vals[j]
            j += 1
        out[i] = last
    return out


def extract_task_avg_eval_iqm_curve(scalars: Dict[str, List[Tuple[int, float]]]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Extract a single task-averaged eval IQM curve from tags:
      eval_reward_iqm/<task_id>

    Align tasks on union of steps using forward-fill, then average across tasks.
    """
    task_tags = [t for t in scalars.keys() if t.startswith("eval_reward_iqm/")]
    if not task_tags:
        return None

    curves = []
    all_steps = set()
    for t in task_tags:
        c = get_curve(scalars, t)
        if c is None:
            continue
        s, v = c
        curves.append((s, v))
        all_steps.update(s.tolist())

    if not curves:
        return None

    target_steps = np.array(sorted(all_steps), dtype=np.int64)
    aligned = []
    for s, v in curves:
        av = forward_fill_align(s, v, target_steps)
        aligned.append(av)

    mat = np.vstack(aligned)  # [n_tasks, n_steps]
    mean = np.nanmean(mat, axis=0)
    return target_steps, mean


def extract_task_avg_dormant_frac_curve(scalars: Dict[str, List[Tuple[int, float]]]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Extract a single task-averaged dormant fraction curve from tags:
      plasticity/dormant_frac/<task_id>

    Align tasks on union of steps using forward-fill, then average across tasks.
    """
    task_tags = [t for t in scalars.keys() if t.startswith("plasticity/dormant_frac/")]
    if not task_tags:
        return None

    curves = []
    all_steps = set()
    for t in task_tags:
        c = get_curve(scalars, t)
        if c is None:
            continue
        s, v = c
        curves.append((s, v))
        all_steps.update(s.tolist())

    if not curves:
        return None

    target_steps = np.array(sorted(all_steps), dtype=np.int64)
    aligned = []
    for s, v in curves:
        av = forward_fill_align(s, v, target_steps)
        aligned.append(av)

    mat = np.vstack(aligned)  # [n_tasks, n_steps]
    mean = np.nanmean(mat, axis=0)
    return target_steps, mean


# ---------------------------
# IMPROVED: Generic task-averaged curve extraction
# ---------------------------

def find_tags_matching_prefix(scalars: Dict[str, List[Tuple[int, float]]], prefix: str) -> List[str]:
    """
    Find all tags in scalars that match: prefix/0, prefix/1, prefix/2, etc.
    
    Args:
        scalars: Tag -> [(step, value)] dict
        prefix: Tag prefix like "forgetting/isolated_task_mean"
    
    Returns:
        List of matching tags sorted by task index
    """
    pattern = re.compile(rf"^{re.escape(prefix)}/(\d+)$")
    matches = []
    for tag in scalars.keys():
        m = pattern.match(tag)
        if m:
            task_id = int(m.group(1))
            matches.append((task_id, tag))
    matches.sort(key=lambda x: x[0])
    return [tag for _, tag in matches]


def extract_task_avg_curve_from_prefix(
    scalars: Dict[str, List[Tuple[int, float]]], 
    prefix: str
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Extract task-averaged curve from task-wise tags like:
      prefix/0, prefix/1, prefix/2, ...
    
    Args:
        scalars: Tag -> [(step, value)] dict
        prefix: Tag prefix (e.g., "forgetting/isolated_task_mean")
    
    Returns:
        (steps, avg_values) or None if no matching tags found
    
    Process:
        1. Find all tags matching prefix/0, prefix/1, etc.
        2. Load curves for each task
        3. Align to union of steps using forward-fill
        4. Average across tasks at each step
    """
    task_tags = find_tags_matching_prefix(scalars, prefix)
    if not task_tags:
        return None
    
    curves = []
    all_steps = set()
    for tag in task_tags:
        c = get_curve(scalars, tag)
        if c is None:
            continue
        s, v = c
        curves.append((s, v))
        all_steps.update(s.tolist())
    
    if not curves:
        return None
    
    target_steps = np.array(sorted(all_steps), dtype=np.int64)
    aligned = []
    for s, v in curves:
        av = forward_fill_align(s, v, target_steps)
        aligned.append(av)
    
    mat = np.vstack(aligned)  # [n_tasks, n_steps]
    mean = np.nanmean(mat, axis=0)
    return target_steps, mean


# ---------------------------
# Bootstrap CI
# ---------------------------

def iqm(values: np.ndarray) -> float:
    """
    Interquartile Mean (IQM): mean of the middle 50% of values.
    
    Drops lowest 25% and highest 25%, then averages remaining values.
    For small sample sizes (n < 4), may return partial range or full mean.
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    n = len(values)
    
    if n == 0:
        return float("nan")
    if n == 1:
        return float(values[0])
    
    # Sort and compute quartile indices
    sorted_vals = np.sort(values)
    q1_idx = int(np.floor(n * 0.25))
    q3_idx = int(np.ceil(n * 0.75))
    
    # Take middle 50%
    middle_vals = sorted_vals[q1_idx:q3_idx]
    
    if len(middle_vals) == 0:
        return float(np.mean(values))
    
    return float(np.mean(middle_vals))


def bootstrap_ci(
    values: np.ndarray, 
    num_bootstrap: int = 2000, 
    alpha: float = 0.05, 
    seed: int = 0,
    statistic: str = "mean"
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval across seeds.
    
    Args:
        values: Array of values from different seeds (one value per seed)
        num_bootstrap: Number of bootstrap resamples
        alpha: Significance level (0.05 = 95% CI)
        seed: Random seed for reproducibility
        statistic: "mean", "median", or "iqm"
    
    Returns:
        (central_estimate, ci_low, ci_high)
        
    IMPORTANT: This computes CI across seeds, not within seeds.
    The uncertainty reflects: "How much does the result vary if I rerun with different random seeds?"
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    n = len(values)
    
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    if n == 1:
        m = float(values[0])
        return (m, m, m)
    
    # Choose statistic function
    if statistic == "mean":
        stat_fn = np.mean
    elif statistic == "median":
        stat_fn = np.median
    elif statistic == "iqm":
        stat_fn = iqm
    else:
        raise ValueError(f"Unknown statistic: {statistic}")
    
    # Bootstrap: resample SEEDS with replacement
    rng = np.random.default_rng(seed)
    boot_stats = np.empty(num_bootstrap, dtype=np.float64)
    for b in range(num_bootstrap):
        samp = rng.choice(values, size=n, replace=True)  # Resampling unit = seed
        boot_stats[b] = stat_fn(samp)
    
    # Compute central estimate and CI
    central = float(stat_fn(values))
    lo = float(np.quantile(boot_stats, alpha / 2))
    hi = float(np.quantile(boot_stats, 1 - alpha / 2))
    
    return (central, lo, hi)


# ---------------------------
# Aggregate curves across seeds
# ---------------------------

def aggregate_curves_across_seeds(
    curves: List[Tuple[np.ndarray, np.ndarray]],
    num_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
    statistic: str = "mean",
) -> Dict[str, List[float]]:
    """
    Aggregate curves from multiple seeds with bootstrap CI.
    
    Args:
        curves: List of (steps, values) tuples, one per seed
        num_bootstrap: Number of bootstrap resamples (10000 recommended)
        alpha: Significance level (0.05 = 95% CI)
        seed: Random seed for reproducibility
        statistic: "mean", "median", or "iqm" - central estimate to use
    
    Returns:
        Dictionary with:
            "steps": [x1, x2, x3, ...]
            "central": [stat(x1), stat(x2), stat(x3), ...]  (mean/median/IQM)
            "ci_low": [lower_bound(x1), lower_bound(x2), ...]
            "ci_high": [upper_bound(x1), upper_bound(x2), ...]
    
    CRITICAL: CI reflects variation ACROSS SEEDS at each time point.
    At each x, we collect values from all seeds, then bootstrap resample those seed values.
    """
    if not curves:
        return {"steps": [], "central": [], "ci_low": [], "ci_high": []}
    
    # Step 1: Find union of all x values (steps) across seeds
    all_steps = set()
    for s, _ in curves:
        all_steps.update(s.tolist())
    target_steps = np.array(sorted(all_steps), dtype=np.int64)
    
    # Step 2: Align all seeds to common x grid using forward-fill
    aligned_vals = []
    for s, v in curves:
        aligned = forward_fill_align(s, v, target_steps)
        aligned_vals.append(aligned)
    
    mat = np.vstack(aligned_vals)  # Shape: [n_seeds, n_steps]
    n_seeds = mat.shape[0]
    n_steps = mat.shape[1]
    
    # Step 3: Choose statistic function
    if statistic == "mean":
        stat_fn = np.nanmean
    elif statistic == "median":
        stat_fn = np.nanmedian
    elif statistic == "iqm":
        stat_fn = lambda arr, axis=0: np.apply_along_axis(iqm, axis, arr)
    else:
        raise ValueError(f"Unknown statistic: {statistic}")
    
    # Step 4: Compute central estimate
    central = stat_fn(mat, axis=0)  # Average across seeds at each step
    
    # Step 5: Bootstrap CI across seeds
    if n_seeds <= 1:
        # Only 1 seed: no variation to estimate
        ci_low = central.copy()
        ci_high = central.copy()
    else:
        # Bootstrap: resample SEEDS (rows), not timesteps
        rng = np.random.default_rng(seed)
        boot_stats = np.empty((num_bootstrap, n_steps), dtype=np.float64)
        
        for b in range(num_bootstrap):
            # Resample seed indices with replacement
            seed_indices = rng.choice(np.arange(n_seeds), size=n_seeds, replace=True)
            boot_sample = mat[seed_indices, :]  # Shape: [n_seeds, n_steps]
            boot_stats[b, :] = stat_fn(boot_sample, axis=0)
        
        # Percentile-based CI at each timestep
        ci_low = np.quantile(boot_stats, alpha / 2, axis=0)
        ci_high = np.quantile(boot_stats, 1 - alpha / 2, axis=0)
    
    return {
        "steps": target_steps.astype(int).tolist(),
        "central": central.astype(float).tolist(),
        "ci_low": ci_low.astype(float).tolist(),
        "ci_high": ci_high.astype(float).tolist(),
    }