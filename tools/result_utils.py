from __future__ import annotations

"""Utility helpers to read TensorBoard event scalars and aggregate metric curves.

This module provides lightweight, dependency-tolerant functions used by the
analysis scripts in /tools. It implements:
- find_event_files: discover event files under a runs directory
- infer_group_and_seed: derive a method/group key and a seed identifier
- load_scalars: read scalar series from a single event file (tensorflow or tensorboard backends)
- merge_scalars_dicts: merge multiple per-file scalar dicts for a seed
- convenience extractors for eval_iqm, dormant fraction and forgetting curves
- bootstrap_ci: per-array bootstrap confidence intervals for mean/median/iqm
- aggregate_curves_across_seeds: align per-seed curves (forward-fill) and compute central + CI

The implementations are intentionally conservative: they avoid hard requirements
on heavy packages (TensorFlow) where possible and raise informative errors when
reading fails.
"""

import os
import re
import json
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd


def find_event_files(runs_dir: str) -> List[str]:
    """Walk runs_dir and return a list of full paths to TensorBoard event files."""
    out = []
    for root, _dirs, files in os.walk(runs_dir):
        for f in files:
            if f.startswith("events.out.tfevents"):
                out.append(os.path.join(root, f))
    return sorted(out)


class RunIdentity:
    def __init__(self, group_key: str, seed: str, event_file: str):
        self.group_key = group_key
        self.seed = seed
        self.event_file = event_file


def infer_group_and_seed(runs_dir: str, event_file: str) -> RunIdentity:
    """Infer a group key (method/intervention) and a seed id from an event file path.

    group_key: relative path from runs_dir to the containing folder (may include method/exp name)
    seed: parsed seed string when possible, otherwise the final folder name
    """
    runs_dir = os.path.abspath(runs_dir)
    event_file = os.path.abspath(event_file)
    parent = os.path.dirname(event_file)
    try:
        rel = os.path.relpath(parent, runs_dir)
    except Exception:
        rel = parent

    # Try to extract seed like 'seed_0' or 'seed-0' or '0' from path components
    parts = rel.replace("\\", "/").split("/")
    seed = None
    for p in reversed(parts):
        m = re.match(r'(?:seed[_-]?)(\d+)$', p, flags=re.IGNORECASE)
        if m:
            seed = m.group(1)
            break
        m2 = re.match(r'^(\d{1,4})$', p)
        if m2:
            seed = m2.group(1)
            break

    if seed is None:
        # fallback to last path component as seed id
        seed = parts[-1] if parts else os.path.basename(parent)

    return RunIdentity(group_key=rel, seed=str(seed), event_file=event_file)


def load_scalars(event_file: str) -> Dict[str, List[Tuple[int, float]]]:
    """Load scalar series from a TensorBoard event file.

    Returns a dict: tag -> [(step, value), ...]
    """
    # Try summary_iterator from tensorflow first (fast, small dependency if TF installed)
    try:
        from tensorflow.python.summary.summary_iterator import summary_iterator

        out = {}
        for ev in summary_iterator(event_file):
            step = int(getattr(ev, "step", 0))
            for v in getattr(ev, "summary", []).value:
                tag = getattr(v, "tag", None)
                val = getattr(v, "simple_value", None)
                if tag is None or val is None:
                    continue
                out.setdefault(tag, []).append((step, float(val)))
        # sort each series by step
        for k in list(out.keys()):
            out[k] = sorted(out[k], key=lambda x: x[0])
        return out
    except Exception:
        pass

    # Fallback: try tensorboard's EventAccumulator by pointing it at the folder
    try:
        from tensorboard.backend.event_processing import event_accumulator

        parent = os.path.dirname(event_file)
        ea = event_accumulator.EventAccumulator(parent)
        ea.Reload()
        out = {}
        tags = ea.Tags().get("scalars", [])
        for tag in tags:
            vals = ea.Scalars(tag)
            out[tag] = [(int(v.step), float(v.value)) for v in vals]
        return out
    except Exception as e:
        raise ValueError(f"Failed to read event file {event_file}: {e}")


def merge_scalars_dicts(dicts: List[Dict[str, List[Tuple[int, float]]]]) -> Dict[str, List[Tuple[int, float]]]:
    """Merge multiple scalar dictionaries produced by `load_scalars`.

    Concatenates series for the same tag and sorts by step. Does not deduplicate identical steps.
    """
    from collections import defaultdict

    merged = defaultdict(list)
    for d in dicts:
        for k, v in d.items():
            merged[k].extend(v)
    out = {}
    for k, v in merged.items():
        out[k] = sorted(v, key=lambda x: x[0])
    return out


def get_curve(scalars: Dict[str, List[Tuple[int, float]]], tag: str) -> Optional[Tuple[List[int], List[float]]]:
    """Return (steps, values) for a tag if present, otherwise None."""
    if tag in scalars:
        series = scalars[tag]
        steps = [int(s) for s, _ in series]
        vals = [float(v) for _, v in series]
        return steps, vals
    return None


def _task_avg_from_matching_tags(scalars: Dict[str, List[Tuple[int, float]]], include_substrs: List[str]) -> Optional[Tuple[List[int], List[float]]]:
    """Generic helper: find tags containing any of include_substrs, average across tags per step.

    This targets the common logging pattern where each task writes a scalar under a tag
    and the per-run task-average is computed by averaging across those tags at each step.
    """
    candidate_tags = [t for t in scalars.keys() if any(s in t.lower() for s in include_substrs)]
    if not candidate_tags:
        return None

    # Build step -> list of values
    from collections import defaultdict

    step_vals = defaultdict(list)
    for tag in candidate_tags:
        for step, val in scalars[tag]:
            step_vals[int(step)].append(float(val))

    if not step_vals:
        return None

    steps = sorted(step_vals.keys())
    vals = [float(np.mean(step_vals[s])) for s in steps]
    return steps, vals


def extract_task_avg_eval_iqm_curve(scalars: Dict[str, List[Tuple[int, float]]]) -> Optional[Tuple[List[int], List[float]]]:
    # match tags like 'eval', 'iqm', 'eval_reward'
    return _task_avg_from_matching_tags(scalars, ['eval', 'iqm', 'eval_reward'])


def extract_task_avg_dormant_frac_curve(scalars: Dict[str, List[Tuple[int, float]]]) -> Optional[Tuple[List[int], List[float]]]:
    return _task_avg_from_matching_tags(scalars, ['dormant', 'dormancy', 'dormant_frac', 'dormant/fr'])


def extract_task_avg_isolated_forgetting_curve(scalars: Dict[str, List[Tuple[int, float]]]) -> Optional[Tuple[List[int], List[float]]]:
    # Try exact aggregate tags first (these are already averaged across tasks)
    for candidate in ['forgetting/isolated_avg_iqm', 'forgetting/isolated_avg_mean', 'forgetting/isolated_avg']:
        result = get_curve(scalars, candidate)
        if result is not None:
            return result
    
    # Try other variations
    for candidate in ['isolated_forgetting', 'forgetting_isolated']:
        result = get_curve(scalars, candidate)
        if result is not None:
            return result
    
    # Fall back to averaging per-task forgetting tags
    return _task_avg_from_matching_tags(scalars, ['forgetting/isolated_task', 'forget'])


def load_effective_rank_curve_from_json(event_file: str) -> Optional[Tuple[List[int], List[float]]]:
    """Try to find a JSON file next to the event file that contains effective-rank history.

    Looks for files named like 'effective_rank*.json' or any JSON containing the key 'effective_rank'.
    """
    parent = os.path.dirname(event_file)
    try:
        for fname in sorted(os.listdir(parent)):
            if not fname.lower().endswith('.json'):
                continue
            p = os.path.join(parent, fname)
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception:
                continue

            if isinstance(data, dict):
                # common keys
                for key in ['effective_rank', 'effective_rank_history', 'eff_rank']:
                    if key in data:
                        arr = data[key]
                        if isinstance(arr, list) and arr:
                            steps = list(range(len(arr)))
                            return steps, [float(x) for x in arr]
    except Exception:
        pass

    return None


def iqm(a: np.ndarray) -> float:
    """Interquartile mean (trim 25% both tails) for 1D array."""
    a = np.asarray(a)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return float('nan')
    a = np.sort(a)
    n = a.size
    lo = int(np.floor(0.25 * n))
    hi = int(np.ceil(0.75 * n))
    if hi <= lo:
        return float(np.mean(a))
    return float(np.mean(a[lo:hi]))


def bootstrap_ci(values: np.ndarray, n_bootstrap: int, alpha: float, seed: int = 0, statistic: str = 'median') -> Tuple[float, float, float]:
    """Compute point estimate and bootstrap percentile CI for a 1D array of values.

    statistic: 'median' | 'mean' | 'iqm'
    Returns: (point_estimate, ci_low, ci_high)
    """
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return float('nan'), float('nan'), float('nan')

    if statistic == 'median':
        stat_fn = lambda x: float(np.median(x))
    elif statistic == 'mean':
        stat_fn = lambda x: float(np.mean(x))
    elif statistic == 'iqm':
        stat_fn = lambda x: iqm(np.asarray(x))
    else:
        raise ValueError(f"Unknown statistic: {statistic}")

    point = stat_fn(vals)

    if n_bootstrap <= 0:
        return point, float('nan'), float('nan')

    rng = np.random.default_rng(seed)
    boots = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        sample = rng.choice(vals, size=vals.size, replace=True)
        boots[i] = stat_fn(sample)

    lo = float(np.quantile(boots, alpha / 2.0))
    hi = float(np.quantile(boots, 1.0 - alpha / 2.0))
    return point, lo, hi


def aggregate_curves_across_seeds(
    curves: List[Tuple[List[int], List[float]]],
    n_bootstrap: int,
    alpha: float,
    seed: int = 0,
    statistic: str = 'median',
) -> Dict[str, List[float]]:
    """Align multiple (steps, values) curves across seeds and compute central + CI per timepoint.

    Alignment strategy: union-of-steps across seeds. For each seed, create a pandas Series
    indexed by its steps, reindex to the union steps and forward-fill missing values.
    This mirrors forward-fill alignment used in other scripts.
    """
    if not curves:
        return {"steps": [], "central": [], "ci_low": [], "ci_high": []}

    # Build union of steps
    all_steps = sorted({int(s) for c in curves for s, _ in [c] for s in c[0]})

    # If any curve has zero steps, skip it
    filtered = []
    for steps, vals in curves:
        if steps is None or vals is None or len(steps) == 0:
            continue
        s = np.asarray(steps, dtype=np.int64)
        v = np.asarray(vals, dtype=np.float64)
        filtered.append((s, v))

    if not filtered:
        return {"steps": [], "central": [], "ci_low": [], "ci_high": []}

    # Reindex each curve to union steps with forward-fill
    df_list = []
    for s, v in filtered:
        ser = pd.Series(v, index=s)
        ser = ser.sort_index()
        ser = ser.reindex(all_steps)
        ser = ser.ffill()
        df_list.append(ser.values)

    arr = np.vstack(df_list)  # shape: n_seeds x n_steps

    # Compute central estimate per column
    if statistic == 'median':
        central = np.nanmedian(arr, axis=0)
    elif statistic == 'mean':
        central = np.nanmean(arr, axis=0)
    elif statistic == 'iqm':
        central = np.apply_along_axis(lambda x: iqm(x), 0, arr)
    else:
        raise ValueError(f"Unknown statistic: {statistic}")

    # Bootstrap CI across seeds per timepoint
    if n_bootstrap <= 0 or arr.shape[0] == 1:
        ci_low = np.full_like(central, np.nan)
        ci_high = np.full_like(central, np.nan)
    else:
        rng = np.random.default_rng(seed)
        n_seeds, n_steps = arr.shape
        boots = np.empty((n_bootstrap, n_steps), dtype=np.float64)
        for b in range(n_bootstrap):
            idx = rng.integers(0, n_seeds, size=n_seeds)
            sample = arr[idx, :]
            if statistic == 'median':
                boots[b, :] = np.nanmedian(sample, axis=0)
            elif statistic == 'mean':
                boots[b, :] = np.nanmean(sample, axis=0)
            else:
                boots[b, :] = np.apply_along_axis(lambda x: iqm(x), 0, sample)

        ci_low = np.quantile(boots, alpha / 2.0, axis=0)
        ci_high = np.quantile(boots, 1.0 - alpha / 2.0, axis=0)

    return {
        "steps": list(map(int, all_steps)),
        "central": [float(x) if np.isfinite(x) else None for x in central.tolist()],
        "ci_low": [float(x) if np.isfinite(x) else None for x in ci_low.tolist()],
        "ci_high": [float(x) if np.isfinite(x) else None for x in ci_high.tolist()],
    }


if __name__ == '__main__':
    # Lightweight CLI to produce JSON summary + optional PNG plots for a runs directory.
    import argparse
    import os
    import json

    from collections import defaultdict

    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-dir', required=True)
    ap.add_argument('--out-dir', default='results')
    ap.add_argument('--bootstrap', type=int, default=1000)
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--statistic', type=str, default='median', choices=['mean', 'median', 'iqm'])
    ap.add_argument('--make-plots', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    event_files = find_event_files(args.runs_dir)
    identities = [infer_group_and_seed(args.runs_dir, ef) for ef in event_files]

    # Group by method/group_key then seed
    grouped = defaultdict(lambda: defaultdict(list))
    for idt in identities:
        mk = idt.group_key
        grouped[mk][idt.seed].append(idt.event_file)

    summary = []
    curves_out = {}

    for mk, seed_map in grouped.items():
        per_seed = {}
        for seed, files in seed_map.items():
            # merge scalars across files for this seed
            dicts = []
            for f in files:
                try:
                    dicts.append(load_scalars(f))
                except Exception:
                    continue
            if not dicts:
                continue
            merged = merge_scalars_dicts(dicts)
            metrics = {
                'curves': {
                    'eval_iqm': extract_task_avg_eval_iqm_curve(merged),
                    'forgetting': extract_task_avg_isolated_forgetting_curve(merged),
                    'dormant_frac': extract_task_avg_dormant_frac_curve(merged),
                    'effective_rank_avg': None,
                }
            }
            per_seed[seed] = metrics

        # Aggregate curves across seeds
        eval_curves = [per_seed[s]['curves']['eval_iqm'] for s in per_seed if per_seed[s]['curves']['eval_iqm'] is not None]
        forget_curves = [per_seed[s]['curves']['forgetting'] for s in per_seed if per_seed[s]['curves']['forgetting'] is not None]
        dorm_curves = [per_seed[s]['curves']['dormant_frac'] for s in per_seed if per_seed[s]['curves']['dormant_frac'] is not None]

        curves_out[mk] = {
            'eval_iqm_curve': aggregate_curves_across_seeds(eval_curves, args.bootstrap, args.alpha, seed=0, statistic=args.statistic),
            'forgetting_curve': aggregate_curves_across_seeds(forget_curves, args.bootstrap, args.alpha, seed=1, statistic=args.statistic),
            'dormant_frac_curve': aggregate_curves_across_seeds(dorm_curves, args.bootstrap, args.alpha, seed=2, statistic=args.statistic),
        }

    out_json = os.path.join(args.out_dir, 'curves_mean_ci.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(curves_out, f, indent=2)
    print(f'Wrote: {out_json}')

