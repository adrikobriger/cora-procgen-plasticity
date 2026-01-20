#!/usr/bin/env python3
"""
Plot grid-search snapshot curves from snapshots.json for an intervention.

- Scans runs_root for snapshots.json under the specified intervention.
- Groups seeds by trial directory (config).
- Plots mean curves for all configs.
- Adds 95% CI only for the best-performing config (based on final snapshot metric).
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def _load_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _extract_series(snapshots: List[dict], metric: str) -> List[float]:
    key = "aggregate_iqm" if metric == "iqm" else "aggregate_mean"
    series = []
    for snap in snapshots:
        val = snap.get(key)
        if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
            series.append(float("nan"))
        else:
            series.append(float(val))
    return series


def _truncate_and_stack(series_list: List[List[float]]) -> np.ndarray:
    if not series_list:
        return np.empty((0, 0), dtype=np.float64)
    min_len = min(len(s) for s in series_list)
    if min_len == 0:
        return np.empty((0, 0), dtype=np.float64)
    stacked = np.array([np.asarray(s[:min_len], dtype=np.float64) for s in series_list])
    return stacked


def _bootstrap_ci(values: np.ndarray, num_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(0)
    n = values.size
    samples = rng.choice(values, size=(num_bootstrap, n), replace=True)
    means = samples.mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lower = np.quantile(means, alpha)
    upper = np.quantile(means, 1.0 - alpha)
    return float(lower), float(upper)


def _best_config_key(config_series: Dict[str, List[List[float]]], metric: str) -> Optional[str]:
    best_key = None
    best_val = -float("inf")
    for key, series_list in config_series.items():
        stacked = _truncate_and_stack(series_list)
        if stacked.size == 0:
            continue
        final_vals = stacked[:, -1]
        final_vals = final_vals[np.isfinite(final_vals)]
        if final_vals.size == 0:
            continue
        mean_final = float(np.mean(final_vals))
        if mean_final > best_val:
            best_val = mean_final
            best_key = key
    return best_key


def _candidate_label(trial_dir: Path) -> str:
    params_path = trial_dir / "candidate_params.json"
    params = _load_json(params_path)
    if not isinstance(params, dict):
        return trial_dir.name
    bits = []
    for k in sorted(params.keys()):
        v = params[k]
        bits.append(f"{k}={v}")
    return ", ".join(bits) if bits else trial_dir.name


def _collect_snapshots(runs_root: Path, intervention: Optional[str]) -> Dict[str, List[List[float]]]:
    config_series: Dict[str, List[List[float]]] = {}

    for root, _dirs, files in os.walk(runs_root):
        if "snapshots.json" not in files:
            continue
        path = Path(root) / "snapshots.json"

        if intervention:
            parts = [p.lower() for p in path.parts]
            if intervention.lower() not in parts:
                continue

        # Expect .../trial_xxx/seed_y/snapshots.json
        seed_dir = path.parent
        trial_dir = seed_dir.parent
        if not trial_dir.name.startswith("trial_"):
            continue

        snapshots = _load_json(path)
        if not isinstance(snapshots, list):
            continue

        key = str(trial_dir.resolve())
        config_series.setdefault(key, []).append(snapshots)

    return config_series


def _collect_event_runs(runs_root: Path, intervention: Optional[str]) -> Dict[str, List[Path]]:
    config_event_dirs: Dict[str, List[Path]] = {}

    for root, _dirs, files in os.walk(runs_root):
        has_event = any(f.startswith("events.out.tfevents") for f in files)
        if not has_event:
            continue

        path = Path(root)
        if intervention:
            parts = [p.lower() for p in path.parts]
            if intervention.lower() not in parts:
                continue

        # Expect .../trial_xxx/seed_y/tb
        if path.name != "tb":
            continue
        seed_dir = path.parent
        trial_dir = seed_dir.parent
        if not trial_dir.name.startswith("trial_"):
            continue

        key = str(trial_dir.resolve())
        config_event_dirs.setdefault(key, []).append(path)

    return config_event_dirs


def _extract_event_series(
    event_dir: Path,
    tag_prefix: str,
    task_ids: Optional[List[int]] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    ea = event_accumulator.EventAccumulator(str(event_dir))
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    matching = [t for t in tags if t.startswith(tag_prefix)]
    if task_ids:
        wanted = {f"t{tid}" for tid in task_ids}
        filtered = []
        for tag in matching:
            run_id = tag[len(tag_prefix):]
            if any(run_id.endswith(w) for w in wanted):
                filtered.append(tag)
        matching = filtered
    if not matching:
        return None

    step_values = {}
    for tag in matching:
        try:
            for ev in ea.Scalars(tag):
                step_values.setdefault(ev.step, []).append(ev.value)
        except KeyError:
            continue

    if not step_values:
        return None

    steps = sorted(step_values.keys())
    values = [float(np.mean(step_values[s])) for s in steps]
    return np.array(steps, dtype=np.int64), np.array(values, dtype=np.float64)


def _parse_task_id(run_id: str, task_ids: Optional[List[int]]) -> Optional[int]:
    if not task_ids:
        return None
    for tid in task_ids:
        if run_id.endswith(f"t{tid}"):
            return tid
    return None


def _collect_task_ranges(
    event_dir: Path,
    tag_prefix: str,
    task_ids: Optional[List[int]],
) -> Dict[int, Tuple[int, int]]:
    ranges: Dict[int, Tuple[int, int]] = {}
    ea = event_accumulator.EventAccumulator(str(event_dir))
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    matching = [t for t in tags if t.startswith(tag_prefix)]
    for tag in matching:
        run_id = tag[len(tag_prefix):]
        tid = _parse_task_id(run_id, task_ids)
        if tid is None:
            continue
        try:
            steps = [ev.step for ev in ea.Scalars(tag)]
        except KeyError:
            continue
        if not steps:
            continue
        lo = int(min(steps))
        hi = int(max(steps))
        if tid in ranges:
            prev_lo, prev_hi = ranges[tid]
            ranges[tid] = (min(prev_lo, lo), max(prev_hi, hi))
        else:
            ranges[tid] = (lo, hi)
    return ranges


def _aggregate_task_boundaries(
    event_dirs: List[Path],
    tag_prefix: str,
    task_ids: Optional[List[int]],
) -> Dict[int, Tuple[int, int]]:
    if not task_ids:
        return {}
    per_tid_mins: Dict[int, List[int]] = {tid: [] for tid in task_ids}
    per_tid_maxs: Dict[int, List[int]] = {tid: [] for tid in task_ids}

    for d in event_dirs:
        ranges = _collect_task_ranges(d, tag_prefix, task_ids)
        for tid, (lo, hi) in ranges.items():
            per_tid_mins[tid].append(lo)
            per_tid_maxs[tid].append(hi)

    boundaries: Dict[int, Tuple[int, int]] = {}
    for tid in task_ids:
        if per_tid_mins[tid] and per_tid_maxs[tid]:
            lo = int(np.median(per_tid_mins[tid]))
            hi = int(np.median(per_tid_maxs[tid]))
            boundaries[tid] = (lo, hi)
    return boundaries


def _align_series(series_list: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    if not series_list:
        return np.array([], dtype=np.int64), np.empty((0, 0), dtype=np.float64)

    common_steps = None
    for steps, _vals in series_list:
        step_set = set(steps.tolist())
        common_steps = step_set if common_steps is None else (common_steps & step_set)
    if not common_steps:
        # fall back to union with interpolation
        all_steps = sorted({s for steps, _ in series_list for s in steps.tolist()})
        grid = np.array(all_steps, dtype=np.int64)
    else:
        grid = np.array(sorted(common_steps), dtype=np.int64)

    aligned = []
    for steps, vals in series_list:
        if grid.size == 0:
            continue
        if np.array_equal(grid, steps):
            aligned.append(vals)
        else:
            aligned.append(np.interp(grid, steps, vals))
    if not aligned:
        return grid, np.empty((0, 0), dtype=np.float64)
    return grid, np.vstack(aligned)


def _plot_grid_snapshots(
    config_series: Dict[str, List[List[dict]]],
    metric: str,
    title: str,
    output_path: Path,
    show_legend: bool,
    num_bootstrap: int,
) -> None:
    if not config_series:
        raise RuntimeError("No snapshots.json files found for the requested intervention.")

    series_by_config: Dict[str, List[List[float]]] = {}
    for cfg, snaps_list in config_series.items():
        series_by_config[cfg] = [_extract_series(snaps, metric) for snaps in snaps_list]

    best_cfg = _best_config_key(series_by_config, metric)

    plt.figure(figsize=(10, 6))

    for cfg, series_list in series_by_config.items():
        stacked = _truncate_and_stack(series_list)
        if stacked.size == 0:
            continue
        mean_curve = np.nanmean(stacked, axis=0)
        x = np.arange(mean_curve.size)
        label = _candidate_label(Path(cfg))

        if cfg == best_cfg:
            plt.plot(x, mean_curve, linewidth=2.5, label=f"BEST: {label}")
        else:
            plt.plot(x, mean_curve, linewidth=1.0, alpha=0.5, label=label)

    if best_cfg is not None:
        stacked = _truncate_and_stack(series_by_config[best_cfg])
        if stacked.size > 0:
            x = np.arange(stacked.shape[1])
            lower = np.empty(stacked.shape[1], dtype=np.float64)
            upper = np.empty(stacked.shape[1], dtype=np.float64)
            for i in range(stacked.shape[1]):
                vals = stacked[:, i]
                vals = vals[np.isfinite(vals)]
                lo, hi = _bootstrap_ci(vals, num_bootstrap=num_bootstrap)
                lower[i] = lo
                upper[i] = hi
            plt.fill_between(x, lower, upper, alpha=0.2, linewidth=0)

    plt.title(title)
    plt.xlabel("snapshot index")
    ylabel = "IQM" if metric == "iqm" else "Mean"
    plt.ylabel(f"Aggregate {ylabel}")

    if show_legend:
        plt.legend(fontsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)


def _plot_grid_events(
    config_event_dirs: Dict[str, List[Path]],
    tag_prefix: str,
    title: str,
    output_path: Path,
    show_legend: bool,
    num_bootstrap: int,
    task_ids: Optional[List[int]],
) -> None:
    if not config_event_dirs:
        raise RuntimeError("No event files found for the requested intervention.")

    series_by_config: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
    for cfg, dirs in config_event_dirs.items():
        per_seed_series = []
        for d in dirs:
            series = _extract_event_series(d, tag_prefix, task_ids=task_ids)
            if series is not None:
                per_seed_series.append(series)
        if per_seed_series:
            series_by_config[cfg] = per_seed_series

    if not series_by_config:
        raise RuntimeError("No matching tag data found in event files.")

    best_cfg = None
    best_val = -float("inf")
    for cfg, series_list in series_by_config.items():
        steps, aligned = _align_series(series_list)
        if aligned.size == 0:
            continue
        final_vals = aligned[:, -1]
        final_vals = final_vals[np.isfinite(final_vals)]
        if final_vals.size == 0:
            continue
        mean_final = float(np.mean(final_vals))
        if mean_final > best_val:
            best_val = mean_final
            best_cfg = cfg

    boundaries = {}
    if best_cfg is not None:
        best_dirs = config_event_dirs.get(best_cfg, [])
        boundaries = _aggregate_task_boundaries(best_dirs, tag_prefix, task_ids)

    plt.figure(figsize=(10, 6))

    for cfg, series_list in series_by_config.items():
        steps, aligned = _align_series(series_list)
        if aligned.size == 0:
            continue
        mean_curve = np.nanmean(aligned, axis=0)
        label = _candidate_label(Path(cfg))
        if cfg == best_cfg:
            plt.plot(steps, mean_curve, linewidth=2.5, label=f"BEST: {label}")
        else:
            plt.plot(steps, mean_curve, linewidth=1.0, alpha=0.5, label=label)

    if best_cfg is not None:
        steps, aligned = _align_series(series_by_config[best_cfg])
        if aligned.size > 0:
            lower = np.empty(aligned.shape[1], dtype=np.float64)
            upper = np.empty(aligned.shape[1], dtype=np.float64)
            for i in range(aligned.shape[1]):
                vals = aligned[:, i]
                vals = vals[np.isfinite(vals)]
                lo, hi = _bootstrap_ci(vals, num_bootstrap=num_bootstrap)
                lower[i] = lo
                upper[i] = hi
            plt.fill_between(steps, lower, upper, alpha=0.2, linewidth=0)

    if boundaries:
        ordered_tids = [tid for tid in task_ids if tid in boundaries] if task_ids else list(boundaries.keys())
        ordered_tids = sorted(ordered_tids, key=lambda t: boundaries[t][0])
        for tid in ordered_tids[:-1]:
            _, hi = boundaries[tid]
            plt.axvline(x=hi, color="gray", linestyle="--", linewidth=1, alpha=0.6)

        y_min, y_max = plt.ylim()
        y_text = y_min + 0.95 * (y_max - y_min)
        for tid in ordered_tids:
            lo, hi = boundaries[tid]
            mid = (lo + hi) / 2.0
            plt.text(mid, y_text, f"task {tid}", ha="center", va="top", fontsize=9, color="gray")

    plt.title(title)
    plt.xlabel("timestep")
    plt.ylabel(tag_prefix.rstrip("/") )

    if show_legend:
        plt.legend(fontsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot grid-search snapshots with CI for best config.")
    parser.add_argument("--runs_root", type=str, default="runs", help="Root directory that contains runs.")
    parser.add_argument("--intervention", type=str, default=None, help="Intervention name (e.g., gmp/set/redo). If omitted, autodetects all.")
    parser.add_argument("--metric", type=str, default="iqm", choices=["iqm", "mean"], help="Metric to plot.")
    parser.add_argument("--title", type=str, default=None, help="Plot title.")
    parser.add_argument("--out", type=str, default=None, help="Output PNG path. If omitted, auto-names under runs/plots.")
    parser.add_argument("--legend", action="store_true", help="Show legend.")
    parser.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap samples for CI.")
    parser.add_argument("--source", type=str, default="events", choices=["events", "snapshots"], help="Data source.")
    parser.add_argument("--tag_prefix", type=str, default="train_reward_iqm/", help="Scalar tag prefix for event files.")
    parser.add_argument("--task_ids", type=str, default="0,2,4", help="Comma-separated task ids to include (e.g., 0,2,4).")

    args = parser.parse_args()
    runs_root = Path(args.runs_root)
    intervention = args.intervention

    task_ids = [int(x) for x in args.task_ids.split(",") if x.strip() != ""] if args.task_ids else None

    if intervention:
        if args.source == "snapshots":
            config_series = _collect_snapshots(runs_root, intervention)
            title = args.title or f"Grid snapshots ({intervention}, metric={args.metric})"
            out_path = Path(args.out) if args.out else runs_root / "plots" / f"grid_snapshots_{intervention}_{args.metric}.png"

            _plot_grid_snapshots(
                config_series=config_series,
                metric=args.metric,
                title=title,
                output_path=out_path,
                show_legend=args.legend,
                num_bootstrap=args.bootstrap,
            )
        else:
            config_events = _collect_event_runs(runs_root, intervention)
            title = args.title or f"Grid events ({intervention}, tag={args.tag_prefix})"
            out_path = Path(args.out) if args.out else runs_root / "plots" / f"grid_events_{intervention}_{args.tag_prefix.rstrip('/')}.png"
            _plot_grid_events(
                config_event_dirs=config_events,
                tag_prefix=args.tag_prefix,
                title=title,
                output_path=out_path,
                show_legend=args.legend,
                num_bootstrap=args.bootstrap,
                task_ids=task_ids,
            )
        return

    # Autodetect interventions
    found = set()
    for root, _dirs, files in os.walk(runs_root):
        if "snapshots.json" not in files:
            continue
        parts = [p.lower() for p in Path(root).parts]
        for token in ("gmp", "set", "redo", "reset", "dense", "online_ewc", "ewc", "pnc"):
            if token in parts:
                found.add(token)

    if not found:
        raise RuntimeError("No interventions found under runs/ with snapshots.json.")

    for token in sorted(found):
        if args.source == "snapshots":
            config_series = _collect_snapshots(runs_root, token)
            title = args.title or f"Grid snapshots ({token}, metric={args.metric})"
            out_path = Path(args.out) if args.out else runs_root / "plots" / f"grid_snapshots_{token}_{args.metric}.png"
            _plot_grid_snapshots(
                config_series=config_series,
                metric=args.metric,
                title=title,
                output_path=out_path,
                show_legend=args.legend,
                num_bootstrap=args.bootstrap,
            )
        else:
            config_events = _collect_event_runs(runs_root, token)
            title = args.title or f"Grid events ({token}, tag={args.tag_prefix})"
            out_path = Path(args.out) if args.out else runs_root / "plots" / f"grid_events_{token}_{args.tag_prefix.rstrip('/')}.png"
            _plot_grid_events(
                config_event_dirs=config_events,
                tag_prefix=args.tag_prefix,
                title=title,
                output_path=out_path,
                show_legend=args.legend,
                num_bootstrap=args.bootstrap,
                task_ids=task_ids,
            )


if __name__ == "__main__":
    main()
