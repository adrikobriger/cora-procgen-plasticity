# tools/analyze_results.py
from __future__ import annotations

import os
import json
import argparse
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

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
    
    Per-seed summaries (BEFORE aggregation across seeds):
      - final_iqm_return: LAST value of eval curve
      - peak_iqm_return: MAX value of eval curve
      - max_isolated_forgetting: MAX value of forgetting curve
      - final_effective_rank: AVERAGE of last K values of rank curve
    
    Returns dict with:
      - per-seed summary scalars
      - curves for time-series aggregation
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

    # ---- sparsity (optional: for curves only) ----
    sparsity_curve = None
    for tag in ["gmp/achieved_sparsity", "set/achieved_sparsity", "sparsity/achieved"]:
        sparsity_curve = get_curve(scalars, tag)
        if sparsity_curve is not None:
            break

    if sparsity_curve is not None:
        s_steps, s_vals = sparsity_curve
    else:
        s_steps, s_vals = None, None

    # ---- effective rank: AVERAGE of last K values (robust) ----
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
            "sparsity": (s_steps, s_vals),
            "effective_rank_avg": (er_steps, er_vals),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=str, required=True, help="Root folder containing all runs")
    ap.add_argument("--out-dir", type=str, default="results", help="Where to write outputs")
    ap.add_argument("--bootstrap", type=int, default=10000, help="Bootstrap resamples (10000 recommended)")
    ap.add_argument("--alpha", type=float, default=0.05, help="CI alpha (0.05 => 95%)")
    ap.add_argument("--last-k-rank", type=int, default=10, help="Average last K effective rank points per seed")
    ap.add_argument("--statistic", type=str, default="median", choices=["mean", "median", "iqm"], 
                    help="Central estimate: median (robust, recommended), mean, or iqm")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) find all event files
    event_files = find_event_files(args.runs_dir)
    if args.verbose:
        print(f"Found {len(event_files)} TensorBoard event files under: {args.runs_dir}")

    # 2) infer group+seed for each event file
    identities = [infer_group_and_seed(args.runs_dir, ef) for ef in event_files]

    # 3) group into: group_key -> seed -> [event_files]
    grouped: Dict[str, Dict[str, List[str]]] = {}
    for rid in identities:
        grouped.setdefault(rid.group_key, {})
        grouped[rid.group_key].setdefault(rid.seed, [])
        grouped[rid.group_key][rid.seed].append(rid.event_file)

    if args.verbose:
        print("Discovered groups (seeds):")
        for gk, seedmap in sorted(grouped.items()):
            print(f"  {gk}: seeds={sorted(seedmap.keys())}")

    # 4) extract per-seed metrics per group (skip invalid event files)
    group_seed_metrics: Dict[str, Dict[str, Dict[str, Any]]] = {}
    skipped = 0

    for group_key, seed_map in grouped.items():
        group_seed_metrics[group_key] = {}
        for seed, files in seed_map.items():
            best = choose_best_event_file(files)

            try:
                scalars = load_scalars(best)
            except ValueError as e:
                skipped += 1
                if args.verbose:
                    print(f"[SKIP] {group_key} seed={seed}")
                    print(f"       File: {best}")
                    print(f"       Reason: {e}")
                continue

            m = extract_run_metrics(scalars, last_k_rank=args.last_k_rank)
            group_seed_metrics[group_key][seed] = m

    if args.verbose:
        print(f"Skipped {skipped} invalid/corrupt event files.")

    # 5) aggregate across seeds: scalars + curves
    summary_rows = []
    curves_out: Dict[str, Any] = {}

    for group_key, seed_dict in sorted(group_seed_metrics.items()):
        seeds = sorted(seed_dict.keys())
        n_seeds = len(seeds)
        if n_seeds == 0:
            continue

        # Collect per-seed summary values
        final_iqm_vals = np.array([seed_dict[s]["final_iqm_return"] for s in seeds], dtype=np.float64)
        peak_iqm_vals = np.array([seed_dict[s]["peak_iqm_return"] for s in seeds], dtype=np.float64)
        max_forget_vals = np.array([seed_dict[s]["max_isolated_forgetting"] for s in seeds], dtype=np.float64)
        final_er_vals = np.array([seed_dict[s]["final_effective_rank"] for s in seeds], dtype=np.float64)
        final_dorm_vals = np.array([seed_dict[s]["final_dormant_frac"] for s in seeds], dtype=np.float64)
        peak_dorm_vals = np.array([seed_dict[s]["peak_dormant_frac"] for s in seeds], dtype=np.float64)

        # Bootstrap CI across seeds with chosen statistic (median recommended)
        final_iqm_stat, final_iqm_lo, final_iqm_hi = bootstrap_ci(final_iqm_vals, args.bootstrap, args.alpha, seed=0, statistic=args.statistic)
        peak_iqm_stat, peak_iqm_lo, peak_iqm_hi = bootstrap_ci(peak_iqm_vals, args.bootstrap, args.alpha, seed=1, statistic=args.statistic)
        max_forget_stat, max_forget_lo, max_forget_hi = bootstrap_ci(max_forget_vals, args.bootstrap, args.alpha, seed=2, statistic=args.statistic)
        final_er_stat, final_er_lo, final_er_hi = bootstrap_ci(final_er_vals, args.bootstrap, args.alpha, seed=3, statistic=args.statistic)
        final_dorm_stat, final_dorm_lo, final_dorm_hi = bootstrap_ci(final_dorm_vals, args.bootstrap, args.alpha, seed=4, statistic=args.statistic)
        peak_dorm_stat, peak_dorm_lo, peak_dorm_hi = bootstrap_ci(peak_dorm_vals, args.bootstrap, args.alpha, seed=5, statistic=args.statistic)

        # Use consistent naming based on statistic choice
        stat_label = args.statistic  # "median", "mean", or "iqm"

        summary_rows.append({
            "method": group_key,
            "n_seeds": n_seeds,
            "last_k_rank": args.last_k_rank,
            "bootstrap_samples": args.bootstrap,
            "statistic": stat_label,

            f"final_iqm_return_{stat_label}": final_iqm_stat,
            "final_iqm_return_ci_low": final_iqm_lo,
            "final_iqm_return_ci_high": final_iqm_hi,

            f"peak_iqm_return_{stat_label}": peak_iqm_stat,
            "peak_iqm_return_ci_low": peak_iqm_lo,
            "peak_iqm_return_ci_high": peak_iqm_hi,

            f"max_isolated_forgetting_{stat_label}": max_forget_stat,
            "max_isolated_forgetting_ci_low": max_forget_lo,
            "max_isolated_forgetting_ci_high": max_forget_hi,

            f"final_effective_rank_{stat_label}": final_er_stat,
            "final_effective_rank_ci_low": final_er_lo,
            "final_effective_rank_ci_high": final_er_hi,

            f"final_dormant_frac_{stat_label}": final_dorm_stat,
            "final_dormant_frac_ci_low": final_dorm_lo,
            "final_dormant_frac_ci_high": final_dorm_hi,

            f"peak_dormant_frac_{stat_label}": peak_dorm_stat,
            "peak_dormant_frac_ci_low": peak_dorm_lo,
            "peak_dormant_frac_ci_high": peak_dorm_hi,
        })

        # curves aggregation per metric
        eval_curves = []
        forget_curves = []
        dorm_curves = []
        spar_curves = []
        er_curves = []

        for s in seeds:
            curves = seed_dict[s]["curves"]
            if curves["eval_iqm"][0] is not None:
                eval_curves.append(curves["eval_iqm"])
            if curves["forgetting"][0] is not None:
                forget_curves.append(curves["forgetting"])
            if curves["dormant_frac"][0] is not None:
                dorm_curves.append(curves["dormant_frac"])
            if curves["sparsity"][0] is not None:
                spar_curves.append(curves["sparsity"])
            if curves["effective_rank_avg"][0] is not None:
                er_curves.append(curves["effective_rank_avg"])

        curves_out[group_key] = {
            "eval_iqm_curve": aggregate_curves_across_seeds(eval_curves, args.bootstrap, args.alpha, seed=10, statistic=args.statistic),
            "forgetting_curve": aggregate_curves_across_seeds(forget_curves, args.bootstrap, args.alpha, seed=11, statistic=args.statistic),
            # dormancy is optimizer-step axis; still aggregated but plot separately
            "dormant_frac_curve": aggregate_curves_across_seeds(dorm_curves, args.bootstrap, args.alpha, seed=12, statistic=args.statistic),
            "sparsity_curve": aggregate_curves_across_seeds(spar_curves, args.bootstrap, args.alpha, seed=13, statistic=args.statistic),
            "effective_rank_avg_curve": aggregate_curves_across_seeds(er_curves, args.bootstrap, args.alpha, seed=14, statistic=args.statistic),
        }

    # 6) write outputs
    df = pd.DataFrame(summary_rows)

    csv_path = os.path.join(args.out_dir, "summary_table.csv")
    json_path = os.path.join(args.out_dir, "summary_table.json")
    curves_path = os.path.join(args.out_dir, "curves_mean_ci.json")

    df.to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    with open(curves_path, "w", encoding="utf-8") as f:
        json.dump(curves_out, f, indent=2)

    print("=" * 140)
    print("CONTINUAL RL RESULTS SUMMARY")
    print("=" * 140)
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {curves_path}")


if __name__ == "__main__":
    main()