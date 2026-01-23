# tools/analyze_results.py
from __future__ import annotations

import os
import json
import argparse
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from result_utils import (
    find_event_files,
    infer_group_and_seed,
    load_scalars,
    merge_scalars_dicts,                 # NEW
    get_curve,
    extract_task_avg_eval_iqm_curve,
    extract_task_avg_dormant_frac_curve,
    extract_task_avg_isolated_forgetting_curve,
    load_effective_rank_curve_from_json,
    bootstrap_ci,
    aggregate_curves_across_seeds,
)


def choose_best_event_file(files: List[str]) -> str:
    """
    Pick best event file as a reference path:
    - prefer larger size
    - then newer modified time

    NOTE: we now MERGE scalars across ALL files,
    but we still keep this to locate JSON files, etc.
    """
    def score(fp: str):
        try:
            return (os.path.getsize(fp), os.path.getmtime(fp))
        except OSError:
            return (0, 0)

    return sorted(files, key=score, reverse=True)[0]


def _normalize_method_only(group_key: str) -> str:
    """
    If runs-dir already points to: .../procgen_3_tasks_1_cycle_500k_starpilot
    then group_key often looks like: gmp/20260.../trial_000
    We want method key: gmp
    """
    parts = group_key.replace("\\", "/").split("/")
    return parts[0] if len(parts) > 0 else group_key


def extract_run_metrics(
    event_file: str,
    scalars: Dict[str, List[Tuple[int, float]]],
    last_k_rank: int = 10,
    debug: bool = False,
) -> Dict[str, Any]:

    # ---- Eval IQM curve ----
    eval_curve = extract_task_avg_eval_iqm_curve(scalars)
    if eval_curve is not None:
        eval_steps, eval_vals = eval_curve
        final_iqm = float(eval_vals[-1])
        peak_iqm = float(np.nanmax(eval_vals))
    else:
        eval_steps, eval_vals = None, None
        final_iqm = float("nan")
        peak_iqm = float("nan")

    # ---- Forgetting curve ----
    forget_curve = get_curve(scalars, "forgetting/isolated_avg")
    if forget_curve is None:
        forget_curve = extract_task_avg_isolated_forgetting_curve(scalars)

    if forget_curve is not None:
        f_steps, f_vals = forget_curve
        max_forgetting = float(np.nanmax(f_vals))
    else:
        f_steps, f_vals = None, None
        max_forgetting = float("nan")

    # ---- Dormant frac ----
    dorm_curve = extract_task_avg_dormant_frac_curve(scalars)
    if dorm_curve is not None:
        d_steps, d_vals = dorm_curve
        final_dormant = float(d_vals[-1])
        peak_dormant = float(np.nanmax(d_vals))
    else:
        d_steps, d_vals = None, None
        final_dormant = float("nan")
        peak_dormant = float("nan")

    # ---- Effective rank ----
    er_curve = get_curve(scalars, "effective_rank/avg")
    if er_curve is None:
        er_curve = get_curve(scalars, "effective_rank/across_tasks_avg")
    if er_curve is None:
        er_curve = load_effective_rank_curve_from_json(event_file)

    if er_curve is not None:
        er_steps, er_vals = er_curve
        tail = er_vals[-last_k_rank:] if len(er_vals) >= last_k_rank else er_vals
        final_er = float(np.nanmean(tail)) if len(tail) > 0 else float("nan")
    else:
        er_steps, er_vals = None, None
        final_er = float("nan")

    if debug:
        def _desc(name: str, steps, vals):
            if steps is None or vals is None:
                print(f"    {name}: NONE")
                return
            steps = np.asarray(steps)
            vals = np.asarray(vals)
            print(f"    {name}: len={len(steps)}  step[min,max]=({steps.min()}, {steps.max()})  nan%={np.isnan(vals).mean()*100:.1f}")

        print(f"[DEBUG] event_file={event_file}")
        _desc("eval_iqm", eval_steps, eval_vals)
        _desc("forget", f_steps, f_vals)
        _desc("dorm", d_steps, d_vals)
        _desc("eff_rank", er_steps, er_vals)

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


# ---------------------------
# Plot: ONE FIGURE PER METHOD PER METRIC
# ---------------------------

def plot_one_method_one_metric(
    method_name: str,
    curve: Dict[str, Any],
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: str,
):
    # Skip if missing
    if (
        curve is None
        or len(curve.get("steps", [])) == 0
        or len(curve.get("central", [])) == 0
        or np.all(np.isnan(np.array(curve.get("central", []), dtype=np.float64)))
    ):
        print(f"[PLOT] Skip (missing): {out_path}")
        return

    steps = np.array(curve["steps"], dtype=np.float64)
    central = np.array(curve["central"], dtype=np.float64)
    lo = np.array(curve["ci_low"], dtype=np.float64)
    hi = np.array(curve["ci_high"], dtype=np.float64)

    plt.figure(figsize=(7.5, 4.8))
    plt.plot(steps, central, linewidth=2)
    plt.fill_between(steps, lo, hi, alpha=0.20)

    plt.title(f"{method_name.upper()} — {title}", fontsize=13, fontweight="bold")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()
    print(f"[PLOT] Wrote: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=str, default="results")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--last-k-rank", type=int, default=10)
    ap.add_argument("--statistic", type=str, default="median", choices=["mean", "median", "iqm"])
    ap.add_argument("--verbose", action="store_true")

    ap.add_argument("--make-plots", action="store_true")
    ap.add_argument("--plots-dir", type=str, default=None)
    ap.add_argument("--debug-curves", action="store_true")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    event_files = find_event_files(args.runs_dir)
    if args.verbose:
        print(f"Found {len(event_files)} event files under {args.runs_dir}")

    identities = [infer_group_and_seed(args.runs_dir, ef) for ef in event_files]

    # method -> seed -> [event files]
    grouped: Dict[str, Dict[str, List[str]]] = {}
    for rid in identities:
        mk = _normalize_method_only(rid.group_key)
        grouped.setdefault(mk, {})
        grouped[mk].setdefault(rid.seed, [])
        grouped[mk][rid.seed].append(rid.event_file)

    if args.verbose:
        print("Interventions discovered:")
        for mk, seedmap in sorted(grouped.items()):
            print(f"  {mk}: seeds={sorted(seedmap.keys())}")

    # extract per seed (MERGE across all event files per seed)
    group_seed_metrics: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for method_key, seed_map in grouped.items():
        group_seed_metrics[method_key] = {}
        for seed, files in seed_map.items():
            best = choose_best_event_file(files)

            all_scalars = []
            for fp in files:
                try:
                    all_scalars.append(load_scalars(fp))
                except ValueError:
                    continue

            if not all_scalars:
                continue

            scalars_merged = merge_scalars_dicts(all_scalars)

            m = extract_run_metrics(
                best,
                scalars_merged,
                last_k_rank=args.last_k_rank,
                debug=args.debug_curves
            )
            group_seed_metrics[method_key][seed] = m

    summary_rows = []
    curves_out: Dict[str, Any] = {}

    for method_key, seed_dict in sorted(group_seed_metrics.items()):
        seeds = sorted(seed_dict.keys())
        if len(seeds) == 0:
            continue

        final_iqm_vals = np.array([seed_dict[s]["final_iqm_return"] for s in seeds], dtype=np.float64)
        peak_iqm_vals = np.array([seed_dict[s]["peak_iqm_return"] for s in seeds], dtype=np.float64)
        max_forget_vals = np.array([seed_dict[s]["max_isolated_forgetting"] for s in seeds], dtype=np.float64)
        final_er_vals = np.array([seed_dict[s]["final_effective_rank"] for s in seeds], dtype=np.float64)
        final_dorm_vals = np.array([seed_dict[s]["final_dormant_frac"] for s in seeds], dtype=np.float64)
        peak_dorm_vals = np.array([seed_dict[s]["peak_dormant_frac"] for s in seeds], dtype=np.float64)

        final_iqm_stat, final_iqm_lo, final_iqm_hi = bootstrap_ci(final_iqm_vals, args.bootstrap, args.alpha, seed=0, statistic=args.statistic)
        peak_iqm_stat, peak_iqm_lo, peak_iqm_hi = bootstrap_ci(peak_iqm_vals, args.bootstrap, args.alpha, seed=1, statistic=args.statistic)
        max_forget_stat, max_forget_lo, max_forget_hi = bootstrap_ci(max_forget_vals, args.bootstrap, args.alpha, seed=2, statistic=args.statistic)
        final_er_stat, final_er_lo, final_er_hi = bootstrap_ci(final_er_vals, args.bootstrap, args.alpha, seed=3, statistic=args.statistic)
        final_dorm_stat, final_dorm_lo, final_dorm_hi = bootstrap_ci(final_dorm_vals, args.bootstrap, args.alpha, seed=4, statistic=args.statistic)
        peak_dorm_stat, peak_dorm_lo, peak_dorm_hi = bootstrap_ci(peak_dorm_vals, args.bootstrap, args.alpha, seed=5, statistic=args.statistic)

        stat_label = args.statistic

        summary_rows.append({
            "method": method_key,
            "n_seeds": len(seeds),
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

        eval_curves, forget_curves, dorm_curves, er_curves = [], [], [], []

        for s in seeds:
            curves = seed_dict[s]["curves"]
            if curves["eval_iqm"][0] is not None:
                eval_curves.append(curves["eval_iqm"])
            if curves["forgetting"][0] is not None:
                forget_curves.append(curves["forgetting"])
            if curves["dormant_frac"][0] is not None:
                dorm_curves.append(curves["dormant_frac"])
            if curves["effective_rank_avg"][0] is not None:
                er_curves.append(curves["effective_rank_avg"])

        curves_out[method_key] = {
            "eval_iqm_curve": aggregate_curves_across_seeds(eval_curves, args.bootstrap, args.alpha, seed=10, statistic=args.statistic),
            "forgetting_curve": aggregate_curves_across_seeds(forget_curves, args.bootstrap, args.alpha, seed=11, statistic=args.statistic),
            "dormant_frac_curve": aggregate_curves_across_seeds(dorm_curves, args.bootstrap, args.alpha, seed=12, statistic=args.statistic),
            "effective_rank_avg_curve": aggregate_curves_across_seeds(er_curves, args.bootstrap, args.alpha, seed=14, statistic=args.statistic),
        }

    df = pd.DataFrame(summary_rows)

    csv_path = os.path.join(args.out_dir, "summary_table.csv")
    json_path = os.path.join(args.out_dir, "summary_table.json")
    curves_path = os.path.join(args.out_dir, "curves_mean_ci.json")

    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)
    with open(curves_path, "w", encoding="utf-8") as f:
        json.dump(curves_out, f, indent=2)

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {curves_path}")

    if args.make_plots:
        plots_root = args.plots_dir or os.path.join(args.out_dir, "plots")
        out_dir = os.path.join(plots_root, "combined")
        os.makedirs(out_dir, exist_ok=True)

        for method_key, payload in sorted(curves_out.items()):
            method_out = os.path.join(out_dir, method_key)
            os.makedirs(method_out, exist_ok=True)

            plots = [
                ("eval_iqm_curve", "Eval IQM Return", "Env steps", "Return", "eval_iqm.png"),
                ("forgetting_curve", "Isolated Forgetting", "Env steps", "Forgetting", "forgetting.png"),
                ("effective_rank_avg_curve", "Effective Rank", "Env steps", "Rank", "effective_rank.png"),
                ("dormant_frac_curve", "Dormant Fraction", "Optimizer steps", "Dormant frac", "dormant_frac.png"),
            ]

            for key, title, xlabel, ylabel, fname in plots:
                curve = payload.get(key, None)
                out_path = os.path.join(method_out, fname)
                plot_one_method_one_metric(method_key, curve, title, xlabel, ylabel, out_path)


if __name__ == "__main__":
    main()
