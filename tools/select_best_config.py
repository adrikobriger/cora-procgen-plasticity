#!/usr/bin/env python3
"""
Select best configuration per intervention based on composite score.
Score = train/eval_reward_iqm_score - lambda * forgetting
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _safe_load_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _is_number(value) -> bool:
    try:
        return value is not None and not (isinstance(value, float) and math.isnan(value))
    except Exception:
        return False


def _compute_reward_score(summary: dict, prefer_eval: bool = True) -> Optional[float]:
    if not isinstance(summary, dict):
        return None
    eval_fallback = summary.get("eval_fallback_used", False)
    eval_score = summary.get("final_iqm_eval", None)
    train_score = summary.get("final_iqm_train", None)

    if prefer_eval and not eval_fallback and _is_number(eval_score):
        return float(eval_score)
    if _is_number(train_score):
        return float(train_score)
    if _is_number(eval_score):
        return float(eval_score)
    return None


def _compute_forgetting(summary: dict) -> float:
    if not isinstance(summary, dict):
        return 0.0
    forgetting = summary.get("forgetting_iqm", None)
    if _is_number(forgetting):
        return float(forgetting)
    return 0.0


def compute_composite_score(summary: dict, lambda_forgetting: float, prefer_eval: bool = True) -> Optional[float]:
    reward_score = _compute_reward_score(summary, prefer_eval=prefer_eval)
    if reward_score is None:
        return None
    forgetting = _compute_forgetting(summary)
    return float(reward_score) - float(lambda_forgetting) * float(forgetting)


def _parse_method_from_path(path: Path) -> Optional[str]:
    parts = [p.lower() for p in path.parts]
    mapping = {
        "dense": "Dense PPO",
        "partial_reinit": "Partial Reinit",
        "partial": "Partial Reinit",
        "reset": "Reset",
        "redo": "ReDo",
        "gmp": "GMP",
        "set": "SET",
    }
    for token, name in mapping.items():
        if token in parts:
            return name
    return None


def collect_trial_seed_summaries(runs_dir: Path) -> Dict[str, Dict[str, List[dict]]]:
    """
    Return mapping: method -> trial_dir -> list of seed_summary dicts.
    """
    results: Dict[str, Dict[str, List[dict]]] = {}
    for candidate_params in runs_dir.rglob("candidate_params.json"):
        trial_dir = candidate_params.parent
        method = _parse_method_from_path(trial_dir)
        if method is None:
            continue
        summaries: List[dict] = []
        for seed_dir in trial_dir.iterdir():
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue
            summary_path = seed_dir / "seed_summary.json"
            summary = _safe_load_json(summary_path)
            if summary is None:
                continue
            summaries.append(summary)
        if summaries:
            results.setdefault(method, {})[str(trial_dir)] = summaries
    return results


def select_best_configs(
    runs_dir: Path,
    lambda_forgetting: float = 0.5,
    prefer_eval: bool = True,
) -> Dict[str, dict]:
    """
    Select the best configuration per method. Returns mapping method -> payload.
    """
    trial_summaries = collect_trial_seed_summaries(runs_dir)
    best: Dict[str, dict] = {}

    for method, trials in trial_summaries.items():
        best_trial = None
        best_score = None
        best_payload = None
        for trial_dir, summaries in trials.items():
            scores = []
            for s in summaries:
                score = compute_composite_score(s, lambda_forgetting=lambda_forgetting, prefer_eval=prefer_eval)
                if score is not None:
                    scores.append(score)
            if not scores:
                continue
            avg_score = float(sum(scores)) / float(len(scores))
            if best_score is None or avg_score > best_score:
                best_score = avg_score
                best_trial = trial_dir
                params_path = Path(trial_dir) / "candidate_params.json"
                params = _safe_load_json(params_path) or {}
                best_payload = {
                    "method": method,
                    "trial_dir": trial_dir,
                    "avg_score": avg_score,
                    "lambda_forgetting": float(lambda_forgetting),
                    "prefer_eval": bool(prefer_eval),
                    "scores": scores,
                    "params": params.get("params", params),
                }
        if best_trial and best_payload:
            best[method] = best_payload
    return best


def write_best_configs(best_configs: Dict[str, dict], results_dir: Path) -> None:
    out_dir = results_dir / "best_configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    for method, payload in best_configs.items():
        out_path = out_dir / f"{method.lower().replace(' ', '_')}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Select best config per intervention")
    parser.add_argument("--runs_dir", type=str, default="runs", help="Runs directory to scan")
    parser.add_argument("--results_dir", type=str, default="results", help="Output directory for best configs")
    parser.add_argument("--lambda_forgetting", type=float, default=0.5)
    parser.add_argument("--prefer_eval", action="store_true", default=False)
    args = parser.parse_args()

    best = select_best_configs(Path(args.runs_dir), args.lambda_forgetting, prefer_eval=args.prefer_eval)
    write_best_configs(best, Path(args.results_dir))
    print(f"Saved {len(best)} best config(s) to {Path(args.results_dir) / 'best_configs'}")


if __name__ == "__main__":
    main()
