"""
PPO baseline hyperparameter tuning.

This script tunes PPO hyperparameters (learning rate, clip range, entropy coefficient, etc.)
using the dense baseline (no intervention) to find optimal PPO settings that will be frozen
when tuning intervention-specific hyperparameters.

Two-stage tuning protocol:
1. Run this script to find best PPO hyperparameters on dense baseline
2. Use best_ppo.json with tune_interventions.py to tune intervention params fairly
"""

import argparse
import csv
import datetime
import itertools
import json
import math
import os
import random
import traceback
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from continual_rl.available_policies import get_available_policies
from continual_rl.experiment_specs import get_available_experiments


# -------------------------
# PPO search space helpers
# -------------------------

def _log_uniform(rng: random.Random, low: float, high: float) -> float:
    lo = math.log(low)
    hi = math.log(high)
    return math.exp(rng.uniform(lo, hi))


def _grid(options: Dict[str, List[Any]], shuffle: bool, rng: random.Random) -> List[Dict[str, Any]]:
    keys = list(options.keys())
    combos = list(itertools.product(*[options[k] for k in keys]))
    if shuffle:
        rng.shuffle(combos)
    return [{k: vals[i] for i, k in enumerate(keys)} for vals in combos]


def _sample_value(spec: Dict[str, Any], rng: random.Random) -> Any:
    stype = spec.get("type", "uniform")
    if stype == "uniform":
        val = rng.uniform(spec["low"], spec["high"])
    elif stype == "loguniform":
        val = _log_uniform(rng, spec["low"], spec["high"])
    elif stype == "categorical":
        val = rng.choice(spec["values"])
    else:
        raise ValueError(f"Unknown sampler type: {stype}")

    if spec.get("round_int", False):
        val = int(round(val))
    if spec.get("as_int", False):
        val = int(val)
    if spec.get("as_bool", False):
        val = bool(val)
    return val


def _random_sample(rng: random.Random, spec: Dict[str, Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for _ in range(n):
        sample = {k: _sample_value(v, rng) for k, v in spec.items()}
        samples.append(sample)
    return samples


def _ppo_search_spaces():
    """
    Define PPO hyperparameter search spaces for grid and random search.
    Based on typical PPO tuning ranges and repo defaults.
    """
    grid_space = {
        "learning_rate": [7e-4, 3e-4, 1e-4, 5e-5],
        "clip_param": [0.1, 0.2, 0.3],
        "entropy_coef": [0.0, 0.01, 0.02],
        "value_loss_coef": [0.5, 1.0],
        "gamma": [0.99, 0.995],
        "gae_lambda": [0.95, 0.98],
        "num_steps": [128, 256],
        "num_mini_batch": [4, 8, 16],
        "ppo_epoch": [3, 4, 5],
        "max_grad_norm": [0.5, 1.0],
    }

    rand_space = {
        "learning_rate": {"type": "loguniform", "low": 1e-5, "high": 1e-3},
        "clip_param": {"type": "uniform", "low": 0.1, "high": 0.3},
        "entropy_coef": {"type": "loguniform", "low": 1e-4, "high": 0.05},
        "value_loss_coef": {"type": "uniform", "low": 0.25, "high": 2.0},
        "gamma": {"type": "uniform", "low": 0.98, "high": 0.999},
        "gae_lambda": {"type": "uniform", "low": 0.9, "high": 0.99},
        "num_steps": {"type": "categorical", "values": [64, 128, 256, 512]},
        "num_mini_batch": {"type": "categorical", "values": [4, 8, 16, 32]},
        "ppo_epoch": {"type": "categorical", "values": [3, 4, 5, 10]},
        "max_grad_norm": {"type": "uniform", "low": 0.3, "high": 2.0},
    }

    return grid_space, rand_space


def _is_candidate_valid(params: Dict[str, Any]) -> bool:
    """Check if PPO hyperparameter combo is valid."""
    num_steps = int(params.get("num_steps", 128))
    num_mini_batch = int(params.get("num_mini_batch", 32))
    num_processes = int(params.get("num_processes", 1))
    
    if (num_steps * num_processes) % num_mini_batch != 0:
        return False
    if num_mini_batch > num_steps:
        return False
    return True


def build_ppo_candidates(search: str, trials: int, seed: int, grid_shuffle: bool) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    grid_space, rand_space = _ppo_search_spaces()

    if search == "grid":
        candidates = _grid(grid_space, shuffle=grid_shuffle, rng=rng)
        candidates = [c for c in candidates if _is_candidate_valid(c)]
        return candidates[:trials]

    samples = []
    attempts = 0
    max_attempts = trials * 10
    while len(samples) < trials and attempts < max_attempts:
        sample = {k: _sample_value(v, rng) for k, v in rand_space.items()}
        if _is_candidate_valid(sample):
            samples.append(sample)
        attempts += 1
    
    if len(samples) < trials:
        print(f"[WARN] Only generated {len(samples)} valid candidates (requested {trials})")
    
    return samples


def _iqm(xs: List[float]) -> float:
    if len(xs) == 0:
        return float("nan")
    arr = np.asarray(xs, dtype=np.float64)
    arr = np.sort(arr)
    n = len(arr)
    lo = int(np.floor(0.25 * n))
    hi = int(np.ceil(0.75 * n))
    if hi <= lo:
        return float(arr.mean())
    return float(arr[lo:hi].mean())


def _select_eval_tasks(experiment) -> List[Any]:
    eval_tasks = []
    for task in experiment.tasks:
        task_id = getattr(task, "task_id", None)
        task_spec = getattr(task, "_task_spec", None)
        is_eval = False
        if task_spec is not None:
            is_eval = getattr(task_spec, "eval_mode", False)
        if isinstance(task_id, str) and task_id.endswith("_eval"):
            is_eval = True or is_eval
        if is_eval:
            eval_tasks.append(task)
    if eval_tasks:
        return eval_tasks
    return list(experiment.tasks)


def evaluate_tasks(experiment, policy, summary_writer, episodes_per_task: int, objective_metric: str):
    per_task = []
    eval_tasks = _select_eval_tasks(experiment)

    for task in eval_tasks:
        task_id = getattr(task, "task_id", "unknown")
        runner = task.continual_eval(
            run_id=task_id,
            policy=policy,
            summary_writer=summary_writer,
            output_dir=experiment.output_dir,
            timestep_log_offset=0,
        )
        rewards: List[float] = []
        done = False
        while not done:
            try:
                _, info = next(runner)
                if isinstance(info, tuple) and len(info) == 2:
                    reward_list, _metrics = info
                    if reward_list is None:
                        continue
                    if isinstance(reward_list, (list, tuple)):
                        rewards.extend([float(r) for r in reward_list])
                    elif isinstance(reward_list, (int, float)):
                        rewards.append(float(reward_list))
            except StopIteration:
                done = True
            if len(rewards) >= episodes_per_task:
                done = True
        if len(rewards) > episodes_per_task:
            rewards = rewards[:episodes_per_task]
        per_task.append(
            {
                "task_id": task_id,
                "mean": float(np.mean(rewards)) if rewards else float("nan"),
                "iqm": _iqm(rewards),
                "count": len(rewards),
                "raw_returns": rewards,
            }
        )

    mean_over_tasks = float(np.nanmean([t["mean"] for t in per_task])) if per_task else float("nan")
    iqm_over_tasks = float(np.nanmean([t["iqm"] for t in per_task])) if per_task else float("nan")
    objective = mean_over_tasks if objective_metric == "mean" else iqm_over_tasks
    return per_task, {"mean_eval_return": mean_over_tasks, "iqm_eval_return": iqm_over_tasks, "objective": objective}



def apply_budget_override(experiment, budget_override: Optional[int]):
    if budget_override is None:
        return
    for task in experiment.tasks:
        task_spec = getattr(task, "_task_spec", None)
        if task_spec is None:
            continue
        if getattr(task_spec, "eval_mode", False):
            continue
        task_spec.num_timesteps = int(budget_override)
        if hasattr(task, "_rolling_return_count"):
            task._rolling_return_count = max(1, min(task._rolling_return_count, 100))


def set_eval_mode(experiment, mode: str):
    if mode == "periodic":
        return
    if mode == "final_only":
        if hasattr(experiment, "_continual_testing_freq"):
            experiment._continual_testing_freq = 10**12
    elif mode == "none":
        if hasattr(experiment, "_continual_testing_freq"):
            experiment._continual_testing_freq = None
    else:
        raise ValueError(f"Unknown eval_mode: {mode}")


# -------------------------
# Trial runner
# -------------------------

def build_experiment(policy_name: str, experiment_name: str, ppo_params: Dict[str, Any], output_dir: str, num_processes: int):
    available_policies = get_available_policies()
    available_experiments = get_available_experiments()

    if policy_name not in available_policies:
        raise ValueError(f"Unknown policy {policy_name}")
    if experiment_name not in available_experiments:
        raise ValueError(f"Unknown experiment {experiment_name}")

    experiment_loader = available_experiments[experiment_name]
    experiment = experiment_loader()
    experiment.set_output_dir(output_dir)

    policy_struct = available_policies[policy_name]
    
    config_dict = ppo_params.copy()
    config_dict.update({
        "intervention_type": "dense",
        "intervention_params": {},
        "num_processes": num_processes,
    })

    config = policy_struct.config().load_from_dict(config_dict)
    config.set_output_dir(output_dir)

    policy = policy_struct.policy(config, experiment.observation_space, experiment.action_spaces)
    policy.set_task_ids(experiment.task_ids)

    try:
        ppo_config_dump = {k: v for k, v in config.__dict__.items() 
                          if not k.startswith("_") and isinstance(v, (int, float, str, bool, type(None)))}
    except Exception:
        ppo_config_dump = config_dict
    
    with open(os.path.join(output_dir, "ppo_config_used.json"), "w", encoding="utf-8") as f:
        json.dump(ppo_config_dump, f, indent=2)

    return experiment, policy


def run_trial(trial_idx: int, args, ppo_params: Dict[str, Any], base_dir: str, timestamp: str, budget_override: Optional[int], eval_mode: str, episodes_per_task: int, objective_metric: str):
    trial_dir = os.path.join(base_dir, f"trial_{trial_idx:03d}")
    os.makedirs(trial_dir, exist_ok=True)
    tb_dir = os.path.join(trial_dir, "tb")

    if args.num_processes > 1 and not args.allow_multiprocess:
        raise ValueError("num_processes > 1 requires --allow_multiprocess due to known rollout bug")
    num_proc = args.num_processes if args.num_processes is not None else 1

    experiment, policy = build_experiment(
        policy_name=args.policy,
        experiment_name=args.experiment,
        ppo_params=ppo_params,
        output_dir=trial_dir,
        num_processes=num_proc,
    )

    set_eval_mode(experiment, eval_mode)
    apply_budget_override(experiment, budget_override)

    base_seed = args.seed + trial_idx
    np.random.seed(base_seed)
    random.seed(base_seed)
    torch.manual_seed(base_seed)

    writer = SummaryWriter(log_dir=tb_dir)
    experiment.try_run(policy, summary_writer=writer)

    aggregates = {"objective": float("nan")}
    per_task = []
    if eval_mode != "none":
        per_task, aggregates = evaluate_tasks(
            experiment=experiment,
            policy=policy,
            summary_writer=writer,
            episodes_per_task=episodes_per_task,
            objective_metric=objective_metric,
        )

    eff_ranks = None
    if hasattr(experiment, "get_current_effective_ranks"):
        eff_ranks = experiment.get_current_effective_ranks()
        try:
            experiment.save_effective_rank_history()
        except Exception:
            pass

    objective_val = aggregates.get("objective", float("nan"))
    result = {
        "trial": trial_idx,
        "timestamp": timestamp,
        "ppo_params": ppo_params,
        "seed": base_seed,
        "objective": objective_val,
        "objective_metric": objective_metric,
        "aggregates": aggregates,
        "per_task": per_task,
        "plasticity": {
            "effective_rank_by_layer": eff_ranks,
        },
        "output_dir": trial_dir,
        "tb_dir": tb_dir,
        "status": "ok",
    }
    writer.flush()
    writer.close()
    return result


# -------------------------
# CLI
# -------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune PPO baseline hyperparameters (learning rate, clip range, entropy coef, etc.) using dense baseline"
    )
    parser.add_argument("--policy", default="ppo", type=str)
    parser.add_argument("--experiment", required=True, type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--budget_override", default=100000, type=int,
                        help="Override num_timesteps for train tasks (default 100k for fast tuning)")
    parser.add_argument("--allow_full_budget", action="store_true",
                        help="Allow using full experiment budget (ignores budget_override)")
    parser.add_argument("--search", default="random", choices=["grid", "random"], type=str)
    parser.add_argument("--trials", default=20, type=int)
    parser.add_argument("--grid_shuffle", default=True, type=lambda x: str(x).lower() == "true")
    parser.add_argument("--objective", default="mean", choices=["mean", "iqm"], help="Scalar objective for selection")
    parser.add_argument("--episodes_per_task", default=5, type=int)
    parser.add_argument("--eval_mode", default="final_only", choices=["final_only", "periodic", "none"],
                        help="Control eval cost; final_only disables continual eval")
    parser.add_argument("--output_root", default="runs/tuning_ppo", type=str)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--num_processes", default=1, type=int,
                        help="Defaults to 1 to avoid rollout broadcast bug")
    parser.add_argument("--allow_multiprocess", action="store_true",
                        help="Allow num_processes > 1 (not recommended due to broadcast bug)")
    return parser.parse_args()


def _write_leaderboard_csv(path: str, records: List[Dict[str, Any]]):
    if not records:
        return
    fieldnames = [
        "trial", "status", "objective", "mean_eval_return", "iqm_eval_return", 
        "seed", "output_dir", "ppo_params"
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            agg = r.get("aggregates", {})
            writer.writerow({
                "trial": r.get("trial"),
                "status": r.get("status"),
                "objective": r.get("objective"),
                "mean_eval_return": agg.get("mean_eval_return"),
                "iqm_eval_return": agg.get("iqm_eval_return"),
                "seed": r.get("seed"),
                "output_dir": r.get("output_dir"),
                "ppo_params": json.dumps(r.get("ppo_params"), sort_keys=True),
            })


def _write_best_ppo_json(path: str, records: List[Dict[str, Any]]):
    ok = [r for r in records if r.get("status") == "ok" and not math.isnan(r.get("objective", float("nan")))]
    if not ok:
        print("[WARN] No successful trials to select best from")
        return
    
    ok = sorted(ok, key=lambda x: (
        x.get("objective", float("-inf")),
        x.get("aggregates", {}).get("iqm_eval_return", float("-inf"))
    ), reverse=True)
    
    best = ok[0]
    out = {
        "best": {
            "ppo_params": best.get("ppo_params"),
            "trial": best.get("trial"),
            "objective": best.get("objective"),
            "aggregates": best.get("aggregates"),
            "seed": best.get("seed"),
            "output_dir": best.get("output_dir"),
        }
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[BEST] Trial {best['trial']}: objective={best['objective']:.4f}, params={json.dumps(best['ppo_params'], indent=2)}")


def main():
    args = parse_args()

    if args.allow_full_budget:
        budget_override = None
        print("[WARN] Using full experiment budget (may be very slow for tuning)")
    else:
        budget_override = args.budget_override
        print(f"[INFO] Using budget override: {budget_override} steps per task")

    if args.num_processes > 1:
        if not args.allow_multiprocess:
            print("[ERROR] num_processes > 1 requires --allow_multiprocess flag due to rollout broadcast bug")
            return
        else:
            print(f"[WARN] Running with num_processes={args.num_processes}; this may trigger rollout broadcast bug")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(args.output_root, args.experiment, timestamp)
    os.makedirs(base_dir, exist_ok=True)

    candidates = build_ppo_candidates(args.search, args.trials, args.seed, args.grid_shuffle)

    results_path = os.path.join(base_dir, "results.jsonl")

    if args.dry_run:
        print("[DRY RUN] Generated PPO hyperparameter candidates:")
        for i, p in enumerate(candidates):
            print(f"trial {i:03d}: {json.dumps(p, indent=2)}")
        return

    try:
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    all_results: List[Dict[str, Any]] = []
    for idx, ppo_params in enumerate(candidates):
        print(f"\n[TRIAL {idx}/{len(candidates)}] Starting with params: {json.dumps(ppo_params, indent=2)}")
        try:
            res = run_trial(
                trial_idx=idx,
                args=args,
                ppo_params=ppo_params,
                base_dir=base_dir,
                timestamp=timestamp,
                budget_override=budget_override,
                eval_mode=args.eval_mode,
                episodes_per_task=args.episodes_per_task,
                objective_metric=args.objective,
            )
            all_results.append(res)
            print(f"[TRIAL {idx}] Completed: objective={res['objective']:.4f}")
        except Exception as e:
            err_msg = f"trial {idx} failed: {e}"
            traceback.print_exc()
            fail = {
                "trial": idx,
                "timestamp": timestamp,
                "ppo_params": ppo_params,
                "seed": args.seed + idx,
                "status": "error",
                "error": err_msg,
                "objective": float("nan"),
                "objective_metric": args.objective,
            }
            all_results.append(fail)
            print(f"[TRIAL {idx}] Failed: {err_msg}")
        finally:
            with open(results_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(all_results[-1]) + "\n")

    leaderboard_csv = os.path.join(base_dir, "leaderboard.csv")
    _write_leaderboard_csv(leaderboard_csv, all_results)
    
    best_ppo_json_timestamped = os.path.join(base_dir, "best_ppo.json")
    _write_best_ppo_json(best_ppo_json_timestamped, all_results)
    
    best_ppo_json_root = os.path.join(args.output_root, args.experiment, "best_ppo.json")
    os.makedirs(os.path.dirname(best_ppo_json_root), exist_ok=True)
    _write_best_ppo_json(best_ppo_json_root, all_results)

    print(f"\n[DONE] Completed {len(all_results)} trials")
    print(f"  Results: {results_path}")
    print(f"  Leaderboard: {leaderboard_csv}")
    print(f"  Best PPO config (easy reference): {best_ppo_json_root}")
    print(f"\nNext step: Use best_ppo.json with tune_interventions.py:")
    print(f"  python tools/tune_interventions.py --experiment {args.experiment} --method set \\")
    print(f"    --ppo_params_path {best_ppo_json_root} --budget_override 100000 --trials 20")


if __name__ == "__main__":
    main()
