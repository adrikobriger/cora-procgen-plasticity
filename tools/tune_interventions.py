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
# Search space helpers
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


def _search_spaces(method: str):
    method = method.lower()
    if method == "set":
        grid_space = {
            "target_sparsity": [0.5, 0.7, 0.85, 0.95],
            "update_interval": [500, 2000, 5000],
            "prune_fraction": [0.05, 0.1, 0.2, 0.3],
            "warmup_steps": [0, 10_000, 50_000],
        }
        rand_space = {
            "target_sparsity": {"type": "uniform", "low": 0.5, "high": 0.95},
            "update_interval": {"type": "loguniform", "low": 500, "high": 5000, "round_int": True},
            "prune_fraction": {"type": "uniform", "low": 0.05, "high": 0.3},
            "warmup_steps": {"type": "loguniform", "low": 1, "high": 50_000, "round_int": True},
        }
    elif method == "gmp":
        grid_space = {
            "final_sparsity": [0.5, 0.7, 0.85, 0.95],
            "tasks_per_cycle": [3, 6],
            "prune_cycle": [0],
            "global_prune": [True],
            "prune_schedule": ["boundary"],
        }
        rand_space = {
            "final_sparsity": {"type": "uniform", "low": 0.5, "high": 0.95},
            "tasks_per_cycle": {"type": "uniform", "low": 3, "high": 8, "round_int": True},
            "prune_cycle": {"type": "categorical", "values": [0, 1]},
            "global_prune": {"type": "categorical", "values": [True]},
        }
    elif method == "redo":
        grid_space = {
            "tau": [0.05, 0.1, 0.2],
            "ema_beta": [0.9, 0.95, 0.99],
            "update_interval": [2000, 5000, 10000],
            "warmup_steps": [0, 10000, 20000],
            "max_recycle_frac": [0.02, 0.05, 0.1],
            "use_activation_buffer": [False, True],
        }
        rand_space = {
            "tau": {"type": "uniform", "low": 0.05, "high": 0.2},
            "ema_beta": {"type": "uniform", "low": 0.9, "high": 0.999},
            "update_interval": {"type": "loguniform", "low": 1000, "high": 10000, "round_int": True},
            "warmup_steps": {"type": "loguniform", "low": 1, "high": 20_000, "round_int": True},
            "max_recycle_frac": {"type": "uniform", "low": 0.01, "high": 0.1},
            "use_activation_buffer": {"type": "categorical", "values": [False, True]},
        }
    elif method == "reset":
        grid_space = {"scope": ["all"]}
        rand_space = {"scope": {"type": "categorical", "values": ["all"]}}
    elif method in ("partial_reinit", "partial-reinit", "partial"):
        grid_space = {"scope": ["head_only"]}
        rand_space = {"scope": {"type": "categorical", "values": ["head_only"]}}
    elif method == "dense":
        grid_space = {"noop": [True]}
        rand_space = {"noop": {"type": "categorical", "values": [True]}}
    else:
        raise ValueError(f"Unknown method: {method}")
    return grid_space, rand_space


def build_candidates(method: str, search: str, trials: int, seed: int, grid_shuffle: bool) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    grid_space, rand_space = _search_spaces(method)

    if search == "grid":
        candidates = _grid(grid_space, shuffle=grid_shuffle, rng=rng)
        return candidates[:trials]

    samples = _random_sample(rng, rand_space, trials)
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


# -------------------------
# Experiment helpers
# -------------------------

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


def build_experiment(policy_name: str, experiment_name: str, intervention_type: str, params: Dict[str, Any], output_dir: str, num_processes: int, ppo_config: Optional[Dict[str, Any]]):
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
    
    base_cfg = ppo_config.copy() if ppo_config else {}
    base_cfg.update({
        "intervention_type": intervention_type,
        "intervention_params": params,
        "num_processes": num_processes,
    })

    config = policy_struct.config().load_from_dict(base_cfg)
    config.set_output_dir(output_dir)

    policy = policy_struct.policy(config, experiment.observation_space, experiment.action_spaces)
    policy.set_task_ids(experiment.task_ids)

    try:
        cfg_dump = {k: v for k, v in config.__dict__.items() 
                   if not k.startswith("_") and isinstance(v, (int, float, str, bool, type(None)))}
    except Exception:
        cfg_dump = base_cfg
    
    with open(os.path.join(output_dir, "ppo_config_used.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_dump, f, indent=2)

    return experiment, policy


def run_trial(trial_idx: int, args, params: Dict[str, Any], base_dir: str, timestamp: str, ppo_config: Optional[Dict[str, Any]], budget_override: Optional[int], eval_mode: str, episodes_per_task: int, objective_metric: str, base_seed_override: Optional[int] = None):
    trial_dir = os.path.join(base_dir, f"trial_{trial_idx:03d}")
    os.makedirs(trial_dir, exist_ok=True)
    tb_dir = os.path.join(trial_dir, "tb")

    if args.num_processes > 1 and not args.allow_multiprocess:
        raise ValueError("num_processes > 1 requires --allow_multiprocess due to known rollout bug")
    num_proc = args.num_processes if args.num_processes is not None else 1

    experiment, policy = build_experiment(
        policy_name=args.policy,
        experiment_name=args.experiment,
        intervention_type=args.method,
        params=params,
        output_dir=trial_dir,
        num_processes=num_proc,
        ppo_config=ppo_config,
    )

    set_eval_mode(experiment, eval_mode)
    apply_budget_override(experiment, budget_override)

    base_seed = base_seed_override if base_seed_override is not None else (args.seed + trial_idx)
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
        "stage": "confirm" if args._stage == "confirm" else "tune",
        "method": args.method,
        "params": params,
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



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="ppo", type=str)
    parser.add_argument("--experiment", required=True, type=str)
    parser.add_argument("--method", required=True, type=str,
                        choices=["dense", "gmp", "set", "reset", "partial_reinit", "partial-reinit", "partial", "redo"])
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--budget_override", default=None, type=int,
                        help="Override num_timesteps for train tasks (e.g., 10000 for quick sweeps)")
    parser.add_argument("--allow_full_budget", action="store_true")
    parser.add_argument("--search", default="random", choices=["grid", "random"], type=str)
    parser.add_argument("--trials", default=10, type=int)
    parser.add_argument("--grid_shuffle", default=True, type=lambda x: str(x).lower() == "true")
    parser.add_argument("--objective", default="mean", choices=["mean", "iqm"], help="Scalar objective for selection")
    parser.add_argument("--episodes_per_task", default=5, type=int)
    parser.add_argument("--eval_mode", default="final_only", choices=["final_only", "periodic", "none"],
                        help="Control eval cost; final_only disables continual eval")
    parser.add_argument("--output_root", default="runs/tuning", type=str)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--num_processes", default=1, type=int,
                        help="Defaults to 1 to avoid rollout broadcast bug")
    parser.add_argument("--allow_multiprocess", action="store_true")
    parser.add_argument("--ppo_config", default=None, type=str, help="Path to fixed PPO config JSON")
    parser.add_argument("--ppo_params_path", default=None, type=str, help="Path to best_ppo.json from tune_ppo.py")
    parser.add_argument("--save_best_k", default=None, type=int)
    parser.add_argument("--confirm_budget_override", default=None, type=int,
                        help="If set with confirm_top_k, rerun best K at this budget")
    parser.add_argument("--confirm_top_k", default=None, type=int,
                        help="Number of top trials to rerun with confirm budget")
    return parser.parse_args()


def _load_json_or_none(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_ppo_params_from_best(path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Load PPO params from best_ppo.json created by tune_ppo.py"""
    if path is None:
        return None
    data = _load_json_or_none(path)
    if data is None:
        return None
    # Try new format first (wrapped in "best" key), then fallback to old format
    best = data.get("best")
    if best is not None:
        return best.get("ppo_params", {})
    # Fallback for old format without "best" wrapper
    return data.get("ppo_params", {})


def _write_summary_csv(path: str, records: List[Dict[str, Any]]):
    if not records:
        return
    fieldnames = [
        "stage", "trial", "status", "objective", "objective_metric", "seed", "output_dir", "params"
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "stage": r.get("stage"),
                "trial": r.get("trial"),
                "status": r.get("status"),
                "objective": r.get("objective"),
                "objective_metric": r.get("objective_metric"),
                "seed": r.get("seed"),
                "output_dir": r.get("output_dir"),
                "params": json.dumps(r.get("params"), sort_keys=True),
            })


def _write_best_json(path: str, records: List[Dict[str, Any]], k: Optional[int]):
    ok = [r for r in records if r.get("status") == "ok" and not math.isnan(r.get("objective", float("nan")))]
    ok = sorted(ok, key=lambda x: x.get("objective", float("nan")), reverse=True)
    out = {
        "best": ok[0] if ok else None,
        "top_k": ok[:k] if k is not None else None,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def main():
    args = parse_args()
    args._stage = "tune"  # internal marker

    if args.budget_override is None and not args.allow_full_budget:
        raise ValueError("--budget_override is required unless --allow_full_budget is set")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(args.output_root, args.experiment, args.method, timestamp)
    os.makedirs(base_dir, exist_ok=True)

    # Load PPO config: prefer --ppo_params_path (from tune_ppo.py), fallback to --ppo_config
    ppo_config = None
    if args.ppo_params_path is not None:
        ppo_config = _load_ppo_params_from_best(args.ppo_params_path)
        print(f"Loaded PPO params from {args.ppo_params_path}")
    elif args.ppo_config is not None:
        ppo_config = _load_json_or_none(args.ppo_config)
        print(f"Loaded PPO config from {args.ppo_config}")

    candidates = build_candidates(args.method, args.search, args.trials, args.seed, args.grid_shuffle)

    results_path = os.path.join(base_dir, "results.jsonl")

    if args.dry_run:
        print("[DRY RUN] Generated commands:")
        for i, p in enumerate(candidates):
            cmd = (
                f"python main.py --policy {args.policy} --experiment {args.experiment} "
                f"--intervention_type {args.method} --intervention_params '{json.dumps(p)}'"
            )
            if args.budget_override is not None:
                cmd += f" # budget_override={args.budget_override} (handled inside tuner)"
            print(f"trial {i:03d}: {cmd}")
        return

    try:
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    all_results: List[Dict[str, Any]] = []
    for idx, params in enumerate(candidates):
        try:
            res = run_trial(
                trial_idx=idx,
                args=args,
                params=params,
                base_dir=base_dir,
                timestamp=timestamp,
                ppo_config=ppo_config,
                budget_override=args.budget_override,
                eval_mode=args.eval_mode,
                episodes_per_task=args.episodes_per_task,
                objective_metric=args.objective,
            )
            all_results.append(res)
        except Exception as e:
            err_msg = f"trial {idx} failed: {e}"
            traceback.print_exc()
            fail = {
                "trial": idx,
                "timestamp": timestamp,
                "stage": args._stage,
                "method": args.method,
                "params": params,
                "seed": args.seed + idx,
                "status": "error",
                "error": err_msg,
                "objective": float("nan"),
                "objective_metric": args.objective,
            }
            all_results.append(fail)
        finally:
            with open(results_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(all_results[-1]) + "\n")

    confirm_results: List[Dict[str, Any]] = []
    if args.confirm_top_k is not None and args.confirm_budget_override is not None:
        args._stage = "confirm"
        ok_sorted = [r for r in all_results if r.get("status") == "ok"]
        ok_sorted = sorted(ok_sorted, key=lambda x: x.get("objective", float("nan")), reverse=True)
        top = ok_sorted[: args.confirm_top_k]
        confirm_dir = os.path.join(base_dir, "confirm")
        os.makedirs(confirm_dir, exist_ok=True)
        confirm_path = os.path.join(confirm_dir, "confirm_results.jsonl")

        for c_idx, r in enumerate(top):
            try:
                res = run_trial(
                    trial_idx=c_idx,
                    args=args,
                    params=r["params"],
                    base_dir=confirm_dir,
                    timestamp=timestamp,
                    ppo_config=ppo_config,
                    budget_override=args.confirm_budget_override,
                    eval_mode=args.eval_mode,
                    episodes_per_task=args.episodes_per_task,
                    objective_metric=args.objective,
                    base_seed_override=r.get("seed"),
                )
                confirm_results.append(res)
            except Exception as e:
                err_msg = f"confirm trial {c_idx} failed: {e}"
                traceback.print_exc()
                fail = {
                    "trial": c_idx,
                    "timestamp": timestamp,
                    "stage": args._stage,
                    "method": args.method,
                    "params": r["params"],
                    "seed": r.get("seed"),
                    "status": "error",
                    "error": err_msg,
                    "objective": float("nan"),
                    "objective_metric": args.objective,
                }
                confirm_results.append(fail)
            finally:
                with open(confirm_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(confirm_results[-1]) + "\n")

    combined = all_results + confirm_results
    summary_csv = os.path.join(base_dir, "summary.csv")
    _write_summary_csv(summary_csv, combined)
    best_json = os.path.join(base_dir, "best.json")
    _write_best_json(best_json, combined if confirm_results else all_results, args.save_best_k)

    print(f"Completed {len(all_results)} trials. Results -> {results_path}")
    if confirm_results:
        print(f"Confirm trials -> {os.path.join(base_dir, 'confirm', 'confirm_results.jsonl')}")


if __name__ == "__main__":
    main()
