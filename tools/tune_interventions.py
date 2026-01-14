#!/usr/bin/env python3
"""
PURPOSE:
    Tune Intervention hyperparameters (SET, GMP, etc.) using fixed PPO hyperparameters.

SNAPSHOTS & FORGETTING:
    - After each training task completes, we run a lightweight eval snapshot on eval tasks (or all tasks if none
      marked eval) and store aggregate mean/IQM plus per-task means/IQM. These snapshots enable computing
      continual-learning metrics like forgetting.
    - Forgetting (per spec): for each train task, take the best snapshot mean after that task was learned minus the
      final eval mean; average across train tasks (and likewise for IQM). Lower is better.

OBJECTIVES:
    - mean / iqm : maximize final aggregate return (existing behavior).
    - forgetting : minimize average forgetting.
    - composite : maximize (final_mean - lambda_forgetting * forgetting_mean). lambda_forgetting is configurable.
"""

import sys
import os
import argparse
import json
import random
import math
import itertools
import datetime
import traceback
import hashlib
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple

# Adjust path to import continual_rl
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.tensorboard import SummaryWriter

from continual_rl.available_policies import get_available_policies
from continual_rl.experiment_specs import get_available_experiments
from continual_rl.experiments.tasks.task_base import TaskBase
from continual_rl.experiments.tasks.task_spec import TaskSpec


# -------------------------
# Search Space Definitions
# -------------------------

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


# -------------------------
# Sampling Helpers
# -------------------------

def _log_uniform(rng: random.Random, low: float, high: float) -> float:
    lo = math.log(low)
    hi = math.log(high)
    return math.exp(rng.uniform(lo, hi))


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


def _grid(options: Dict[str, List[Any]], shuffle: bool, rng: random.Random) -> List[Dict[str, Any]]:
    keys = list(options.keys())
    combos = list(itertools.product(*[options[k] for k in keys]))
    if shuffle:
        rng.shuffle(combos)
    return [{k: vals[i] for i, k in enumerate(keys)} for vals in combos]


def build_candidates(method: str, search: str, trials: int, seed: int, grid_shuffle: bool) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    grid_space, rand_space = _search_spaces(method)

    if search == "grid":
        combos = _grid(grid_space, grid_shuffle, rng)
        return combos[:trials] if trials is not None else combos
    
    return _random_sample(rng, rand_space, trials)


# -------------------------
# PPO Geometry Checks
# -------------------------

MIN_MINIBATCH_SIZE = 32


def _validate_minibatch_geometry(num_steps: int, num_processes: int, num_mini_batch: int) -> bool:
    """
    Validate PPO minibatch geometry constraint to avoid invalid rollouts.

    PPO requires (num_steps * num_processes) % num_mini_batch == 0 and minibatches should not be tiny.
    """
    batch_size = num_steps * num_processes
    if num_mini_batch <= 0 or num_mini_batch > batch_size:
        return False
    if batch_size % num_mini_batch != 0:
        return False
    minibatch_size = batch_size // num_mini_batch
    return minibatch_size >= MIN_MINIBATCH_SIZE


# -------------------------
# Evaluation Helpers (Matched to tune_ppo.py)
# -------------------------

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
        # Check task object itself first (make_procgen_task sets eval_mode attr)
        if getattr(task, "eval_mode", False):
            eval_tasks.append(task)
        # Also check name based convention
        elif task.task_id.endswith("_eval"):
            eval_tasks.append(task)
    
    # Fallback: if no eval tasks, use all tasks
    if not eval_tasks:
        eval_tasks = experiment.tasks
        
    return list(eval_tasks)


def evaluate_policy_on_tasks(
    experiment,
    policy,
    summary_writer,
    episodes_per_task: int,
    objective_metric: str,
    include_raw_returns: bool = False,
    tasks_override: Optional[List[Any]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    
    per_task = []
    eval_tasks = list(tasks_override) if tasks_override is not None else _select_eval_tasks(experiment)
    if not eval_tasks:
        return [], {"mean_eval_return": float("nan"), "iqm_eval_return": float("nan"), "objective": float("nan")}
    
    for task in eval_tasks:
        returns = []
        # Build a temporary eval TaskSpec to collect a fixed number of episodes in eval mode
        eval_spec = TaskSpec(
            task_id=task.task_id,
            action_space_id=task.action_space_id,
            preprocessor=task._task_spec.preprocessor,
            env_spec=task._task_spec.env_spec,
            num_timesteps=100000,  # large enough to finish requested episodes
            eval_mode=True,
            return_after_episode_num=episodes_per_task,
            with_continual_eval=False,
        )

        # Run the eval loop; collect returns emitted by the runner
        for _, data in task._run(
            eval_spec,
            run_id=f"eval_{task.task_id}",
            policy=policy,
            summary_writer=summary_writer,
            output_dir=experiment.output_dir,
            timestep_log_offset=0,
            wait_to_report=False,
            log_with_task_timestep=False,
            reward_tag="eval_reward",
            task_timestep_start=0,
        ):
            if data is None:
                continue
            returns_batch, _logs = data
            returns.extend(returns_batch)

        mean_ret = float(np.mean(returns)) if returns else float("nan")
        iqm_ret = float(_iqm(returns))
        
        per_task.append({
            "task_id": task.task_id,
            "mean": mean_ret,
            "iqm": iqm_ret,
            "raw_returns": returns if include_raw_returns else None,
        })
        
        print(f"  Eval Task {task.task_id}: Mean={mean_ret:.2f}, IQM={iqm_ret:.2f}")

    # Aggregate across tasks
    mean_over_tasks = float(np.nanmean([t["mean"] for t in per_task])) if per_task else float("nan")
    iqm_over_tasks = float(np.nanmean([t["iqm"] for t in per_task])) if per_task else float("nan")
    
    if objective_metric == "iqm":
        objective = iqm_over_tasks
    else:
        objective = mean_over_tasks
    
    aggregates = {
        "mean_eval_return": mean_over_tasks,
        "iqm_eval_return": iqm_over_tasks,
        "objective": objective,
    }

    return per_task, aggregates


# -------------------------
# Experiment Helpers
# -------------------------

def apply_budget_override(experiment, budget_override: Optional[int]):
    if budget_override is None:
        return
    for task in experiment.tasks:
        # Override only TRAIN tasks (usually not _eval ones)
        if not getattr(task, "eval_mode", False):
            task._num_timesteps = budget_override
            # Ensure task spec reflects override
            if hasattr(task, "_task_spec"):
                try:
                    task._task_spec._num_timesteps = budget_override
                except Exception:
                    pass


def set_eval_mode(experiment, mode: str):
    if mode == "periodic":
        # Keep default continual eval
        pass
    elif mode == "final_only":
        # Disable periodic eval by making freq huge
        experiment._continual_testing_freq = 10**12
    elif mode == "none":
        experiment._continual_testing_freq = 10**12
    # Ensure tasks themselves know we might be in a different mode if needed
    # (Usually handled by experiment runner loop)


def build_experiment_and_policy(
    policy_name: str,
    experiment_name: str,
    intervention_type: str,
    intervention_params: Dict[str, Any],
    ppo_config: Dict[str, Any],
    output_dir: str,
    num_processes: int,
) -> Tuple[Any, Any]:
    
    available_policies = get_available_policies()
    available_experiments = get_available_experiments()

    if policy_name not in available_policies:
        raise ValueError(f"Policy {policy_name} not found.")
    if experiment_name not in available_experiments:
        raise ValueError(f"Experiment {experiment_name} not found.")

    # Load experiment
    experiment = available_experiments[experiment_name]
    experiment.set_output_dir(output_dir)

    # Load policy struct
    policy_struct = available_policies[policy_name]
    
    # Create Policy Config
    # Start with default config from struct
    config = policy_struct.config()
    
    # Apply loaded PPO configs (e.g. from best_ppo.json)
    if ppo_config:
        config.load_from_dict(ppo_config.copy())
    
    # Apply Intervention configs
    override_dict = {
        "intervention_type": intervention_type,
        "intervention_params": intervention_params,
        "num_processes": num_processes,
    }
    config.load_from_dict(override_dict)
    
    config.set_output_dir(output_dir)

    # Initialize policy
    policy = policy_struct.policy(config, experiment.observation_space, experiment.action_spaces)
    policy.set_task_ids(experiment.task_ids)

    # Save effective config used
    try:
        with open(os.path.join(output_dir, "policy_config_used.json"), "w", encoding="utf-8") as f:
            json.dump(config.__dict__, f, indent=2, default=str)
    except Exception:
        pass

    return experiment, policy


# -------------------------
# Main Execution / Reporting
# -------------------------

def write_results_jsonl(path: str, result: Dict[str, Any]):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")


def write_leaderboard_csv(path: str, results: List[Dict[str, Any]]):
    ok_results = [r for r in results if r.get("status") == "ok" and not math.isnan(r.get("objective", float("nan")))]
    # Determine sorting direction based on objective type
    objective_type = None
    if ok_results:
        objective_type = ok_results[0].get("objective_type", "mean")
    if objective_type == "forgetting":
        ok_results.sort(key=lambda x: x.get("objective", float("inf")))  # minimize forgetting
    else:
        ok_results.sort(key=lambda x: x.get("objective", -1e9), reverse=True)
    
    if not ok_results:
        return

    fieldnames = [
        "rank", "trial", "objective", "objective_type", "method",
        "intervention_params", "output_dir",
        "final_mean_eval_return", "final_iqm_eval_return",
        "forgetting_mean", "forgetting_iqm", "composite"
    ]
    
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, r in enumerate(ok_results, 1):
            row = {
                "rank": rank,
                "trial": r["trial"],
                "objective": r.get("objective"),
                "objective_type": r.get("objective_type"),
                "method": r.get("method"),
                "intervention_params": json.dumps(r.get("params", {})),
                "output_dir": r.get("output_dir"),
                "final_mean_eval_return": r.get("final_mean"),
                "final_iqm_eval_return": r.get("final_iqm"),
                "forgetting_mean": r.get("forgetting_mean"),
                "forgetting_iqm": r.get("forgetting_iqm"),
                "composite": r.get("composite"),
            }
            writer.writerow(row)


def write_best_json(path: str, results: List[Dict[str, Any]], k: Optional[int] = 1):
    ok_results = [r for r in results if r.get("status") == "ok" and not math.isnan(r.get("objective", float("nan")))]
    if ok_results:
        objective_type = ok_results[0].get("objective_type", "mean")
        if objective_type == "forgetting":
            ok_results.sort(key=lambda x: x.get("objective", float("inf")))
        else:
            ok_results.sort(key=lambda x: x.get("objective", -1e9), reverse=True)
    
    out_data = {
        "best": ok_results[0] if ok_results else None,
        "top_k": ok_results[:k] if k is not None and ok_results else None
    }
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)


def run_trial(
    trial_idx: int,
    args,
    params: Dict[str, Any],
    base_dir: str,
    timestamp: str,
    ppo_config: Dict[str, Any]
) -> Dict[str, Any]:

    # Clean up task IDs from previous trials to avoid collision
    TaskBase.ALL_TASK_IDS.clear()
    
    trial_dir = os.path.join(base_dir, f"trial_{trial_idx:03d}")
    os.makedirs(trial_dir, exist_ok=True)
    tb_dir = os.path.join(trial_dir, "tb")

    # Set seed
    trial_seed = args.seed + trial_idx
    np.random.seed(trial_seed)
    random.seed(trial_seed)
    torch.manual_seed(trial_seed)

    print(f"\n=== Running Trial {trial_idx} ===")
    print(f"Params: {params}")

    try:
        # Validate PPO minibatch geometry when provided
        if ppo_config:
            num_steps = ppo_config.get("num_steps")
            num_mini_batch = ppo_config.get("num_mini_batch")
            if num_steps is not None and num_mini_batch is not None:
                if not _validate_minibatch_geometry(num_steps, args.num_processes, num_mini_batch):
                    batch_size = num_steps * args.num_processes
                    minibatch_size = batch_size // num_mini_batch if num_mini_batch else 0
                    raise ValueError(
                        "Invalid PPO minibatch geometry: "
                        f"num_steps={num_steps}, num_processes={args.num_processes}, num_mini_batch={num_mini_batch}. "
                        f"Require divisible batches and minibatch_size >= {MIN_MINIBATCH_SIZE} (got {minibatch_size})."
                    )
        experiment, policy = build_experiment_and_policy(
            policy_name="ppo",
            experiment_name=args.experiment,
            intervention_type=args.method,
            intervention_params=params,
            ppo_config=ppo_config,
            output_dir=trial_dir,
            num_processes=args.num_processes,
        )
        
        apply_budget_override(experiment, args.budget_override)
        set_eval_mode(experiment, args.eval_mode)

        writer = SummaryWriter(log_dir=tb_dir)

        # ----------------
        # Training loop with snapshots
        # ----------------
        snapshots = []
        total_train_timesteps = 0
        # Identify train tasks (non-eval)
        train_tasks = [t for t in experiment.tasks if not getattr(t, "eval_mode", False) and not t.task_id.endswith("_eval")]
        if not train_tasks:
            print("Warning: No train-task filter found; using all tasks as train tasks.")
            train_tasks = experiment.tasks
        eval_tasks = _select_eval_tasks(experiment)
        if not eval_tasks:
            print("Warning: No explicit eval tasks found; using all tasks for evaluation snapshots.")
            eval_tasks = experiment.tasks

        cycle_count = getattr(experiment, "_cycle_count", 1) or 1

        def run_snapshot(cycle_id, task_run_idx, label):
            snap_per_task, snap_aggs = evaluate_policy_on_tasks(
                experiment,
                policy,
                writer,
                episodes_per_task=min(args.episodes_per_task, args.snapshot_episodes_per_task),
                objective_metric="mean",
                include_raw_returns=args.save_raw_returns,
                tasks_override=train_tasks,
            )
            snapshot = {
                "cycle": cycle_id,
                "task_run_idx": task_run_idx,
                "label": label,
                "timestamp": datetime.datetime.now().isoformat(),
                "aggregate_mean": snap_aggs.get("mean_eval_return"),
                "aggregate_iqm": snap_aggs.get("iqm_eval_return"),
                "per_task": snap_per_task,
            }
            snapshots.append(snapshot)

        # Initial snapshot before training
        run_snapshot(cycle_id=0, task_run_idx=-1, label="pre_train")

        for cycle_id in range(cycle_count):
            for task_run_idx, task in enumerate(train_tasks):
                run_id = f"train_c{cycle_id}_t{task_run_idx}"
                for task_timesteps, _ in task._run(
                    task._task_spec,
                    run_id=run_id,
                    policy=policy,
                    summary_writer=writer,
                    output_dir=experiment.output_dir,
                    timestep_log_offset=total_train_timesteps,
                    wait_to_report=False,
                    log_with_task_timestep=True,
                    reward_tag="train_reward",
                    task_timestep_start=0,
                ):
                    total_train_timesteps = max(total_train_timesteps, task_timesteps)
                # Snapshot after each train task
                run_snapshot(cycle_id=cycle_id, task_run_idx=task_run_idx, label="post_task")

        # Final eval snapshot (full episodes_per_task) if enabled
        per_task, aggregates = [], {"objective": float("nan")}
        per_task_train_final: List[Dict[str, Any]] = []
        # Final train-task eval for forgetting metric
        per_task_train_final, _ = evaluate_policy_on_tasks(
            experiment,
            policy,
            writer,
            episodes_per_task=min(args.episodes_per_task, args.snapshot_episodes_per_task),
            objective_metric="mean",
            include_raw_returns=args.save_raw_returns,
            tasks_override=train_tasks,
        )
        if args.eval_mode != "none":
            print("Evaluating...")
            eval_objective_metric = args.objective if args.objective in ("mean", "iqm") else "mean"
            per_task, aggregates = evaluate_policy_on_tasks(
                experiment,
                policy,
                writer,
                args.episodes_per_task,
                eval_objective_metric,
                include_raw_returns=args.save_raw_returns,
            )
            print(f"Trial {trial_idx} Objective ({args.objective}): {aggregates['objective']:.4f}")

        eff_ranks = None
        if hasattr(experiment, "get_current_effective_ranks"):
            eff_ranks = experiment.get_current_effective_ranks()

        # Compute forgetting
        forgetting_mean = float("nan")
        forgetting_iqm = float("nan")
        if per_task_train_final:
            final_map_mean = {t["task_id"]: t["mean"] for t in per_task_train_final}
            final_map_iqm = {t["task_id"]: t["iqm"] for t in per_task_train_final}
            train_task_ids = [t.task_id for t in train_tasks]
            per_task_forgetting_mean = []
            per_task_forgetting_iqm = []
            for tid in train_task_ids:
                best_mean = None
                best_iqm = None
                for snap in snapshots:
                    for pt in snap.get("per_task", []):
                        if pt.get("task_id") == tid:
                            cand_mean = pt.get("mean")
                            cand_iqm = pt.get("iqm")
                            if cand_mean is not None and not math.isnan(cand_mean):
                                if best_mean is None or cand_mean > best_mean:
                                    best_mean = cand_mean
                            if cand_iqm is not None and not math.isnan(cand_iqm):
                                if best_iqm is None or cand_iqm > best_iqm:
                                    best_iqm = cand_iqm
                if best_mean is not None and tid in final_map_mean and not math.isnan(final_map_mean[tid]):
                    per_task_forgetting_mean.append(best_mean - final_map_mean[tid])
                if best_iqm is not None and tid in final_map_iqm and not math.isnan(final_map_iqm[tid]):
                    per_task_forgetting_iqm.append(best_iqm - final_map_iqm[tid])
            if per_task_forgetting_mean:
                forgetting_mean = float(np.mean(per_task_forgetting_mean))
            if per_task_forgetting_iqm:
                forgetting_iqm = float(np.mean(per_task_forgetting_iqm))

        final_mean = aggregates.get("mean_eval_return") if aggregates else float("nan")
        final_iqm = aggregates.get("iqm_eval_return") if aggregates else float("nan")

        # Composite objective
        objective_type = args.objective
        objective_value = aggregates.get("objective")
        if args.objective == "forgetting":
            objective_value = forgetting_mean
        elif args.objective == "composite":
            # maximize final_mean - lambda * forgetting_mean
            lambda_f = args.lambda_forgetting
            objective_value = (final_mean if not math.isnan(final_mean) else 0.0) - lambda_f * (forgetting_mean if not math.isnan(forgetting_mean) else 0.0)
        
        # Save snapshots
        snapshots_path = os.path.join(trial_dir, "snapshots.json")
        with open(snapshots_path, "w", encoding="utf-8") as f:
            json.dump(snapshots, f, indent=2)

        # Trial summary
        ppo_hash = hashlib.md5(json.dumps(ppo_config, sort_keys=True).encode("utf-8") if ppo_config else b"no_ppo").hexdigest()
        trial_summary = {
            "trial": trial_idx,
            "timestamp": timestamp,
            "method": args.method,
            "params": params,
            "seed": trial_seed,
            "status": "ok",
            "objective_type": objective_type,
            "objective": objective_value,
            "final_mean": final_mean,
            "final_iqm": final_iqm,
            "forgetting_mean": forgetting_mean,
            "forgetting_iqm": forgetting_iqm,
            "composite": objective_value if args.objective == "composite" else None,
            "ppo_config_hash": ppo_hash,
            "snapshots_path": snapshots_path,
            "output_dir": trial_dir,
        }
        with open(os.path.join(trial_dir, "trial_summary.json"), "w", encoding="utf-8") as f:
            json.dump(trial_summary, f, indent=2)

        # Save lean best_config.json
        best_cfg = {
            "method": args.method,
            "params": params,
            "objective_type": objective_type,
            "objective": objective_value,
            "final_mean": final_mean,
            "final_iqm": final_iqm,
            "forgetting_mean": forgetting_mean,
            "forgetting_iqm": forgetting_iqm,
        }
        with open(os.path.join(trial_dir, "best_config.json"), "w", encoding="utf-8") as f:
            json.dump(best_cfg, f, indent=2)

        result = {
            "trial": trial_idx,
            "timestamp": timestamp,
            "method": args.method,
            "params": params,
            "seed": trial_seed,
            "objective": objective_value,
            "objective_type": objective_type,
            "final_mean": final_mean,
            "final_iqm": final_iqm,
            "forgetting_mean": forgetting_mean,
            "forgetting_iqm": forgetting_iqm,
            "aggregates": aggregates,
            "per_task": per_task,
            "per_task_train_final": per_task_train_final,
            "plasticity_metrics": {"effective_rank": eff_ranks} if eff_ranks else None,
            "output_dir": trial_dir,
            "status": "ok",
        }
        
        writer.close()
        return result

    except Exception as e:
        print(f"Trial {trial_idx} failed: {e}")
        traceback.print_exc()
        return {
            "trial": trial_idx,
            "status": "failed",
            "error": str(e),
            "method": args.method,
            "params": params,
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=str)
    parser.add_argument("--method", required=True, type=str,
                        choices=["dense", "gmp", "set", "reset", "partial_reinit", "redo"])
    parser.add_argument("--trials", default=10, type=int)
    parser.add_argument("--search", default="random", choices=["grid", "random"], type=str)
    parser.add_argument("--seed", default=0, type=int)
    
    parser.add_argument("--budget_override", default=None, type=int)
    parser.add_argument("--eval_mode", default="final_only", choices=["final_only", "periodic", "none"])
    parser.add_argument("--episodes_per_task", default=5, type=int)
    parser.add_argument("--snapshot_episodes_per_task", default=2, type=int,
                        help="Episodes per task for lightweight snapshots/forgetting")
    parser.add_argument("--objective", default="mean", choices=["mean", "iqm", "forgetting", "composite"])
    parser.add_argument("--lambda_forgetting", default=1.0, type=float,
                        help="Weight for forgetting in composite objective")
    parser.add_argument("--save_raw_returns", action="store_true", default=False,
                        help="If set, store raw episode returns in outputs")
    
    parser.add_argument("--ppo_config", default=None, type=str, 
                        help="Path to JSON file with fixed PPO hyperparameters (tuned)")
    
    parser.add_argument("--num_processes", default=1, type=int)
    parser.add_argument("--output_root", default="runs/tuning", type=str)
    parser.add_argument("--grid_shuffle", action="store_true", default=True)
    
    # Save top K best to a json
    parser.add_argument("--save_best_k", default=5, type=int)

    return parser.parse_args()


def main():
    args = parse_args()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(args.output_root, args.experiment, args.method, timestamp)
    os.makedirs(base_dir, exist_ok=True)
    
    print(f"Starting tuning for method: {args.method}")
    print(f"Experiment: {args.experiment}")
    print(f"Output Directory: {base_dir}")
    
    # Load PPO Config
    ppo_config = {}
    if args.ppo_config:
        print(f"Loading fixed PPO params from: {args.ppo_config}")
        with open(args.ppo_config, "r") as f:
            ppo_config = json.load(f)
            # Handle if it's wrapped in a list (some configs are lists of dicts)
            if isinstance(ppo_config, list):
                ppo_config = ppo_config[0]
            # Handle if it's wrapped in 'best' (best_ppo.json format)
            if "best" in ppo_config:
                ppo_config = ppo_config["best"].get("ppo_params", ppo_config["best"])

    # Generate Candidates
    candidates = build_candidates(args.method, args.search, args.trials, args.seed, args.grid_shuffle)
    
    results = []
    
    # Run Trials
    for i, params in enumerate(candidates):
        res = run_trial(i, args, params, base_dir, timestamp, ppo_config)
        results.append(res)
        
        # Live updates
        write_results_jsonl(os.path.join(base_dir, "results.jsonl"), res)
        write_leaderboard_csv(os.path.join(base_dir, "leaderboard.csv"), results)
        write_best_json(os.path.join(base_dir, "best_interventions.json"), results, args.save_best_k)

    print(f"Tuning complete. Best results saved to {os.path.join(base_dir, 'best_interventions.json')}")

if __name__ == "__main__":
    main()
