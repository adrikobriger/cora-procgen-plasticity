#!/usr/bin/env python3
"""
PURPOSE:
    Tune intervention hyperparameters (SET, GMP, ReDo, etc.) while keeping PPO hyperparameters fixed.

SNAPSHOTS & FORGETTING (DEFENSIBLE):
    - After each training task completes, we run a lightweight evaluation snapshot on the TRAIN tasks
      (because forgetting is defined on the train tasks).
    - Forgetting (per spec): for each train task, take the best snapshot return AFTER that task was learned
      (i.e., from the first post_task snapshot for that task onward) minus the final return on that task.
      Average across train tasks. Lower is better.

OBJECTIVES:
    - mean / iqm      : maximize final aggregate return (mean or IQM) across eval tasks.
    - forgetting      : minimize average forgetting (mean or IQM, consistent with primary metric).
    - composite       : maximize (final_primary - lambda_forgetting * forgetting_primary)

ROBUSTNESS:
    - Each intervention candidate is evaluated across multiple seeds (trial_seeds) and aggregated as mean ± stderr.
"""

import sys
import os
import argparse
import json
import random
import math
import itertools
import numbers
import datetime
import uuid
import warnings

# Suppress Gym deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym")
warnings.filterwarnings("ignore", category=UserWarning, module="gym")
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
warnings.filterwarnings("ignore", message=".*old step API.*")
warnings.filterwarnings("ignore", message=".*np.bool8.*")
import traceback
import concurrent.futures
import multiprocessing
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
# Small utilities
# -------------------------

def _parse_int_list(s: str) -> List[int]:
    s = (s or "").strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _stderr(xs: List[float]) -> float:
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    if len(xs) <= 1:
        return float("nan")
    arr = np.asarray(xs, dtype=np.float64)
    return float(arr.std(ddof=1) / np.sqrt(len(arr)))


def _mean(xs: List[float]) -> float:
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    if not xs:
        return float("nan")
    return float(np.mean(np.asarray(xs, dtype=np.float64)))


def _is_eval_task(task) -> bool:
    """Explicit eval tasks: flagged eval_mode or id suffixed with _eval (task or spec)."""
    if getattr(task, "eval_mode", False):
        return True
    if getattr(task, "task_id", "").endswith("_eval"):
        return True
    if hasattr(task, "_task_spec") and getattr(task._task_spec, "eval_mode", False):
        return True
    return False


def _is_train_task(task) -> bool:
    """Train tasks are those not marked eval and without _eval suffix."""
    return not _is_eval_task(task)


def _task_timesteps(task: Any) -> int:
    for attr in ("_num_timesteps", "num_timesteps"):
        if hasattr(task, attr):
            try:
                return int(getattr(task, attr))
            except Exception:
                pass
    if hasattr(task, "_task_spec"):
        ts = getattr(task._task_spec, "_num_timesteps", None)
        if ts is None:
            ts = getattr(task._task_spec, "num_timesteps", None)
        try:
            return int(ts) if ts is not None else 0
        except Exception:
            return 0
    return 0


# -------------------------
# Search Space Definitions
# -------------------------

def _search_spaces(method: str, opt_steps_total: Optional[int] = None):
    """Return (grid_space, rand_space) for the given intervention.

    Notes:
      - For SET, update_interval and warmup_steps are measured in *optimizer steps* (minibatch updates),
        not environment timesteps. If opt_steps_total is provided, we scale these ranges to be meaningful
        for the current run budget + PPO update geometry.
    """
    method = method.lower()
    if method == "set":
        if opt_steps_total is not None and int(opt_steps_total) > 0:
            opt_steps_total = int(opt_steps_total)

            # We want SET to actually execute multiple prune+regrow updates within the run.
            # Target roughly 5-50 updates across training.
            def _clamp_int(x: int, lo: int, hi: int) -> int:
                return int(max(lo, min(hi, x)))

            # Update interval candidates now represent "updates per run" (relative schedule)
            update_interval_grid = [10, 20, 40]

            # Warmup candidates now represent fraction of total optimizer steps
            warmup_grid = [0.0, 0.03]

            # Random sampling ranges (relative schedule)
            min_update = 5
            max_update = 120
            max_warmup = 0.20

            rand_space = {
                "target_sparsity": {"type": "uniform", "low": 0.5, "high": 0.95},
                "update_interval": {"type": "loguniform", "low": float(min_update), "high": float(max_update), "round_int": True},
                "prune_fraction": {"type": "uniform", "low": 0.05, "high": 0.3},
                "warmup_steps": {"type": "uniform", "low": 0.0, "high": float(max_warmup)},
            }

            grid_space = {
                "target_sparsity": [0.6],
                "update_interval": update_interval_grid,
                "prune_fraction": [0.1, 0.2],
                "warmup_steps": warmup_grid,
            }
        else:
            # Fallback (should rarely be used): conservative defaults.
            grid_space = {
                "target_sparsity": [0.6, 0.8, 0.9],
                "update_interval": [200, 500, 1000],
                "prune_fraction": [0.05, 0.1, 0.2],
                "warmup_steps": [0, 200, 1000],
            }
            rand_space = {
                "target_sparsity": {"type": "uniform", "low": 0.5, "high": 0.95},
                "update_interval": {"type": "loguniform", "low": 50, "high": 2000, "round_int": True},
                "prune_fraction": {"type": "uniform", "low": 0.05, "high": 0.3},
                "warmup_steps": {"type": "uniform", "low": 0.0, "high": 2000.0, "round_int": True},
            }

    elif method == "gmp":
        grid_space = {
            "final_sparsity": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
        }
        rand_space = {
            "final_sparsity": {"type": "uniform", "low": 0.5, "high": 0.9},
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


def build_candidates(method: str, search: str, trials: int, seed: int, grid_shuffle: bool, opt_steps_total: Optional[int] = None) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    grid_space, rand_space = _search_spaces(method, opt_steps_total=opt_steps_total)

    if search == "grid":
        combos = _grid(grid_space, grid_shuffle, rng)
        if trials is None or int(trials) <= 0:
            return combos
        return combos[:trials]

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


def _estimate_total_optimizer_steps(args, ppo_config: Dict[str, Any]) -> Optional[int]:
    """Estimate total optimizer steps for the TRAIN portion of a run.

    This is used to scale SET hyperparameters that are defined in optimizer steps.

    Assumptions (standard PPO):
      - Each rollout collects (num_steps * num_processes) environment steps.
      - PPO performs `epochs` passes over the rollout, split into `num_mini_batch` minibatches.
      - One optimizer step per minibatch.

    If we cannot estimate reliably, returns None.
    """
    try:
        # Instantiate the experiment (cheap) to count train tasks and cycles.
        exps = get_available_experiments()
        if args.experiment not in exps:
            return None
        exp = exps[args.experiment]
        train_tasks = [t for t in exp.tasks if _is_train_task(t)]
        if not train_tasks:
            train_tasks = list(exp.tasks)
        cycle_count = int(getattr(exp, "_cycle_count", 1) or 1)

        if args.budget_override is not None:
            per_task_steps = int(args.budget_override)
            total_env_steps = per_task_steps * len(train_tasks) * cycle_count
        else:
            total_env_steps = 0
            for t in train_tasks:
                total_env_steps += int(getattr(t, "_num_timesteps", 0) or 0)
            total_env_steps *= cycle_count

        # PPO geometry
        num_steps = int(ppo_config.get("num_steps", ppo_config.get("n_steps", 256)) or 256)
        num_mini_batch = int(ppo_config.get("num_mini_batch", ppo_config.get("num_minibatches", 4)) or 4)

        # Try several common keys for epochs
        epochs = (
            ppo_config.get("ppo_epoch")
            or ppo_config.get("ppo_epochs")
            or ppo_config.get("update_epochs")
            or ppo_config.get("num_epochs")
            or ppo_config.get("epochs")
            or 4
        )
        epochs = int(epochs)

        denom = max(1, num_steps * int(args.num_processes))
        rollouts = int(math.ceil(total_env_steps / float(denom))) if total_env_steps > 0 else 0
        opt_steps_total = rollouts * epochs * num_mini_batch
        if opt_steps_total <= 0:
            return None
        return int(opt_steps_total)
    except Exception as e:
        print(f"WARNING: could not estimate optimizer steps (SET scaling). Falling back to defaults. Error: {e}")
        return None


def _infer_train_tasks_per_cycle(experiment_name: str) -> Optional[int]:
    try:
        exps = get_available_experiments()
        if experiment_name not in exps:
            return None
        exp = exps[experiment_name]
        train_tasks = [t for t in exp.tasks if _is_train_task(t)]
        if not train_tasks:
            train_tasks = list(exp.tasks)
        cycle_count = int(getattr(exp, "_cycle_count", 1) or 1)
        tasks_per_cycle = int(len(train_tasks))
        if cycle_count > 1 and tasks_per_cycle % cycle_count == 0:
            tasks_per_cycle = int(tasks_per_cycle // cycle_count)
        return tasks_per_cycle if tasks_per_cycle > 0 else None
    except Exception:
        return None


# -------------------------
# Evaluation Helpers
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


def _select_eval_tasks(experiment) -> Tuple[List[Any], bool]:
    eval_tasks = [t for t in experiment.tasks if _is_eval_task(t)]
    if eval_tasks:
        return list(eval_tasks), False
    return list(experiment.tasks), True


def _append_episode_returns(dst: List[float], batch, max_n: int) -> None:
    """Append up to max_n numeric episode returns into dst, skipping None/NaN/inf and non-scalars."""
    if batch is None or max_n <= 0:
        return

    # Some implementations may emit a single scalar instead of a list.
    if isinstance(batch, numbers.Real):
        batch_iter = [batch]
    else:
        try:
            batch_iter = list(batch)
        except Exception:
            return

    for r in batch_iter:
        if r is None:
            continue

        # Handle common scalar containers (torch/np) defensively.
        try:
            import torch
            if isinstance(r, torch.Tensor):
                if r.numel() != 1:
                    continue
                r = r.item()
        except Exception:
            pass

        try:
            import numpy as _np
            if isinstance(r, _np.ndarray):
                if r.shape != ():
                    continue
                r = float(r)
        except Exception:
            pass

        if not isinstance(r, numbers.Real):
            continue

        rf = float(r)
        if math.isnan(rf) or math.isinf(rf):
            continue

        dst.append(rf)
        if len(dst) >= max_n:
            return


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
    if tasks_override is not None:
        eval_tasks = list(tasks_override)
    else:
        eval_tasks, _fallback_used = _select_eval_tasks(experiment)
    if not eval_tasks:
        return [], {"mean_eval_return": float("nan"), "iqm_eval_return": float("nan"), "objective": float("nan")}

    for task in eval_tasks:
        returns = []

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
            _append_episode_returns(returns, returns_batch, episodes_per_task)
            if len(returns) >= episodes_per_task:
                break

        mean_ret = float(np.mean(returns)) if returns else float("nan")
        iqm_ret = float(_iqm(returns))

        per_task.append({
            "task_id": task.task_id,
            "mean": mean_ret,
            "iqm": iqm_ret,
            "raw_returns": returns if include_raw_returns else None,
        })

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
        if _is_eval_task(task):
            continue
        task._num_timesteps = budget_override
        if hasattr(task, "_task_spec"):
            try:
                task._task_spec._num_timesteps = budget_override
            except Exception:
                pass


def set_eval_mode(experiment, mode: str):
    if mode == "periodic":
        # Keep default continual eval
        return
    # final_only / none -> effectively disable periodic eval
    experiment._continual_testing_freq = 10**12


def build_experiment_and_policy(
    policy_name: str,
    experiment_name: str,
    intervention_type: str,
    intervention_params: Dict[str, Any],
    ppo_config: Dict[str, Any],
    output_dir: str,
    num_processes: int,
    budget_override: Optional[int] = None,
) -> Tuple[Any, Any]:

    available_policies = get_available_policies()
    available_experiments = get_available_experiments()

    if policy_name not in available_policies:
        raise ValueError(f"Policy {policy_name} not found.")
    if experiment_name not in available_experiments:
        raise ValueError(f"Experiment {experiment_name} not found.")

    experiment = available_experiments[experiment_name]
    experiment.set_output_dir(output_dir)

    policy_struct = available_policies[policy_name]
    config = policy_struct.config()

    if ppo_config:
        config.load_from_dict(ppo_config.copy())

    # Apply budget override before computing total_train_steps (if any)
    apply_budget_override(experiment, budget_override)

    # Compute total_train_steps for global-step interventions (e.g., GMP)
    total_train_steps = None
    try:
        train_tasks = [t for t in experiment.tasks if _is_train_task(t)]
        if not train_tasks:
            train_tasks = list(experiment.tasks)
        cycle_count = int(getattr(experiment, "_cycle_count", 1) or 1)
        total_train_timesteps = sum(_task_timesteps(t) for t in train_tasks) * cycle_count

        if total_train_timesteps <= 0 and budget_override is not None:
            total_train_timesteps = int(budget_override) * len(train_tasks) * cycle_count

        num_steps = int(getattr(config, "num_steps", 256) or 256)
        num_mini_batch = int(getattr(config, "num_mini_batch", 4) or 4)
        ppo_epoch = int(getattr(config, "ppo_epoch", 4) or 4)
        denom = max(1, num_steps * int(num_processes))
        rollouts = int(math.ceil(total_train_timesteps / float(denom))) if total_train_timesteps > 0 else 0
        if rollouts > 0 and num_mini_batch > 0 and ppo_epoch > 0:
            total_train_steps = rollouts * ppo_epoch * num_mini_batch
    except Exception:
        total_train_steps = None

    if total_train_steps is not None:
        intervention_params = dict(intervention_params or {})
        intervention_params["total_train_steps"] = int(total_train_steps)

    override_dict = {
        "intervention_type": intervention_type,
        "intervention_params": intervention_params,
        "num_processes": num_processes,
    }
    config.load_from_dict(override_dict)
    if total_train_steps is not None:
        if getattr(config, "intervention_params", None) is None:
            config.intervention_params = {}
        config.intervention_params["total_train_steps"] = int(total_train_steps)
    config.set_output_dir(output_dir)

    policy = policy_struct.policy(config, experiment.observation_space, experiment.action_spaces)
    policy.set_task_ids(experiment.task_ids)

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
    if not ok_results:
        return

    objective_type = ok_results[0].get("objective_type", "mean")
    if objective_type == "forgetting":
        ok_results.sort(key=lambda x: x.get("objective", float("inf")))
    else:
        ok_results.sort(key=lambda x: x.get("objective", -1e9), reverse=True)

    import csv
    fieldnames = [
        "rank", "trial", "objective", "objective_stderr", "objective_type", "primary_metric", "final_eval_set",
        "method", "intervention_params", "output_dir",
        "final_mean", "final_mean_stderr", "final_iqm", "final_iqm_stderr", "final_primary", "final_primary_stderr",
        "final_mean_train", "final_mean_train_stderr", "final_iqm_train", "final_iqm_train_stderr", "final_primary_train", "final_primary_train_stderr",
        "final_mean_eval", "final_mean_eval_stderr", "final_iqm_eval", "final_iqm_eval_stderr", "final_primary_eval", "final_primary_eval_stderr",
        "forgetting_mean", "forgetting_mean_stderr", "forgetting_iqm", "forgetting_iqm_stderr", "forgetting_primary_mean", "forgetting_primary_stderr",
        "eval_fallback_used", "composite"
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, r in enumerate(ok_results, 1):
            writer.writerow({
                "rank": rank,
                "trial": r.get("trial"),
                "objective": r.get("objective"),
                "objective_stderr": r.get("objective_stderr"),
                "objective_type": r.get("objective_type"),
                "primary_metric": r.get("primary_metric"),
                "final_eval_set": r.get("final_eval_set"),
                "method": r.get("method"),
                "intervention_params": json.dumps(r.get("params", {})),
                "output_dir": r.get("output_dir"),
                "final_mean": r.get("final_mean"),
                "final_mean_stderr": r.get("final_mean_stderr"),
                "final_iqm": r.get("final_iqm"),
                "final_iqm_stderr": r.get("final_iqm_stderr"),
                "final_primary": r.get("final_primary"),
                "final_primary_stderr": r.get("final_primary_stderr"),
                "final_mean_train": r.get("final_mean_train"),
                "final_mean_train_stderr": r.get("final_mean_train_stderr"),
                "final_iqm_train": r.get("final_iqm_train"),
                "final_iqm_train_stderr": r.get("final_iqm_train_stderr"),
                "final_primary_train": r.get("final_primary_train"),
                "final_primary_train_stderr": r.get("final_primary_train_stderr"),
                "final_mean_eval": r.get("final_mean_eval"),
                "final_mean_eval_stderr": r.get("final_mean_eval_stderr"),
                "final_iqm_eval": r.get("final_iqm_eval"),
                "final_iqm_eval_stderr": r.get("final_iqm_eval_stderr"),
                "final_primary_eval": r.get("final_primary_eval"),
                "final_primary_eval_stderr": r.get("final_primary_eval_stderr"),
                "forgetting_mean": r.get("forgetting_mean"),
                "forgetting_mean_stderr": r.get("forgetting_mean_stderr"),
                "forgetting_iqm": r.get("forgetting_iqm"),
                "forgetting_iqm_stderr": r.get("forgetting_iqm_stderr"),
                "forgetting_primary_mean": r.get("forgetting_primary"),
                "forgetting_primary_stderr": r.get("forgetting_primary_stderr"),
                "eval_fallback_used": r.get("eval_fallback_used"),
                "composite": r.get("composite"),
            })


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
        "top_k": ok_results[:k] if (k is not None and ok_results) else None
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, default=str)


def _compute_forgetting_after_learned(
    snapshots: List[Dict[str, Any]],
    train_tasks: List[Any],
    per_task_train_final: List[Dict[str, Any]],
) -> Tuple[float, float]:
    """
    Forgetting (per task): (best snapshot AFTER learned) - (final).
    We define "learned" for task i as the first snapshot labeled post_task for task_run_idx == i in cycle 0.
    """
    final_map_mean = {t["task_id"]: t["mean"] for t in per_task_train_final}
    final_map_iqm = {t["task_id"]: t["iqm"] for t in per_task_train_final}

    train_id_to_index = {t.task_id: i for i, t in enumerate(train_tasks)}

    # Find learn point per task
    learn_snap_idx: Dict[str, int] = {}
    for si, snap in enumerate(snapshots):
        if snap.get("label") != "post_task":
            continue
        if snap.get("cycle") != 0:
            continue
        tri = snap.get("task_run_idx")
        for tid, idx in train_id_to_index.items():
            if idx == tri and tid not in learn_snap_idx:
                learn_snap_idx[tid] = si

    per_task_forgetting_mean = []
    per_task_forgetting_iqm = []

    for tid in [t.task_id for t in train_tasks]:
        if tid not in final_map_mean:
            continue

        start_si = learn_snap_idx.get(tid, None)
        if start_si is None:
            continue

        best_mean = None
        best_iqm = None

        for snap in snapshots[start_si:]:
            for pt in snap.get("per_task", []):
                if pt.get("task_id") != tid:
                    continue
                cand_mean = pt.get("mean")
                cand_iqm = pt.get("iqm")

                if cand_mean is not None and not math.isnan(cand_mean):
                    best_mean = cand_mean if (best_mean is None or cand_mean > best_mean) else best_mean

                if cand_iqm is not None and not math.isnan(cand_iqm):
                    best_iqm = cand_iqm if (best_iqm is None or cand_iqm > best_iqm) else best_iqm

        if best_mean is not None and not math.isnan(final_map_mean.get(tid, float("nan"))):
            per_task_forgetting_mean.append(best_mean - final_map_mean[tid])

        if best_iqm is not None and not math.isnan(final_map_iqm.get(tid, float("nan"))):
            per_task_forgetting_iqm.append(best_iqm - final_map_iqm[tid])

    forgetting_mean = float(np.mean(per_task_forgetting_mean)) if per_task_forgetting_mean else float("nan")
    forgetting_iqm = float(np.mean(per_task_forgetting_iqm)) if per_task_forgetting_iqm else float("nan")
    return forgetting_mean, forgetting_iqm


def _run_single_seed(
    args,
    params: Dict[str, Any],
    trial_dir: str,
    seed: int,
    ppo_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run one candidate intervention with one RNG seed, returning per-seed metrics + saved artifacts under seed_dir.
    """
    # Avoid collisions
    TaskBase.ALL_TASK_IDS.clear()

    seed_dir = os.path.join(trial_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    tb_dir = os.path.join(seed_dir, "tb")
    os.makedirs(tb_dir, exist_ok=True)

    # Save PPO config for traceability at seed level
    try:
        with open(os.path.join(seed_dir, "ppo_frozen_used.json"), "w", encoding="utf-8") as f:
            json.dump(ppo_config, f, indent=2)
    except Exception:
        pass

    # PPO geometry validation
    if ppo_config:
        num_steps = ppo_config.get("num_steps")
        num_mini_batch = ppo_config.get("num_mini_batch")
        if num_steps is not None and num_mini_batch is not None:
            try:
                num_steps = int(num_steps)
                num_mini_batch = int(num_mini_batch)
            except Exception:
                raise ValueError(f"PPO config has non-int num_steps/num_mini_batch: {num_steps}, {num_mini_batch}")

            if not _validate_minibatch_geometry(num_steps, args.num_processes, num_mini_batch):
                batch_size = num_steps * args.num_processes
                minibatch_size = batch_size // num_mini_batch if num_mini_batch else 0
                raise ValueError(
                    "Invalid PPO minibatch geometry: "
                    f"num_steps={num_steps}, num_processes={args.num_processes}, num_mini_batch={num_mini_batch}. "
                    f"Require divisible batches and minibatch_size >= {MIN_MINIBATCH_SIZE} (got {minibatch_size})."
                )

    # Ensure per-seed intervention RNG independence (important for SET-style random regrowth)
    seeded_params = dict(params)
    if args.method.lower() == 'set':
        seeded_params.setdefault('seed', int(seed))

    experiment, policy = build_experiment_and_policy(
        policy_name="ppo",
        experiment_name=args.experiment,
        intervention_type=args.method,
        intervention_params=seeded_params,
        ppo_config=ppo_config,
        output_dir=seed_dir,
        num_processes=args.num_processes,
        budget_override=args.budget_override,
    )

    set_eval_mode(experiment, args.eval_mode)

    writer = SummaryWriter(log_dir=tb_dir)

    # Identify train/eval tasks
    train_tasks = [t for t in experiment.tasks if _is_train_task(t)]
    if not train_tasks:
        train_tasks = list(experiment.tasks)
        print(f"[Seed {seed}] No explicit train tasks found; using all tasks as train tasks for snapshots and forgetting.")

    eval_tasks = [t for t in experiment.tasks if _is_eval_task(t)]
    eval_fallback_used = False
    if not eval_tasks:
        eval_tasks = list(train_tasks)
        eval_fallback_used = True
        print(f"[Seed {seed}] No eval tasks found; using train tasks for final objective evaluation (fallback).")

    def _task_timesteps(task: Any) -> int:
        for attr in ("_num_timesteps", "num_timesteps"):
            if hasattr(task, attr):
                try:
                    return int(getattr(task, attr))
                except Exception:
                    pass
        if hasattr(task, "_task_spec"):
            ts = getattr(task._task_spec, "_num_timesteps", None)
            if ts is None:
                ts = getattr(task._task_spec, "num_timesteps", None)
            try:
                return int(ts) if ts is not None else 0
            except Exception:
                return 0
        return 0

    # Provide total training budget to interventions (for relative schedules like SET)
    cycle_count = getattr(experiment, "_cycle_count", 1) or 1
    total_train_timesteps = sum(_task_timesteps(t) for t in train_tasks) * int(cycle_count)
    total_train_steps = None
    try:
        num_steps = int(ppo_config.get("num_steps", ppo_config.get("n_steps", 256)) or 256)
        num_mini_batch = int(ppo_config.get("num_mini_batch", ppo_config.get("num_minibatches", 4)) or 4)
        ppo_epoch = int(
            ppo_config.get("ppo_epoch")
            or ppo_config.get("ppo_epochs")
            or ppo_config.get("update_epochs")
            or ppo_config.get("num_epochs")
            or ppo_config.get("epochs")
            or 4
        )
        denom = max(1, num_steps * int(args.num_processes))
        rollouts = int(math.ceil(total_train_timesteps / float(denom))) if total_train_timesteps > 0 else 0
        if rollouts > 0 and num_mini_batch > 0 and ppo_epoch > 0:
            total_train_steps = rollouts * ppo_epoch * num_mini_batch
    except Exception:
        total_train_steps = None
    try:
        if hasattr(policy, "_intervention") and policy._intervention is not None:
            policy._intervention.ctx.params["total_train_timesteps"] = int(total_train_timesteps)
            policy._intervention.ctx.params["train_tasks_per_cycle"] = int(len(train_tasks))
            policy._intervention.ctx.params["num_cycles"] = int(cycle_count)
            if total_train_steps is not None:
                policy._intervention.ctx.params["total_train_steps"] = int(total_train_steps)
    except Exception:
        pass

    # ----------------
    # Training loop with snapshots (on train tasks, for forgetting)
    # ----------------
    snapshots: List[Dict[str, Any]] = []
    total_train_timesteps = 0
    cycle_count = getattr(experiment, "_cycle_count", 1) or 1

    def run_snapshot(cycle_id: int, task_run_idx: int, label: str):
        snap_per_task, snap_aggs = evaluate_policy_on_tasks(
            experiment,
            policy,
            writer,
            episodes_per_task=min(args.episodes_per_task, args.snapshot_episodes_per_task),
            objective_metric=args.primary_metric,   # store both aggs but choose a consistent metric label
            include_raw_returns=args.save_raw_returns,
            tasks_override=train_tasks,
        )
        snapshots.append({
            "cycle": cycle_id,
            "task_run_idx": task_run_idx,
            "label": label,
            "timestamp": datetime.datetime.now().isoformat(),
            "aggregate_mean": snap_aggs.get("mean_eval_return"),
            "aggregate_iqm": snap_aggs.get("iqm_eval_return"),
            "per_task": snap_per_task,
        })

    # Initial snapshot (pre-train)
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

            run_snapshot(cycle_id=cycle_id, task_run_idx=task_run_idx, label="post_task")

    # Final train-task eval (for objective metrics on train set)
    per_task_train_final_obj, aggregates_train_obj = evaluate_policy_on_tasks(
        experiment,
        policy,
        writer,
        episodes_per_task=args.episodes_per_task,
        objective_metric=args.primary_metric,
        include_raw_returns=args.save_raw_returns,
        tasks_override=train_tasks,
    )

    # Final eval-task eval (held-out generalization) if enabled
    per_task_eval_obj: List[Dict[str, Any]] = []
    aggregates_eval_obj: Dict[str, float] = {"objective": float("nan"), "mean_eval_return": float("nan"), "iqm_eval_return": float("nan")}
    if args.eval_mode != "none":
        per_task_eval_obj, aggregates_eval_obj = evaluate_policy_on_tasks(
            experiment,
            policy,
            writer,
            args.episodes_per_task,
            args.primary_metric,
            include_raw_returns=args.save_raw_returns,
            tasks_override=eval_tasks,
        )

    # Final train-task eval for forgetting (uses dedicated episodes)
    per_task_train_final_forgetting, _ = evaluate_policy_on_tasks(
        experiment,
        policy,
        writer,
        episodes_per_task=args.forgetting_episodes_per_task,
        objective_metric=args.primary_metric,
        include_raw_returns=args.save_raw_returns,
        tasks_override=train_tasks,
    )

    # Plasticity metrics (optional)
    eff_ranks = None
    if hasattr(experiment, "get_current_effective_ranks"):
        try:
            eff_ranks = experiment.get_current_effective_ranks()
        except Exception:
            eff_ranks = None

    # Forgetting (defensible: only after task learned)
    forgetting_mean, forgetting_iqm = _compute_forgetting_after_learned(
        snapshots=snapshots,
        train_tasks=train_tasks,
        per_task_train_final=per_task_train_final_forgetting,
    )

    # Final metrics per set
    final_mean_train = aggregates_train_obj.get("mean_eval_return", float("nan"))
    final_iqm_train = aggregates_train_obj.get("iqm_eval_return", float("nan"))
    final_primary_train = final_iqm_train if args.primary_metric == "iqm" else final_mean_train

    final_mean_eval = aggregates_eval_obj.get("mean_eval_return", float("nan"))
    final_iqm_eval = aggregates_eval_obj.get("iqm_eval_return", float("nan"))
    final_primary_eval = final_iqm_eval if args.primary_metric == "iqm" else final_mean_eval

    if args.final_eval_set == "eval":
        final_mean = final_mean_eval
        final_iqm = final_iqm_eval
        final_primary = final_primary_eval
        objective_source = "eval_fallback_to_train" if eval_fallback_used else "eval"
        if eval_fallback_used:
            final_primary = final_primary_train
            final_mean = final_mean_train
            final_iqm = final_iqm_train
    else:
        final_mean = final_mean_train
        final_iqm = final_iqm_train
        final_primary = final_primary_train
        objective_source = "train"

    forgetting_primary = forgetting_iqm if args.primary_metric == "iqm" else forgetting_mean

    # Objective
    if args.objective in ("mean", "iqm"):
        objective_value = final_primary
    elif args.objective == "forgetting":
        objective_value = forgetting_primary
    elif args.objective == "composite":
        objective_value = (final_primary if not math.isnan(final_primary) else 0.0) - args.lambda_forgetting * (
            forgetting_primary if not math.isnan(forgetting_primary) else 0.0
        )
    else:
        objective_value = float("nan")

    # Save snapshots
    snapshots_path = os.path.join(seed_dir, "snapshots.json")
    with open(snapshots_path, "w", encoding="utf-8") as f:
        json.dump(snapshots, f, indent=2)

    writer.close()

    seed_summary = {
        "seed": seed,
        "params": params,
        "objective_type": args.objective,
        "primary_metric": args.primary_metric,
        "final_eval_set": args.final_eval_set,
        "objective_source": objective_source,
        "objective": objective_value,
        "final_mean": final_mean,
        "final_iqm": final_iqm,
        "final_primary": final_primary,
        "final_mean_train": final_mean_train,
        "final_iqm_train": final_iqm_train,
        "final_primary_train": final_primary_train,
        "final_mean_eval": final_mean_eval,
        "final_iqm_eval": final_iqm_eval,
        "final_primary_eval": final_primary_eval,
        "forgetting_mean": forgetting_mean,
        "forgetting_iqm": forgetting_iqm,
        "forgetting_primary": forgetting_primary,
        "eval_mode": args.eval_mode,
        "train_task_ids": [getattr(t, "task_id", "") for t in train_tasks],
        "eval_task_ids": [getattr(t, "task_id", "") for t in eval_tasks],
        "eval_fallback_used": eval_fallback_used,
        "snapshots_path": snapshots_path,
        "tb_dir": tb_dir,
    }

    try:
        with open(os.path.join(seed_dir, "seed_summary.json"), "w", encoding="utf-8") as f:
            json.dump(seed_summary, f, indent=2)
    except Exception:
        pass

    return {
        "seed": seed,
        "seed_dir": seed_dir,
        "objective": objective_value,
        "final_mean": final_mean,
        "final_iqm": final_iqm,
        "final_primary": final_primary,
        "final_mean_train": final_mean_train,
        "final_iqm_train": final_iqm_train,
        "final_primary_train": final_primary_train,
        "final_mean_eval": final_mean_eval,
        "final_iqm_eval": final_iqm_eval,
        "final_primary_eval": final_primary_eval,
        "forgetting_mean": forgetting_mean,
        "forgetting_iqm": forgetting_iqm,
        "forgetting_primary": forgetting_primary,
        "per_task_eval": per_task_eval_obj,
        "per_task_train_final": per_task_train_final_forgetting,
        "plasticity_metrics": {"effective_rank": eff_ranks} if eff_ranks is not None else None,
        "snapshots_path": snapshots_path,
        "train_task_ids": seed_summary["train_task_ids"],
        "eval_task_ids": seed_summary["eval_task_ids"],
        "eval_fallback_used": eval_fallback_used,
        "objective_source": objective_source,
    }


def run_trial(
    trial_idx: int,
    args,
    params: Dict[str, Any],
    base_dir: str,
    timestamp: str,
    ppo_config: Dict[str, Any]
) -> Dict[str, Any]:

    # Keep per-process threading low for RL env stepping
    torch.set_num_threads(1)

    # Avoid collisions across trials
    TaskBase.ALL_TASK_IDS.clear()

    trial_dir = os.path.join(base_dir, f"trial_{trial_idx:03d}")
    os.makedirs(trial_dir, exist_ok=True)

    # Save candidate params for traceability
    try:
        with open(os.path.join(trial_dir, "candidate_params.json"), "w", encoding="utf-8") as f:
            json.dump({"trial": trial_idx, "params": params}, f, indent=2)
    except Exception:
        pass

    print(f"\n=== Running Trial {trial_idx} ===")
    print(f"Params: {params}")

    trial_seeds = list(args.trial_seeds)
    print(f"Seeds: {trial_seeds}")

    try:
        seed_results: List[Dict[str, Any]] = []
        for s in trial_seeds:
            print(f"  [Seed {s}] starting...")
            sr = _run_single_seed(args=args, params=params, trial_dir=trial_dir, seed=s, ppo_config=ppo_config)
            seed_results.append(sr)
            print(f"  [Seed {s}] done. objective={sr['objective']:.6f}")

        # Aggregate across seeds
        obj_vals = [sr["objective"] for sr in seed_results]
        final_primary_vals = [sr["final_primary"] for sr in seed_results]
        forgetting_primary_vals = [sr["forgetting_primary"] for sr in seed_results]
        final_mean_vals = [sr["final_mean"] for sr in seed_results]
        final_iqm_vals = [sr["final_iqm"] for sr in seed_results]
        final_mean_train_vals = [sr["final_mean_train"] for sr in seed_results]
        final_iqm_train_vals = [sr["final_iqm_train"] for sr in seed_results]
        final_primary_train_vals = [sr["final_primary_train"] for sr in seed_results]
        final_mean_eval_vals = [sr["final_mean_eval"] for sr in seed_results]
        final_iqm_eval_vals = [sr["final_iqm_eval"] for sr in seed_results]
        final_primary_eval_vals = [sr["final_primary_eval"] for sr in seed_results]
        fmean_vals = [sr["forgetting_mean"] for sr in seed_results]
        fiqm_vals = [sr["forgetting_iqm"] for sr in seed_results]

        objective_mean = _mean(obj_vals)
        objective_stderr = _stderr(obj_vals)

        final_mean = _mean(final_mean_vals)
        final_mean_stderr = _stderr(final_mean_vals)

        final_iqm = _mean(final_iqm_vals)
        final_iqm_stderr = _stderr(final_iqm_vals)

        final_primary = _mean(final_primary_vals)
        final_primary_stderr = _stderr(final_primary_vals)

        final_mean_train = _mean(final_mean_train_vals)
        final_mean_train_stderr = _stderr(final_mean_train_vals)

        final_iqm_train = _mean(final_iqm_train_vals)
        final_iqm_train_stderr = _stderr(final_iqm_train_vals)

        final_primary_train = _mean(final_primary_train_vals)
        final_primary_train_stderr = _stderr(final_primary_train_vals)

        final_mean_eval = _mean(final_mean_eval_vals)
        final_mean_eval_stderr = _stderr(final_mean_eval_vals)

        final_iqm_eval = _mean(final_iqm_eval_vals)
        final_iqm_eval_stderr = _stderr(final_iqm_eval_vals)

        final_primary_eval = _mean(final_primary_eval_vals)
        final_primary_eval_stderr = _stderr(final_primary_eval_vals)

        forgetting_mean = _mean(fmean_vals)
        forgetting_mean_stderr = _stderr(fmean_vals)

        forgetting_iqm = _mean(fiqm_vals)
        forgetting_iqm_stderr = _stderr(fiqm_vals)

        forgetting_primary = _mean(forgetting_primary_vals)
        forgetting_primary_stderr = _stderr(forgetting_primary_vals)

        # Composite is just "objective" when objective_type == composite; else None for readability
        composite_value = objective_mean if args.objective == "composite" else None

        # Hash PPO config (plus save it once at trial level)
        ppo_hash = hashlib.md5(
            json.dumps(ppo_config, sort_keys=True).encode("utf-8") if ppo_config else b"no_ppo"
        ).hexdigest()
        try:
            with open(os.path.join(trial_dir, "ppo_frozen_used.json"), "w", encoding="utf-8") as f:
                json.dump(ppo_config, f, indent=2)
        except Exception:
            pass

        trial_summary = {
            "trial": trial_idx,
            "timestamp": timestamp,
            "method": args.method,
            "params": params,
            "status": "ok",
            "objective_type": args.objective,
            "primary_metric": args.primary_metric,
            "objective": objective_mean,
            "objective_stderr": objective_stderr,
            "final_mean": final_mean,
            "final_mean_stderr": final_mean_stderr,
            "final_iqm": final_iqm,
            "final_iqm_stderr": final_iqm_stderr,
            "final_primary": final_primary,
            "final_primary_stderr": final_primary_stderr,
            "final_mean_train": final_mean_train,
            "final_mean_train_stderr": final_mean_train_stderr,
            "final_iqm_train": final_iqm_train,
            "final_iqm_train_stderr": final_iqm_train_stderr,
            "final_primary_train": final_primary_train,
            "final_primary_train_stderr": final_primary_train_stderr,
            "final_mean_eval": final_mean_eval,
            "final_mean_eval_stderr": final_mean_eval_stderr,
            "final_iqm_eval": final_iqm_eval,
            "final_iqm_eval_stderr": final_iqm_eval_stderr,
            "final_primary_eval": final_primary_eval,
            "final_primary_eval_stderr": final_primary_eval_stderr,
            "forgetting_mean": forgetting_mean,
            "forgetting_mean_stderr": forgetting_mean_stderr,
            "forgetting_iqm": forgetting_iqm,
            "forgetting_iqm_stderr": forgetting_iqm_stderr,
            "forgetting_primary": forgetting_primary,
            "forgetting_primary_stderr": forgetting_primary_stderr,
            "composite": composite_value,
            "trial_seeds": trial_seeds,
            "ppo_config_hash": ppo_hash,
            "output_dir": trial_dir,
            "final_eval_set": args.final_eval_set,
            "train_task_ids": seed_results[0].get("train_task_ids", []),
            "eval_task_ids": seed_results[0].get("eval_task_ids", []),
            "eval_fallback_used": any(sr.get("eval_fallback_used") for sr in seed_results),
            "seed_results": [{
                "seed": sr["seed"],
                "seed_dir": sr["seed_dir"],
                "objective": sr.get("objective"),
                "final_mean": sr.get("final_mean"),
                "final_iqm": sr.get("final_iqm"),
                "final_primary": sr.get("final_primary"),
                "final_mean_train": sr.get("final_mean_train"),
                "final_iqm_train": sr.get("final_iqm_train"),
                "final_primary_train": sr.get("final_primary_train"),
                "final_mean_eval": sr.get("final_mean_eval"),
                "final_iqm_eval": sr.get("final_iqm_eval"),
                "final_primary_eval": sr.get("final_primary_eval"),
                "forgetting_mean": sr.get("forgetting_mean"),
                "forgetting_iqm": sr.get("forgetting_iqm"),
                "forgetting_primary": sr.get("forgetting_primary"),
                "objective_source": sr.get("objective_source"),
            } for sr in seed_results],
        }

        with open(os.path.join(trial_dir, "trial_summary.json"), "w", encoding="utf-8") as f:
            json.dump(trial_summary, f, indent=2, default=str)

        # Save lean best_config.json
        best_cfg = {
            "method": args.method,
            "params": params,
            "objective_type": args.objective,
            "primary_metric": args.primary_metric,
            "objective": objective_mean,
            "objective_stderr": objective_stderr,
            "final_mean": final_mean,
            "final_mean_stderr": final_mean_stderr,
            "final_iqm": final_iqm,
            "final_iqm_stderr": final_iqm_stderr,
            "final_primary": final_primary,
            "final_primary_stderr": final_primary_stderr,
            "final_mean_train": final_mean_train,
            "final_mean_train_stderr": final_mean_train_stderr,
            "final_iqm_train": final_iqm_train,
            "final_iqm_train_stderr": final_iqm_train_stderr,
            "final_primary_train": final_primary_train,
            "final_primary_train_stderr": final_primary_train_stderr,
            "final_mean_eval": final_mean_eval,
            "final_mean_eval_stderr": final_mean_eval_stderr,
            "final_iqm_eval": final_iqm_eval,
            "final_iqm_eval_stderr": final_iqm_eval_stderr,
            "final_primary_eval": final_primary_eval,
            "final_primary_eval_stderr": final_primary_eval_stderr,
            "forgetting_mean": forgetting_mean,
            "forgetting_mean_stderr": forgetting_mean_stderr,
            "forgetting_iqm": forgetting_iqm,
            "forgetting_iqm_stderr": forgetting_iqm_stderr,
            "forgetting_primary": forgetting_primary,
            "forgetting_primary_stderr": forgetting_primary_stderr,
            "trial_seeds": trial_seeds,
            "final_eval_set": args.final_eval_set,
            "train_task_ids": seed_results[0].get("train_task_ids", []),
            "eval_task_ids": seed_results[0].get("eval_task_ids", []),
            "eval_fallback_used": any(sr.get("eval_fallback_used") for sr in seed_results),
        }
        with open(os.path.join(trial_dir, "best_config.json"), "w", encoding="utf-8") as f:
            json.dump(best_cfg, f, indent=2)

        print(f"Trial {trial_idx} objective_mean={objective_mean:.6f} ± {objective_stderr:.6f}")

        return {
            "trial": trial_idx,
            "timestamp": timestamp,
            "method": args.method,
            "params": params,
            "status": "ok",
            "objective_type": args.objective,
            "primary_metric": args.primary_metric,
            "objective": objective_mean,
            "objective_stderr": objective_stderr,
            "final_mean": final_mean,
            "final_mean_stderr": final_mean_stderr,
            "final_iqm": final_iqm,
            "final_iqm_stderr": final_iqm_stderr,
            "final_primary": final_primary,
            "final_primary_stderr": final_primary_stderr,
            "final_mean_train": final_mean_train,
            "final_mean_train_stderr": final_mean_train_stderr,
            "final_iqm_train": final_iqm_train,
            "final_iqm_train_stderr": final_iqm_train_stderr,
            "final_primary_train": final_primary_train,
            "final_primary_train_stderr": final_primary_train_stderr,
            "final_mean_eval": final_mean_eval,
            "final_mean_eval_stderr": final_mean_eval_stderr,
            "final_iqm_eval": final_iqm_eval,
            "final_iqm_eval_stderr": final_iqm_eval_stderr,
            "final_primary_eval": final_primary_eval,
            "final_primary_eval_stderr": final_primary_eval_stderr,
            "forgetting_mean": forgetting_mean,
            "forgetting_mean_stderr": forgetting_mean_stderr,
            "forgetting_iqm": forgetting_iqm,
            "forgetting_iqm_stderr": forgetting_iqm_stderr,
            "forgetting_primary": forgetting_primary,
            "forgetting_primary_stderr": forgetting_primary_stderr,
            "composite": composite_value,
            "trial_seeds": trial_seeds,
            "seed_results": seed_results,
            "final_eval_set": args.final_eval_set,
            "train_task_ids": seed_results[0].get("train_task_ids", []),
            "eval_task_ids": seed_results[0].get("eval_task_ids", []),
            "eval_fallback_used": any(sr.get("eval_fallback_used") for sr in seed_results),
            "output_dir": trial_dir,
        }

    except Exception as e:
        print(f"Trial {trial_idx} failed: {e}")
        traceback.print_exc()
        return {
            "trial": trial_idx,
            "status": "failed",
            "error": str(e),
            "method": args.method,
            "params": params,
            "output_dir": trial_dir,
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=str)
    parser.add_argument("--method", required=True, type=str,
                        choices=["dense", "gmp", "set", "reset", "partial_reinit", "redo"])
    parser.add_argument("--trials", default=10, type=int)
    parser.add_argument("--search", default="random", choices=["grid", "random"], type=str)

    # Candidate generation seed (does NOT control training randomness anymore)
    parser.add_argument("--seed", default=0, type=int, help="Seed for candidate generation (grid shuffle / random sampling).")

    parser.add_argument("--trial_seeds", default="0,1,2", type=str,
                        help="Comma-separated seeds to run each candidate with, e.g. '0,1,2'.")

    parser.add_argument("--budget_override", default=None, type=int)
    parser.add_argument("--eval_mode", default="final_only", choices=["final_only", "periodic", "none"])
    parser.add_argument("--episodes_per_task", default=10, type=int)

    parser.add_argument("--final_eval_set", default="train", choices=["train", "eval"],
                        help="Which set to use for the final objective: train (default, aligns with forgetting) or eval (held-out generalization). Eval falls back to train if no eval tasks are defined.")

    parser.add_argument("--snapshot_episodes_per_task", default=2, type=int,
                        help="Episodes per task for lightweight snapshots used in forgetting (cheap).")

    parser.add_argument("--forgetting_episodes_per_task", default=5, type=int,
                        help="Episodes per task for FINAL train-task eval used in forgetting (less noisy).")

    parser.add_argument("--objective", default="mean", choices=["mean", "iqm", "forgetting", "composite"])
    parser.add_argument("--primary_metric", default="iqm", choices=["mean", "iqm"],
                        help="Metric family used consistently for objective/composite/forgetting (mean or iqm).")

    parser.add_argument("--lambda_forgetting", default=0.5, type=float,
                        help="Weight for forgetting in composite objective (final - lambda * forgetting).")

    parser.add_argument("--save_raw_returns", action="store_true", default=False,
                        help="If set, store raw episode returns in outputs")

    parser.add_argument("--ppo_config", default=None, type=str,
                        help="Path to JSON file with fixed PPO hyperparameters (tuned)")

    parser.add_argument("--params_json", default=None, type=str,
                        help="Path to JSON dict (single candidate) or list of dicts (explicit candidates).")
    parser.add_argument("--params_inline", default=None, type=str,
                        help="Inline JSON dict or list of dicts for explicit candidates.")

    parser.add_argument("--num_processes", default=1, type=int)
    parser.add_argument("--output_root", default="runs/tuning", type=str)

    # NOTE: store_true should default False; user can opt-in to shuffle
    parser.add_argument("--grid_shuffle", action="store_true", default=False)

    parser.add_argument("--save_best_k", default=5, type=int)

    parser.add_argument("--parallel_trials", default=1, type=int,
                        help="Number of trials to run in parallel. Use <=0 to run all candidates in parallel.")

    return parser.parse_args()


def main():
    args = parse_args()

    # Fix thread oversubscription for parallel RL env stepping
    torch.set_num_threads(1)

    # Enforce primary metric consistency with objective
    if args.objective in {"mean", "iqm"} and args.primary_metric != args.objective:
        print(f"Overriding primary_metric to '{args.objective}' to match objective '{args.objective}'.")
        args.primary_metric = args.objective

    # Parse trial seeds once
    parsed_trial_seeds = _parse_int_list(args.trial_seeds)
    if not parsed_trial_seeds:
        parsed_trial_seeds = [0]
    args.trial_seeds = parsed_trial_seeds
    trial_seeds_str = ",".join(str(s) for s in args.trial_seeds)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    suffix_components = [timestamp]
    if slurm_job_id:
        suffix_components.append(slurm_job_id)
    else:
        suffix_components.append(str(os.getpid()))
        suffix_components.append(uuid.uuid4().hex[:6])
    unique_suffix = "-".join(suffix_components)
    base_dir = os.path.join(args.output_root, args.experiment, args.method, unique_suffix)
    os.makedirs(base_dir, exist_ok=False)

    print(f"Starting tuning for method: {args.method}")
    print(f"Experiment: {args.experiment}")
    print(f"Output Directory: {base_dir}")
    print(f"Trials: {args.trials} | Search: {args.search} | Candidate seed: {args.seed} | Grid shuffle: {args.grid_shuffle}")
    print(f"Eval mode: {args.eval_mode} | Episodes/task: {args.episodes_per_task}")
    print(f"Snapshot eps/task: {args.snapshot_episodes_per_task} | Forgetting final eps/task: {args.forgetting_episodes_per_task}")
    print(f"Objective: {args.objective} | Primary metric: {args.primary_metric} | Final eval set: {args.final_eval_set} | Lambda_forgetting: {args.lambda_forgetting}")
    print(f"Trial seeds: {trial_seeds_str}")

    # Load PPO Config
    ppo_config: Dict[str, Any] = {}
    if args.ppo_config:
        print(f"Loading fixed PPO params from: {args.ppo_config}")
        with open(args.ppo_config, "r", encoding="utf-8") as f:
            ppo_config = json.load(f)
            if isinstance(ppo_config, list):
                ppo_config = ppo_config[0]
            if "best" in ppo_config:
                # Handle both "best_ppo.json" wrappers and raw dicts
                b = ppo_config["best"]
                if isinstance(b, dict) and "ppo_params" in b:
                    ppo_config = b["ppo_params"]
                else:
                    ppo_config = b

    # Generate candidates
    opt_steps_total = _estimate_total_optimizer_steps(args, ppo_config) if args.method.lower() == 'set' else None

    if opt_steps_total is not None:
        print(f"[SET scaling] Estimated total optimizer steps: {opt_steps_total}")
        gs, rs = _search_spaces('set', opt_steps_total=opt_steps_total)
        print(f"[SET scaling] Grid update_interval: {gs.get('update_interval')}")
        print(f"[SET scaling] Grid warmup_steps: {gs.get('warmup_steps')}")

    explicit_candidates: Optional[List[Dict[str, Any]]] = None
    if args.params_json or args.params_inline:
        try:
            if args.params_json:
                with open(args.params_json, "r", encoding="utf-8") as f:
                    explicit = json.load(f)
            else:
                explicit = json.loads(args.params_inline)
            if isinstance(explicit, list):
                explicit_candidates = list(explicit)
            elif isinstance(explicit, dict):
                explicit_candidates = [explicit]
            else:
                raise ValueError("params must be a dict or list of dicts")
        except Exception as e:
            raise ValueError(f"Failed to parse explicit params: {e}")

    if explicit_candidates is not None:
        candidates = explicit_candidates
        print(f"Using explicit candidates: {len(candidates)}")
    else:
        candidates = build_candidates(args.method, args.search, args.trials, args.seed, args.grid_shuffle, opt_steps_total=opt_steps_total)

    # GMP: enforce prune_cycle=0 and set tasks_per_cycle from experiment (if known)
    if args.method.lower() == "gmp":
        tpc = _infer_train_tasks_per_cycle(args.experiment)
        for c in candidates:
            c["prune_cycle"] = 0
            if tpc is not None:
                c["tasks_per_cycle"] = int(tpc)
        print(f"[GMP] prune_cycle fixed at 0; tasks_per_cycle={tpc if tpc is not None else 'unknown'}")

    # Save candidates list for traceability
    try:
        with open(os.path.join(base_dir, "candidates.json"), "w", encoding="utf-8") as f:
            json.dump({"method": args.method, "search": args.search, "seed": args.seed, "candidates": candidates}, f, indent=2)
    except Exception:
        pass

    results: List[Dict[str, Any]] = []

    parallel_trials = int(args.parallel_trials)
    if parallel_trials <= 0:
        parallel_trials = max(1, len(candidates))

    if parallel_trials <= 1 or len(candidates) <= 1:
        for i, params in enumerate(candidates):
            res = run_trial(i, args, params, base_dir, timestamp, ppo_config)
            results.append(res)

            write_results_jsonl(os.path.join(base_dir, "results.jsonl"), res)
            write_leaderboard_csv(os.path.join(base_dir, "leaderboard.csv"), results)
            write_best_json(os.path.join(base_dir, "best_interventions.json"), results, args.save_best_k)
    else:
        print(f"Running trials in parallel with {parallel_trials} workers...")
        mp_ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=parallel_trials, mp_context=mp_ctx) as ex:
            futures = {
                ex.submit(run_trial, i, args, params, base_dir, timestamp, ppo_config): i
                for i, params in enumerate(candidates)
            }
            for fut in concurrent.futures.as_completed(futures):
                res = fut.result()
                results.append(res)

                write_results_jsonl(os.path.join(base_dir, "results.jsonl"), res)
                write_leaderboard_csv(os.path.join(base_dir, "leaderboard.csv"), results)
                write_best_json(os.path.join(base_dir, "best_interventions.json"), results, args.save_best_k)

    print(f"Tuning complete. Best results saved to {os.path.join(base_dir, 'best_interventions.json')}")


if __name__ == "__main__":
    main()
