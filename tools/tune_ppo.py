#!/usr/bin/env python3
"""
PURPOSE:
    Tune PPO *base* hyperparameters on the **dense** baseline (no intervention),
    then freeze them for intervention-specific tuning.

"Bulletproof" upgrades (CORA-aligned):
  - Default objective = IQM (robust to outliers)
  - Default eval episodes/task = 10 (CORA uses E=10 eval episodes)
  - Multi-seed evaluation per trial (recommended; CORA reports across seeds)
  - Optional override of continual testing frequency for periodic eval

IMPORTANT (Project alignment):
  - This script now supports choosing WHICH tasks to score PPO on via --score_tasks.
    If your project metrics/forgetting are defined on TRAIN tasks (retention),
    set --score_tasks train (DEFAULT).
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from continual_rl.available_policies import get_available_policies
from continual_rl.experiment_specs import get_available_experiments
from continual_rl.experiments.tasks.task_base import TaskBase
from continual_rl.utils.utils import Utils

# Prefer the same evaluation mechanism as your intervention tuner.
# If TaskSpec import fails in your continual_rl version, we fall back to continual_eval.
try:
    from continual_rl.experiments.tasks.task_spec import TaskSpec  # type: ignore
    _HAS_TASKSPEC = True
except Exception:
    TaskSpec = None  # type: ignore
    _HAS_TASKSPEC = False


# -------------------------
# PPO hyperparameter search spaces
# -------------------------

PPO_SEARCH_SPACE_RANDOM = {
    "learning_rate": {"type": "loguniform", "low": 1e-5, "high": 1e-3},
    "clip_param": {"type": "uniform", "low": 0.1, "high": 0.3},
    "entropy_coef": {"type": "loguniform", "low": 1e-4, "high": 0.05},
    "value_loss_coef": {"type": "uniform", "low": 0.25, "high": 2.0},
    "gamma": {"type": "uniform", "low": 0.98, "high": 0.999},
    "gae_lambda": {"type": "uniform", "low": 0.9, "high": 0.99},
    "num_steps": {"type": "categorical", "values": [64, 128, 256, 512]},
    "num_mini_batch": {"type": "categorical", "values": [4, 8, 16, 32, 64]},
    "ppo_epoch": {"type": "categorical", "values": [3, 4, 5, 8, 10]},
    "max_grad_norm": {"type": "uniform", "low": 0.3, "high": 2.0},
}

PPO_SEARCH_SPACE_GRID = {
    "learning_rate": [1e-4, 3e-4, 7e-4],
    "clip_param": [0.1, 0.2, 0.3],
    "entropy_coef": [0.001, 0.01, 0.05],
    "value_loss_coef": [0.5, 1.0],
    "gamma": [0.99, 0.995],
    "gae_lambda": [0.9, 0.95],
    "num_steps": [128, 256],
    "num_mini_batch": [8, 16, 32],
    "ppo_epoch": [3, 4, 5],
    "max_grad_norm": [0.5, 1.0],
}

PPO_PARAM_ALIASES = {
    "learning_rate": "learning_rate",
    "lr": "learning_rate",
    "clip_param": "clip_param",
    "clip_epsilon": "clip_param",
    "entropy_coef": "entropy_coef",
    "entropy_coefficient": "entropy_coef",
    "value_loss_coef": "value_loss_coef",
    "value_coefficient": "value_loss_coef",
    "vf_coef": "value_loss_coef",
    "gamma": "gamma",
    "discount": "gamma",
    "gae_lambda": "gae_lambda",
    "gae_tau": "gae_lambda",
    "lambda": "gae_lambda",
    "num_steps": "num_steps",
    "rollout_length": "num_steps",
    "n_steps": "num_steps",
    "num_mini_batch": "num_mini_batch",
    "minibatch_count": "num_mini_batch",
    "n_minibatches": "num_mini_batch",
    "ppo_epoch": "ppo_epoch",
    "n_epochs": "ppo_epoch",
    "update_epochs": "ppo_epoch",
    "max_grad_norm": "max_grad_norm",
    "gradient_clip": "max_grad_norm",
}

REQUIRED_PPO_PARAMS = [
    "learning_rate", "clip_param", "entropy_coef", "value_loss_coef",
    "gamma", "gae_lambda", "num_steps", "num_mini_batch", "ppo_epoch", "max_grad_norm"
]


# -------------------------
# Sampling utilities
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


# -------------------------
# PPO minibatch geometry checks
# -------------------------

MIN_MINIBATCH_SIZE = 32


def _validate_minibatch_geometry(num_steps: int, num_processes: int, num_mini_batch: int) -> bool:
    batch_size = num_steps * num_processes
    if num_mini_batch <= 0 or num_mini_batch > batch_size:
        return False
    if batch_size % num_mini_batch != 0:
        return False
    minibatch_size = batch_size // num_mini_batch
    return minibatch_size >= MIN_MINIBATCH_SIZE


def _fix_minibatch_geometry(params: Dict[str, Any], num_processes: int) -> Dict[str, Any]:
    params = params.copy()
    num_steps = int(params.get("num_steps", 128))
    num_mini_batch = int(params.get("num_mini_batch", 32))
    batch_size = num_steps * num_processes

    if _validate_minibatch_geometry(num_steps, num_processes, num_mini_batch):
        return params

    # Find largest valid divisor <= requested num_mini_batch
    for candidate in range(num_mini_batch, 0, -1):
        if candidate <= batch_size and batch_size % candidate == 0:
            minibatch_size = batch_size // candidate
            if minibatch_size >= MIN_MINIBATCH_SIZE:
                params["num_mini_batch"] = candidate
                return params

    # Worst-case fallback
    params["num_mini_batch"] = 1
    return params


def build_ppo_candidates(
    search: str,
    trials: int,
    seed: int,
    num_processes: int,
    grid_shuffle: bool = True,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    if search == "grid":
        candidates = _grid(PPO_SEARCH_SPACE_GRID, shuffle=grid_shuffle, rng=rng)
        candidates = candidates[:trials]
    else:
        candidates = _random_sample(rng, PPO_SEARCH_SPACE_RANDOM, trials)

    candidates = [_fix_minibatch_geometry(c, num_processes) for c in candidates]
    return candidates


# -------------------------
# Metrics
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


def _nanmean(xs: List[float]) -> float:
    arr = np.asarray(xs, dtype=np.float64)
    return float(np.nanmean(arr)) if arr.size else float("nan")


def _nanstd(xs: List[float]) -> float:
    arr = np.asarray(xs, dtype=np.float64)
    return float(np.nanstd(arr)) if arr.size else float("nan")


def _stderr(xs: List[float]) -> float:
    arr = np.asarray(xs, dtype=np.float64)
    if arr.size <= 1:
        return float("nan")
    return float(np.nanstd(arr) / math.sqrt(arr.size))


# -------------------------
# Task selection (train/eval/all/auto)
# -------------------------

def _is_eval_task(task: Any) -> bool:
    if getattr(task, "task_id", "").endswith("_eval"):
        return True
    task_spec = getattr(task, "_task_spec", None)
    if task_spec is not None and getattr(task_spec, "eval_mode", False):
        return True
    if getattr(task, "eval_mode", False):
        return True
    return False


def _is_train_task(task: Any) -> bool:
    return not _is_eval_task(task)


def _select_scoring_tasks(experiment: Any, score_tasks: str) -> Tuple[List[Any], bool]:
    """
    Returns (tasks, fallback_used).
    score_tasks:
      - train: only train tasks (retention-aligned)
      - eval:  only eval tasks (generalization)
      - auto:  eval tasks if present else all tasks (old behavior)
      - all:   all tasks
    """
    all_tasks = list(getattr(experiment, "tasks", []))
    eval_tasks = [t for t in all_tasks if _is_eval_task(t)]
    train_tasks = [t for t in all_tasks if _is_train_task(t)]

    fallback = False

    if score_tasks == "all":
        return all_tasks, False

    if score_tasks == "train":
        if train_tasks:
            return train_tasks, False
        # Fallback: if nothing marked train, use all
        return all_tasks, True

    if score_tasks == "eval":
        if eval_tasks:
            return eval_tasks, False
        return all_tasks, True

    # auto
    if eval_tasks:
        return eval_tasks, False
    return all_tasks, True


# -------------------------
# Evaluation helpers
# -------------------------

def _extract_rewards_from_eval_info(info: Any) -> List[float]:
    rewards: List[float] = []
    if info is None:
        return rewards

    if isinstance(info, tuple) and len(info) == 2:
        info = info[0]

    if isinstance(info, dict):
        for key in ["episode_returns", "returns", "episode_return", "reward", "rewards"]:
            if key in info:
                val = info[key]
                if isinstance(val, (list, tuple, np.ndarray)):
                    rewards.extend([float(x) for x in val])
                elif isinstance(val, (int, float)):
                    rewards.append(float(val))
                return rewards
        return rewards

    if isinstance(info, (list, tuple, np.ndarray)):
        rewards.extend([float(x) for x in info])
        return rewards

    if isinstance(info, (int, float)):
        rewards.append(float(info))
        return rewards

    return rewards


def _evaluate_with_taskspec(
    experiment: Any,
    policy: Any,
    summary_writer: SummaryWriter,
    episodes_per_task: int,
    tasks: List[Any],
) -> List[Dict[str, Any]]:
    """
    Preferred evaluation: build a TaskSpec with return_after_episode_num=E and run task._run.
    This matches your intervention tuner style and is usually more reliable than continual_eval.
    """
    per_task: List[Dict[str, Any]] = []

    for task in tasks:
        returns: List[float] = []

        # Build an eval TaskSpec based on the task's spec.
        ts = getattr(task, "_task_spec", None)
        if ts is None:
            raise RuntimeError("Task has no _task_spec; cannot use TaskSpec evaluation.")

        eval_spec = TaskSpec(
            task_id=task.task_id,
            action_space_id=task.action_space_id,
            preprocessor=ts.preprocessor,
            env_spec=ts.env_spec,
            num_timesteps=10**9,  # should be plenty to finish E episodes
            eval_mode=True,
            return_after_episode_num=episodes_per_task,
            with_continual_eval=False,
        )

        # Collect returns from the runner
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
            returns.extend(list(returns_batch))

        returns = returns[:episodes_per_task]
        per_task.append({
            "task_id": getattr(task, "task_id", "unknown"),
            "mean": float(np.mean(returns)) if returns else float("nan"),
            "iqm": _iqm(returns),
            "count": len(returns),
            "raw_returns": returns,
        })

    return per_task


def _evaluate_with_continual_eval(
    experiment: Any,
    policy: Any,
    summary_writer: SummaryWriter,
    episodes_per_task: int,
    tasks: List[Any],
) -> List[Dict[str, Any]]:
    """
    Fallback evaluation using task.continual_eval (old behavior).
    """
    per_task: List[Dict[str, Any]] = []

    for task in tasks:
        task_id = getattr(task, "task_id", "unknown")
        if not hasattr(task, "continual_eval"):
            raise RuntimeError("Task has no continual_eval; cannot fall back to continual_eval evaluation.")

        runner = task.continual_eval(
            run_id=str(task_id),
            policy=policy,
            summary_writer=summary_writer,
            output_dir=experiment.output_dir,
            timestep_log_offset=0,
        )

        rewards: List[float] = []
        while len(rewards) < episodes_per_task:
            try:
                _, info = next(runner)
                rewards.extend(_extract_rewards_from_eval_info(info))
            except StopIteration:
                break
            except Exception:
                continue

        rewards = rewards[:episodes_per_task]
        per_task.append({
            "task_id": task_id,
            "mean": float(np.mean(rewards)) if rewards else float("nan"),
            "iqm": _iqm(rewards),
            "count": len(rewards),
            "raw_returns": rewards,
        })

    return per_task


def evaluate_policy_on_tasks(
    experiment: Any,
    policy: Any,
    summary_writer: SummaryWriter,
    episodes_per_task: int,
    objective_metric: str,
    tasks: List[Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Evaluate policy on the provided tasks and compute aggregate metrics.
    objective_metric in {"mean","iqm"} controls which aggregate is "objective".
    """
    if not tasks:
        return [], {"mean_eval_return": float("nan"), "iqm_eval_return": float("nan"), "objective": float("nan")}

    per_task: List[Dict[str, Any]] = []

    # Preferred path: TaskSpec + task._run
    if _HAS_TASKSPEC:
        try:
            per_task = _evaluate_with_taskspec(experiment, policy, summary_writer, episodes_per_task, tasks)
        except Exception:
            # Fallback
            per_task = _evaluate_with_continual_eval(experiment, policy, summary_writer, episodes_per_task, tasks)
    else:
        per_task = _evaluate_with_continual_eval(experiment, policy, summary_writer, episodes_per_task, tasks)

    mean_over_tasks = _nanmean([t["mean"] for t in per_task]) if per_task else float("nan")
    iqm_over_tasks = _nanmean([t["iqm"] for t in per_task]) if per_task else float("nan")
    objective = mean_over_tasks if objective_metric == "mean" else iqm_over_tasks

    return per_task, {
        "mean_eval_return": mean_over_tasks,
        "iqm_eval_return": iqm_over_tasks,
        "objective": objective,
    }


# -------------------------
# Experiment helpers
# -------------------------

def apply_budget_override(experiment: Any, budget_override: Optional[int]) -> None:
    """Override num_timesteps for TRAIN tasks only (not eval tasks)."""
    if budget_override is None:
        return

    for task in getattr(experiment, "tasks", []):
        task_spec = getattr(task, "_task_spec", None)
        if task_spec is None:
            continue
        if getattr(task_spec, "eval_mode", False):
            continue

        task_spec._num_timesteps = int(budget_override)

        if hasattr(task, "_rolling_return_count"):
            task._rolling_return_count = max(1, min(task._rolling_return_count, 100))


def apply_total_timesteps(experiment: Any, total_timesteps: int, cycle_count: int) -> Dict[str, Any]:
    """
    Split total_timesteps across TRAIN tasks and cycles.
    Returns metadata including per-task budgets.
    """
    train_tasks = [t for t in getattr(experiment, "tasks", []) if _is_train_task(t)]
    if not train_tasks:
        raise ValueError("No train tasks found to apply total_timesteps.")

    if cycle_count < 1:
        raise ValueError("cycle_count must be >= 1")

    per_task_total = total_timesteps // (len(train_tasks) * cycle_count)
    remainder = total_timesteps - per_task_total * len(train_tasks) * cycle_count

    per_task_budgets = []
    for idx, task in enumerate(train_tasks):
        extra = 1 if idx < remainder else 0
        task_budget = per_task_total + extra
        task_spec = getattr(task, "_task_spec", None)
        if task_spec is not None:
            task_spec._num_timesteps = int(task_budget)
        if hasattr(task, "_rolling_return_count"):
            task._rolling_return_count = max(1, min(task._rolling_return_count, 100))
        per_task_budgets.append(task_budget)

    # enforce cycles
    if hasattr(experiment, "_cycle_count"):
        experiment._cycle_count = int(cycle_count)

    expected_total = sum(per_task_budgets) * cycle_count
    return {
        "train_task_count": len(train_tasks),
        "cycle_count": cycle_count,
        "per_task_budgets": per_task_budgets,
        "expected_total": expected_total,
    }


def _wrap_env_spec_with_seed(env_spec, base_seed: int):
    counter = {"i": 0}

    def _next_seed() -> int:
        seed = int(base_seed + counter["i"])
        counter["i"] += 1
        return seed

    def _make():
        seed = _next_seed()
        env, _ = Utils.make_env(env_spec, seed_to_set=seed)
        return env

    # Attach seed provider for parallel envs
    _make._seed_to_set = _next_seed  # type: ignore[attr-defined]
    return _make


def apply_env_seed(experiment: Any, base_seed: int) -> None:
    """Ensure envs use deterministic seeds (per-env) based on base_seed."""
    for task in getattr(experiment, "tasks", []):
        task_spec = getattr(task, "_task_spec", None)
        if task_spec is None:
            continue
        task_spec._env_spec = _wrap_env_spec_with_seed(task_spec.env_spec, base_seed)


def _extract_procgen_env_ids(experiment: Any) -> List[str]:
    env_ids: List[str] = []
    for task in getattr(experiment, "tasks", []):
        if _is_eval_task(task):
            continue
        task_spec = getattr(task, "_task_spec", None)
        if task_spec is None:
            continue
        env, _ = Utils.make_env(task_spec.env_spec)
        try:
            env_ids.append(getattr(getattr(env, "spec", None), "id", None) or "unknown")
        finally:
            try:
                env.close()
            except Exception:
                pass
    return env_ids


def _normalize_procgen_name(env_id: str) -> str:
    if env_id is None:
        return "unknown"
    if env_id.startswith("procgen-"):
        return env_id.replace("procgen-", "", 1)
    return env_id


def set_eval_mode(experiment: Any, mode: str, continual_testing_freq: Optional[int] = None) -> None:
    """
    Configure evaluation mode:
      - periodic: keep normal continual eval frequency (optionally override freq)
      - final_only: disable periodic eval, only do post-training eval
      - none: no evaluation at all
    """
    if mode == "periodic":
        if continual_testing_freq is not None and hasattr(experiment, "_continual_testing_freq"):
            experiment._continual_testing_freq = int(continual_testing_freq)
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
# Config / policy build
# -------------------------

def normalize_ppo_params(params: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in params.items():
        canonical = PPO_PARAM_ALIASES.get(key, key)
        normalized[canonical] = value
    return normalized


def verify_ppo_params(params: Dict[str, Any], config_obj: Any, strict: bool) -> bool:
    params = normalize_ppo_params(params)

    for param_name in REQUIRED_PPO_PARAMS:
        if param_name not in params:
            continue

        expected = params[param_name]
        actual = getattr(config_obj, param_name, None)

        if isinstance(expected, float) and isinstance(actual, float):
            if not math.isclose(expected, actual, rel_tol=1e-6):
                msg = f"PPO param {param_name} mismatch: expected {expected}, got {actual}"
                if strict:
                    raise ValueError(msg)
                print(f"WARNING: {msg}")
                return False
        elif expected != actual:
            msg = f"PPO param {param_name} mismatch: expected {expected}, got {actual}"
            if strict:
                raise ValueError(msg)
            print(f"WARNING: {msg}")
            return False

    return True


def build_experiment_and_policy(
    policy_name: str,
    experiment_name: str,
    ppo_params: Dict[str, Any],
    output_dir: str,
    num_processes: int,
    strict_verify: bool,
) -> Tuple[Any, Any]:
    """
    Build experiment and policy with given PPO hyperparameters.
    Uses dense baseline (intervention_type="dense") for PPO tuning.
    """
    available_policies = get_available_policies()
    available_experiments = get_available_experiments()

    if policy_name not in available_policies:
        raise ValueError(f"Unknown policy: {policy_name}")
    if experiment_name not in available_experiments:
        raise ValueError(f"Unknown experiment: {experiment_name}")

    experiment = available_experiments[experiment_name]
    experiment.set_output_dir(output_dir)

    policy_struct = available_policies[policy_name]
    ppo_params = normalize_ppo_params(ppo_params)

    config_dict = ppo_params.copy()
    config_dict.update({
        "intervention_type": "dense",
        "intervention_params": {},
        "num_processes": num_processes,
        "use_gae": True,
    })

    config = policy_struct.config()
    config.load_from_dict(config_dict)
    config.set_output_dir(output_dir)

    verify_ppo_params(ppo_params, config, strict=strict_verify)

    policy = policy_struct.policy(config, experiment.observation_space, experiment.action_spaces)
    policy.set_task_ids(experiment.task_ids)

    # Save what was actually used
    try:
        cfg_dump = {k: v for k, v in config.__dict__.items()
                    if not k.startswith("_") and isinstance(v, (int, float, str, bool, type(None)))}
    except Exception:
        cfg_dump = config_dict

    with open(os.path.join(output_dir, "ppo_config_used.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_dump, f, indent=2)

    return experiment, policy


# -------------------------
# Run helpers
# -------------------------

def clear_task_registry() -> None:
    TaskBase.ALL_TASK_IDS.clear()


def _set_global_seeds(seed: int, deterministic_torch: bool = False) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _parse_seeds_arg(seeds_arg: Optional[str], fallback_seed: int) -> List[int]:
    if seeds_arg is None:
        return [int(fallback_seed)]
    s = seeds_arg.strip()
    if not s:
        return [int(fallback_seed)]
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    return [int(p) for p in parts]


def run_trial(
    trial_idx: int,
    args: argparse.Namespace,
    ppo_params: Dict[str, Any],
    base_dir: str,
    timestamp: str,
    scoring_tasks_info: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run a single tuning trial. If multiple seeds are provided, run one full
    train+eval per seed and aggregate metrics across seeds.
    """
    trial_dir = os.path.join(base_dir, f"trial_{trial_idx:03d}")
    os.makedirs(trial_dir, exist_ok=True)

    # Validate minibatch geometry once (seed-independent)
    num_steps = int(ppo_params.get("num_steps", 128))
    num_mini_batch = int(ppo_params.get("num_mini_batch", 32))
    if not _validate_minibatch_geometry(num_steps, args.num_processes, num_mini_batch):
        batch_size = num_steps * args.num_processes
        minibatch_size = batch_size // num_mini_batch if num_mini_batch > 0 else 0
        raise ValueError(
            f"Invalid minibatch geometry: num_steps={num_steps}, "
            f"num_processes={args.num_processes}, num_mini_batch={num_mini_batch}. "
            f"Require: (num_steps * num_processes) % num_mini_batch == 0 "
            f"AND minibatch_size >= {MIN_MINIBATCH_SIZE} (got {minibatch_size})"
        )

    seeds = _parse_seeds_arg(args.seeds, args.seed)

    print(f"[Trial {trial_idx:03d}] PPO params: {ppo_params}")
    print(f"[Trial {trial_idx:03d}] Seeds: {seeds}")

    seed_runs: List[Dict[str, Any]] = []

    for seed in seeds:
        seed_dir = os.path.join(trial_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        tb_dir = os.path.join(seed_dir, "tb")

        clear_task_registry()
        _set_global_seeds(seed, deterministic_torch=args.deterministic_torch)

        experiment, policy = build_experiment_and_policy(
            policy_name=args.policy,
            experiment_name=args.experiment,
            ppo_params=ppo_params,
            output_dir=seed_dir,
            num_processes=args.num_processes,
            strict_verify=args.strict_verify,
        )

        # Apply deterministic env seeding per seed
        apply_env_seed(experiment, seed)

        # Verify train tasks (exactly 3) and expected procgen env ids
        train_tasks = [t for t in getattr(experiment, "tasks", []) if _is_train_task(t)]
        if len(train_tasks) != 3:
            raise ValueError(f"Expected exactly 3 train tasks, got {len(train_tasks)}")

        env_ids = _extract_procgen_env_ids(experiment)
        env_names = [_normalize_procgen_name(eid) for eid in env_ids]
        expected = [t.strip() for t in args.tasks.split(",") if t.strip()]
        if sorted(env_names) != sorted(expected):
            raise ValueError(
                f"Train task mismatch. Expected {expected}, got {env_names} (env_ids={env_ids})"
            )

        set_eval_mode(experiment, args.eval_mode, continual_testing_freq=args.continual_testing_freq)

        budget_info = apply_total_timesteps(experiment, args.total_timesteps, args.cycles)

        # Enforce exact total timesteps
        if budget_info["expected_total"] != int(args.total_timesteps):
            raise ValueError(
                f"Total timesteps mismatch: expected {budget_info['expected_total']} vs requested {args.total_timesteps}"
            )

        # Determine scoring tasks for THIS experiment instance
        scoring_tasks, fallback_used = _select_scoring_tasks(experiment, args.score_tasks)
        scoring_ids = [getattr(t, "task_id", "unknown") for t in scoring_tasks]
        if fallback_used:
            print(f"[Trial {trial_idx:03d} | Seed {seed}] WARNING: score_tasks='{args.score_tasks}' had no matches; "
                  f"falling back to scoring on ALL tasks.")

        writer = SummaryWriter(log_dir=tb_dir)

        # Save scoring task info per seed for auditability
        with open(os.path.join(seed_dir, "scoring_tasks.json"), "w", encoding="utf-8") as f:
            json.dump({
                "score_tasks": args.score_tasks,
                "fallback_used": fallback_used,
                "scoring_task_ids": scoring_ids,
            }, f, indent=2)

        print(f"[Trial {trial_idx:03d} | Seed {seed}] Starting training...")
        experiment.try_run(policy, summary_writer=writer)

        # Verify executed timesteps from run metadata
        from continual_rl.experiments.run_metadata import RunMetadata
        run_meta = RunMetadata(seed_dir)
        final_steps = int(run_meta.total_train_timesteps)
        print(f"[Trial {trial_idx:03d} | Seed {seed}] Final train timesteps: {final_steps}")
        if final_steps != int(args.total_timesteps):
            raise ValueError(
                f"Executed timesteps mismatch: {final_steps} vs target {args.total_timesteps}"
            )

        aggregates = {"objective": float("nan")}
        per_task: List[Dict[str, Any]] = []

        if args.eval_mode != "none":
            print(f"[Trial {trial_idx:03d} | Seed {seed}] Scoring PPO on tasks ({args.score_tasks}) "
                  f"with {args.episodes_per_task} eps/task...")
            per_task, aggregates = evaluate_policy_on_tasks(
                experiment=experiment,
                policy=policy,
                summary_writer=writer,
                episodes_per_task=args.episodes_per_task,
                objective_metric=args.objective,
                tasks=scoring_tasks,
            )

        writer.flush()
        writer.close()

        seed_runs.append({
            "seed": seed,
            "aggregates": aggregates,
            "per_task": per_task,
            "output_dir": seed_dir,
            "tb_dir": tb_dir,
            "scoring_task_ids": scoring_ids,
            "budget_info": budget_info,
            "final_train_timesteps": final_steps,
            "train_env_ids": env_ids,
        })

        obj_val = aggregates.get("objective", float("nan"))
        print(f"[Trial {trial_idx:03d} | Seed {seed}] Done. Objective ({args.objective})={obj_val:.4f}")

    # Aggregate across seeds
    seed_objectives = [r["aggregates"].get("objective", float("nan")) for r in seed_runs]
    seed_means = [r["aggregates"].get("mean_eval_return", float("nan")) for r in seed_runs]
    seed_iqms = [r["aggregates"].get("iqm_eval_return", float("nan")) for r in seed_runs]

    objective_mean = _nanmean(seed_objectives)
    objective_std = _nanstd(seed_objectives)
    objective_stderr = _stderr(seed_objectives)

    mean_eval_return_mean = _nanmean(seed_means)
    iqm_eval_return_mean = _nanmean(seed_iqms)

    result = {
        "trial": trial_idx,
        "timestamp": timestamp,
        "status": "ok",
        "ppo_params": ppo_params,
        "seeds": seeds,
        "seed_runs": seed_runs,
        "aggregates": {
            "objective": objective_mean,
            "objective_std": objective_std,
            "objective_stderr": objective_stderr,
            "mean_eval_return": mean_eval_return_mean,
            "iqm_eval_return": iqm_eval_return_mean,
        },
        "output_dir": trial_dir,
        "experiment": args.experiment,
        "policy": args.policy,
        "num_processes": args.num_processes,
        "total_timesteps": args.total_timesteps,
        "cycle_count": args.cycles,
        "task_list": [t.strip() for t in args.tasks.split(",") if t.strip()],
        "eval_mode": args.eval_mode,
        "episodes_per_task": args.episodes_per_task,
        "objective_metric": args.objective,
        "continual_testing_freq": args.continual_testing_freq,
        "score_tasks": args.score_tasks,
        "scoring_tasks_info": scoring_tasks_info,
        "train_env_ids": seed_runs[0].get("train_env_ids", []) if seed_runs else [],
    }

    print(
        f"[Trial {trial_idx:03d}] Aggregated objective ({args.objective}) = "
        f"{objective_mean:.4f} ± {objective_std:.4f} (stderr={objective_stderr:.4f})"
    )

    # Per-trial summary JSON (single file per trial)
    summary_path = os.path.join(trial_dir, "trial_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def write_results_jsonl(path: str, result: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")


def write_leaderboard_csv(path: str, results: List[Dict[str, Any]]) -> None:
    if not results:
        return

    sorted_results = sorted(
        results,
        key=lambda x: x.get("aggregates", {}).get("objective", float("-inf")),
        reverse=True,
    )

    fieldnames = [
        "rank", "trial", "status",
        "objective", "objective_std", "objective_stderr",
        "mean_eval_return", "iqm_eval_return",
        "seeds",
        "score_tasks",
        "learning_rate", "clip_param", "entropy_coef", "value_loss_coef",
        "gamma", "gae_lambda", "num_steps", "num_mini_batch", "ppo_epoch", "max_grad_norm",
        "output_dir",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for rank, r in enumerate(sorted_results, 1):
            agg = r.get("aggregates", {})
            ppo = r.get("ppo_params", {})

            writer.writerow({
                "rank": rank,
                "trial": r.get("trial"),
                "status": r.get("status"),
                "objective": agg.get("objective"),
                "objective_std": agg.get("objective_std"),
                "objective_stderr": agg.get("objective_stderr"),
                "mean_eval_return": agg.get("mean_eval_return"),
                "iqm_eval_return": agg.get("iqm_eval_return"),
                "seeds": ",".join(map(str, r.get("seeds", []))),
                "score_tasks": r.get("score_tasks"),
                "learning_rate": ppo.get("learning_rate"),
                "clip_param": ppo.get("clip_param"),
                "entropy_coef": ppo.get("entropy_coef"),
                "value_loss_coef": ppo.get("value_loss_coef"),
                "gamma": ppo.get("gamma"),
                "gae_lambda": ppo.get("gae_lambda"),
                "num_steps": ppo.get("num_steps"),
                "num_mini_batch": ppo.get("num_mini_batch"),
                "ppo_epoch": ppo.get("ppo_epoch"),
                "max_grad_norm": ppo.get("max_grad_norm"),
                "output_dir": r.get("output_dir"),
            })


def write_best_ppo_json(path: str, results: List[Dict[str, Any]]) -> None:
    ok_results = [r for r in results if r.get("status") == "ok"]
    ok_results = [r for r in ok_results
                  if not math.isnan(r.get("aggregates", {}).get("objective", float("nan")))]

    if not ok_results:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"best": None, "all_trials": []}, f, indent=2)
        return

    sorted_results = sorted(
        ok_results,
        key=lambda x: x.get("aggregates", {}).get("objective", float("-inf")),
        reverse=True,
    )

    best = sorted_results[0]

    output = {
        "best": {
            "trial": best.get("trial"),
            "ppo_params": best.get("ppo_params"),
            "objective": best.get("aggregates", {}).get("objective"),
            "objective_std": best.get("aggregates", {}).get("objective_std"),
            "objective_stderr": best.get("aggregates", {}).get("objective_stderr"),
            "objective_metric": best.get("objective_metric"),
            "mean_eval_return": best.get("aggregates", {}).get("mean_eval_return"),
            "iqm_eval_return": best.get("aggregates", {}).get("iqm_eval_return"),
            "seeds": best.get("seeds"),
            "score_tasks": best.get("score_tasks"),
            "output_dir": best.get("output_dir"),
        },
        "metadata": {
            "experiment": best.get("experiment"),
            "policy": best.get("policy"),
            "seed": best.get("seeds", [None])[0],
            "seeds": best.get("seeds"),
            "num_processes": best.get("num_processes"),
            "total_timesteps": best.get("total_timesteps"),
            "cycle_count": best.get("cycle_count"),
            "task_list": best.get("task_list"),
            "eval_mode": best.get("eval_mode"),
            "episodes_per_task": best.get("episodes_per_task"),
            "continual_testing_freq": best.get("continual_testing_freq"),
            "score_tasks": best.get("score_tasks"),
            "timestamp": best.get("timestamp"),
            "total_trials": len(results),
            "successful_trials": len(ok_results),
        },
        "all_successful_trials": [
            {
                "trial": r.get("trial"),
                "objective": r.get("aggregates", {}).get("objective"),
                "objective_std": r.get("aggregates", {}).get("objective_std"),
                "objective_stderr": r.get("aggregates", {}).get("objective_stderr"),
                "ppo_params": r.get("ppo_params"),
                "seeds": r.get("seeds"),
                "score_tasks": r.get("score_tasks"),
            }
            for r in sorted_results
        ],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PPO Baseline Hyperparameter Tuning (CORA-style robustness)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
Example usage (RETENTION-aligned PPO baseline):
    python tools/tune_ppo.py --experiment procgen_3_tasks_1_cycle_1M \
            --total_timesteps 1000000 --trials 50 --eval_mode final_only \
            --episodes_per_task 10 --objective iqm --num_processes 1 \
            --seeds "0,1,2" --score_tasks train --strict_verify

Notes:
  - CORA describes evaluating E=10 episodes at evaluation points and aggregating across seeds.
  - If your project defines forgetting/retention on TRAIN tasks, use --score_tasks train (default).
"""
    )

    parser.add_argument("--experiment", required=True, type=str,
                        help="Name of the experiment from experiment_specs.py")
    parser.add_argument("--policy", default="ppo", type=str,
                        help="Policy to tune (default: ppo)")

    parser.add_argument("--trials", default=20, type=int,
                        help="Number of hyperparameter configurations to try")
    parser.add_argument("--search", default="random", choices=["random", "grid"], type=str,
                        help="Search strategy: random sampling or grid search")

    # Default True, but allow disabling via flag
    parser.add_argument("--no_grid_shuffle", action="store_false", dest="grid_shuffle",
                        help="Disable shuffling grid combinations (default: shuffle enabled)")
    parser.set_defaults(grid_shuffle=True)

    parser.add_argument("--total_timesteps", default=1_000_000, type=int,
                        help="Total TRAIN timesteps across all tasks/cycles (default: 1,000,000)")
    parser.add_argument("--cycles", default=1, type=int,
                        help="Number of cycles through tasks (default: 1)")
    parser.add_argument("--tasks", default="climber-v0,dodgeball-v0,fruitbot-v0", type=str,
                        help="Comma-separated task env names (exactly 3) for tuning")
    parser.add_argument("--allow_nonfinal_budget", action="store_true",
                        help="Allow budgets other than 1,000,000 total timesteps")
    parser.add_argument("--num_processes", default=1, type=int,
                        help="Number of parallel environments (default: 1)")

    parser.add_argument("--eval_mode", default="final_only",
                        choices=["final_only", "periodic", "none"],
                        help="Evaluation mode: final_only (recommended), periodic, or none")
    parser.add_argument("--continual_testing_freq", default=None, type=int,
                        help="If eval_mode=periodic, override experiment._continual_testing_freq (timesteps)")

    parser.add_argument("--episodes_per_task", default=10, type=int,
                        help="Number of evaluation episodes per task (default: 10)")

    parser.add_argument("--objective", default="iqm", choices=["mean", "iqm"],
                        help="Objective metric for selection: mean or IQM (default: iqm)")

    parser.add_argument("--score_tasks", default="train", choices=["train", "eval", "auto", "all"],
                        help="Which tasks to score PPO on. Default=train (retention-aligned). "
                             "auto=eval if present else all (old behavior).")

    parser.add_argument("--output_root", default="runs/tuning", type=str,
                        help="Root directory for tuning outputs")

    parser.add_argument("--seed", default=0, type=int,
                        help="Base seed (used for candidate generation; also used if --seeds not provided)")

    parser.add_argument("--seeds", default="0,1,2", type=str,
                        help='Comma-separated seeds to run per trial, e.g. "0,1,2". Use "0" for single-seed.')

    parser.add_argument("--deterministic_torch", action="store_true",
                        help="Enable deterministic cuDNN settings for reproducibility (may slow down).")

    parser.add_argument("--strict_verify", action="store_true",
                        help="Raise error if any PPO hyperparameter didn't apply correctly")

    parser.add_argument("--dry_run", action="store_true",
                        help="Print generated configurations without running trials")

    return parser.parse_args()


def main():
    args = parse_args()

    # Final-budget guardrails
    if args.total_timesteps != 1_000_000 and not args.allow_nonfinal_budget:
        raise ValueError(
            "--total_timesteps must be 1,000,000 for final-budget tuning. "
            "Use --allow_nonfinal_budget to override explicitly."
        )
    if args.cycles != 1 and not args.allow_nonfinal_budget:
        raise ValueError(
            "--cycles must be 1 for final-budget tuning. "
            "Use --allow_nonfinal_budget to override explicitly."
        )

    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if len(task_list) != 3:
        raise ValueError("--tasks must list exactly 3 tasks.")

    seed_list = _parse_seeds_arg(args.seeds, args.seed)
    if seed_list != [0, 1, 2] and not args.allow_nonfinal_budget:
        raise ValueError("--seeds must be exactly '0,1,2' for final-budget tuning. Use --allow_nonfinal_budget to override.")

    try:
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(args.output_root, args.experiment, "ppo_tune", timestamp)
    os.makedirs(base_dir, exist_ok=True)

    print("PPO Baseline Tuning")
    print("=" * 60)
    print(f"PPO tuning: {args.total_timesteps:,} total timesteps | 3 seeds | 3 tasks | {args.cycles} cycle")
    print(f"Experiment: {args.experiment}")
    print(f"Policy: {args.policy}")
    print(f"Search: {args.search}")
    print(f"Trials: {args.trials}")
    print(f"Total timesteps: {args.total_timesteps}")
    print(f"Cycles: {args.cycles}")
    print(f"Tasks: {args.tasks}")
    print(f"Eval mode: {args.eval_mode}")
    print(f"Objective: {args.objective}")
    print(f"Episodes/task: {args.episodes_per_task}")
    print(f"Seeds per trial: {args.seeds}")
    print(f"Score tasks: {args.score_tasks}")
    print(f"Grid shuffle: {args.grid_shuffle}")
    print(f"Output: {base_dir}")
    print("=" * 60)

    candidates = build_ppo_candidates(
        search=args.search,
        trials=args.trials,
        seed=args.seed,
        num_processes=args.num_processes,
        grid_shuffle=args.grid_shuffle,
    )

    print(f"Generated {len(candidates)} PPO configurations")

    # Save candidates for traceability
    with open(os.path.join(base_dir, "candidates.json"), "w", encoding="utf-8") as f:
        json.dump({"candidates": candidates}, f, indent=2)

    if args.dry_run:
        print("\n[DRY RUN] Generated configurations:")
        for i, params in enumerate(candidates):
            print(f"\nTrial {i:03d}:")
            for k, v in sorted(params.items()):
                print(f"  {k}: {v}")
        return

    # Save tuning config
    config_path = os.path.join(base_dir, "tuning_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": args.experiment,
            "policy": args.policy,
            "search": args.search,
            "trials": args.trials,
            "seed": args.seed,
            "seeds": args.seeds,
            "total_timesteps": args.total_timesteps,
            "cycles": args.cycles,
            "tasks": task_list,
            "num_processes": args.num_processes,
            "eval_mode": args.eval_mode,
            "continual_testing_freq": args.continual_testing_freq,
            "episodes_per_task": args.episodes_per_task,
            "objective": args.objective,
            "score_tasks": args.score_tasks,
            "deterministic_torch": args.deterministic_torch,
            "timestamp": timestamp,
        }, f, indent=2)

    results_path = os.path.join(base_dir, "results.jsonl")
    all_results: List[Dict[str, Any]] = []

    # One-time scoring task info (documented intent)
    scoring_tasks_info = {
        "score_tasks": args.score_tasks,
        "tasks": task_list,
        "total_timesteps": args.total_timesteps,
        "cycles": args.cycles,
        "definition": {
            "train": "Only tasks not marked eval_mode and not suffixed _eval (retention-aligned).",
            "eval": "Only tasks marked eval_mode or suffixed _eval (generalization).",
            "auto": "Eval tasks if present else all tasks (old behavior).",
            "all": "All tasks in the experiment.",
        }
    }
    with open(os.path.join(base_dir, "scoring_tasks.json"), "w", encoding="utf-8") as f:
        json.dump(scoring_tasks_info, f, indent=2)

    for idx, ppo_params in enumerate(candidates):
        print(f"\n{'=' * 60}")
        print(f"Trial {idx + 1}/{len(candidates)}")
        print(f"{'=' * 60}")

        try:
            result = run_trial(
                trial_idx=idx,
                args=args,
                ppo_params=ppo_params,
                base_dir=base_dir,
                timestamp=timestamp,
                scoring_tasks_info=scoring_tasks_info,
            )
            all_results.append(result)

        except Exception as e:
            error_msg = f"Trial {idx} failed: {e}"
            traceback.print_exc()

            fail_result = {
                "trial": idx,
                "timestamp": timestamp,
                "status": "error",
                "error": error_msg,
                "ppo_params": ppo_params,
                "seeds": args.seeds,
                "aggregates": {"objective": float("nan")},
                "seed_runs": [],
                "output_dir": os.path.join(base_dir, f"trial_{idx:03d}"),
                "experiment": args.experiment,
                "policy": args.policy,
                "objective_metric": args.objective,
                "score_tasks": args.score_tasks,
                "total_timesteps": args.total_timesteps,
                "cycle_count": args.cycles,
                "task_list": [t.strip() for t in args.tasks.split(",") if t.strip()],
            }
            all_results.append(fail_result)

        finally:
            write_results_jsonl(results_path, all_results[-1])

    leaderboard_path = os.path.join(base_dir, "leaderboard.csv")
    write_leaderboard_csv(leaderboard_path, all_results)

    best_ppo_path = os.path.join(base_dir, "best_ppo.json")
    write_best_ppo_json(best_ppo_path, all_results)

    print(f"\n{'=' * 60}")
    print("PPO Tuning Complete")
    print(f"{'=' * 60}")

    successful = [r for r in all_results if r.get("status") == "ok"]
    failed = [r for r in all_results if r.get("status") != "ok"]

    print(f"Total trials: {len(all_results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        best = max(successful, key=lambda x: x.get("aggregates", {}).get("objective", float("-inf")))
        best_obj = best.get("aggregates", {}).get("objective", float("nan"))
        print(f"\nBest trial: {best.get('trial')}")
        print(f"Best objective ({args.objective}) across seeds: {best_obj:.4f}")
        print(f"Score tasks: {best.get('score_tasks')}")
        print("Best PPO params:")
        for k, v in sorted(best.get("ppo_params", {}).items()):
            print(f"  {k}: {v}")

    print("\nOutputs:")
    print(f"  Results: {results_path}")
    print(f"  Leaderboard: {leaderboard_path}")
    print(f"  Best config: {best_ppo_path}")
    print("\nTo use best config for intervention tuning:")
    print(f"  python tools/tune_interventions.py --ppo_params_path {best_ppo_path} ...")


if __name__ == "__main__":
    main()
