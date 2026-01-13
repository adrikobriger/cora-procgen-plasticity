#!/usr/bin/env python3
"""
PPO Baseline Hyperparameter Tuning Script (CORA-style)
======================================================

PURPOSE:
    Tune PPO *base* hyperparameters on the **dense** baseline (no intervention),
    then freeze them for intervention-specific tuning.

CORA-LIKE PROTOCOL:
    1. Define a small sweep over base PPO settings.
    2. For each candidate config:
       - Build experiment with dense baseline (no intervention)
       - Train on TRAIN tasks only
       - Evaluate on EVAL tasks (held-out generalization)
    3. Pick best config by objective (mean or IQM over eval returns).
    4. Save best_ppo.json for use in tune_interventions.py.

WHAT ARE EVAL TASKS?
    Eval tasks are tasks where `task_spec.eval_mode=True` or `task_id` ends with "_eval".
    These are held-out evaluation environments used for generalization testing.
    In procgen, eval tasks typically use the full level distribution (num_levels=0)
    while train tasks use a restricted set (e.g., num_levels=200).

WHAT IS eval_mode="final_only"?
    - Disables periodic continual evaluations during training (expensive).
    - BUT still runs the post-training evaluation on eval tasks.
    This is the recommended setting for tuning sweeps to reduce compute cost.

WHY TUNE ON EVAL RETURNS?
    We want PPO hyperparameters that generalize well to unseen levels/tasks.
    Training returns can overfit to the specific train levels, so eval returns
    provide a better signal for hyperparameter selection.

MINIBATCH GEOMETRY CONSTRAINTS:
    PPO requires: (num_steps * num_processes) % num_mini_batch == 0
    This ensures even division of rollout data into minibatches.
    We enforce this constraint during candidate generation and verification.

USAGE:
    python tools/tune_ppo.py --experiment procgen_3_tasks_1_cycle_5m_tuning \\
        --budget_override 100000 --trials 20 --eval_mode final_only \\
        --episodes_per_task 5 --objective mean --num_processes 1 --strict_verify

OUTPUT:
    <output_root>/<experiment>/ppo_tune/<timestamp>/
        ├── results.jsonl         # One JSON line per trial
        ├── leaderboard.csv       # Sorted by objective
        ├── best_ppo.json         # Best trial's PPO params + metadata
        └── trial_XXX/
            ├── ppo_config_used.json
            └── tb/               # TensorBoard logs
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


# =============================================================================
# PPO Hyperparameter Search Space (CORA-like small sweep)
# =============================================================================

# Canonical PPO hyperparameter names and their search ranges.
# These map directly to PPOPolicyConfig attributes.

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

# Map of canonical names to PPO config attribute names (for aliases/verification)
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

# Required PPO params for strict verification
REQUIRED_PPO_PARAMS = [
    "learning_rate", "clip_param", "entropy_coef", "value_loss_coef",
    "gamma", "gae_lambda", "num_steps", "num_mini_batch", "ppo_epoch", "max_grad_norm"
]


# =============================================================================
# Search Space Sampling Utilities
# =============================================================================

def _log_uniform(rng: random.Random, low: float, high: float) -> float:
    """Sample from log-uniform distribution."""
    lo = math.log(low)
    hi = math.log(high)
    return math.exp(rng.uniform(lo, hi))


def _sample_value(spec: Dict[str, Any], rng: random.Random) -> Any:
    """Sample a single value from a spec dictionary."""
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
    """Generate n random samples from the spec."""
    samples: List[Dict[str, Any]] = []
    for _ in range(n):
        sample = {k: _sample_value(v, rng) for k, v in spec.items()}
        samples.append(sample)
    return samples


def _grid(options: Dict[str, List[Any]], shuffle: bool, rng: random.Random) -> List[Dict[str, Any]]:
    """Generate all combinations from grid options."""
    keys = list(options.keys())
    combos = list(itertools.product(*[options[k] for k in keys]))
    if shuffle:
        rng.shuffle(combos)
    return [{k: vals[i] for i, k in enumerate(keys)} for vals in combos]


# Minimum minibatch size for meaningful gradient estimates
# Too small minibatches lead to high variance and poor learning
MIN_MINIBATCH_SIZE = 32


def _validate_minibatch_geometry(num_steps: int, num_processes: int, num_mini_batch: int) -> bool:
    """
    Validate PPO minibatch geometry constraint.
    
    PPO requires: (num_steps * num_processes) % num_mini_batch == 0
    This ensures the rollout buffer can be evenly divided into minibatches.
    
    Also enforces:
    - num_mini_batch <= (num_steps * num_processes)
    - minibatch_size >= MIN_MINIBATCH_SIZE (for meaningful gradient estimates)
    """
    batch_size = num_steps * num_processes
    if num_mini_batch > batch_size:
        return False
    if batch_size % num_mini_batch != 0:
        return False
    # Ensure minibatch size is large enough for meaningful gradients
    minibatch_size = batch_size // num_mini_batch
    if minibatch_size < MIN_MINIBATCH_SIZE:
        return False
    return True


def _fix_minibatch_geometry(params: Dict[str, Any], num_processes: int) -> Dict[str, Any]:
    """
    Adjust num_mini_batch to satisfy geometry constraint if needed.
    
    Strategy: Find the largest valid divisor <= original num_mini_batch
    that also ensures minibatch_size >= MIN_MINIBATCH_SIZE.
    """
    params = params.copy()
    num_steps = params.get("num_steps", 128)
    num_mini_batch = params.get("num_mini_batch", 32)
    batch_size = num_steps * num_processes
    
    # If already valid, return as-is
    if _validate_minibatch_geometry(num_steps, num_processes, num_mini_batch):
        return params
    
    # Find largest valid divisor <= original that satisfies all constraints
    for candidate in range(num_mini_batch, 0, -1):
        if batch_size % candidate == 0 and candidate <= batch_size:
            minibatch_size = batch_size // candidate
            if minibatch_size >= MIN_MINIBATCH_SIZE:
                params["num_mini_batch"] = candidate
                return params
    
    # Fallback: use 1 (full batch) if nothing else works
    # This happens when batch_size < MIN_MINIBATCH_SIZE
    params["num_mini_batch"] = 1
    return params


def build_ppo_candidates(
    search: str,
    trials: int,
    seed: int,
    num_processes: int,
    grid_shuffle: bool = True,
) -> List[Dict[str, Any]]:
    """
    Build candidate PPO hyperparameter configurations.
    
    Args:
        search: "random" or "grid"
        trials: Number of candidates to generate
        seed: Random seed for reproducibility
        num_processes: Number of parallel environments (needed for geometry check)
        grid_shuffle: Whether to shuffle grid combinations
        
    Returns:
        List of PPO hyperparameter dictionaries
    """
    rng = random.Random(seed)
    
    if search == "grid":
        candidates = _grid(PPO_SEARCH_SPACE_GRID, shuffle=grid_shuffle, rng=rng)
        candidates = candidates[:trials]
    else:
        candidates = _random_sample(rng, PPO_SEARCH_SPACE_RANDOM, trials)
    
    # Fix minibatch geometry for all candidates
    candidates = [_fix_minibatch_geometry(c, num_processes) for c in candidates]
    
    return candidates


# =============================================================================
# Evaluation Utilities
# =============================================================================

def _iqm(xs: List[float]) -> float:
    """
    Compute Interquartile Mean (IQM): mean of middle 50% of samples.
    
    IQM is more robust to outliers than mean, making it useful for RL evaluation
    where episode returns can have high variance.
    """
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
    """
    Select evaluation tasks from the experiment.
    
    Eval tasks are identified by:
    1. task_spec.eval_mode == True
    2. task_id ends with "_eval"
    
    These are held-out environments for generalization testing.
    """
    eval_tasks = []
    for task in experiment.tasks:
        task_id = getattr(task, "task_id", None)
        task_spec = getattr(task, "_task_spec", None)
        is_eval = False
        
        if task_spec is not None:
            is_eval = getattr(task_spec, "eval_mode", False)
        if isinstance(task_id, str) and task_id.endswith("_eval"):
            is_eval = True
            
        if is_eval:
            eval_tasks.append(task)
    
    # Fallback: if no eval tasks, use all tasks
    if eval_tasks:
        return eval_tasks
    return list(experiment.tasks)


def evaluate_policy_on_tasks(
    experiment,
    policy,
    summary_writer,
    episodes_per_task: int,
    objective_metric: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Evaluate policy on eval tasks and compute aggregate metrics.
    
    Args:
        experiment: The experiment object containing tasks
        policy: Trained policy to evaluate
        summary_writer: TensorBoard writer
        episodes_per_task: Number of episodes to collect per eval task
        objective_metric: "mean" or "iqm"
        
    Returns:
        per_task: List of per-task results with raw returns
        aggregates: Dict with mean_eval_return, iqm_eval_return, objective
    """
    per_task = []
    eval_tasks = _select_eval_tasks(experiment)
    
    for task in eval_tasks:
        task_id = getattr(task, "task_id", "unknown")
        
        # Use continual_eval method to run evaluation episodes
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
        
        # Truncate to exact number requested
        if len(rewards) > episodes_per_task:
            rewards = rewards[:episodes_per_task]
        
        per_task.append({
            "task_id": task_id,
            "mean": float(np.mean(rewards)) if rewards else float("nan"),
            "iqm": _iqm(rewards),
            "count": len(rewards),
            "raw_returns": rewards,
        })
    
    # Aggregate across tasks
    mean_over_tasks = float(np.nanmean([t["mean"] for t in per_task])) if per_task else float("nan")
    iqm_over_tasks = float(np.nanmean([t["iqm"] for t in per_task])) if per_task else float("nan")
    
    objective = mean_over_tasks if objective_metric == "mean" else iqm_over_tasks
    
    return per_task, {
        "mean_eval_return": mean_over_tasks,
        "iqm_eval_return": iqm_over_tasks,
        "objective": objective,
    }


# =============================================================================
# Experiment Setup Utilities
# =============================================================================

def apply_budget_override(experiment, budget_override: Optional[int]) -> None:
    """
    Override num_timesteps for TRAIN tasks only (not eval tasks).
    
    This allows running shorter experiments for hyperparameter sweeps
    while still evaluating on full eval tasks.
    """
    if budget_override is None:
        return
    
    for task in experiment.tasks:
        task_spec = getattr(task, "_task_spec", None)
        if task_spec is None:
            continue
        
        # Skip eval tasks - they don't train anyway
        if getattr(task_spec, "eval_mode", False):
            continue
        
        # Override the budget
        task_spec._num_timesteps = int(budget_override)
        
        # Ensure rolling return count is reasonable
        if hasattr(task, "_rolling_return_count"):
            task._rolling_return_count = max(1, min(task._rolling_return_count, 100))


def set_eval_mode(experiment, mode: str) -> None:
    """
    Configure evaluation mode for the experiment.
    
    - "periodic": Keep normal continual eval frequency (expensive)
    - "final_only": Disable periodic eval, only do post-training eval (cheap)
    - "none": No evaluation at all (fastest, for debugging)
    
    IMPORTANT: final_only still runs the post-training evaluation function,
    it just disables the expensive periodic evaluations during training.
    """
    if mode == "periodic":
        return
    
    if mode == "final_only":
        # Set continual testing freq to effectively infinity
        if hasattr(experiment, "_continual_testing_freq"):
            experiment._continual_testing_freq = 10**12
    elif mode == "none":
        if hasattr(experiment, "_continual_testing_freq"):
            experiment._continual_testing_freq = None
    else:
        raise ValueError(f"Unknown eval_mode: {mode}")


def normalize_ppo_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize PPO parameter names using the alias map.
    
    Converts any aliases (e.g., "lr" -> "learning_rate") to canonical names.
    """
    normalized = {}
    for key, value in params.items():
        canonical = PPO_PARAM_ALIASES.get(key, key)
        normalized[canonical] = value
    return normalized


def verify_ppo_params(params: Dict[str, Any], config_obj, strict: bool) -> bool:
    """
    Verify that PPO parameters were correctly applied to the config.
    
    In strict mode, raises ValueError if any required param wasn't applied.
    """
    params = normalize_ppo_params(params)
    
    for param_name in REQUIRED_PPO_PARAMS:
        if param_name not in params:
            continue
        
        expected = params[param_name]
        actual = getattr(config_obj, param_name, None)
        
        # Handle floating point comparison
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
    Build experiment and policy with the given PPO hyperparameters.
    
    Uses dense baseline (intervention_type="dense") for PPO tuning.
    """
    available_policies = get_available_policies()
    available_experiments = get_available_experiments()
    
    if policy_name not in available_policies:
        raise ValueError(f"Unknown policy: {policy_name}")
    if experiment_name not in available_experiments:
        raise ValueError(f"Unknown experiment: {experiment_name}")
    
    # Load experiment (LazyDict already calls the loader, so we get an Experiment directly)
    experiment = available_experiments[experiment_name]
    experiment.set_output_dir(output_dir)
    
    # Load policy struct
    policy_struct = available_policies[policy_name]
    
    # Normalize and prepare PPO config
    ppo_params = normalize_ppo_params(ppo_params)
    
    # Build config dict with dense baseline (no intervention)
    config_dict = ppo_params.copy()
    config_dict.update({
        "intervention_type": "dense",
        "intervention_params": {},
        "num_processes": num_processes,
        "use_gae": True,  # Ensure GAE is enabled when gae_lambda is tuned
    })
    
    # Create config and load parameters
    config = policy_struct.config().load_from_dict(config_dict)
    config.set_output_dir(output_dir)
    
    # Verify parameters were applied correctly
    verify_ppo_params(ppo_params, config, strict=strict_verify)
    
    # Create policy
    policy = policy_struct.policy(config, experiment.observation_space, experiment.action_spaces)
    policy.set_task_ids(experiment.task_ids)
    
    # Save the config used for this trial
    try:
        cfg_dump = {k: v for k, v in config.__dict__.items()
                    if not k.startswith("_") and isinstance(v, (int, float, str, bool, type(None)))}
    except Exception:
        cfg_dump = config_dict
    
    with open(os.path.join(output_dir, "ppo_config_used.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_dump, f, indent=2)
    
    return experiment, policy


def clear_task_registry() -> None:
    """
    Clear the global task ID registry to avoid duplicate task_id errors between trials.
    
    TaskBase.ALL_TASK_IDS tracks all created task IDs to ensure uniqueness.
    Between tuning trials, we need to reset this to allow task recreation.
    """
    TaskBase.ALL_TASK_IDS.clear()


# =============================================================================
# Trial Runner
# =============================================================================

def run_trial(
    trial_idx: int,
    args: argparse.Namespace,
    ppo_params: Dict[str, Any],
    base_dir: str,
    timestamp: str,
) -> Dict[str, Any]:
    """
    Run a single tuning trial with the given PPO hyperparameters.
    
    Returns a result dictionary with trial metadata, metrics, and paths.
    """
    trial_dir = os.path.join(base_dir, f"trial_{trial_idx:03d}")
    os.makedirs(trial_dir, exist_ok=True)
    tb_dir = os.path.join(trial_dir, "tb")
    
    # Clear task registry to avoid duplicate ID errors
    clear_task_registry()
    
    # Validate minibatch geometry
    num_steps = ppo_params.get("num_steps", 128)
    num_mini_batch = ppo_params.get("num_mini_batch", 32)
    
    if not _validate_minibatch_geometry(num_steps, args.num_processes, num_mini_batch):
        batch_size = num_steps * args.num_processes
        minibatch_size = batch_size // num_mini_batch if num_mini_batch > 0 else 0
        raise ValueError(
            f"Invalid minibatch geometry: num_steps={num_steps}, "
            f"num_processes={args.num_processes}, num_mini_batch={num_mini_batch}. "
            f"Require: (num_steps * num_processes) % num_mini_batch == 0 "
            f"AND minibatch_size >= {MIN_MINIBATCH_SIZE} (got {minibatch_size})"
        )
    
    # Build experiment and policy
    experiment, policy = build_experiment_and_policy(
        policy_name=args.policy,
        experiment_name=args.experiment,
        ppo_params=ppo_params,
        output_dir=trial_dir,
        num_processes=args.num_processes,
        strict_verify=args.strict_verify,
    )
    
    # Configure evaluation mode
    set_eval_mode(experiment, args.eval_mode)
    
    # Apply budget override to train tasks only
    apply_budget_override(experiment, args.budget_override)
    
    # Set random seeds (same base seed for fair comparison)
    base_seed = args.seed
    np.random.seed(base_seed)
    random.seed(base_seed)
    torch.manual_seed(base_seed)
    
    # Create tensorboard writer
    writer = SummaryWriter(log_dir=tb_dir)
    
    # Run training
    print(f"[Trial {trial_idx:03d}] Starting training with PPO params: {ppo_params}")
    experiment.try_run(policy, summary_writer=writer)
    
    # Run evaluation (unless eval_mode is "none")
    aggregates = {"objective": float("nan")}
    per_task = []
    
    if args.eval_mode != "none":
        print(f"[Trial {trial_idx:03d}] Running evaluation on eval tasks...")
        per_task, aggregates = evaluate_policy_on_tasks(
            experiment=experiment,
            policy=policy,
            summary_writer=writer,
            episodes_per_task=args.episodes_per_task,
            objective_metric=args.objective,
        )
    
    # Build result dictionary
    result = {
        "trial": trial_idx,
        "timestamp": timestamp,
        "status": "ok",
        "ppo_params": ppo_params,
        "seed": base_seed,
        "aggregates": aggregates,
        "per_task": per_task,
        "output_dir": trial_dir,
        "tb_dir": tb_dir,
        "experiment": args.experiment,
        "policy": args.policy,
        "num_processes": args.num_processes,
        "budget_override": args.budget_override,
        "eval_mode": args.eval_mode,
        "episodes_per_task": args.episodes_per_task,
        "objective_metric": args.objective,
    }
    
    # Log final objective
    objective_val = aggregates.get("objective", float("nan"))
    print(f"[Trial {trial_idx:03d}] Completed. Objective ({args.objective}): {objective_val:.4f}")
    
    writer.flush()
    writer.close()
    
    return result


# =============================================================================
# Output Writers
# =============================================================================

def write_results_jsonl(path: str, result: Dict[str, Any]) -> None:
    """Append a single result to the JSONL file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")


def write_leaderboard_csv(path: str, results: List[Dict[str, Any]]) -> None:
    """
    Write leaderboard CSV sorted by objective (descending).
    
    Fields: trial, status, objective, objective_metric, seed, ppo_params, output_dir
    """
    if not results:
        return
    
    # Sort by objective descending
    sorted_results = sorted(
        results,
        key=lambda x: x.get("aggregates", {}).get("objective", float("-inf")),
        reverse=True,
    )
    
    fieldnames = [
        "rank", "trial", "status", "objective", "mean_eval_return", "iqm_eval_return",
        "seed", "learning_rate", "clip_param", "entropy_coef", "value_loss_coef",
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
                "mean_eval_return": agg.get("mean_eval_return"),
                "iqm_eval_return": agg.get("iqm_eval_return"),
                "seed": r.get("seed"),
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
    """
    Write best_ppo.json with the best trial's PPO params and metadata.
    
    This file is designed to be loaded by tune_interventions.py via --ppo_params_path.
    """
    # Filter successful trials
    ok_results = [r for r in results if r.get("status") == "ok"]
    ok_results = [r for r in ok_results 
                  if not math.isnan(r.get("aggregates", {}).get("objective", float("nan")))]
    
    if not ok_results:
        # No successful trials - write empty
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"best": None, "all_trials": []}, f, indent=2)
        return
    
    # Sort by objective descending
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
            "objective_metric": best.get("objective_metric"),
            "mean_eval_return": best.get("aggregates", {}).get("mean_eval_return"),
            "iqm_eval_return": best.get("aggregates", {}).get("iqm_eval_return"),
            "output_dir": best.get("output_dir"),
        },
        "metadata": {
            "experiment": best.get("experiment"),
            "policy": best.get("policy"),
            "seed": best.get("seed"),
            "num_processes": best.get("num_processes"),
            "budget_override": best.get("budget_override"),
            "eval_mode": best.get("eval_mode"),
            "episodes_per_task": best.get("episodes_per_task"),
            "timestamp": best.get("timestamp"),
            "total_trials": len(results),
            "successful_trials": len(ok_results),
        },
        "all_successful_trials": [
            {
                "trial": r.get("trial"),
                "objective": r.get("aggregates", {}).get("objective"),
                "ppo_params": r.get("ppo_params"),
            }
            for r in sorted_results
        ],
    }
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)


# =============================================================================
# CLI Argument Parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PPO Baseline Hyperparameter Tuning (CORA-style)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python tools/tune_ppo.py --experiment procgen_3_tasks_1_cycle_5m_tuning \\
      --budget_override 100000 --trials 20 --eval_mode final_only \\
      --episodes_per_task 5 --objective mean --num_processes 1 --strict_verify
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--experiment", required=True, type=str,
        help="Name of the experiment from experiment_specs.py"
    )
    
    # Policy selection
    parser.add_argument(
        "--policy", default="ppo", type=str,
        help="Policy to tune (default: ppo)"
    )
    
    # Search configuration
    parser.add_argument(
        "--trials", default=20, type=int,
        help="Number of hyperparameter configurations to try"
    )
    parser.add_argument(
        "--search", default="random", choices=["random", "grid"], type=str,
        help="Search strategy: random sampling or grid search"
    )
    parser.add_argument(
        "--grid_shuffle", default=True, type=lambda x: str(x).lower() == "true",
        help="Shuffle grid combinations (default: True)"
    )
    
    # Budget and training
    parser.add_argument(
        "--budget_override", default=None, type=int,
        help="Override num_timesteps for train tasks (required unless --allow_full_budget)"
    )
    parser.add_argument(
        "--allow_full_budget", action="store_true",
        help="Allow running with full budget (no override required)"
    )
    parser.add_argument(
        "--num_processes", default=1, type=int,
        help="Number of parallel environments (default: 1 for safety)"
    )
    
    # Evaluation configuration
    parser.add_argument(
        "--eval_mode", default="final_only", choices=["final_only", "periodic", "none"],
        help="Evaluation mode: final_only (recommended), periodic, or none"
    )
    parser.add_argument(
        "--episodes_per_task", default=5, type=int,
        help="Number of evaluation episodes per task"
    )
    parser.add_argument(
        "--objective", default="mean", choices=["mean", "iqm"],
        help="Objective metric for selection: mean or IQM of eval returns"
    )
    
    # Output configuration
    parser.add_argument(
        "--output_root", default="runs/tuning", type=str,
        help="Root directory for tuning outputs"
    )
    
    # Reproducibility
    parser.add_argument(
        "--seed", default=0, type=int,
        help="Base random seed (same for all trials for fair comparison)"
    )
    
    # Verification
    parser.add_argument(
        "--strict_verify", action="store_true",
        help="Raise error if any PPO hyperparameter didn't apply correctly"
    )
    
    # Utility
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print generated configurations without running trials"
    )
    
    return parser.parse_args()


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    args = parse_args()
    
    # Validate arguments
    if args.budget_override is None and not args.allow_full_budget:
        raise ValueError(
            "--budget_override is required unless --allow_full_budget is set. "
            "This prevents accidentally running full-budget experiments during tuning."
        )
    
    # Set up multiprocessing
    try:
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass  # Already set
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(args.output_root, args.experiment, "ppo_tune", timestamp)
    os.makedirs(base_dir, exist_ok=True)
    
    print(f"PPO Baseline Tuning")
    print(f"=" * 60)
    print(f"Experiment: {args.experiment}")
    print(f"Policy: {args.policy}")
    print(f"Search: {args.search}")
    print(f"Trials: {args.trials}")
    print(f"Budget override: {args.budget_override}")
    print(f"Eval mode: {args.eval_mode}")
    print(f"Objective: {args.objective}")
    print(f"Output: {base_dir}")
    print(f"=" * 60)
    
    # Generate candidates
    candidates = build_ppo_candidates(
        search=args.search,
        trials=args.trials,
        seed=args.seed,
        num_processes=args.num_processes,
        grid_shuffle=args.grid_shuffle,
    )
    
    print(f"Generated {len(candidates)} PPO configurations")
    
    # Dry run mode
    if args.dry_run:
        print("\n[DRY RUN] Generated configurations:")
        for i, params in enumerate(candidates):
            print(f"\nTrial {i:03d}:")
            for k, v in sorted(params.items()):
                print(f"  {k}: {v}")
        return
    
    # Save configuration
    config_path = os.path.join(base_dir, "tuning_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": args.experiment,
            "policy": args.policy,
            "search": args.search,
            "trials": args.trials,
            "seed": args.seed,
            "budget_override": args.budget_override,
            "num_processes": args.num_processes,
            "eval_mode": args.eval_mode,
            "episodes_per_task": args.episodes_per_task,
            "objective": args.objective,
            "timestamp": timestamp,
        }, f, indent=2)
    
    # Run trials
    results_path = os.path.join(base_dir, "results.jsonl")
    all_results: List[Dict[str, Any]] = []
    
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
                "seed": args.seed,
                "aggregates": {"objective": float("nan")},
                "per_task": [],
                "output_dir": os.path.join(base_dir, f"trial_{idx:03d}"),
                "tb_dir": os.path.join(base_dir, f"trial_{idx:03d}", "tb"),
                "experiment": args.experiment,
                "policy": args.policy,
                "objective_metric": args.objective,
            }
            all_results.append(fail_result)
        
        finally:
            # Write result immediately (crash resilience)
            write_results_jsonl(results_path, all_results[-1])
    
    # Write summary outputs
    leaderboard_path = os.path.join(base_dir, "leaderboard.csv")
    write_leaderboard_csv(leaderboard_path, all_results)
    
    best_ppo_path = os.path.join(base_dir, "best_ppo.json")
    write_best_ppo_json(best_ppo_path, all_results)
    
    # Print summary
    print(f"\n{'=' * 60}")
    print(f"PPO Tuning Complete")
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
        print(f"Best objective ({args.objective}): {best_obj:.4f}")
        print(f"Best PPO params:")
        for k, v in sorted(best.get("ppo_params", {}).items()):
            print(f"  {k}: {v}")
    
    print(f"\nOutputs:")
    print(f"  Results: {results_path}")
    print(f"  Leaderboard: {leaderboard_path}")
    print(f"  Best config: {best_ppo_path}")
    print(f"\nTo use best config for intervention tuning:")
    print(f"  python tools/tune_interventions.py --ppo_params_path {best_ppo_path} ...")


if __name__ == "__main__":
    main()
