#!/usr/bin/env python3
"""
PURPOSE:
    Tune PPO *base* hyperparameters on the **dense** baseline (no intervention),
    then freeze them for intervention-specific tuning.

"Bulletproof" upgrades (CORA-aligned):
  - Default objective = IQM (robust to outliers)
  - Default eval episodes/task = 10 (CORA uses E=10 eval episodes)
  - Optional multi-seed evaluation per trial (recommended; CORA reports across seeds)
  - Optional override of continual testing frequency for periodic eval

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
MIN_MINIBATCH_SIZE = 32


def _validate_minibatch_geometry(num_steps: int, num_processes: int, num_mini_batch: int) -> bool:
    """
    Validate PPO minibatch geometry constraint.

    PPO requires: (num_steps * num_processes) % num_mini_batch == 0
    - minibatch_size >= MIN_MINIBATCH_SIZE
    """
    batch_size = num_steps * num_processes
    if num_mini_batch > batch_size:
        return False
    if batch_size % num_mini_batch != 0:
        return False
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

    if _validate_minibatch_geometry(num_steps, num_processes, num_mini_batch):
        return params

    for candidate in range(num_mini_batch, 0, -1):
        if batch_size % candidate == 0 and candidate <= batch_size:
            minibatch_size = batch_size // candidate
            if minibatch_size >= MIN_MINIBATCH_SIZE:
                params["num_mini_batch"] = candidate
                return params

    params["num_mini_batch"] = 1
    return params


def build_ppo_candidates(
    search: str,
    trials: int,
    seed: int,
    num_processes: int,
    grid_shuffle: bool = True,
) -> List[Dict[str, Any]]:
    """Build candidate PPO hyperparameter configurations."""
    rng = random.Random(seed)

    if search == "grid":
        candidates = _grid(PPO_SEARCH_SPACE_GRID, shuffle=grid_shuffle, rng=rng)
        candidates = candidates[:trials]
    else:
        candidates = _random_sample(rng, PPO_SEARCH_SPACE_RANDOM, trials)

    candidates = [_fix_minibatch_geometry(c, num_processes) for c in candidates]
    return candidates


def _iqm(xs: List[float]) -> float:
    """Interquartile mean: mean of the middle 50% of samples."""
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


def _select_eval_tasks(experiment) -> List[Any]:
    """
    Select evaluation tasks from the experiment.

    Eval tasks are identified by:
      1) task_spec.eval_mode == True
      2) task_id ends with "_eval"

    CORA uses test/unseen environments when available.
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

    if eval_tasks:
        return eval_tasks
    return list(experiment.tasks)


def _extract_rewards_from_eval_info(info: Any) -> List[float]:
    """
    Best-effort extraction of episode returns from whatever continual_eval yields.

    We support common formats:
      - (reward_list, metrics)
      - reward_list
      - dict containing 'episode_returns' / 'returns' / 'reward'
      - numpy arrays
    """
    rewards: List[float] = []

    if info is None:
        return rewards

    # If it's a tuple like (reward_list, metrics)
    if isinstance(info, tuple) and len(info) == 2:
        info = info[0]

    # If dict
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

    # List/tuple/ndarray
    if isinstance(info, (list, tuple, np.ndarray)):
        rewards.extend([float(x) for x in info])
        return rewards

    # Scalar
    if isinstance(info, (int, float)):
        rewards.append(float(info))
        return rewards

    return rewards


def evaluate_policy_on_tasks(
    experiment,
    policy,
    summary_writer,
    episodes_per_task: int,
    objective_metric: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Evaluate policy on eval tasks and compute aggregate metrics.

    Returns:
      per_task: [{task_id, mean, iqm, count, raw_returns}]
      aggregates: {mean_eval_return, iqm_eval_return, objective}
    """
    per_task: List[Dict[str, Any]] = []
    eval_tasks = _select_eval_tasks(experiment)

    for task in eval_tasks:
        task_id = getattr(task, "task_id", "unknown")

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
                # If eval is flaky for one step, don't kill the whole trial; just continue.
                continue

        rewards = rewards[:episodes_per_task]

        per_task.append({
            "task_id": task_id,
            "mean": float(np.mean(rewards)) if rewards else float("nan"),
            "iqm": _iqm(rewards),
            "count": len(rewards),
            "raw_returns": rewards,
        })

    mean_over_tasks = _nanmean([t["mean"] for t in per_task]) if per_task else float("nan")
    iqm_over_tasks = _nanmean([t["iqm"] for t in per_task]) if per_task else float("nan")
    objective = mean_over_tasks if objective_metric == "mean" else iqm_over_tasks

    return per_task, {
        "mean_eval_return": mean_over_tasks,
        "iqm_eval_return": iqm_over_tasks,
        "objective": objective,
    }


def apply_budget_override(experiment, budget_override: Optional[int]) -> None:
    """Override num_timesteps for TRAIN tasks only (not eval tasks)."""
    if budget_override is None:
        return

    for task in experiment.tasks:
        task_spec = getattr(task, "_task_spec", None)
        if task_spec is None:
            continue
        if getattr(task_spec, "eval_mode", False):
            continue

        task_spec._num_timesteps = int(budget_override)

        if hasattr(task, "_rolling_return_count"):
            task._rolling_return_count = max(1, min(task._rolling_return_count, 100))


def set_eval_mode(experiment, mode: str, continual_testing_freq: Optional[int] = None) -> None:
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


def normalize_ppo_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize PPO parameter names using the alias map."""
    normalized: Dict[str, Any] = {}
    for key, value in params.items():
        canonical = PPO_PARAM_ALIASES.get(key, key)
        normalized[canonical] = value
    return normalized


def verify_ppo_params(params: Dict[str, Any], config_obj, strict: bool) -> bool:
    """Verify PPO parameters were correctly applied to the config."""
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

    config = policy_struct.config().load_from_dict(config_dict)
    config.set_output_dir(output_dir)

    verify_ppo_params(ppo_params, config, strict=strict_verify)

    policy = policy_struct.policy(config, experiment.observation_space, experiment.action_spaces)
    policy.set_task_ids(experiment.task_ids)

    try:
        cfg_dump = {k: v for k, v in config.__dict__.items()
                    if not k.startswith("_") and isinstance(v, (int, float, str, bool, type(None)))}
    except Exception:
        cfg_dump = config_dict

    with open(os.path.join(output_dir, "ppo_config_used.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_dump, f, indent=2)

    return experiment, policy


def clear_task_registry() -> None:
    """Clear the global task ID registry to avoid duplicate task_id errors between trials."""
    TaskBase.ALL_TASK_IDS.clear()


def _set_global_seeds(seed: int, deterministic_torch: bool = False) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Optional determinism (can slow things down, but helps reproducibility)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _parse_seeds_arg(seeds_arg: Optional[str], fallback_seed: int) -> List[int]:
    """
    Parse seeds argument:
      --seeds "0,1,2"  -> [0,1,2]
      --seeds "0"      -> [0]
    If None, fall back to [fallback_seed].
    """
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
) -> Dict[str, Any]:
    """
    Run a single tuning trial. If multiple seeds are provided, run one full
    train+eval per seed and aggregate metrics across seeds.
    """
    trial_dir = os.path.join(base_dir, f"trial_{trial_idx:03d}")
    os.makedirs(trial_dir, exist_ok=True)

    # Validate minibatch geometry once (seed-independent)
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

    seeds = _parse_seeds_arg(args.seeds, args.seed)

    print(f"[Trial {trial_idx:03d}] PPO params: {ppo_params}")
    print(f"[Trial {trial_idx:03d}] Seeds: {seeds}")

    seed_runs: List[Dict[str, Any]] = []

    for sidx, seed in enumerate(seeds):
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

        set_eval_mode(experiment, args.eval_mode, continual_testing_freq=args.continual_testing_freq)
        apply_budget_override(experiment, args.budget_override)

        writer = SummaryWriter(log_dir=tb_dir)

        print(f"[Trial {trial_idx:03d} | Seed {seed}] Starting training...")
        experiment.try_run(policy, summary_writer=writer)

        aggregates = {"objective": float("nan")}
        per_task: List[Dict[str, Any]] = []

        if args.eval_mode != "none":
            print(f"[Trial {trial_idx:03d} | Seed {seed}] Evaluating ({args.episodes_per_task} eps/task)...")
            per_task, aggregates = evaluate_policy_on_tasks(
                experiment=experiment,
                policy=policy,
                summary_writer=writer,
                episodes_per_task=args.episodes_per_task,
                objective_metric=args.objective,
            )

        writer.flush()
        writer.close()

        seed_runs.append({
            "seed": seed,
            "aggregates": aggregates,
            "per_task": per_task,
            "output_dir": seed_dir,
            "tb_dir": tb_dir,
        })

        obj_val = aggregates.get("objective", float("nan"))
        print(f"[Trial {trial_idx:03d} | Seed {seed}] Done. Objective ({args.objective})={obj_val:.4f}")

    # Aggregate across seeds (for ranking)
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
        "budget_override": args.budget_override,
        "eval_mode": args.eval_mode,
        "episodes_per_task": args.episodes_per_task,
        "objective_metric": args.objective,
        "continual_testing_freq": args.continual_testing_freq,
    }

    print(
        f"[Trial {trial_idx:03d}] Aggregated objective ({args.objective}) = "
        f"{objective_mean:.4f} ± {objective_std:.4f} (stderr={objective_stderr:.4f})"
    )
    return result


def write_results_jsonl(path: str, result: Dict[str, Any]) -> None:
    """Append a single result to the JSONL file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")


def write_leaderboard_csv(path: str, results: List[Dict[str, Any]]) -> None:
    """Write leaderboard CSV sorted by aggregated objective (descending)."""
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
    Backwards-compatible: still provides best.ppo_params.
    """
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
            "output_dir": best.get("output_dir"),
        },
        "metadata": {
            "experiment": best.get("experiment"),
            "policy": best.get("policy"),
            "seed": best.get("seeds", [None])[0],  # keep old field shape-ish
            "seeds": best.get("seeds"),
            "num_processes": best.get("num_processes"),
            "budget_override": best.get("budget_override"),
            "eval_mode": best.get("eval_mode"),
            "episodes_per_task": best.get("episodes_per_task"),
            "continual_testing_freq": best.get("continual_testing_freq"),
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
        epilog="""
Example usage:
  python tools/tune_ppo.py --experiment procgen_3_tasks_1_cycle_5m_tuning \\
      --budget_override 100000 --trials 50 --eval_mode final_only \\
      --episodes_per_task 10 --objective iqm --num_processes 1 \\
      --seeds "0,1,2" --strict_verify

Notes:
  - CORA describes evaluating E=10 episodes at evaluation points, and aggregating across seeds.
  - For extra stability when selecting the final PPO baseline, consider --episodes_per_task 25 or 50.
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
    parser.add_argument("--grid_shuffle", default=True, type=lambda x: str(x).lower() == "true",
                        help="Shuffle grid combinations (default: True)")

    parser.add_argument("--budget_override", default=None, type=int,
                        help="Override num_timesteps for train tasks (required unless --allow_full_budget)")
    parser.add_argument("--allow_full_budget", action="store_true",
                        help="Allow running with full budget (no override required)")
    parser.add_argument("--num_processes", default=1, type=int,
                        help="Number of parallel environments (default: 1)")

    parser.add_argument("--eval_mode", default="final_only",
                        choices=["final_only", "periodic", "none"],
                        help="Evaluation mode: final_only (recommended), periodic, or none")
    parser.add_argument("--continual_testing_freq", default=None, type=int,
                        help="If eval_mode=periodic, override experiment._continual_testing_freq (timesteps)")

    # CORA uses E=10 evaluation episodes; default to 10 here.
    parser.add_argument("--episodes_per_task", default=10, type=int,
                        help="Number of evaluation episodes per task (default: 10)")

    # Default to IQM for robustness.
    parser.add_argument("--objective", default="iqm", choices=["mean", "iqm"],
                        help="Objective metric for selection: mean or IQM (default: iqm)")

    parser.add_argument("--output_root", default="runs/tuning", type=str,
                        help="Root directory for tuning outputs")

    # Candidate generation seed (and fallback single-seed run if --seeds omitted)
    parser.add_argument("--seed", default=0, type=int,
                        help="Base seed (used for candidate generation; also used if --seeds not provided)")

    # Multi-seed trial evaluation (recommended)
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

    if args.budget_override is None and not args.allow_full_budget:
        raise ValueError(
            "--budget_override is required unless --allow_full_budget is set. "
            "This prevents accidentally running full-budget experiments during tuning."
        )

    try:
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(args.output_root, args.experiment, "ppo_tune", timestamp)
    os.makedirs(base_dir, exist_ok=True)

    print("PPO Baseline Tuning")
    print("=" * 60)
    print(f"Experiment: {args.experiment}")
    print(f"Policy: {args.policy}")
    print(f"Search: {args.search}")
    print(f"Trials: {args.trials}")
    print(f"Budget override: {args.budget_override}")
    print(f"Eval mode: {args.eval_mode}")
    print(f"Objective: {args.objective}")
    print(f"Episodes/task: {args.episodes_per_task}")
    print(f"Seeds per trial: {args.seeds}")
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

    if args.dry_run:
        print("\n[DRY RUN] Generated configurations:")
        for i, params in enumerate(candidates):
            print(f"\nTrial {i:03d}:")
            for k, v in sorted(params.items()):
                print(f"  {k}: {v}")
        return

    config_path = os.path.join(base_dir, "tuning_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": args.experiment,
            "policy": args.policy,
            "search": args.search,
            "trials": args.trials,
            "seed": args.seed,
            "seeds": args.seeds,
            "budget_override": args.budget_override,
            "num_processes": args.num_processes,
            "eval_mode": args.eval_mode,
            "continual_testing_freq": args.continual_testing_freq,
            "episodes_per_task": args.episodes_per_task,
            "objective": args.objective,
            "deterministic_torch": args.deterministic_torch,
            "timestamp": timestamp,
        }, f, indent=2)

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
                "seeds": args.seeds,
                "aggregates": {"objective": float("nan")},
                "seed_runs": [],
                "output_dir": os.path.join(base_dir, f"trial_{idx:03d}"),
                "experiment": args.experiment,
                "policy": args.policy,
                "objective_metric": args.objective,
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
