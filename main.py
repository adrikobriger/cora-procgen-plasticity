import sys
import math
from torch import multiprocessing
from torch.utils.tensorboard.writer import SummaryWriter
from continual_rl.utils.argparse_manager import ArgparseManager
import logging
logging.basicConfig(level=logging.INFO)

# ADDED: Suppress gym warnings
import os, warnings
os.environ["GYM_DISABLE_WARNINGS"] = "1"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# END ADDED


if __name__ == "__main__":

    # Pytorch multiprocessing requires either forkserver or spawn.
    try:
        multiprocessing.set_start_method("spawn")
    except ValueError as e:
        # Windows doesn't support forking, so fall back to spawn instead
        assert "cannot find context" in str(e)
        multiprocessing.set_start_method("spawn")

    experiment, policy = ArgparseManager.parse(sys.argv[1:])

    if experiment is None:
        raise RuntimeError("No experiment started. Most likely there is no new run to start.")

    summary_writer = SummaryWriter(log_dir=experiment.output_dir)

    # Provide total training budget for relative intervention schedules (e.g., SET)
    def _is_eval_task(task) -> bool:
        if getattr(task, "eval_mode", False):
            return True
        if hasattr(task, "_task_spec") and getattr(task._task_spec, "eval_mode", False):
            return True
        task_id = getattr(task, "task_id", "") or ""
        return task_id.endswith("_eval")

    def _task_timesteps(task) -> int:
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

    try:
        train_tasks = [t for t in experiment.tasks if not _is_eval_task(t)]
        if not train_tasks:
            train_tasks = list(experiment.tasks)
        cycle_count = getattr(experiment, "_cycle_count", 1) or 1
        total_train_timesteps = sum(_task_timesteps(t) for t in train_tasks) * int(cycle_count)
        total_train_steps = None
        try:
            cfg = getattr(policy, "_config", None)
            if cfg is not None:
                num_steps = int(getattr(cfg, "num_steps", 0) or 0)
                num_processes = int(getattr(cfg, "num_processes", 0) or 0)
                num_mini_batch = int(getattr(cfg, "num_mini_batch", 0) or 0)
                ppo_epoch = int(getattr(cfg, "ppo_epoch", 0) or 0)
                denom = max(1, num_steps * num_processes)
                rollouts = int(math.ceil(total_train_timesteps / float(denom))) if total_train_timesteps > 0 else 0
                if rollouts > 0 and num_mini_batch > 0 and ppo_epoch > 0:
                    total_train_steps = rollouts * ppo_epoch * num_mini_batch
        except Exception:
            total_train_steps = None
        if hasattr(policy, "_intervention") and policy._intervention is not None:
            policy._intervention.ctx.params["total_train_timesteps"] = int(total_train_timesteps)
            policy._intervention.ctx.params["train_tasks_per_cycle"] = int(len(train_tasks))
            policy._intervention.ctx.params["num_cycles"] = int(cycle_count)
            if total_train_steps is not None:
                policy._intervention.ctx.params["total_train_steps"] = int(total_train_steps)
    except Exception:
        pass

    experiment.try_run(policy, summary_writer=summary_writer)

    import os, random
    import numpy as np
    import torch

    def _set_global_seeds(seed: int, deterministic_torch: bool = False) -> None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    _seed_env = os.getenv("CONTINUAL_RL_SEED")
    if _seed_env is not None and _seed_env.strip() != "":
        seed = int(_seed_env)
        _set_global_seeds(seed)
        print(f"[main.py] Using CONTINUAL_RL_SEED={seed}")