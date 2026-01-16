import sys
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
        if hasattr(policy, "_intervention") and policy._intervention is not None:
            policy._intervention.ctx.params["total_train_timesteps"] = int(total_train_timesteps)
    except Exception:
        pass

    experiment.try_run(policy, summary_writer=summary_writer)
