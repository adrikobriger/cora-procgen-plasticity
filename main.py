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
    experiment.try_run(policy, summary_writer=summary_writer)
