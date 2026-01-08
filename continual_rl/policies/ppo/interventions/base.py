from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class InterventionContext:
    """
    Shared objects interventions may need access to.
    Keep this minimal and explicit to avoid spaghetti dependencies.
    """
    actor_critic: Any
    ppo_trainer: Any
    rollout_storage: Any
    device: Any
    logger: Any
    params: Dict[str, Any]


class InterventionBase:
    """
    Base interface for all interventions.
    Default behavior is no-op everywhere.
    """

    def __init__(self, ctx: InterventionContext):
        self.ctx = ctx

        # dedicated logger namespace for interventions
        self.logger = ctx.logger.getChild("intervention")

    # Task-boundary hooks (already wired via Experiment -> Policy)
    def on_task_start(self, cycle_id: int, task_run_id: int) -> None:
        pass

    def on_task_end(self, cycle_id: int, task_run_id: int) -> None:
        pass

    # Optimizer-step hooks (used later for GMP/SET/ReDo)
    def before_optimizer_step(self) -> None:
        pass

    def after_optimizer_step(self) -> None:
        pass

    def on_optimizer_step(self) -> None:
        """
        Called once per optimizer step (minibatch update).
        Useful for "every K steps" schedules.
        """
        pass
