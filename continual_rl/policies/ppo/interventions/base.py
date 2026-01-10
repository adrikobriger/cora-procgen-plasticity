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

        # logs to be forwarded to TaskBase -> TensorBoard
        self._pending_logs = []

    # Task-boundary hooks (wired via Experiment -> Policy)
    def on_task_start(self, cycle_id: int, task_run_id: int) -> None:
        pass

    def on_task_end(self, cycle_id: int, task_run_id: int) -> None:
        pass

    # Optimizer-step hooks (used for GMP/SET/ReDo)
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

    # METHODS FOR LOGGING METRICS
    def _emit_scalar(self, tag: str, value: float, timestep: Optional[int] = None) -> None:
        """
        Queue a scalar metric to be logged via TaskBase.
        If timestep is None, TaskBase will use default_timestep passed to _report_log.
        """
        self._pending_logs.append({
            "type": "scalar",
            "tag": tag,
            "value": float(value),
            **({ "timestep": int(timestep) } if timestep is not None else {})
        })

    def drain_logs(self):
        """
        Return and clear any queued logs since last drain.
        PPOPolicy.train() will call this and append results to its own logs list.
        """
        if not self._pending_logs:
            return []
        out = self._pending_logs
        self._pending_logs = []
        return out

