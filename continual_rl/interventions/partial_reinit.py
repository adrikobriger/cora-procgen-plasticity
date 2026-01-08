from .base import InterventionBase


class PartialReinitIntervention(InterventionBase):
    def on_task_end(self, cycle_id: int, task_run_id: int) -> None:
        # TODO: implement heads-only reinit
        self.ctx.logger.info(f"[PARTIAL] (stub) cycle={cycle_id} task_run={task_run_id}")
