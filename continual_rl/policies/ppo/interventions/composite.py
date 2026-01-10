from __future__ import annotations

from typing import List

from .base import InterventionBase, InterventionContext


class CompositeIntervention(InterventionBase):
    """
    Runs multiple interventions together by forwarding all hook calls.

    Important: PPOPolicy only drains logs from the top-level intervention,
    so CompositeIntervention.drain_logs() must aggregate logs from children.
    """

    def __init__(self, ctx: InterventionContext, interventions: List[InterventionBase]):
        super().__init__(ctx)
        self._interventions = interventions

    # Task-boundary hooks
    def on_task_start(self, cycle_id: int, task_run_id: int) -> None:
        for itv in self._interventions:
            itv.on_task_start(cycle_id, task_run_id)

    def on_task_end(self, cycle_id: int, task_run_id: int) -> None:
        for itv in self._interventions:
            itv.on_task_end(cycle_id, task_run_id)

    # Optimizer-step hooks
    def before_optimizer_step(self) -> None:
        for itv in self._interventions:
            itv.before_optimizer_step()

    def after_optimizer_step(self) -> None:
        for itv in self._interventions:
            itv.after_optimizer_step()

    def on_optimizer_step(self) -> None:
        for itv in self._interventions:
            itv.on_optimizer_step()

    def drain_logs(self):
        logs = []
        # drain children first
        for itv in self._interventions:
            if hasattr(itv, "drain_logs"):
                logs.extend(itv.drain_logs())
        # drain any logs emitted directly on composite (rare)
        logs.extend(super().drain_logs())
        return logs

    def __del__(self):
        # ensure hooks cleanup if children define __del__
        for itv in getattr(self, "_interventions", []):
            try:
                itv.__del__()
            except Exception:
                pass
