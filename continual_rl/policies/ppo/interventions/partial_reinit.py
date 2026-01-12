from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.init as init

from .base import InterventionBase


class PartialReinitIntervention(InterventionBase):
    """
    Partial reinitialization (per project slides):
    - at task boundary, reinitialize ONLY the policy head
    - keep the feature extractor/backbone intact
    - clear optimizer state for the reinitialized params (avoid stale Adam moments)
    """

    @staticmethod
    def _snapshot_params(module: nn.Module) -> Dict[str, torch.Tensor]:
        # Save a copy of all trainable parameters (for proving reinit happened)
        return {
            name: p.detach().clone()
            for name, p in module.named_parameters()
            if p.requires_grad
        }

    @staticmethod
    def _mean_abs_param_delta(module: nn.Module, before: Dict[str, torch.Tensor]) -> float:
        # Method to verify that reinit actually changed weights
        with torch.no_grad():
            total = 0.0
            count = 0
            for name, p in module.named_parameters():
                if not p.requires_grad:
                    continue
                if name not in before:
                    continue
                total += (p - before[name]).abs().mean().item()
                count += 1
            return total / max(count, 1)

    @staticmethod
    def _reset_linear(layer: nn.Linear) -> None:
        init.orthogonal_(layer.weight)
        if layer.bias is not None:
            init.constant_(layer.bias, 0.0)

    def on_task_end(self, cycle_id: int, task_run_id: int) -> None:
        ac = self.ctx.actor_critic

        if not hasattr(ac, "dist"):
            raise AttributeError("actor_critic has no attribute 'dist' (expected policy head).")

        head = ac.dist
        if not isinstance(head, nn.Linear):
            raise TypeError(f"actor_critic.dist is {type(head)} but expected torch.nn.Linear")

        # 1) snapshot head params to verify reinit happened
        before = self._snapshot_params(head)

        # 2) reinitialize ONLY the policy head
        self._reset_linear(head)

        # 3) clear optimizer state for ONLY these parameters
        opt = self.ctx.ppo_trainer.optimizer
        for p in head.parameters():
            if p in opt.state:
                opt.state[p].clear()

        # 4) rollout bookkeeping
        self.ctx.rollout_storage.after_update()

        # 5) log a sanity metric
        delta = self._mean_abs_param_delta(head, before)
        self.logger.info(
            "partial reinit applied | cycle=%d task=%d head_mean_param_delta=%.6f",
            cycle_id,
            task_run_id,
            delta,
        )
