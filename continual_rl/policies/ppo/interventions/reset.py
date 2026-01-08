from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.init as init

from .base import InterventionBase


class ResetIntervention(InterventionBase):
    """
    Full reset at task boundary:
    - reinitialize all trainable weights
    - clear optimizer state (Adam moments)
    - clear rollout bookkeeping
    """

    @staticmethod
    def _reset_module_parameters(m: nn.Module) -> None:
        # only reset layers that actually have parameters we want to reinit
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.orthogonal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0.0)

    @staticmethod
    def _snapshot_params(model: nn.Module) -> Dict[str, torch.Tensor]:
        # save a copy of all trainable parameters (for proving reset happened)
        return {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }

    @staticmethod
    def _mean_abs_param_delta(model: nn.Module, before: Dict[str, torch.Tensor]) -> float:
        # average(mean(abs(delta))) across all trainable parameter tensors
        with torch.no_grad():
            total = 0.0
            count = 0
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if name not in before:
                    continue
                total += (p - before[name]).abs().mean().item()
                count += 1
            return total / max(count, 1)

    def on_task_end(self, cycle_id: int, task_run_id: int) -> None:

        # 1) snapshot params to verify reset actually changed weights
        before = self._snapshot_params(self.ctx.actor_critic)

        # 2) reset model weights
        self.ctx.actor_critic.apply(self._reset_module_parameters)

        # 3) print a stronger sanity check across ALL params
        delta = self._mean_abs_param_delta(self.ctx.actor_critic, before)

        # 4) clear optimizer state (Adam moments etc.)
        self.ctx.ppo_trainer.optimizer.state.clear()

        # 5) clear rollout bookkeeping (safe)
        self.ctx.rollout_storage.after_update()

        self.logger.info(
            "reset applied | cycle=%d task=%d mean_param_delta=%.6f",
            cycle_id,
            task_run_id,
            delta,
        )
