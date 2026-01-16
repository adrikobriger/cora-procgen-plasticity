from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from .base import InterventionBase


@dataclass
class _PrunableParam:
    name: str
    param: torch.nn.Parameter
    numel: int


class GMPIntervention(InterventionBase):
    """
    Gradual Magnitude Pruning (GMP) adapted for continual RL:

    - Prune ONLY at task boundaries in cycle 0 ("train-then-sparsify")
    - Ramp sparsity to final_sparsity across tasks_per_cycle boundaries (e.g., 6)
    - Once pruned, weights stay pruned permanently (hard masks)
    - Exclude CNN trunk (base.main) and exclude critic head (base.critic_linear)
    - Prune weights only (dim >= 2); do not prune biases / 1D params
    """

    def __init__(self, ctx):
        super().__init__(ctx)

        p = ctx.params or {}
        self.final_sparsity: float = float(p.get("final_sparsity", 0.80))
        self.tasks_per_cycle: int = int(p.get("tasks_per_cycle", 3))
        self.prune_cycle: int = int(p.get("prune_cycle", 0))  # prune only on this cycle

        # boundary pruning counter (counts only when we actually prune)
        self._boundary_prune_step: int = 0

        # masks keyed by parameter name
        self._masks: Dict[str, torch.Tensor] = {}

        # cache prunable params list (names + tensors)
        self._prunable: List[_PrunableParam] = self._collect_prunable_params()
        self.logger.info("gmp prunable params: %s", [p.name for p in self._prunable])

        # init masks to ones (dense)
        for item in self._prunable:
            self._masks[item.name] = torch.ones_like(item.param.data, device=item.param.data.device)

        self.logger.info(
            "gmp init | final_sparsity=%.2f tasks_per_cycle=%d prune_cycle=%d prunable_tensors=%d",
            self.final_sparsity,
            self.tasks_per_cycle,
            self.prune_cycle,
            len(self._prunable),
        )

    # PARAMETER SELECTION
    def _collect_prunable_params(self) -> List[_PrunableParam]:
        ac = self.ctx.actor_critic

        prunable: List[_PrunableParam] = []
        for name, p in ac.named_parameters():
            if not p.requires_grad:
                continue

            # prune weights only (skip bias / layernorm scale, etc.)
            if p.dim() < 2:
                continue

            # EXCLUDE CNN (keep post-CNN FC layer)
            # For us, conv weights are 4D; the post-CNN FC is 2D but also lives in base.main.
            if name.startswith("base.main.") and p.dim() == 4:
                continue

            # EXCLUDE critic head
            if name.startswith("base.critic_linear."):
                continue

            prunable.append(_PrunableParam(name=name, param=p, numel=p.numel()))

        if len(prunable) == 0:
            raise RuntimeError("GMP found no prunable parameters. Check name filters.")

        return prunable

    # HELPERS FOR MASK APPLICATION
    def _apply_masks_to_params_(self) -> None:
        # hard enforce zeros on weights
        for item in self._prunable:
            mask = self._masks[item.name]
            item.param.data.mul_(mask)

    def _apply_masks_to_grads_(self) -> None:
        for item in self._prunable:
            if item.param.grad is None:
                continue
            mask = self._masks[item.name]
            item.param.grad.mul_(mask)

    def _current_sparsity(self) -> float:
        total = 0
        active = 0
        for item in self._prunable:
            m = self._masks[item.name]
            total += m.numel()
            active += int(m.sum().item())
        if total == 0:
            return 0.0
        return 1.0 - (active / total)

    # TASK BOUNDARY PRUNING
    def on_task_end(self, cycle_id: int, task_run_id: int) -> None:
        # Always enforce masks at boundaries too (safety)
        self._apply_masks_to_params_()

        # Logging
        current = self._current_sparsity()
        self.logger.info(
            "gmp status | cycle=%d task=%d boundary_step=%d current_sparsity=%.3f",
            cycle_id, task_run_id, self._boundary_prune_step, current
            )

        # Only prune during prune_cycle (default cycle 0)
        if cycle_id != self.prune_cycle:
            return

        # Increment boundary prune step (only for TRAIN tasks; experiment hook already handles that)
        self._boundary_prune_step += 1

        # target sparsity ramp across tasks_per_cycle boundaries
        # after 1st boundary -> 1/tasks_per_cycle * final_sparsity
        frac = min(1.0, self._boundary_prune_step / float(self.tasks_per_cycle))
        target_sparsity = frac * self.final_sparsity

        # prune only if we need to increase sparsity
        current = self._current_sparsity()
        if target_sparsity <= current + 1e-8:
            self.logger.info(
                "gmp boundary | cycle=%d task=%d step=%d target=%.3f current=%.3f (no-op)",
                cycle_id,
                task_run_id,
                self._boundary_prune_step,
                target_sparsity,
                current,
            )
            return

        self._prune_to_target_sparsity(target_sparsity)

        # enforce weights immediately after pruning
        self._apply_masks_to_params_()

        # clear optimizer state for pruned weights?
        # simplest + safe: clear ALL optimizer state so momentum doesn't revive "near-zero" dynamics
        # but since we enforce hard masks, it's not strictly necessary.
        # We keep it minimal: do nothing here.

        new_sparsity = self._current_sparsity()
        self.logger.info(
            "gmp boundary | cycle=%d task=%d step=%d target=%.3f new=%.3f",
            cycle_id,
            task_run_id,
            self._boundary_prune_step,
            target_sparsity,
            new_sparsity,
        )

    def _prune_to_target_sparsity(self, target_sparsity: float) -> None:
        # compute total and desired active count
        total = sum(item.numel for item in self._prunable)
        desired_active = int(round((1.0 - target_sparsity) * total))

        # current active
        current_active = 0
        for item in self._prunable:
            current_active += int(self._masks[item.name].sum().item())

        to_prune = current_active - desired_active
        if to_prune <= 0:
            return

        # gather magnitudes of currently-active weights
        mags: List[torch.Tensor] = []
        owners: List[Tuple[str, torch.Tensor]] = []  # (name, flat_active_mags)
        for item in self._prunable:
            w = item.param.data
            m = self._masks[item.name]
            active = (m > 0)
            if active.any():
                flat = w[active].abs().flatten()
                mags.append(flat)
                owners.append((item.name, flat))

        if len(mags) == 0:
            return

        all_mags = torch.cat(mags)
        if to_prune >= all_mags.numel():
            # extreme case: prune everything (shouldn't happen if target < 1.0)
            for item in self._prunable:
                self._masks[item.name].zero_()
            return

        # find global threshold for pruning the smallest 'to_prune' active weights
        # kthvalue is 1-indexed in torch
        kth = to_prune
        threshold = torch.kthvalue(all_mags, kth).values.item()

        # prune weights with magnitude <= threshold, but make sure we prune exactly 'to_prune'
        # To do exact pruning, we do a second pass using sorting indices.
        # Build a global list of (mag, name, index_in_tensor_flat_active)
        # For simplicity and speed, do approximate exactness:
        #   prune <= threshold then, if over-pruned, unprune some = threshold ties.
        # This is fine for research use and stable.

        pruned_count = 0
        tie_candidates: List[Tuple[str, torch.Tensor]] = []

        for item in self._prunable:
            w = item.param.data
            m = self._masks[item.name]
            active = (m > 0)
            if not active.any():
                continue
            abs_w = w.abs()

            prune_mask = active & (abs_w < threshold)
            if prune_mask.any():
                m[prune_mask] = 0.0
                pruned_count += int(prune_mask.sum().item())

            tie_mask = active & (abs_w == threshold)
            if tie_mask.any():
                tie_candidates.append((item.name, tie_mask))

        remaining = to_prune - pruned_count
        if remaining <= 0:
            return

        # prune a subset of ties to reach exact count (safe for any tensor dimensionality)
        for name, tie_mask in tie_candidates:
            if remaining <= 0:
                break
            m = self._masks[name]
            idx = tie_mask.nonzero(as_tuple=False)
            if idx.shape[0] == 0:
                continue
            take = min(remaining, idx.shape[0])
            # Deterministic: prune the first `take` indices.
            for i in range(take):
                m[tuple(idx[i].tolist())] = 0.0
            remaining -= take

        # If we still haven't pruned enough (should be rare), warn rather than silently under-pruning.
        if remaining > 0:
            self.logger.warning(
                'gmp tie-prune underflow | remaining=%d after processing all tie candidates (target_to_prune=%d)',
                remaining,
                to_prune,
            )

    # OPTIMIZER STEP HOOKS
    def before_optimizer_step(self) -> None:
        # enforce gradient masking so pruned weights never update
        self._apply_masks_to_grads_()

    def after_optimizer_step(self) -> None:
        # enforce parameter masking so pruned weights stay zero
        self._apply_masks_to_params_()
