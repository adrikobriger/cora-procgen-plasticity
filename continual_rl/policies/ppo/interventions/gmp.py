from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import math

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

    - Prune using a global step schedule (Zhu & Gupta cubic) across the full run
    - Pruning is NOT gated by cycles; it is timestep-based and consistent across 1-cycle/2-cycle runs
    - Once pruned, weights stay pruned permanently (hard masks)
    - Exclude CNN trunk (base.main) and exclude critic head (base.critic_linear)
    - Prune weights only (dim >= 2); do not prune biases / 1D params
    """

    def __init__(self, ctx):
        super().__init__(ctx)

        p = ctx.params or {}
        self.final_sparsity: float = float(p.get("final_sparsity", 0.80))
        self.tstart_frac: float = float(p.get("tstart_frac", 0.05))
        self.tend_frac: float = float(p.get("tend_frac", 0.80))
        self.pruning_freq_steps: int = int(p.get("pruning_freq_steps", 500))

        # Backward compatibility: accept prune_cycle but ignore it.
        if "prune_cycle" in p:
            self.logger.warning(
                "gmp deprecation | prune_cycle is ignored in global-step GMP schedule."
            )

        self.total_train_steps: int = int(p.get("total_train_steps", 0) or 0)

        self._validate_schedule_params()
        self._tstart: int = int(math.floor(self.tstart_frac * self.total_train_steps))
        self._tend: int = int(math.floor(self.tend_frac * self.total_train_steps))
        if self._tend <= self._tstart:
            raise ValueError(
                "GMP schedule invalid: tend <= tstart (tstart=%d, tend=%d). "
                "Check tstart_frac/tend_frac and total_train_steps."
                % (self._tstart, self._tend)
            )

        self._global_step: int = 0
        self._pruning_activated: bool = False
        self._logged_guidance: bool = False
        self._last_target_sparsity: float = 0.0

        # masks keyed by parameter name
        self._masks: Dict[str, torch.Tensor] = {}

        # cache prunable params list (names + tensors)
        self._prunable: List[_PrunableParam] = self._collect_prunable_params()
        self.logger.info("gmp prunable params: %s", [p.name for p in self._prunable])

        # init masks to ones (dense)
        for item in self._prunable:
            self._masks[item.name] = torch.ones_like(item.param.data, device=item.param.data.device)

        self.logger.info(
            "gmp init | final_sparsity=%.2f tstart_frac=%.3f tend_frac=%.3f freq=%d total_train_steps=%d prunable_tensors=%d",
            self.final_sparsity,
            self.tstart_frac,
            self.tend_frac,
            self.pruning_freq_steps,
            self.total_train_steps,
            len(self._prunable),
        )

    def _validate_schedule_params(self) -> None:
        if not (0.0 <= self.tstart_frac < self.tend_frac <= 1.0):
            raise ValueError(
                "GMP schedule invalid: require 0 <= tstart_frac < tend_frac <= 1. "
                f"Got tstart_frac={self.tstart_frac}, tend_frac={self.tend_frac}."
            )
        if self.pruning_freq_steps <= 0:
            raise ValueError("GMP schedule invalid: pruning_freq_steps must be > 0.")
        if not (0.0 < self.final_sparsity < 1.0):
            raise ValueError("GMP schedule invalid: final_sparsity must be in (0, 1).")
        if self.total_train_steps <= 0:
            raise ValueError(
                "GMP requires total_train_steps in ctx.params (computed from run budget)."
            )

    def _pruning_active(self, step: int) -> bool:
        return self._tstart <= step <= self._tend

    def _target_sparsity(self, step: int) -> float:
        if step <= self._tstart:
            return 0.0
        if step >= self._tend:
            return float(self.final_sparsity)
        p = (step - self._tstart) / float(self._tend - self._tstart)
        p = max(0.0, min(1.0, p))
        return float(self.final_sparsity * (1.0 - (1.0 - p) ** 3))

    def on_task_start(self, cycle_id: int, task_run_id: int) -> None:
        active = self._pruning_active(self._global_step)
        self.logger.info(
            "gmp task start | global_step=%d tstart=%d tend=%d active=%s",
            self._global_step,
            self._tstart,
            self._tend,
            str(active),
        )
        if active and not self._logged_guidance:
            self.logger.info(
                "gmp note | expect achieved_sparsity to approach final_sparsity=%.3f by tend",
                self.final_sparsity,
            )
            self._logged_guidance = True

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

    # TASK BOUNDARY LOGGING
    def on_task_end(self, cycle_id: int, task_run_id: int) -> None:
        # Always enforce masks at boundaries too (safety)
        self._apply_masks_to_params_()

        current = self._current_sparsity()
        active = self._pruning_active(self._global_step)
        target = self._target_sparsity(self._global_step)

        self.logger.info(
            "gmp status | global_step=%d active=%s target=%.3f achieved=%.3f",
            self._global_step,
            str(active),
            target,
            current,
        )

        try:
            self._emit_scalar("gmp/target_sparsity", float(target))
            self._emit_scalar("gmp/achieved_sparsity", float(current))
            self._emit_scalar("gmp/pruning_active", 1.0 if active else 0.0)
        except Exception:
            pass

        if self._global_step >= self.total_train_steps and not self._pruning_activated:
            raise ValueError(
                "GMP pruning never activated during run. "
                "Check total_train_steps or tstart_frac/tend_frac."
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

    def on_optimizer_step(self) -> None:
        self._global_step += 1
        active = self._pruning_active(self._global_step)
        if not active:
            return
        if (self._global_step % self.pruning_freq_steps) != 0:
            return

        target = self._target_sparsity(self._global_step)
        self._last_target_sparsity = target

        current = self._current_sparsity()
        if target <= current + 1e-8:
            return

        self._prune_to_target_sparsity(target)
        self._apply_masks_to_params_()
        self._pruning_activated = True

        new_sparsity = self._current_sparsity()
        self.logger.info(
            "gmp prune | global_step=%d target=%.3f achieved=%.3f",
            self._global_step,
            target,
            new_sparsity,
        )

        try:
            self._emit_scalar("gmp/target_sparsity", float(target))
            self._emit_scalar("gmp/achieved_sparsity", float(new_sparsity))
            self._emit_scalar("gmp/pruning_active", 1.0)
        except Exception:
            pass
