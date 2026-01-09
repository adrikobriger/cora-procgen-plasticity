from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch

from .base import InterventionBase


@dataclass
class _PrunableParam:
    name: str
    param: torch.nn.Parameter


class SETIntervention(InterventionBase):
    """
    Sparse Evolutionary Training (SET):

    - Maintain fixed sparsity (target_sparsity)
    - Every `update_interval` optimizer steps:
        * prune `prune_fraction` of active weights by smallest magnitude
        * regrow same number at random among inactive weights
    - Hard masks enforced each optimizer step
    - Scope: prune actor-side FC + head; exclude conv trunk and critic head; prune weights only (dim>=2)
    """

    def __init__(self, ctx):
        super().__init__(ctx)
        p = ctx.params or {}

        self.target_sparsity: float = float(p.get("target_sparsity", 0.80))
        self.update_interval: int = int(p.get("update_interval", 200))
        self.prune_fraction: float = float(p.get("prune_fraction", 0.10))
        self.warmup_steps: int = int(p.get("warmup_steps", 0))
        self.seed: int = int(p.get("seed", 0))

        # internal step counter (optimizer steps / minibatch updates)
        self._opt_step: int = 0
        self._initialized_sparse: bool = False

        self._prunable: List[_PrunableParam] = self._collect_prunable_params()
        self.log_interval: int = int(p.get("log_interval", 1000))

        # masks keyed by parameter name
        self._masks: Dict[str, torch.Tensor] = {
            item.name: torch.ones_like(item.param.data)
            for item in self._prunable
        }

        # rng for regrowth
        self._g = torch.Generator()
        self._g.manual_seed(self.seed)

        self.logger.info(
            "set init | target_sparsity=%.2f update_interval=%d prune_fraction=%.2f warmup_steps=%d prunable=%d",
            self.target_sparsity,
            self.update_interval,
            self.prune_fraction,
            self.warmup_steps,
            len(self._prunable),
        )
        self.logger.info("set prunable params: %s", [x.name for x in self._prunable])

        # initialize sparse mask immediately if no warmup
        if self.warmup_steps == 0:
            self._initialize_to_target_sparsity()
            self._initialized_sparse = True
            self._apply_masks_to_params_()

    # PARAMETER SELECTION
    def _collect_prunable_params(self) -> List[_PrunableParam]:
        ac = self.ctx.actor_critic

        prunable: List[_PrunableParam] = []
        for name, p in ac.named_parameters():
            if not p.requires_grad:
                continue

            # weights only
            if p.dim() < 2:
                continue

            # exclude conv trunk (4D params inside base.main)
            if name.startswith("base.main.") and p.dim() == 4:
                continue

            # exclude critic head
            if name.startswith("base.critic_linear."):
                continue

            prunable.append(_PrunableParam(name=name, param=p))

        if not prunable:
            raise RuntimeError("SET found no prunable parameters. Check filters.")
        return prunable

    # HELPERS FOR MASKS 
    def _apply_masks_to_params_(self) -> None:
        for item in self._prunable:
            item.param.data.mul_(self._masks[item.name])

    def _apply_masks_to_grads_(self) -> None:
        for item in self._prunable:
            if item.param.grad is None:
                continue
            item.param.grad.mul_(self._masks[item.name])

    def _current_sparsity(self) -> float:
        total = 0
        active = 0
        for item in self._prunable:
            m = self._masks[item.name]
            total += m.numel()
            active += int(m.sum().item())
        return 1.0 - (active / total)

    # INITIALIZATION TO SPARSITY
    def _initialize_to_target_sparsity(self) -> None:
        """Randomly initialize masks to achieve target sparsity (classic SET starts sparse)."""
        with torch.no_grad():
            for item in self._prunable:
                m = self._masks[item.name]
                total = m.numel()
                keep = int(round((1.0 - self.target_sparsity) * total))
                keep = max(1, min(total, keep))
                flat = m.view(-1)
                flat.zero_()
                idx = torch.randperm(total, generator=self._g, device=flat.device)[:keep]
                flat[idx] = 1.0
                self._masks[item.name] = flat.view_as(m)

        self.logger.info("set init sparse | achieved_sparsity=%.3f", self._current_sparsity())

    # PRUNING + REGROWTH
    def _count_active(self) -> int:
        return sum(int(self._masks[it.name].sum().item()) for it in self._prunable)

    def _count_total(self) -> int:
        return sum(self._masks[it.name].numel() for it in self._prunable)

    def _prune_and_regrow(self, track_flips: bool = False) -> int:
        """
        Global magnitude prune among active weights, then random regrow among inactive.
        Keeps total active weight count constant.
        """

        before_active = self._count_active()
        before_masks = None
        if track_flips:
            before_masks = {it.name: self._masks[it.name].clone() for it in self._prunable}


        with torch.no_grad():
            total_active = self._count_active()
            if total_active <= 0:
                return 0

            k = int(round(self.prune_fraction * total_active))
            if k <= 0:
                return 0
            
            # collect magnitudes of active weights globally
            mags = []
            owners: List[Tuple[str, torch.Tensor, torch.Tensor]] = []
            # owners: (name, active_indices_flat, active_magnitudes)
            for item in self._prunable:
                w = item.param.data.view(-1)
                m = self._masks[item.name].view(-1)
                active_idx = (m > 0).nonzero(as_tuple=False).view(-1)
                if active_idx.numel() == 0:
                    continue
                active_mag = w[active_idx].abs()
                mags.append(active_mag)
                owners.append((item.name, active_idx, active_mag))

            if not mags:
                return 0

            all_mags = torch.cat(mags)
            k = min(k, all_mags.numel())

            # threshold = k-th smallest magnitude (1-indexed kthvalue)
            thr = torch.kthvalue(all_mags, k).values

            # prune: mark active weights with mag < thr, then handle ties to hit k exactly
            pruned = 0
            tie_pool: List[Tuple[str, torch.Tensor]] = []  # (name, tie_indices_flat)

            for name, active_idx, active_mag in owners:
                m_flat = self._masks[name].view(-1)

                prune_mask = active_mag < thr
                if prune_mask.any():
                    idx_to_prune = active_idx[prune_mask]
                    m_flat[idx_to_prune] = 0.0
                    pruned += int(idx_to_prune.numel())

                tie_mask = active_mag == thr
                if tie_mask.any():
                    tie_pool.append((name, active_idx[tie_mask]))

            remaining = k - pruned
            if remaining > 0 and tie_pool:
                # prune some ties at random to reach exact k
                for name, tie_idx in tie_pool:
                    if remaining <= 0:
                        break
                    m_flat = self._masks[name].view(-1)
                    take = min(remaining, tie_idx.numel())
                    perm = torch.randperm(tie_idx.numel(), generator=self._g, device=tie_idx.device)[:take]
                    m_flat[tie_idx[perm]] = 0.0
                    remaining -= take

            # regrow same number of weights randomly among inactive
            regrow = k
            # gather all inactive indices globally
            inactive_owners: List[Tuple[str, torch.Tensor]] = []
            inactive_total = 0
            for item in self._prunable:
                m_flat = self._masks[item.name].view(-1)
                inactive_idx = (m_flat == 0).nonzero(as_tuple=False).view(-1)
                if inactive_idx.numel() == 0:
                    continue
                inactive_owners.append((item.name, inactive_idx))
                inactive_total += int(inactive_idx.numel())

            if inactive_total == 0:
                return 0

            regrow = min(regrow, inactive_total)
            # sample global indices by walking owners
            # (simple: regrow layer-by-layer proportionally)
            for name, inactive_idx in inactive_owners:
                if regrow <= 0:
                    break
                # proportional allocation
                alloc = int(round(regrow * (inactive_idx.numel() / inactive_total)))
                alloc = max(0, min(regrow, alloc))
                if alloc == 0:
                    continue
                m_flat = self._masks[name].view(-1)
                perm = torch.randperm(inactive_idx.numel(), generator=self._g, device=inactive_idx.device)[:alloc]
                m_flat[inactive_idx[perm]] = 1.0
                regrow -= alloc

            # if rounding left some remaining, fill greedily
            if regrow > 0:
                for name, inactive_idx in inactive_owners:
                    if regrow <= 0:
                        break
                    m_flat = self._masks[name].view(-1)
                    # recompute inactive indices after previous regrowth
                    inactive_idx2 = (m_flat == 0).nonzero(as_tuple=False).view(-1)
                    if inactive_idx2.numel() == 0:
                        continue
                    take = min(regrow, inactive_idx2.numel())
                    perm = torch.randperm(inactive_idx2.numel(), generator=self._g, device=inactive_idx2.device)[:take]
                    m_flat[inactive_idx2[perm]] = 1.0
                    regrow -= take

            # enforce weights
            self._apply_masks_to_params_()

            after_active = self._count_active()
            if before_active != after_active:
                raise RuntimeError(f"SET violated constant active count: {before_active} -> {after_active}")

            flips = 0
            if track_flips and before_masks is not None:
                for it in self._prunable:
                    flips += int((before_masks[it.name] != self._masks[it.name]).sum().item())

            return flips


    # HOOKS
    def on_task_start(self, cycle_id: int, task_run_id: int) -> None:
        # optional visibility
        self.logger.info(
            "set task start | cycle=%d task=%d sparsity=%.3f opt_step=%d",
            cycle_id, task_run_id, self._current_sparsity(), self._opt_step
        )

    def on_task_end(self, cycle_id: int, task_run_id: int) -> None:
        # ensure weights obey mask at boundaries too
        self._apply_masks_to_params_()
        self.logger.info(
            "set task end | cycle=%d task=%d sparsity=%.3f opt_step=%d",
            cycle_id, task_run_id, self._current_sparsity(), self._opt_step
        )

    def before_optimizer_step(self) -> None:
        # enforce gradient mask so pruned weights never update
        self._apply_masks_to_grads_()

    def after_optimizer_step(self) -> None:
        # enforce parameter mask so pruned weights stay zero
        self._apply_masks_to_params_()

    def on_optimizer_step(self) -> None:
        self._opt_step += 1

        # warmup: run dense, then initialize sparse once
        if (not self._initialized_sparse) and (self._opt_step >= self.warmup_steps):
            self._initialize_to_target_sparsity()
            self._initialized_sparse = True
            self._apply_masks_to_params_()
            return

        if not self._initialized_sparse:
            return

        if self.update_interval > 0 and (self._opt_step % self.update_interval == 0):
            before = self._current_sparsity()

            track = (self.log_interval > 0 and self._opt_step % self.log_interval == 0)
            flips = self._prune_and_regrow(track_flips=track)

            after = self._current_sparsity()

            # Logging
            if (self.log_interval > 0 and self._opt_step % self.log_interval == 0) or (abs(after - before) > 1e-6):
                self.logger.info(
                    "set update | opt_step=%d sparsity=%.3f->%.3f flips=%d",
                    self._opt_step, before, after, flips
                )

