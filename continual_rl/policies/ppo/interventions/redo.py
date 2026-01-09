from __future__ import annotations

from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.init as init

from .base import InterventionBase


class ReDoIntervention(InterventionBase):
    """
    ReDo (Recycling Dormant Neurons) for this repo's PPO+Procgen setup.

    What we do (actor-side only, critic untouched):
      - Track activations of the post-CNN FC layer (hidden_size=512) AFTER ReLU via a forward hook.
      - Maintain EMA of normalized mean |activation| per unit.
      - Every `update_interval` optimizer steps:
          * find units with EMA < tau (dormant)
          * recycle up to `max_recycle_frac` of units (lowest EMA first)
          * reinit incoming weights of those units in FC
          * zero outgoing weights from those units in policy head
          * clear Adam moments for modified weights/biases

    Notes:
      - Scheduling is in optimizer steps (minibatch updates), consistent with SET/GMP in this repo.
      - This intervention is intentionally minimal and does not touch the conv trunk or critic head.
    """

    def __init__(self, ctx):
        super().__init__(ctx)
        p = ctx.params or {}

        # HYPERPARAMETERS
        self.update_interval: int = int(p.get("update_interval", 5000))
        self.warmup_steps: int = int(p.get("warmup_steps", 0))
        self.tau: float = float(p.get("tau", 0.10))
        self.ema_beta: float = float(p.get("ema_beta", 0.99))
        self.max_recycle_frac: float = float(p.get("max_recycle_frac", 0.05))
        self.log_interval: int = int(p.get("log_interval", 1000))

        # INTERNAL STATE/COUNTERRS
        self._opt_step: int = 0
        self._forward_calls: int = 0

        # TARGET MODULES
        self._fc: nn.Linear = self._get_post_cnn_fc()
        self._relu: nn.Module = self._get_post_cnn_relu()
        self._head: nn.Linear = self._get_actor_head()

        self.hidden: int = int(self._fc.out_features)
        device = self.ctx.device

        # EMA stats: normalized mean abs activation per unit
        self._ema = torch.ones(self.hidden, device=device)
        self._ema_initialized = False

        # one-time proof flags
        self._logged_fc_proof = False
        self._logged_relu_proof = False

        # forward hook handles
        # - FC hook is for debugging only (pre-ReLU)
        self._hook_handle_fc = self._fc.register_forward_hook(self._fc_output_hook)
        # - ReLU hook is the real signal used for EMA (post-ReLU)
        self._hook_handle_relu = self._relu.register_forward_hook(self._activation_hook)

        self.logger.info(
            "redo init | update_interval=%d warmup_steps=%d tau=%.3f ema_beta=%.3f max_recycle_frac=%.3f hidden=%d",
            self.update_interval,
            self.warmup_steps,
            self.tau,
            self.ema_beta,
            self.max_recycle_frac,
            self.hidden,
        )

    # MODULE OUTPUT HOOK (FOR DEBUGGING)
    @torch.no_grad()
    def _fc_output_hook(self, module: nn.Module, inp, out) -> None:
        # out should be [*, hidden]
        if out is None or (not torch.is_tensor(out)):
            return
        if out.shape[-1] != self.hidden:
            return

        if self._logged_fc_proof:
            return

        x = out.detach().reshape(-1, self.hidden)
        abs_x = x.abs()

        # Only log proof when there's a non-trivial activation present
        if float(abs_x.max().item()) <= 1e-8:
            return

        self._logged_fc_proof = True

        self.logger.info(
            "redo fc proof | out_shape=%s abs_min=%.3e abs_max=%.3e abs_mean=%.3e nnz_frac=%.4f raw_min=%.3e raw_max=%.3e",
            tuple(out.shape),
            float(abs_x.min().item()),
            float(abs_x.max().item()),
            float(abs_x.mean().item()),
            float((abs_x > 0).float().mean().item()),
            float(x.min().item()),
            float(x.max().item()),
        )


    def _get_post_cnn_fc(self) -> nn.Linear:
        ac = self.ctx.actor_critic
        if not hasattr(ac, "base") or not hasattr(ac.base, "main"):
            raise AttributeError("actor_critic has no base.main (unexpected architecture).")

        # CNNBase.main = [..., Flatten(), Linear(..., hidden_size), ReLU()]
        # In this repo's model.py, Linear is at index 8, ReLU at 9. :contentReference[oaicite:6]{index=6}
        fc = ac.base.main[8]
        if not isinstance(fc, nn.Linear):
            raise TypeError(f"Expected base.main[8] to be nn.Linear, got {type(fc)}")
        return fc

    def _get_post_cnn_relu(self) -> nn.Module:
        ac = self.ctx.actor_critic
        relu = ac.base.main[9]
        # could be nn.ReLU or similar; just needs forward hook capability
        return relu

    def _get_actor_head(self) -> nn.Linear:
        ac = self.ctx.actor_critic
        if not hasattr(ac, "dist"):
            raise AttributeError("actor_critic has no attribute 'dist' (expected policy head).")
        head = ac.dist
        if not isinstance(head, nn.Linear):
            raise TypeError(f"actor_critic.dist is {type(head)} but expected torch.nn.Linear")
        return head

    # ACTIVATION TRACKING
    @torch.no_grad()
    def _activation_hook(self, module: nn.Module, inp, out) -> None:
        """
        Post-ReLU activation hook.
        Accepts any tensor whose last dim is hidden (e.g., [B,H], [T,B,H], etc.)
        Updates EMA of normalized mean abs activation per unit.
        """
        if out is None or (not torch.is_tensor(out)):
            return
        if out.shape[-1] != self.hidden:
            return

        self._forward_calls += 1

        x = out.detach().reshape(-1, self.hidden)  # [N, H]
        m = x.abs().mean(dim=0)                    # [H]
        denom = m.mean().clamp_min(1e-8)
        m_norm = m / denom

        # EMA init: avoid initializing from a degenerate all-zero batch
        if not self._ema_initialized:
            # if the whole layer is silent, skip init and wait for a non-trivial batch
            if float(m.max().item()) <= 1e-8:
                return
            self._ema.copy_(m_norm)
            self._ema_initialized = True
            self.logger.info(
                "redo ema init | forward_calls=%d ema_mean=%.4f m_max=%.3e",
                self._forward_calls,
                float(self._ema.mean().item()),
                float(m.max().item()),
            )
        else:
            self._ema.mul_(self.ema_beta).add_(m_norm, alpha=(1.0 - self.ema_beta))

        # one-time proof: do we see non-trivial post-ReLU activations?
        if not self._logged_relu_proof:
            self._logged_relu_proof = True
            abs_x = x.abs()

            if float(abs_x.max().item()) <= 1e-8:
                return

            self.logger.info(
                "redo relu proof | out_shape=%s abs_min=%.3e abs_max=%.3e abs_mean=%.3e nnz_frac=%.4f raw_min=%.3e raw_max=%.3e",
                tuple(out.shape),
                float(abs_x.min().item()),
                float(abs_x.max().item()),
                float(abs_x.mean().item()),
                float((abs_x > 0).float().mean().item()),
                float(x.min().item()),
                float(x.max().item()),
            )

    # OPTIMIZER STEP SCHEDULE
    def on_optimizer_step(self) -> None:
        self._opt_step += 1

        # periodic stats
        if self.log_interval > 0 and (self._opt_step % self.log_interval == 0):
            if self._ema_initialized:
                dormant_frac = float((self._ema < self.tau).float().mean().item())
                ema_mean = float(self._ema.mean().item())
                ema_min = float(self._ema.min().item())
                ema_p10 = float(torch.quantile(self._ema, 0.10).item())
                ema_p50 = float(torch.quantile(self._ema, 0.50).item())
                ema_p90 = float(torch.quantile(self._ema, 0.90).item())
            else:
                dormant_frac, ema_mean, ema_min, ema_p10, ema_p50, ema_p90 = 0.0, -1.0, -1.0, -1.0, -1.0, -1.0


            self.logger.info(
                "redo stats | opt_step=%d forward_calls=%d ema_init=%s dormant_frac=%.4f ema_mean=%.4f ema_min=%.4f ema_p10=%.4f ema_p50=%.4f ema_p90=%.4f"
,
                self._opt_step,
                self._forward_calls,
                str(self._ema_initialized),
                dormant_frac,
                ema_mean,
                ema_min,
                ema_p10,
                ema_p50,
                ema_p90,
            )

        if self._opt_step < self.warmup_steps:
            return
        if self.update_interval <= 0:
            return
        if (self._opt_step % self.update_interval) != 0:
            return

        self._maybe_recycle()


    # NEURON RECYCLE LOGIC
    @torch.no_grad()
    def _maybe_recycle(self) -> None:
        if not self._ema_initialized:
            self.logger.info("redo recycle skipped | opt_step=%d reason=no_ema_yet", self._opt_step)
            return

        hidden = self._ema.numel()
        max_k = max(1, int(round(self.max_recycle_frac * hidden)))

        dormant_mask = (self._ema < self.tau)
        dormant_idxs = dormant_mask.nonzero(as_tuple=False).view(-1)

        dormant_count = int(dormant_idxs.numel())
        if dormant_count == 0:
            self.logger.info("redo recycle | opt_step=%d dormant=0 -> nothing to do", self._opt_step)
            return

        # choose lowest-EMA units first, capped by max_k
        ema_vals = self._ema[dormant_idxs]
        order = torch.argsort(ema_vals)  # ascending
        chosen = dormant_idxs[order[:max_k]]
        chosen_list = chosen.tolist()

        self._recycle_units(chosen_list)

        self.logger.info(
            "redo recycle applied | opt_step=%d dormant=%d recycled=%d (cap=%d) tau=%.3f",
            self._opt_step,
            dormant_count,
            len(chosen_list),
            max_k,
            self.tau,
        )

    @torch.no_grad()
    def _recycle_units(self, unit_idxs: List[int]) -> None:
        fc = self._fc
        head = self._head
        opt = self.ctx.ppo_trainer.optimizer

        device = fc.weight.device
        idx = torch.tensor(unit_idxs, device=device, dtype=torch.long)

        # DEBUG: snapshot a few rows/cols before changes
        sample = idx[:3] if idx.numel() >= 3 else idx
        pre_fc = fc.weight[sample].detach().clone()            # [k, in_features]
        pre_head = head.weight[:, sample].detach().clone()     # [num_actions, k]
        # END DEBUG

        # --- (1) reinit incoming weights for selected units in the FC ---
        # fc.weight shape: [hidden, in_features]
        in_features = fc.weight.shape[1]
        k = idx.numel()

        # create fresh rows, orthogonal like the repo init style for Linear
        fresh = torch.empty((k, in_features), device=device, dtype=fc.weight.dtype)
        init.orthogonal_(fresh)
        fc.weight.index_copy_(0, idx, fresh)

        if fc.bias is not None:
            fc.bias.index_fill_(0, idx, 0.0)

        # --- (2) zero outgoing weights from those units in actor head ---
        # head.weight shape: [num_actions, hidden]
        head.weight.index_fill_(1, idx, 0.0)

        # DEBUG: verify changes happened (fc rows changed, head cols zeroed)
        post_fc = fc.weight[sample].detach().clone()
        post_head = head.weight[:, sample].detach().clone()

        self.logger.info(
            "redo recycle proof | fc_row_abs_delta_mean=%.4g head_col_abs_sum_after=%.4g",
            float((post_fc - pre_fc).abs().mean().item()),
            float(post_head.abs().sum().item()),
        )
        # END DEBUG

        # --- (3) clear Adam state slices for touched params ---
        # We must avoid stale exp_avg/exp_avg_sq on modified entries.
        self._zero_adam_slices_(opt, fc.weight, row_idx=idx)
        if fc.bias is not None:
            self._zero_adam_slices_(opt, fc.bias, vec_idx=idx)
        self._zero_adam_slices_(opt, head.weight, col_idx=idx)

        # DEBUG: confirm Adam moments zeroed on a small sample of recycled slices ---
        st_fc = opt.state.get(fc.weight, None)
        if st_fc is not None and "exp_avg" in st_fc:
            self.logger.info(
                "redo adam proof | fc_exp_avg_sample_abs_mean=%.4g fc_exp_avg_sq_sample_abs_mean=%.4g",
                float(st_fc["exp_avg"][sample].abs().mean().item()),
                float(st_fc["exp_avg_sq"][sample].abs().mean().item()),
            )

        st_head = opt.state.get(head.weight, None)
        if st_head is not None and "exp_avg" in st_head:
            # columns sample: take [:, sample]
            self.logger.info(
                "redo adam proof | head_exp_avg_sample_abs_mean=%.4g head_exp_avg_sq_sample_abs_mean=%.4g",
                float(st_head["exp_avg"][:, sample].abs().mean().item()),
                float(st_head["exp_avg_sq"][:, sample].abs().mean().item()),
            )
        # END DEBUG

        # --- (4) optional: nudge EMA upward for recycled units so they don't immediately re-trigger ---
        # This doesn't change weights; it just prevents pathological "recycle same unit every interval".
        self._ema[idx] = 1.0

    @staticmethod
    @torch.no_grad()
    def _zero_adam_slices_(
        opt: torch.optim.Optimizer,
        param: torch.nn.Parameter,
        row_idx: Optional[torch.Tensor] = None,
        col_idx: Optional[torch.Tensor] = None,
        vec_idx: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Zero exp_avg / exp_avg_sq for selected slices of a parameter tensor.
        Supports:
          - row slices (2D): param[row_idx, :]
          - col slices (2D): param[:, col_idx]
          - vector slices (1D): param[vec_idx]
        """
        if param not in opt.state:
            return
        st = opt.state[param]
        if "exp_avg" not in st or "exp_avg_sq" not in st:
            return

        exp_avg = st["exp_avg"]
        exp_avg_sq = st["exp_avg_sq"]

        if vec_idx is not None:
            exp_avg.index_fill_(0, vec_idx, 0.0)
            exp_avg_sq.index_fill_(0, vec_idx, 0.0)
            return

        if row_idx is not None:
            exp_avg.index_fill_(0, row_idx, 0.0)
            exp_avg_sq.index_fill_(0, row_idx, 0.0)

        if col_idx is not None:
            exp_avg.index_fill_(1, col_idx, 0.0)
            exp_avg_sq.index_fill_(1, col_idx, 0.0)

    # CLEANUP
    def __del__(self):
        try:
            if hasattr(self, "_hook_handle_fc") and self._hook_handle_fc is not None:
                self._hook_handle_fc.remove()
        except Exception:
            pass
        try:
            if hasattr(self, "_hook_handle_relu") and self._hook_handle_relu is not None:
                self._hook_handle_relu.remove()
        except Exception:
            pass
