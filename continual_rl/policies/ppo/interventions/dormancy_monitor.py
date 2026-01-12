from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from .base import InterventionBase


class DormancyMonitorIntervention(InterventionBase):
    """
    Logs neuron dormancy (fraction of units with low activation) for any run.

    Definition matches ReDo's EMA-normalized mean-abs activation on post-FC ReLU.
    A unit is "dormant" if EMA(unit) < tau.
    """

    def __init__(self, ctx):
        super().__init__(ctx)
        p: Dict[str, Any] = ctx.params or {}

        # Use dedicated keys to avoid clashing with other interventions' params
        self.tau: float = float(p.get("dormancy_tau", 0.00001))
        self.ema_beta: float = float(p.get("dormancy_ema_beta", 0.99))
        self.log_interval: int = int(p.get("dormancy_log_interval", 1000))

        self._opt_step: int = 0
        self._forward_calls: int = 0

        # locate the same layer ReDo uses: base.main[8] = FC, base.main[9] = ReLU
        ac = self.ctx.actor_critic
        if not hasattr(ac, "base") or not hasattr(ac.base, "main"):
            raise AttributeError("actor_critic has no base.main (unexpected architecture).")

        relu = ac.base.main[9]
        if not isinstance(relu, nn.Module):
            raise TypeError(f"Expected base.main[9] to be nn.Module, got {type(relu)}")

        # infer hidden size by looking at FC out_features (base.main[8])
        fc = ac.base.main[8]
        if not isinstance(fc, nn.Linear):
            raise TypeError(f"Expected base.main[8] to be nn.Linear, got {type(fc)}")
        self.hidden = int(fc.out_features)

        device = self.ctx.device
        self._ema = torch.ones(self.hidden, device=device)
        self._ema_initialized = False

        # NEW: Track raw activation statistics
        self._activation_mean_ema = torch.zeros(1, device=device)
        self._activation_max_ema = torch.zeros(1, device=device)
        self._activation_std_ema = torch.zeros(1, device=device)
        self._activation_stats_initialized = False

        self._hook_handle_relu = relu.register_forward_hook(self._activation_hook)

        self.logger.info(
            "dormancy monitor init | tau=%.4g ema_beta=%.3f log_interval=%d hidden=%d",
            self.tau, self.ema_beta, self.log_interval, self.hidden
        )

    @torch.no_grad()
    def _activation_hook(self, module: nn.Module, inp, out) -> None:
        if out is None or (not torch.is_tensor(out)):
            return
        if out.shape[-1] != self.hidden:
            return

        self._forward_calls += 1

        x = out.detach().reshape(-1, self.hidden)  # [N, H]
        m = x.abs().mean(dim=0)                    # [H]
        denom = m.mean().clamp_min(1e-8)
        m_norm = m / denom

        # NEW: Track raw activation statistics
        batch_mean = x.abs().mean().item()
        batch_max = x.abs().max().item()
        batch_std = x.std().item()
        
        if not self._activation_stats_initialized:
            self._activation_mean_ema[0] = batch_mean
            self._activation_max_ema[0] = batch_max
            self._activation_std_ema[0] = batch_std
            self._activation_stats_initialized = True
        else:
            # Use same EMA beta for consistency
            self._activation_mean_ema[0] = (self.ema_beta * self._activation_mean_ema[0] + 
                                            (1.0 - self.ema_beta) * batch_mean)
            self._activation_max_ema[0] = (self.ema_beta * self._activation_max_ema[0] + 
                                           (1.0 - self.ema_beta) * batch_max)
            self._activation_std_ema[0] = (self.ema_beta * self._activation_std_ema[0] + 
                                          (1.0 - self.ema_beta) * batch_std)

        if not self._ema_initialized:
            if float(m.max().item()) <= 1e-8:
                return
            self._ema.copy_(m_norm)
            self._ema_initialized = True
        else:
            self._ema.mul_(self.ema_beta).add_(m_norm, alpha=(1.0 - self.ema_beta))

    def on_optimizer_step(self) -> None:
        self._opt_step += 1

        if self.log_interval > 0 and (self._opt_step % self.log_interval == 0):
            if self._ema_initialized:
                dormant_frac = float((self._ema < self.tau).float().mean().item())
                # IMPORTANT: use optimizer-step timestep so the curve is clean/monotonic
                self._emit_scalar("plasticity/dormant_frac", dormant_frac, timestep=self._opt_step)
                
                # NEW: Log activation statistics
                self._emit_scalar("plasticity/activation_mean", 
                                 float(self._activation_mean_ema[0].item()), timestep=self._opt_step)
                self._emit_scalar("plasticity/activation_max", 
                                 float(self._activation_max_ema[0].item()), timestep=self._opt_step)
                self._emit_scalar("plasticity/activation_std", 
                                 float(self._activation_std_ema[0].item()), timestep=self._opt_step)
                
                # NEW: Log EMA statistics (per-neuron EMA values)
                self._emit_scalar("plasticity/ema_mean", 
                                 float(self._ema.mean().item()), timestep=self._opt_step)
                self._emit_scalar("plasticity/ema_min", 
                                 float(self._ema.min().item()), timestep=self._opt_step)
                self._emit_scalar("plasticity/ema_max", 
                                 float(self._ema.max().item()), timestep=self._opt_step)
                
                # NEW: Log how EMA values compare to threshold
                self._emit_scalar("plasticity/tau_threshold", self.tau, timestep=self._opt_step)
            # If EMA isn't initialized yet, we just skip emitting (same behavior as ReDo)

    def __del__(self):
        try:
            if getattr(self, "_hook_handle_relu", None) is not None:
                self._hook_handle_relu.remove()
        except Exception:
            pass