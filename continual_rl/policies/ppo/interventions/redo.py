from __future__ import annotations

from typing import Optional, List, Dict, Any
from collections import deque

import torch
import torch.nn as nn
import torch.nn.init as init

from .base import InterventionBase


class ReDoIntervention(InterventionBase):
    """
    ReDo (Recycling Dormant Neurons) for PPO+Procgen.

    Tracks post-FC ReLU activations (hidden=512) and identifies dormant units.
    Optionally evaluates dormancy using a replay buffer of FC inputs.
    Recycles dormant units by:
      - reinitializing the corresponding FC incoming weights (rows)
      - zeroing the corresponding actor-head outgoing weights (columns)
      - clearing Adam moments for those slices
    """

    def __init__(self, ctx):
        super().__init__(ctx)
        p = ctx.params or {}

        # HYPERPARAMETERS
        self.update_interval: int = int(p.get("update_interval", 5000))
        self.warmup_steps: int = int(p.get("warmup_steps", 12000))
        self.tau: float = float(p.get("tau", 0.10))
        self.ema_beta: float = float(p.get("ema_beta", 0.99))
        self.max_recycle_frac: float = float(p.get("max_recycle_frac", 0.10))
        self.log_interval: int = int(p.get("log_interval", 1000))

        # BUFFER SETTINGS
        self.use_activation_buffer: bool = bool(p.get("use_activation_buffer", False))
        self.buffer_size: int = int(p.get("buffer_size", 50000))          # rows per task
        self.store_every: int = int(p.get("store_every", 5))              # store every N forwards (FC hook calls)
        self.eval_batch: int = int(p.get("eval_batch", 4096))             # rows to evaluate at recycle time
        self.mix_current_frac: float = float(p.get("mix_current_frac", 0.5))
        self.max_tasks_in_buffer: int = int(p.get("max_tasks_in_buffer", 50))
        self._disable_store: bool = False

        # STATE COUNTERS
        self._opt_step: int = 0
        self._forward_calls: int = 0
        self._store_counter: int = 0

        # MODEL PARTS
        self._fc: nn.Linear = self._get_post_cnn_fc()
        self._relu: nn.Module = self._get_post_cnn_relu()
        self._head: nn.Linear = self._get_actor_head()

        self.hidden: int = int(self._fc.out_features)
        self._fc_in_dim: int = int(self._fc.in_features)

        # EMA over normalized mean abs activation per unit
        device = self.ctx.device
        self._ema = torch.ones(self.hidden, device=device)
        self._ema_initialized = False

        # BUFFER STATE
        self._task_id: int = 0
        self._buffers: Dict[int, Dict[str, Any]] = {}  # tid -> {data, ptr, full}
        self._seen_tasks = deque(maxlen=self.max_tasks_in_buffer)

        # HOOKS
        self._hook_handle_fc = self._fc.register_forward_hook(self._fc_input_hook)
        self._hook_handle_relu = self._relu.register_forward_hook(self._activation_hook)

        self.logger.info(
            "redo init | update_interval=%d warmup_steps=%d tau=%.3f ema_beta=%.3f max_recycle_frac=%.3f hidden=%d use_buffer=%s",
            self.update_interval,
            self.warmup_steps,
            self.tau,
            self.ema_beta,
            self.max_recycle_frac,
            self.hidden,
            str(self.use_activation_buffer),
        )

    # MODEL PARTS ACCESSORS
    def _get_post_cnn_fc(self) -> nn.Linear:
        ac = self.ctx.actor_critic
        if not hasattr(ac, "base") or not hasattr(ac.base, "main"):
            raise AttributeError("actor_critic has no base.main (unexpected architecture).")
        fc = ac.base.main[8]
        if not isinstance(fc, nn.Linear):
            raise TypeError(f"Expected base.main[8] to be nn.Linear, got {type(fc)}")
        return fc

    def _get_post_cnn_relu(self) -> nn.Module:
        ac = self.ctx.actor_critic
        return ac.base.main[9]

    def _get_actor_head(self) -> nn.Linear:
        ac = self.ctx.actor_critic
        if not hasattr(ac, "dist"):
            raise AttributeError("actor_critic has no attribute 'dist' (expected policy head).")
        head = ac.dist
        if not isinstance(head, nn.Linear):
            raise TypeError(f"actor_critic.dist is {type(head)} but expected torch.nn.Linear")
        return head

    # BUFFER COLLECTION HOOK (FC input)
    @torch.no_grad()
    def _fc_input_hook(self, module: nn.Module, inp, out) -> None:
        """
        Collect FC input vectors into a per-task ring buffer.
        We MUST cap rows-per-hook-call, otherwise occasional [T,B,dim] inputs
        explode the buffer in a single step (your 5246 jump).
        """
        if not self.use_activation_buffer:
            return
        
        if self._disable_store:
            return

        if not inp or (not torch.is_tensor(inp[0])):
            return

        fc_in = inp[0]
        if fc_in.shape[-1] != self._fc_in_dim:
            return

        self._store_counter += 1
        if (self._store_counter % self.store_every) != 0:
            return

        x = fc_in.detach().reshape(-1, self._fc_in_dim)

        # hard cap: max rows we accept from one hook call
        max_rows = 256  # could maybe be tuned but this fixes the buffer explosion
        n = int(x.shape[0])

        if n > max_rows:
            # proof log (only when it happens)
            self.logger.info(
                "redo WARN big fc_in store | opt_step=%d task=%d fc_in_shape=%s rows=%d -> capped=%d",
                self._opt_step,
                self._task_id,
                tuple(fc_in.shape),
                n,
                max_rows,
            )
            # uniform random subsample of rows
            idx = torch.randint(0, n, (max_rows,), device=x.device)
            x = x.index_select(0, idx)

        self._push_fc_in(x)


    @torch.no_grad()
    def _push_fc_in(self, fc_in: torch.Tensor) -> None:
        x = fc_in.to("cpu", non_blocking=True)

        tid = int(self._task_id)
        if tid not in self._buffers:
            data = torch.empty((self.buffer_size, self._fc_in_dim), dtype=x.dtype)
            self._buffers[tid] = {"data": data, "ptr": 0, "full": False}
            self._seen_tasks.append(tid)

        buf = self._buffers[tid]
        data, ptr = buf["data"], int(buf["ptr"])
        full = bool(buf["full"])

        n = int(x.shape[0])
        if n <= 0:
            return

        if n >= self.buffer_size:
            data[:] = x[-self.buffer_size:]
            buf["ptr"] = 0
            buf["full"] = True
            return

        end = ptr + n
        if end <= self.buffer_size:
            data[ptr:end] = x
        else:
            k1 = self.buffer_size - ptr
            data[ptr:] = x[:k1]
            data[: (n - k1)] = x[k1:]
            full = True

        buf["ptr"] = (ptr + n) % self.buffer_size
        buf["full"] = full

    # ACTIVATION TRACKING HOOK (post-FC ReLU)
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

        if not self._ema_initialized:
            if float(m.max().item()) <= 1e-8:
                return
            self._ema.copy_(m_norm)
            self._ema_initialized = True
        else:
            self._ema.mul_(self.ema_beta).add_(m_norm, alpha=(1.0 - self.ema_beta))

    # TASK BOUNDARY HOOKS
    def on_task_start(self, cycle_id: int, task_run_id: int) -> None:
        # keep compatibility with base
        try:
            super().on_task_start(cycle_id, task_run_id)
        except TypeError:
            pass

        self._task_id = int(task_run_id)

        #  reset store counter so we don’t “inherit” modulo state across tasks
        self._store_counter = 0

        self.logger.info(
            "redo task start | cycle=%d task=%d opt_step=%d buffers=%d",
            cycle_id, task_run_id, self._opt_step, len(self._buffers)
        )


    # OPTIMIZER STEP SCHEDULE HOOK
    def on_optimizer_step(self) -> None:
        self._opt_step += 1

        if self.log_interval > 0 and (self._opt_step % self.log_interval == 0):
            if self._ema_initialized:
                dormant_frac = float((self._ema < self.tau).float().mean().item())

                # forward to TensorBoard via PPOPolicy.train -> TaskBase
                self._emit_scalar("plasticity/dormant_frac", dormant_frac, timestep=self._opt_step)
                # self._emit_scalar("plasticity/dormant_pct", 100.0 * dormant_frac)

                self.logger.info(
                    "redo stats | opt_step=%d forward_calls=%d dormant_frac=%.4f buffer_rows_curr=%d",
                    self._opt_step,
                    self._forward_calls,
                    dormant_frac,
                    self._task_size(self._task_id) if self.use_activation_buffer and (self._task_id in self._buffers) else 0,
                )
            else:
                self.logger.info(
                    "redo stats | opt_step=%d forward_calls=%d ema_init=False buffer_rows_curr=%d",
                    self._opt_step,
                    self._forward_calls,
                    self._task_size(self._task_id) if self.use_activation_buffer and (self._task_id in self._buffers) else 0,
                )

        if self._opt_step < self.warmup_steps:
            return
        if self.update_interval <= 0:
            return
        if (self._opt_step % self.update_interval) != 0:
            return

        self._maybe_recycle()

    # RECYCLE LOGIC
    @torch.no_grad()
    def _maybe_recycle(self) -> None:
        if not self._ema_initialized:
            self.logger.info("redo recycle skipped | opt_step=%d reason=no_ema_yet", self._opt_step)
            return

        dormancy_scores = None
        if self.use_activation_buffer:
            dormancy_scores = self._buffer_activity()

        if dormancy_scores is None:
            dormancy_scores = self._ema  # fallback
            source = "ema"
        else:
            source = "buffer"

        dormant_mask = (dormancy_scores < self.tau)
        dormant_idxs = dormant_mask.nonzero(as_tuple=False).view(-1)

        dormant_count = int(dormant_idxs.numel())
        if dormant_count == 0:
            self.logger.info("redo recycle | opt_step=%d source=%s dormant=0", self._opt_step, source)
            return

        max_k = max(1, int(round(self.max_recycle_frac * self.hidden)))

        # choose lowest-score units first
        vals = dormancy_scores[dormant_idxs]
        order = torch.argsort(vals)  # ascending
        chosen = dormant_idxs[order[:max_k]]
        chosen_list = chosen.tolist()

        self.logger.info(
            "redo recycle select | opt_step=%d source=%s dormant=%d recycled=%d cap=%d",
            self._opt_step,
            source,
            dormant_count,
            len(chosen_list),
            max_k,
        )

        self._recycle_units(chosen_list)
        # bump EMA for recycled units to avoid instant re-trigger
        self._ema[chosen] = 1.0

    @torch.no_grad()
    def _recycle_units(self, unit_idxs: List[int]) -> None:
        fc = self._fc
        head = self._head
        opt = self.ctx.ppo_trainer.optimizer

        device = fc.weight.device
        idx = torch.tensor(unit_idxs, device=device, dtype=torch.long)

        pre_fc = fc.weight[idx].detach().clone()
        pre_head = head.weight[:, idx].detach().clone()

        # 1. reinit incoming FC weights for selected units
        in_features = fc.weight.shape[1]
        k = idx.numel()

        fresh = torch.empty((k, in_features), device=device, dtype=fc.weight.dtype)
        init.orthogonal_(fresh)
        fc.weight.index_copy_(0, idx, fresh)
        if fc.bias is not None:
            fc.bias.index_fill_(0, idx, 0.0)

        # 2. zero outgoing weights in actor head
        head.weight.index_fill_(1, idx, 0.0)

        # 3. clear Adam state slices
        self._zero_adam_slices_(opt, fc.weight, row_idx=idx)
        if fc.bias is not None:
            self._zero_adam_slices_(opt, fc.bias, vec_idx=idx)
        self._zero_adam_slices_(opt, head.weight, col_idx=idx)

        self.logger.info(
            "redo recycle applied | opt_step=%d k=%d fc_abs_delta=%.4g head_abs_delta=%.4g head_cols_abs_sum_after=%.4g",
            self._opt_step,
            int(idx.numel()),
            float((fc.weight[idx] - pre_fc).abs().mean().item()),
            float((head.weight[:, idx] - pre_head).abs().mean().item()),
            float(head.weight[:, idx].abs().sum().item()),
        )

    @staticmethod
    @torch.no_grad()
    def _zero_adam_slices_(
        opt: torch.optim.Optimizer,
        param: torch.nn.Parameter,
        row_idx: Optional[torch.Tensor] = None,
        col_idx: Optional[torch.Tensor] = None,
        vec_idx: Optional[torch.Tensor] = None,
    ) -> None:
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

    # BUFFER ACTIVITY EVALUATION
    @torch.no_grad()
    def _buffer_activity(self) -> Optional[torch.Tensor]:
        if len(self._buffers) == 0:
            return None

        curr = int(self._task_id)
        if curr not in self._buffers:
            return None

        k = int(self.eval_batch)
        if k <= 0:
            return None

        k_curr = int(round(self.mix_current_frac * k))
        k_past = k - k_curr

        xs = []

        x_curr = self._sample_from_task(curr, k_curr)
        if x_curr is None:
            return None
        xs.append(x_curr)

        past_tasks = [t for t in self._buffers.keys() if t != curr]
        if k_past > 0 and len(past_tasks) > 0:
            x_past = self._sample_from_tasks(past_tasks, k_past)
            if x_past is not None:
                xs.append(x_past)

        x = torch.cat(xs, dim=0).to(self._fc.weight.device)  # [K, in_features]

        self._disable_store = True
        try:
            acts = self._relu(self._fc(x))
        finally:
            self._disable_store = False

        m = acts.abs().mean(dim=0)                            # [hidden]
        denom = m.mean().clamp_min(1e-8)
        return m / denom

    @torch.no_grad()
    def _task_size(self, tid: int) -> int:
        buf = self._buffers[tid]
        return self.buffer_size if bool(buf["full"]) else int(buf["ptr"])

    @torch.no_grad()
    def _sample_from_task(self, tid: int, k: int) -> Optional[torch.Tensor]:
        if k <= 0:
            return torch.empty((0, self._fc_in_dim))
        n = self._task_size(tid)
        if n <= 0:
            return None
        buf = self._buffers[tid]["data"][:n]
        idx = torch.randint(0, n, (k,))
        return buf[idx]

    @torch.no_grad()
    def _sample_from_tasks(self, tids: list[int], k: int) -> Optional[torch.Tensor]:
        if k <= 0 or len(tids) == 0:
            return torch.empty((0, self._fc_in_dim))

        out = []
        for _ in range(k):
            t = tids[int(torch.randint(0, len(tids), (1,)).item())]
            n = self._task_size(t)
            if n <= 0:
                continue
            buf = self._buffers[t]["data"][:n]
            j = int(torch.randint(0, n, (1,)).item())
            out.append(buf[j].unsqueeze(0))

        if len(out) == 0:
            return None
        return torch.cat(out, dim=0)

    # CLEANUP
    def __del__(self):
        try:
            if getattr(self, "_hook_handle_fc", None) is not None:
                self._hook_handle_fc.remove()
        except Exception:
            pass
        try:
            if getattr(self, "_hook_handle_relu", None) is not None:
                self._hook_handle_relu.remove()
        except Exception:
            pass
