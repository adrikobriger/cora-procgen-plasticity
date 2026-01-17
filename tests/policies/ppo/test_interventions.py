import logging

import pytest
import torch
import torch.nn as nn

from continual_rl.policies.ppo.interventions.base import InterventionContext
from continual_rl.policies.ppo.interventions.reset import ResetIntervention
from continual_rl.policies.ppo.interventions.partial_reinit import PartialReinitIntervention
from continual_rl.policies.ppo.interventions.set import SETIntervention
from continual_rl.policies.ppo.interventions.redo import ReDoIntervention
from continual_rl.policies.ppo.interventions.gmp import GMPIntervention


class _Base(nn.Module):
    def __init__(self):
        super().__init__()
        # Ensure base.main[8] is Linear, base.main[9] is ReLU
        self.main = nn.Sequential(
            nn.Conv2d(3, 3, 3),  # 0 (4D, should be excluded by GMP/SET)
            nn.ReLU(),           # 1
            nn.Flatten(),        # 2
            nn.Linear(27, 16),   # 3
            nn.ReLU(),           # 4
            nn.Linear(16, 16),   # 5
            nn.ReLU(),           # 6
            nn.Linear(16, 16),   # 7
            nn.Linear(16, 16),   # 8  <-- post-CNN FC
            nn.ReLU(),           # 9  <-- post-CNN ReLU
        )
        self.critic_linear = nn.Linear(16, 1)


class _DummyActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = _Base()
        self.dist = nn.Linear(16, 4)


class _DummyTrainer:
    def __init__(self, params):
        self.optimizer = torch.optim.Adam(params, lr=1e-3)
        self.ppo_epoch = 4
        self.num_mini_batch = 4


class _DummyRollout:
    def __init__(self):
        self.after_update_called = False
        self.num_steps = 4
        self.rewards = torch.zeros(4, 2)

    def after_update(self):
        self.after_update_called = True


def _make_ctx(params: dict):
    ac = _DummyActorCritic()
    trainer = _DummyTrainer(ac.parameters())
    rollout = _DummyRollout()
    return InterventionContext(
        actor_critic=ac,
        ppo_trainer=trainer,
        rollout_storage=rollout,
        device=torch.device("cpu"),
        logger=logging.getLogger("test_interventions"),
        params=params,
    )


def _seed_optimizer_state(opt, params):
    for p in params:
        opt.state[p] = {
            "exp_avg": torch.ones_like(p.data),
            "exp_avg_sq": torch.ones_like(p.data),
        }


def _param_snapshot(model: nn.Module):
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def test_reset_reinitializes_all_params_and_clears_state():
    torch.manual_seed(0)
    ctx = _make_ctx({})
    reset = ResetIntervention(ctx)

    before = _param_snapshot(ctx.actor_critic)
    _seed_optimizer_state(ctx.ppo_trainer.optimizer, ctx.actor_critic.parameters())

    reset.on_task_end(cycle_id=0, task_run_id=0)

    changed = any(
        not torch.allclose(before[name], p.detach())
        for name, p in ctx.actor_critic.named_parameters()
    )
    assert changed, "Reset should reinitialize at least one parameter"
    assert ctx.ppo_trainer.optimizer.state == {}, "Optimizer state should be cleared"
    assert ctx.rollout_storage.after_update_called, "Rollout storage should be cleared"


def test_partial_reinit_only_head_changes_and_optimizer_state_cleared_for_head():
    torch.manual_seed(0)
    ctx = _make_ctx({})
    preinit = PartialReinitIntervention(ctx)

    base_before = _param_snapshot(ctx.actor_critic.base)
    head_before = _param_snapshot(ctx.actor_critic.dist)
    _seed_optimizer_state(ctx.ppo_trainer.optimizer, ctx.actor_critic.parameters())

    preinit.on_task_end(cycle_id=0, task_run_id=0)

    # base should stay the same
    for name, p in ctx.actor_critic.base.named_parameters():
        assert torch.allclose(base_before[name], p.detach()), "Base params should not change"

    # head should change
    changed = any(
        not torch.allclose(head_before[name], p.detach())
        for name, p in ctx.actor_critic.dist.named_parameters()
    )
    assert changed, "Policy head params should change"

    # optimizer state cleared only for head params
    opt = ctx.ppo_trainer.optimizer
    for p in ctx.actor_critic.dist.parameters():
        assert (p not in opt.state) or (opt.state[p] == {}), "Head optimizer state should be cleared"
    for p in ctx.actor_critic.base.parameters():
        assert p in opt.state and opt.state[p] != {}, "Base optimizer state should remain"

    assert ctx.rollout_storage.after_update_called, "Rollout storage should be cleared"


def test_set_prune_regrow_preserves_active_count_and_masks_weights():
    torch.manual_seed(0)
    ctx = _make_ctx({
        "target_sparsity": 0.5,
        "warmup_steps": 0,
        "update_interval": 1,
        "prune_fraction": 0.2,
        "seed": 123,
    })
    set_itv = SETIntervention(ctx)

    masks_before = {k: v.clone() for k, v in set_itv._masks.items()}
    zero_count = sum(int((m == 0).sum().item()) for m in masks_before.values())
    assert zero_count > 0, "SET should initialize with some masked (zero) weights"

    active_before = sum(int(m.sum().item()) for m in masks_before.values())

    set_itv.on_optimizer_step()

    masks_after = set_itv._masks
    active_after = sum(int(m.sum().item()) for m in masks_after.values())
    assert active_after == active_before, "SET should preserve active weight count"

    flips = sum(int((masks_after[k] != masks_before[k]).sum().item()) for k in masks_after.keys())
    assert flips > 0, "SET should prune/regrow and flip some mask entries"

    set_itv._apply_masks_to_params_()
    for item in set_itv._prunable:
        mask = set_itv._masks[item.name]
        assert torch.all(item.param.data[mask == 0] == 0), "Masked weights should be zero"


def test_redo_recycles_dormant_units_and_clears_optimizer_slices():
    torch.manual_seed(0)
    ctx = _make_ctx({
        "update_interval": 1,
        "warmup_steps": 0,
        "tau": 1.1,
        "ema_beta": 0.0,
        "max_recycle_frac": 0.5,
        "use_activation_buffer": False,
    })
    redo = ReDoIntervention(ctx)

    # force non-zero head/FC weights for clear detection
    redo._head.weight.data.fill_(0.5)
    redo._fc.weight.data.fill_(0.5)

    _seed_optimizer_state(ctx.ppo_trainer.optimizer, [redo._fc.weight, redo._fc.bias, redo._head.weight])

    # initialize EMA using a fake activation
    out = torch.ones((4, redo.hidden))
    redo._activation_hook(None, None, out)
    assert redo._ema_initialized

    head_before = redo._head.weight.detach().clone()
    fc_before = redo._fc.weight.detach().clone()

    redo.on_optimizer_step()

    zero_cols = (redo._head.weight.abs().sum(dim=0) == 0)
    assert zero_cols.any(), "ReDo should zero at least one head column"

    assert torch.any(redo._fc.weight[zero_cols] != fc_before[zero_cols]), "ReDo should change recycled FC rows"

    st = ctx.ppo_trainer.optimizer.state[redo._head.weight]
    assert torch.all(st["exp_avg"][:, zero_cols] == 0)
    assert torch.all(st["exp_avg_sq"][:, zero_cols] == 0)

    assert torch.any(head_before[:, zero_cols] != redo._head.weight[:, zero_cols])


def test_gmp_prunes_toward_target_and_excludes_layers():
    torch.manual_seed(0)
    ctx = _make_ctx({
        "final_sparsity": 0.8,
        "total_train_steps": 10,
        "tstart_frac": 0.0,
        "tend_frac": 1.0,
        "pruning_freq_steps": 1,
    })
    gmp = GMPIntervention(ctx)

    keys = list(gmp._masks.keys())
    assert not any("base.main.0" in k for k in keys), "GMP should exclude conv trunk"
    assert not any("base.critic_linear" in k for k in keys), "GMP should exclude critic head"
    assert any("base.main.8" in k for k in keys), "GMP should include post-CNN FC"
    assert any("dist" in k for k in keys), "GMP should include actor head"

    initial = gmp._current_sparsity()
    for _ in range(10):
        gmp.on_optimizer_step()
    final = gmp._current_sparsity()

    assert final > initial, "GMP should increase sparsity over time"
    assert final >= 0.2, "GMP should achieve non-trivial sparsity by end of schedule"

    gmp._apply_masks_to_params_()
    for item in gmp._prunable:
        mask = gmp._masks[item.name]
        assert torch.all(item.param.data[mask == 0] == 0), "Masked weights should be zero"


def test_gmp_requires_total_train_steps():
    ctx = _make_ctx({"final_sparsity": 0.8})
    with pytest.raises(ValueError, match="total_train_steps"):
        GMPIntervention(ctx)
