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


class _DummyRollout:
    def __init__(self):
        self.after_update_called = False

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


def test_reset_intervention_resets_and_clears_state():
    ctx = _make_ctx({})
    reset = ResetIntervention(ctx)

    before = {name: p.detach().clone() for name, p in ctx.actor_critic.named_parameters()}
    _seed_optimizer_state(ctx.ppo_trainer.optimizer, ctx.actor_critic.parameters())

    reset.on_task_end(cycle_id=0, task_run_id=0)

    # params should change
    changed = False
    for name, p in ctx.actor_critic.named_parameters():
        if not torch.allclose(before[name], p.detach()):
            changed = True
            break
    assert changed, "Reset should change at least one parameter"

    assert ctx.ppo_trainer.optimizer.state == {}, "Optimizer state should be cleared"
    assert ctx.rollout_storage.after_update_called, "Rollout storage should be cleared"


def test_partial_reinit_only_head_changes():
    ctx = _make_ctx({})
    preinit = PartialReinitIntervention(ctx)

    base_before = {name: p.detach().clone() for name, p in ctx.actor_critic.base.named_parameters()}
    head_before = {name: p.detach().clone() for name, p in ctx.actor_critic.dist.named_parameters()}

    _seed_optimizer_state(ctx.ppo_trainer.optimizer, ctx.actor_critic.parameters())

    preinit.on_task_end(cycle_id=0, task_run_id=0)

    # base should stay the same
    for name, p in ctx.actor_critic.base.named_parameters():
        assert torch.allclose(base_before[name], p.detach()), "Base params should not change"

    # head should change
    changed = False
    for name, p in ctx.actor_critic.dist.named_parameters():
        if not torch.allclose(head_before[name], p.detach()):
            changed = True
            break
    assert changed, "Policy head params should change"

    assert ctx.rollout_storage.after_update_called, "Rollout storage should be cleared"


def test_set_intervention_initializes_sparse_masks():
    ctx = _make_ctx({"target_sparsity": 0.5, "warmup_steps": 0, "update_interval": 10})
    set_itv = SETIntervention(ctx)

    sparsity = set_itv._current_sparsity()
    assert 0.1 < sparsity < 0.9, "SET should initialize to a non-trivial sparsity"


def test_redo_intervention_smoke():
    ctx = _make_ctx({"update_interval": 10, "warmup_steps": 100, "use_activation_buffer": False})
    redo = ReDoIntervention(ctx)

    redo.on_task_start(cycle_id=0, task_run_id=0)
    redo.on_optimizer_step()
    assert redo._opt_step == 1


def test_gmp_tasks_per_cycle_mismatch_raises():
    ctx = _make_ctx({
        "final_sparsity": 0.8,
        "tasks_per_cycle": 7,
        "prune_cycle": 0,
        "train_tasks_per_cycle": 3,
    })
    gmp = GMPIntervention(ctx)

    with pytest.raises(ValueError, match="tasks_per_cycle mismatch"):
        gmp.on_task_start(cycle_id=0, task_run_id=0)


def test_gmp_tasks_per_cycle_match_ok():
    ctx = _make_ctx({
        "final_sparsity": 0.8,
        "tasks_per_cycle": 3,
        "prune_cycle": 0,
        "train_tasks_per_cycle": 3,
    })
    gmp = GMPIntervention(ctx)
    gmp.on_task_start(cycle_id=0, task_run_id=0)


def test_gmp_tasks_per_cycle_override_env(monkeypatch):
    monkeypatch.setenv("GMP_ALLOW_TASKS_PER_CYCLE_MISMATCH", "1")
    ctx = _make_ctx({
        "final_sparsity": 0.8,
        "tasks_per_cycle": 7,
        "prune_cycle": 0,
        "train_tasks_per_cycle": 3,
    })
    gmp = GMPIntervention(ctx)
    gmp.on_task_start(cycle_id=0, task_run_id=0)