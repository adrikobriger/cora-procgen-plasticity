"""
From https://raw.githubusercontent.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/84a7582477fb0d5c82ad6d850fe476829dddd2e1/a2c_ppo_acktr/algo/ppo.py
With minor changes
"""

import torch
import math
import logging
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PPO():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        # ADDED FOR INTERVENTIONS
        self.intervention = None

        # Diagnostics for zero-reward task issue.
        self.debug_enabled = False
        self._debug_logs = []
        self._debug_last_stats = {}
        self._debug_zero_delta_count = 0
        self._logger = logging.getLogger(__name__)

    def enable_debug(self, enabled: bool = True):
        self.debug_enabled = bool(enabled)

    def pop_debug_logs(self):
        if not self._debug_logs:
            return []
        out = self._debug_logs
        self._debug_logs = []
        return out

    def update(self, rollouts, action_space):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch, action_space)

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                # self.optimizer.zero_grad()
                # (value_loss * self.value_loss_coef + action_loss -
                #  dist_entropy * self.entropy_coef).backward()
                # nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                #                          self.max_grad_norm)
                # self.optimizer.step()

                # ADDED FOR INTERVENTIONS:
                self.optimizer.zero_grad()

                loss = (value_loss * self.value_loss_coef + action_loss -
                        dist_entropy * self.entropy_coef)

                loss.backward()

                # intervention hook: before backward/step
                if self.intervention is not None:
                    self.intervention.before_optimizer_step()

                total_norm = nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)

                # DEBUG: grad/param integrity checks
                if self.debug_enabled:
                    grad_has_nan = False
                    grad_has_inf = False
                    for p in self.actor_critic.parameters():
                        if p.grad is None:
                            continue
                        if torch.isnan(p.grad).any():
                            grad_has_nan = True
                        if torch.isinf(p.grad).any():
                            grad_has_inf = True

                    # snapshot params before step for delta norm
                    params_before = [p.detach().clone() for p in self.actor_critic.parameters() if p.requires_grad]

                self.optimizer.step()

                if self.debug_enabled:
                    delta_sq = 0.0
                    param_has_nan = False
                    param_has_inf = False
                    for p, p0 in zip((p for p in self.actor_critic.parameters() if p.requires_grad), params_before):
                        if torch.isnan(p).any():
                            param_has_nan = True
                        if torch.isinf(p).any():
                            param_has_inf = True
                        delta_sq += float((p.detach() - p0).pow(2).sum().item())
                    delta_norm = math.sqrt(delta_sq) if delta_sq > 0 else 0.0

                    if delta_norm == 0.0:
                        self._debug_zero_delta_count += 1
                    else:
                        self._debug_zero_delta_count = 0

                    self._debug_last_stats = {
                        "grad_norm": float(total_norm) if total_norm is not None else float("nan"),
                        "grad_has_nan": grad_has_nan,
                        "grad_has_inf": grad_has_inf,
                        "param_delta_norm": float(delta_norm),
                        "param_has_nan": param_has_nan,
                        "param_has_inf": param_has_inf,
                        "zero_delta_count": int(self._debug_zero_delta_count),
                        "loss": float(loss.item()),
                    }

                    self._debug_logs.extend([
                        {"type": "scalar", "tag": "debug/grad_norm", "value": float(total_norm)},
                        {"type": "scalar", "tag": "debug/param_delta_norm", "value": float(delta_norm)},
                        {"type": "scalar", "tag": "debug/zero_param_delta_count", "value": int(self._debug_zero_delta_count)},
                        {"type": "scalar", "tag": "debug/loss", "value": float(loss.item())},
                    ])

                # intervention hooks: after step + step counter 
                if self.intervention is not None:
                    self.intervention.after_optimizer_step()
                    self.intervention.on_optimizer_step()
                # END ADDED


                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
