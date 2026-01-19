import torch
import os
import logging
import numpy as np
from continual_rl.policies.policy_base import PolicyBase
from continual_rl.policies.ppo.ppo_policy_config import PPOPolicyConfig
from continual_rl.policies.ppo.ppo_timestep_data import PPOTimestepData
from continual_rl.policies.ppo.a2c_ppo_acktr_gail.ppo import PPO
from continual_rl.policies.ppo.a2c_ppo_acktr_gail.model import Policy
from continual_rl.policies.ppo.a2c_ppo_acktr_gail.storage import RolloutStorage
from continual_rl.experiments.environment_runners.environment_runner_batch import EnvironmentRunnerBatch
from continual_rl.utils.utils import Utils
import continual_rl.policies.ppo.a2c_ppo_acktr_gail.utils as utils
from continual_rl.policies.ppo.interventions import make_intervention, InterventionContext
from continual_rl.utils.debug_stats import RollingStats


class PPOPolicy(PolicyBase):
    """
    A simple implementation of policy as a sample of how policies can be created.
    Refer to policy_base itself for more detailed descriptions of the method signatures.

    Some of the code in this file is adapted from:
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/84a7582477fb0d5c82ad6d850fe476829dddd2e1/main.py

    This method is NOT multi-headed. I.e. if the tasks have mismatched action spaces, the biggest one is used,
    and the rest are subsets.
    """
    def __init__(self, config: PPOPolicyConfig, observation_space, action_spaces):  # Switch to your config type
        super().__init__(config)
        self._logger = logging.getLogger(__name__)
        max_action_space = Utils.get_max_discrete_action_space(action_spaces)
        self._action_spaces = action_spaces

        # Original observation_space is [time, channels, width, height]
        # Compact it into [time * channels, width, height]
        observation_size = observation_space.shape
        compressed_observation_size = [observation_size[0] * observation_size[1], observation_size[2], observation_size[3]]
        self._config = config

        # Diagnostics for zero-reward task issue.
        self._debug_reward_pipeline = bool(getattr(self._config, "debug_reward_pipeline", False))
        self._debug_interval = int(getattr(self._config, "debug_reward_pipeline_interval", 10000) or 10000)
        self._debug_first_steps = int(getattr(self._config, "debug_reward_pipeline_first_steps", 0) or 0)
        self._debug_max_actions = int(getattr(self._config, "debug_reward_pipeline_max_actions", 20) or 20)
        self._debug_step_counter = 0
        self._debug_last_log_step = 0
        self._debug_raw_reward_stats = RollingStats()
        self._debug_shaped_reward_stats = RollingStats()
        self._debug_done_count = 0
        self._debug_terminated_count = 0
        self._debug_truncated_count = 0
        self._debug_action_counts = None
        self._debug_action_total = 0
        self._debug_argmax_count = 0
        self._debug_entropy_sum = 0.0
        self._debug_obs_nan = 0
        self._debug_obs_inf = 0
        self._debug_logits_nan = 0
        self._debug_logits_inf = 0
        self._debug_value_nan = 0
        self._debug_value_inf = 0
        self._debug_logprob_nan = 0
        self._debug_logprob_inf = 0
        self._debug_pending_logs = []
        self._debug_task_run_id = None

        # ADDED:
        # Continual RL intervention settings (default: dense/no-op)
        self._intervention_type = getattr(self._config, "intervention_type", "dense")
        self._logger.info("ppo | intervention_type=%s", self._intervention_type)
        self._intervention_params = getattr(self._config, "intervention_params", {})
        # END ADDED

        self._device = torch.device("cuda:0" if self._config.cuda else "cpu")

        self._actor_critic = Policy(obs_shape=compressed_observation_size,
                                    action_space=max_action_space)
        self._actor_critic.to(self._device)

        self._rollout_storage = RolloutStorage(num_steps=config.num_steps,
                                               num_processes=config.num_processes,
                                               obs_shape=compressed_observation_size,
                                               action_space=max_action_space,
                                               recurrent_hidden_state_size=self._actor_critic.recurrent_hidden_state_size)
        self._rollout_storage.to(self._device)

        self._ppo_trainer = PPO(
            self._actor_critic,
            self._config.clip_param,
            self._config.ppo_epoch,
            self._config.num_mini_batch,
            self._config.value_loss_coef,
            self._config.entropy_coef,
            lr=self._config.learning_rate,
            eps=self._config.eps,
            max_grad_norm=self._config.max_grad_norm)
        self._step_id = 0  # What collection step we're at, in the current num_steps size collection
        self._train_step_id = 0  # How many times we've trained

        self._ppo_trainer.enable_debug(self._debug_reward_pipeline)

        # ADDED:
        # Build intervention handler (keeps PPOPolicy clean)
        ctx = InterventionContext(
            actor_critic=self._actor_critic,
            ppo_trainer=self._ppo_trainer,
            rollout_storage=self._rollout_storage,
            device=self._device,
            logger=self._logger,
            params=self._intervention_params,
        )
        self._intervention = make_intervention(self._intervention_type, ctx)
        self._logger.info("ppo | intervention_class=%s", self._intervention.__class__.__name__)

        # Wire intervention into PPO update loop (optimizer-step hooks)
        self._ppo_trainer.intervention = self._intervention
        # END ADDED

    # ADDED METHODS FOR POLICY HOOKS
    def on_task_start(self, cycle_id: int, task_run_id: int):
        self._intervention.on_task_start(cycle_id, task_run_id)
        if self._debug_reward_pipeline:
            self._debug_reset_task(task_run_id)
    
    # ADDED METHODS FOR POLICY HOOKS
    def on_task_end(self, cycle_id: int, task_run_id: int):
        self._intervention.on_task_end(cycle_id, task_run_id)
        if self._debug_reward_pipeline:
            self._debug_emit_summary(force=True)

    def get_debug_reward_pipeline_config(self):
        return {
            "enabled": self._debug_reward_pipeline,
            "interval": self._debug_interval,
            "first_steps": self._debug_first_steps,
        }

    def get_environment_runner(self, task_spec):
        # See note in policy_base.get_environment_runner
        num_parallel_envs = 1 if task_spec.eval_mode else self._config.num_processes

        # Since this method is using a shared memory storage (self._rollout_storage), FullParallel cannot be supported.
        # To support it, move to using only what is returned in TimestepData from compute_action
        runner = EnvironmentRunnerBatch(policy=self, num_parallel_envs=num_parallel_envs,
                                        timesteps_per_collection=self._config.num_steps,
                                        render_collection_freq=self._config.render_collection_freq,
                                        output_dir=self._config.output_dir)
        return runner

    def _update_rollout_storage(self, observation, last_timestep_data):
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in last_timestep_data.done])

        # The original a2c_ppo_acktr_gail uses a TimeLimit gym wrapper, and that sets bad_transition
        # This is analogous to utils/env_wrappers/TimeLimit, which uses TimeLimit.truncated
        # This is not currently fully tested
        def _is_truncated(info):
            if not isinstance(info, dict):
                return False
            return bool(info.get("_truncated") or info.get("TimeLimit.truncated") or info.get("truncated"))

        bad_masks = torch.FloatTensor(
            [[0.0] if _is_truncated(info) else [1.0]
             for info in last_timestep_data.info])
        rewards = torch.FloatTensor(last_timestep_data.reward).unsqueeze(1)

        if self._config.clip_reward:
            rewards = torch.sign(rewards)

        if self._debug_reward_pipeline:
            self._debug_step_counter += len(last_timestep_data.reward)
            self._debug_raw_reward_stats.update(last_timestep_data.reward)
            self._debug_shaped_reward_stats.update(rewards)
            self._debug_done_count += int(np.sum(last_timestep_data.done))
            for info in last_timestep_data.info:
                if isinstance(info, dict):
                    if info.get("_terminated") is True or info.get("terminated") is True:
                        self._debug_terminated_count += 1
                    if info.get("_truncated") is True or info.get("truncated") is True or info.get("TimeLimit.truncated") is True:
                        self._debug_truncated_count += 1

            # If terminated/truncated were never annotated, mirror done to avoid split-brain diagnostics
            if self._debug_terminated_count == 0 and self._debug_truncated_count == 0 and self._debug_done_count > 0:
                self._debug_terminated_count = self._debug_done_count
            self._debug_emit_summary()

        # The codebase being used expects the resultant observation, not the producer observation.
        self._rollout_storage.insert(observation, last_timestep_data.recurrent_hidden_states,
                                     last_timestep_data.actions, last_timestep_data.action_log_probs,
                                     last_timestep_data.values, rewards, masks, bad_masks)

    def _update_learning_rate(self):
        if self._config.use_linear_lr_decay:
            num_updates = self._config.decay_over_steps // (self._config.num_steps * self._config.num_processes)

            # decrease learning rate linearly
            utils.update_linear_schedule(
                self._ppo_trainer.optimizer, self._train_step_id, num_updates, self._config.learning_rate)

    def compute_action(self, observation, task_id, action_space_id, last_timestep_data, eval_mode):
        action_space = self._action_spaces[action_space_id]

        # The observation now includes the batch
        observation = observation.view((observation.shape[0], -1, observation.shape[3], observation.shape[4]))

        # Insert the previous step's data, now that it has been populated with reward and done
        if last_timestep_data is not None:
            self._update_rollout_storage(observation, last_timestep_data)

        # We could get this from the timestep data itself, but doing it this way for consistency with the original
        # codebase (a2c_ppo_acktr_gail)
        observation = self._rollout_storage.obs[self._step_id]
        recurrent_hidden_state = self._rollout_storage.recurrent_hidden_states[self._step_id]
        masks = self._rollout_storage.masks[self._step_id]

        with torch.no_grad():
            value, action, action_log_prob, recurrent_hidden_states = \
                self._actor_critic.act(observation, recurrent_hidden_state, masks, action_space=action_space)

        if self._debug_reward_pipeline:
            try:
                obs_np = observation.detach().cpu().numpy()
                self._debug_obs_nan += int(np.isnan(obs_np).sum())
                self._debug_obs_inf += int(np.isinf(obs_np).sum())
            except Exception:
                pass

            # Action stats + entropy (recompute logits under debug only)
            is_discrete = (action_space.__class__.__name__ == "Discrete") or hasattr(action_space, "n")
            if is_discrete:
                with torch.no_grad():
                    _, actor_features, _ = self._actor_critic.base(observation, recurrent_hidden_state, masks)
                    num_outputs = int(getattr(action_space, "n", None) or 0)
                    logits = self._actor_critic.dist(actor_features)
                    if num_outputs > 0:
                        logits = logits[:, :num_outputs]

                    try:
                        logits_np = logits.detach().cpu().numpy()
                        self._debug_logits_nan += int(np.isnan(logits_np).sum())
                        self._debug_logits_inf += int(np.isinf(logits_np).sum())
                    except Exception:
                        pass

                    dist = torch.distributions.Categorical(logits=logits)
                    entropy = dist.entropy()
                    self._debug_entropy_sum += float(entropy.sum().item())

                    argmax = logits.argmax(dim=-1)
                    action_flat = action.squeeze(-1)
                    self._debug_argmax_count += int((argmax == action_flat).sum().item())

                    actions_np = action_flat.detach().cpu().numpy().astype(np.int64, copy=False)
                    if self._debug_action_counts is None or len(self._debug_action_counts) != num_outputs:
                        self._debug_action_counts = np.zeros(num_outputs, dtype=np.int64)
                    self._debug_action_counts += np.bincount(actions_np, minlength=num_outputs)
                    self._debug_action_total += int(actions_np.size)

            if torch.isnan(value).any():
                self._debug_value_nan += int(torch.isnan(value).sum().item())
            if torch.isinf(value).any():
                self._debug_value_inf += int(torch.isinf(value).sum().item())
            if torch.isnan(action_log_prob).any():
                self._debug_logprob_nan += int(torch.isnan(action_log_prob).sum().item())
            if torch.isinf(action_log_prob).any():
                self._debug_logprob_inf += int(torch.isinf(action_log_prob).sum().item())

        # Keep storage actions as (N, 1); return env actions as (N,)
        action_for_env = action
        if isinstance(action_for_env, torch.Tensor) and action_for_env.dim() == 2 and action_for_env.size(1) == 1:
            action_for_env = action_for_env.squeeze(1)   # (N,1) -> (N,)

        timestep_data = PPOTimestepData(observation=observation, recurrent_hidden_states=recurrent_hidden_states,
                                        actions=action, action_log_probs=action_log_prob, values=value,
                                        action_space=action_space)

        self._step_id = (self._step_id + 1) % self._config.num_steps

        return action_for_env, timestep_data

    def train(self, storage_buffer):
        self._update_learning_rate()

        with torch.no_grad():
            next_value = self._actor_critic.get_value(
                self._rollout_storage.obs[-1], self._rollout_storage.recurrent_hidden_states[-1],
                self._rollout_storage.masks[-1]).detach()

        self._rollout_storage.compute_returns(next_value, self._config.use_gae, self._config.gamma,
                                 self._config.gae_lambda, self._config.use_proper_time_limits)

        if self._debug_reward_pipeline:
            returns = self._rollout_storage.returns[:-1]
            values = self._rollout_storage.value_preds[:-1]
            adv = returns - values

            adv_stats = RollingStats()
            ret_stats = RollingStats()
            adv_stats.update(adv)
            ret_stats.update(returns)

            adv_np = adv.detach().cpu().numpy() if adv.numel() > 0 else np.asarray([])
            adv_nan = int(np.isnan(adv_np).sum()) if adv_np.size else 0
            adv_inf = int(np.isinf(adv_np).sum()) if adv_np.size else 0
            ret_np = returns.detach().cpu().numpy() if returns.numel() > 0 else np.asarray([])
            ret_nan = int(np.isnan(ret_np).sum()) if ret_np.size else 0
            ret_inf = int(np.isinf(ret_np).sum()) if ret_np.size else 0

            logs = [
                {"type": "scalar", "tag": "debug/adv_mean", "value": adv_stats.mean},
                {"type": "scalar", "tag": "debug/adv_abs_mean", "value": float(np.mean(np.abs(adv.detach().cpu().numpy()))) if adv.numel() > 0 else 0.0},
                {"type": "scalar", "tag": "debug/adv_min", "value": adv_stats.summary()["min"]},
                {"type": "scalar", "tag": "debug/adv_max", "value": adv_stats.summary()["max"]},
                {"type": "scalar", "tag": "debug/return_mean", "value": ret_stats.mean},
                {"type": "scalar", "tag": "debug/return_min", "value": ret_stats.summary()["min"]},
                {"type": "scalar", "tag": "debug/return_max", "value": ret_stats.summary()["max"]},
                {"type": "scalar", "tag": "debug/adv_nan", "value": adv_nan},
                {"type": "scalar", "tag": "debug/adv_inf", "value": adv_inf},
                {"type": "scalar", "tag": "debug/return_nan", "value": ret_nan},
                {"type": "scalar", "tag": "debug/return_inf", "value": ret_inf},
            ]
        else:
            logs = []

        # Initial experiments seem to indicate that truncating the evaluate_action using the action_space
        # makes learning worse. So disabling it by setting action_space to None.
        value_loss, action_loss, dist_entropy = self._ppo_trainer.update(self._rollout_storage,
                                                                         action_space=None)
        self._rollout_storage.after_update()
        self._train_step_id += 1

        logs.extend([
            {"type": "scalar", "tag": "value_loss", "value": value_loss},
            {"type": "scalar", "tag": "action_loss", "value": action_loss},
            {"type": "scalar", "tag": "dist_entropy", "value": dist_entropy},
        ])
        
        # ADDED: include any intervention-emitted metrics (eg. ReDo dormant fraction)
        if self._intervention is not None and hasattr(self._intervention, "drain_logs"):
            logs.extend(self._intervention.drain_logs())

        if self._debug_reward_pipeline:
            logs.extend(self._ppo_trainer.pop_debug_logs())
            if self._debug_pending_logs:
                logs.extend(self._debug_pending_logs)
                self._debug_pending_logs = []

            stats = getattr(self._ppo_trainer, "_debug_last_stats", {})
            if stats.get("zero_delta_count", 0) >= 20:
                self._logger.warning(
                    "WARNING: optimizer step produced 0 parameter delta for %d updates; check optimizer/grad flow.",
                    stats.get("zero_delta_count")
                )
            if stats.get("grad_has_nan") or stats.get("grad_has_inf"):
                self._logger.warning("WARNING: NaN/Inf detected in gradients.")
            if stats.get("param_has_nan") or stats.get("param_has_inf"):
                self._logger.warning("WARNING: NaN/Inf detected in parameters after step.")

        return logs

    def _debug_reset_task(self, task_run_id: int):
        self._debug_task_run_id = task_run_id
        self._debug_step_counter = 0
        self._debug_last_log_step = 0
        self._debug_raw_reward_stats.reset()
        self._debug_shaped_reward_stats.reset()
        self._debug_done_count = 0
        self._debug_terminated_count = 0
        self._debug_truncated_count = 0
        self._debug_action_counts = None
        self._debug_action_total = 0
        self._debug_argmax_count = 0
        self._debug_entropy_sum = 0.0
        self._debug_obs_nan = 0
        self._debug_obs_inf = 0
        self._debug_logits_nan = 0
        self._debug_logits_inf = 0
        self._debug_value_nan = 0
        self._debug_value_inf = 0
        self._debug_logprob_nan = 0
        self._debug_logprob_inf = 0
        self._debug_pending_logs = []

    def _debug_emit_summary(self, force: bool = False):
        if not self._debug_reward_pipeline:
            return
        should_trace = self._debug_step_counter <= self._debug_first_steps
        should_log = (self._debug_step_counter - self._debug_last_log_step) >= self._debug_interval
        if not (force or should_trace or should_log):
            return

        self._debug_last_log_step = self._debug_step_counter

        raw_stats = self._debug_raw_reward_stats.summary()
        shaped_stats = self._debug_shaped_reward_stats.summary()
        argmax_frac = (self._debug_argmax_count / max(1, self._debug_action_total)) if self._debug_action_total else 0.0
        entropy_mean = (self._debug_entropy_sum / max(1, self._debug_action_total)) if self._debug_action_total else 0.0

        self._logger.info(
            "[DEBUG] Task %s window(%d steps): raw_reward nonzero=%d mean=%.4f max=%.4f | "
            "shaped_reward nonzero=%d mean=%.4f max=%.4f | done=%d term=%d trunc=%d | entropy=%.4f | argmax_frac=%.4f",
            self._debug_task_run_id,
            self._debug_step_counter,
            raw_stats["nonzero_count"],
            raw_stats["mean"],
            raw_stats["max"],
            shaped_stats["nonzero_count"],
            shaped_stats["mean"],
            shaped_stats["max"],
            self._debug_done_count,
            self._debug_terminated_count,
            self._debug_truncated_count,
            entropy_mean,
            argmax_frac,
        )

        if raw_stats["nonzero_count"] > 0 and shaped_stats["nonzero_count"] == 0:
            self._logger.warning(
                "WARNING: reward is being zeroed after wrappers; inspect clip/normalize wrappers."
            )

        if argmax_frac > 0.99 and self._debug_step_counter > max(10, self._debug_first_steps):
            self._logger.warning(
                "WARNING: action sampling appears near-deterministic (argmax_frac=%.3f)", argmax_frac
            )

        self._debug_pending_logs.extend([
            {"type": "scalar", "tag": "debug/raw_reward_mean", "value": raw_stats["mean"]},
            {"type": "scalar", "tag": "debug/raw_reward_nonzero_frac", "value": raw_stats["nonzero_count"] / max(1, raw_stats["count"])},
            {"type": "scalar", "tag": "debug/raw_reward_max", "value": raw_stats["max"]},
            {"type": "scalar", "tag": "debug/raw_reward_nan", "value": int(raw_stats["nan"])},
            {"type": "scalar", "tag": "debug/raw_reward_inf", "value": int(raw_stats["inf"])},
            {"type": "scalar", "tag": "debug/shaped_reward_mean", "value": shaped_stats["mean"]},
            {"type": "scalar", "tag": "debug/shaped_reward_nonzero_frac", "value": shaped_stats["nonzero_count"] / max(1, shaped_stats["count"])},
            {"type": "scalar", "tag": "debug/shaped_reward_max", "value": shaped_stats["max"]},
            {"type": "scalar", "tag": "debug/shaped_reward_nan", "value": int(shaped_stats["nan"])},
            {"type": "scalar", "tag": "debug/shaped_reward_inf", "value": int(shaped_stats["inf"])},
            {"type": "scalar", "tag": "debug/done_count", "value": int(self._debug_done_count)},
            {"type": "scalar", "tag": "debug/terminated_count", "value": int(self._debug_terminated_count)},
            {"type": "scalar", "tag": "debug/truncated_count", "value": int(self._debug_truncated_count)},
            {"type": "scalar", "tag": "debug/argmax_frac", "value": float(argmax_frac)},
            {"type": "scalar", "tag": "debug/entropy_mean", "value": float(entropy_mean)},
            {"type": "scalar", "tag": "debug/obs_nan", "value": int(self._debug_obs_nan)},
            {"type": "scalar", "tag": "debug/obs_inf", "value": int(self._debug_obs_inf)},
            {"type": "scalar", "tag": "debug/logits_nan", "value": int(self._debug_logits_nan)},
            {"type": "scalar", "tag": "debug/logits_inf", "value": int(self._debug_logits_inf)},
            {"type": "scalar", "tag": "debug/value_nan", "value": int(self._debug_value_nan)},
            {"type": "scalar", "tag": "debug/value_inf", "value": int(self._debug_value_inf)},
            {"type": "scalar", "tag": "debug/logprob_nan", "value": int(self._debug_logprob_nan)},
            {"type": "scalar", "tag": "debug/logprob_inf", "value": int(self._debug_logprob_inf)},
        ])

        if self._debug_action_counts is not None:
            total = max(1, self._debug_action_total)
            max_actions = min(self._debug_max_actions, len(self._debug_action_counts))
            for i in range(max_actions):
                self._debug_pending_logs.append({
                    "type": "scalar",
                    "tag": f"debug/action_frac/{i}",
                    "value": float(self._debug_action_counts[i] / total),
                })

        # Reset window stats
        self._debug_raw_reward_stats.reset()
        self._debug_shaped_reward_stats.reset()
        self._debug_done_count = 0
        self._debug_terminated_count = 0
        self._debug_truncated_count = 0
        self._debug_action_counts = None
        self._debug_action_total = 0
        self._debug_argmax_count = 0
        self._debug_entropy_sum = 0.0
        self._debug_obs_nan = 0
        self._debug_obs_inf = 0
        self._debug_logits_nan = 0
        self._debug_logits_inf = 0
        self._debug_value_nan = 0
        self._debug_value_inf = 0
        self._debug_logprob_nan = 0
        self._debug_logprob_inf = 0

    def save(self, output_path_dir, cycle_id, task_id, task_total_steps):
        checkpoint_data = {
                "model_state_dict": self._actor_critic.state_dict(),
                "optimizer_state_dict": self._ppo_trainer.optimizer.state_dict(),
            }
        model_path = os.path.join(output_path_dir, "actor_critic.pt")
        torch.save(checkpoint_data, model_path)

    def load(self, output_path_dir):
        model_path = os.path.join(output_path_dir, "actor_critic.pt")
        if os.path.exists(model_path):
            checkpoint_data = torch.load(model_path)
            self._actor_critic.load_state_dict(checkpoint_data["model_state_dict"])
            self._ppo_trainer.optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
