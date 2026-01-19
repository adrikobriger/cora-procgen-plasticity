import torch
import numpy as np
import logging
from collections import deque
from continual_rl.experiments.environment_runners.parallel_env import ParallelEnv
from continual_rl.experiments.environment_runners.environment_runner_base import EnvironmentRunnerBase
import copy
from continual_rl.utils.debug_stats import RollingStats


class EnvironmentRunnerBatch(EnvironmentRunnerBase):
    """
    Passes a batch of observations into the policy, gets a batch of actions out, and runs the environments in parallel.

    The arguments provided to __init__ are from the policy.
    The arguments provided to collect_data are from the task.
    """
    def __init__(self, policy, num_parallel_envs, timesteps_per_collection, render_collection_freq=None,
                 output_dir=None):
        super().__init__()
        self._policy = policy
        self._num_parallel_envs = num_parallel_envs
        self._timesteps_per_collection = timesteps_per_collection
        self._render_collection_freq = render_collection_freq  # In timesteps
        self._output_dir = output_dir

        self._parallel_env = None
        self._last_observations = None  # To allow returning mid-episode
        self._last_timestep_data = None  # Always stores the last thing seen, even across "dones"
        self._cumulative_rewards = np.array([0 for _ in range(num_parallel_envs)], dtype=np.float64)

        # Used to determine what to save off to logs and when
        self._observations_to_render = []
        self._timesteps_since_last_render = 0
        self._total_timesteps = 0
        
        self._logger = logging.getLogger(__name__)

        # Diagnostics for zero-reward task issue.
        self._debug_cfg = getattr(policy, "get_debug_reward_pipeline_config", lambda: {})()
        self._debug_enabled = bool(self._debug_cfg.get("enabled", False))
        self._debug_interval = int(self._debug_cfg.get("interval", 10000) or 10000)
        self._debug_first_steps = int(self._debug_cfg.get("first_steps", 0) or 0)
        self._debug_window_steps = 0
        self._debug_total_steps = 0
        self._debug_last_log_step = 0
        self._debug_raw_reward_stats = RollingStats()
        self._debug_done_count = 0
        self._debug_terminated_count = 0
        self._debug_truncated_count = 0
        self._debug_episode_end_count = 0
        self._debug_episode_returns = deque(maxlen=100)
        self._debug_episode_lengths = deque(maxlen=100)
        self._debug_current_episode_lengths = np.zeros(num_parallel_envs, dtype=np.int32)
        self._debug_task_info_logged = False

    def _preprocess_raw_observations(self, preprocessor, raw_observations):
        return preprocessor.preprocess(raw_observations)

    def _initialize_envs(self, env_spec, preprocessor):
        if self._parallel_env is None:
            env_specs = [env_spec for _ in range(self._num_parallel_envs)]
            self._parallel_env = ParallelEnv(env_specs, self._output_dir)

        # Initialize the observation time-batch with n of the first observation.
        raw_observations = self._parallel_env.reset()
        processed_observations = self._preprocess_raw_observations(preprocessor, raw_observations)
        return processed_observations

    def _debug_log_task_info(self, task_spec):
        if not self._debug_enabled or self._debug_task_info_logged:
            return

        env = getattr(self._parallel_env, "_local_env", None)
        env_id = None
        wrappers = []
        num_levels = None
        start_level = None
        distribution_mode = None
        if env is not None:
            try:
                env_id = getattr(getattr(env, "spec", None), "id", None)
            except Exception:
                env_id = None

            # unwrap to get base env attributes
            base_env = env
            while hasattr(base_env, "env"):
                wrappers.append(base_env.__class__.__name__)
                base_env = base_env.env
            wrappers.append(base_env.__class__.__name__)

            for attr in ("num_levels", "start_level", "distribution_mode"):
                if hasattr(base_env, attr):
                    try:
                        val = getattr(base_env, attr)
                        if attr == "num_levels":
                            num_levels = val
                        elif attr == "start_level":
                            start_level = val
                        elif attr == "distribution_mode":
                            distribution_mode = val
                    except Exception:
                        pass

        self._logger.info(
            "[DEBUG] Task %s | action_space_id=%s | eval=%s | env=%s | wrappers=%s | num_levels=%s start_level=%s distribution_mode=%s",
            task_spec.task_id,
            task_spec.action_space_id,
            task_spec.eval_mode,
            env_id,
            "->".join(wrappers) if wrappers else "unknown",
            num_levels,
            start_level,
            distribution_mode,
        )

        self._debug_task_info_logged = True

    def _reset_env(self, env_id):
        """
        ParallelEnv doesn't readily expose manually resetting an environment, so doing that here.
        """
        if env_id == 0:
            observation = self._parallel_env.envs[0].reset()
        else:
            local = self._parallel_env.locals[env_id-1]
            local.send(("reset", None))
            observation = local.recv()

        return observation

    def _render_video(self, preprocessor):
        """
        Only renders if it's time, per the render_collection_freq
        """
        video_log = None

        if self._render_collection_freq is not None and \
                self._timesteps_since_last_render >= self._render_collection_freq:
            try:
                # As with resetting, remove the last element because it's from the next episode
                rendered_episode = preprocessor.render_episode(self._observations_to_render[:-1])
                video_log = {"type": "video",
                             "tag": "behavior_video",
                             "value": rendered_episode,
                             "timestep": self._total_timesteps}
            except NotImplementedError:
                # If the task hasn't implemented rendering, it may simply not be feasible, so just
                # let it go.
                pass

            self._timesteps_since_last_render = 0

        # Reset the observations to render except keep the last frame because it's from the next episode
        self._observations_to_render = [self._observations_to_render[-1]]
        return video_log

    def collect_data(self, task_spec):
        """
        Passes observations to the policy of shape [#envs, time, **env.observation_shape]
        """
        env_spec = task_spec.env_spec
        preprocessor = task_spec.preprocessor
        task_id = task_spec.task_id
        action_space_id = task_spec.action_space_id
        eval_mode = task_spec.eval_mode
        return_after_episode_num = task_spec.return_after_episode_num

        # If the task requires fewer collections than the policy specifies, only collect that number
        timesteps_to_collect = min(self._timesteps_per_collection, task_spec.num_timesteps)

        # The per-environment data is contained within each TimestepData object, stored within per_timestep_data
        per_timestep_data = []
        returns_to_report = []
        logs_to_report = []  # {tag, type ("video", "scalar"), value, timestep}
        num_timesteps = 0

        # Grabbed the saved-off observations, if applicable.
        if self._last_observations is None:
            processed_observations = self._initialize_envs(env_spec, preprocessor)
        else:
            processed_observations = self._last_observations

        if self._debug_enabled:
            self._debug_log_task_info(task_spec)

        for timestep_id in range(timesteps_to_collect):
            actions, timestep_data = self._policy.compute_action(processed_observations,
                                                                 task_id,
                                                                 action_space_id,
                                                                 self._last_timestep_data,
                                                                 eval_mode)

            # --- FIX: procgen/gym3 expects actions shape (num_envs,), not (num_envs, 1)
            import numpy as np
            if hasattr(actions, "detach"):  # torch tensor
                actions = actions.detach().cpu().numpy()
            actions = np.asarray(actions)

            if actions.ndim == 2 and actions.shape[1] == 1:
                actions = actions[:, 0]  # <-- squeeze (N,1) -> (N,)
            # --- end fix

            # ParallelEnv automatically resets the env and returns the new observation when a "done" occurs
            result = self._parallel_env.step(actions)
            raw_observations, rewards, dones, infos = list(result)

            self._total_timesteps += self._num_parallel_envs
            self._last_timestep_data = timestep_data
            processed_observations = self._preprocess_raw_observations(preprocessor, raw_observations)
            self._last_observations = processed_observations  # Save it off so we can resume if we finish the collection

            # If we're expecting the environment to keep track of this for us (EpisodicLifeEnv) use that.
            # Otherwise accumulate ourselves
            if "episode_return" in infos[0]:
                for env_id, env_info in enumerate(infos):
                    # The episode return will be None if the episode is not yet over, but Nones can't be stored in
                    # numpy arrays, so convert to np.nan.
                    val_to_store = env_info["episode_return"] if env_info["episode_return"] is not None else np.nan
                    self._cumulative_rewards[env_id] = val_to_store
            else:
                self._cumulative_rewards += np.array(rewards)

            # For logging video, take the first env's most recent observation and save it.
            # Without the deepcopy, the reset overwrites the end of observations_to_render
            self._observations_to_render.append(copy.deepcopy(processed_observations[0][-1]))
            self._timesteps_since_last_render += self._num_parallel_envs

            if self._debug_enabled:
                self._debug_current_episode_lengths += 1

            for env_id, done in enumerate(dones):
                if done:
                    # It may not be a "real" done (e.g. EpisodicLifeEnv), so only log it out if it is
                    if not np.isnan(self._cumulative_rewards[env_id]):
                        returns_to_report.append(self._cumulative_rewards[env_id])

                        if self._debug_enabled:
                            self._debug_episode_returns.append(float(self._cumulative_rewards[env_id]))
                            self._debug_episode_lengths.append(int(self._debug_current_episode_lengths[env_id]))
                            self._debug_episode_end_count += 1

                    self._cumulative_rewards[env_id] = 0
                    if self._debug_enabled:
                        self._debug_current_episode_lengths[env_id] = 0

                    # Save off observations to enable viewing behavior
                    if env_id == 0:
                        render_log = self._render_video(preprocessor)
                        if render_log is not None:
                            logs_to_report.append(render_log)

            # Finish populating the info to store with the collected data
            timestep_data.reward = rewards
            timestep_data.done = dones
            timestep_data.info = infos
            per_timestep_data.append(timestep_data)
            num_timesteps += self._num_parallel_envs

            if self._debug_enabled:
                self._debug_window_steps += self._num_parallel_envs
                self._debug_total_steps += self._num_parallel_envs
                self._debug_raw_reward_stats.update(rewards)
                self._debug_done_count += int(np.sum(dones))

                for info in infos:
                    if isinstance(info, dict):
                        if info.get("_terminated") is True or info.get("terminated") is True:
                            self._debug_terminated_count += 1
                        if info.get("_truncated") is True or info.get("truncated") is True or info.get("TimeLimit.truncated") is True:
                            self._debug_truncated_count += 1

                if self._debug_enabled:
                    assert len(dones) == self._num_parallel_envs, "terminal shape mismatch"

                should_trace = self._debug_total_steps <= self._debug_first_steps
                should_log = (self._debug_total_steps - self._debug_last_log_step) >= self._debug_interval

                if should_trace or should_log:
                    self._debug_last_log_step = self._debug_total_steps
                    raw_stats = self._debug_raw_reward_stats.summary()
                    done_rate = (self._debug_done_count / max(1, self._debug_window_steps))
                    ep_mean = float(np.mean(self._debug_episode_returns)) if self._debug_episode_returns else 0.0
                    ep_min = float(np.min(self._debug_episode_returns)) if self._debug_episode_returns else 0.0
                    ep_max = float(np.max(self._debug_episode_returns)) if self._debug_episode_returns else 0.0

                    self._logger.info(
                        "[DEBUG] Task %s window(%d steps): raw_reward nonzero=%d mean=%.4f max=%.4f | "
                        "episode_ends=%d done_rate=%.4f terminated=%d truncated=%d ep_return_mean=%.4f",
                        task_spec.task_id,
                        self._debug_window_steps,
                        raw_stats["nonzero_count"],
                        raw_stats["mean"],
                        raw_stats["max"],
                        self._debug_episode_end_count,
                        done_rate,
                        self._debug_terminated_count,
                        self._debug_truncated_count,
                        ep_mean,
                    )

                    logs_to_report.extend([
                        {"type": "scalar", "tag": "debug/raw_reward_mean", "value": raw_stats["mean"]},
                        {"type": "scalar", "tag": "debug/raw_reward_nonzero_frac", "value": raw_stats["nonzero_count"] / max(1, raw_stats["count"])},
                        {"type": "scalar", "tag": "debug/raw_reward_max", "value": raw_stats["max"]},
                        {"type": "scalar", "tag": "debug/done_rate", "value": done_rate},
                        {"type": "scalar", "tag": "debug/episode_ends", "value": int(self._debug_episode_end_count)},
                        {"type": "scalar", "tag": "debug/episode_return_mean", "value": ep_mean},
                        {"type": "scalar", "tag": "debug/episode_return_min", "value": ep_min},
                        {"type": "scalar", "tag": "debug/episode_return_max", "value": ep_max},
                    ])

                    if self._debug_episode_end_count == 0 and self._debug_total_steps >= 10000:
                        self._logger.warning(
                            "WARNING: no episode terminations detected; check terminated/truncated handling."
                        )

                    self._debug_raw_reward_stats.reset()
                    self._debug_done_count = 0
                    self._debug_terminated_count = 0
                    self._debug_truncated_count = 0
                    self._debug_episode_end_count = 0
                    self._debug_window_steps = 0

            if return_after_episode_num is not None and len(returns_to_report) >= return_after_episode_num:
                break

        # Tasks expect a list of lists for timestep data, to support different forms of parallelization, so return
        # per_timestep_data as a list
        return num_timesteps, [per_timestep_data], returns_to_report, logs_to_report

    def cleanup(self, task_spec):
        """
        Safely cleanup the parallel environment. Idempotent and handles None gracefully.
        Called even if environment construction failed, so must be robust.
        """
        if self._parallel_env is None:
            self._logger.debug("Cleanup called but parallel environment was never initialized. "
                             "This can occur if environment construction failed early.")
            return
        
        try:
            self._parallel_env.close()
            self._logger.debug("Successfully closed parallel environment.")
        except Exception as e:
            self._logger.warning(f"Exception occurred while closing parallel environment: {e}")
        finally:
            # Clear the reference to prevent accidental reuse or double-close
            self._parallel_env = None

