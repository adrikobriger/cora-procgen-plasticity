import os
import json
import numbers
import numpy as np
import torch
import torch.nn as nn

from continual_rl.experiments.run_metadata import RunMetadata
from continual_rl.utils.utils import Utils
from continual_rl.utils.common_exceptions import OutputDirectoryNotSetException


class InvalidTaskAttributeException(Exception):
    def __init__(self, error_msg):
        super().__init__(error_msg)


class Experiment(object):
    """
    Experiment runner with continual evaluation, forgetting metrics, and optional plasticity diagnostics
    via activation effective-rank.
    """

    # ---------------------------------------------------------------------
    # Effective-rank configuration (activation-based)
    # ---------------------------------------------------------------------
    # NOTE:
    # - "features" is captured via a forward-hook (module OUTPUT).
    # - "actor_in" and "critic_in" are captured via forward PRE-hooks (module INPUTS).
    # - Patterns are substring matches on model.named_modules() keys.
    EFFECTIVE_RANK_LAYER_PATTERNS = {
        "features": ["base.main"],
        "actor_in": ["base.actor"],
        "critic_in": ["base.critic"],
    }

    MAX_EFFECTIVE_RANK_BATCHES = 10
    MAX_EFFECTIVE_RANK_ROWS = 512

    # Guardrail: if activation dim is huge, project down before SVD
    MAX_EFFECTIVE_RANK_COLS = 4096
    EFFECTIVE_RANK_PROJ_DIM = 1024
    EFFECTIVE_RANK_PROJ_SEED = 0

    # Whether to center activations (recommended for stability/comparability)
    EFFECTIVE_RANK_CENTER = True

    def __init__(self, tasks, continual_testing_freq=None, cycle_count=1):
        self.tasks = tasks
        self.action_spaces = self._get_action_spaces(self.tasks)
        self.observation_space = self._get_common_attribute(
            [task.observation_space for task in self.tasks]
        )
        self.task_ids = [task.task_id for task in tasks]
        self._output_dir = None
        self._continual_testing_freq = continual_testing_freq
        self._cycle_count = cycle_count
        self._core_logger = None

        # Continual-eval returns used for forgetting metrics
        self._eval_last_return_mean = {}
        self._eval_last_return_iqm = {}
        self._ref_return_end_of_task_mean = {}
        self._ref_return_end_of_task_iqm = {}

        # Effective-rank tracking
        self._effective_rank_history = []          # list of (timestep, layer_key, eff_rank)
        self._effective_rank_by_layer = {}         # latest per-layer_key
        self._activation_buffers = {}              # layer_key -> list[np.ndarray]  (2D)
        self._activation_rows_total = {}           # layer_key -> rows cached

        # Hook management
        self._rank_layer_modules = {}              # layer_key -> (module_name, module)
        self._rank_hook_handles = []               # forward hooks
        self._rank_prehook_handles = []            # forward pre-hooks
        self._rank_hook_model_id = None
        self._rank_layers_logged = False

        # Projection cache to avoid rebuilding matrices every eval checkpoint
        self._proj_cache = {}                      # (in_dim, proj_dim, seed) -> proj_matrix

        # Temporary: last aggregate stats for the most recent compute call
        self._last_effective_rank_stats = None

    # ---------------------------------------------------------------------
    # Output directory and logging
    # ---------------------------------------------------------------------
    def set_output_dir(self, output_dir):
        self._output_dir = output_dir

    @property
    def output_dir(self):
        if self._output_dir is None:
            raise OutputDirectoryNotSetException(
                "Output directory not set, but is attempting to be used. Call set_output_dir."
            )
        return self._output_dir

    @property
    def _logger(self):
        return Utils.create_logger(f"{self.output_dir}/core_process.log")

    def _console(self, msg: str) -> None:
        print(msg, flush=True)

    # ---------------------------------------------------------------------
    # Effective-rank utilities
    # ---------------------------------------------------------------------
    @staticmethod
    def _compute_effective_rank_from_matrix(matrix: np.ndarray, epsilon: float = 1e-10) -> float:
        """
        Effective rank via entropy of normalized singular-value energy:

            p_i = sigma_i^2 / sum_j sigma_j^2
            H   = -sum_i p_i log(p_i)
            er  = exp(H)

        Returns:
            float or None (if matrix is degenerate/unusable)
        """
        if matrix is None or not isinstance(matrix, np.ndarray):
            return None
        if matrix.ndim != 2:
            return None
        if matrix.shape[0] < 2 or matrix.shape[1] < 2:
            return None

        try:
            s = np.linalg.svd(matrix, compute_uv=False)
        except np.linalg.LinAlgError:
            return None

        s = s[s > epsilon]
        if s.size == 0:
            return 0.0

        energy = s * s
        total = float(np.sum(energy))
        if total < epsilon:
            return 0.0

        p = energy / total
        entropy = -float(np.sum(p * np.log(p + epsilon)))
        return float(np.exp(entropy))

    def _get_or_make_projection(self, in_dim: int) -> np.ndarray:
        key = (int(in_dim), int(self.EFFECTIVE_RANK_PROJ_DIM), int(self.EFFECTIVE_RANK_PROJ_SEED))
        proj = self._proj_cache.get(key, None)
        if proj is None:
            rng = np.random.default_rng(self.EFFECTIVE_RANK_PROJ_SEED)
            proj = rng.normal(size=(in_dim, self.EFFECTIVE_RANK_PROJ_DIM)).astype(np.float32)
            self._proj_cache[key] = proj
        return proj

    def _select_rank_layers(self, model):
        """
        Best-effort module selection based on EFFECTIVE_RANK_LAYER_PATTERNS.
        We prefer "features" first, then try actor/critic inputs, ensuring we do not reuse the same module.
        """
        selected = {}
        try:
            named_modules = list(model.named_modules()) if hasattr(model, "named_modules") else []
            if not named_modules:
                return {}

            used = set()

            # 1) features (preferred)
            for pattern in self.EFFECTIVE_RANK_LAYER_PATTERNS.get("features", []):
                for name, module in named_modules:
                    if pattern in name:
                        selected["features"] = (name, module)
                        used.add(module)
                        break
                if "features" in selected:
                    break

            # 2) actor_in / critic_in (optional)
            for key in ("actor_in", "critic_in"):
                for pattern in self.EFFECTIVE_RANK_LAYER_PATTERNS.get(key, []):
                    for name, module in named_modules:
                        if pattern in name and module not in used:
                            selected[key] = (name, module)
                            used.add(module)
                            break
                    if key in selected:
                        break

        except Exception:
            return {}

        if not selected:
            self._logger.warning("Effective-rank: no modules matched patterns; skipping activation capture.")
        return selected

    def _activation_to_matrix(self, tensor) -> torch.Tensor:
        """
        Convert a model tensor (input or output) to a 2D tensor [N, D] suitable for SVD.
        """
        if tensor is None:
            return None

        # Handle tuples/lists produced by some modules
        if isinstance(tensor, (tuple, list)) and tensor:
            tensor = tensor[0]

        if not torch.is_tensor(tensor):
            return None

        act = tensor.detach()

        # Common RL shapes:
        #  [B, C, H, W] -> [B, C*H*W]
        #  [T, B, D]    -> [T*B, D]
        #  [B, D]       -> [B, D]
        #  [D]          -> [1, D]
        if act.dim() == 4:
            act = act.reshape(act.size(0), -1)
        elif act.dim() == 3:
            act = act.reshape(-1, act.size(-1))
        elif act.dim() == 2:
            pass
        elif act.dim() == 1:
            act = act.view(1, -1)
        else:
            act = act.reshape(act.size(0), -1) if act.dim() > 1 else act.view(1, -1)

        return act

    def _collect_activation(self, layer_key: str, tensor) -> None:
        """
        Collect up to MAX_EFFECTIVE_RANK_BATCHES and MAX_EFFECTIVE_RANK_ROWS rows total.
        """
        try:
            if layer_key not in self._activation_buffers:
                self._activation_buffers[layer_key] = []
                self._activation_rows_total[layer_key] = 0

            if len(self._activation_buffers[layer_key]) >= self.MAX_EFFECTIVE_RANK_BATCHES:
                return

            act = self._activation_to_matrix(tensor)
            if act is None:
                return

            rows_left = self.MAX_EFFECTIVE_RANK_ROWS - self._activation_rows_total[layer_key]
            if rows_left <= 0:
                return

            if act.size(0) > rows_left:
                act = act[:rows_left]

            self._activation_buffers[layer_key].append(act.cpu().float().numpy())
            self._activation_rows_total[layer_key] += int(act.size(0))
        except Exception:
            # best-effort metrics; never break training/eval
            return

    def _reset_rank_buffers(self) -> None:
        self._activation_buffers = {}
        self._activation_rows_total = {}

    def _install_rank_hooks(self, model) -> None:
        """
        Installs hooks once per model instance:
          - forward-hook for "features" (module output)
          - forward PRE-hook for actor_in / critic_in (module input)
        """
        if model is None:
            return

        model_id = id(model)
        if self._rank_hook_model_id == model_id and (self._rank_hook_handles or self._rank_prehook_handles):
            return

        # Remove old hooks
        for h in self._rank_hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        for h in self._rank_prehook_handles:
            try:
                h.remove()
            except Exception:
                pass

        self._rank_hook_handles = []
        self._rank_prehook_handles = []
        self._rank_hook_model_id = model_id
        self._rank_layers_logged = False

        self._rank_layer_modules = self._select_rank_layers(model)
        if not self._rank_layer_modules:
            return

        # Install hooks per layer_key
        for layer_key, (name, module) in self._rank_layer_modules.items():
            try:
                if layer_key == "features":
                    # forward hook captures OUTPUT of module
                    def _fwd_hook(_module, _inp, out, _lk=layer_key):
                        self._collect_activation(_lk, out)

                    self._rank_hook_handles.append(module.register_forward_hook(_fwd_hook))
                else:
                    # pre-hook captures INPUTS to module (actor_in/critic_in)
                    def _pre_hook(_module, inputs, _lk=layer_key):
                        if isinstance(inputs, (tuple, list)) and inputs:
                            self._collect_activation(_lk, inputs[0])
                        else:
                            self._collect_activation(_lk, inputs)

                    self._rank_prehook_handles.append(module.register_forward_pre_hook(_pre_hook))
            except Exception:
                continue

        if not self._rank_layers_logged:
            matched = {k: v[0] for k, v in self._rank_layer_modules.items()}
            missing = [k for k in self.EFFECTIVE_RANK_LAYER_PATTERNS.keys() if k not in matched]
            self._logger.info(f"Effective-rank layer matches: {matched}")
            if missing:
                self._logger.info(f"Effective-rank missing layer keys: {missing}")
            if list(matched.keys()) == ["features"]:
                self._logger.warning("Effective-rank: only 'features' matched (actor/critic not found).")
            self._rank_layers_logged = True

    def _get_policy_model(self, policy):
        model = None

        if hasattr(policy, "_actor_critic") and policy._actor_critic is not None:
            model = policy._actor_critic
        elif hasattr(policy, "impala_trainer"):
            trainer = policy.impala_trainer
            if hasattr(trainer, "learner_model") and trainer.learner_model is not None:
                model = trainer.learner_model
            elif hasattr(trainer, "actor_model") and trainer.actor_model is not None:
                model = trainer.actor_model

        if model is None:
            model_attr_names = [
                "model", "network", "actor_critic", "net", "policy_net",
                "q_network", "actor", "actor_net"
            ]
            for attr_name in model_attr_names:
                if hasattr(policy, attr_name):
                    attr = getattr(policy, attr_name)
                    if attr is not None:
                        model = attr
                        break

        return model

    def _compute_effective_rank(
        self,
        policy,
        total_timesteps: int,
        summary_writer,
        log_prefix: str = "effective_rank",
        do_log: bool = True,
    ) -> dict:
        """
        Compute activation effective-rank for the activations collected during the most recent eval run.
        """
        effective_ranks = {}
        self._last_effective_rank_stats = None

        model = self._get_policy_model(policy)
        if model is None:
            return effective_ranks

        try:
            self._install_rank_hooks(model)

            if not self._activation_buffers:
                return effective_ranks

            layer_ranks = []
            for layer_key, buffers in self._activation_buffers.items():
                if not buffers:
                    continue

                try:
                    X = np.concatenate(buffers, axis=0).astype(np.float32, copy=False)
                except Exception:
                    continue

                # Optional centering improves stability
                if self.EFFECTIVE_RANK_CENTER and X.shape[0] >= 2:
                    X = X - X.mean(axis=0, keepdims=True)

                # Projection guardrail for extremely wide matrices
                if X.shape[1] > self.MAX_EFFECTIVE_RANK_COLS:
                    proj = self._get_or_make_projection(X.shape[1])
                    try:
                        X = X @ proj
                    except Exception:
                        continue

                if X.shape[0] < 2 or X.shape[1] < 2:
                    continue

                eff_rank = self._compute_effective_rank_from_matrix(X)
                if eff_rank is None:
                    continue

                effective_ranks[layer_key] = eff_rank
                layer_ranks.append(eff_rank)
                self._effective_rank_history.append((total_timesteps, layer_key, eff_rank))

                if do_log and log_prefix:
                    summary_writer.add_scalar(
                        f"{log_prefix}/layer/{layer_key}",
                        eff_rank,
                        global_step=total_timesteps,
                    )

            if layer_ranks:
                avg_rank = float(np.mean(layer_ranks))
                min_rank = float(np.min(layer_ranks))
                max_rank = float(np.max(layer_ranks))
                self._last_effective_rank_stats = {"avg": avg_rank, "min": min_rank, "max": max_rank}

                if do_log and log_prefix:
                    summary_writer.add_scalar(f"{log_prefix}/layers_avg", avg_rank, global_step=total_timesteps)
                    summary_writer.add_scalar(f"{log_prefix}/layers_min", min_rank, global_step=total_timesteps)
                    summary_writer.add_scalar(f"{log_prefix}/layers_max", max_rank, global_step=total_timesteps)
                    summary_writer.flush()

        except Exception as e:
            self._logger.warning(f"Error computing effective rank: {e}")

        self._effective_rank_by_layer = effective_ranks
        return effective_ranks

    def get_effective_rank_history(self) -> list:
        return self._effective_rank_history.copy()

    def get_current_effective_ranks(self) -> dict:
        return self._effective_rank_by_layer.copy()

    def save_effective_rank_history(self, filepath: str = None) -> str:
        if filepath is None:
            filepath = os.path.join(self.output_dir, "effective_rank_history.json")

        history_data = {
            "measurements": [
                {"timestep": t, "layer": layer, "effective_rank": rank}
                for t, layer, rank in self._effective_rank_history
            ],
            "latest_by_layer": self._effective_rank_by_layer,
        }

        with open(filepath, "w") as f:
            json.dump(history_data, f, indent=2)

        self._logger.info(f"Saved effective rank history to {filepath}")
        return filepath

    # ---------------------------------------------------------------------
    # Task attribute helpers
    # ---------------------------------------------------------------------
    @classmethod
    def _get_action_spaces(cls, tasks):
        action_space_map = {}
        for task in tasks:
            if task.action_space_id not in action_space_map:
                action_space_map[task.action_space_id] = task.action_space
            elif action_space_map[task.action_space_id] != task.action_space:
                raise InvalidTaskAttributeException(
                    f"Action sizes were mismatched for task {task.action_space_id}"
                )
        return action_space_map

    @classmethod
    def _get_common_attribute(cls, task_attributes):
        common_attribute = None
        for task_attribute in task_attributes:
            if common_attribute is None:
                common_attribute = task_attribute
            if task_attribute != common_attribute:
                raise InvalidTaskAttributeException("Tasks do not have a common attribute.")
        return common_attribute

    # ---------------------------------------------------------------------
    # Continual evaluation and forgetting metrics
    # ---------------------------------------------------------------------
    def _run_continual_eval(self, task_run_id, policy, summary_writer, total_timesteps, set_ref_task_run_id=None):
        model_for_hooks = self._get_policy_model(policy)
        if model_for_hooks is not None:
            self._install_rank_hooks(model_for_hooks)
        else:
            self._logger.warning("Effective-rank: no model found; skipping activation capture.")

        def _iqm_local(xs):
            xs = np.asarray(xs, dtype=np.float64)
            if xs.size == 0:
                return np.nan
            xs = np.sort(xs)
            n = xs.size
            lo = int(np.floor(0.25 * n))
            hi = int(np.ceil(0.75 * n))
            if hi <= lo:
                lo = 0
                hi = n
            return float(xs[lo:hi].mean())

        per_task_avgs = []
        per_task_mins = []
        per_task_maxs = []

        for test_task_run_id, test_task in enumerate(self.tasks):
            if not test_task._task_spec.with_continual_eval:
                continue

            # Clear buffers so activations are from THIS continual-eval run
            if model_for_hooks is not None:
                self._reset_rank_buffers()

            episodes_cap = None
            if hasattr(test_task, "_continual_eval_task_spec"):
                episodes_cap = getattr(test_task._continual_eval_task_spec, "return_after_episode_num", None)
            if episodes_cap is None:
                episodes_cap = float("inf")

            self._logger.info(f"Continual eval for task: {test_task_run_id}")

            test_task_runner = self.tasks[test_task_run_id].continual_eval(
                test_task_run_id,
                policy,
                summary_writer,
                output_dir=self.output_dir,
                timestep_log_offset=total_timesteps,
            )

            test_complete = False
            returns_all = []

            while not test_complete:
                try:
                    info = next(test_task_runner)
                    if not (isinstance(info, tuple) and len(info) == 2):
                        continue

                    _, data = info
                    if data is None:
                        continue

                    if isinstance(data, tuple) and len(data) == 2:
                        rewards, _ = data
                        if rewards is None:
                            continue

                        if isinstance(rewards, (list, tuple)):
                            for r in rewards:
                                if isinstance(r, numbers.Real):
                                    returns_all.append(float(r))
                                    if len(returns_all) >= episodes_cap:
                                        test_complete = True
                                        break
                        elif isinstance(rewards, numbers.Real):
                            returns_all.append(float(rewards))

                        if len(returns_all) >= episodes_cap:
                            test_complete = True
                            break

                except StopIteration:
                    test_complete = True

            # Effective rank from activations collected during this eval run
            self._compute_effective_rank(
                policy,
                total_timesteps,
                summary_writer,
                log_prefix=f"effective_rank/task_{test_task_run_id}",
                do_log=True,
            )

            if self._last_effective_rank_stats is not None:
                per_task_avgs.append(self._last_effective_rank_stats["avg"])
                per_task_mins.append(self._last_effective_rank_stats["min"])
                per_task_maxs.append(self._last_effective_rank_stats["max"])

            # Store eval returns for forgetting
            if returns_all:
                mean_ret = float(np.mean(returns_all))
                iqm_ret = _iqm_local(returns_all)
                self._eval_last_return_mean[test_task_run_id] = mean_ret
                self._eval_last_return_iqm[test_task_run_id] = iqm_ret

            self._logger.info(f"Completed continual eval for task: {test_task_run_id}")

        # Lock in reference return at end-of-task boundary (mean and IQM)
        if set_ref_task_run_id is not None:
            if set_ref_task_run_id in self._eval_last_return_mean:
                self._ref_return_end_of_task_mean[set_ref_task_run_id] = self._eval_last_return_mean[set_ref_task_run_id]
            if set_ref_task_run_id in self._eval_last_return_iqm:
                self._ref_return_end_of_task_iqm[set_ref_task_run_id] = self._eval_last_return_iqm[set_ref_task_run_id]

        # Isolated forgetting: ref_end_of_task(i) - current_eval(i)
        forgetting_mean_vals = []
        forgetting_iqm_vals = []

        for tid, ref in self._ref_return_end_of_task_mean.items():
            cur = self._eval_last_return_mean.get(tid, None)
            if cur is None:
                continue
            f = float(ref) - float(cur)
            forgetting_mean_vals.append(f)
            summary_writer.add_scalar(f"forgetting/isolated_task_mean/{tid}", f, global_step=total_timesteps)

        for tid, ref in self._ref_return_end_of_task_iqm.items():
            cur = self._eval_last_return_iqm.get(tid, None)
            if cur is None:
                continue
            f = float(ref) - float(cur)
            forgetting_iqm_vals.append(f)
            summary_writer.add_scalar(f"forgetting/isolated_task_iqm/{tid}", f, global_step=total_timesteps)

        if forgetting_mean_vals:
            summary_writer.add_scalar(
                "forgetting/isolated_avg_mean",
                float(np.mean(forgetting_mean_vals)),
                global_step=total_timesteps,
            )

        if forgetting_iqm_vals:
            summary_writer.add_scalar(
                "forgetting/isolated_avg_iqm",
                float(np.mean(forgetting_iqm_vals)),
                global_step=total_timesteps,
            )

        if forgetting_mean_vals or forgetting_iqm_vals:
            summary_writer.flush()

        # Across-task aggregate effective-rank at this checkpoint
        if per_task_avgs:
            summary_writer.add_scalar(
                "effective_rank/across_tasks_avg",
                float(np.mean(per_task_avgs)),
                global_step=total_timesteps,
            )
            summary_writer.add_scalar(
                "effective_rank/across_tasks_min",
                float(np.min(per_task_mins)),
                global_step=total_timesteps,
            )
            summary_writer.add_scalar(
                "effective_rank/across_tasks_max",
                float(np.max(per_task_maxs)),
                global_step=total_timesteps,
            )
            summary_writer.flush()

    # ---------------------------------------------------------------------
    # Main experiment execution
    # ---------------------------------------------------------------------
    def _run(self, policy, summary_writer):
        policy.load(self.output_dir)
        run_metadata = RunMetadata(self._output_dir)
        start_cycle_id = run_metadata.cycle_id
        start_task_id = run_metadata.task_id
        start_task_timesteps = run_metadata.task_timesteps

        total_train_timesteps = run_metadata.total_train_timesteps
        timesteps_per_save = policy.config.timesteps_per_save

        for cycle_id in range(start_cycle_id, self._cycle_count):
            for task_run_id, task in enumerate(self.tasks[start_task_id:], start=start_task_id):
                self._logger.info(f"Starting cycle {cycle_id} task {task_run_id}")
                self._console(f"[TASK] start | cycle={cycle_id} task={task_run_id}")

                if not task._task_spec.eval_mode:
                    policy.on_task_start(cycle_id=cycle_id, task_run_id=task_run_id)

                task_complete = False
                task_runner = task.run(
                    task_run_id,
                    policy,
                    summary_writer,
                    self.output_dir,
                    timestep_log_offset=total_train_timesteps,
                    task_timestep_start=start_task_timesteps,
                )

                task_timesteps = start_task_timesteps
                continual_freq = self._continual_testing_freq
                last_timestep_saved = None
                last_continual_testing_step = -10 * continual_freq if continual_freq is not None else None

                while not task_complete:
                    try:
                        task_timesteps, info = next(task_runner)

                        if (not task._task_spec.eval_mode) and (task_timesteps % 1024 == 0):
                            r = None
                            stats = {}
                            if isinstance(info, tuple) and len(info) == 2:
                                reward_list, metric_list = info
                                if reward_list:
                                    r = reward_list[-1]
                                if metric_list:
                                    for m in metric_list:
                                        if m.get("type") == "scalar":
                                            stats[m["tag"]] = m["value"]

                            r_str = f"{r:.3f}" if isinstance(r, (int, float)) else "NA"
                            vloss = stats.get("value_loss")
                            aloss = stats.get("action_loss")
                            ent = stats.get("dist_entropy")
                            vloss_str = f"{vloss:.4f}" if isinstance(vloss, (int, float)) else "NA"
                            aloss_str = f"{aloss:.4f}" if isinstance(aloss, (int, float)) else "NA"
                            ent_str = f"{ent:.3f}" if isinstance(ent, (int, float)) else "NA"

                            self._console(
                                f"[TRAIN] cycle={cycle_id} task={task_run_id} "
                                f"t={total_train_timesteps + task_timesteps} "
                                f"r={r_str} vloss={vloss_str} aloss={aloss_str} ent={ent_str}"
                            )

                    except StopIteration:
                        task_complete = True

                    if not task._task_spec.eval_mode:
                        if (
                            last_timestep_saved is None
                            or task_timesteps - last_timestep_saved >= timesteps_per_save
                            or task_complete
                        ):
                            run_metadata.save(cycle_id, task_run_id, task_timesteps, total_train_timesteps)
                            policy.save(self.output_dir, cycle_id, task_run_id, task_timesteps)

                            if task_complete:
                                task_boundary_dir = os.path.join(self.output_dir, f"cycle{cycle_id}_task{task_run_id}")
                                os.makedirs(task_boundary_dir, exist_ok=True)
                                policy.save(task_boundary_dir, cycle_id, task_run_id, task_timesteps)

                            last_timestep_saved = task_timesteps

                    if (
                        continual_freq is not None
                        and not task._task_spec.eval_mode
                        and total_train_timesteps + task_timesteps > last_continual_testing_step + continual_freq
                    ):
                        self._run_continual_eval(
                            task_run_id,
                            policy,
                            summary_writer,
                            total_train_timesteps + task_timesteps,
                        )
                        last_continual_testing_step = total_train_timesteps + task_timesteps

                self._logger.info(f"Task {task_run_id} complete")
                self._console(f"[TASK] end   | cycle={cycle_id} task={task_run_id} steps={task_timesteps}")

                if not task._task_spec.eval_mode:
                    self._run_continual_eval(
                        task_run_id,
                        policy,
                        summary_writer,
                        total_train_timesteps + task_timesteps,
                        set_ref_task_run_id=task_run_id,
                    )
                    policy.on_task_end(cycle_id=cycle_id, task_run_id=task_run_id)

                if not task._task_spec.eval_mode:
                    total_train_timesteps += task_timesteps

                start_task_timesteps = 0

            start_task_id = 0

        if self._effective_rank_history:
            self.save_effective_rank_history()

    def try_run(self, policy, summary_writer):
        try:
            self._run(policy, summary_writer)
        except Exception as e:
            self._logger.exception(f"Failed with exception: {e}")
            policy.shutdown()
            raise e
