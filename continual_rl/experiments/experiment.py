import os
import json
import numbers
import numpy as np
import torch
from continual_rl.experiments.run_metadata import RunMetadata
from continual_rl.utils.utils import Utils
from continual_rl.utils.common_exceptions import OutputDirectoryNotSetException


class InvalidTaskAttributeException(Exception):
    def __init__(self, error_msg):
        super().__init__(error_msg)


class Experiment(object):
    def __init__(self, tasks, continual_testing_freq=None, cycle_count=1):
        """
        The Experiment class contains everything that should be held consistent when the experiment is used as a
        setting for a baseline.

        A single experiment can cover tasks with a variety of action spaces. It is up to the policy on how they wish
        to handle this, but what the Experiment does is create a dictionary mapping action_space_id to action space, and
        ensures that all tasks claiming the same id use the same action space.

        The observation space and time batch sizes are both restricted to being the same for all tasks. This
        initialization will assert if this is violated.

        :param tasks: A list of subclasses of TaskBase. These need to have a consistent observation space.
        :param output_dir: The directory in which logs will be stored.
        :param continual_testing_freq: The number of timesteps between evaluation steps on the not-currently-training
        tasks.
        :param cycle count: The number of times to cycle through the list of tasks.
        """
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

        # ADDED: tracking continual-eval returns for forgetting metrics
        self._eval_last_return = {}
        self._eval_last_return_iqm = {}
        self._ref_return_end_of_task = {}

        # ADDED: tracking effective rank metrics
        self._effective_rank_history = []  # List of (timestep, layer_name, effective_rank) tuples
        self._effective_rank_by_layer = {}  # Most recent effective rank per layer

    def set_output_dir(self, output_dir):
        self._output_dir = output_dir

    @property
    def output_dir(self):
        if self._output_dir is None:
            raise OutputDirectoryNotSetException("Output directory not set, but is attempting to be used. Call set_output_dir.")
        return self._output_dir

    @property
    def _logger(self):
        return Utils.create_logger(f"{self.output_dir}/core_process.log")
    
    # ADDED: Trying to clean up loggers and terminal output
    def _console(self, msg: str) -> None:
        # clean human-readable terminal output (no timestamps, no logger prefixes)
        print(msg, flush=True)

    # ADDED: Effective rank computation for plasticity metrics
    @staticmethod
    def _compute_effective_rank_from_matrix(matrix: np.ndarray, epsilon: float = 1e-10) -> float:
        """
        Compute the effective rank of a matrix using the entropy-based formula.
        
        Effective rank = exp(H) where H is the entropy of normalized singular values.
        H = -sum(p_i * log(p_i)) where p_i = sigma_i / sum(sigma_j)
        
        This measures the "effective dimensionality" of the matrix.
        A higher effective rank indicates more distributed singular values (more "plastic").
        A lower effective rank indicates more concentrated singular values (potential plasticity loss).
        
        :param matrix: 2D numpy array (weight matrix)
        :param epsilon: Small value to avoid log(0)
        :return: Effective rank (float between 1 and min(rows, cols))
        """
        if matrix.ndim != 2:
            return None
        
        # Compute singular values
        try:
            singular_values = np.linalg.svd(matrix, compute_uv=False)
        except np.linalg.LinAlgError:
            return None
        
        # Filter out near-zero singular values
        singular_values = singular_values[singular_values > epsilon]
        
        if len(singular_values) == 0:
            return 0.0
        
        # Normalize to get probability distribution
        total = np.sum(singular_values)
        if total < epsilon:
            return 0.0
        
        p = singular_values / total
        
        # Compute entropy: H = -sum(p_i * log(p_i))
        # Use natural log for standard entropy
        entropy = -np.sum(p * np.log(p + epsilon))
        
        # Effective rank = exp(H)
        effective_rank = np.exp(entropy)
        
        return float(effective_rank)

    def _compute_effective_rank(self, policy, total_timesteps: int, summary_writer) -> dict:
        """
        Compute effective rank for all applicable weight matrices in the policy's model.
        Effective rank measures the "effective dimensionality" of weight matrices and is used
        to track plasticity loss in continual learning.
        
        :param policy: The policy object containing the neural network
        :param total_timesteps: Current total timesteps (for logging)
        :param summary_writer: Tensorboard summary writer
        :return: Dictionary mapping layer names to their effective ranks
        """
        effective_ranks = {}
        
        # Try to access the model from the policy - handle different policy types
        model = None
        
        # PPO and other single-model policies
        if hasattr(policy, '_actor_critic') and policy._actor_critic is not None:
            model = policy._actor_critic
        
        # Impala and other trainer-based policies
        elif hasattr(policy, 'impala_trainer'):
            trainer = policy.impala_trainer
            # Try learner_model first (on device, actively used for training)
            if hasattr(trainer, 'learner_model') and trainer.learner_model is not None:
                model = trainer.learner_model
            elif hasattr(trainer, 'actor_model') and trainer.actor_model is not None:
                model = trainer.actor_model
        
        # Fallback: try common attribute names
        if model is None:
            model_attr_names = ['model', 'network', 'actor_critic', 'net', 'policy_net', 'q_network', 'actor', 'actor_net']
            for attr_name in model_attr_names:
                if hasattr(policy, attr_name):
                    attr = getattr(policy, attr_name)
                    if attr is not None:
                        model = attr
                        break
        
        if model is None:
            return effective_ranks
        
        # Iterate through named parameters and compute effective rank for weight matrices
        try:
            named_params = list(model.named_parameters()) if hasattr(model, 'named_parameters') else []
            
            if not named_params:
                return effective_ranks
            
            layer_ranks = []
            
            for name, param in named_params:
                # Only compute for 2D weight matrices (skip biases, embeddings, etc.)
                if param.dim() == 2 and 'weight' in name.lower():
                    weight_matrix = param.detach().cpu().numpy()
                    eff_rank = self._compute_effective_rank_from_matrix(weight_matrix)
                    
                    if eff_rank is not None:
                        effective_ranks[name] = eff_rank
                        layer_ranks.append(eff_rank)
                        self._effective_rank_history.append((total_timesteps, name, eff_rank))
                
                # Also handle Conv2d layers by reshaping to 2D
                elif param.dim() == 4 and 'weight' in name.lower():
                    # Conv weights are (out_channels, in_channels, H, W)
                    # Reshape to (out_channels, in_channels * H * W)
                    weight = param.detach().cpu().numpy()
                    out_ch = weight.shape[0]
                    reshaped = weight.reshape(out_ch, -1)
                    eff_rank = self._compute_effective_rank_from_matrix(reshaped)
                    
                    if eff_rank is not None:
                        effective_ranks[name] = eff_rank
                        layer_ranks.append(eff_rank)
                        self._effective_rank_history.append((total_timesteps, name, eff_rank))
            
            # Log only aggregate statistics
            if layer_ranks:
                avg_rank = float(np.mean(layer_ranks))
                min_rank = float(np.min(layer_ranks))
                max_rank = float(np.max(layer_ranks))
                
                summary_writer.add_scalar("effective_rank/avg", avg_rank, global_step=total_timesteps)
                summary_writer.add_scalar("effective_rank/min", min_rank, global_step=total_timesteps)
                summary_writer.add_scalar("effective_rank/max", max_rank, global_step=total_timesteps)
                summary_writer.flush()
            
        except Exception as e:
            self._logger.warning(f"Error computing effective rank: {e}")
        
        # Update most recent values
        self._effective_rank_by_layer = effective_ranks
        
        return effective_ranks

    def get_effective_rank_history(self) -> list:
        """
        Get the full history of effective rank measurements.
        
        :return: List of (timestep, layer_name, effective_rank) tuples
        """
        return self._effective_rank_history.copy()

    def get_current_effective_ranks(self) -> dict:
        """
        Get the most recent effective rank values per layer.
        
        :return: Dictionary mapping layer names to effective rank values
        """
        return self._effective_rank_by_layer.copy()

    def save_effective_rank_history(self, filepath: str = None) -> str:
        """
        Save effective rank history to a JSON file.
        
        :param filepath: Path to save the file. If None, saves to output_dir/effective_rank_history.json
        :return: The filepath where data was saved
        """
        if filepath is None:
            filepath = os.path.join(self.output_dir, "effective_rank_history.json")
        
        # Convert history to a more structured format for JSON
        history_data = {
            "measurements": [
                {"timestep": t, "layer": layer, "effective_rank": rank}
                for t, layer, rank in self._effective_rank_history
            ],
            "latest_by_layer": self._effective_rank_by_layer
        }
        
        with open(filepath, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        self._logger.info(f"Saved effective rank history to {filepath}")
        return filepath

    @classmethod
    def _get_action_spaces(self, tasks):
        action_space_map = {}  # Maps task id to its action space

        for task in tasks:
            if task.action_space_id not in action_space_map:
                action_space_map[task.action_space_id] = task.action_space
            elif action_space_map[task.action_space_id] != task.action_space:
                raise InvalidTaskAttributeException(f"Action sizes were mismatched for task {task.action_space_id}")

        return action_space_map

    @classmethod
    def _get_common_attribute(self, task_attributes):
        common_attribute = None

        for task_attribute in task_attributes:
            if common_attribute is None:
                common_attribute = task_attribute

            if task_attribute != common_attribute:
                raise InvalidTaskAttributeException("Tasks do not have a common attribute.")

        return common_attribute

    def _run_continual_eval(self, task_run_id, policy, summary_writer, total_timesteps, set_ref_task_run_id=None):
        # ADDED: Compute effective rank at each continual eval point
        self._compute_effective_rank(policy, total_timesteps, summary_writer)

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

        # Run a small amount of eval on all non-eval, not-currently-running tasks
        for test_task_run_id, test_task in enumerate(self.tasks):
            # not checking test_task._task_spec.eval_mode anymore since some eval tasks
            # (for train/test pairs) should be continual eval
            if not test_task._task_spec.with_continual_eval:
                continue

            episodes_cap = None
            if hasattr(test_task, "_continual_eval_task_spec"):
                episodes_cap = getattr(test_task._continual_eval_task_spec, "return_after_episode_num", None)
            if episodes_cap is None:
                episodes_cap = float("inf")

            self._logger.info(f"Continual eval for task: {test_task_run_id}")

            # Don't increment the total_timesteps counter for continual tests
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

                    # Task generator yields: (task_timesteps, data); where data is (returns, logs) or None
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

            # store aggregate eval return (mean over collected episodes) for this task
            if returns_all:
                mean_ret = float(np.mean(returns_all))
                iqm_ret = _iqm_local(returns_all)
                self._eval_last_return[test_task_run_id] = mean_ret
                self._eval_last_return_iqm[test_task_run_id] = iqm_ret

            self._logger.info(f"Completed continual eval for task: {test_task_run_id}")
        

        # ADDED: If requested, lock in the "reference" return for a task at end-of-task boundary
        if set_ref_task_run_id is not None and set_ref_task_run_id in self._eval_last_return:
            self._ref_return_end_of_task[set_ref_task_run_id] = self._eval_last_return[set_ref_task_run_id]

        # ADDED: log isolated forgetting scalars
        # isolated forgetting for task i at time t := ref_end_of_task(i) - current_eval(i)
        forgetting_vals = []
        for tid, ref in self._ref_return_end_of_task.items():
            cur = self._eval_last_return.get(tid, None)
            if cur is None:
                continue
            f = float(ref) - float(cur)
            forgetting_vals.append(f)

            # per-task forgetting (optional but very useful)
            summary_writer.add_scalar(f"forgetting/isolated_task/{tid}", f, global_step=total_timesteps)

        if forgetting_vals:
            avg_f = float(sum(forgetting_vals) / len(forgetting_vals))
            summary_writer.add_scalar("forgetting/isolated_avg", avg_f, global_step=total_timesteps)
            summary_writer.flush()


    def _run(self, policy, summary_writer):
        # Load as necessary
        policy.load(self.output_dir)
        run_metadata = RunMetadata(self._output_dir)
        start_cycle_id = run_metadata.cycle_id
        start_task_id = run_metadata.task_id
        start_task_timesteps = run_metadata.task_timesteps

        # Only updated after a task is complete. To get the current within-task number, add task_timesteps
        total_train_timesteps = run_metadata.total_train_timesteps

        timesteps_per_save = policy.config.timesteps_per_save

        for cycle_id in range(start_cycle_id, self._cycle_count):
            for task_run_id, task in enumerate(self.tasks[start_task_id:], start=start_task_id):
                # Run the current task as a generator so we can intersperse testing tasks during the run
                self._logger.info(f"Starting cycle {cycle_id} task {task_run_id}")
                self._console(f"[TASK] start | cycle={cycle_id} task={task_run_id}")
                
                # ADDED: INTEGRATION WITH POLICY HOOKS
                # Policy hook: task is about to start (train or eval)
                if not task._task_spec.eval_mode:
                    policy.on_task_start(cycle_id=cycle_id, task_run_id=task_run_id)
                # END ADDED

                task_complete = False
                task_runner = task.run(
                    task_run_id,
                    policy,
                    summary_writer,
                    self.output_dir,
                    timestep_log_offset=total_train_timesteps,
                    task_timestep_start=start_task_timesteps,
                )
                task_timesteps = start_task_timesteps  # What timestep the task is currently on. Cumulative during a task.
                continual_freq = self._continual_testing_freq
                last_timestep_saved = None  # Ensures a save at the beginning of every new task (after one train step)

                # The last step at which continual testing was done. Initializing to be more negative
                # than the frequency we collect at, to ensure we do a collection right away
                last_continual_testing_step = -10 * continual_freq if continual_freq is not None else None
                last_printed_t = None  # For logging training progress

                while not task_complete:
                    try:
                        task_timesteps, info = next(task_runner)
                        # ADDED: For better logging of training progress
                        if (not task._task_spec.eval_mode) and (task_timesteps % 1024 == 0):
                            # info is usually: ([reward], list_of_metric_dicts)
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
                            ent   = stats.get("dist_entropy")

                            vloss_str = f"{vloss:.4f}" if isinstance(vloss, (int, float)) else "NA"
                            aloss_str = f"{aloss:.4f}" if isinstance(aloss, (int, float)) else "NA"
                            ent_str   = f"{ent:.3f}"   if isinstance(ent, (int, float)) else "NA"

                            self._console(
                                f"[TRAIN] cycle={cycle_id} task={task_run_id} "
                                f"t={total_train_timesteps + task_timesteps} "
                                f"r={r_str} vloss={vloss_str} aloss={aloss_str} ent={ent_str}"
                            )
                            # END ADDED

                    except StopIteration:
                        task_complete = True

                    if not task._task_spec.eval_mode:
                        if last_timestep_saved is None or task_timesteps - last_timestep_saved >= timesteps_per_save or \
                                task_complete:
                            # Save the metadata that allows us to resume where we left off.
                            # This will not copy files in large_file_path such as 
                            # replay buffers, and is intended for debugging model changes
                            # at task boundaries.
                            run_metadata.save(cycle_id, task_run_id, task_timesteps, total_train_timesteps)
                            policy.save(self.output_dir, cycle_id, task_run_id, task_timesteps)
                            if task_complete:
                                task_boundary_dir = os.path.join(self.output_dir, f'cycle{cycle_id}_task{task_run_id}')
                                os.makedirs(task_boundary_dir, exist_ok=True)

                                policy.save(task_boundary_dir, cycle_id, task_run_id, task_timesteps)

                            last_timestep_saved = task_timesteps

                    # If we're already doing eval, don't do a forced eval run (nothing has trained to warrant it anyway)
                    # Evaluate intermittently. Every time is too slow
                    if continual_freq is not None and not task._task_spec.eval_mode and \
                            total_train_timesteps + task_timesteps > last_continual_testing_step + continual_freq:
                        self._run_continual_eval(
                            task_run_id,
                            policy,
                            summary_writer,
                            total_train_timesteps + task_timesteps,
                        )
                        last_continual_testing_step = total_train_timesteps + task_timesteps

                # Log out some info about the just-completed task
                self._logger.info(f"Task {task_run_id} complete")
                self._console(f"[TASK] end   | cycle={cycle_id} task={task_run_id} steps={task_timesteps}")

                # ADDED: INTEGRATION WITH POLICY HOOKS
                # Policy hook: task has finished (train or eval)
                if not task._task_spec.eval_mode:
                    self._run_continual_eval(
                        task_run_id,
                        policy,
                        summary_writer,
                        total_train_timesteps + task_timesteps,
                        set_ref_task_run_id=task_run_id,
                    )
                    policy.on_task_end(cycle_id=cycle_id, task_run_id=task_run_id)
                # END ADDED

                # Only increment the global counter for training (it's supposed to represent number of frames *trained on*)
                if not task._task_spec.eval_mode:
                    total_train_timesteps += task_timesteps

                # On the next task, start from the beginning (regardless of where we loaded from)
                start_task_timesteps = 0

            # On the next cycle, start from the beginning again (regardless of where we loaded from)
            start_task_id = 0

        # ADDED: Save effective rank history at end of experiment
        if self._effective_rank_history:
            self.save_effective_rank_history()

    def try_run(self, policy, summary_writer):
        try:
            self._run(policy, summary_writer)
        except Exception as e:
            self._logger.exception(f"Failed with exception: {e}")
            policy.shutdown()

            raise e
