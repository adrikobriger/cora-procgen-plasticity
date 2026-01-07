import torch
import torchvision
from gym.spaces.box import Box
from continual_rl.experiments.tasks.task_base import TaskBase
from continual_rl.experiments.tasks.preprocessor_base import PreprocessorBase
from continual_rl.utils.utils import Utils
from continual_rl.utils.env_wrappers import FrameStack, WarpFrame, ImageToPyTorch


class ImagePreprocessor(PreprocessorBase):
    def __init__(self, time_batch_size, image_size, grayscale, env_spec, resize_interp_method):
        self._resize_interp_method = resize_interp_method
        self.env_spec = self._wrap_env(env_spec, time_batch_size, image_size, grayscale)
        dummy_env, _ = Utils.make_env(self.env_spec)
        super().__init__(dummy_env.observation_space)

    def _wrap_env(self, env_spec, time_batch_size, image_size, grayscale):
        # Leverage the existing env wrappers for simplicity
        frame_stacked_env_spec = lambda: FrameStack(ImageToPyTorch(
            WarpFrame(Utils.make_env(env_spec)[0], image_size[0], image_size[1], grayscale=grayscale,
                      resize_interp_method=self._resize_interp_method)), time_batch_size)
        return frame_stacked_env_spec

    def preprocess(self, batched_env_image):
        """
        The preprocessed image will have values in range [0, 255] and shape [batch, time, channels, width, height].
        Handled as a batch for speed.
        """
        # OLD (breaks because `image` is already a torch.Tensor)
        # processed_image = torch.stack([image.to_tensor() for image in batched_env_image])

        # NEW (works for both old "image objects" + new torch/np arrays)
        tensors = []
        for image in batched_env_image:
            if hasattr(image, "to_tensor"):
                # --- FIX: keep backward compatibility (old image wrapper types) ---
                t = image.to_tensor()
            else:
                # --- FIX: handle torch.Tensor / numpy.ndarray directly ---
                import numpy as np

                if isinstance(image, torch.Tensor):
                    t = image
                else:
                    t = torch.as_tensor(image)

                # if it's HWC (common for procgen), convert to CHW (what the CNN expects)
                if t.ndim == 3 and t.shape[-1] in (1, 3, 4) and t.shape[0] not in (1, 3, 4):
                    t = t.permute(2, 0, 1)

            tensors.append(t)

        processed_image = torch.stack(tensors)
        return processed_image

    def render_episode(self, episode_observations):
        """
        Turn a list of observations gathered from the episode into a video that can be saved off to view behavior.
        """
        return torch.stack(episode_observations).unsqueeze(0)


class ImageTask(TaskBase):
    def __init__(self, task_id, action_space_id, env_spec, num_timesteps, time_batch_size, eval_mode,
                 image_size, grayscale, continual_eval=True, resize_interp_method="INTER_AREA"):
        preprocessor = ImagePreprocessor(time_batch_size, image_size, grayscale, env_spec, resize_interp_method)
        dummy_env, _ = Utils.make_env(preprocessor.env_spec)

        super().__init__(task_id, action_space_id, preprocessor, preprocessor.env_spec, preprocessor.observation_space,
                         dummy_env.action_space, num_timesteps, eval_mode, continual_eval=continual_eval)
