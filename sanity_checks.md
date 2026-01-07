# Sanity Checks & Setup Guide (Procgen + PPO)

This document explains:
- how to set up the environment using `uv`
- how to run the tiny Procgen sanity checks
- which files were modified and why

## Environment Setup (using `uv`)

### 1. Create and activate a virtual environment
```bash
uv venv
source .venv/bin/activate
```

### 2. Install the repo **without pulling in Atari dependencies**

**Important:**  
`pip install -e .` pulls in Atari-related packages that are not needed and cause issues.  
We intentionally install the repo **without dependencies** and install only what we need manually.

```bash
uv pip install -e . --no-deps
```

### 3. Install required runtime dependencies

```bash
uv pip install torch torchvision
uv pip install gym==0.25.2 tensorboard cloudpickle opencv-python
uv pip install procgen gym3
```

**Notes:**
- `gym==0.25.2` is required for compatibility with this codebase
- `procgen` depends on `gym3`
- CUDA is **not required** for sanity checks

### 4. Quick import checks

```bash
python -c "import continual_rl; print('continual_rl ok')"
python -c "from procgen import ProcgenEnv; env=ProcgenEnv(num_envs=1, env_name='fruitbot'); env.close(); print('procgen ok')"
```

## Sanity Check Experiment

A **tiny Procgen experiment** was added for fast verification:

- runs in ~1–2 minutes on CPU
- single environment
- very small number of timesteps
- intended only to check that PPO + Procgen works end-to-end

### Run the tiny PPO sanity check

```bash
OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
python main.py \
  --policy ppo \
  --experiment procgen_fruitbot_tiny \
  --output-dir tmp \
  --num_processes 1 \
  --num_steps 64 \
  --cuda False
```

**Expected behavior:**
- No crashes
- Logs printed to terminal
- Output directory created under:

```
tmp/ppo_procgen_fruitbot_tiny_<timestamp>/
```

## Files Modified & Why

### 1. `continual_rl/experiment_specs.py`

**Change:** Added `procgen_fruitbot_tiny`

**Why:**
- Original Procgen experiments run for millions of steps
- Tiny experiment allows quick sanity checking on CPU

### 2. `continual_rl/experiments/environment_runners/environment_runner_batch.py`

**Changes:**
- Replaced deprecated `np.float` → `np.float64`
- Ensured actions are passed to Procgen as shape `(N,)` instead of `(N,1)`

**Why:**
- NumPy removed `np.float`
- `gym3/procgen` is strict about action shapes

### 3. `continual_rl/experiments/tasks/image_task.py`

**Changes:**
- Removed assumption that observations have `.to_tensor()`
- Added support for:
  - NumPy arrays
  - Torch tensors
- Ensured observations are converted to CHW format

**Why:**
- Procgen returns raw arrays/tensors, not Atari-style wrappers

### 4. `continual_rl/policies/ppo/a2c_ppo_acktr_gail/model.py`

**Change:**
- Fixed CNN to support Procgen’s 64×64 inputs

**Why:**
- Original CNN assumed Atari’s 84×84 resolution
- Caused linear-layer shape mismatch