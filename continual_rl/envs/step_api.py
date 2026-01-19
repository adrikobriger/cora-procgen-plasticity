"""Gym vs Gymnasium step API compatibility.

Provides a unified terminal signal and annotates info with _terminated/_truncated.
"""

from typing import Any, Tuple

import numpy as np


def _bool_array(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x.astype(bool)
    return np.asarray(x, dtype=bool)


def _detect_truncation_from_info(info: Any, done=None):
    """
    Detect time-limit truncation from info dict(s) for 4-tuple Gym API.
    Returns array/bool matching done shape when possible.
    """
    if isinstance(info, list):
        trunc = []
        for i, inf in enumerate(info):
            val = False
            if isinstance(inf, dict):
                val = bool(inf.get("TimeLimit.truncated", False) or inf.get("time_limit", False))
            trunc.append(val)
        return _bool_array(trunc)

    if isinstance(info, dict):
        return bool(info.get("TimeLimit.truncated", False) or info.get("time_limit", False))

    if done is None:
        return False

    return _bool_array(False)


def _annotate_info(info: Any, terminated, truncated) -> Any:
    if isinstance(info, list):
        for i, inf in enumerate(info):
            if isinstance(inf, dict):
                try:
                    inf["_terminated"] = bool(terminated[i])
                    inf["_truncated"] = bool(truncated[i])
                except Exception:
                    pass
        return info

    if isinstance(info, dict):
        try:
            info["_terminated"] = bool(terminated)
            info["_truncated"] = bool(truncated)
        except Exception:
            pass
    return info


def unpack_step(step_result) -> Tuple[Any, Any, Any, Any, Any, Any]:
    """
    Normalize env.step outputs to:
      obs, reward, terminal, info, terminated, truncated

    - For Gym 4-tuple: (obs, reward, done, info)
    - For Gymnasium 5-tuple: (obs, reward, terminated, truncated, info)
    """
    if len(step_result) == 5:
        obs, reward, terminated, truncated, info = step_result
        terminated = _bool_array(terminated)
        truncated = _bool_array(truncated)
        terminal = _bool_array(terminated | truncated)
        info = _annotate_info(info, terminated, truncated)
        return obs, reward, terminal, info, terminated, truncated

    if len(step_result) == 4:
        obs, reward, done, info = step_result
        terminal = _bool_array(done)
        truncated = _detect_truncation_from_info(info, done)
        terminated = _bool_array(done)
        try:
            # If time-limit truncation detected, mark terminated false
            if np.any(truncated):
                terminated = _bool_array(terminated & ~truncated)
        except Exception:
            pass
        info = _annotate_info(info, terminated, truncated)
        return obs, reward, terminal, info, terminated, truncated

    raise ValueError(f"Unexpected step() return length: {len(step_result)}")
