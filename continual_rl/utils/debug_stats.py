"""Diagnostics for zero-reward task issue."""

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch optional for stats
    torch = None


@dataclass
class RollingStats:
    count: int = 0
    nonzero_count: int = 0
    sum: float = 0.0
    min: float = field(default_factory=lambda: float("inf"))
    max: float = field(default_factory=lambda: float("-inf"))
    nan_count: int = 0
    inf_count: int = 0

    def reset(self) -> None:
        self.count = 0
        self.nonzero_count = 0
        self.sum = 0.0
        self.min = float("inf")
        self.max = float("-inf")
        self.nan_count = 0
        self.inf_count = 0

    def _to_numpy(self, x) -> np.ndarray:
        if x is None:
            return np.asarray([])
        if torch is not None and isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def update(self, x) -> None:
        arr = self._to_numpy(x).astype(np.float64, copy=False)
        if arr.size == 0:
            return
        arr = arr.reshape(-1)

        nan_mask = np.isnan(arr)
        inf_mask = np.isinf(arr)
        self.nan_count += int(nan_mask.sum())
        self.inf_count += int(inf_mask.sum())

        finite = arr[~(nan_mask | inf_mask)]
        if finite.size == 0:
            return

        self.count += int(finite.size)
        self.nonzero_count += int((finite != 0).sum())
        self.sum += float(finite.sum())
        self.min = float(min(self.min, float(finite.min())))
        self.max = float(max(self.max, float(finite.max())))

    @property
    def mean(self) -> float:
        return float(self.sum / self.count) if self.count > 0 else 0.0

    @property
    def nonzero_frac(self) -> float:
        return float(self.nonzero_count / self.count) if self.count > 0 else 0.0

    def summary(self) -> dict:
        return {
            "count": int(self.count),
            "nonzero_count": int(self.nonzero_count),
            "mean": float(self.mean),
            "min": float(self.min if self.count > 0 else 0.0),
            "max": float(self.max if self.count > 0 else 0.0),
            "nan": int(self.nan_count),
            "inf": int(self.inf_count),
        }