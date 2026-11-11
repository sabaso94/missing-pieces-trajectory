"""Percentile bootstrap that resamples patients (each patient is one row)."""
from typing import Callable

import numpy as np


def bootstrap_ci(stat_fn: Callable[[np.ndarray], float | np.ndarray], n: int, n_boot: int = 1000, seed: int = 0,
                 level: float = 0.95):
    """Percentile interval of stat_fn(indices) over n_boot resamples of the n rows (with replacement).
    stat_fn may return one number or an array of statistics; the bounds then have the same shape."""
    rng = np.random.default_rng(seed)
    stats = np.array([stat_fn(rng.integers(0, n, size=n)) for _ in range(n_boot)])
    tail = (1 - level) / 2 * 100
    return np.nanpercentile(stats, tail, axis=0), np.nanpercentile(stats, 100 - tail, axis=0)
