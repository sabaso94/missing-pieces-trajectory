"""Percentile bootstrap that resamples patients (each patient is one row); coverage of a nested truth."""
from typing import Callable

import numpy as np
from scipy.stats import norm


def bootstrap_ci(stat_fn: Callable[[np.ndarray], float | np.ndarray], n: int, n_boot: int = 1000, seed: int = 0,
                 level: float = 0.95):
    """Percentile interval of stat_fn(indices) over n_boot resamples of the n rows (with replacement).
    stat_fn may return one number or an array of statistics; the bounds then have the same shape."""
    rng = np.random.default_rng(seed)
    stats = np.array([stat_fn(rng.integers(0, n, size=n)) for _ in range(n_boot)])
    tail = (1 - level) / 2 * 100
    return np.nanpercentile(stats, tail, axis=0), np.nanpercentile(stats, 100 - tail, axis=0)


def coverage_of_truth(est: np.ndarray, se: np.ndarray, truth: float, se_truth: float,
                      level: float = 0.95) -> np.ndarray:
    """Whether each interval covers a truth estimated on data that contain the sample.

    The sample estimate and the truth share patients, so their difference varies less than the sample's standard
    error says. When the truth is the more efficient estimate, Var(est - truth) = se^2 - se_truth^2 (the Hausman
    form); intervals use that variance. An approximation: it assumes the truth's estimator is efficient."""
    half = norm.ppf(0.5 + level / 2) * np.sqrt(np.maximum(np.asarray(se) ** 2 - se_truth ** 2, 0.0))
    return (np.asarray(est) - half <= truth) & (truth <= np.asarray(est) + half)
