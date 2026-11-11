import numpy as np

from trial_trajectories.metrics import bootstrap_ci


def test_bootstrap_ci_of_a_mean_matches_normal_theory():
    y = np.random.default_rng(0).normal(10, 2, size=500)
    low, high = bootstrap_ci(lambda idx: y[idx].mean(), len(y), n_boot=2000, seed=0)
    half = 1.96 * y.std(ddof=1) / np.sqrt(len(y))
    assert abs((high - low) / 2 - half) < 0.1 * half and low < y.mean() < high


def test_bootstrap_ci_of_several_statistics_at_once():
    y = np.random.default_rng(1).normal(size=300)
    low, high = bootstrap_ci(lambda idx: np.array([y[idx].mean(), y[idx].std()]), len(y), n_boot=500)
    assert low.shape == (2,) and (low < high).all()
