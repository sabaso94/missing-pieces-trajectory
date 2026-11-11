import numpy as np
import pandas as pd
from scipy.special import expit

from trial_trajectories.confounding import confounded_sample, severity_score


def _trial(n=20_000, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random(n) < 0.75).astype(int), rng.normal(size=n)


def test_confounded_sample_keeps_covariates_and_sets_propensity():
    t, score = _trial()
    idx = confounded_sample(t, score, gamma=1.5, rng=np.random.default_rng(1))
    assert abs(score[idx].mean() - score.mean()) < 0.05
    deciles = pd.qcut(score[idx], 10, labels=False)
    for d in range(10):
        in_d = idx[deciles == d]
        assert abs(t[in_d].mean() - expit(1.5 * score[in_d]).mean()) < 0.05, d


def test_gamma_zero_is_a_random_subsample():
    t, score = _trial()
    idx = confounded_sample(t, score, gamma=0.0, rng=np.random.default_rng(2))
    assert abs(t[idx].mean() - 0.5) < 0.03
    assert abs(np.corrcoef(t[idx], score[idx])[0, 1]) < 0.05
    assert len(np.unique(idx)) == len(idx)


def test_severity_is_higher_for_sicker_patients():
    df = pd.DataFrame({"cd40": [500, 200, 350, 300], "karnof": [100, 70, 90, 90], "symptom": [0, 1, 0, 1]})
    score = severity_score(df)
    assert score.argmax() == 1 and score.argmin() == 0
    assert abs(score.mean()) < 1e-9 and abs(score.std() - 1) < 1e-9
