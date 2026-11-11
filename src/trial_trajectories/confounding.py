"""Turning the randomised trial into confounded "observational" data with a known answer.

Patients are subsampled so that in the sample sicker patients are more likely to be treated:
P(T = 1 | x) = expit(gamma * severity(x)), while the covariates keep the trial's distribution. The average
effect in the sample is then still the trial's, and any estimator can be scored against it
(the idea of Gentzel et al., 2021; here the sampling also preserves the covariate distribution).
"""
import numpy as np
import pandas as pd
from scipy.special import expit


def severity_score(df: pd.DataFrame) -> np.ndarray:
    """Mean of standardised low CD4, low Karnofsky score and symptoms; standardised again (mean 0, SD 1)."""
    parts = [-df["cd40"], -df["karnof"], df["symptom"]]
    score = np.mean([(p - p.mean()) / p.std() for p in parts], axis=0)
    return (score - score.mean()) / score.std()


def confounded_sample(t: np.ndarray, score: np.ndarray, gamma: float, rng: np.random.Generator) -> np.ndarray:
    """Indices of a subsample in which P(T = 1 | score) = expit(gamma * score) and the score keeps its
    distribution.

    With p the trial's treated share and e the target propensity, a treated patient is kept with probability
    k * e / p and a control with k * (1 - e) / (1 - p): the chance of being kept is then k for everyone, so the
    covariates are not reweighted, and among kept patients the treated share at each score is e. k is the largest
    value keeping both probabilities at most 1.
    """
    p = t.mean()
    e = expit(gamma * score)
    k = min((p / e).min(), ((1 - p) / (1 - e)).min())
    keep = np.where(t == 1, k * e / p, k * (1 - e) / (1 - p))
    return np.flatnonzero(rng.random(len(t)) < keep)
