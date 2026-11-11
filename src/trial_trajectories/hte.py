"""Who benefits: effects that differ between patients, in a randomised trial.

- t_learner: separate outcome models per arm, the effect is their difference;
- dr_learner: the doubly robust pseudo-outcome (Kennedy, 2023) with the trial's known treatment probability,
  regressed on the covariates with a ridge regression; the outcome models and the final regression are fitted on
  disjoint folds (sample splitting).
Every prediction is out of fold, and no model behind a patient's prediction has seen that patient's outcome.
Predicted effects are checked against randomised contrasts within groups of patients (group_effects). Each fold's
predictions come from different models, so their level can differ between folds; quartiles are therefore formed
within each fold (quartiles_within_folds), so that a fold's level does not decide its patients' quartile.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from trial_trajectories.estimators import cross_fit_predict, outcome_model
from trial_trajectories.missing import effect_in_group


def t_learner(y: np.ndarray, t: np.ndarray, x: np.ndarray, nuisance: str, folds: np.ndarray, seed: int = 0) -> np.ndarray:
    mu1 = cross_fit_predict(lambda: outcome_model(nuisance, seed), x, y, folds, rows=t == 1)
    mu0 = cross_fit_predict(lambda: outcome_model(nuisance, seed), x, y, folds, rows=t == 0)
    return mu1 - mu0


def dr_learner(y: np.ndarray, t: np.ndarray, x: np.ndarray, folds: np.ndarray, p_treat: float,
               nuisance: str = "boosting", seed: int = 0) -> np.ndarray:
    """For each fold k, the other folds are split in two halves: outcome models on one half, pseudo-outcomes and
    the final ridge regression on the other; then the halves swap, and fold k gets the mean of both predictions."""
    pred = np.zeros(len(y))
    for k in np.unique(folds):
        others = [f for f in np.unique(folds) if f != k]
        halves = [np.isin(folds, others[0::2]), np.isin(folds, others[1::2])]
        for nuis, fin in [(halves[0], halves[1]), (halves[1], halves[0])]:
            mu1 = outcome_model(nuisance, seed).fit(x[nuis & (t == 1)], y[nuis & (t == 1)]).predict(x[fin])
            mu0 = outcome_model(nuisance, seed).fit(x[nuis & (t == 0)], y[nuis & (t == 0)]).predict(x[fin])
            tf, yf = t[fin], y[fin]
            pseudo = mu1 - mu0 + tf * (yf - mu1) / p_treat - (1 - tf) * (yf - mu0) / (1 - p_treat)
            final = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25))).fit(x[fin], pseudo)
            pred[folds == k] += final.predict(x[folds == k]) / 2
    return pred


def quartiles_within_folds(pred: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Quartile (1-4) of each prediction among the predictions of its own fold."""
    q = np.empty(len(pred), dtype=int)
    for k in np.unique(folds):
        rows = folds == k
        q[rows] = pd.qcut(pd.Series(pred[rows]).rank(method="first"), 4, labels=False).to_numpy() + 1
    return q


def _effects(y, t, groups, labels):
    return np.array([effect_in_group(y, t, groups == g) for g in labels])


def group_effects(y: np.ndarray, t: np.ndarray, groups: np.ndarray, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Randomised difference in means within each group, with percentile intervals over patient resamples."""
    labels = pd.unique(groups)
    point = _effects(y, t, groups, labels)
    rng = np.random.default_rng(seed)
    boot = np.array([_effects(y[i], t[i], groups[i], labels) for i in rng.integers(0, len(y), size=(n_boot, len(y)))])
    return pd.DataFrame({"group": labels, "n": [(groups == g).sum() for g in labels],
                         "n_treated": [((groups == g) & (t == 1)).sum() for g in labels], "effect": point,
                         "low": np.nanpercentile(boot, 2.5, axis=0), "high": np.nanpercentile(boot, 97.5, axis=0)})


def group_effect_difference(y: np.ndarray, t: np.ndarray, groups: np.ndarray, a, b, n_boot: int = 1000,
                            seed: int = 0) -> tuple[float, float, float]:
    """Effect in group b minus effect in group a, with a percentile interval."""
    diff = lambda yy, tt, gg: effect_in_group(yy, tt, gg == b) - effect_in_group(yy, tt, gg == a)
    rng = np.random.default_rng(seed)
    boot = [diff(y[i], t[i], groups[i]) for i in rng.integers(0, len(y), size=(n_boot, len(y)))]
    return float(diff(y, t, groups)), float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))
