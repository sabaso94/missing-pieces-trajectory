"""Who benefits: effects that differ between patients, in a randomised trial.

- t_learner: separate outcome models per arm, the effect is their difference;
- dr_learner: the doubly robust pseudo-outcome (Kennedy, 2023) with the trial's known treatment probability,
  regressed on the covariates with a ridge regression.
Every prediction is out of fold (cross-fitting), so a patient's predicted effect never uses their own outcome.
Predicted effects are checked against randomised contrasts within groups of patients (group_effects).
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
    mu1 = cross_fit_predict(lambda: outcome_model(nuisance, seed), x, y, folds, rows=t == 1)
    mu0 = cross_fit_predict(lambda: outcome_model(nuisance, seed), x, y, folds, rows=t == 0)
    pseudo = mu1 - mu0 + t * (y - mu1) / p_treat - (1 - t) * (y - mu0) / (1 - p_treat)
    final = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))
    return cross_fit_predict(final, x, pseudo, folds)


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
