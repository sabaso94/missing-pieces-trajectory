import numpy as np
from scipy.special import expit

from trial_trajectories.estimators import aipw, ancova, cross_fit_predict, diff_in_means, folds_for, ipw, propensity_model

ATE = 2.0


def _confounded(n=4000, seed=0):
    """x0 drives both treatment and outcome; the true average effect is ATE."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    t = (rng.random(n) < expit(1.0 * x[:, 0])).astype(int)
    y = ATE * t + 2 * x[:, 0] + x[:, 1] + rng.normal(size=n)
    return y, t, x


def test_unadjusted_is_biased_and_adjusted_estimators_are_not():
    y, t, x = _confounded()
    assert abs(diff_in_means(y, t).est - ATE) > 0.5
    for estimate in [ancova(y, t, x), ipw(y, t, x), aipw(y, t, x, "linear"), aipw(y, t, x, "boosting")]:
        assert abs(estimate.est - ATE) < 0.25, estimate


def test_aipw_is_doubly_robust():
    # outcome model wrong (linear for a quadratic outcome), propensity model right; averaged over 5 samples
    ancova_bias, aipw_bias = [], []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(20_000, 3))
        t = (rng.random(20_000) < expit(x[:, 0] + 0.5)).astype(int)
        y = ATE * t + 3 * x[:, 0] ** 2 + x[:, 1] + rng.normal(size=20_000)
        ancova_bias.append(ancova(y, t, x).est - ATE)
        aipw_bias.append(aipw(y, t, x, "linear").est - ATE)
    assert abs(np.mean(ancova_bias)) > 0.3
    assert abs(np.mean(aipw_bias)) < 0.15


class _Memoriser:
    """Predicts a training row's own y, and the training mean for rows it has not seen."""

    def fit(self, x, y):
        self.seen = {tuple(row): v for row, v in zip(x, y)}
        self.mean = float(np.mean(y))
        return self

    def predict(self, x):
        return np.array([self.seen.get(tuple(row), self.mean) for row in x])


def test_cross_fit_predictions_are_out_of_fold():
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(100, 2)), rng.normal(size=100)
    folds = folds_for(100, 5, seed=0)
    pred = cross_fit_predict(_Memoriser, x, y, folds)
    for k in range(5):
        assert np.allclose(pred[folds == k], y[folds != k].mean())


def test_ipw_and_aipw_survive_near_perfect_separation():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(500, 2))
    t = (x[:, 0] > 0).astype(int)
    t[:3] = 1 - t[:3]
    y = t + x[:, 0] + rng.normal(size=500)
    for estimate in [ipw(y, t, x), aipw(y, t, x, "linear"), aipw(y, t, x, "boosting")]:
        assert np.isfinite(estimate.est) and np.isfinite(estimate.se) and estimate.se > 0


def test_intervals_cover_near_nominal():
    covered = 0
    for seed in range(200):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(500, 3))
        t = (rng.random(500) < 0.5).astype(int)
        y = ATE * t + x @ np.array([1.0, -1.0, 0.5]) + rng.normal(size=500)
        low, high = ancova(y, t, x).ci()
        covered += low <= ATE <= high
    assert 0.90 <= covered / 200 <= 0.99


def test_boosted_propensity_does_not_overfit_a_randomised_trial():
    # randomised with P(T = 1) = 0.75 and 15 irrelevant covariates: estimated propensities should stay near 0.75
    rng = np.random.default_rng(3)
    x = rng.normal(size=(2000, 15))
    t = (rng.random(2000) < 0.75).astype(int)
    e = cross_fit_predict(lambda: propensity_model("boosting"), x, t, folds_for(2000, 5, seed=0))
    assert np.percentile(e, 1) > 0.55 and np.percentile(e, 99) < 0.92


class _Degenerate:
    """A propensity model that is certain: 0 for half the rows, 1 for the rest."""

    def fit(self, x, t):
        return self

    def predict_proba(self, x):
        p = (x[:, 0] > 0).astype(float)
        return np.column_stack([1 - p, p])


def test_estimated_propensities_are_clipped(monkeypatch):
    import trial_trajectories.estimators as est
    monkeypatch.setattr(est, "propensity_model", lambda nuisance, seed=0: _Degenerate())
    rng = np.random.default_rng(4)
    x = rng.normal(size=(400, 2))
    t = (rng.random(400) < 0.5).astype(int)
    y = t + rng.normal(size=400)
    for estimate in [est.ipw(y, t, x), est.aipw(y, t, x, "linear")]:
        assert np.isfinite(estimate.est) and np.isfinite(estimate.se)
    assert est.clipped_propensity(np.array([0.0, 0.5, 1.0]), 0.01).tolist() == [0.01, 0.5, 0.99]
