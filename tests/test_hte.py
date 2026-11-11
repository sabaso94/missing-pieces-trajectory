import numpy as np

from trial_trajectories.estimators import folds_for
from trial_trajectories.hte import dr_learner, group_effect_difference, group_effects, t_learner


def _rct(n=4000, seed=0):
    """Randomised with P(T = 1) = 0.75; the effect is 1 + 2 * x0."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 5))
    t = (rng.random(n) < 0.75).astype(int)
    tau = 1 + 2 * x[:, 0]
    y = x[:, 1] + tau * t + rng.normal(size=n)
    return y, t, x, tau


def test_learners_recover_a_linear_effect_modifier():
    y, t, x, tau = _rct()
    folds = folds_for(len(y), 5, seed=0)
    for pred in [t_learner(y, t, x, "linear", folds), t_learner(y, t, x, "boosting", folds),
                 dr_learner(y, t, x, folds, p_treat=0.75)]:
        assert np.corrcoef(pred, tau)[0, 1] > 0.7


def test_group_effects_are_the_randomised_contrasts():
    y = np.array([5.0, 3.0, 1.0, 2.0, 10.0, 4.0])
    t = np.array([1, 0, 0, 1, 1, 0])
    groups = np.array(["a", "a", "a", "b", "b", "b"])
    out = group_effects(y, t, groups, n_boot=50, seed=0).set_index("group")
    assert out.loc["a", "effect"] == 5.0 - 2.0 and out.loc["b", "effect"] == 6.0 - 4.0
    assert out.loc["a", "n"] == 3
    est, low, high = group_effect_difference(y, t, groups, "a", "b", n_boot=50, seed=0)
    assert est == 2.0 - 3.0
