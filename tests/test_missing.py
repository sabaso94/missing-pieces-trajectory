import numpy as np
import pandas as pd
from scipy.special import expit

from trial_trajectories.missing import HISTORY96, arm_mean, effect_in_group, week96_history

METHODS = ["complete", "ipcw", "aipw"]


def _arm(n=20_000, seed=0, mechanism="mar"):
    """y depends on h0; the true mean of y is 3. Under MAR dropout depends on h0, under MNAR on y itself."""
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(n, 2))
    y = 3 + 2 * h[:, 0] + rng.normal(size=n)
    driver = h[:, 0] if mechanism == "mar" else (y - 3) / y.std()
    observed = rng.random(n) < expit(1.5 * driver)
    return np.where(observed, y, np.nan), observed, h


def test_complete_cases_are_biased_and_corrections_are_not_under_mar():
    y, observed, h = _arm()
    err = {m: abs(arm_mean(y, observed, h, m) - 3) for m in METHODS}
    assert err["complete"] > 0.5
    assert err["ipcw"] < 0.1 and err["aipw"] < 0.1
    assert err["complete"] > 3 * max(err["ipcw"], err["aipw"])


def test_corrections_fail_when_dropout_depends_on_the_outcome():
    y, observed, h = _arm(mechanism="mnar")
    for m in METHODS:
        assert abs(arm_mean(y, observed, h, m) - 3) > 0.3, m


def test_week96_history_excludes_the_outcome():
    assert "cd420" in HISTORY96
    assert not {"cd496", "y96", "observed96", "treat", "arms", "cens", "days", "offtrt"} & set(HISTORY96)
    df = pd.DataFrame({c: [1.0, 2.0] for c in HISTORY96} | {"cd496": [5.0, np.nan]})
    assert week96_history(df).shape == (2, len(HISTORY96))


def test_randomised_effect_in_a_group_with_one_arm_is_nan():
    y, t = np.array([1.0, 2.0, 3.0, 4.0]), np.array([1, 1, 0, 0])
    assert np.isnan(effect_in_group(y, t, np.array([True, True, False, False])))
    assert effect_in_group(y, t, np.ones(4, dtype=bool)) == -2.0
