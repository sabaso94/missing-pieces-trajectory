"""E3: one patient in three has no week-96 CD4 count. Who drops out, and does it bias the long-term answer?

    python scripts/run_e3.py

(a) Who drops out: a logistic model of having a week-96 count, from baseline and arm, then with the week-20
    count added; standardised coefficients and 5-fold cross-validated AUROC. Also the week-20 change of patients
    with and without a week-96 count.
(b) A check with a known answer: week 20 is complete, so dropout is imposed on it, 500 times per mechanism, and
    the three estimators are scored against the full week-20 data. Mechanisms: the real dropout model fitted on
    baseline covariates in each arm ("as observed"), the same with its coefficients tripled ("3x stronger"), and
    dropout driven by the week-20 change itself ("depends on the outcome"); each keeps each arm's real dropout rate.
(c) Week 96: CD4 at week 96 per arm and the week-96 effect with complete cases, IPCW and AIPW, using everything
    known before the visit (baseline, week-20 count); bootstrap intervals.
Writes results/e3_dropout_model.csv, results/e3_dropout_profile.csv, results/e3_simulation.csv and
results/e3_week96.csv.
"""
import os

for _var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_var, "1")  # small models: one thread is fastest and keeps results identical

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score

from trial_trajectories.data import COVARIATES, ROOT, covariate_matrix, load_trial
from trial_trajectories.metrics import bootstrap_ci
from trial_trajectories.missing import HISTORY96, arm_mean, dropout_model, week96_history

METHODS = ["complete", "ipcw", "aipw"]
N_MASKS = 500


def who_drops_out(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    observed = df["observed96"].to_numpy()
    for name, cols in [("baseline + arm", COVARIATES + ["treat"]), ("baseline + arm + week-20 CD4", HISTORY96 + ["treat"])]:
        h = df[cols].to_numpy(float)
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        p = cross_val_predict(dropout_model(h, observed), h, observed.astype(int), cv=cv, method="predict_proba")[:, 1]
        coefs = dropout_model(h, observed)[-1].coef_[0]
        auroc = roc_auc_score(observed, p)
        rows += [{"model": name, "feature": c, "std_coef": b, "auroc_cv": auroc} for c, b in zip(cols, coefs)]
    return pd.DataFrame(rows)


def dropout_profile(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["treat", "observed96"])
    return g.agg(n=("y20", "size"), cd4_week0=("cd40", "mean"), cd4_week20=("cd420", "mean"),
                 change_week20=("y20", "mean")).reset_index()


def _match_rate(lp: np.ndarray, rate: float) -> np.ndarray:
    """P(observed) = expit(c + lp) with c chosen so that the mean equals rate."""
    c = brentq(lambda c: expit(c + lp).mean() - rate, -50, 50)
    return expit(c + lp)


def mechanisms(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """P(observed at week 20) per patient under each simulated dropout mechanism."""
    out = {name: np.empty(len(df)) for name in ["as observed", "3x stronger", "depends on the outcome"]}
    x = covariate_matrix(df)
    for arm in (0, 1):
        rows = (df["treat"] == arm).to_numpy()
        model = dropout_model(x[rows], df.loc[rows, "observed96"].to_numpy())
        lp = model.decision_function(x[rows])
        lp = lp - lp.mean()
        rate = df.loc[rows, "observed96"].mean()
        y = df.loc[rows, "y20"].to_numpy()
        out["as observed"][rows] = _match_rate(lp, rate)
        out["3x stronger"][rows] = _match_rate(3 * lp, rate)
        out["depends on the outcome"][rows] = _match_rate((y - y.mean()) / y.std(), rate)
    return out


def estimate(y: np.ndarray, observed: np.ndarray, h: np.ndarray, t: np.ndarray, method: str) -> dict:
    m0 = arm_mean(y[t == 0], observed[t == 0], h[t == 0], method)
    m1 = arm_mean(y[t == 1], observed[t == 1], h[t == 1], method)
    return {"control mean": m0, "treated mean": m1, "effect": m1 - m0}


def simulate(df: pd.DataFrame) -> pd.DataFrame:
    y, t, x = df["y20"].to_numpy(float), df["treat"].to_numpy(), covariate_matrix(df)
    truth = {"control mean": y[t == 0].mean(), "treated mean": y[t == 1].mean()}
    truth["effect"] = truth["treated mean"] - truth["control mean"]
    rows = []
    for k, (name, p_obs) in enumerate(mechanisms(df).items()):
        for draw in range(N_MASKS):
            observed = np.random.default_rng(1000 * k + draw).random(len(y)) < p_obs
            for method in METHODS:
                for quantity, value in estimate(y, observed, x, t, method).items():
                    rows.append({"mechanism": name, "draw": draw, "method": method, "quantity": quantity,
                                 "est": value, "truth": truth[quantity]})
    sims = pd.DataFrame(rows).assign(err=lambda d: d["est"] - d["truth"])
    return sims.groupby(["mechanism", "method", "quantity"], sort=False).agg(
        truth=("truth", "first"), bias=("err", "mean"), sd=("est", "std"),
        rmse=("err", lambda e: np.sqrt((e ** 2).mean()))).reset_index()


def week96(df: pd.DataFrame) -> pd.DataFrame:
    t, h, observed = df["treat"].to_numpy(), week96_history(df), df["observed96"].to_numpy()
    level, change = df["cd496"].to_numpy(float), df["y96"].to_numpy(float)
    rows = []
    for method in METHODS:
        def stats(idx):
            ti, hi, oi = t[idx], h[idx], observed[idx]
            lv, ch = estimate(level[idx], oi, hi, ti, method), estimate(change[idx], oi, hi, ti, method)
            return np.array([lv["control mean"], lv["treated mean"], ch["control mean"], ch["treated mean"],
                             ch["effect"]])
        point = stats(np.arange(len(t)))
        low, high = bootstrap_ci(stats, len(t), seed=96)
        names = ["CD4 at week 96, zidovudine alone", "CD4 at week 96, newer regimens",
                 "change to week 96, zidovudine alone", "change to week 96, newer regimens", "effect at week 96"]
        rows += [{"method": method, "quantity": q, "est": p, "low": lo, "high": hi}
                 for q, p, lo, hi in zip(names, point, low, high)]
    return pd.DataFrame(rows)


def main() -> None:
    df = load_trial()
    out = ROOT / "results"
    model = who_drops_out(df)
    model.to_csv(out / "e3_dropout_model.csv", index=False, float_format="%.4f")
    profile = dropout_profile(df)
    profile.to_csv(out / "e3_dropout_profile.csv", index=False, float_format="%.3f")
    sims = simulate(df)
    sims.to_csv(out / "e3_simulation.csv", index=False, float_format="%.4f")
    w96 = week96(df)
    w96.to_csv(out / "e3_week96.csv", index=False, float_format="%.3f")
    print(model.groupby("model")["auroc_cv"].first().round(3).to_string())
    print(model.assign(a=model.std_coef.abs()).sort_values("a", ascending=False).groupby("model").head(4)
          [["model", "feature", "std_coef"]].round(3).to_string(index=False))
    print(profile.round(1).to_string(index=False))
    print(sims.round(2).to_string(index=False))
    print(w96.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
