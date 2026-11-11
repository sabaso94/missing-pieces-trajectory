"""All figures of the README, from the CSVs in results/.

    python scripts/make_figures.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trial_trajectories.data import ROOT

RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
ARMS = {"zidovudine alone": "#4d4d4d", "newer regimens": "#2166ac"}
ESTIMATORS = {"difference in means": "#999999", "ANCOVA": "#2166ac", "IPW": "#e08214", "AIPW, linear": "#1b7837",
              "AIPW, boosting": "#b2182b"}
METHODS = {"complete": ("complete cases", "#999999"), "ipcw": ("IPCW", "#e08214"), "aipw": ("AIPW", "#1b7837")}
LEARNERS = {"T-learner, linear": "#2166ac", "T-learner, boosting": "#b2182b", "DR-learner": "#1b7837"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=150)
    plt.close(fig)


def errorbar(ax, x, est, low, high, **kw):
    ax.errorbar(x, est, yerr=[np.asarray(est) - np.asarray(low), np.asarray(high) - np.asarray(est)], **kw)


def e1_figure() -> None:
    traj = pd.read_csv(RESULTS / "e1_trajectory.csv")
    eff = pd.read_csv(RESULTS / "e1_effects.csv")
    eff = eff[eff["comparison"] == "newer regimens vs zidovudine alone"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for k, (group, color) in enumerate(ARMS.items()):
        g = traj[traj["group"] == group].sort_values("week")
        errorbar(ax1, g["week"] + (k - 0.5) * 2, g["mean_cd4"], g["low"], g["high"], fmt="o-", color=color,
                 capsize=3, label=group)
    ax1.set_xticks([0, 20, 96], ["0", "20", "96\n(patients seen)"])
    ax1.set_xlabel("week")
    ax1.set_ylabel("mean CD4 count (cells/mm³)")
    ax1.set_title("CD4 counts by arm")
    ax1.legend(frameon=False)
    quantities = {"natural course (change on zidovudine alone)": ("natural course", "#4d4d4d"),
                  "before-after change in the treated": ("before-after change, treated", "#e08214"),
                  "effect, ANCOVA": ("treatment effect (randomised)", "#2166ac")}
    for j, (q, (label, color)) in enumerate(quantities.items()):
        r = eff[eff["quantity"] == q].sort_values("week")
        x = np.arange(2) + (j - 1) * 0.22
        ax2.bar(x, r["est"], width=0.2, color=color, label=label)
        errorbar(ax2, x, r["est"], r["low"], r["high"], fmt="none", color="black", capsize=3, lw=1)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xticks([0, 1], ["week 20", "week 96"])
    ax2.set_ylabel("change in CD4 count (cells/mm³)")
    ax2.set_title("Natural course, before-after change and effect")
    ax2.legend(frameon=False, loc="lower left")
    save(fig, "e1_trajectory.png")


def e2_figure() -> None:
    s = pd.read_csv(RESULTS / "e2_summary.csv")
    panels = [("measured", "All covariates measured: bias"), ("cd40 hidden", "Baseline CD4 hidden: bias"),
              ("measured", "All covariates measured: spread (SD)")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    left_out = []
    for ax, (adjustment, title) in zip(axes, panels):
        col = "sd" if "spread" in title else "bias"
        for name, color in ESTIMATORS.items():
            r = s[(s["adjustment"] == adjustment) & (s["estimator"] == name)]
            ax.plot(r["gamma"], r[col], "o-", color=color, label=name)
        if adjustment == "cd40 hidden":
            r = s[(s["adjustment"] == "cd40 hidden, karnof and symptom left out") & (s["estimator"] == "ANCOVA")]
            if len(r):
                left_out = ax.plot(r["gamma"], r["bias"], "o--", color=ESTIMATORS["ANCOVA"],
                                   label="ANCOVA without Karnofsky\nscore and symptoms")
        if col == "bias":
            ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(sorted(s["gamma"].unique()))
        ax.set_xlabel("confounding strength γ")
        ax.set_ylabel("CD4 cells/mm³")
        ax.set_title(title)
    axes[0].legend(frameon=False, fontsize=8)
    if left_out:
        axes[1].legend(frameon=False, fontsize=8, handles=left_out)
    save(fig, "e2_confounding.png")


def e3_figure() -> None:
    sims = pd.read_csv(RESULTS / "e3_simulation.csv")
    w96 = pd.read_csv(RESULTS / "e3_week96.csv")
    traj = pd.read_csv(RESULTS / "e1_trajectory.csv")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.2))
    mechanisms = list(pd.unique(sims["mechanism"]))
    for ax, quantity, title in [(ax1, "control mean", "Week 20, zidovudine alone: bias of the mean change"),
                                (ax2, "effect", "Week 20: bias of the effect")]:
        for j, (method, (label, color)) in enumerate(METHODS.items()):
            r = sims[(sims["method"] == method) & (sims["quantity"] == quantity)].set_index("mechanism").loc[mechanisms]
            x = np.arange(len(mechanisms)) + (j - 1) * 0.25
            ax.bar(x, r["bias"], width=0.23, color=color, label=label, yerr=r["sd"], capsize=2,
                   error_kw={"lw": 0.8})
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(np.arange(len(mechanisms)), [m.replace(" on the ", "\non the ") for m in mechanisms])
        ax.set_ylabel("bias (cells/mm³), bar: ±1 SD over masks")
        ax.set_title(title, fontsize=10)
    ax1.legend(frameon=False, fontsize=8)
    for k, (group, color) in enumerate(ARMS.items()):
        g = traj[(traj["group"] == group) & (traj["week"] < 96)].sort_values("week")
        ax3.plot(g["week"], g["mean_cd4"], "o-", color=color, label=group)
        for j, (method, (label, mcolor)) in enumerate(METHODS.items()):
            r = w96[(w96["method"] == method) & (w96["quantity"] == f"CD4 at week 96, {group}")]
            errorbar(ax3, [96 + (j - 1) * 3], r["est"], r["low"], r["high"], fmt="o", color=mcolor, capsize=3,
                     label=label if k == 0 else None)
            ax3.plot([20, 96 + (j - 1) * 3], [g["mean_cd4"].iloc[-1], r["est"].iloc[0]], color=color, lw=0.6, alpha=0.6)
    ax3.set_xticks([0, 20, 96])
    ax3.set_xlabel("week")
    ax3.set_ylabel("mean CD4 count (cells/mm³)")
    ax3.set_title("Week 96 with and without correcting for dropout", fontsize=10)
    ax3.legend(frameon=False, fontsize=8)
    save(fig, "e3_dropout.png")


def e4_figure() -> None:
    cal = pd.read_csv(RESULTS / "e4_calibration.csv")
    sub = pd.read_csv(RESULTS / "e4_subgroups.csv")
    overall = pd.read_csv(RESULTS / "e1_effects.csv")
    overall = overall[(overall["comparison"] == "newer regimens vs zidovudine alone") & (overall["week"] == 20)
                      & (overall["quantity"] == "effect, difference in means")]["est"].iloc[0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    lim = [-20, 125]
    ax1.plot(lim, lim, color="#999999", lw=1, ls="--")
    for name, color in LEARNERS.items():
        r = cal[cal["learner"] == name].sort_values("group")
        errorbar(ax1, r["predicted"], r["effect"], r["low"], r["high"], fmt="o-", color=color, capsize=3, label=name)
    ax1.axhline(overall, color="black", lw=0.8, ls=":")
    ax1.set_xlim(lim)
    ax1.set_xlabel("predicted effect, mean in quartile")
    ax1.set_ylabel("randomised effect in quartile")
    ax1.set_title("Predicted benefit against randomised benefit")
    ax1.legend(frameon=False, fontsize=8, loc="upper left")
    labels = [f"{f}: {g}" for f, g in zip(sub["factor"], sub["group"])]
    y = np.arange(len(sub))[::-1]
    ax2.errorbar(sub["effect"], y, xerr=[sub["effect"] - sub["low"], sub["high"] - sub["effect"]], fmt="o",
                 color="#2166ac", capsize=3)
    ax2.axvline(overall, color="black", lw=0.8, ls=":")
    ax2.set_yticks(y, labels, fontsize=8)
    ax2.set_xlabel("randomised effect at week 20 (cells/mm³)")
    ax2.set_title("Prespecified subgroups")
    save(fig, "e4_benefit.png")


if __name__ == "__main__":
    e1_figure()
    e2_figure()
    e3_figure()
    e4_figure()
