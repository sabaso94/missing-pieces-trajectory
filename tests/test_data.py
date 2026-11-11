import numpy as np

from trial_trajectories.data import COVARIATES, load_trial

HEADER = ("pidnum age wtkg hemo homo drugs karnof oprior z30 zprior preanti race gender str2 strat symptom treat "
          "offtrt cd40 cd420 cd496 r cd80 cd820 cens days arms")
ROWS = ["10056 48 89.8128 0 0 0 100 0 0 1 0 0 0 0 1 0 1 0 422 477 660 1 566 324 0 948 2",
        "10059 61 49.4424 0 0 0 90 0 1 1 895 0 0 1 3 0 1 0 162 218 NA 0 392 564 1 1002 3",
        "10089 45 88.4520 0 1 1 90 0 1 1 707 0 1 1 3 0 0 1 326 274 122 1 2063 1893 0 961 0"]


def test_loader_derives_outcomes_from_a_small_file(tmp_path):
    path = tmp_path / "ACTG175.txt"
    path.write_text("\n".join([HEADER, *ROWS]) + "\n")
    df = load_trial(path)
    assert df["y20"].tolist() == [55, 56, -52]
    assert df["y96"].iloc[0] == 238 and np.isnan(df["y96"].iloc[1]) and df["y96"].iloc[2] == -204
    assert df["observed96"].tolist() == [True, False, True]


def test_covariates_exclude_post_randomisation_columns():
    after = {"offtrt", "cd420", "cd496", "cd820", "cens", "days", "arms", "r", "treat", "y20", "y96"}
    assert not after & set(COVARIATES)
    assert len(COVARIATES) == 15
