import numpy as np
import pytest

from firstpr.eval.bootstrap import paired_bootstrap, pairwise_cis, seed_averaged_per_user
from firstpr.eval.evaluator import Evaluator
from firstpr.models.popularity import Popularity
from firstpr.train.runner import final, load_runs

EVAL = {"k": 10, "batch_size": 16, "seeds": [0, 1, 2], "tune_seed": 0, "primary_metric": "ndcg@10"}


def test_identical_models_ci_contains_zero():
    a = np.random.default_rng(0).random(500)
    res = paired_bootstrap(a, a.copy(), n_resamples=200, ci=0.95, seed=0)
    assert res["diff"] == 0 and res["ci_low"] == 0 and res["ci_high"] == 0


def test_clear_difference_excludes_zero_and_is_deterministic():
    rng = np.random.default_rng(1)
    b = rng.random(2000)
    a = b + 0.05 + rng.normal(0, 0.01, 2000)
    r1 = paired_bootstrap(a, b, n_resamples=500, ci=0.95, seed=3)
    r2 = paired_bootstrap(a, b, n_resamples=500, ci=0.95, seed=3)
    assert r1 == r2
    assert r1["ci_low"] > 0.04 and r1["ci_high"] < 0.06
    assert r1["p_not_positive"] == 0.0


def test_pairing_gives_narrower_ci_than_unpaired():
    # strongly correlated per-user scores: pairing removes between-user variance
    rng = np.random.default_rng(2)
    base = rng.random(1000)
    a, b = base + 0.01, base + rng.normal(0, 0.001, 1000)
    paired = paired_bootstrap(a, b, 500, 0.95, 0)
    width_paired = paired["ci_high"] - paired["ci_low"]
    se_unpaired = np.sqrt(a.var() / len(a) + b.var() / len(b))
    assert width_paired < 2 * 1.96 * se_unpaired / 10


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        paired_bootstrap(np.zeros(3), np.zeros(4), 10, 0.95, 0)


def test_evaluator_per_user_matches_overall(toy_data):
    pop = Popularity()
    pop.fit(toy_data, {})
    res = Evaluator(toy_data, k=10).evaluate(pop, "test", per_user=True)
    pu = res["per_user"]
    assert len(pu["users"]) == res["n_users"]
    assert pu["ndcg@10"].mean() == pytest.approx(res["overall"]["ndcg@10"])
    assert pu["recall@10"].mean() == pytest.approx(res["overall"]["recall@10"])


def test_runner_saves_per_user_files_and_pairwise_cis(tmp_path, toy_data):
    final("popularity", toy_data, {"params": {}, "search": {}}, EVAL, "toy", tmp_path)
    knn = {"params": {"k": 10, "shrink": 0}, "search": {}}
    final("itemknn", toy_data, knn, EVAL, "toy", tmp_path)
    runs = load_runs(tmp_path, "test")
    assert all((tmp_path / "runs" / "toy" / r["model"] / r["per_user_file"]).exists() for r in runs)
    pop_runs = [r for r in runs if r["model"] == "popularity"]
    _, vals = seed_averaged_per_user(pop_runs, "ndcg@10")
    assert vals.mean() == pytest.approx(pop_runs[0]["metrics"]["overall"]["ndcg@10"])
    final_runs: dict = {}
    for r in runs:
        final_runs.setdefault((r["dataset"], r["model"]), []).append(r)
    cis = pairwise_cis(final_runs, "ndcg@10", 100, 0.95, 0)
    assert len(cis) == 1
    row = cis.iloc[0]
    assert row["ci_low"] <= row["diff"] <= row["ci_high"]
