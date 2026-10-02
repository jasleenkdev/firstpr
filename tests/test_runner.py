import pytest

from firstpr.eval.leaderboard import aggregate, to_markdown
from firstpr.train.runner import expand_grid, final, load_runs, tune
from firstpr.utils.io import load_yaml

EVAL = {"k": 10, "batch_size": 16, "seeds": [0, 1, 2], "tune_seed": 0, "primary_metric": "ndcg@10"}


def test_expand_grid():
    grid = expand_grid({"a": 1, "b": 2}, {"b": [3, 4], "c": [5]})
    assert grid == [{"a": 1, "b": 3, "c": 5}, {"a": 1, "b": 4, "c": 5}]
    assert expand_grid({"a": 1}, {}) == [{"a": 1}]


def test_tune_then_final_then_leaderboard(tmp_path, toy_data, monkeypatch):
    import firstpr.eval.leaderboard as lb

    monkeypatch.setattr(lb, "METRICS", [m.replace("@20", "@10") for m in lb.METRICS])
    monkeypatch.setattr(lb, "SLICE_METRICS", [("tail", "recall@10"), ("head", "recall@10")])

    knn = {"params": {"k": 10, "shrink": 0}, "search": {"k": [5, 10], "shrink": [0, 10]}}
    best = tune("itemknn", toy_data, knn, EVAL, "toy", tmp_path)
    assert best["n_configs"] == 4
    saved = load_yaml(tmp_path / "best" / "itemknn_toy.yaml")
    assert saved["params"] == best["params"]
    assert saved["val_score"] == max(g["ndcg@10"] for g in saved["grid"])

    recs = final("itemknn", toy_data, knn, EVAL, "toy", tmp_path)
    assert [r["seed"] for r in recs] == [0, 1, 2]
    assert all(r["config"] == best["params"] for r in recs)
    with pytest.raises(RuntimeError):  # test is touched once per final config
        final("itemknn", toy_data, knn, EVAL, "toy", tmp_path)

    final("popularity", toy_data, {"params": {}, "search": {}}, EVAL, "toy", tmp_path)
    runs = load_runs(tmp_path, "test")
    assert len(runs) == 6
    rec = runs[0]
    for key in ("config", "seed", "git", "train_time_s", "inference_time_s", "metrics"):
        assert key in rec

    df = aggregate(runs, model_order=["popularity", "itemknn"])
    assert list(df["model"]) == ["popularity", "itemknn"]
    assert (df["n_seeds"] == 3).all()
    assert df.loc[0, "ndcg@10_std"] == 0.0  # deterministic model
    md = to_markdown(df.rename(columns=lambda c: c.replace("@10", "@20")))
    assert "| popularity | toy | 3 |" in md


def test_final_requires_tuning_when_there_is_a_grid(tmp_path, toy_data):
    with pytest.raises(FileNotFoundError):
        final("itemknn", toy_data, {"params": {}, "search": {"k": [5]}}, EVAL, "toy", tmp_path)


def test_tune_reuses_finished_val_runs(tmp_path, toy_data):
    knn = {"params": {"k": 10, "shrink": 0}, "search": {"k": [5, 10]}}
    tune("itemknn", toy_data, knn, EVAL, "toy", tmp_path)
    n_before = len(list((tmp_path / "runs").rglob("val_*.json")))
    knn["search"]["k"].append(20)  # extend the grid: only the new config runs
    best = tune("itemknn", toy_data, knn, EVAL, "toy", tmp_path)
    assert len(list((tmp_path / "runs").rglob("val_*.json"))) == n_before + 1
    assert best["n_configs"] == 3
