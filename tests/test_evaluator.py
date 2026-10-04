import numpy as np
import pytest

from firstpr.eval.evaluator import Evaluator
from firstpr.models.base import Recommender


class FixedScores(Recommender):
    """Scores given by a function of the data; used to build oracle / adversarial models."""

    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = matrix

    def fit(self, data, config, val_fn=None):
        return {}

    def score(self, user_ids):
        return self.matrix[user_ids]


class RandomScores(Recommender):
    def __init__(self, n_items: int, seed: int) -> None:
        self.n_items, self.rng = n_items, np.random.default_rng(seed)

    def fit(self, data, config, val_fn=None):
        return {}

    def score(self, user_ids):
        return self.rng.random((len(user_ids), self.n_items))


def test_oracle_scores_one(toy_data):
    ev = Evaluator(toy_data, k=20)
    for mode, targets in (("val", toy_data.val), ("test", toy_data.test)):
        res = ev.evaluate(FixedScores(targets.toarray()), mode)
        for m in ("recall@20", "ndcg@20", "hit@20"):
            assert res["overall"][m] == pytest.approx(1.0)
        assert res["n_users"] == toy_data.n_users


def test_train_items_are_masked(toy_data):
    ev = Evaluator(toy_data, k=20)
    model = FixedScores(toy_data.train.toarray() * 100.0)  # loves its train items
    users = np.arange(toy_data.n_users)
    top = ev.topk(model, users, toy_data.train)
    train = toy_data.train.toarray().astype(bool)
    for u in users:
        rec = top[u][top[u] >= 0]
        assert not train[u, rec].any()


def test_val_items_masked_in_test_mode(toy_data):
    ev = Evaluator(toy_data, k=20)
    seen = (toy_data.train + toy_data.val).toarray()
    model = FixedScores(toy_data.val.toarray() * 100.0)  # loves its val items
    _, mask = ev._targets_and_mask("test")
    top = ev.topk(model, np.arange(toy_data.n_users), mask)
    for u in range(toy_data.n_users):
        assert not seen[u, top[u][top[u] >= 0]].any()
    # in val mode the same model is an oracle
    assert ev.evaluate(model, "val")["overall"]["recall@20"] == pytest.approx(1.0)


def test_masked_slots_are_empty_when_catalog_too_small(make_data):
    data = make_data(n_users=10, n_items=25, per_user=(10, 15))
    ev = Evaluator(data, k=20)
    top = ev.topk(FixedScores(np.ones((10, 25))), np.arange(10), data.train)
    n_rankable = 25 - data.train.getnnz(axis=1)
    assert ((top >= 0).sum(axis=1) == np.minimum(20, n_rankable)).all()


def test_random_model_recall_matches_expectation(make_data):
    # ~5000 users x ~2.5 targets x 4 random models -> ~2% relative noise; tolerance 8% (~4 sigma)
    data = make_data(n_users=5000, n_items=400, per_user=(10, 40), seed=3)
    ev = Evaluator(data, k=20)
    recalls = [ev.evaluate(RandomScores(data.n_items, seed=s), "test") for s in range(4)]
    observed = np.mean([r["overall"]["recall@20"] for r in recalls])
    # each target lands in the top-K with prob K / (n_items - |masked items of u|)
    users = ev.users_to_evaluate("test")
    masked = (data.train + data.val).getnnz(axis=1)[users]
    expected = np.mean(20 / (data.n_items - masked))
    assert observed == pytest.approx(expected, rel=0.08)
    assert expected > 20 / data.n_items  # masking makes random slightly better than K/n


def test_slices_split_targets_by_head_and_tail(toy_data):
    ev = Evaluator(toy_data, k=20)
    res = ev.evaluate(FixedScores(toy_data.test.toarray()), "test")
    for name in ("head", "tail"):
        assert res["slices"][name]["recall@20"] == pytest.approx(1.0)
        assert res["slices"][name]["n_users"] > 0


def test_wrong_score_shape_raises(toy_data):
    ev = Evaluator(toy_data, k=20)
    with pytest.raises(ValueError):
        ev.evaluate(FixedScores(np.zeros((toy_data.n_users, 3))), "val")


def test_cold_item_slice_only_when_cold_items_exist():
    import pandas as pd

    from firstpr.data.dataset import InteractionData
    from firstpr.eval.evaluator import Evaluator
    from firstpr.models.popularity import Popularity

    train = pd.DataFrame({"user": [0, 0, 0, 0, 1, 1], "item": [0, 1, 2, 4, 0, 1]})
    val = pd.DataFrame({"user": [0, 1], "item": [5, 2]})
    test = pd.DataFrame({"user": [0, 1], "item": [3, 4]})  # item 3: no train interactions
    data = InteractionData.from_frames(train, val, test, n_users=2, n_items=6)
    pop = Popularity()
    pop.fit(data, {})
    res = Evaluator(data, k=2).evaluate(pop, "test", per_user=True)
    assert set(res["slices"]) == {"head", "tail", "cold"}
    assert res["slices"]["cold"]["n_users"] == 1
    cold = res["per_user"]["cold_ndcg@2"]
    assert np.isnan(cold).sum() == 1 and np.nanmax(cold) >= 0
