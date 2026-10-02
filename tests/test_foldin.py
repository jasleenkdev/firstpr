import numpy as np

from firstpr.eval.evaluator import Evaluator
from firstpr.models.itemknn import ItemKNN
from firstpr.models.sasrec import SASRec
from firstpr.train.foldin import set_foldin_context
from firstpr.utils.seed import set_seed


def test_val_histories_are_ordered_and_complete(toy_data):
    for u in range(toy_data.n_users):
        assert sorted(toy_data.val_histories[u]) == sorted(toy_data.val[u].indices)


def test_itemknn_foldin_changes_scores_only_through_context(toy_data):
    m = ItemKNN()
    m.fit(toy_data, {"k": 10, "shrink": 0})
    users = np.arange(toy_data.n_users)
    before = m.score(users)
    w = m.w_.copy()
    set_foldin_context(m, "itemknn", toy_data)
    after = m.score(users)
    assert abs(m.w_ - w).max() == 0  # similarities untouched (no refit)
    expected = ((toy_data.train + toy_data.val) @ w).toarray()
    np.testing.assert_allclose(after, expected, rtol=1e-5)
    assert not np.allclose(before, after)


def test_sasrec_foldin_appends_val_items(toy_data):
    set_seed(0)
    m = SASRec()
    cfg = {
        "max_len": 50,
        "hidden": 8,
        "blocks": 1,
        "heads": 1,
        "dropout": 0.0,
        "lr": 0.01,
        "batch_size": 32,
        "loss": "ce",
        "max_epochs": 1,
        "patience": 1,
        "seed": 0,
    }
    m.fit(toy_data, cfg)
    set_foldin_context(m, "sasrec", toy_data)
    u = 3
    ctx = m.context_[u][m.context_[u] > 0] - 1
    expected = np.concatenate([toy_data.train_histories[u], toy_data.val_histories[u]])[-50:]
    np.testing.assert_array_equal(ctx, expected)
    assert Evaluator(toy_data, k=5).evaluate(m, "test")["n_users"] == toy_data.n_users
