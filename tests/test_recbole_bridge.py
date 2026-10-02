import shutil

import numpy as np
import pytest

from firstpr.recbole_bridge.export import export_benchmark
from firstpr.recbole_bridge.model import RecBoleModel


def _read(path):
    rows = [line.rstrip("\n").split("\t") for line in open(path)][1:]
    return [(int(u), int(i), float(t)) for u, i, t in rows]


def test_export_preserves_split_and_order(toy_data, tmp_path):
    d = export_benchmark(toy_data, tmp_path, "toy")
    train, val, test = (_read(d / f"toy.{s}.inter") for s in ("train", "valid", "test"))
    assert len(train) == toy_data.train.nnz
    assert len(val) == toy_data.val.nnz
    assert len(test) == toy_data.test.nnz
    # train order key = position in our tie-broken history
    for u in (0, 5):
        rows = sorted((t, i) for uu, i, t in train if uu == u)
        assert [i for _, i in rows] == list(toy_data.train_histories[u])
    # val after train, test after val, per user
    last_train = {u: max(t for uu, _, t in train if uu == u) for u in range(toy_data.n_users)}
    assert all(t > last_train[u] for u, _, t in val)
    last_val = {u: max(t for uu, _, t in val if uu == u) for u in {u for u, _, _ in val}}
    assert all(t > last_val[u] for u, _, t in test)


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv for the recbole env")
def test_recbole_pop_scores_map_back_to_our_ids(toy_data, tmp_path):
    # RecBole's Pop model scores items by train count: after mapping back to our ids the score
    # matrix must be proportional to our own item popularity -> checks the id mapping exactly.
    # Batch size 1: Pop increments `item_cnt[items] += 1` per batch, so duplicates inside a
    # batch would be counted once (a RecBole quirk), which would blur the check.
    model = RecBoleModel("Pop", work_dir=tmp_path)
    # no negative sampling: Pop is pointwise and would count sampled negatives as items too
    model.fit(
        toy_data,
        {"epochs": 1, "train_batch_size": 1, "train_neg_sample_args": None, "seed": 0},
    )
    s = model.score(np.arange(toy_data.n_users))
    assert s.shape == (toy_data.n_users, toy_data.n_items)
    pop = toy_data.item_popularity.astype(float)
    row = s[0] / s[0].max()
    np.testing.assert_allclose(row, pop / pop.max(), rtol=1e-4, atol=1e-6)
