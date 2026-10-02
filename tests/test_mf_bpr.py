import numpy as np
import pandas as pd
import torch

from firstpr.data.dataset import InteractionData
from firstpr.eval.evaluator import Evaluator
from firstpr.models.mf_bpr import MFBPR
from firstpr.models.popularity import Popularity
from firstpr.train.samplers import UniformNegativeSampler
from firstpr.utils.seed import set_seed

SMALL = {
    "dim": 16,
    "init_std": 0.1,
    "lr": 0.01,
    "reg": 1e-4,
    "batch_size": 256,
    "max_epochs": 40,
    "patience": 5,
    "eval_every": 1,
    "seed": 0,
}


def test_sampler_never_returns_a_positive(toy_data):
    s = UniformNegativeSampler(toy_data.train, seed=0)
    train = toy_data.train.toarray().astype(bool)
    n = 0
    for u, i, j in s.epoch(batch_size=64):
        assert train[u, i].all()
        assert not train[u, j].any()
        n += len(u)
    assert n == toy_data.train.nnz  # each positive once per epoch


def test_sampler_is_seeded(toy_data):
    a = next(UniformNegativeSampler(toy_data.train, seed=1).epoch(32))
    b = next(UniformNegativeSampler(toy_data.train, seed=1).epoch(32))
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)


def _planted() -> InteractionData:
    """Two user groups with disjoint item clusters; group A is bigger, so Popularity serves
    cluster A to everyone and fails group B. A personalised model should not."""
    rng = np.random.default_rng(0)
    rows = []
    for u in range(200):
        cluster = np.arange(0, 30) if u < 130 else np.arange(30, 60)
        items = rng.choice(cluster, size=15, replace=False)
        rows += [(u, int(i), t) for t, i in enumerate(items)]
    df = pd.DataFrame(rows, columns=["user", "item", "timestamp"])
    df = df.sort_values(["user", "timestamp"])
    last = df.groupby("user").cumcount(ascending=False)
    test, val, train = df[last == 0], df[last == 1], df[last >= 2]
    return InteractionData.from_frames(train, val, test, n_users=200, n_items=60)


def test_loss_decreases_and_beats_popularity_on_planted_clusters():
    set_seed(0)
    data = _planted()
    ev = Evaluator(data, k=10)
    model = MFBPR()
    info = model.fit(data, SMALL, val_fn=lambda m: ev.evaluate(m, "val")["overall"]["ndcg@10"])
    losses = [h["loss"] for h in info["history"]]
    assert losses[-1] < losses[0]
    assert info["best_epoch"] >= 1

    pop = Popularity()
    pop.fit(data, {})
    mf_ndcg = ev.evaluate(model, "test")["overall"]["ndcg@10"]
    pop_ndcg = ev.evaluate(pop, "test")["overall"]["ndcg@10"]
    assert mf_ndcg > pop_ndcg + 0.1


def test_bpr_loss_matches_formula():
    from firstpr.models.mf_bpr import MFModule

    torch.manual_seed(0)
    m = MFModule(3, 4, dim=2, init_std=1.0)
    u, i, j = torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([3, 0])
    total, bpr = m.bpr_loss(u, i, j, reg=0.5)
    eu, ei, ej = m.user(u), m.item(i), m.item(j)
    x = (eu * ei).sum(-1) - (eu * ej).sum(-1)
    expected_bpr = -torch.log(torch.sigmoid(x)).mean()
    expected_l2 = 0.5 * (eu.pow(2).sum() + ei.pow(2).sum() + ej.pow(2).sum()) / 2
    assert torch.allclose(bpr, expected_bpr, atol=1e-6)
    assert torch.allclose(total, expected_bpr + 0.5 * expected_l2, atol=1e-6)
