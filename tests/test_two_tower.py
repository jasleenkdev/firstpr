import numpy as np
import pytest
import torch

from firstpr.eval.evaluator import Evaluator
from firstpr.models.popularity import Popularity
from firstpr.models.two_tower import TwoTower
from firstpr.utils.seed import set_seed

SMALL = {
    "dim": 16,
    "hidden": [32, 16],
    "init_std": 0.1,
    "lr": 0.01,
    "weight_decay": 0.0,
    "temperature": 0.1,
    "batch_size": 128,
    "logq": True,
    "max_epochs": 30,
    "patience": 5,
    "eval_every": 1,
    "seed": 0,
}


def test_hist_mean_excludes_target(toy_data):
    m = TwoTower()
    set_seed(0)
    m.fit(toy_data, {**SMALL, "max_epochs": 1})
    u = np.array([0, 1])
    items = [toy_data.train[x].indices for x in u]
    target = torch.tensor([items[0][0], items[1][0]])
    emb = m.module.item_emb.weight.detach()
    got = m._hist_mean(u, exclude=target).detach()
    for r in range(2):
        rest = [j for j in items[r] if j != target[r].item()]
        torch.testing.assert_close(got[r], emb[rest].mean(0), atol=1e-5, rtol=1e-5)


def test_scores_are_bounded_cosines(toy_data):
    m = TwoTower()
    set_seed(0)
    m.fit(toy_data, {**SMALL, "max_epochs": 1})
    s = m.score(np.arange(toy_data.n_users))
    assert s.shape == (toy_data.n_users, toy_data.n_items)
    assert np.abs(s).max() <= 1.0 + 1e-5


@pytest.mark.parametrize("logq", [True, False])
def test_two_tower_personalises_on_planted_clusters(logq):
    from test_mf_bpr import _planted

    set_seed(0)
    data = _planted()
    ev = Evaluator(data, k=10)
    model = TwoTower()
    info = model.fit(
        data,
        {**SMALL, "logq": logq},
        val_fn=lambda m: ev.evaluate(m, "val")["overall"]["ndcg@10"],
    )
    assert info["best_epoch"] >= 1
    users = np.arange(130, 200)
    mask = (data.train + data.val).tocsr()
    pop = Popularity()
    pop.fit(data, {})
    assert (ev.topk(pop, users, mask) >= 30).mean() < 0.5
    assert (ev.topk(model, users, mask) >= 30).mean() > 0.9
