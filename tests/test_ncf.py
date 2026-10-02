import numpy as np
import pytest
import torch

from firstpr.eval.evaluator import Evaluator
from firstpr.models.ncf import NCF, NCFModule
from firstpr.models.popularity import Popularity
from firstpr.train.samplers import PointwiseSampler
from firstpr.utils.seed import set_seed

SMALL = {
    "factors": 8,
    "mlp_layers": [32, 16, 8],
    "init_std": 0.1,
    "lr": 0.01,
    "reg": 0.0,
    "n_neg": 4,
    "batch_size": 128,
    "max_epochs": 30,
    "patience": 5,
    "eval_every": 1,
    "seed": 0,
    "pretrain": False,
    "alpha": 0.5,
    "finetune_lr_scale": 0.5,
}


@pytest.mark.parametrize("variant", ["gmf", "mlp", "neumf"])
def test_score_all_matches_forward(variant):
    torch.manual_seed(0)
    m = NCFModule(5, 7, variant, factors=4, mlp_layers=[16, 8, 4], init_std=0.5)
    users = torch.tensor([0, 3, 4])
    full = m.score_all(users, chunk=2)
    for r, u in enumerate(users):
        pairs = m(u.repeat(7), torch.arange(7))
        torch.testing.assert_close(full[r], pairs, atol=1e-5, rtol=1e-5)


def test_pointwise_sampler_labels(toy_data):
    s = PointwiseSampler(toy_data.train, n_neg=4, seed=0)
    train = toy_data.train.toarray().astype(bool)
    n_pos = n_neg = 0
    for u, i, y in s.epoch(batch_size=50):
        assert (train[u, i] == (y == 1)).all()
        n_pos += int(y.sum())
        n_neg += int((y == 0).sum())
    assert n_pos == toy_data.train.nnz
    assert n_neg == 4 * n_pos


@pytest.mark.parametrize("variant,pretrain", [("gmf", False), ("mlp", False), ("neumf", True)])
def test_ncf_beats_popularity_on_planted_clusters(variant, pretrain):
    from test_mf_bpr import _planted

    set_seed(0)
    data = _planted()
    ev = Evaluator(data, k=10)
    model = NCF(variant)
    info = model.fit(
        data,
        {**SMALL, "pretrain": pretrain},
        val_fn=lambda m: ev.evaluate(m, "val")["overall"]["ndcg@10"],
    )
    assert info["best_epoch"] >= 1
    if pretrain:
        assert set(info["pretrain"]) == {"pretrain_gmf", "pretrain_mlp"}
    # minority-cluster users (130-199) like items 30-59; Popularity serves them cluster A
    users = np.arange(130, 200)
    mask = (data.train + data.val).tocsr()
    in_cluster = lambda m: (ev.topk(m, users, mask) >= 30).mean()  # noqa: E731
    pop = Popularity()
    pop.fit(data, {})
    assert in_cluster(pop) < 0.5
    assert in_cluster(model) > 0.9
    assert np.isfinite(model.score(np.arange(5))).all()
