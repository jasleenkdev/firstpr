import numpy as np
import pytest
import scipy.sparse as sp
import torch

from firstpr.eval.evaluator import Evaluator
from firstpr.models.lightgcn import LightGCN, LightGCNModule, normalized_adjacency, to_torch_sparse
from firstpr.models.popularity import Popularity
from firstpr.utils.seed import set_seed

SMALL = {
    "dim": 16,
    "n_layers": 2,
    "init_std": 0.1,
    "lr": 0.01,
    "reg": 1e-4,
    "batch_size": 256,
    "max_epochs": 40,
    "patience": 5,
    "eval_every": 1,
    "seed": 0,
}


def test_normalized_adjacency_hand_computed():
    # u0-i0, u0-i1, u1-i1 : deg u0=2, u1=1, i0=1, i1=2
    r = sp.csr_matrix(np.array([[1, 1], [0, 1]], dtype=np.float32))
    a = normalized_adjacency(r).toarray()
    assert a.shape == (4, 4)
    assert a[0, 2] == pytest.approx(1 / np.sqrt(2 * 1))  # u0 - i0
    assert a[0, 3] == pytest.approx(1 / np.sqrt(2 * 2))  # u0 - i1
    assert a[1, 3] == pytest.approx(1 / np.sqrt(1 * 2))  # u1 - i1
    np.testing.assert_allclose(a, a.T)
    assert a[0, 1] == 0 and a[2, 3] == 0  # bipartite: no user-user / item-item edges


def test_propagation_matches_dense_layer_mean():
    torch.manual_seed(0)
    r = sp.random(5, 7, density=0.4, format="csr", random_state=0)
    r.data[:] = 1
    adj = normalized_adjacency(r)
    m = LightGCNModule(5, 7, dim=3, n_layers=3, init_std=1.0)
    users, items = m.propagate(to_torch_sparse(adj))
    e = torch.cat([m.user.weight, m.item.weight]).detach().numpy()
    a = adj.toarray()
    expected = (e + a @ e + a @ a @ e + a @ a @ a @ e) / 4
    np.testing.assert_allclose(torch.cat([users, items]).detach().numpy(), expected, atol=1e-5)


def test_zero_layers_is_plain_mf():
    m = LightGCNModule(4, 6, dim=3, n_layers=0, init_std=1.0)
    users, items = m.propagate(to_torch_sparse(normalized_adjacency(sp.eye(4, 6, format="csr"))))
    torch.testing.assert_close(users, m.user.weight)
    torch.testing.assert_close(items, m.item.weight)


def test_lightgcn_personalises_on_planted_clusters():
    from test_mf_bpr import _planted

    set_seed(0)
    data = _planted()
    ev = Evaluator(data, k=10)
    model = LightGCN()
    model.fit(data, SMALL, val_fn=lambda m: ev.evaluate(m, "val")["overall"]["ndcg@10"])
    users = np.arange(130, 200)
    mask = (data.train + data.val).tocsr()
    pop = Popularity()
    pop.fit(data, {})
    assert (ev.topk(pop, users, mask) >= 30).mean() < 0.5
    assert (ev.topk(model, users, mask) >= 30).mean() > 0.9
