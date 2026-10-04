"""LightGCN (He, Deng, Wang, Li, Zhang & Wang, SIGIR 2020).

Graph: users and items are nodes of one bipartite graph built from *train* interactions only.
Adjacency A = [[0, R], [R^T, 0]] (R = users x items), symmetric normalisation
A_hat = D^-1/2 A D^-1/2. Propagation has no feature transform and no non-linearity:

    E^(k+1) = A_hat E^(k),    final E = mean_k=0..K E^(k)   (layer combination, alpha_k = 1/(K+1))

score(u, i) = <e_u, e_i>. Training: BPR with uniform negatives, L2 on the *layer-0* embeddings of
the batch (as in the paper's code), Adam. With K = 0 this is exactly MF-BPR.
"""

from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

from firstpr.data.dataset import InteractionData
from firstpr.models.base import Recommender, ValFn
from firstpr.train.samplers import UniformNegativeSampler
from firstpr.train.trainer import train_with_early_stopping


def normalized_adjacency(train: sp.csr_matrix) -> sp.csr_matrix:
    """D^-1/2 A D^-1/2 for the bipartite user-item graph; isolated nodes get 0 rows."""
    n_users, n_items = train.shape
    r = train.astype(np.float32)
    a = sp.bmat([[None, r], [r.T, None]], format="csr", dtype=np.float32)
    deg = np.asarray(a.sum(axis=1)).ravel()
    inv_sqrt = np.zeros_like(deg)
    nz = deg > 0
    inv_sqrt[nz] = deg[nz] ** -0.5
    d = sp.diags(inv_sqrt)
    return (d @ a @ d).tocsr()


def to_torch_sparse(m: sp.csr_matrix) -> torch.Tensor:
    m = m.tocoo()
    idx = torch.from_numpy(np.vstack([m.row, m.col]).astype(np.int64))
    return torch.sparse_coo_tensor(
        idx, torch.from_numpy(m.data.astype(np.float32)), m.shape, check_invariants=False
    ).coalesce()


class LightGCNModule(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int, n_layers: int, init_std: float):
        super().__init__()
        self.n_users, self.n_items, self.n_layers = n_users, n_items, n_layers
        self.user = nn.Embedding(n_users, dim)
        self.item = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user.weight, std=init_std)
        nn.init.normal_(self.item.weight, std=init_std)

    def propagate(self, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e = torch.cat([self.user.weight, self.item.weight])
        layers = [e]
        for _ in range(self.n_layers):
            e = torch.sparse.mm(adj, e)
            layers.append(e)
        final = torch.stack(layers).mean(0)
        return final[: self.n_users], final[self.n_users :]


class LightGCN(Recommender):
    name = "lightgcn"

    def fit(
        self, data: InteractionData, config: dict[str, Any], val_fn: ValFn | None = None
    ) -> dict[str, Any]:
        seed = int(config.get("seed", 0))
        self.adj_ = to_torch_sparse(normalized_adjacency(data.train))
        self.module = LightGCNModule(
            data.n_users,
            data.n_items,
            int(config["dim"]),
            int(config["n_layers"]),
            float(config["init_std"]),
        )
        opt = torch.optim.Adam(self.module.parameters(), lr=float(config["lr"]))
        sampler = UniformNegativeSampler(data.train, seed=seed)
        reg, batch_size = float(config["reg"]), int(config["batch_size"])
        self._cache = None

        def run_epoch(epoch: int) -> float:
            self.module.train()
            self._cache = None
            total, n = 0.0, 0
            for u, i, j in sampler.epoch(batch_size):
                u_t, i_t, j_t = (torch.from_numpy(a) for a in (u, i, j))
                users, items = self.module.propagate(self.adj_)
                eu, ei, ej = users[u_t], items[i_t], items[j_t]
                x_uij = (eu * ei).sum(-1) - (eu * ej).sum(-1)
                bpr = torch.nn.functional.softplus(-x_uij).mean()
                e0 = (self.module.user(u_t), self.module.item(i_t), self.module.item(j_t))
                l2 = 0.5 * sum(e.pow(2).sum() for e in e0) / len(u)
                loss = bpr + reg * l2
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += bpr.item() * len(u)
                n += len(u)
            self.module.eval()
            return total / n

        info = train_with_early_stopping(
            self.module,
            run_epoch,
            (lambda: val_fn(self)) if val_fn is not None else None,
            max_epochs=int(config["max_epochs"]),
            patience=int(config["patience"]),
            eval_every=int(config.get("eval_every", 1)),
            min_epochs=int(config.get("min_epochs", 0)),
        )
        self._cache = None
        self.module.eval()
        return info

    @torch.no_grad()
    def score(self, user_ids: np.ndarray) -> np.ndarray:
        if self._cache is None:  # embeddings are fixed between training epochs
            self._cache = self.module.propagate(self.adj_)
        users, items = self._cache
        return (users[torch.as_tensor(user_ids)] @ items.T).numpy()
