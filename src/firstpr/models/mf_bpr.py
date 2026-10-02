"""Matrix factorisation trained with Bayesian Personalized Ranking (Rendle et al., UAI 2009).

score(u, i) = <e_u, e_i>. For a train positive i and a sampled negative j:
loss = -log sigmoid(x_ui - x_uj) + reg * (|e_u|^2 + |e_i|^2 + |e_j|^2) / 2, averaged over the
batch (L2 on the batch embeddings, as in the LightGCN reference code).
"""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from firstpr.data.dataset import InteractionData
from firstpr.models.base import Recommender, ValFn
from firstpr.train.samplers import UniformNegativeSampler
from firstpr.train.trainer import train_with_early_stopping


class MFModule(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int, init_std: float) -> None:
        super().__init__()
        self.user = nn.Embedding(n_users, dim)
        self.item = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user.weight, std=init_std)
        nn.init.normal_(self.item.weight, std=init_std)

    def bpr_loss(
        self, u: torch.Tensor, i: torch.Tensor, j: torch.Tensor, reg: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eu, ei, ej = self.user(u), self.item(i), self.item(j)
        x_uij = (eu * ei).sum(-1) - (eu * ej).sum(-1)
        bpr = F.softplus(-x_uij).mean()  # = -log sigmoid(x_uij), numerically stable
        l2 = 0.5 * (eu.pow(2).sum() + ei.pow(2).sum() + ej.pow(2).sum()) / len(u)
        return bpr + reg * l2, bpr


class MFBPR(Recommender):
    name = "mf_bpr"

    def fit(
        self, data: InteractionData, config: dict[str, Any], val_fn: ValFn | None = None
    ) -> dict[str, Any]:
        seed = int(config.get("seed", 0))
        self.module = MFModule(
            data.n_users, data.n_items, int(config["dim"]), float(config["init_std"])
        )
        opt = torch.optim.Adam(self.module.parameters(), lr=float(config["lr"]))
        sampler = UniformNegativeSampler(data.train, seed=seed)
        reg, batch_size = float(config["reg"]), int(config["batch_size"])

        def run_epoch(epoch: int) -> float:
            self.module.train()
            total, n = 0.0, 0
            for u, i, j in sampler.epoch(batch_size):
                u_t, i_t, j_t = (torch.from_numpy(a) for a in (u, i, j))
                loss, bpr = self.module.bpr_loss(u_t, i_t, j_t, reg)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += bpr.item() * len(u)
                n += len(u)
            return total / n

        info = train_with_early_stopping(
            self.module,
            run_epoch,
            (lambda: val_fn(self)) if val_fn is not None else None,
            max_epochs=int(config["max_epochs"]),
            patience=int(config["patience"]),
            eval_every=int(config.get("eval_every", 1)),
        )
        self.module.eval()
        return info

    @torch.no_grad()
    def score(self, user_ids: np.ndarray) -> np.ndarray:
        u = self.module.user.weight[torch.as_tensor(user_ids)]
        return (u @ self.module.item.weight.T).numpy()
