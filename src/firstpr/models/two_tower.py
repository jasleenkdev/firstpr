"""Two-tower retrieval model with in-batch softmax and logQ correction (Yi et al., RecSys 2019).

- User tower: MLP([user-id embedding ; mean of the user's train-history item embeddings]).
- Item tower: MLP(item-id embedding).
- Both outputs are L2-normalised; score(u, i) = <u, v_i> / tau.
- Loss: softmax over the batch's items (in-batch negatives). Popular items appear as in-batch
  negatives far more often than under uniform sampling, which biases the softmax against them.
  logQ correction subtracts log p_j (p_j = probability item j is a batch item = its share of
  train interactions, known exactly here) from every logit, so the model estimates the full
  softmax instead of the popularity-biased one.
- Duplicate items in a batch are not used as negatives for each other.
- During training the target item is excluded from the history mean (otherwise the user tower
  could simply find the target inside its own input).
"""

from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import nn

from firstpr.data.dataset import InteractionData
from firstpr.models.base import Recommender, ValFn
from firstpr.train.trainer import train_with_early_stopping


def _mlp(dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for a, b in zip(dims[:-1], dims[1:], strict=True):
        layers += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*layers[:-1])  # no activation on the output layer


def _csr_to_torch(m: sp.csr_matrix) -> torch.Tensor:
    m = m.tocoo()
    idx = torch.from_numpy(np.vstack([m.row, m.col]).astype(np.int64))
    return torch.sparse_coo_tensor(
        idx, torch.from_numpy(m.data.astype(np.float32)), m.shape, check_invariants=False
    )


class TwoTowerModule(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int, hidden: list[int], init_std: float):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)  # shared by the item tower and user histories
        self.user_tower = _mlp([2 * dim, *hidden])
        self.item_tower = _mlp([dim, *hidden])
        nn.init.normal_(self.user_emb.weight, std=init_std)
        nn.init.normal_(self.item_emb.weight, std=init_std)

    def users(self, u: torch.Tensor, hist_mean: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.user_emb(u), hist_mean], dim=-1)
        return F.normalize(self.user_tower(x), dim=-1)

    def items(self, i: torch.Tensor | None = None) -> torch.Tensor:
        e = self.item_emb.weight if i is None else self.item_emb(i)
        return F.normalize(self.item_tower(e), dim=-1)


class TwoTower(Recommender):
    name = "two_tower"

    def _hist_mean(self, users: np.ndarray, exclude: torch.Tensor | None = None) -> torch.Tensor:
        """Mean train-history item embedding per user; optionally minus one item (the target)."""
        rows = _csr_to_torch(self.train_[users])
        total = torch.sparse.mm(rows, self.module.item_emb.weight)
        n = torch.from_numpy(self.hist_len_[users]).float()[:, None]
        if exclude is not None:
            total = total - self.module.item_emb(exclude)
            n = n - 1
        return total / n.clamp(min=1)

    def fit(
        self, data: InteractionData, config: dict[str, Any], val_fn: ValFn | None = None
    ) -> dict[str, Any]:
        seed = int(config.get("seed", 0))
        rng = np.random.default_rng(seed)
        self.train_ = data.train.tocsr()
        self.hist_len_ = np.diff(self.train_.indptr).astype(np.int64)
        dim, hidden = int(config["dim"]), [int(h) for h in config["hidden"]]
        self.module = TwoTowerModule(
            data.n_users, data.n_items, dim, hidden, float(config["init_std"])
        )
        opt = torch.optim.Adam(
            self.module.parameters(),
            lr=float(config["lr"]),
            weight_decay=float(config["weight_decay"]),
        )
        tau, batch_size = float(config["temperature"]), int(config["batch_size"])
        log_q = torch.from_numpy(
            np.log(data.item_popularity / data.item_popularity.sum() + 1e-12)
        ).float()
        use_logq = bool(config["logq"])

        coo = self.train_.tocoo()
        pairs_u, pairs_i = coo.row.astype(np.int64), coo.col.astype(np.int64)

        def run_epoch(epoch: int) -> float:
            self.module.train()
            order = rng.permutation(len(pairs_u))
            total, n = 0.0, 0
            for start in range(0, len(order), batch_size):
                b = order[start : start + batch_size]
                u_np, i_np = pairs_u[b], pairs_i[b]
                u, i = torch.from_numpy(u_np), torch.from_numpy(i_np)
                user_vec = self.module.users(u, self._hist_mean(u_np, exclude=i))
                item_vec = self.module.items(i)
                logits = user_vec @ item_vec.T / tau  # [B, B], diagonal = positives
                if use_logq:
                    logits = logits - log_q[i][None, :]
                same = i[:, None] == i[None, :]
                same.fill_diagonal_(False)
                logits = logits.masked_fill(same, float("-inf"))
                loss = F.cross_entropy(logits, torch.arange(len(b)))
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(b)
                n += len(b)
            self.module.eval()
            return total / n

        return train_with_early_stopping(
            self.module,
            run_epoch,
            (lambda: val_fn(self)) if val_fn is not None else None,
            max_epochs=int(config["max_epochs"]),
            patience=int(config["patience"]),
            eval_every=int(config.get("eval_every", 1)),
        )

    @torch.no_grad()
    def score(self, user_ids: np.ndarray) -> np.ndarray:
        u = torch.as_tensor(user_ids)
        user_vec = self.module.users(u, self._hist_mean(np.asarray(user_ids)))
        return (user_vec @ self.module.items().T).numpy()
