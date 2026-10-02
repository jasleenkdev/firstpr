"""Neural Collaborative Filtering (He et al., WWW 2017): GMF, MLP and NeuMF.

- GMF:   y = h^T (p_u ⊙ q_i)                       (a weighted dot product)
- MLP:   y = h^T MLP([p'_u ; q'_i])                (a learned similarity function)
- NeuMF: y = h^T [p_u ⊙ q_i ; MLP([p'_u ; q'_i])]  (separate embeddings for the two branches)

Trained pointwise with binary cross-entropy on each positive plus `n_neg` uniform negatives, as in
the paper. NeuMF can be initialised from separately trained GMF and MLP models (alpha = 0.5).
Rendle et al. (RecSys 2020) argue that a well-tuned dot product (MF) beats the learned MLP
similarity; phase 2 tests that on our split.

Full-catalog scoring for the MLP uses the identity W1 [p ; q] = W1_u p + W1_i q: the first layer
is precomputed per user and per item, so scoring is [users, items, d1] adds, not a giant MLP batch.
"""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from firstpr.data.dataset import InteractionData
from firstpr.models.base import Recommender, ValFn
from firstpr.train.samplers import PointwiseSampler
from firstpr.train.trainer import train_with_early_stopping
from firstpr.utils.logging import get_logger

log = get_logger(__name__)


class NCFModule(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        variant: str,
        factors: int,
        mlp_layers: list[int],
        init_std: float,
    ) -> None:
        super().__init__()
        if variant not in ("gmf", "mlp", "neumf"):
            raise ValueError(variant)
        self.variant = variant
        self.use_gmf = variant in ("gmf", "neumf")
        self.use_mlp = variant in ("mlp", "neumf")
        out_dim = 0
        if self.use_gmf:
            self.gmf_user = nn.Embedding(n_users, factors)
            self.gmf_item = nn.Embedding(n_items, factors)
            out_dim += factors
        if self.use_mlp:
            # embedding size = first layer width / 2 (paper: last layer = predictive factors)
            emb = mlp_layers[0] // 2
            self.mlp_user = nn.Embedding(n_users, emb)
            self.mlp_item = nn.Embedding(n_items, emb)
            self.layers = nn.ModuleList(
                nn.Linear(a, b) for a, b in zip(mlp_layers[:-1], mlp_layers[1:], strict=True)
            )
            out_dim += mlp_layers[-1]
        self.h = nn.Linear(out_dim, 1, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=init_std)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def embeddings(self, u: torch.Tensor, i: torch.Tensor) -> list[torch.Tensor]:
        out = []
        if self.use_gmf:
            out += [self.gmf_user(u), self.gmf_item(i)]
        if self.use_mlp:
            out += [self.mlp_user(u), self.mlp_item(i)]
        return out

    def _mlp_tail(self, x: torch.Tensor) -> torch.Tensor:
        """Layers after the first linear layer, applied to the first layer's pre-activation."""
        x = F.relu(x)
        for layer in self.layers[1:]:
            x = F.relu(layer(x))
        return x

    def forward(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        parts = []
        if self.use_gmf:
            parts.append(self.gmf_user(u) * self.gmf_item(i))
        if self.use_mlp:
            x = torch.cat([self.mlp_user(u), self.mlp_item(i)], dim=-1)
            parts.append(self._mlp_tail(self.layers[0](x)))
        return self.h(torch.cat(parts, dim=-1)).squeeze(-1)

    @torch.no_grad()
    def score_all(self, users: torch.Tensor, chunk: int = 64) -> torch.Tensor:
        """Logits for every item, [len(users), n_items]."""
        out = []
        for start in range(0, len(users), chunk):
            u = users[start : start + chunk]
            parts = []
            if self.use_gmf:
                parts.append(self.gmf_user(u)[:, None, :] * self.gmf_item.weight[None, :, :])
            if self.use_mlp:
                first = self.layers[0]
                emb = self.mlp_user.weight.shape[1]
                w_u, w_i = first.weight[:, :emb], first.weight[:, emb:]
                a = self.mlp_user(u) @ w_u.T + first.bias  # [B, d1]
                b = self.mlp_item.weight @ w_i.T  # [n_items, d1]
                parts.append(self._mlp_tail(a[:, None, :] + b[None, :, :]))
            out.append(self.h(torch.cat(parts, dim=-1)).squeeze(-1))
        return torch.cat(out)


class NCF(Recommender):
    """`variant` in {gmf, mlp, neumf}; registered as ncf_gmf / ncf_mlp / ncf_neumf."""

    def __init__(self, variant: str) -> None:
        self.variant = variant
        self.name = f"ncf_{variant}"

    def _new_module(self, data: InteractionData, config: dict[str, Any], variant: str) -> NCFModule:
        return NCFModule(
            data.n_users,
            data.n_items,
            variant,
            int(config["factors"]),
            [int(x) for x in config["mlp_layers"]],
            float(config["init_std"]),
        )

    def _train(
        self,
        module: NCFModule,
        data: InteractionData,
        config: dict[str, Any],
        val_fn: ValFn | None,
        lr: float,
    ) -> dict[str, Any]:
        sampler = PointwiseSampler(data.train, int(config["n_neg"]), int(config.get("seed", 0)))
        opt = torch.optim.Adam(module.parameters(), lr=lr)
        reg, batch_size = float(config["reg"]), int(config["batch_size"])
        self.module = module

        def run_epoch(epoch: int) -> float:
            module.train()
            total, n = 0.0, 0
            for u, i, y in sampler.epoch(batch_size):
                u_t, i_t, y_t = torch.from_numpy(u), torch.from_numpy(i), torch.from_numpy(y)
                logits = module(u_t, i_t)
                bce = F.binary_cross_entropy_with_logits(logits, y_t)
                l2 = sum(e.pow(2).sum() for e in module.embeddings(u_t, i_t)) / (2 * len(u))
                loss = bce + reg * l2
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += bce.item() * len(u)
                n += len(u)
            module.eval()
            return total / n

        return train_with_early_stopping(
            module,
            run_epoch,
            (lambda: val_fn(self)) if val_fn is not None else None,
            max_epochs=int(config["max_epochs"]),
            patience=int(config["patience"]),
            eval_every=int(config.get("eval_every", 1)),
        )

    def fit(
        self, data: InteractionData, config: dict[str, Any], val_fn: ValFn | None = None
    ) -> dict[str, Any]:
        lr = float(config["lr"])
        if self.variant == "neumf" and config.get("pretrain", False):
            # paper: train GMF and MLP separately, initialise NeuMF from both (alpha = 0.5)
            info: dict[str, Any] = {}
            subs = {}
            for sub in ("gmf", "mlp"):
                m = self._new_module(data, config, sub)
                info[f"pretrain_{sub}"] = self._train(m, data, config, val_fn, lr)
                subs[sub] = m
            neu = self._new_module(data, config, "neumf")
            with torch.no_grad():
                for name in ("gmf_user", "gmf_item"):
                    getattr(neu, name).weight.copy_(getattr(subs["gmf"], name).weight)
                for name in ("mlp_user", "mlp_item"):
                    getattr(neu, name).weight.copy_(getattr(subs["mlp"], name).weight)
                for a, b in zip(neu.layers, subs["mlp"].layers, strict=True):
                    a.load_state_dict(b.state_dict())
                alpha = float(config.get("alpha", 0.5))
                neu.h.weight.copy_(
                    torch.cat([alpha * subs["gmf"].h.weight, (1 - alpha) * subs["mlp"].h.weight], 1)
                )
            info["neumf"] = self._train(
                neu, data, config, val_fn, lr * float(config["finetune_lr_scale"])
            )
            final = info["neumf"]
            return {
                **{k: v for k, v in final.items()},
                "pretrain": {k: info[k] for k in info if k != "neumf"},
            }
        return self._train(self._new_module(data, config, self.variant), data, config, val_fn, lr)

    def score(self, user_ids: np.ndarray) -> np.ndarray:
        return self.module.score_all(torch.as_tensor(user_ids)).numpy()
