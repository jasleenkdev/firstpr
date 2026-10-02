"""SASRec: self-attentive sequential recommendation (Kang & McAuley, ICDM 2018).

The user's train history (oldest -> newest, `seq` order) is a sequence; a causal Transformer
encoder predicts the next item at every position. score(u, ·) = h_last · E_items^T, where h_last
is the hidden state after the user's most recent train item and E_items is the shared item
embedding table.

Loss: full-catalog cross-entropy at every position by default. The original paper uses binary
cross-entropy with one sampled negative per position ("bce" here); later work (Klenitskiy &
Vasilev, RecSys 2023; Petrov & Macdonald, RecSys 2023) shows the full softmax is much stronger.

`shuffle_history: true` randomly permutes every user's train history (seeded) for training and
scoring: an ablation that measures how much item order matters (66% of ML-1M interactions are
timestamp-tied, so part of the "order" is already random).
"""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from firstpr.data.dataset import InteractionData
from firstpr.models.base import Recommender, ValFn
from firstpr.train.trainer import train_with_early_stopping

PAD = 0  # item ids are shifted by +1 so that 0 can be padding


def left_pad(histories: list[np.ndarray], max_len: int) -> np.ndarray:
    """[n, max_len] int64 with the last `max_len` items (shifted +1), left-padded with 0."""
    out = np.zeros((len(histories), max_len), dtype=np.int64)
    for r, h in enumerate(histories):
        h = h[-max_len:]
        if len(h):
            out[r, -len(h) :] = h + 1
    return out


class SASRecModule(nn.Module):
    def __init__(
        self, n_items: int, max_len: int, hidden: int, blocks: int, heads: int, dropout: float
    ):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, hidden, padding_idx=PAD)
        self.pos_emb = nn.Embedding(max_len, hidden)
        self.drop = nn.Dropout(dropout)
        self.attn_norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(blocks))
        self.attns = nn.ModuleList(
            nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
            for _ in range(blocks)
        )
        self.ffn_norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(blocks))
        self.ffns = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.Dropout(dropout),
            )
            for _ in range(blocks)
        )
        self.out_norm = nn.LayerNorm(hidden)
        self.heads = heads
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[PAD].zero_()

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """seq [B, L] (0 = pad) -> hidden states [B, L, H]."""
        b, length = seq.shape
        pad = seq == PAD
        x = self.item_emb(seq) * (self.item_emb.embedding_dim**0.5)
        x = x + self.pos_emb(torch.arange(length))[None]
        x = self.drop(x) * (~pad)[..., None]
        # mask[q, k] = True means "may not attend": future keys, and padding keys -- but every
        # query may attend to itself, so left-padded rows never get an all-masked softmax (NaN)
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
        mask = causal[None] | pad[:, None, :]
        mask = mask & ~torch.eye(length, dtype=torch.bool)[None]
        mask = mask.repeat_interleave(self.heads, dim=0)
        for norm, attn, fnorm, ffn in zip(
            self.attn_norms, self.attns, self.ffn_norms, self.ffns, strict=True
        ):
            q = norm(x)
            a, _ = attn(q, q, q, attn_mask=mask, need_weights=False)
            x = x + a
            x = x + ffn(fnorm(x))
            x = x * (~pad)[..., None]
        return self.out_norm(x)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        """Hidden states -> scores for every real item (padding column dropped)."""
        return h @ self.item_emb.weight[1:].T


class SASRec(Recommender):
    name = "sasrec"

    def _histories(self, data: InteractionData, config: dict[str, Any]) -> list[np.ndarray]:
        hist = data.train_histories
        if config.get("shuffle_history", False):
            rng = np.random.default_rng(int(config.get("seed", 0)) + 10_000)
            hist = [rng.permutation(h) for h in hist]
        return hist

    def fit(
        self, data: InteractionData, config: dict[str, Any], val_fn: ValFn | None = None
    ) -> dict[str, Any]:
        seed = int(config.get("seed", 0))
        rng = np.random.default_rng(seed)
        self.max_len = int(config["max_len"])
        self.n_items = data.n_items
        hist = self._histories(data, config)
        self.context_ = left_pad(hist, self.max_len)  # scoring input: last max_len train items
        # training: input = s_1..s_{n-1}, target = s_2..s_n (last max_len + 1 items)
        seqs = left_pad(hist, self.max_len + 1)
        inputs, targets = seqs[:, :-1], seqs[:, 1:]
        keep = (targets != PAD).any(axis=1)
        inputs, targets = inputs[keep], targets[keep]

        self.module = SASRecModule(
            data.n_items,
            self.max_len,
            int(config["hidden"]),
            int(config["blocks"]),
            int(config["heads"]),
            float(config["dropout"]),
        )
        opt = torch.optim.Adam(self.module.parameters(), lr=float(config["lr"]), betas=(0.9, 0.98))
        batch_size, loss_type = int(config["batch_size"]), config["loss"]
        train_sets = data.train.tocsr()

        def run_epoch(epoch: int) -> float:
            self.module.train()
            order = rng.permutation(len(inputs))
            total, n = 0.0, 0
            for start in range(0, len(order), batch_size):
                b = order[start : start + batch_size]
                x, y = torch.from_numpy(inputs[b]), torch.from_numpy(targets[b])
                h = self.module(x)
                valid = y != PAD
                if loss_type == "ce":
                    logits = self.module.logits(h[valid])
                    loss = F.cross_entropy(logits, y[valid] - 1)
                elif loss_type == "bce":  # original paper: one uniform negative per position
                    neg = torch.from_numpy(rng.integers(1, self.n_items + 1, size=y.shape))
                    pos_s = (h * self.module.item_emb(y)).sum(-1)[valid]
                    neg_s = (h * self.module.item_emb(neg)).sum(-1)[valid]
                    loss = F.binary_cross_entropy_with_logits(
                        pos_s, torch.ones_like(pos_s)
                    ) + F.binary_cross_entropy_with_logits(neg_s, torch.zeros_like(neg_s))
                else:
                    raise ValueError(loss_type)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * int(valid.sum())
                n += int(valid.sum())
            self.module.eval()
            return total / n

        del train_sets
        return train_with_early_stopping(
            self.module,
            run_epoch,
            (lambda: val_fn(self)) if val_fn is not None else None,
            max_epochs=int(config["max_epochs"]),
            patience=int(config["patience"]),
            eval_every=int(config.get("eval_every", 1)),
        )

    def set_context(self, histories: list[np.ndarray]) -> None:
        """Replace the scoring input sequences without retraining (fold-in ablation)."""
        self.context_ = left_pad(histories, self.max_len)

    @torch.no_grad()
    def score(self, user_ids: np.ndarray) -> np.ndarray:
        out = []
        for start in range(0, len(user_ids), 256):
            seq = torch.from_numpy(self.context_[user_ids[start : start + 256]])
            out.append(self.module.logits(self.module(seq)[:, -1, :]))
        return torch.cat(out).numpy()
