"""Text-based SASRec: item representations come from frozen text embeddings through an adapter.

Item vector = adapter(text_emb(item metadata))           (UniSRec "inductive": text only)
            [+ ID embedding]                             (UniSRec "transductive": text + ID)
            [+ kar_adapter(text_emb(LLM item knowledge))] (KAR item-side augmentation)

The same item matrix is used for the input sequence and the output softmax, as in SASRec. Items
with no train interactions (cold items) still get a meaningful vector from their text, which an
ID-only model cannot do.

KAR user-side augmentation (Xi et al. 2024, "Towards Open-World Recommendation with Knowledge
Augmentation from Large Language Models"): an LLM summarises the user's preferences from their
history; its embedding goes through a gated-expert adapter and is added to the final hidden
state. To avoid leaking the training target, the preference is generated from the train history
*without its last item*; training applies the user term only at the last position, whose target
(the last train item) the LLM never saw. At scoring time the same preference is used.

`cold_negatives: false` restricts the training softmax to items with train interactions. With
the full catalog, items that are never a training target are pushed down as negatives at every
step, which a real system cannot do to an item that arrives after training (ZESRec / UniSRec's
inductive setting scores new items that training never saw). Scoring always covers every item.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from firstpr.data.dataset import InteractionData
from firstpr.models.base import Recommender, ValFn
from firstpr.models.sasrec import PAD, SASRecModule, left_pad
from firstpr.text.adapters import MoEAdapter, build_adapter
from firstpr.text.encoder import DEFAULT_ENCODER, cached_embeddings
from firstpr.train.trainer import train_with_early_stopping


def side_texts(data: InteractionData, file: str, key: str, n: int) -> list[str]:
    """Texts per id from `<processed_dir>/<file>` (columns key, text), '' where missing."""
    if data.processed_dir is None:
        raise ValueError("text models need data loaded with InteractionData.from_processed")
    df = pd.read_parquet(Path(data.processed_dir) / file)
    out = [""] * n
    for k, t in zip(df[key], df["text"], strict=True):
        out[int(k)] = t
    return out


def text_matrix(data: InteractionData, file: str, key: str, n: int, encoder: str) -> np.ndarray:
    texts = side_texts(data, file, key, n)
    cache = Path(data.processed_dir) / "embeddings" / Path(file).stem  # type: ignore[arg-type]
    return cached_embeddings(texts, cache, encoder)


class TextItemTable(nn.Module):
    """Stands in for SASRec's nn.Embedding: `table(idx)`, `.weight`, `.embedding_dim`."""

    def __init__(
        self,
        text: np.ndarray,
        hidden: int,
        adapter: str,
        dropout: float,
        use_id: bool,
        knowledge: np.ndarray | None,
    ) -> None:
        super().__init__()
        pad = np.zeros((1, text.shape[1]), dtype=np.float32)
        self.register_buffer("text", torch.from_numpy(np.vstack([pad, text])))
        self.adapter = build_adapter(adapter, text.shape[1], hidden, dropout)
        self.id_emb = nn.Embedding(len(text) + 1, hidden, padding_idx=PAD) if use_id else None
        if self.id_emb is not None:
            nn.init.normal_(self.id_emb.weight, std=0.02)
        self.has_knowledge = knowledge is not None
        if knowledge is not None:
            self.register_buffer("knowledge", torch.from_numpy(np.vstack([pad, knowledge])))
            self.kar = MoEAdapter(knowledge.shape[1], hidden, dropout)
        self.embedding_dim = hidden
        mask = torch.ones(len(text) + 1, 1)
        mask[PAD] = 0
        self.register_buffer("pad_mask", mask)

    @property
    def weight(self) -> torch.Tensor:
        m = self.adapter(self.text)
        if self.id_emb is not None:
            m = m + self.id_emb.weight
        if self.has_knowledge:
            m = m + self.kar(self.knowledge)
        return m * self.pad_mask

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.weight[idx]


class TextSASRec(Recommender):
    name = "sasrec_text"

    def fit(
        self, data: InteractionData, config: dict[str, Any], val_fn: ValFn | None = None
    ) -> dict[str, Any]:
        seed = int(config.get("seed", 0))
        rng = np.random.default_rng(seed)
        encoder = config.get("text_encoder", DEFAULT_ENCODER)
        hidden, dropout = int(config["hidden"]), float(config["dropout"])
        self.max_len = int(config["max_len"])

        text = text_matrix(data, "items.parquet", "item", data.n_items, encoder)
        knowledge = None
        if config.get("item_knowledge", False):
            knowledge = text_matrix(data, "kar_items.parquet", "item", data.n_items, encoder)
        self.module = SASRecModule(
            data.n_items, self.max_len, hidden, int(config["blocks"]), int(config["heads"]), dropout
        )
        self.module.item_emb = TextItemTable(
            text, hidden, config["adapter"], dropout, bool(config.get("use_id", False)), knowledge
        )
        self.user_vec = None
        if config.get("user_preference", False):
            pref = text_matrix(data, "kar_users.parquet", "user", data.n_users, encoder)
            self.user_text = torch.from_numpy(pref)
            self.user_adapter = MoEAdapter(pref.shape[1], hidden, dropout)
            self.module.add_module("user_adapter", self.user_adapter)
            self.user_vec = lambda users: self.user_adapter(self.user_text[users])

        hist = data.train_histories
        self.context_ = left_pad(hist, self.max_len)
        seqs = left_pad(hist, self.max_len + 1)
        inputs, targets = seqs[:, :-1], seqs[:, 1:]
        users = np.arange(data.n_users)
        keep = (targets != PAD).any(axis=1)
        inputs, targets, users = inputs[keep], targets[keep], users[keep]

        softmax_items = torch.arange(data.n_items)
        if not config.get("cold_negatives", True):
            softmax_items = torch.from_numpy(np.flatnonzero(data.item_popularity > 0))
        to_softmax = torch.full((data.n_items,), -1, dtype=torch.long)
        to_softmax[softmax_items] = torch.arange(len(softmax_items))

        opt = torch.optim.Adam(self.module.parameters(), lr=float(config["lr"]), betas=(0.9, 0.98))
        batch_size = int(config["batch_size"])

        def run_epoch(epoch: int) -> float:
            self.module.train()
            order = rng.permutation(len(inputs))
            total, n = 0.0, 0
            for start in range(0, len(order), batch_size):
                b = order[start : start + batch_size]
                x, y = torch.from_numpy(inputs[b]), torch.from_numpy(targets[b])
                h = self.module(x)
                if self.user_vec is not None:  # user term at the last position only
                    last = torch.zeros_like(h)
                    last[:, -1, :] = self.user_vec(torch.from_numpy(users[b]))
                    h = h + last
                valid = y != PAD
                items = self.module.item_emb.weight[1:][softmax_items]
                loss = F.cross_entropy(h[valid] @ items.T, to_softmax[y[valid] - 1])
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * int(valid.sum())
                n += int(valid.sum())
            self.module.eval()
            self._items = None
            return total / n

        self._items = None
        info = train_with_early_stopping(
            self.module,
            run_epoch,
            (lambda: val_fn(self)) if val_fn is not None else None,
            max_epochs=int(config["max_epochs"]),
            patience=int(config["patience"]),
            eval_every=int(config.get("eval_every", 1)),
        )
        self._items = None
        return info

    @torch.no_grad()
    def score(self, user_ids: np.ndarray) -> np.ndarray:
        if self._items is None:  # item matrix is fixed between training epochs
            self._items = self.module.item_emb.weight[1:]
        out = []
        for start in range(0, len(user_ids), 256):
            u = user_ids[start : start + 256]
            h = self.module(torch.from_numpy(self.context_[u]))[:, -1, :]
            if self.user_vec is not None:
                h = h + self.user_vec(torch.from_numpy(u))
            out.append(h @ self._items.T)
        return torch.cat(out).numpy()
