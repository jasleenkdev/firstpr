"""Replay models: trained only on the phase-5 train split (Mar–Aug 2025), fixed epochs, scored
for users outside the training data from their pre-PR profile.

- popularity: train interaction counts.
- LightGCN: transductive; a cohort user (mostly not in training) is folded in as the mean of the
  propagated item embeddings of their profile items (zero vector for an empty profile).
- SASRec / text-SASRec: the profile (oldest -> newest, last max_len catalog items) is the input
  sequence; no retraining.
Epochs = median best epoch of the phase-5 test runs (no early stopping on Sep / Oct data).
All scores get a 1e-6 x popularity tie-break so empty or uninformative profiles rank
deterministically.
"""

from typing import Any

import numpy as np
import torch

from firstpr.data.dataset import InteractionData
from firstpr.models.lightgcn import LightGCN
from firstpr.models.sasrec import SASRec, left_pad
from firstpr.models.sasrec_text import TextSASRec
from firstpr.utils.io import load_yaml
from firstpr.utils.seed import set_seed

FIXED_EPOCHS = {"lightgcn": 86, "sasrec": 15, "sasrec_text_mlp_warm": 24}
MODELS = ["popularity", "lightgcn", "sasrec", "sasrec_text_mlp_warm"]


def config(name: str, seed: int) -> dict[str, Any]:
    base = "sasrec_text_mlp" if name == "sasrec_text_mlp_warm" else name
    params = load_yaml(f"results/best/{base}_github.yaml")["params"]
    epochs = FIXED_EPOCHS[name]
    out = {**params, "max_epochs": epochs, "patience": epochs, "min_epochs": 0, "seed": seed}
    if name == "sasrec_text_mlp_warm":
        out["cold_negatives"] = False
    return out


class Scorer:
    def __init__(self, name: str, data: InteractionData, seed: int) -> None:
        self.name = name
        pop = data.item_popularity.astype(np.float64)
        self.tie = 1e-6 * pop / pop.max()
        self.pop = pop
        if name == "popularity":
            return
        set_seed(seed)
        torch.set_num_threads(4)
        cfg = config(name, seed)
        if name == "lightgcn":
            m = LightGCN()
            m.fit(data, cfg, val_fn=None)
            with torch.no_grad():
                self.items = m.module.propagate(m.adj_)[1].numpy()
        else:
            m = SASRec() if name == "sasrec" else TextSASRec()
            m.fit(data, cfg, val_fn=None)
            m.module.eval()
            self.max_len = int(cfg["max_len"])
            with torch.no_grad():
                self.items = m.module.item_emb.weight[1:].detach().numpy().copy()
        self.module = m.module

    @torch.no_grad()
    def scores(self, profiles: list[np.ndarray]) -> np.ndarray:
        """[len(profiles), n_items] scores; profiles = catalog item indices, oldest first."""
        n = len(profiles)
        if self.name == "popularity":
            return np.broadcast_to(self.pop + self.tie, (n, len(self.pop))).copy()
        if self.name == "lightgcn":
            u = np.stack(
                [
                    self.items[p].mean(0) if len(p) else np.zeros(self.items.shape[1])
                    for p in profiles
                ]
            )
            return u @ self.items.T + self.tie
        h = self.module(torch.from_numpy(left_pad(profiles, self.max_len)))[:, -1, :].numpy()
        return h @ self.items.T + self.tie
