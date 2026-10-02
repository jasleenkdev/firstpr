"""Negative samplers for pairwise (BPR-style) training."""

from collections.abc import Iterator

import numpy as np
import scipy.sparse as sp


class UniformNegativeSampler:
    """One epoch = every train (u, i) pair once, shuffled, each with one negative j.

    j is drawn uniformly from all items and redrawn (vectorised) while (u, j) is a train positive,
    following the uniform sampling of BPR (Rendle et al., UAI 2009).
    """

    def __init__(self, train: sp.csr_matrix, seed: int) -> None:
        coo = train.tocoo()
        self.users = coo.row.astype(np.int64)
        self.items = coo.col.astype(np.int64)
        self.n_items = train.shape[1]
        self._pos_keys = np.sort(self.users * self.n_items + self.items)
        self.rng = np.random.default_rng(seed)

    def is_positive(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        keys = users * self.n_items + items
        idx = np.searchsorted(self._pos_keys, keys)
        idx = np.minimum(idx, len(self._pos_keys) - 1)
        return self._pos_keys[idx] == keys

    def sample_negatives(self, users: np.ndarray) -> np.ndarray:
        neg = self.rng.integers(0, self.n_items, size=len(users))
        bad = self.is_positive(users, neg)
        while bad.any():
            neg[bad] = self.rng.integers(0, self.n_items, size=int(bad.sum()))
            bad[bad] = self.is_positive(users[bad], neg[bad])
        return neg

    def __len__(self) -> int:
        return len(self.users)

    def epoch(self, batch_size: int) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        order = self.rng.permutation(len(self.users))
        for start in range(0, len(order), batch_size):
            b = order[start : start + batch_size]
            u = self.users[b]
            yield u, self.items[b], self.sample_negatives(u)
