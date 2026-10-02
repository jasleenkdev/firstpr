"""Item-based kNN collaborative filtering (Sarwar et al., WWW 2001; Deshpande & Karypis, TOIS 2004).

sim(i, j) = |U_i ∩ U_j| / (sqrt(|U_i|) * sqrt(|U_j|) + shrink)   (cosine on binary vectors with a
shrinkage term that pulls similarities computed from few co-occurrences toward 0).
For each target item j only its k most similar items are kept, and
score(u, j) = sum of sim(i, j) over the user's train items i that are among j's neighbours.
"""

from typing import Any

import numpy as np
import scipy.sparse as sp

from firstpr.data.dataset import InteractionData
from firstpr.models.base import Recommender, ValFn


def topk_per_row(dense_rows: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Column indices and values of the k largest entries per row (k clipped to n_cols)."""
    k = min(k, dense_rows.shape[1])
    idx = np.argpartition(-dense_rows, k - 1, axis=1)[:, :k]
    return idx, np.take_along_axis(dense_rows, idx, axis=1)


def cosine_shrink_knn(
    x: sp.csr_matrix, k: int, shrink: float, block_size: int = 1024
) -> sp.csr_matrix:
    """Item-item similarity [n_items, n_items]; row j holds the top-k neighbours of item j."""
    x = x.astype(np.float32).tocsc()
    co = (x.T @ x).tocsr()  # co-occurrence counts |U_i ∩ U_j|
    norms = np.sqrt(np.asarray(x.sum(axis=0)).ravel())
    n = co.shape[0]

    rows, cols, vals = [], [], []
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        block = co[start:stop].toarray()
        block /= np.outer(norms[start:stop], norms) + shrink + 1e-12
        block[np.arange(stop - start), np.arange(start, stop)] = 0.0  # no self-similarity
        idx, v = topk_per_row(block, k)
        keep = v > 0
        rows.append(np.repeat(np.arange(start, stop), idx.shape[1])[keep.ravel()])
        cols.append(idx[keep])
        vals.append(v[keep])
    return sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)
    )


class ItemKNN(Recommender):
    name = "itemknn"

    def fit(
        self, data: InteractionData, config: dict[str, Any], val_fn: ValFn | None = None
    ) -> dict[str, Any]:
        self.x_ = data.train.astype(np.float32).tocsr()
        neighbours = cosine_shrink_knn(self.x_, k=int(config["k"]), shrink=float(config["shrink"]))
        self.w_ = neighbours.T.tocsr()  # column j = neighbours of target j -> score = X_u @ W
        return {"nnz_similarity": int(self.w_.nnz)}

    def set_context(self, matrix: sp.csr_matrix) -> None:
        """Score from a different interaction matrix without refitting (fold-in ablation)."""
        self.x_ = matrix.astype(np.float32).tocsr()

    def score(self, user_ids: np.ndarray) -> np.ndarray:
        return (self.x_[user_ids] @ self.w_).toarray()
