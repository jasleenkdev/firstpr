"""Top-K ranking metrics with binary relevance.

Per-user metrics take `hits` [U, K] (True where the item at rank r is a target) and `n_targets`
[U]. Beyond-accuracy metrics take `topk` [U, K] item ids, where -1 marks an empty slot (fewer
than K rankable items).
"""

import numpy as np


def hits_matrix(topk: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """`targets` is a dense bool [U, n_items] row block; returns bool [U, K]."""
    valid = topk >= 0
    safe = np.where(valid, topk, 0)
    return np.take_along_axis(targets, safe, axis=1) & valid


def discounts(k: int) -> np.ndarray:
    return 1.0 / np.log2(np.arange(k) + 2.0)


def recall_at_k(hits: np.ndarray, n_targets: np.ndarray) -> np.ndarray:
    """|top-K ∩ targets| / |targets| (denominator is |targets|, not min(|targets|, K))."""
    return hits.sum(axis=1) / n_targets


def ndcg_at_k(hits: np.ndarray, n_targets: np.ndarray) -> np.ndarray:
    """DCG@K / IDCG@K with binary gains; IDCG puts min(|targets|, K) hits at the top."""
    k = hits.shape[1]
    disc = discounts(k)
    dcg = (hits * disc).sum(axis=1)
    idcg = np.cumsum(disc)[np.minimum(n_targets, k) - 1]
    return dcg / idcg


def hit_rate_at_k(hits: np.ndarray) -> np.ndarray:
    return hits.any(axis=1).astype(np.float64)


def catalog_coverage(topk: np.ndarray, n_items: int) -> float:
    """Share of the catalog that appears in at least one user's top-K."""
    return len(np.unique(topk[topk >= 0])) / n_items


def average_popularity(topk: np.ndarray, popularity: np.ndarray) -> float:
    """Mean train interaction count of recommended items (over all filled slots)."""
    items = topk[topk >= 0]
    return float(popularity[items].mean())


def long_tail_share(topk: np.ndarray, tail_mask: np.ndarray) -> float:
    """Share of recommended slots filled with tail items."""
    items = topk[topk >= 0]
    return float(tail_mask[items].mean())
