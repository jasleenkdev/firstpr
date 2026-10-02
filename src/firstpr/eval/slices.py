"""Item slices: head (top 20% by train popularity) vs long tail.

A slice metric restricts the targets to items in the slice: e.g. head Recall@K is the share of a
user's head targets found in the top-K. Users with no targets in the slice are skipped.
"""

import numpy as np

from firstpr.eval.metrics import hits_matrix, ndcg_at_k, recall_at_k


def slice_metrics(
    topk: np.ndarray, targets: np.ndarray, item_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-user recall and ndcg for targets inside `item_mask`; returns (recall, ndcg, keep)."""
    sliced = targets & item_mask[None, :]
    n = sliced.sum(axis=1)
    keep = n > 0
    hits = hits_matrix(topk[keep], sliced[keep])
    return recall_at_k(hits, n[keep]), ndcg_at_k(hits, n[keep]), keep
