"""The single evaluator: full ranking over all items, seen items masked, top-K metrics.

Every model in the benchmark is scored here; models never compute their own metrics.
- mode="val": targets = val items, mask = train items.
- mode="test": targets = test items, mask = train + val items.
"""

from typing import Any, Literal

import numpy as np
import scipy.sparse as sp

from firstpr.data.dataset import InteractionData
from firstpr.eval import metrics as M
from firstpr.eval.slices import slice_metrics
from firstpr.models.base import Recommender

Mode = Literal["val", "test"]


class Evaluator:
    def __init__(self, data: InteractionData, k: int = 20, batch_size: int = 1024) -> None:
        self.data = data
        self.k = k
        self.batch_size = batch_size

    def _targets_and_mask(self, mode: Mode) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        if mode == "val":
            return self.data.val, self.data.train
        if mode == "test":
            return self.data.test, (self.data.train + self.data.val).tocsr()
        raise ValueError(f"unknown mode {mode!r}")

    def users_to_evaluate(self, mode: Mode) -> np.ndarray:
        targets, _ = self._targets_and_mask(mode)
        return np.flatnonzero(targets.getnnz(axis=1) > 0)

    def topk(self, model: Recommender, users: np.ndarray, mask: sp.csr_matrix) -> np.ndarray:
        """Masked full-ranking top-K item ids [len(users), K]; -1 where fewer than K items."""
        scores = np.array(model.score(users), dtype=np.float32, copy=True)
        if scores.shape != (len(users), self.data.n_items):
            raise ValueError(
                f"score() returned {scores.shape}, expected {(len(users), self.data.n_items)}"
            )
        m = mask[users].tocoo()
        scores[m.row, m.col] = -np.inf

        k = min(self.k, scores.shape[1])
        part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        part_scores = np.take_along_axis(scores, part, axis=1)
        order = np.argsort(-part_scores, axis=1, kind="stable")
        top = np.take_along_axis(part, order, axis=1)
        top_scores = np.take_along_axis(part_scores, order, axis=1)
        top[~np.isfinite(top_scores)] = -1  # masked items never count as recommendations
        if k < self.k:
            top = np.pad(top, ((0, 0), (0, self.k - k)), constant_values=-1)
        return top

    def evaluate(
        self, model: Recommender, mode: Mode = "val", per_user: bool = False
    ) -> dict[str, Any]:
        """Overall + slice metrics. With per_user=True the result also holds
        `per_user = {"users", "recall@K", "ndcg@K"}` arrays (for paired bootstrap tests)."""
        targets_m, mask = self._targets_and_mask(mode)
        users = self.users_to_evaluate(mode)
        k = self.k
        head = self.data.head_mask
        # cold items (no train interactions) only exist on some datasets (Amazon, not ML-1M)
        cold = self.data.item_popularity == 0
        slice_masks = [("head", head), ("tail", ~head)] + ([("cold", cold)] if cold.any() else [])

        recall, ndcg, hit, tops = [], [], [], []
        sl: dict[str, tuple[list, list]] = {name: ([], []) for name, _ in slice_masks}
        cold_ndcg = []
        for start in range(0, len(users), self.batch_size):
            batch = users[start : start + self.batch_size]
            top = self.topk(model, batch, mask)
            targets = targets_m[batch].toarray().astype(bool)
            n_targets = targets.sum(axis=1)
            hits = M.hits_matrix(top, targets)
            recall.append(M.recall_at_k(hits, n_targets))
            ndcg.append(M.ndcg_at_k(hits, n_targets))
            hit.append(M.hit_rate_at_k(hits))
            tops.append(top)
            for name, item_mask in slice_masks:
                r, n, keep = slice_metrics(top, targets, item_mask)
                sl[name][0].append(r)
                sl[name][1].append(n)
                if name == "cold":  # per user, NaN where the user has no cold target
                    full = np.full(len(batch), np.nan)
                    full[keep] = n
                    cold_ndcg.append(full)

        top_all = np.concatenate(tops)
        overall = {
            f"recall@{k}": float(np.concatenate(recall).mean()),
            f"ndcg@{k}": float(np.concatenate(ndcg).mean()),
            f"hit@{k}": float(np.concatenate(hit).mean()),
            f"coverage@{k}": M.catalog_coverage(top_all, self.data.n_items),
            f"avg_pop@{k}": M.average_popularity(top_all, self.data.item_popularity),
            f"long_tail_share@{k}": M.long_tail_share(top_all, self.data.tail_mask),
        }
        slices = {}
        for name, (r, n) in sl.items():
            r, n = np.concatenate(r), np.concatenate(n)
            slices[name] = {
                f"recall@{k}": float(r.mean()) if len(r) else float("nan"),
                f"ndcg@{k}": float(n.mean()) if len(n) else float("nan"),
                "n_users": int(len(r)),
            }
        result = {"mode": mode, "n_users": int(len(users)), "overall": overall, "slices": slices}
        if per_user:
            result["per_user"] = {
                "users": users,
                f"recall@{k}": np.concatenate(recall),
                f"ndcg@{k}": np.concatenate(ndcg),
            }
            if cold_ndcg:
                result["per_user"][f"cold_ndcg@{k}"] = np.concatenate(cold_ndcg)
        return result
