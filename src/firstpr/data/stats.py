"""Dataset statistics saved next to the processed split (stats.json)."""

from typing import Any

import numpy as np
import pandas as pd

from firstpr.data.dataset import InteractionData


def compute_stats(
    data: InteractionData, n_short_users: int, ties: dict[str, Any] | None = None
) -> dict[str, Any]:
    n_inter = data.train.nnz + data.val.nnz + data.test.nnz
    per_user = np.asarray((data.train + data.val + data.test).sum(axis=1)).ravel()
    head_share = float(data.item_popularity[data.head_mask].sum() / data.item_popularity.sum())
    return {
        "name": data.name,
        "n_users": data.n_users,
        "n_items": data.n_items,
        "n_interactions": int(n_inter),
        "density": n_inter / (data.n_users * data.n_items),
        "interactions_per_user": {
            "min": int(per_user.min()),
            "median": float(np.median(per_user)),
            "max": int(per_user.max()),
        },
        "split_counts": {
            "train": int(data.train.nnz),
            "val": int(data.val.nnz),
            "test": int(data.test.nnz),
        },
        "users_with_val": int((data.val.getnnz(axis=1) > 0).sum()),
        "users_with_test": int((data.test.getnnz(axis=1) > 0).sum()),
        "n_short_users": n_short_users,
        "n_head_items": int(data.head_mask.sum()),
        "n_tail_items": int(data.tail_mask.sum()),
        "head_share_of_train_interactions": head_share,
        "items_without_train_interactions": int((data.item_popularity == 0).sum()),
        "timestamp_ties": ties or {},
    }


def tie_stats(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    """How often a user's interactions share a timestamp, and how often a split boundary
    falls inside such a tied group (where the tie-break decides which split an item lands in)."""
    df = pd.concat([train, val, test])
    tied = df.groupby(["user", "timestamp"])["item"].transform("size") > 1
    per_user = tied.groupby(df["user"]).mean()
    ts = "timestamp"
    tv = train.groupby("user")[ts].max() == val.groupby("user")[ts].min()
    vt = val.groupby("user")[ts].max() == test.groupby("user")[ts].min()
    return {
        "interaction_share_tied": float(tied.mean()),
        "per_user_tied_share_mean": float(per_user.mean()),
        "per_user_tied_share_median": float(per_user.median()),
        "users_with_any_tie_share": float((per_user > 0).mean()),
        "users_train_val_boundary_in_tie": int(tv.sum()),
        "users_val_test_boundary_in_tie": int(vt.sum()),
        "users_any_boundary_in_tie": int((tv | vt).sum()),
    }
