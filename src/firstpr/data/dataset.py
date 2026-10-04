"""InteractionData: the one data object every model and the evaluator share."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from firstpr.utils.io import load_json


def head_item_mask(popularity: np.ndarray, head_fraction: float) -> np.ndarray:
    """Boolean mask of head items: the top `head_fraction` of items by popularity.

    Ties are broken by item id (lower id first) so the head set is deterministic.
    """
    n_items = len(popularity)
    n_head = int(np.ceil(head_fraction * n_items))
    order = np.lexsort((np.arange(n_items), -popularity))  # popularity desc, then id asc
    mask = np.zeros(n_items, dtype=bool)
    mask[order[:n_head]] = True
    return mask


def _to_csr(df: pd.DataFrame, n_users: int, n_items: int) -> sp.csr_matrix:
    m = sp.csr_matrix(
        (np.ones(len(df), dtype=np.float32), (df["user"].to_numpy(), df["item"].to_numpy())),
        shape=(n_users, n_items),
    )
    m.sum_duplicates()
    m.data[:] = 1.0
    return m


def _ordered_histories(df: pd.DataFrame, n_users: int) -> list[np.ndarray]:
    """Per-user item arrays in chronological order (`seq` if present, else timestamp, item)."""
    if "seq" in df.columns:  # position in the user's chronological history
        ordered = df.sort_values(["user", "seq"], kind="stable")
    elif "timestamp" in df.columns:
        ordered = df.sort_values(["user", "timestamp", "item"], kind="stable")
    else:
        ordered = df.sort_values(["user"], kind="stable")
    bounds = np.searchsorted(ordered["user"].to_numpy(), np.arange(n_users + 1))
    items = ordered["item"].to_numpy()
    return [items[bounds[u] : bounds[u + 1]] for u in range(n_users)]


@dataclass
class InteractionData:
    """Train / val / test interactions as binary CSR matrices (users x items) plus helpers."""

    n_users: int
    n_items: int
    train: sp.csr_matrix
    val: sp.csr_matrix
    test: sp.csr_matrix
    train_histories: list[np.ndarray]  # per user, train items in time order (oldest first)
    val_histories: list[np.ndarray]  # per user, val items in time order (fold-in ablation)
    item_popularity: np.ndarray  # train interaction count per item
    head_mask: np.ndarray  # True for head items (top head_fraction by train popularity)
    name: str = "toy"
    stats: dict = field(default_factory=dict)
    processed_dir: str | None = None  # where side files live (e.g. item text, text embeddings)

    @property
    def tail_mask(self) -> np.ndarray:
        return ~self.head_mask

    @classmethod
    def from_frames(
        cls,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        n_users: int,
        n_items: int,
        head_fraction: float = 0.2,
        name: str = "toy",
        stats: dict | None = None,
    ) -> "InteractionData":
        train_m = _to_csr(train, n_users, n_items)
        popularity = np.asarray(train_m.sum(axis=0)).ravel().astype(np.int64)

        histories = _ordered_histories(train, n_users)

        return cls(
            n_users=n_users,
            n_items=n_items,
            train=train_m,
            val=_to_csr(val, n_users, n_items),
            test=_to_csr(test, n_users, n_items),
            train_histories=histories,
            val_histories=_ordered_histories(val, n_users),
            item_popularity=popularity,
            head_mask=head_item_mask(popularity, head_fraction),
            name=name,
            stats=stats or {},
        )

    @classmethod
    def from_processed(
        cls, processed_dir: str | Path, head_fraction: float = 0.2
    ) -> "InteractionData":
        d = Path(processed_dir)
        stats = load_json(d / "stats.json")
        data = cls.from_frames(
            pd.read_parquet(d / "train.parquet"),
            pd.read_parquet(d / "val.parquet"),
            pd.read_parquet(d / "test.parquet"),
            n_users=stats["n_users"],
            n_items=stats["n_items"],
            head_fraction=head_fraction,
            name=stats["name"],
            stats=stats,
        )
        data.processed_dir = str(d)
        return data
