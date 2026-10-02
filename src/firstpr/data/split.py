"""Per-user chronological train / val / test split."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from firstpr.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    n_short_users: int  # users with too few interactions to fill train, val and test


def _round_half_up(x: np.ndarray) -> np.ndarray:
    return np.floor(x + 0.5).astype(np.int64)


def split_sizes(n: np.ndarray, val_ratio: float, test_ratio: float) -> tuple[np.ndarray, ...]:
    """Per-user (n_train, n_val, n_test).

    Users with n >= 3 get at least one val and one test item. Users with n < 3 keep one train
    item first, then fill test, then val.
    """
    n = np.asarray(n, dtype=np.int64)
    n_test = np.maximum(1, _round_half_up(n * test_ratio))
    n_val = np.maximum(1, _round_half_up(n * val_ratio))
    n_train = n - n_test - n_val

    short = n < 3
    n_train = np.where(short, np.minimum(n, 1), n_train)
    n_test = np.where(short, np.minimum(n - n_train, 1), n_test)
    n_val = np.where(short, n - n_train - n_test, n_val)
    return n_train, n_val, n_test


def chronological_order(df: pd.DataFrame, tie_break_seed: int) -> pd.DataFrame:
    """Sort interactions by user, then time; break timestamp ties in a seeded random order.

    Rows are first put in a canonical order (user, timestamp, item) so the random keys do not
    depend on the input row order; the result is identical for a fixed seed. A `seq` column
    holds each interaction's position in its user's chronological history.
    """
    df = df.sort_values(["user", "timestamp", "item"], kind="stable").reset_index(drop=True)
    rng = np.random.default_rng(tie_break_seed)
    df["_tie"] = rng.permutation(len(df))
    df = df.sort_values(["user", "timestamp", "_tie"], kind="stable").drop(columns="_tie")
    df = df.reset_index(drop=True)
    df["seq"] = df.groupby("user").cumcount().astype(np.int64)
    return df


def chronological_split(
    df: pd.DataFrame, val_ratio: float, test_ratio: float, tie_break_seed: int
) -> SplitResult:
    """Split each user's interactions by time: oldest -> train, then val, newest -> test.

    Ties on timestamp are broken in a seeded random order (see `chronological_order`).
    """
    df = chronological_order(df, tie_break_seed)
    pos = df["seq"].to_numpy()
    n = df.groupby("user")["item"].transform("size").to_numpy()
    n_train, n_val, _ = split_sizes(n, val_ratio, test_ratio)

    part = np.where(pos < n_train, 0, np.where(pos < n_train + n_val, 1, 2))
    train, val, test = (df.loc[part == p].reset_index(drop=True) for p in range(3))

    n_short = int((df.groupby("user").size() < 3).sum())
    log.info(
        "split: train=%d val=%d test=%d, users too short for all three splits=%d",
        len(train), len(val), len(test), n_short,
    )
    return SplitResult(train, val, test, n_short)
