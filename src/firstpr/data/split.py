"""Per-user chronological train / val / test split."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from firstpr.data.preprocess import k_core as apply_k_core
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


def global_temporal_split(
    df: pd.DataFrame,
    val_start: int,
    test_start: int,
    k_core: int,
    tie_break_seed: int,
    keep_cold_items: set[int] | None = None,
    max_users: int | None = None,
    sample_seed: int = 0,
) -> SplitResult:
    """One global cutoff for everyone (Meng et al., RecSys 2020): train = t < val_start,
    val = val_start <= t < test_start, test = t >= test_start (timestamps in seconds).

    The k-core filter runs on train pairs only (no future information decides who stays).
    Val / test keep users that survive it and items that survive it or are listed in
    `keep_cold_items` (new items with no train interactions: the cold slice). Users with no val
    or test pair simply have no targets there; pairs already in train are dropped from val/test.
    `max_users`: a seeded uniform sample of the users that survive the core (CPU budget), then
    the core is applied again so every kept user and item still has k train interactions.
    """
    df = chronological_order(df, tie_break_seed)
    t = df["timestamp"].to_numpy()
    part = np.where(t < val_start, 0, np.where(t < test_start, 1, 2))
    train = apply_k_core(df.loc[part == 0], k_core)
    if max_users is not None and train["user"].nunique() > max_users:
        pool = np.sort(train["user"].unique())
        keep = np.random.default_rng(sample_seed).choice(pool, max_users, replace=False)
        train = apply_k_core(train.loc[train["user"].isin(keep)], k_core)
        log.info("sampled %d of %d core users (seed %d)", max_users, len(pool), sample_seed)
    users = set(train["user"])
    items = set(train["item"]) | set(keep_cold_items or ())
    seen = set(zip(train["user"], train["item"], strict=True))

    def restrict(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.loc[frame["user"].isin(users) & frame["item"].isin(items)]
        pairs = list(zip(frame["user"], frame["item"], strict=True))
        return frame.loc[[p not in seen for p in pairs]].reset_index(drop=True)

    val, test = restrict(df.loc[part == 1]), restrict(df.loc[part == 2])
    log.info(
        "global split: train=%d (%d users, %d items) val=%d (%d users) test=%d (%d users)",
        len(train),
        len(users),
        train["item"].nunique(),
        len(val),
        val["user"].nunique(),
        len(test),
        test["user"].nunique(),
    )
    return SplitResult(train.reset_index(drop=True), val, test, n_short_users=0)


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
        len(train),
        len(val),
        len(test),
        n_short,
    )
    return SplitResult(train, val, test, n_short)
