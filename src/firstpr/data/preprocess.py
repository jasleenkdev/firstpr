"""Raw explicit ratings -> implicit interactions with k-core filtering and contiguous ids."""

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from firstpr.utils.logging import get_logger

log = get_logger(__name__)


def load_ml1m_ratings(zip_path: str | Path, member: str) -> pd.DataFrame:
    """Read `ratings.dat` (UserID::MovieID::Rating::Timestamp) from the ML-1M archive."""
    with zipfile.ZipFile(zip_path) as zf:
        text = zf.read(member).replace(b"::", b"\t")  # C parser needs a 1-char separator
    return pd.read_csv(
        io.BytesIO(text),
        sep="\t",
        header=None,
        names=["user", "item", "rating", "timestamp"],
        dtype="int64",
    )


def to_implicit(ratings: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Keep ratings >= threshold as positive interactions; drop the rest."""
    kept = ratings.loc[ratings["rating"] >= threshold, ["user", "item", "timestamp"]]
    log.info(
        "implicit conversion: kept %d of %d ratings (>= %s)", len(kept), len(ratings), threshold
    )
    return kept.reset_index(drop=True)


def k_core(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Repeatedly drop users and items with fewer than k interactions until none remain."""
    for n_pass in range(1, 1000):
        user_counts = df["user"].map(df["user"].value_counts())
        item_counts = df["item"].map(df["item"].value_counts())
        keep = (user_counts >= k) & (item_counts >= k)
        n_drop = int((~keep).sum())
        log.info("k-core pass %d: dropping %d interactions", n_pass, n_drop)
        if n_drop == 0:
            return df.reset_index(drop=True)
        df = df.loc[keep]
    raise RuntimeError("k-core filtering did not converge")


def remap_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Map raw user/item ids to contiguous 0..n-1 ints (sorted by raw id, so deterministic)."""
    user_map = pd.DataFrame({"raw_id": np.sort(df["user"].unique())})
    user_map["user"] = np.arange(len(user_map))
    item_map = pd.DataFrame({"raw_id": np.sort(df["item"].unique())})
    item_map["item"] = np.arange(len(item_map))

    out = df.copy()
    out["user"] = np.searchsorted(user_map["raw_id"].to_numpy(), df["user"].to_numpy())
    out["item"] = np.searchsorted(item_map["raw_id"].to_numpy(), df["item"].to_numpy())
    return out.reset_index(drop=True), user_map, item_map
