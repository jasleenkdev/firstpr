"""Amazon Reviews 2023 (Hou, Li, He, Yan, Chen & McAuley, 2024): ratings and item text.

Ratings come from the McAuley lab's 5-core rating file (user_id, parent_asin, rating, timestamp
in ms). Item text comes from the category's metadata file (title, store, categories, feature
bullets, description); only text fields are kept, never images, prices or review text.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from firstpr.utils.logging import get_logger

log = get_logger(__name__)


def load_amazon_ratings(csv_path: str | Path, min_date: str | None = None) -> pd.DataFrame:
    """-> columns user, item (raw string ids), rating, timestamp (seconds)."""
    df = pd.read_csv(csv_path, dtype={"user_id": str, "parent_asin": str})
    df = df.rename(columns={"user_id": "user", "parent_asin": "item"})
    df["timestamp"] = (df["timestamp"] // 1000).astype("int64")
    n = len(df)
    if min_date is not None:
        cutoff = int(pd.Timestamp(min_date, tz="UTC").timestamp())
        df = df.loc[df["timestamp"] >= cutoff]
        log.info("kept %d of %d ratings on or after %s", len(df), n, min_date)
    # a user can rate the same product twice (variants share a parent_asin): keep the first
    df = df.sort_values(["user", "item", "timestamp"], kind="stable")
    df = df.drop_duplicates(["user", "item"], keep="first")
    return df[["user", "item", "rating", "timestamp"]].reset_index(drop=True)


def _clean(x: Any) -> str:
    return " ".join(str(x).split()) if x is not None else ""


def item_text(record: dict[str, Any], max_features: int, max_description_chars: int) -> str:
    """One plain-text description per item, built only from facts on the product page."""
    parts = []
    if title := _clean(record.get("title")):
        parts.append(f"Title: {title}")
    if store := _clean(record.get("store")):
        parts.append(f"Brand: {store}")
    cats = [_clean(c) for c in record.get("categories") or [] if _clean(c)]
    if cats:
        parts.append("Category: " + " > ".join(cats))
    feats = [_clean(f) for f in record.get("features") or [] if _clean(f)][:max_features]
    if feats:
        parts.append("Features: " + "; ".join(feats))
    desc = _clean(" ".join(str(d) for d in record.get("description") or []))
    if desc:
        if len(desc) > max_description_chars:
            desc = desc[:max_description_chars].rsplit(" ", 1)[0] + " ..."
        parts.append(f"Description: {desc}")
    return "\n".join(parts)


def item_table(
    records: list[dict[str, Any]], item_map: pd.DataFrame, max_features: int, max_desc: int
) -> pd.DataFrame:
    """Item text per contiguous item id (items without metadata get an empty title/text)."""
    by_asin = {r["parent_asin"]: r for r in records}
    rows = []
    for raw_id, item in zip(item_map["raw_id"], item_map["item"], strict=True):
        rec = by_asin.get(raw_id, {})
        rows.append(
            {
                "item": int(item),
                "raw_id": raw_id,
                "title": _clean(rec.get("title")),
                "text": item_text(rec, max_features, max_desc),
            }
        )
    out = pd.DataFrame(rows)
    log.info("item text: %d items, %d without metadata", len(out), int((out["text"] == "").sum()))
    return out
