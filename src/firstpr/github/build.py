"""Daily event files -> benchmark interactions (DuckDB over parquet).

1. Keep events on the repo set; drop an (actor, repo)'s events when the actor had pushed to the
   repo before the end of the event's window (maintainer by then: write access), so PR / issue /
   comment actors are external contributors. PushEvents themselves are never interactions.
2. One binary interaction per (user, repo): first touch over the kept kinds; its kind and
   whether the pair also has a contribution (PR / issue / comment / review) in that window.
3. Users touching more than `max_user_items` repos in train are dropped (star farms, scripts).
Then `data.split.global_temporal_split` (k-core on train only, cold items kept).
"""

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from firstpr.data.dataset import InteractionData
from firstpr.data.stats import compute_stats
from firstpr.github.gharchive import CONTRIB_KINDS
from firstpr.github.privacy import scrub_text
from firstpr.utils.io import save_json
from firstpr.utils.logging import get_logger

log = get_logger(__name__)


def _epoch(date: str) -> int:
    return int(pd.Timestamp(date, tz="UTC").timestamp())


def interactions(cfg: dict[str, Any], repo_ids: list[int]) -> pd.DataFrame:
    """-> user (hash), item (repo_id), timestamp (s), kind (first touch), contrib (bool)."""
    w, pp = cfg["windows"], cfg["preprocess"]
    events = str(Path(cfg["github_dir"]) / "events" / "*.parquet")
    con = duckdb.connect()
    con.register("repos", pd.DataFrame({"repo_id": repo_ids}))
    kinds = ", ".join(f"'{k}'" for k in pp["types"])
    contrib = ", ".join(f"'{k}'" for k in CONTRIB_KINDS)
    v, t, e = _epoch(w["val_start"]), _epoch(w["test_start"]), _epoch(w["test_end"])
    s = _epoch(w["train_start"])
    df = con.execute(
        f"""
        WITH ev AS (
          SELECT kind, "user", repo_id, epoch(first_at)::BIGINT AS ts, n
          FROM read_parquet('{events}') JOIN repos USING (repo_id)
          WHERE epoch(first_at) >= {s} AND epoch(first_at) < {e}
        ),
        win AS (  -- end of the window each event falls in
          SELECT *, CASE WHEN ts < {v} THEN {v} WHEN ts < {t} THEN {t} ELSE {e} END AS win_end
          FROM ev
        ),
        push AS (SELECT "user", repo_id, MIN(ts) AS first_push FROM ev WHERE kind = 'push'
                 GROUP BY ALL),
        kept AS (
          SELECT w.* FROM win w LEFT JOIN push p USING ("user", repo_id)
          WHERE w.kind IN ({kinds}) AND (p.first_push IS NULL OR p.first_push >= w.win_end)
        ),
        first AS (
          SELECT "user", repo_id, MIN(ts) AS ts, arg_min(kind, ts) AS kind,
                 arg_min(win_end, ts) AS win_end
          FROM kept GROUP BY ALL
        ),
        contrib AS (
          SELECT DISTINCT "user", repo_id, win_end FROM kept WHERE kind IN ({contrib})
        )
        SELECT f."user", f.repo_id AS item, f.ts AS timestamp, f.kind,
               c."user" IS NOT NULL AS contrib
        FROM first f LEFT JOIN contrib c USING ("user", repo_id, win_end)
        """
    ).df()
    n_pairs = len(df)
    train_counts = df.loc[df["timestamp"] < v].groupby("user").size()
    heavy = set(train_counts.index[train_counts > pp["max_user_items"]])
    df = df.loc[~df["user"].isin(heavy)].reset_index(drop=True)
    log.info(
        "interactions: %d first-touch pairs, %d users over %d train repos dropped (%d pairs)",
        n_pairs,
        len(heavy),
        pp["max_user_items"],
        n_pairs - len(df),
    )
    return df


def maintainer_share(cfg: dict[str, Any], repo_ids: list[int]) -> dict[str, float]:
    """Share of events per kind removed as maintainer actions (reported in stats.json)."""
    events = str(Path(cfg["github_dir"]) / "events" / "*.parquet")
    con = duckdb.connect()
    con.register("repos", pd.DataFrame({"repo_id": repo_ids}))
    rows = con.execute(
        f"""
        WITH ev AS (SELECT kind, "user", repo_id FROM read_parquet('{events}')
                    JOIN repos USING (repo_id)),
        m AS (SELECT DISTINCT "user", repo_id FROM ev WHERE kind = 'push')
        SELECT kind, AVG(CASE WHEN m."user" IS NULL THEN 0 ELSE 1 END) AS share
        FROM ev LEFT JOIN m USING ("user", repo_id) WHERE kind != 'push' GROUP BY kind
        """
    ).fetchall()
    return {k: float(v) for k, v in rows}


def item_text(row: pd.Series, readme_chars: int, description_chars: int) -> str:
    """Repo text from facts on the repo page: name, language, description, README (as of the
    train cutoff), cleaned of URLs, emails, @mentions and code blocks."""

    def field(key: str) -> str:
        v = row.get(key)
        return v if isinstance(v, str) else ""  # missing metadata arrives as None / NaN

    parts = [f"Repository: {field('name').split('/')[-1]}"]
    if lang := field("language"):
        parts.append(f"Language: {lang}")
    if desc := scrub_text(field("description"), description_chars):
        parts.append(f"Description: {desc}")
    if readme := scrub_text(field("readme"), readme_chars):
        parts.append(f"README: {readme}")
    return "\n".join(parts)


def items_table(cfg: dict[str, Any], item_map: pd.DataFrame) -> pd.DataFrame:
    """items.parquet for text models: item, raw_id, title, text (README as of the train cutoff
    from `<github_dir>/details.parquet`)."""
    it = cfg["item_text"]
    details = pd.read_parquet(Path(cfg["github_dir"]) / "details.parquet")
    df = item_map[["item", "raw_id", "name"]].merge(
        details.drop(columns="name").rename(columns={"repo_id": "raw_id"}), on="raw_id", how="left"
    )
    df["text"] = [
        item_text(r, it["readme_max_chars"], it["description_max_chars"]) for _, r in df.iterrows()
    ]
    df["title"] = df["name"].str.split("/").str[-1]
    n_readme = int((df["readme"].fillna("") != "").sum())
    log.info("item text: %d items, %d with a README at the cutoff", len(df), n_readme)
    return df[["item", "raw_id", "title", "text"]]


def github_stats(
    cfg: dict[str, Any], split: Any, item_map: pd.DataFrame, pairs: pd.DataFrame
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ("train", "val", "test"):
        frame = getattr(split, name)
        out[f"{name}_kinds"] = frame["kind"].value_counts().to_dict()
        out[f"{name}_contrib_share"] = float(frame["contrib"].mean()) if len(frame) else 0.0
        out[f"{name}_users"] = int(frame["user"].nunique())
    out["item_groups"] = item_map["group"].value_counts().to_dict()
    train_items = set(split.train["item"])
    cold = [i for i in item_map["item"] if i not in train_items]
    out["cold_items"] = len(cold)
    out["test_pairs_on_cold_items"] = int(split.test["item"].isin(cold).sum())
    out["pairs_before_filters"] = len(pairs)
    out["maintainer_event_share"] = maintainer_share(cfg, item_map["raw_id"].tolist())
    return out


def write_contrib_variant(
    cfg: dict[str, Any],
    out: Path,
    split: Any,
    user_map: pd.DataFrame,
    item_map: pd.DataFrame,
) -> None:
    """`<processed_dir>_contrib`: same train, val/test targets only where the pair contributed
    (the FirstPR question: where will this student contribute?). Tuned configs come from the main
    dataset (`--variant contrib`)."""
    if not cfg["split"].get("contrib_variant"):
        return
    vdir = Path(f"{out}_contrib")
    vdir.mkdir(parents=True, exist_ok=True)
    val = split.val.loc[split.val["contrib"]].reset_index(drop=True)
    test = split.test.loc[split.test["contrib"]].reset_index(drop=True)
    for name, frame in [("train", split.train), ("val", val), ("test", test)]:
        frame.to_parquet(vdir / f"{name}.parquet", index=False)
    user_map.to_parquet(vdir / "user_map.parquet", index=False)
    item_map.to_parquet(vdir / "item_map.parquet", index=False)
    (vdir / "items.parquet").write_bytes((out / "items.parquet").read_bytes())
    data = InteractionData.from_frames(
        split.train,
        val,
        test,
        n_users=len(user_map),
        n_items=len(item_map),
        head_fraction=cfg["head_fraction"],
        name=f"{cfg['name']}_contrib",
    )
    stats = compute_stats(data, 0)
    stats["config"] = cfg
    save_json(stats, vdir / "stats.json")
    log.info("contrib variant: val %d / test %d target pairs", len(val), len(test))
