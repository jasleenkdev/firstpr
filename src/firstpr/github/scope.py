"""Repo set for the GitHub benchmark (all decisions from the train window, except cold items).

1. activity: distinct external actors per repo (BigQuery, month tables) for the train window and
   for val+test (the latter only to find new repos).
2. beginner: candidates with >= `candidate_min_contrib` external contributors in train; a repo is
   a beginner repo if it has a beginner-style label used on an issue created in the train window
   (GraphQL: label names, then labelled issues).
3. broad: repos sharing >= `broad_min_shared` external actors with beginner repos (collaborative
   signal beyond beginner repos).
4. cold: repos created on/after val_start with >= `cold_min_actors` external actors in val+test
   (items without train interactions, for the cold slice).
Each step writes a parquet under `<github_dir>/scope/` and is skipped when it exists.
"""

import re
from pathlib import Path
from typing import Any

import pandas as pd

from firstpr.github import gharchive as gha
from firstpr.github.api import GraphQLClient
from firstpr.github.bigquery import BigQueryRunner
from firstpr.utils.logging import get_logger

log = get_logger(__name__)


def _yyyymm(date: str) -> str:
    return date[:4] + date[5:7]


def _prev_month(date: str) -> str:
    ts = pd.Timestamp(date) - pd.offsets.MonthBegin(1)
    return ts.strftime("%Y%m")


def scope_dir(cfg: dict[str, Any]) -> Path:
    return Path(cfg["github_dir"]) / "scope"


def step_activity(cfg: dict[str, Any], bq: BigQueryRunner) -> None:
    w, sc = cfg["windows"], cfg["scoping"]
    params = {**gha.base_params(cfg), "min_actors": sc["min_actors"]}
    for name, start, end in [
        ("train", w["train_start"], w["val_start"]),
        ("valtest", w["val_start"], w["test_end"]),
    ]:
        out = scope_dir(cfg) / f"activity_{name}.parquet"
        if out.exists():
            continue
        sql = gha.repo_activity_sql(_yyyymm(start), _prev_month(end))
        table = bq.query(sql, f"activity_{name}", params)
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_pandas().to_parquet(out, index=False)
        log.info(
            "activity %s: %d repos with >= %d actors", name, table.num_rows, params["min_actors"]
        )


def beginner_pattern(labels: list[str]) -> re.Pattern[str]:
    """Label names that count as beginner labels: a configured phrase, ignoring case, spaces,
    hyphens, underscores and emoji / punctuation around it."""
    norm = sorted({re.sub(r"[\s_-]+", " ", lab.lower()).strip() for lab in labels})
    alts = "|".join(re.escape(n).replace(r"\ ", r"[\s_-]*") for n in norm)
    return re.compile(rf"(^|[^a-z])({alts})([^a-z]|$)", re.I)


def step_beginner(cfg: dict[str, Any], gql: GraphQLClient) -> None:
    out = scope_dir(cfg) / "beginner.parquet"
    if out.exists():
        return
    w, sc = cfg["windows"], cfg["scoping"]
    act = pd.read_parquet(scope_dir(cfg) / "activity_train.parquet")
    cand = act.loc[act["n_contrib"] >= sc["candidate_min_contrib"]]
    cand = cand.sort_values(["n_contrib", "repo_id"], ascending=[False, True])
    cand = cand.head(sc["max_candidates"])
    log.info("beginner check: %d candidates", len(cand))
    meta = gql.repo_labels(cand["repo_id"].tolist())
    pattern = beginner_pattern(cfg["beginner_labels"])
    meta["beginner_labels"] = meta["labels"].map(
        lambda names: [n for n in names if pattern.search(n)]
    )
    has = meta.loc[meta["beginner_labels"].map(len) > 0]
    used = gql.labelled_issue_dates(
        dict(zip(has["repo_id"], has["beginner_labels"], strict=True)), since=w["train_start"]
    )
    start, end = pd.Timestamp(w["train_start"], tz="UTC"), pd.Timestamp(w["val_start"], tz="UTC")
    meta["n_beginner_issues_train"] = meta["repo_id"].map(
        lambda r: sum(start <= pd.Timestamp(d) < end for d in used.get(r, []))
    )
    meta = meta.merge(cand, on="repo_id", how="left", suffixes=("", "_gha"))
    meta["is_beginner"] = meta["n_beginner_issues_train"] > 0
    meta["labels"] = meta["labels"].map(list)
    meta.to_parquet(out, index=False)
    log.info(
        "beginner: %d of %d candidates had beginner-labelled issues created in train",
        int(meta["is_beginner"].sum()),
        len(meta),
    )


def step_broad(cfg: dict[str, Any], bq: BigQueryRunner) -> None:
    out = scope_dir(cfg) / "broad.parquet"
    if out.exists():
        return
    w, sc = cfg["windows"], cfg["scoping"]
    beg = beginner_set(cfg)
    params = {
        **gha.base_params(cfg),
        "seed_ids": [int(x) for x in beg["repo_id"]],
        "min_shared": sc["broad_min_shared"],
    }
    sql = gha.co_touch_sql(_yyyymm(w["train_start"]), _prev_month(w["val_start"]))
    table = bq.query(sql, "co_touch_train", params)
    table.to_pandas().to_parquet(out, index=False)
    log.info("broad: %d repos share >= %d actors", table.num_rows, sc["broad_min_shared"])


def step_cold(cfg: dict[str, Any], gql: GraphQLClient) -> None:
    out = scope_dir(cfg) / "cold.parquet"
    if out.exists():
        return
    w, sc = cfg["windows"], cfg["scoping"]
    vt = pd.read_parquet(scope_dir(cfg) / "activity_valtest.parquet")
    train_ids = set(pd.read_parquet(scope_dir(cfg) / "activity_train.parquet")["repo_id"])
    # repo ids grow with creation time: new repos have ids above every train-active repo
    cand = vt.loc[(vt["n_actors"] >= sc["cold_min_actors"]) & ~vt["repo_id"].isin(train_ids)]
    cand = cand.loc[cand["repo_id"] > max(train_ids)]
    meta = gql.repo_labels(cand["repo_id"].tolist())
    meta = meta.merge(cand, on="repo_id", how="left", suffixes=("", "_gha"))
    meta["created"] = pd.to_datetime(meta["created_at"], utc=True)
    meta = meta.loc[meta["created"] >= pd.Timestamp(w["val_start"], tz="UTC")]
    meta = meta.loc[~meta["is_fork"]].drop(columns="created")
    meta["labels"] = meta["labels"].map(list)
    meta.to_parquet(out, index=False)
    log.info("cold: %d new repos (created >= %s)", len(meta), w["val_start"])


def beginner_set(cfg: dict[str, Any]) -> pd.DataFrame:
    sc = cfg["scoping"]
    beg = pd.read_parquet(scope_dir(cfg) / "beginner.parquet")
    beg = beg.loc[beg["is_beginner"] & ~beg["is_fork"]]
    beg = beg.sort_values(["n_contrib", "repo_id"], ascending=[False, True])
    return beg.head(sc["max_beginner"])


def finalize(cfg: dict[str, Any]) -> pd.DataFrame:
    """repo_set.parquet: repo_id, repo_name, group (beginner | broad | cold)."""
    sc, d = cfg["scoping"], scope_dir(cfg)
    beg = beginner_set(cfg)[["repo_id", "name"]].assign(group="beginner")
    broad = pd.read_parquet(d / "broad.parquet")
    broad = broad.loc[~broad["repo_id"].isin(beg["repo_id"])]
    broad = broad.sort_values(["n_shared", "repo_id"], ascending=[False, True]).head(
        sc["max_broad"]
    )
    broad = broad.rename(columns={"repo_name": "name"})[["repo_id", "name"]].assign(group="broad")
    cold = pd.read_parquet(d / "cold.parquet")
    cold = cold.sort_values(["n_actors", "repo_id"], ascending=[False, True]).head(sc["max_cold"])
    cold = cold[["repo_id", "name"]].assign(group="cold")
    out = pd.concat([beg, broad, cold], ignore_index=True)
    out = out.drop_duplicates("repo_id").reset_index(drop=True)
    out.to_parquet(d / "repo_set.parquet", index=False)
    log.info("repo set: %s", out["group"].value_counts().to_dict())
    return out
