"""Daily batch ingestion of GH Archive events for the repo set.

One BigQuery query per day table (~0.27 GB scan), aggregated per (type, actor, repo): first
timestamp and count. Actor ids are replaced by salted hashes in memory, the frame is checked for
raw ids, then written to `<github_dir>/events/<YYYY-MM-DD>.parquet`. Existing days are skipped,
so the same command backfills a range or appends yesterday.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

from firstpr.github import gharchive as gha
from firstpr.github.bigquery import BigQueryRunner, BudgetExceededError
from firstpr.github.privacy import assert_hashed, find_leaks, hash_column, load_salt
from firstpr.utils.disk import check_disk
from firstpr.utils.logging import get_logger

log = get_logger(__name__)


NON_ID_COLUMNS = ("repo_id", "n", "number")  # numbers that can equal an actor id by chance


def anonymise(df: pd.DataFrame, salt: bytes) -> pd.DataFrame:
    """actor_id -> user (salted hash); raises if a raw actor id survives in any column other
    than repo ids, counts and issue / PR numbers."""
    raw = set(df["actor_id"].astype(str))
    out = hash_column(df, "actor_id", salt, out="user")
    assert_hashed(out["user"])
    leaks = find_leaks(out, raw, skip=NON_ID_COLUMNS)
    if leaks:
        raise RuntimeError(f"raw actor ids found in columns {leaks}")
    return out


def day_path(events_dir: Path, day: pd.Timestamp) -> Path:
    return events_dir / f"{day:%Y-%m-%d}.parquet"


def ingest_day(
    day: pd.Timestamp, repo_ids: list[int], cfg: dict[str, Any], bq: BigQueryRunner, salt: bytes
) -> int:
    out = day_path(Path(cfg["github_dir"]) / "events", day)
    if out.exists():
        return 0
    params = {**gha.base_params(cfg), "repo_ids": repo_ids}
    table = bq.query(gha.day_events_sql(f"{day:%Y%m%d}"), f"day_{day:%Y%m%d}", params)
    df = table.to_pandas()
    df = anonymise(df, salt)
    df["kind"] = df["type"].map({**gha.INTERACTION_TYPES, gha.MAINTAINER_TYPE: "push"})
    df = df[["kind", "user", "repo_id", "first_at", "n"]]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.rename(out)  # atomic: a crashed day is re-ingested, never half-written
    return len(df)


def ingest_day_retry(
    day: pd.Timestamp,
    repo_ids: list[int],
    cfg: dict[str, Any],
    bq: BigQueryRunner,
    salt: bytes,
    attempts: int = 4,
) -> int:
    """`ingest_day` with backoff on transient network / API errors (budget errors are raised)."""
    for k in range(attempts):
        try:
            return ingest_day(day, repo_ids, cfg, bq, salt)
        except BudgetExceededError:
            raise
        except Exception as e:  # noqa: BLE001 - transport errors come in many types
            if k == attempts - 1:
                raise
            log.warning(
                "day %s failed (%s), retry in %ds", f"{day:%Y-%m-%d}", type(e).__name__, 30 * 2**k
            )
            time.sleep(30 * 2**k)
    return 0


def ingest_range(
    start: str, end: str, repo_ids: list[int], cfg: dict[str, Any], workers: int = 4
) -> None:
    """Days in [start, end) (YYYY-MM-DD), `workers` BigQuery jobs in flight."""
    bqc = cfg["bigquery"]
    gdir = Path(cfg["github_dir"])
    bq = BigQueryRunner(gdir / "bq_ledger.jsonl", bqc["max_gb_per_query"], bqc["monthly_budget_gb"])
    salt = load_salt()
    days = list(pd.date_range(start, end, freq="D", inclusive="left"))
    todo = [d for d in days if not day_path(gdir / "events", d).exists()]
    log.info("ingest %s..%s: %d days, %d to fetch", start, end, len(days), len(todo))
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for d, n in zip(
            todo,
            pool.map(lambda d: ingest_day_retry(d, repo_ids, cfg, bq, salt), todo),
            strict=True,
        ):
            done += 1
            log.info("day %s: %d rows (%d/%d)", f"{d:%Y-%m-%d}", n, done, len(todo))
            if done % 10 == 0:
                check_disk(gdir, cfg["disk"]["max_data_gb"], cfg["disk"]["warn_free_gb"])


def ingest_payload_check(
    start: str, end: str, repo_ids: list[int], cfg: dict[str, Any]
) -> pd.DataFrame:
    """PR / issue events with payload facts over [start, end] (inclusive days, YYYY-MM-DD) ->
    `<github_dir>/payload_<start>_<end>.parquet` (hashed users)."""
    gdir = Path(cfg["github_dir"])
    out = gdir / f"payload_{start}_{end}.parquet"
    if out.exists():
        return pd.read_parquet(out)
    bqc = cfg["bigquery"]
    bq = BigQueryRunner(gdir / "bq_ledger.jsonl", bqc["max_gb_per_query"], bqc["monthly_budget_gb"])
    sql = gha.payload_check_sql(start.replace("-", ""), end.replace("-", ""))
    params = {"bot_regex": cfg["bot_regex"], "repo_ids": repo_ids}
    df = anonymise(bq.query(sql, f"payload_{start}_{end}", params).to_pandas(), load_salt())
    df.to_parquet(out, index=False)
    return df
