"""Replay cohorts and leakage-safe profiles from the daily GH Archive event files (DuckDB).

First PR: the earliest PullRequestEvent of a (user, repo) pair in our window (from 2025-03-01),
by a user who had not pushed to that repo at or before that moment (not a maintainer then).
"First-ever" therefore means "first since March 2025 in our data": earlier PRs are not visible.
A cohort = users whose earliest first PR to a catalog repo falls in the month; one query per
user (that earliest PR). Profile = the user's first-touch interactions (star, fork, PR, issue,
comment, review) on any repo of the repo set, strictly before the PR timestamp, with the same
time-local maintainer rule.
"""

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


def _con(cfg: dict[str, Any]) -> tuple[duckdb.DuckDBPyConnection, str]:
    return duckdb.connect(), str(Path(cfg["github_dir"]) / "events" / "*.parquet")


def cohort(cfg: dict[str, Any], catalog_repos: list[int], month: str) -> pd.DataFrame:
    """-> user, repo_id, pr_ts (s): each user's earliest first PR to a catalog repo in `month`
    (YYYY-MM)."""
    con, events = _con(cfg)
    con.register("cat", pd.DataFrame({"repo_id": catalog_repos}))
    return con.execute(
        f"""
        WITH ev AS (SELECT kind, "user", repo_id, epoch(first_at)::BIGINT AS ts
                    FROM read_parquet('{events}')),
        push AS (SELECT "user", repo_id, MIN(ts) AS fp FROM ev WHERE kind = 'push' GROUP BY ALL),
        pr AS (SELECT "user", repo_id, MIN(ts) AS ts FROM ev WHERE kind = 'pr' GROUP BY ALL),
        first AS (SELECT pr.* FROM pr LEFT JOIN push USING ("user", repo_id)
                  WHERE push.fp IS NULL OR push.fp > pr.ts),
        inmonth AS (SELECT f.* FROM first f JOIN cat USING (repo_id)
                    WHERE strftime(to_timestamp(f.ts), '%Y-%m') = '{month}')
        SELECT "user", arg_min(repo_id, ts) AS repo_id, MIN(ts) AS pr_ts
        FROM inmonth GROUP BY "user" ORDER BY "user"
        """
    ).df()


def profiles(cfg: dict[str, Any], users: pd.DataFrame, kinds: list[str]) -> pd.DataFrame:
    """-> user, repo_id, ts, kind: first touches strictly before each user's pr_ts."""
    con, events = _con(cfg)
    con.register("q", users[["user", "pr_ts"]])
    kind_list = ", ".join(f"'{k}'" for k in kinds)
    return con.execute(
        f"""
        WITH ev AS (SELECT e.kind, e."user", e.repo_id, epoch(e.first_at)::BIGINT AS ts
                    FROM read_parquet('{events}') e JOIN q USING ("user")
                    WHERE epoch(e.first_at) < q.pr_ts),
        push AS (SELECT "user", repo_id, MIN(ts) AS fp FROM ev WHERE kind = 'push' GROUP BY ALL),
        kept AS (SELECT ev.* FROM ev LEFT JOIN push USING ("user", repo_id)
                 WHERE ev.kind IN ({kind_list}) AND (push.fp IS NULL OR push.fp > ev.ts))
        SELECT "user", repo_id, MIN(ts) AS ts, arg_min(kind, ts) AS kind
        FROM kept GROUP BY ALL ORDER BY "user", ts
        """
    ).df()
