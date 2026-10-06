"""Replay: cohort definition, leakage-safe profiles, rank metrics."""

import numpy as np
import pandas as pd

from firstpr.replay import cohorts as C
from firstpr.replay.evaluate import history_bin, rank_metrics


def _events(tmp_path, rows):
    ev = pd.DataFrame(rows, columns=["kind", "user", "repo_id", "first_at", "n"])
    ev["first_at"] = pd.to_datetime(ev["first_at"], utc=True)
    d = tmp_path / "events"
    d.mkdir()
    for day, part in ev.groupby(ev["first_at"].dt.strftime("%Y-%m-%d")):
        part.to_parquet(d / f"{day}.parquet", index=False)
    return {"github_dir": str(tmp_path)}


def test_cohort_first_pr_maintainer_and_month(tmp_path):
    a, b, m, x = "a" * 16, "b" * 16, "c" * 16, "d" * 16
    cfg = _events(
        tmp_path,
        [
            ("star", a, 1, "2025-08-01", 1),
            ("pr", a, 1, "2025-09-10", 1),  # a: first PR to 1 in Sep -> cohort
            ("pr", a, 2, "2025-09-20", 1),  # later first PR in Sep: not the query
            ("pr", b, 1, "2025-08-15", 1),  # b: first PR to 1 in Aug ...
            ("pr", b, 1, "2025-09-12", 1),  # ... so not a first PR in Sep
            ("push", m, 1, "2025-09-01", 1),
            ("pr", m, 1, "2025-09-05", 1),  # m pushed before: maintainer, excluded
            ("pr", x, 3, "2025-09-07", 1),  # repo 3 not in the catalog
        ],
    )
    co = C.cohort(cfg, [1, 2], "2025-09")
    assert list(co["user"]) == [a]
    assert co["repo_id"].iloc[0] == 1
    assert co["pr_ts"].iloc[0] == int(pd.Timestamp("2025-09-10", tz="UTC").timestamp())
    prof = C.profiles(cfg, co, ["star", "fork", "pr", "issue", "comment", "review"])
    assert list(prof["repo_id"]) == [1]  # the Aug star only: nothing at or after the PR
    assert prof["kind"].iloc[0] == "star"


def test_rank_metrics_hits_ndcg_and_near_misses():
    scores = np.array([[0.9, 0.8, 0.1, 0.0], [0.1, 0.9, 0.8, 0.0]])
    lang = np.array(["py", "go", "py", "rs"], dtype=object)
    owner = np.array(["o1", "o2", "o1", "o3"], dtype=object)
    m = rank_metrics(scores, np.array([1, 0]), lang, owner, k=2)
    assert m["hit"].tolist() == [1.0, 0.0]
    assert np.isclose(m["ndcg"][0], 1 / np.log2(3))
    assert m["near_lang"].tolist() == [0.0, 1.0]  # miss with a python repo (2) in the top 2
    assert m["near_org"].tolist() == [0.0, 1.0]  # repo 2 shares the owner of repo 0


def test_history_bins():
    assert [history_bin(n) for n in (0, 1, 5, 6, 20, 21)] == [
        "0",
        "1-5",
        "1-5",
        "6-20",
        "6-20",
        "20+",
    ]
