"""GitHub pipeline: global temporal split, node ids, beginner labels, DuckDB build on toy events."""

import base64

import numpy as np
import pandas as pd

from firstpr.data.split import global_temporal_split
from firstpr.github import build
from firstpr.github.api import repo_node_id
from firstpr.github.gharchive import repo_activity_sql
from firstpr.github.scope import beginner_pattern


def test_repo_node_id_matches_github_format():
    # verified against the API: pallets/flask = 596892
    assert repo_node_id(596892).startswith("R_kgDO")
    raw = base64.urlsafe_b64decode(repo_node_id(596892)[2:] + "==")
    assert raw == b"\x92\x00\xce" + (596892).to_bytes(4, "big")
    assert base64.urlsafe_b64decode(repo_node_id(5)[2:] + "==") == b"\x92\x00\x05"


def test_beginner_pattern():
    p = beginner_pattern(["good first issue", "help wanted", "easy"])
    for yes in ["Good-First-Issue", "good first issue :wave:", "status: help wanted", "Easy"]:
        assert p.search(yes), yes
    for no in ["bug", "easyocr", "uneasy", "good first"]:
        assert not p.search(no), no


def test_activity_sql_never_selects_logins():
    sql = repo_activity_sql("202503", "202508")
    selected = sql.split("FROM")[0]
    assert "login" not in selected
    assert "githubarchive.month.20*" in sql


def _toy(rows: list[tuple[int, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["user", "item", "timestamp"])


def test_global_split_uses_one_cutoff_and_train_only_core():
    # users 0..5 each touch items 0..4 in train; user 0 also touches item 9 (cold) in test
    rows = [(u, i, 10 + i) for u in range(6) for i in range(5)]
    rows += [(0, 5, 150), (1, 6, 250), (0, 9, 260), (7, 1, 260)]  # 5,6 train-unknown; 7 new user
    rows += [(2, 0, 270)]  # repeat of a train pair
    s = global_temporal_split(_toy(rows), 100, 200, k_core=5, tie_break_seed=0, keep_cold_items={9})
    assert s.train["timestamp"].max() < 100
    assert set(s.test["item"]) == {9}  # item 6 never in train and not cold; user 7 unknown
    assert s.val.empty  # item 5 has no train interactions and is not listed as cold
    assert set(s.test["user"]) == {0}


def test_global_split_core_ignores_future():
    # user 6 has 4 train items + many test items: removed by the train-only 5-core
    rows = [(u, i, i) for u in range(6) for i in range(5)]
    rows += [(6, i, i) for i in range(4)] + [(6, 4, 300), (6, 0, 301)]
    s = global_temporal_split(_toy(rows), 100, 200, k_core=5, tie_break_seed=0)
    assert 6 not in set(s.train["user"]) and s.test.empty


def test_global_split_user_sample_keeps_core():
    rng = np.random.default_rng(0)
    rows = [(u, int(i), int(t)) for u in range(200) for t, i in enumerate(rng.choice(30, 8, False))]
    full = global_temporal_split(_toy(rows), 100, 200, k_core=5, tie_break_seed=0)
    s = global_temporal_split(_toy(rows), 100, 200, 5, 0, max_users=50, sample_seed=1)
    assert s.train["user"].nunique() <= 50 < full.train["user"].nunique()
    assert s.train.groupby("user").size().min() >= 5
    assert s.train.groupby("item").size().min() >= 5
    again = global_temporal_split(_toy(rows), 100, 200, 5, 0, max_users=50, sample_seed=1)
    assert again.train.equals(s.train)


def test_item_text_handles_missing_metadata():
    row = pd.Series({"name": "o/r", "language": None, "description": np.nan, "readme": np.nan})
    assert build.item_text(row, 100, 100) == "Repository: r"


def _write_events(tmp_path, rows):
    ev = pd.DataFrame(rows, columns=["kind", "user", "repo_id", "first_at", "n"])
    ev["first_at"] = pd.to_datetime(ev["first_at"], utc=True)
    d = tmp_path / "events"
    d.mkdir()
    for day, part in ev.groupby(ev["first_at"].dt.strftime("%Y-%m-%d")):
        part.to_parquet(d / f"{day}.parquet", index=False)


def _cfg(tmp_path, max_user_items=100):
    return {
        "github_dir": str(tmp_path),
        "windows": {
            "train_start": "2025-03-01",
            "val_start": "2025-09-01",
            "test_start": "2025-10-01",
            "test_end": "2025-11-01",
        },
        "preprocess": {
            "types": ["star", "fork", "pr", "issue", "comment", "review"],
            "max_user_items": max_user_items,
        },
    }


def test_interactions_first_touch_and_maintainer_filter(tmp_path):
    a, b, m = "a" * 16, "b" * 16, "c" * 16
    _write_events(
        tmp_path,
        [
            ("star", a, 1, "2025-04-01", 1),
            ("pr", a, 1, "2025-10-05", 2),  # same pair later: not a new interaction
            ("comment", b, 1, "2025-09-10", 3),  # contribution in val
            ("push", m, 1, "2025-05-01", 9),  # m maintains repo 1 from May
            ("pr", m, 1, "2025-06-01", 1),  # dropped (maintainer by end of train)
            ("issue", m, 2, "2025-06-01", 1),  # other repo: kept
            ("pr", b, 2, "2025-03-15", 1),
            ("push", b, 2, "2025-10-20", 1),  # b becomes maintainer of 2 only in test
            ("star", a, 3, "2025-02-01", 1),  # before the window
        ],
    )
    df = build.interactions(_cfg(tmp_path), [1, 2, 3])
    got = {(r.user, r.item): (r.kind, bool(r.contrib)) for r in df.itertuples()}
    assert got == {
        (a, 1): ("star", False),
        (b, 1): ("comment", True),
        (m, 2): ("issue", True),
        (b, 2): ("pr", True),  # train event kept: b had not pushed by the end of train
    }
    first = df.set_index(["user", "item"]).loc[(a, 1), "timestamp"]
    assert first == int(pd.Timestamp("2025-04-01", tz="UTC").timestamp())


def test_interactions_drop_heavy_users(tmp_path):
    heavy, light = "d" * 16, "e" * 16
    rows = [("star", heavy, r, "2025-04-01", 1) for r in range(5)]
    rows += [("star", light, 0, "2025-04-01", 1)]
    _write_events(tmp_path, rows)
    df = build.interactions(_cfg(tmp_path, max_user_items=3), list(range(5)))
    assert set(df["user"]) == {light}


def test_item_text_is_scrubbed_and_owner_free():
    row = pd.Series(
        {
            "name": "someowner/tool",
            "language": "Python",
            "description": "A tool by @someowner",
            "readme": "# Tool\nContact someowner@mail.com https://x.y",
        }
    )
    text = build.item_text(row, 1500, 300)
    assert "someowner" not in text and "http" not in text
    assert text.startswith("Repository: tool") and "Language: Python" in text
    assert np.isfinite(len(text))
