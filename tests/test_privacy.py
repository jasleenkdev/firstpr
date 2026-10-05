"""Privacy: salted hashing, leak checks, prompt scrubbing, and a scan of processed GitHub data."""

import re
from pathlib import Path

import pandas as pd
import pytest

from firstpr.github.ingest import anonymise
from firstpr.github.privacy import (
    HASH_PATTERN,
    assert_hashed,
    find_leaks,
    hash_column,
    hash_ids,
    scrub_text,
)
from firstpr.utils.io import REPO_ROOT

SALT = b"test-salt"


def test_hash_is_deterministic_salted_and_opaque():
    a = hash_ids([123, 456], SALT)
    assert list(a) == list(hash_ids([123, 456], SALT))
    assert list(a) != list(hash_ids([123, 456], b"other-salt"))
    assert all(HASH_PATTERN.match(h) for h in a)
    assert "123" not in a[0]


def test_hash_column_replaces_raw_ids():
    df = pd.DataFrame({"actor_id": [7, 8, 7], "repo_id": [1, 2, 3]})
    out = hash_column(df, "actor_id", SALT)
    assert "actor_id" not in out.columns
    assert out["user"].iloc[0] == out["user"].iloc[2] != out["user"].iloc[1]
    assert_hashed(out["user"])


def test_find_leaks_and_assert_hashed():
    df = pd.DataFrame({"user": ["Octocat", "x"], "repo": ["octocat/hello", "a/b"]})
    assert find_leaks(df, {"octocat"}) == ["user"]  # repo names are not exact matches
    nums = pd.DataFrame({"item": [100, 2], "user": ["ab", "100"]})
    assert find_leaks(nums, {"100"}) == ["item", "user"]
    assert find_leaks(nums, {"100"}, text_only=True) == ["user"]
    with pytest.raises(ValueError):
        assert_hashed(df["user"])


def test_anonymise_drops_raw_actor_ids():
    df = pd.DataFrame(
        {"type": ["WatchEvent"] * 2, "actor_id": [111, 222], "repo_id": [5, 6], "n": [1, 1]}
    )
    out = anonymise(df, SALT)
    assert "actor_id" not in out.columns
    assert not find_leaks(out, {"111", "222"}, skip=("repo_id", "n"))


def test_scrub_text_removes_people_identifiers():
    text = (
        "Maintained by @alice-dev and @bob. Mail me@example.com. "
        "![badge](https://img.shields.io/x) See [docs](https://x.io/docs) or https://y.org. "
        "```bash\npip install x\n``` <img src='a.png'> End"
    )
    out = scrub_text(text)
    assert "alice" not in out and "@bob" not in out and "example.com" not in out
    assert "http" not in out and "pip install" not in out and "<img" not in out
    assert "@user" in out and "docs" in out and out.endswith("End")
    assert scrub_text("word " * 100, max_chars=20).endswith("...")


def _processed_dirs() -> list[Path]:
    root = REPO_ROOT / "data" / "processed"
    return sorted(p for p in root.glob("github*") if (p / "train.parquet").exists())


@pytest.mark.skipif(not _processed_dirs(), reason="GitHub data not prepared")
def test_processed_github_data_holds_no_logins():
    """Repo owners are public logins we do hold (in repo names): none of them may appear as a
    user value or in any non-repo column of the processed data, and every user is a hash."""
    for d in _processed_dirs():
        items = pd.read_parquet(d / "item_map.parquet")
        owners = {n.split("/")[0].lower() for n in items["name"].dropna()}
        users = pd.read_parquet(d / "user_map.parquet")
        assert_hashed(users["raw_id"])
        for name in ("train", "val", "test", "user_map"):
            frame = pd.read_parquet(d / f"{name}.parquet")
            assert not find_leaks(frame, owners, text_only=True), f"{d.name}/{name}"
        text = pd.read_parquet(d / "items.parquet")["text"]
        assert not text.str.contains(r"https?://|[\w.+-]+@[\w-]+\.[\w.-]+", regex=True).any()
        mentions = text.str.findall(r"(?<![\w.])@([A-Za-z0-9][A-Za-z0-9-]*)").explode().dropna()
        assert set(mentions) <= {"user"}


@pytest.mark.skipif(
    not (REPO_ROOT / "data" / "github" / "events").exists(), reason="no GitHub events"
)
def test_event_files_hold_only_hashed_users():
    files = sorted((REPO_ROOT / "data" / "github" / "events").glob("*.parquet"))
    for f in files[:: max(1, len(files) // 20)]:  # ~20 days spread over the window
        df = pd.read_parquet(f)
        assert set(df.columns) == {"kind", "user", "repo_id", "first_at", "n"}
        assert df["user"].map(lambda u: bool(re.fullmatch(HASH_PATTERN.pattern, u))).all()
