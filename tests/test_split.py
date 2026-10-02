import numpy as np
import pandas as pd
import pytest

from firstpr.data.dataset import InteractionData, head_item_mask
from firstpr.data.split import chronological_order, chronological_split, split_sizes


@pytest.fixture
def interactions() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for u in range(40):
        n = int(rng.integers(5, 30))
        items = rng.choice(200, size=n, replace=False)
        ts = rng.integers(0, 50, size=n)  # small range -> many timestamp ties
        rows += [(u, int(i), int(t)) for i, t in zip(items, ts, strict=True)]
    return pd.DataFrame(rows, columns=["user", "item", "timestamp"])


def test_no_overlap_and_no_lost_interactions(interactions):
    s = chronological_split(interactions, 0.1, 0.1, tie_break_seed=0)
    assert len(s.train) + len(s.val) + len(s.test) == len(interactions)
    keys = [set(zip(f["user"], f["item"], strict=True)) for f in (s.train, s.val, s.test)]
    assert not keys[0] & keys[1] and not keys[0] & keys[2] and not keys[1] & keys[2]


def test_every_user_has_train_val_test(interactions):
    s = chronological_split(interactions, 0.1, 0.1, tie_break_seed=0)
    users = set(interactions["user"])
    for f in (s.train, s.val, s.test):
        assert set(f["user"]) == users
    assert s.n_short_users == 0


def test_chronology_holds_per_user(interactions):
    s = chronological_split(interactions, 0.1, 0.1, tie_break_seed=0)
    ts, seq = "timestamp", "seq"
    # timestamps never go backwards across splits
    assert (s.train.groupby("user")[ts].max() <= s.val.groupby("user")[ts].min()).all()
    assert (s.val.groupby("user")[ts].max() <= s.test.groupby("user")[ts].min()).all()
    # and the full order (time + tie-break) is strict
    assert (s.train.groupby("user")[seq].max() < s.val.groupby("user")[seq].min()).all()
    assert (s.val.groupby("user")[seq].max() < s.test.groupby("user")[seq].min()).all()


def test_tie_break_is_deterministic_for_fixed_seed(interactions):
    a = chronological_split(interactions, 0.1, 0.1, tie_break_seed=7)
    shuffled = interactions.sample(frac=1.0, random_state=3)
    b = chronological_split(shuffled, 0.1, 0.1, tie_break_seed=7)
    for x, y in [(a.train, b.train), (a.val, b.val), (a.test, b.test)]:
        pd.testing.assert_frame_equal(x, y)


def test_tie_break_depends_on_seed(interactions):
    # the fixture has many same-timestamp interactions, so a different seed reorders some ties
    a = chronological_split(interactions, 0.1, 0.1, tie_break_seed=0)
    b = chronological_split(interactions, 0.1, 0.1, tie_break_seed=1)
    assert not a.test.equals(b.test)


def test_tie_break_is_not_item_id_order():
    # one user, 40 interactions all at the same timestamp: random order, not sorted by item id
    df = pd.DataFrame({"user": 0, "item": np.arange(40), "timestamp": 5})
    order = chronological_order(df, tie_break_seed=0)["item"].to_numpy()
    assert sorted(order) == list(range(40))
    assert not (np.diff(order) > 0).all()


def test_split_sizes_rounding_and_short_users():
    n_train, n_val, n_test = split_sizes(np.array([1, 2, 3, 5, 10, 15, 25]), 0.1, 0.1)
    assert list(n_train) == [1, 1, 1, 3, 8, 11, 19]
    assert list(n_val) == [0, 0, 1, 1, 1, 2, 3]
    assert list(n_test) == [0, 1, 1, 1, 1, 2, 3]


def test_short_users_are_counted():
    df = pd.DataFrame({"user": [0, 0, 1], "item": [0, 1, 0], "timestamp": [1, 2, 1]})
    s = chronological_split(df, 0.1, 0.1, tie_break_seed=0)
    assert s.n_short_users == 2
    assert set(s.train["user"]) == {0, 1}
    assert list(s.test["item"]) == [1]


def test_interaction_data_histories_and_popularity(interactions):
    s = chronological_split(interactions, 0.1, 0.1, tie_break_seed=0)
    data = InteractionData.from_frames(s.train, s.val, s.test, n_users=40, n_items=200)
    assert data.train.shape == (40, 200)
    assert data.item_popularity.sum() == len(s.train)
    for u in (0, 7):
        hist = data.train_histories[u]
        rows = s.train[s.train["user"] == u].sort_values("seq")
        assert list(hist) == list(rows["item"])


def test_head_mask_top_fraction_with_id_tie_break():
    pop = np.array([5, 1, 5, 3, 0, 9, 2, 2, 1, 0])
    mask = head_item_mask(pop, 0.2)
    assert list(np.flatnonzero(mask)) == [0, 5]  # 9 first, then 5 (items 0 and 2) -> lower id
