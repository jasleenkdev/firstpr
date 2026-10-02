import numpy as np
import pandas as pd

from firstpr.data.preprocess import k_core, remap_ids, to_implicit


def test_to_implicit_keeps_only_ratings_at_or_above_threshold():
    df = pd.DataFrame(
        {"user": [1, 1, 2, 2], "item": [1, 2, 1, 3], "rating": [5, 3, 4, 1], "timestamp": [1] * 4}
    )
    out = to_implicit(df, threshold=4)
    assert list(zip(out["user"], out["item"], strict=True)) == [(1, 1), (2, 1)]
    assert "rating" not in out.columns


def test_k_core_is_iterative_and_result_is_k_core():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "user": rng.integers(0, 60, 800),
            "item": rng.integers(0, 120, 800),
            "timestamp": rng.integers(0, 10_000, 800),
        }
    ).drop_duplicates(["user", "item"])
    out = k_core(df, k=5)
    assert len(out) > 0
    assert out["user"].value_counts().min() >= 5
    assert out["item"].value_counts().min() >= 5


def test_k_core_cascade():
    # users 0-5 each rate items 0-4 (a stable 5-core block).
    # user 6 rates items 0, 1, 2, 9 (only 4 -> dropped); item 9 is rated by users 0-3 and 6, so
    # it has 5 users at first but only 4 after user 6 goes -> dropped in the second pass.
    rows = [(u, i) for u in range(6) for i in range(5)]
    rows += [(u, 9) for u in range(4)] + [(6, i) for i in (0, 1, 2, 9)]
    df = pd.DataFrame(rows, columns=["user", "item"]).assign(timestamp=0)
    out = k_core(df, k=5)
    assert set(out["user"]) == set(range(6))
    assert set(out["item"]) == set(range(5))
    assert len(out) == 30


def test_remap_ids_contiguous_and_deterministic():
    df = pd.DataFrame({"user": [30, 10, 20, 10], "item": [7, 5, 7, 9], "timestamp": [0] * 4})
    out, user_map, item_map = remap_ids(df)
    assert sorted(out["user"].unique()) == [0, 1, 2]
    assert sorted(out["item"].unique()) == [0, 1, 2]
    assert list(user_map["raw_id"]) == [10, 20, 30]
    # mapping round-trips
    assert list(user_map["raw_id"].to_numpy()[out["user"]]) == [30, 10, 20, 10]
    assert list(item_map["raw_id"].to_numpy()[out["item"]]) == [7, 5, 7, 9]
