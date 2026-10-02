import numpy as np
import pandas as pd
import pytest

from firstpr.data.dataset import InteractionData
from firstpr.data.split import chronological_split


def make_random_data(
    n_users: int, n_items: int, per_user: tuple[int, int], seed: int = 0, head_fraction: float = 0.2
) -> InteractionData:
    """Random interactions with a popularity skew, split chronologically."""
    rng = np.random.default_rng(seed)
    p = 1.0 / np.arange(1, n_items + 1) ** 0.8
    p /= p.sum()
    rows = []
    for u in range(n_users):
        n = int(rng.integers(*per_user))
        items = rng.choice(n_items, size=n, replace=False, p=p)
        rows += [(u, int(i), t) for t, i in enumerate(items)]
    df = pd.DataFrame(rows, columns=["user", "item", "timestamp"])
    s = chronological_split(df, 0.1, 0.1, tie_break_seed=0)
    return InteractionData.from_frames(
        s.train, s.val, s.test, n_users, n_items, head_fraction=head_fraction
    )


@pytest.fixture
def toy_data() -> InteractionData:
    return make_random_data(n_users=50, n_items=80, per_user=(10, 30))


@pytest.fixture
def make_data():
    """Factory fixture: `make_data(n_users=..., n_items=..., per_user=(lo, hi), seed=...)`."""
    return make_random_data
