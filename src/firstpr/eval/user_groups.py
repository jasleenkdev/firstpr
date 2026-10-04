"""Test metrics by user activity group (train history length quantiles).

Graph models claim to help users with few interactions, because propagation gives them signal
from neighbours of their few items (Wang et al. 2019; He et al. 2020). This splits test users
into quantile groups of train history length and reports each model's seed-averaged per-user
metric per group, plus paired bootstrap CIs inside each group for requested model pairs.
"""

from typing import Any

import numpy as np
import pandas as pd

from firstpr.eval.bootstrap import paired_bootstrap, seed_averaged_per_user


def activity_groups(history_len: np.ndarray, n_groups: int) -> tuple[np.ndarray, list[str]]:
    """Group id per user (0 = least active) from quantiles of `history_len`, with labels."""
    edges = np.quantile(history_len, np.linspace(0, 1, n_groups + 1))
    group = np.clip(np.searchsorted(edges[1:-1], history_len, side="right"), 0, n_groups - 1)
    labels = []
    for g in range(n_groups):
        members = history_len[group == g]
        labels.append(f"{int(members.min())}-{int(members.max())}" if len(members) else "empty")
    return group, labels


def group_table(
    final_runs: dict[str, list[dict[str, Any]]],
    history_len: np.ndarray,
    metric: str,
    n_groups: int,
) -> pd.DataFrame:
    """One row per (model, group): users, mean metric. `history_len` is indexed by user id."""
    rows = []
    for model, runs in final_runs.items():
        users, values = seed_averaged_per_user(runs, metric)
        group, labels = activity_groups(history_len[users], n_groups)
        for g, label in enumerate(labels):
            sel = group == g
            rows.append(
                {
                    "model": model,
                    "group": g,
                    "history_len": label,
                    "n_users": int(sel.sum()),
                    metric: float(values[sel].mean()),
                }
            )
    return pd.DataFrame(rows)


def group_cis(
    final_runs: dict[str, list[dict[str, Any]]],
    history_len: np.ndarray,
    pairs: list[tuple[str, str]],
    metric: str,
    n_groups: int,
    n_resamples: int,
    ci: float,
    seed: int,
) -> pd.DataFrame:
    """Paired bootstrap of A - B inside each activity group."""
    rows = []
    for a, b in pairs:
        if a not in final_runs or b not in final_runs:
            continue
        (ua, va), (ub, vb) = (seed_averaged_per_user(final_runs[m], metric) for m in (a, b))
        if not np.array_equal(ua, ub):
            raise ValueError(f"{a} and {b} were evaluated on different users")
        group, labels = activity_groups(history_len[ua], n_groups)
        for g, label in enumerate(labels):
            sel = group == g
            res = paired_bootstrap(va[sel], vb[sel], n_resamples, ci, seed)
            rows.append({"model_a": a, "model_b": b, "group": g, "history_len": label, **res})
    return pd.DataFrame(rows)
