"""Paired bootstrap over users for differences between two models on the same test users.

For models A and B, d_u = metric_A(u) - metric_B(u), where each model's per-user metric is first
averaged over its seeds (the CI is about the model difference, not seed noise; seed spread is the
± std on the leaderboard). Resample users with replacement, recompute mean(d), and take the
percentile interval. Pairing removes the large between-user variance that two independent CIs
would wrongly include. The same resample indices are used for every pair.
"""

from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def paired_bootstrap(
    a: np.ndarray, b: np.ndarray, n_resamples: int, ci: float, seed: int
) -> dict[str, float]:
    """Mean of a - b with a percentile CI; `p_not_positive` = share of resamples with mean <= 0."""
    if a.shape != b.shape:
        raise ValueError("paired bootstrap needs per-user arrays over the same users")
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_resamples, len(d)))
    means = d[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return {
        "diff": float(d.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_not_positive": float((means <= 0).mean()),
        "n_users": int(len(d)),
    }


def seed_averaged_per_user(
    runs: list[dict[str, Any]], metric: str
) -> tuple[np.ndarray, np.ndarray]:
    """(users, per-user metric averaged over the runs' seeds) from the runs' saved .npz files."""
    key = metric.replace("@", "_at_")
    users, values = None, []
    for r in runs:
        z = np.load(Path(r["_path"]).with_name(r["per_user_file"]))
        if users is None:
            users = z["users"]
        elif not np.array_equal(users, z["users"]):
            raise ValueError(f"per-user files of {r['model']} cover different users")
        values.append(z[key])
    return users, np.mean(values, axis=0)


def pairwise_cis(
    final_runs: dict[tuple[str, str], list[dict[str, Any]]],
    metric: str,
    n_resamples: int,
    ci: float,
    seed: int,
) -> pd.DataFrame:
    """CI for every pair of models per dataset (A = higher mean metric). Models whose runs have no
    per-user files are skipped."""
    rows = []
    for dataset in sorted({d for d, _ in final_runs}):
        per_model = {}
        for (d, model), runs in final_runs.items():
            if d == dataset and all("per_user_file" in r for r in runs):
                per_model[model] = seed_averaged_per_user(runs, metric)
        ranked = sorted(per_model, key=lambda m: -per_model[m][1].mean())
        for rank_a, rank_b in combinations(range(len(ranked)), 2):
            a, b = ranked[rank_a], ranked[rank_b]
            (ua, va), (ub, vb) = per_model[a], per_model[b]
            if not np.array_equal(ua, ub):
                raise ValueError(f"{a} and {b} were evaluated on different users")
            res = paired_bootstrap(va, vb, n_resamples, ci, seed)
            rows.append(
                {
                    "dataset": dataset,
                    "model_a": a,
                    "model_b": b,
                    "metric": metric,
                    "adjacent": rank_b == rank_a + 1,
                    **res,
                    "n_resamples": n_resamples,
                    "ci": ci,
                }
            )
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame, pairs: list[tuple[str, str]] | None = None) -> str:
    """Adjacent pairs in ranking order, plus any explicitly requested pairs."""
    keep = df["adjacent"].copy()
    for a, b in pairs or []:
        keep |= ((df["model_a"] == a) & (df["model_b"] == b)) | (
            (df["model_a"] == b) & (df["model_b"] == a)
        )
    lines = [
        "| Dataset | A | B | Δ "
        + str(df["metric"].iloc[0])
        + " (A − B) | 95% CI | CI excludes 0 |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in df[keep].iterrows():
        excl = "yes" if r["ci_low"] > 0 or r["ci_high"] < 0 else "no"
        lines.append(
            f"| {r['dataset']} | {r['model_a']} | {r['model_b']} | {r['diff']:+.4f} | "
            f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | {excl} |"
        )
    return "\n".join(lines)
