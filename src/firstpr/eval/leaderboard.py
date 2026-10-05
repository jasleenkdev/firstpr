"""Aggregate test runs into mean ± std per (dataset, model) and render a markdown table."""

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

METRICS = ["recall@20", "ndcg@20", "hit@20", "coverage@20", "avg_pop@20", "long_tail_share@20"]
SLICE_METRICS = [("tail", "recall@20"), ("head", "recall@20")]
OPTIONAL_SLICE_METRICS = [("cold", "recall@20"), ("cold", "ndcg@20")]  # datasets with cold items


def select_final_runs(runs: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Per (dataset, model): the runs of the latest final config, newest run per seed."""
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in runs:
        groups[(r["dataset"], r["model"], r["config_hash"])].append(r)

    latest: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for (dataset, model, _), rs in groups.items():
        key = (dataset, model)
        if key not in latest or max(x["finished_at"] for x in rs) > max(
            x["finished_at"] for x in latest[key]
        ):
            latest[key] = rs
    final = {}
    for key, rs in latest.items():
        by_seed = {}  # if a config was forced to re-run, keep the newest run per seed
        for r in sorted(rs, key=lambda x: x["finished_at"]):
            by_seed[r["seed"]] = r
        final[key] = [by_seed[s] for s in sorted(by_seed)]
    return final


def aggregate(runs: list[dict[str, Any]], model_order: list[str]) -> pd.DataFrame:
    """One row per (dataset, model): the latest final config, mean and sample std (ddof=1)
    over its seeds."""
    latest = select_final_runs(runs)
    rows = []
    for (dataset, model), rs in latest.items():
        row: dict[str, Any] = {"dataset": dataset, "model": model, "n_seeds": len(rs)}
        values = {m: [r["metrics"]["overall"][m] for r in rs] for m in METRICS}
        for s, m in SLICE_METRICS:
            values[f"{s}_{m}"] = [r["metrics"]["slices"][s][m] for r in rs]
        for s, m in OPTIONAL_SLICE_METRICS:
            if all(s in r["metrics"]["slices"] for r in rs):
                values[f"{s}_{m}"] = [r["metrics"]["slices"][s][m] for r in rs]
        values["train_time_s"] = [r["train_time_s"] for r in rs]
        for name, v in values.items():
            arr = np.asarray(v, dtype=float)
            row[f"{name}_mean"] = float(arr.mean())
            row[f"{name}_std"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        row["config_hash"] = rs[0]["config_hash"]
        row["config"] = str(rs[0]["config"])
        row["git_commit"] = rs[-1]["git"]["commit"][:10]
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    rank = {m: i for i, m in enumerate(model_order)}
    df["_order"] = df["model"].map(lambda m: rank.get(m, len(rank)))
    return df.sort_values(["dataset", "_order"]).drop(columns="_order").reset_index(drop=True)


def to_markdown(df: pd.DataFrame) -> str:
    cols = [
        "recall@20",
        "ndcg@20",
        "hit@20",
        "coverage@20",
        "avg_pop@20",
        "long_tail_share@20",
        "tail_recall@20",
        "train_time_s",
    ]
    head = "| Model | Dataset | n | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 3)
    lines = [head, sep]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            fmt = "{:.1f}" if c in ("avg_pop@20", "train_time_s") else "{:.4f}"
            cells.append(f"{fmt.format(r[c + '_mean'])} ± {fmt.format(r[c + '_std'])}")
        lines.append(
            f"| {r['model']} | {r['dataset']} | {r['n_seeds']} | " + " | ".join(cells) + " |"
        )
    return "\n".join(lines)
