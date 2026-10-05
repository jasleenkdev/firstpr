"""Test NDCG@20 by user activity quartile -> results/analysis/user_groups_<dataset>.csv (+ CIs)."""

import argparse
from pathlib import Path

import numpy as np

from firstpr.data.dataset import InteractionData
from firstpr.eval.leaderboard import select_final_runs
from firstpr.eval.user_groups import group_cis, group_table
from firstpr.models.registry import MODELS
from firstpr.train.runner import load_runs
from firstpr.utils.io import load_yaml

PAIRS = [
    ("lightgcn", "mf_bpr"),
    ("recbole_ngcf", "mf_bpr"),
    ("lightgcn", "recbole_ngcf"),
    ("lightgcn", "itemknn"),
    ("sasrec", "lightgcn"),
    ("sasrec_text_linear_id", "sasrec"),  # amazon: text + ID vs ID only
    ("sasrec_text_moe", "sasrec"),  # amazon: text only vs ID only
    ("sasrec_text_kar", "sasrec_text_linear_id"),  # amazon: KAR on top of its base
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="ml1m")
    p.add_argument("--n-groups", type=int, default=4)
    a = p.parse_args()
    data_cfg = load_yaml(f"configs/data/{a.data}.yaml")
    boot = load_yaml("configs/eval/default.yaml")["bootstrap"]
    data = InteractionData.from_processed(data_cfg["processed_dir"], data_cfg["head_fraction"])
    history_len = np.diff(data.train.indptr)

    results = Path("results")
    final = {
        model: runs
        for (dataset, model), runs in select_final_runs(load_runs(results, "test")).items()
        if dataset == data_cfg["name"] and all("per_user_file" in r for r in runs)
    }
    final = {m: final[m] for m in MODELS if m in final}  # registry order
    table = group_table(final, history_len, boot["metric"], a.n_groups)
    cis = group_cis(
        final,
        history_len,
        PAIRS,
        boot["metric"],
        a.n_groups,
        boot["n_resamples"],
        boot["ci"],
        boot["seed"],
    )
    out = results / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"user_groups_{data_cfg['name']}.csv", index=False, float_format="%.6g")
    cis.to_csv(out / f"user_groups_cis_{data_cfg['name']}.csv", index=False, float_format="%.6g")
    wide = table.pivot(index="model", columns="history_len", values=boot["metric"])
    wide = wide.reindex(index=list(final), columns=table["history_len"].unique())
    print(wide.round(4).to_string())
    if not cis.empty:
        print(cis[["model_a", "model_b", "history_len", "diff", "ci_low", "ci_high"]].round(4))


if __name__ == "__main__":
    main()
