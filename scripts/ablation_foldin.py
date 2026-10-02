"""Fold-in ablation for history-based models -> results/ablations/foldin_<dataset>.csv."""

import argparse
from pathlib import Path

import pandas as pd

from firstpr.data.dataset import InteractionData
from firstpr.train.foldin import FOLDIN_MODELS, run_foldin
from firstpr.utils.io import load_yaml


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="ml1m")
    p.add_argument("--models", nargs="+", default=list(FOLDIN_MODELS))
    a = p.parse_args()
    data_cfg = load_yaml(f"configs/data/{a.data}.yaml")
    eval_cfg = load_yaml("configs/eval/default.yaml")
    data = InteractionData.from_processed(data_cfg["processed_dir"], data_cfg["head_fraction"])
    results = Path("results")
    frames = []
    for m in a.models:
        params = load_yaml(results / "best" / f"{m}_{data_cfg['name']}.yaml")["params"]
        frames.append(run_foldin(m, data, params, eval_cfg, results))
    df = pd.concat(frames)
    out = results / "ablations" / f"foldin_{data_cfg['name']}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = df.groupby(["model", "context"]).agg(["mean", "std"]).drop(columns="seed")
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.reset_index().to_csv(out, index=False, float_format="%.6g")
    print(summary[["recall@20_mean", "ndcg@20_mean", "ndcg@20_std"]].to_markdown())


if __name__ == "__main__":
    main()
