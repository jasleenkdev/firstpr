"""Is SASRec's weakness on very active users stale context? Fit the tuned model on train (as in
the main runs), score test with context = train and = train + val, and report NDCG@20 per user
activity quartile -> results/analysis/foldin_user_groups_<dataset>.csv.

Very active users have long val periods (10% of their history) between the end of the train
context and the first test item, so a next-item model scored from train only is predicting tens
of steps ahead for them, but only 1-2 steps ahead for light users.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from firstpr.data.dataset import InteractionData
from firstpr.eval.evaluator import Evaluator
from firstpr.eval.user_groups import activity_groups
from firstpr.models.registry import build_model
from firstpr.train.foldin import FOLDIN_MODELS, set_foldin_context
from firstpr.utils.io import load_yaml
from firstpr.utils.seed import set_seed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="ml1m")
    p.add_argument("--models", nargs="+", default=list(FOLDIN_MODELS))
    p.add_argument("--n-groups", type=int, default=4)
    a = p.parse_args()
    data_cfg = load_yaml(f"configs/data/{a.data}.yaml")
    eval_cfg = load_yaml("configs/eval/default.yaml")
    data = InteractionData.from_processed(data_cfg["processed_dir"], data_cfg["head_fraction"])
    evaluator = Evaluator(data, k=eval_cfg["k"], batch_size=eval_cfg["batch_size"])
    primary = eval_cfg["primary_metric"]
    history_len = np.diff(data.train.indptr)
    val_len = np.diff(data.val.indptr)

    rows = []
    for name in a.models:
        best = Path("results") / "best" / f"{name}_{data_cfg['name']}.yaml"
        params = load_yaml(best)["params"]
        for seed in eval_cfg["seeds"]:
            set_seed(seed)
            model = build_model(name)
            model.fit(
                data,
                {**params, "seed": seed},
                val_fn=lambda m: evaluator.evaluate(m, "val")["overall"][primary],
            )
            for context in ("train", "train+val"):
                if context == "train+val":
                    set_foldin_context(model, name, data)
                per_user = evaluator.evaluate(model, "test", per_user=True)["per_user"]
                users, values = per_user["users"], per_user[primary]
                group, labels = activity_groups(history_len[users], a.n_groups)
                for g, label in enumerate(labels):
                    sel = group == g
                    rows.append(
                        {
                            "model": name,
                            "seed": seed,
                            "context": context,
                            "group": g,
                            "history_len": label,
                            "median_val_len": float(np.median(val_len[users[sel]])),
                            "n_users": int(sel.sum()),
                            primary: float(values[sel].mean()),
                        }
                    )
    df = pd.DataFrame(rows)
    out = Path("results") / "analysis" / f"foldin_user_groups_{data_cfg['name']}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = ["model", "context", "group", "history_len", "median_val_len", "n_users"]
    summary = df.groupby(keys)[primary].agg(["mean", "std"]).reset_index()
    summary.to_csv(out, index=False, float_format="%.6g")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
