"""Cold-item slice: test targets on items with no train interactions -> results/analysis/.

Per model: NDCG@20 and Recall@20 against cold targets only, averaged over the users that have
at least one cold target (per-user values averaged over seeds first). Paired bootstrap CIs over
those users for the pairs below. ID-only models cannot score an unseen item above chance; text
models can, which is the inductive argument of ZESRec / UniSRec.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from firstpr.eval.bootstrap import paired_bootstrap, seed_averaged_per_user
from firstpr.eval.leaderboard import select_final_runs
from firstpr.models.registry import MODELS
from firstpr.train.runner import load_runs
from firstpr.utils.io import load_yaml

METRIC = "cold_ndcg@20"
PAIRS = [
    ("sasrec_text_moe", "sasrec"),
    ("sasrec_text_mlp", "sasrec"),
    ("sasrec_text_linear", "sasrec"),
    ("sasrec_text_linear_id", "sasrec"),
    ("sasrec_text_moe_id", "sasrec"),
    ("sasrec_text_moe", "sasrec_text_moe_id"),
    ("sasrec_text_kar", "sasrec_text_linear_id"),
    ("sasrec_text_kar_item", "sasrec_text_linear_id"),
    ("sasrec_text_moe_warm", "sasrec_text_moe"),
    ("sasrec_text_linear_id_warm", "sasrec_text_linear_id"),
    ("sasrec_text_moe_warm", "sasrec"),
    ("sasrec_text_mlp_warm", "sasrec_text_mlp"),  # github: warm-only softmax
    ("sasrec_text_mlp_warm", "sasrec"),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="amazon_sci")
    a = p.parse_args()
    name = load_yaml(f"configs/data/{a.data}.yaml")["name"]
    boot = load_yaml("configs/eval/default.yaml")["bootstrap"]
    final = {
        model: runs
        for (dataset, model), runs in select_final_runs(load_runs(Path("results"), "test")).items()
        if dataset == name and all("per_user_file" in r for r in runs)
    }
    per_model = {m: seed_averaged_per_user(final[m], METRIC) for m in MODELS if m in final}
    users = next(iter(per_model.values()))[0]
    has_cold = ~np.isnan(next(iter(per_model.values()))[1])
    rows = []
    for m, (u, v) in per_model.items():
        if not np.array_equal(u, users) or not np.array_equal(~np.isnan(v), has_cold):
            raise ValueError(f"{m}: per-user cold values cover different users")
        recall = seed_averaged_per_user(final[m], "cold_recall@20")[1] if _has(final[m]) else None
        rows.append(
            {
                "model": m,
                "cold_ndcg@20": float(v[has_cold].mean()),
                "cold_recall@20": float(recall[has_cold].mean()) if recall is not None else None,
                "n_users": int(has_cold.sum()),
            }
        )
    cis = []
    for x, y in PAIRS:
        if x in per_model and y in per_model:
            res = paired_bootstrap(
                per_model[x][1][has_cold],
                per_model[y][1][has_cold],
                boot["n_resamples"],
                boot["ci"],
                boot["seed"],
            )
            cis.append({"model_a": x, "model_b": y, "metric": METRIC, **res})
    out = Path("results") / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / f"cold_items_{name}.csv", index=False, float_format="%.6g")
    pd.DataFrame(cis).to_csv(out / f"cold_items_cis_{name}.csv", index=False, float_format="%.6g")
    print(table.round(4).to_string(index=False))
    if cis:
        print(pd.DataFrame(cis)[["model_a", "model_b", "diff", "ci_low", "ci_high"]].round(4))


def _has(runs: list[dict]) -> bool:
    z = np.load(Path(runs[0]["_path"]).with_name(runs[0]["per_user_file"]))
    return "cold_recall_at_20" in z.files


if __name__ == "__main__":
    main()
