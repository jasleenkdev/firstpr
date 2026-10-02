"""Tune (mode val) or run final test seeds (mode test) for one model on one dataset."""

import argparse
from pathlib import Path

from firstpr.data.dataset import InteractionData
from firstpr.train.runner import final, tune
from firstpr.utils.io import load_yaml


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", default="ml1m")
    p.add_argument("--mode", choices=["val", "test"], required=True)
    p.add_argument(
        "--tie-break-seed",
        type=int,
        default=None,
        help="use the data variant prepared with this tie-break seed (robustness check)",
    )
    p.add_argument("--force", action="store_true", help="allow re-running test for a config")
    p.add_argument("--results-dir", default="results")
    a = p.parse_args()

    data_cfg = load_yaml(f"configs/data/{a.data}.yaml")
    model_cfg = load_yaml(f"configs/models/{a.model}.yaml")
    eval_cfg = load_yaml("configs/eval/default.yaml")

    processed, dataset = data_cfg["processed_dir"], data_cfg["name"]
    if a.tie_break_seed is not None and a.tie_break_seed != data_cfg["split"]["tie_break_seed"]:
        processed, dataset = f"{processed}_tb{a.tie_break_seed}", f"{dataset}_tb{a.tie_break_seed}"
    data = InteractionData.from_processed(processed, head_fraction=data_cfg["head_fraction"])

    results = Path(a.results_dir)
    if a.mode == "val":
        if dataset != data_cfg["name"]:
            raise SystemExit("tuning is only done on the main dataset; variants reuse its config")
        tune(a.model, data, model_cfg, eval_cfg, dataset, results)
    else:
        final(
            a.model,
            data,
            model_cfg,
            eval_cfg,
            dataset,
            results,
            force=a.force,
            tuned_from=data_cfg["name"],
        )


if __name__ == "__main__":
    main()
