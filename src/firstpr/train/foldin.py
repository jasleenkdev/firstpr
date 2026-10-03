"""Fold-in ablation: does giving a fitted model the user's val interactions as *input context*
at test time (no retraining) help?

The main protocol scores every model from train interactions only. History-based models can use
new interactions without retraining (ItemKNN: X_u · W; SASRec: a longer input sequence), which is
how they would be deployed. This ablation fits on train exactly as in the main runs, then
evaluates test twice: context = train (main protocol) and context = train + val.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from firstpr.data.dataset import InteractionData
from firstpr.eval.evaluator import Evaluator
from firstpr.models.registry import build_model
from firstpr.utils.io import git_info, save_json
from firstpr.utils.seed import set_seed

FOLDIN_MODELS = ("itemknn", "sasrec")


def set_foldin_context(model: Any, model_name: str, data: InteractionData) -> None:
    if model_name == "itemknn":
        model.set_context((data.train + data.val).tocsr())
    elif model_name == "sasrec":
        model.set_context(
            [
                np.concatenate([t, v])
                for t, v in zip(data.train_histories, data.val_histories, strict=True)
            ]
        )
    else:
        raise ValueError(f"fold-in not defined for {model_name}")


def run_foldin(
    model_name: str,
    data: InteractionData,
    params: dict[str, Any],
    eval_cfg: dict[str, Any],
    results_dir: Path,
) -> pd.DataFrame:
    evaluator = Evaluator(data, k=eval_cfg["k"], batch_size=eval_cfg["batch_size"])
    primary = eval_cfg["primary_metric"]
    rows = []
    for seed in eval_cfg["seeds"]:
        set_seed(seed)
        model = build_model(model_name)
        model.fit(
            data,
            {**params, "seed": seed},
            val_fn=lambda m: evaluator.evaluate(m, "val")["overall"][primary],
        )
        for context in ("train", "train+val"):
            if context == "train+val":
                set_foldin_context(model, model_name, data)
            res = evaluator.evaluate(model, "test")
            rows.append({"model": model_name, "seed": seed, "context": context, **res["overall"]})
            save_json(
                {
                    "model": model_name,
                    "seed": seed,
                    "context": context,
                    "config": params,
                    "git": git_info(),
                    "metrics": res,
                },
                results_dir
                / "runs"
                / f"{data.name}_foldin"
                / model_name
                / f"foldin_{context}_s{seed}.json",  # not test_*: kept out of the leaderboard
            )
    return pd.DataFrame(rows)
