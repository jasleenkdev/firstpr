"""KAR features: LLM item knowledge + user preferences -> <processed_dir>/kar_{items,users}.parquet.

--estimate prints the number of prompts and a token estimate (and how many are already cached)
without calling the LLM.
"""

import argparse
from pathlib import Path

import pandas as pd

from firstpr.data.dataset import InteractionData
from firstpr.llm.client import LLMClient
from firstpr.llm.kar import generate_all, item_prompt, user_prompt
from firstpr.utils.io import load_yaml, save_parquet


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="amazon_sci")
    p.add_argument("--config", default="configs/llm/kar.yaml")
    p.add_argument("--part", choices=["items", "users", "both"], default="both")
    p.add_argument("--estimate", action="store_true")
    a = p.parse_args()
    cfg = load_yaml(a.config)
    data_cfg = load_yaml(f"configs/data/{a.data}.yaml")
    data = InteractionData.from_processed(data_cfg["processed_dir"], data_cfg["head_fraction"])
    items = pd.read_parquet(Path(data_cfg["processed_dir"]) / "items.parquet")
    client = LLMClient(cfg["model"], cfg["backend"])
    opts = cfg["options"]

    jobs = {}
    if a.part in ("items", "both"):
        jobs["items"] = [item_prompt(t) for t in items["text"]]
    if a.part in ("users", "both"):
        titles = items["title"].tolist()
        u = cfg["user"]
        jobs["users"] = [  # train history without its last item (see llm/kar.py)
            user_prompt([titles[i] for i in h[:-1]], u["max_items"], u["max_title_chars"])
            for h in data.train_histories
        ]
    for part, prompts in jobs.items():
        n_cached = sum(client.cached(q, opts) is not None for q in prompts)
        chars = sum(len(q) for q in prompts)
        print(
            f"{part}: {len(prompts)} prompts, {n_cached} cached, ~{chars / 4 / 1e6:.2f}M prompt "
            f"tokens + up to {len(prompts) * opts['num_predict'] / 1e6:.2f}M output tokens"
        )
        if a.estimate:
            continue
        texts = generate_all(client, prompts, opts, workers=cfg["workers"])
        key = "item" if part == "items" else "user"
        out = pd.DataFrame({key: range(len(texts)), "text": texts})
        save_parquet(out, Path(data_cfg["processed_dir"]) / f"kar_{part}.parquet")
    print(f"uncached LLM calls made: {client.calls}")


if __name__ == "__main__":
    main()
