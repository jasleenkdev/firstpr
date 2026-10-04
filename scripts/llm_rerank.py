"""LLM listwise reranking on a fixed random sample of test users -> results/llm_rerank/.

Candidate generator: the tuned LightGCN (seed 0) scored from train only, as in the main protocol.
Variants (see firstpr/llm/rerank.py): random-20 + text, LightGCN-20 + text (Q3), LightGCN-20 +
text + LightGCN scores (Q2). Baselines on the same candidate sets: LightGCN's own order and
popularity order. Paired bootstrap CIs over the sampled users.

--estimate prints the number of LLM calls and prompt sizes without calling the LLM.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from firstpr.data.dataset import InteractionData
from firstpr.eval import metrics as M
from firstpr.eval.bootstrap import paired_bootstrap
from firstpr.eval.evaluator import Evaluator
from firstpr.llm.client import LLMClient
from firstpr.llm.rerank import format_prompt, rerank_user, scaled_scores
from firstpr.models.registry import build_model
from firstpr.utils.io import load_yaml, save_json
from firstpr.utils.seed import set_seed

VARIANTS = [  # (name, candidate set, show recommender scores)
    ("random20_text", "random20", False),
    ("lightgcn20_text", "lightgcn20", False),
    ("lightgcn20_text_score", "lightgcn20", True),
]


def per_user_metrics(ranked: np.ndarray, targets: np.ndarray, k: int) -> dict[str, np.ndarray]:
    """ranked [U, 20] item ids; targets dense bool [U, n_items]."""
    hits = M.hits_matrix(ranked, targets)
    n = targets.sum(axis=1)
    return {
        f"ndcg@{k}": M.ndcg_at_k(hits[:, :k], n),
        f"recall@{k}": M.recall_at_k(hits[:, :k], n),
        "ndcg@20": M.ndcg_at_k(hits, n),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="amazon_sci")
    p.add_argument("--config", default="configs/llm/rerank.yaml")
    p.add_argument("--estimate", action="store_true")
    a = p.parse_args()
    cfg = load_yaml(a.config)
    data_cfg = load_yaml(f"configs/data/{a.data}.yaml")
    data = InteractionData.from_processed(data_cfg["processed_dir"], data_cfg["head_fraction"])
    items = pd.read_parquet(Path(data_cfg["processed_dir"]) / "items.parquet")
    titles = items["title"].tolist()
    k, n_cand = cfg["k"], cfg["n_candidates"]

    rng = np.random.default_rng(cfg["seed"])
    ev = Evaluator(data, k=n_cand)
    users = np.sort(rng.choice(ev.users_to_evaluate("test"), size=cfg["n_users"], replace=False))
    seen = (data.train + data.val).tocsr()
    targets = data.test[users].toarray().astype(bool)

    # candidate generator: tuned LightGCN, seed 0, train-only context
    params = load_yaml(Path("results/best") / f"lightgcn_{data_cfg['name']}.yaml")["params"]
    set_seed(0)
    lgcn = build_model("lightgcn")
    lgcn.fit(data, {**params, "seed": 0})
    lgcn_scores = lgcn.score(users)
    lgcn_top = ev.topk(lgcn, users, seen)

    random_sets = np.zeros((len(users), n_cand), dtype=np.int64)
    random_targets = np.zeros_like(targets)
    for r, u in enumerate(users):
        urng = np.random.default_rng([cfg["seed"], int(u)])
        target = urng.choice(np.flatnonzero(targets[r]))
        pool = np.setdiff1d(np.arange(data.n_items), np.r_[seen[u].indices, data.test[u].indices])
        random_sets[r] = urng.permutation(np.r_[target, urng.choice(pool, n_cand - 1, False)])
        random_targets[r, target] = True
    cand = {"lightgcn20": lgcn_top, "random20": random_sets}
    tgt = {"lightgcn20": targets, "random20": random_targets}

    hist_titles = [
        [titles[i] for i in data.train_histories[u][-cfg["max_history"] :]] for u in users
    ]
    n_calls = len(VARIANTS) * len(users) * cfg["n_shuffles"]
    example = format_prompt(
        hist_titles[0], [titles[i] for i in lgcn_top[0]], [0.5] * n_cand, cfg["max_title_chars"]
    )
    print(
        f"{n_calls} LLM calls ({len(VARIANTS)} variants x {len(users)} users x "
        f"{cfg['n_shuffles']} shuffles); example prompt ~{len(example) / 4:.0f} tokens"
    )
    if a.estimate:
        print(example)
        return

    client = LLMClient(cfg["model"], cfg["backend"])

    def generate(prompt: str) -> str:
        return client.generate(prompt, cfg["options"])

    out_dir = Path("results") / "llm_rerank"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_user: dict[str, dict[str, np.ndarray]] = {}
    summary: dict[str, dict] = {}
    pop = data.item_popularity
    for set_name in ("random20", "lightgcn20"):  # baselines on each candidate set
        c = cand[set_name]
        lgcn_order = np.take_along_axis(
            c, np.argsort(-np.take_along_axis(lgcn_scores, c, 1), 1, kind="stable"), 1
        )
        pop_order = np.take_along_axis(c, np.argsort(-pop[c], 1, kind="stable"), 1)
        per_user[f"{set_name}_lightgcn_order"] = per_user_metrics(lgcn_order, tgt[set_name], k)
        per_user[f"{set_name}_popularity_order"] = per_user_metrics(pop_order, tgt[set_name], k)
    for name, set_name, show_scores in VARIANTS:
        c = cand[set_name]

        def one(r: int, c=c, show_scores=show_scores, name=name) -> tuple[np.ndarray, int]:
            s = scaled_scores(lgcn_scores[r, c[r]]) if show_scores else None
            urng = np.random.default_rng([cfg["seed"], int(users[r]), len(name)])
            return rerank_user(
                generate,
                hist_titles[r],
                c[r],
                titles,
                s,
                cfg["n_shuffles"],
                urng,
                cfg["max_title_chars"],
            )

        with ThreadPoolExecutor(max_workers=cfg["workers"]) as pool:
            res = list(tqdm(pool.map(one, range(len(users))), total=len(users), desc=name))
        ranked = np.stack([r for r, _ in res])
        per_user[name] = per_user_metrics(ranked, tgt[set_name], k)
        n_prompts = len(users) * cfg["n_shuffles"]
        summary[name] = {"missing_ids_per_call": sum(m for _, m in res) / n_prompts}

    rows = []
    for name, m in per_user.items():
        rows.append({"variant": name, **{key: float(v.mean()) for key, v in m.items()}})
    pd.DataFrame(rows).to_csv(out_dir / f"summary_{data_cfg['name']}.csv", index=False)
    b = cfg["bootstrap"]
    comparisons = [
        ("random20_text", "random20_lightgcn_order"),
        ("random20_text", "random20_popularity_order"),
        ("lightgcn20_text", "lightgcn20_lightgcn_order"),
        ("lightgcn20_text_score", "lightgcn20_lightgcn_order"),
        ("lightgcn20_text_score", "lightgcn20_text"),
    ]
    cis = []
    for x, y in comparisons:
        res = paired_bootstrap(
            per_user[x][f"ndcg@{k}"], per_user[y][f"ndcg@{k}"], b["n_resamples"], b["ci"], b["seed"]
        )
        cis.append({"a": x, "b": y, "metric": f"ndcg@{k}", **res})
    pd.DataFrame(cis).to_csv(out_dir / f"cis_{data_cfg['name']}.csv", index=False)
    np.savez_compressed(
        out_dir / f"per_user_{data_cfg['name']}.npz",
        users=users,
        **{f"{n}__{key}": v for n, m in per_user.items() for key, v in m.items()},
    )
    save_json(
        {"config": cfg, "n_users": len(users), "uncached_calls": client.calls, **summary},
        out_dir / f"meta_{data_cfg['name']}.json",
    )
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(pd.DataFrame(cis)[["a", "b", "diff", "ci_low", "ci_high"]].round(4).to_string())


if __name__ == "__main__":
    main()
