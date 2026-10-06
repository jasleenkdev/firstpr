"""Replay pipeline: cohorts -> profiles -> models (3 seeds) -> per-user metrics, plus the serving
ladder grid (text-SASRec + text profile + language boost) on both cohorts."""

import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from firstpr.data.dataset import InteractionData
from firstpr.models.sasrec_text import text_matrix
from firstpr.replay import cohorts as C
from firstpr.replay.evaluate import Queries, evaluate, rank_metrics
from firstpr.replay.models import MODELS, Scorer
from firstpr.text.encoder import DEFAULT_ENCODER
from firstpr.utils.logging import get_logger

log = get_logger(__name__)

MONTHS = {"tune": "2025-09", "report": "2025-10"}
SEEDS = [0, 1, 2]
LADDER_K = [1, 2, 3, 5, 10, 20, 10**9]  # weight on the model = min(1, n_catalog_stars / k)
LADDER_BETA = [0.0, 0.15, 0.3, 0.6]  # language-match boost
SERVING_NOW = (3, 0.15)  # phase-6 deployed values


def build_queries(cfg: dict[str, Any], data_dir: Path, out: Path) -> dict[str, Any]:
    item_map = pd.read_parquet(data_dir / "item_map.parquet")
    idx = dict(zip(item_map["raw_id"], item_map["item"], strict=True))
    res = {}
    for split, month in MONTHS.items():
        f = out / f"queries_{month}.parquet"
        if not f.exists():
            co = C.cohort(cfg, list(idx), month)
            prof = C.profiles(cfg, co, cfg["preprocess"]["types"])
            prof = prof.merge(co[["user"]], on="user")
            grouped = prof.groupby("user")
            rows = []
            for u, repo, ts in co.itertuples(index=False):
                p = grouped.get_group(u) if u in grouped.groups else prof.iloc[:0]
                items = [idx[r] for r in p["repo_id"] if r in idx]
                rows.append(
                    {
                        "user": u,
                        "target": idx[repo],
                        "pr_ts": ts,
                        "profile": items,
                        "n_history": len(p),
                        "target_in_profile": idx[repo] in items,
                    }
                )
            pd.DataFrame(rows).to_parquet(f, index=False)
            log.info("%s cohort %s: %d users", split, month, len(rows))
        q = pd.read_parquet(f)
        res[split] = Queries(
            profiles=[np.asarray(p, dtype=np.int64) for p in q["profile"]],
            target=q["target"].to_numpy(),
            n_history=q["n_history"].to_numpy(),
            new_repo=~q["target_in_profile"].to_numpy(),
        )
    return res


def _z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-9)


def ladder_grid(
    scorer: Scorer,
    q: Queries,
    text: np.ndarray,
    lang: np.ndarray,
    owner: np.ndarray,
    batch: int = 1024,
) -> dict[tuple[int, float], dict[str, np.ndarray]]:
    """Serving-architecture scores for every (k, beta): w*z(model) + (1-w)*z(text profile) +
    beta*language match, w = min(1, n/k); n = catalog items in the profile."""
    out: dict[tuple[int, float], list[dict[str, np.ndarray]]] = {}
    for s in range(0, len(q.target), batch):
        prof = q.profiles[s : s + batch]
        zm = _z(scorer.scores(prof))
        tv = np.stack([text[p].mean(0) if len(p) else np.zeros(text.shape[1]) for p in prof])
        zt = _z(tv @ text.T + scorer.tie)
        langs = [set(lang[p]) - {""} for p in prof]
        lm = np.stack([np.isin(lang, list(ls)) for ls in langs]).astype(float)
        n = np.array([len(p) for p in prof], dtype=float)
        for k, beta in itertools.product(LADDER_K, LADDER_BETA):
            w = np.minimum(1.0, n / k)[:, None]
            sc = w * zm + (1 - w) * zt + beta * lm + scorer.tie
            out.setdefault((k, beta), []).append(
                rank_metrics(sc, q.target[s : s + batch], lang, owner)
            )
    return {c: {m: np.concatenate([p[m] for p in v]) for m in v[0]} for c, v in out.items()}


def run(cfg: dict[str, Any], out: Path) -> None:
    data_dir = Path(cfg["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    queries = build_queries(cfg, data_dir, out)
    data = InteractionData.from_processed(data_dir, cfg["head_fraction"])
    item_map = pd.read_parquet(data_dir / "item_map.parquet").sort_values("item")
    details = pd.read_parquet(Path(cfg["github_dir"]) / "details.parquet").set_index("repo_id")
    lang = np.array(
        [
            details["language"].get(r) or "" if r in details.index else ""
            for r in item_map["raw_id"]
        ],
        dtype=object,
    )
    lang = np.array([x if isinstance(x, str) else "" for x in lang], dtype=object)
    owner = np.array([n.split("/")[0].lower() for n in item_map["name"]], dtype=object)
    text = text_matrix(data, "items.parquet", "item", data.n_items, DEFAULT_ENCODER)
    for name in MODELS:
        for seed in [0] if name == "popularity" else SEEDS:
            done = all((out / f"{name}_s{seed}_{m}.npz").exists() for m in MONTHS.values())
            if done and name != "sasrec_text_mlp_warm":
                continue
            if done and (out / f"ladder_s{seed}_{MONTHS['report']}.npz").exists():
                continue
            log.info("replay: training %s seed %d", name, seed)
            scorer = Scorer(name, data, seed)
            for month in MONTHS.values():
                q = queries["tune" if month == MONTHS["tune"] else "report"]
                res = evaluate(scorer.scores, q, lang, owner)
                np.savez(out / f"{name}_s{seed}_{month}.npz", **res)
                log.info(
                    "%s s%d %s: hit@20 %.4f ndcg@20 %.4f",
                    name,
                    seed,
                    month,
                    res["hit"].mean(),
                    res["ndcg"].mean(),
                )
                if name == "sasrec_text_mlp_warm":
                    grid = ladder_grid(scorer, q, text, lang, owner)
                    np.savez(
                        out / f"ladder_s{seed}_{month}.npz",
                        **{f"k{k}_b{b}_{m}": v[m] for (k, b), v in grid.items() for m in v},
                    )
