"""Replay metrics per query user: was the actual first-PR repo in the top K?

- hit@K (= Recall@K with one target) and NDCG@K (1 / log2(rank + 2) for a hit at rank 0..K-1).
- near-misses (misses only): the target is not in the top K but a repo with its primary
  language / of its owner (org) is.
- unmasked ranking: repos already in the profile stay rankable (the app shows starred repos;
  a first PR often follows a star). `new_repo` marks queries whose target is not in the profile.
"""

from dataclasses import dataclass

import numpy as np

HISTORY_BINS = [(0, 0, "0"), (1, 5, "1-5"), (6, 20, "6-20"), (21, 10**9, "20+")]


def history_bin(n: int) -> str:
    return next(label for lo, hi, label in HISTORY_BINS if lo <= n <= hi)


@dataclass
class Queries:
    profiles: list[np.ndarray]  # catalog item indices before the PR, oldest first
    target: np.ndarray  # catalog item index of the first-PR repo
    n_history: np.ndarray  # profile size over the whole repo set (catalog or not)
    new_repo: np.ndarray  # target not in the profile


def rank_metrics(
    scores: np.ndarray, target: np.ndarray, lang: np.ndarray, owner: np.ndarray, k: int = 20
) -> dict[str, np.ndarray]:
    top = np.argpartition(-scores, k, axis=1)[:, :k]
    order = np.take_along_axis(scores, top, 1).argsort(1)[:, ::-1]
    top = np.take_along_axis(top, order, 1)
    pos = np.where(top == target[:, None], np.arange(k)[None, :], -1).max(1)
    hit = pos >= 0
    ndcg = np.where(hit, 1.0 / np.log2(np.maximum(pos, 0) + 2), 0.0)
    # near-misses: the target is not in the top K, but a repo in its language / of its owner is
    near_lang = ~hit & (lang[top] == lang[target][:, None]).any(1) & (lang[target] != "")
    near_org = ~hit & (owner[top] == owner[target][:, None]).any(1)
    return {
        "hit": hit.astype(float),
        "ndcg": ndcg,
        "near_lang": near_lang.astype(float),
        "near_org": near_org.astype(float),
    }


def evaluate(
    score_fn, q: Queries, lang: np.ndarray, owner: np.ndarray, k: int = 20, batch: int = 1024
) -> dict[str, np.ndarray]:
    parts = []
    for s in range(0, len(q.target), batch):
        sl = slice(s, s + batch)
        parts.append(rank_metrics(score_fn(q.profiles[sl]), q.target[sl], lang, owner, k))
    return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
