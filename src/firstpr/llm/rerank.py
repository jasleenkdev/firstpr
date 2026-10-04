"""LLM listwise reranking (Hou et al., "Large Language Models are Zero-Shot Rankers for
Recommender Systems", ECIR 2024).

The LLM sees the user's past products (titles, oldest first) and 20 candidates, and returns the
candidates in ranked order as bracketed numbers, e.g. "[3] > [12] > ...". Variants:
- Q3 (text only): candidate titles only;
- Q2 (text + CRM): each candidate also shows the conventional recommender's score;
and two candidate sets: LightGCN's top-20 (hard, realistic) vs the held-out item + 19 random
items (easy: what most LLM-ranking papers report).

Position bias: LLMs prefer candidates near the start of the list. Each (user, variant) is asked
`n_shuffles` times with the candidates in different random orders; rankings are combined with a
Borda count (Hou et al. bootstrap the order the same way).
"""

import re
from typing import Any

import numpy as np

from firstpr.llm.kar import truncate

PROMPT = """I've purchased the following products in the past, in order:
{history}

Now there are {n} candidate products that I might buy next:
{candidates}

Rank all {n} candidates by how likely I am to buy them next, most likely first. Answer only \
with the candidate numbers in brackets separated by " > ", for example: [3] > [1] > [2]. \
Include every number exactly once."""


def format_prompt(
    history_titles: list[str],
    cand_titles: list[str],
    cand_scores: list[float] | None,
    max_title_chars: int,
) -> str:
    hist = "\n".join(
        f"{i + 1}. {truncate(t, max_title_chars)}" for i, t in enumerate(history_titles)
    )
    lines = []
    for i, t in enumerate(cand_titles):
        line = f"[{i + 1}] {truncate(t, max_title_chars)}"
        if cand_scores is not None:
            line += f" (recommender score: {cand_scores[i]:.2f})"
        lines.append(line)
    return PROMPT.format(history=hist, n=len(cand_titles), candidates="\n".join(lines))


def parse_ranking(text: str, n: int) -> tuple[list[int], int]:
    """0-based positions in the order the LLM ranked them, deduplicated and range-checked, then
    the missing positions in presented order. Also returns how many were missing."""
    seen: list[int] = []
    for m in re.findall(r"\[(\d+)\]", text):
        k = int(m) - 1
        if 0 <= k < n and k not in seen:
            seen.append(k)
    missing = [k for k in range(n) if k not in seen]
    return seen + missing, len(missing)


def borda(rankings: list[list[int]], n: int) -> np.ndarray:
    """Candidate indices ordered by total Borda points (n - position); ties by first ranking."""
    points = np.zeros(n)
    for r in rankings:
        for pos, c in enumerate(r):
            points[c] += n - pos
    first = np.empty(n)
    first[np.asarray(rankings[0])] = np.arange(n)
    return np.lexsort((first, -points))


def scaled_scores(scores: np.ndarray) -> np.ndarray:
    """Recommender scores of the candidates min-max scaled to [0, 1] for the prompt."""
    lo, hi = scores.min(), scores.max()
    return np.zeros_like(scores) if hi == lo else (scores - lo) / (hi - lo)


def rerank_user(
    generate: Any,
    history_titles: list[str],
    cand_items: np.ndarray,
    titles: list[str],
    cand_scores: np.ndarray | None,
    n_shuffles: int,
    rng: np.random.Generator,
    max_title_chars: int,
) -> tuple[np.ndarray, int]:
    """Reranked candidate items (Borda over shuffles) and the number of missing ids in total."""
    n = len(cand_items)
    rankings, n_missing = [], 0
    for _ in range(n_shuffles):
        order = rng.permutation(n)  # presented position -> candidate index
        prompt = format_prompt(
            history_titles,
            [titles[cand_items[c]] for c in order],
            None if cand_scores is None else [float(cand_scores[c]) for c in order],
            max_title_chars,
        )
        ranked_positions, miss = parse_ranking(generate(prompt), n)
        rankings.append([int(order[p]) for p in ranked_positions])
        n_missing += miss
    return cand_items[borda(rankings, n)], n_missing
