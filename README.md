# FirstPR

FirstPR is my project to help CS students find a first open-source contribution: beginner-friendly
repositories and specific issues matched to their skills, interests and available time, with
explanations grounded in real signals.

It has two parts:

- **Evolution benchmark.** I implement recommender systems in roughly historical order
  (Popularity → ItemKNN → MF-BPR → NCF → two-tower → SASRec → NGCF → LightGCN → text adapters →
  LLM feature augmentation → LLM rerankers → conversational agent) and score all of them with one
  split and one evaluator, first on MovieLens-1M, then Amazon Reviews, then GitHub data.
- **FirstPR app.** The best pipeline from the benchmark, applied to GitHub repositories and issues.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11.

```bash
uv python install 3.11
uv sync
make test
```

## Reproducing phase 1 (MovieLens-1M baselines)

```bash
make data                       # download ML-1M (checksum-verified), build implicit data + split
make tune MODEL=itemknn         # grid search on validation
make tune MODEL=mf_bpr
make run MODEL=popularity       # final runs on test, 3 seeds
make run MODEL=itemknn
make run MODEL=mf_bpr
make leaderboard                # aggregate to results/leaderboard.csv
```

Evaluation protocol: per-user chronological 80/10/10 split, full ranking over all items with
previously seen items masked, Recall@20 and NDCG@20 as primary metrics, mean ± std over 3 seeds.

## Phase 1 results: MovieLens-1M

Data after preprocessing: 6,034 users, 3,125 items, 574,376 positive interactions (rating ≥ 4,
5-core), density 3.05%. The 20% most popular items hold 71.5% of training interactions.

Test set, full ranking, mean ± std over 3 seeds (`results/leaderboard.csv`):

| Model | Recall@20 | NDCG@20 | HitRate@20 | Coverage@20 | Long-tail share | Tail Recall@20 |
|---|---|---|---|---|---|---|
| Popularity | 0.0777 ± 0.0000 | 0.0562 ± 0.0000 | 0.3490 ± 0.0000 | 0.051 | 0.000 | 0.0000 |
| ItemKNN (k=50, shrink=10) | 0.1145 ± 0.0000 | 0.0746 ± 0.0000 | 0.4208 ± 0.0000 | 0.433 | 0.044 | 0.0146 |
| MF-BPR (dim 64, lr 5e-3, L2 1e-2) | 0.1187 ± 0.0027 | 0.0772 ± 0.0012 | 0.4364 ± 0.0072 | 0.520 | 0.086 | 0.0202 |

What I take from it:

- Personalisation matters, but Popularity is a strong floor on this dataset: ItemKNN and MF-BPR
  improve NDCG@20 by roughly 33–37% over it.
- MF-BPR only moved ahead of ItemKNN after tuning: my first grid picked a config at the edge of
  the grid, and that config tied ItemKNN (NDCG@20 0.0745). Extending the grid to stronger L2 gave
  the final config. Under-tuned baselines can change a conclusion.
- All three models find very few of the tail items users liked (tail recall ≤ 0.02 against
  ≥ 0.10 on head items). MF-BPR recommends tail items about twice as often as ItemKNN.
- Popularity and ItemKNN are deterministic, so their std over seeds is 0.

Robustness: 66% of ML-1M interactions share a timestamp with another interaction from the same
user, so the tie-break decides which items land in validation and test. Re-running the final
configs with two other tie-break seeds (`results/robustness_tiebreak.csv`) keeps the order
MF-BPR > ItemKNN > Popularity on NDCG@20 for every seed.
