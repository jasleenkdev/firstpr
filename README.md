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
