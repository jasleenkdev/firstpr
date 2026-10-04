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

## Phase 2 results: deep and sequential models on MovieLens-1M

Same split, evaluator and tuning protocol as phase 1 (val-only grid search, extended while the best
configuration sat on a grid edge; 3 seeds on test). Test set, full ranking, mean ± std over 3 seeds
(`results/leaderboard.csv`):

| Model | Recall@20 | NDCG@20 | Coverage@20 | Long-tail share | Tail Recall@20 |
|---|---|---|---|---|---|
| Popularity | 0.0777 ± 0.0000 | 0.0562 ± 0.0000 | 0.051 | 0.000 | 0.0000 |
| ItemKNN | 0.1145 ± 0.0000 | 0.0746 ± 0.0000 | 0.433 | 0.044 | 0.0146 |
| MF-BPR | 0.1187 ± 0.0027 | 0.0772 ± 0.0012 | 0.520 | 0.086 | 0.0202 |
| NCF-GMF | 0.1220 ± 0.0004 | 0.0788 ± 0.0005 | 0.483 | 0.079 | 0.0184 |
| NCF-MLP | 0.1104 ± 0.0020 | 0.0735 ± 0.0010 | 0.500 | 0.068 | 0.0135 |
| NeuMF (pretrained) | 0.1187 ± 0.0008 | 0.0774 ± 0.0005 | 0.575 | 0.098 | 0.0194 |
| Two-tower (logQ) | 0.1197 ± 0.0028 | 0.0773 ± 0.0014 | 0.578 | 0.093 | 0.0214 |
| RecBole BPR (via my bridge) | 0.1212 ± 0.0026 | 0.0794 ± 0.0004 | 0.477 | 0.059 | 0.0137 |
| **SASRec** | **0.1766 ± 0.0019** | **0.1095 ± 0.0015** | 0.799 | 0.262 | 0.0711 |

Ablations (same tuned configs, one change each):

| Ablation | NDCG@20 | vs the full model (paired bootstrap, 95% CI) |
|---|---|---|
| Two-tower without logQ correction | 0.0400 ± 0.0008 | −0.0373 [−0.0402, −0.0346] |
| SASRec with the paper's BCE loss | 0.0919 ± 0.0004 | −0.0176 [−0.0198, −0.0152] |
| SASRec on shuffled histories | 0.0719 ± 0.0006 | −0.0376 [−0.0422, −0.0333] |
| ItemKNN scored with val items as context (no retraining) | 0.0889 | +0.0143 vs ItemKNN; +0.0117 [+0.0089, +0.0145] vs MF-BPR |
| SASRec scored with val items as context (no retraining) | 0.2088 ± 0.0012 | +0.0993 vs SASRec |

What I take from it:

- **Order is the big lever on this dataset.** SASRec improves NDCG@20 by 42% over MF-BPR, and
  all of that disappears when each user's history is shuffled: shuffled SASRec (0.0719) ends up
  slightly below MF-BPR. That holds even though 66%
  of ML-1M interactions share a timestamp with another of the same user. A model-free check agrees:
  ItemKNN scored from only a user's last 5 items does much better on validation than from the full
  history.
- **A learned MLP similarity does not beat a dot product.** NCF-MLP is the weakest learned model
  (−0.0037 vs MF-BPR, CI [−0.0055, −0.0019]). GMF, NeuMF, two-tower and MF-BPR all lie within
  0.0016 NDCG@20 of each other; of those pairs, only GMF vs NeuMF has a CI that excludes zero, and
  only just. That matches Rendle et al. (2020).
- **Without logQ correction, in-batch softmax falls below Popularity.** It recommends long-tail items
  in 65% of slots and collapses on a popularity-heavy dataset.
- **The full-softmax loss matters for SASRec** (+19% NDCG@20 over the original BCE with one negative).
- **Up-to-date context beats a better model class.** Using the newest interactions at scoring time,
  without retraining, helps ItemKNN overtake MF-BPR and nearly doubles SASRec.
- **RecBole's BPR, run through my bridge on my split and evaluator, lands in the same range as my
  MF-BPR** (slightly higher: +0.0022, CI [+0.0008, +0.0036]). That validates the bridge and suggests
  my MF-BPR still has a little tuning headroom.
