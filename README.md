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

## Phase 3 results: graph models on MovieLens-1M

LightGCN is my own implementation; NGCF and a second LightGCN run through RecBole via the same
bridge, as a reference check. Same split, evaluator and tuning protocol as before. Test set, full
ranking, mean ± std over 3 seeds (`results/leaderboard.csv`, `results/bootstrap_cis.csv`):

| Model | Recall@20 | NDCG@20 | Coverage@20 | Long-tail share | Tail Recall@20 |
|---|---|---|---|---|---|
| MF-BPR (phase 1) | 0.1187 ± 0.0027 | 0.0772 ± 0.0012 | 0.520 | 0.086 | 0.0202 |
| NGCF (RecBole) | 0.1124 ± 0.0009 | 0.0723 ± 0.0011 | 0.559 | 0.110 | 0.0227 |
| LightGCN (mine, 3 layers) | 0.1219 ± 0.0008 | 0.0781 ± 0.0004 | 0.559 | 0.094 | 0.0230 |
| LightGCN (RecBole) | 0.1232 ± 0.0030 | 0.0799 ± 0.0012 | 0.508 | 0.079 | 0.0233 |
| SASRec (phase 2) | 0.1766 ± 0.0019 | 0.1095 ± 0.0015 | 0.799 | 0.262 | 0.0711 |

| Comparison | Δ NDCG@20 (paired bootstrap, 95% CI) |
|---|---|
| LightGCN − MF-BPR | +0.0009 [−0.0005, +0.0023] |
| LightGCN − NGCF | +0.0058 [+0.0043, +0.0074] |
| MF-BPR − NGCF | +0.0049 [+0.0033, +0.0064] |
| RecBole LightGCN − my LightGCN | +0.0018 [+0.0003, +0.0031] |
| LightGCN 3 layers − 1 layer | +0.0021 [+0.0011, +0.0030] |
| LightGCN 3 layers − 2 layers | +0.0005 [−0.0003, +0.0012] |
| LightGCN 4 layers − 3 layers | +0.0002 [−0.0005, +0.0008] |
| LightGCN, init std 0.1 − 0.01 | +0.0004 [−0.0008, +0.0016] |
| MF-BPR, init std 0.1 − 0.01 | +0.0003 [−0.0008, +0.0014] |

NDCG@20 by user activity (quartiles of train history length; `results/analysis/`):

| Users with … train items | 3–20 | 21–45 | 46–98 | 99–1133 |
|---|---|---|---|---|
| MF-BPR | 0.0930 | 0.0788 | 0.0641 | 0.0737 |
| NGCF | 0.0858 | 0.0752 | 0.0611 | 0.0677 |
| LightGCN | 0.0994 | 0.0793 | 0.0629 | 0.0719 |
| LightGCN − MF-BPR (95% CI) | +0.0065 [+0.0023, +0.0107] | +0.0005 [−0.0021, +0.0033] | −0.0012 [−0.0031, +0.0009] | −0.0019 [−0.0037, −0.0002] |
| SASRec | 0.1670 | 0.1317 | 0.0850 | 0.0569 |
| SASRec with val items as context | 0.2429 | 0.2241 | 0.1986 | 0.1664 |

What I take from it:

- **Removing NGCF's feature transforms and non-linearities helps**, as the LightGCN paper argues:
  NGCF is significantly below both LightGCN and plain MF-BPR overall, and below MF-BPR in every
  activity group.
- **Graph propagation helps sparse users, not everyone.** Overall LightGCN ties MF-BPR, but for
  the quarter of users with at most 20 train items it is clearly better (+7% NDCG@20), and for the
  most active quarter it is slightly worse. Averaged over all users, the two effects roughly cancel.
- **Depth saturates at 2 layers.** One layer is significantly worse than three; two, three and four
  layers are tied.
- **LightGCN starts out as a popularity model.** At a small learning rate it sat at Popularity's
  validation score for 20+ epochs before it began to personalise, both in my implementation and in
  RecBole's, so early stopping with patience 10 stopped it there. I added a minimum number of
  epochs before early stopping can fire.
- **RecBole's LightGCN lands slightly above mine,** the same small gap as RecBole BPR vs my MF-BPR.
  Initialisation is not the cause: changing the init std from 0.1 to 0.01 changes neither model.
- **SASRec's weakness on very active users is stale context.** Scored from train only, SASRec is
  below Popularity for the most active quarter. Those users have long validation periods (median 20
  items) between the end of the input sequence and the first test item. With the validation items
  as context, their NDCG@20 almost triples.
