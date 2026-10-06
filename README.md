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

## Phase 4 results: text and LLMs on Amazon Reviews

Data: Amazon Reviews 2023 (Hou et al., 2024), Industrial_and_Scientific, interactions from 2020
on, rating ≥ 4, 5-core. That leaves 6,192 users × 5,105 items and 61,088 interactions, with a
median of 7 per user. It is much sparser than MovieLens-1M. The split, evaluator and tuning
protocol are the same as before. 303 test targets fall on 68 items that have no train
interactions ("cold items"). Item text is the title, brand, category, up to three feature
bullets and a short description, encoded once by a frozen `BAAI/bge-small-en-v1.5`. Every LLM
call goes to a local `llama3.2:3b` through Ollama and is cached on disk.

```bash
uv run python scripts/prepare_data.py --config configs/data/amazon_sci.yaml
ollama pull llama3.2:3b
uv run python scripts/kar_generate.py --data amazon_sci          # ~11k cached LLM calls
uv run python scripts/run_experiment.py --model sasrec_text_mlp --data amazon_sci --mode val
uv run python scripts/llm_rerank.py --data amazon_sci           # 1,800 cached LLM calls
uv run python scripts/cold_items.py && uv run python scripts/user_groups.py --data amazon_sci --n-groups 3
```

Test set, full ranking, mean ± std over 3 seeds:

| Model | Recall@20 | NDCG@20 | Tail Recall@20 | Cold NDCG@20 | Coverage@20 |
|---|---|---|---|---|---|
| Popularity | 0.0422 | 0.0182 | 0.000 | 0.0000 | 0.005 |
| MF-BPR | 0.0612 ± 0.0018 | 0.0259 ± 0.0007 | 0.012 | 0.0000 | 0.83 |
| ItemKNN | 0.0616 | 0.0290 | 0.016 | 0.0000 | 0.96 |
| LightGCN | 0.0693 ± 0.0017 | 0.0295 ± 0.0005 | 0.009 | 0.0000 | 0.52 |
| SASRec (ID embeddings) | 0.0835 ± 0.0013 | 0.0368 ± 0.0007 | 0.029 | 0.0000 | 0.57 |
| SASRec + text, linear adapter | 0.0700 ± 0.0016 | 0.0312 ± 0.0007 | 0.037 | 0.0040 | 0.77 |
| SASRec + text, linear adapter + ID | 0.0864 ± 0.0016 | 0.0377 ± 0.0010 | 0.032 | 0.0000 | 0.68 |
| SASRec + text, MLP adapter | **0.0953 ± 0.0028** | **0.0423 ± 0.0015** | 0.036 | 0.0013 | 0.62 |
| SASRec + text, MLP adapter + ID | 0.0857 ± 0.0039 | 0.0377 ± 0.0014 | 0.032 | 0.0000 | 0.62 |
| SASRec + text, MoE-whitening adapter | 0.0908 ± 0.0028 | 0.0402 ± 0.0013 | 0.039 | 0.0018 | 0.74 |
| SASRec + text, MoE-whitening adapter + ID | 0.0868 ± 0.0014 | 0.0379 ± 0.0013 | 0.033 | 0.0000 | 0.67 |
| KAR item knowledge (on linear + ID) | 0.0878 ± 0.0017 | 0.0382 ± 0.0011 | 0.036 | 0.0000 | 0.70 |
| KAR item knowledge + user preference | 0.0886 ± 0.0027 | 0.0389 ± 0.0009 | 0.037 | 0.0000 | 0.55 |
| MoE adapter, warm-item training softmax | 0.0919 ± 0.0040 | 0.0401 ± 0.0014 | 0.042 | **0.0087** | 0.75 |

A uniform random ranking scores 0.0072 cold NDCG@20.

| Comparison | Δ NDCG@20 (paired bootstrap, 95% CI) |
|---|---|
| LightGCN − MF-BPR | +0.0035 [+0.0017, +0.0053] |
| LightGCN − ItemKNN | +0.0004 [−0.0021, +0.0028] |
| SASRec − LightGCN | +0.0073 [+0.0050, +0.0098] |
| text MLP − SASRec | +0.0055 [+0.0030, +0.0079] |
| text MoE − SASRec | +0.0034 [+0.0011, +0.0059] |
| text MLP − text MLP + ID | +0.0046 [+0.0023, +0.0069] |
| text MLP − text MoE | +0.0021 [+0.0003, +0.0039] |
| text MoE − text linear | +0.0090 [+0.0068, +0.0111] |
| KAR (item + user) − its base | +0.0012 [−0.0005, +0.0027] |
| KAR item only − its base | +0.0005 [−0.0008, +0.0018] |
| cold items: MoE warm softmax − MoE | +0.0069 [+0.0033, +0.0111] (287 users) |
| overall: MoE − MoE warm softmax | +0.0001 [−0.0009, +0.0010] |

NDCG@20 by user activity (thirds of train history length; quartiles collapse because 28% of
users have exactly 3 train items):

| Users with … train items | 3 | 4–6 | 7–116 |
|---|---|---|---|
| MF-BPR | 0.0300 | 0.0299 | 0.0184 |
| ItemKNN | 0.0307 | 0.0333 | 0.0232 |
| LightGCN | 0.0359 | 0.0350 | 0.0185 |
| LightGCN − MF-BPR (95% CI) | +0.0059 [+0.0021, +0.0101] | +0.0051 [+0.0015, +0.0083] | +0.0000 [−0.0022, +0.0023] |
| SASRec | 0.0358 | 0.0407 | 0.0335 |
| SASRec − LightGCN (95% CI) | −0.0001 [−0.0045, +0.0039] | +0.0057 [+0.0019, +0.0095] | +0.0150 [+0.0111, +0.0190] |
| text MLP | 0.0400 | 0.0472 | 0.0389 |

LLM listwise reranking (Hou et al., ECIR 2024), 200 fixed random test users. The LLM sees the
user's last 20 product titles and 20 candidates, and returns a ranking. Each list is shown in 3
shuffled orders and combined with a Borda count, to cancel position bias. "Random-20" = one
held-out item + 19 random items (the setup most LLM-ranking papers report). "LightGCN-20" =
LightGCN's real top-20 from train-only context. NDCG@10 on the reranked list:

| Candidates | Ranker | NDCG@10 | Δ vs reference (95% CI) |
|---|---|---|---|
| random-20 | random order (analytic) | 0.227 | |
| random-20 | popularity order | 0.304 | |
| random-20 | LightGCN order | 0.529 | |
| random-20 | LLM, titles only | 0.309 | vs popularity +0.005 [−0.059, +0.061]; vs LightGCN −0.220 [−0.278, −0.163] |
| LightGCN-20 | LightGCN order | 0.0295 | |
| LightGCN-20 | popularity order | 0.0201 | |
| LightGCN-20 | LLM, titles only (Q3) | 0.0130 | vs LightGCN −0.0166 [−0.0297, −0.0067] |
| LightGCN-20 | LLM, titles + LightGCN scores (Q2) | 0.0225 | vs LightGCN −0.0070 [−0.0199, +0.0041]; vs Q3 +0.0095 [−0.0050, +0.0273] |

The LLM left out 3.1 of the 20 candidate numbers per answer in Q3 and 0.3 in Q2. Missing
candidates are appended in the order shown, so a malformed answer costs the LLM, not the
baseline. 1,800 uncached calls to a local `llama3.2:3b`, about 4 hours on a laptop.

What I take from it:

- **Item text helps most when it stands alone.** The best model is SASRec over frozen text
  embeddings with an MLP adapter: +15% NDCG@20 over ID-based SASRec. With a median of 7
  interactions per user, an item's ID embedding gets very few updates, while its text vector is
  informative from the start.
- **Validation chose the wrong variant.** On validation the variants with an added ID embedding
  won. On test the text-only MLP and MoE models beat every ID variant. Selection stays on
  validation only, and KAR was attached to the validation winner (linear + ID). So KAR's base is
  not the best test model. I report that rather than re-pick after seeing test.
- **Adapter capacity matters more than whitening here.** A single linear map is far behind. The
  MLP edges out UniSRec's MoE with parametric whitening, which is built for transfer across
  domains, not for one domain.
- **Cold-start results depend on the training softmax.** With the whole catalogue in the softmax,
  items that never appear as a target are pushed down at every step. Every model, text models
  included, ranked cold items *below random*. When cold items are left out of the training softmax
  (a real system never trained on an item that arrives later), the text MoE model's cold NDCG@20
  rises from 0.0018 to 0.0087, above random, with no cost overall. Models with an ID embedding
  stay near zero on cold items.
- **KAR with a small local LLM adds nothing measurable.** LLM-written item knowledge and user
  preference summaries, encoded and added through expert adapters, gave +0.0012 (CI includes 0).
  The frozen encoder already reads the listing. A 3B model's "knowledge" mostly paraphrases it.
- **Graph propagation helps sparse users, again.** On this sparse data LightGCN beats MF-BPR
  overall. The whole gain comes from users with at most 6 train items. ItemKNN ties LightGCN.
- **Sequence models gain as history grows.** SASRec ties LightGCN for users with 3 items and
  leads clearly for users with 7 or more.
- **A small zero-shot LLM is not a reranker.** On the easy random-20 setup, which most LLM-ranking
  papers report, the 3B model beats a random order but only matches popularity order, far below
  LightGCN. On LightGCN's real top-20 it makes LightGCN's ranking significantly worse. Given
  LightGCN's scores in the prompt (Q2 in Lin et al.'s survey), it mostly falls back to them and
  the loss shrinks to a tie. In this setup the LLM's value is not in ranking.
- **Sparse data needs heavy regularisation for ID models.** MF-BPR's best L2 is 30× its
  MovieLens value, and SASRec's best dropout is 0.8 (0.2 on MovieLens). The text models preferred
  low dropout: frozen text features already regularise.

## Phase 5 results: GitHub

Data: GH Archive events (BigQuery, March–October 2025) on 8,771 repositories, plus repository
text from the GitHub GraphQL API. The repositories are 5,271 "beginner" repos (a beginner-style
label such as *good first issue* used on an issue created in the training window), 3,000 repos
that share the most contributors with them, and 500 repos created after the training cutoff.
An interaction is a user's first star, fork, pull request, issue, comment or review on a repo.
Events by people who pushed to the repo (maintainers) are dropped, so pull-request and issue
actors are outside contributors. On October 2025, where the event payload is affordable, 93% of
kept pull-request actors had opened a pull request themselves that month (82% for issues).
User ids are salted hashes computed before anything is written; logins never leave the query.

Split: one global cutoff. Train = March–August 2025, validation = September, test = October
(Hacktoberfest). The 5-core filter uses training data only. A seeded sample of 30,000 of the
521,591 qualifying users keeps CPU costs at MovieLens size: 29,554 users × 6,021 repos,
422,191 train / 33,888 validation / 20,081 test interactions, 91% of them stars. 402 repos
created after the cutoff carry 18% of test targets ("cold items"). Repo text is the name,
language, description and the README as of the last commit before the cutoff.

```bash
uv run python scripts/gh_scope.py                 # BigQuery activity + GraphQL label checks
uv run python scripts/gh_ingest.py                # 245 daily batches, idempotent
uv run python scripts/gh_fetch.py                 # README at the cutoff, open beginner issues
uv run python scripts/prepare_data.py --config configs/data/github.yaml
uv run python scripts/run_experiment.py --model lightgcn --data github --mode val
uv run python scripts/run_experiment.py --model lightgcn --data github --variant contrib --mode test
```

Tuning: log-spaced grids over about three orders of magnitude, then the four half-step
neighbours of the best configuration (validation only, repeated once when the best moved by
≥ 1%). Test set, full ranking, mean ± std over 3 seeds:

| Model | Recall@20 | NDCG@20 | Tail Recall@20 | Cold NDCG@20 | Coverage@20 |
|---|---|---|---|---|---|
| Popularity | 0.0300 | 0.0133 | 0.000 | 0.0000 | 0.008 |
| ItemKNN | 0.0531 | 0.0237 | 0.012 | 0.0000 | 0.51 |
| MF-BPR | 0.0582 ± 0.0020 | 0.0267 ± 0.0006 | 0.009 | 0.0000 | 0.45 |
| LightGCN | 0.0631 ± 0.0005 | 0.0284 ± 0.0003 | 0.012 | 0.0000 | 0.44 |
| SASRec | 0.0647 ± 0.0020 | 0.0286 ± 0.0007 | 0.023 | 0.0000 | 0.49 |
| SASRec + README text, MLP adapter | **0.0688 ± 0.0011** | **0.0311 ± 0.0004** | 0.013 | 0.0000 | 0.30 |
| same, warm-item training softmax | 0.0681 ± 0.0006 | 0.0307 ± 0.0004 | 0.013 | **0.0022** | 0.30 |

| Comparison | Δ NDCG@20 (paired bootstrap, 95% CI) |
|---|---|
| text MLP − SASRec | +0.0025 [+0.0013, +0.0038] |
| text MLP − LightGCN | +0.0026 [+0.0009, +0.0043] |
| SASRec − LightGCN | +0.0001 [−0.0019, +0.0019] |
| LightGCN − MF-BPR | +0.0017 [+0.0005, +0.0029] |
| MF-BPR − ItemKNN | +0.0030 [+0.0013, +0.0045] |
| ItemKNN − Popularity | +0.0105 [+0.0087, +0.0124] |
| cold items: warm softmax − full softmax | +0.0022 [+0.0016, +0.0030] (2,563 users) |

Contribution targets only (same training data; validation/test keep only pairs with a pull
request, issue, comment or review: 828 test targets):

| Model | NDCG@20 | Recall@20 |
|---|---|---|
| Popularity | 0.0112 | 0.033 |
| MF-BPR | 0.0437 ± 0.0023 | 0.097 |
| ItemKNN | 0.0439 | 0.096 |
| SASRec | 0.0454 ± 0.0062 | 0.110 |
| text MLP, warm softmax | 0.0464 ± 0.0051 | 0.107 |
| LightGCN | 0.0484 ± 0.0016 | 0.117 |
| text MLP | **0.0509 ± 0.0060** | 0.112 |

Every model beats Popularity by about 4×; among the learned models no difference is
significant (e.g. text MLP − LightGCN +0.0025 [−0.0054, +0.0103]).

NDCG@20 by user activity (quartiles of train history length):

| Users with … train repos | 5–6 | 7–12 | 13–23 | 24+ |
|---|---|---|---|---|
| MF-BPR | 0.0309 | 0.0299 | 0.0270 | 0.0196 |
| LightGCN | 0.0363 | 0.0330 | 0.0259 | 0.0198 |
| LightGCN − MF-BPR (95% CI) | +0.0054 [+0.0019, +0.0090] | +0.0030 [+0.0007, +0.0055] | −0.0011 [−0.0032, +0.0010] | +0.0003 [−0.0011, +0.0017] |
| SASRec | 0.0361 | 0.0317 | 0.0296 | 0.0186 |
| text MLP | 0.0375 | 0.0361 | 0.0305 | 0.0210 |

What I take from it:

- **README text helps, again.** SASRec over frozen README embeddings is the best model overall
  (+9% NDCG@20 over ID-based SASRec, CI excludes 0), as it was on Amazon.
- **The sequence model's validation lead did not survive the month boundary.** On validation
  (September) SASRec was 17% above LightGCN; on test (October) they tie. With a global split,
  validation is one month and test the next, and October is Hacktoberfest: what users did last
  month predicts less than on MovieLens, where SASRec won by 40%.
- **Graph propagation helps sparse users, for the third dataset in a row.** LightGCN beats MF-BPR
  only for users with ≤ 12 train repos.
- **New repos are almost unreachable.** No ID model can rank a repo that did not exist in
  training, and the text model with a full-catalogue softmax scores exactly 0 on them. A
  warm-only training softmax lifts that to 0.0022 (cold Recall@20 0.0073, about 2.2× a random
  ranking's 0.0033), at no significant overall cost. October's new repos were mostly hype-driven; a README predicts poorly who will star them.
- **ItemKNN's shrinkage matters a little more than on MovieLens, much less than on Amazon.** The
  best shrinkage beats none by 2.7% on validation (MovieLens 0.5%, Amazon 53%). What drives it
  is co-occurrence support per item pair, not popularity skew: star co-occurrences on GitHub are
  large counts.
- **For contribution targets the models tie.** Contributions are 4% of interactions (828 test
  targets), so the confidence intervals are wide. Every model is ~4× Popularity, so the
  collaborative signal from stars does transfer to where people contribute.

LLM features for the app (Groq free tier, cached): README summary and skills for 515 beginner
repos (`gpt-oss-20b`) and difficulty, skills and a quoted evidence span for 1,154 open beginner
issues (`gpt-oss-120b`). 99% of repo skills and 82% of issue skills are named in the source text;
82% of evidence quotes are copied from the issue. A manual check of 30 outputs found 3 grounding
errors (10%): an inverted fact, a wrong currency and an invented skill.

## Phase 6: the FirstPR app

Live: **https://firstpr-pied.vercel.app** (API: https://firstpr-api.vercel.app/health).

Two ways in:

- **I have GitHub.** Enter a username. The API reads that account's public stars once (the
  username is hashed immediately and never stored or logged) and feeds the stars that are in the
  catalog to the phase-5 text-SASRec model. Because it reads README text, the model can score repos
  it never saw during training and users who were never in the training data. With fewer than three
  catalog stars it blends in a text profile built from the stars' languages and topics.
- **New to GitHub.** Pick languages, interests and hours per week. Repos are matched through
  README embeddings of the chosen interests, language and skill overlap.

Each answer has about 10 repositories, each with up to 3 open beginner issues. Repos come from
star signals; issues are ranked by difficulty match (hours per week → target difficulty), skill
match, recency, whether someone is already assigned, and how active outside contributors are in
the repo. A "New projects" section surfaces repositories created in the last six months with
fresh beginner issues, which collaborative models cannot reach. "Why this?" asks an LLM to
rephrase facts the server collects itself (README summary, labels, matching skills, repos you
starred that share stargazers); if the LLM is unavailable a template built from the same facts
is shown.

| Piece | Where | Notes |
|---|---|---|
| Model | `src/firstpr/serve/build.py` | text-SASRec, MLP adapter, warm-only softmax, retrained on all GitHub data; encoder re-implemented in numpy (no torch at serving time) |
| API | `src/firstpr/serve/`, `services/api/` | FastAPI on Vercel serverless; 18 MB of artifacts; daily data pulled hourly from a Hugging Face dataset |
| Frontend | `web/` | Next.js on Vercel |
| Refresh | `.github/workflows/refresh.yml` | daily: open issues, new repos, LLM features within the free tier |

Measured on the live API (`scripts/measure_latency.py`, 20 accounts, 30 onboarding requests):

| Stage | p50 | p95 |
|---|---|---|
| Read GitHub stars (first time) | 1,563 ms | 2,082 ms |
| Retrieve + rank (GitHub path) | 15 ms | 24 ms |
| Retrieve + rank (onboarding) | 44 ms | 52 ms |
| Explanation (LLM) | 241 ms | 525 ms |
| Whole GitHub request, first time / repeated | 2.05 s / 0.38 s | 2.54 s / 0.61 s |

Exact numpy search over 8,513 repos (0.1–0.3 ms) and 32,728 issues (1.4 ms) is 5–10× faster
than Qdrant's local mode with identical results, so the API uses numpy only
(`results/analysis/vector_search_phase6.csv`).
