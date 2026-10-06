"""Build-time (torch): serving model + static artifacts for the API.

Serving model = text-SASRec with the MLP adapter and a warm-only training softmax (phase 5: ties
the full softmax overall and is the only variant above zero on new repos), with its tuned GitHub
config, retrained on *all* GitHub interactions (train + val + test, Mar–Oct 2025) for a fixed
number of epochs (median best epoch of its three test runs; no validation set is left).

Static artifacts (`<out>/static/`):
- model.npz      encoder weights for `serve.encoder.SASRecEncoder`
- item_vecs.npy  adapter(bge(text)) for every catalog repo, float32 [n, 64] (inductive: repos
                 outside the training data get vectors too)
- text_emb.npy   bge-small text embeddings, float16 [n, 384] (onboarding / cold-start matching)
- repos.json     catalog metadata, index-aligned with the matrices
- topics.json + topic_emb.npy  onboarding interest options and their embeddings
- costar.json    top co-starred neighbours per training repo (for "why this?")
- meta.json      versions and provenance
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from firstpr.data.dataset import InteractionData
from firstpr.github.build import item_text
from firstpr.github.privacy import scrub_text
from firstpr.models.sasrec_text import TextSASRec
from firstpr.text.encoder import DEFAULT_ENCODER, cached_embeddings, encode_texts
from firstpr.utils.io import git_info, load_json, load_yaml
from firstpr.utils.logging import get_logger
from firstpr.utils.seed import set_seed

log = get_logger(__name__)

# onboarding interests: id, label, a descriptive phrase that is embedded (bge space) and keywords
# used for keyword matching on repos outside the embedded catalog
TOPICS: list[dict[str, Any]] = [
    {
        "id": "web",
        "label": "Web development",
        "kw": [
            "web",
            "frontend",
            "react",
            "vue",
            "css",
            "html",
            "nextjs",
            "backend",
            "http",
            "api",
        ],
    },
    {
        "id": "mobile",
        "label": "Mobile apps",
        "kw": ["android", "ios", "flutter", "mobile", "kotlin", "swift", "react-native"],
    },
    {
        "id": "ml",
        "label": "Machine learning",
        "kw": [
            "machine-learning",
            "deep-learning",
            "pytorch",
            "tensorflow",
            "model",
            "training",
            "neural",
        ],
    },
    {
        "id": "llm",
        "label": "LLMs and AI agents",
        "kw": ["llm", "agent", "agents", "gpt", "rag", "prompt", "ai", "chatbot"],
    },
    {
        "id": "data",
        "label": "Data science and analytics",
        "kw": ["data", "pandas", "analytics", "visualization", "jupyter", "dataframe", "sql"],
    },
    {
        "id": "devops",
        "label": "DevOps and cloud",
        "kw": [
            "kubernetes",
            "docker",
            "devops",
            "cloud",
            "terraform",
            "ci",
            "deployment",
            "infrastructure",
        ],
    },
    {
        "id": "security",
        "label": "Security",
        "kw": ["security", "vulnerability", "crypto", "auth", "privacy", "pentest"],
    },
    {
        "id": "tools",
        "label": "Developer tools and CLIs",
        "kw": ["cli", "terminal", "tool", "editor", "vscode", "neovim", "linter", "formatter"],
    },
    {
        "id": "games",
        "label": "Games and graphics",
        "kw": ["game", "games", "graphics", "engine", "opengl", "emulator", "3d"],
    },
    {
        "id": "docs",
        "label": "Documentation and translation",
        "kw": ["documentation", "docs", "translation", "i18n", "localization", "tutorial"],
    },
    {
        "id": "education",
        "label": "Learning resources",
        "kw": ["learning", "course", "tutorial", "interview", "roadmap", "awesome", "education"],
    },
    {
        "id": "systems",
        "label": "Systems and performance",
        "kw": ["rust", "compiler", "kernel", "database", "performance", "systems", "runtime"],
    },
    {
        "id": "science",
        "label": "Scientific computing",
        "kw": [
            "scientific",
            "physics",
            "biology",
            "bioinformatics",
            "math",
            "simulation",
            "research",
        ],
    },
    {
        "id": "selfhost",
        "label": "Self-hosted apps",
        "kw": ["self-hosted", "selfhosted", "home-assistant", "server", "dashboard", "homelab"],
    },
    {
        "id": "design",
        "label": "UI and design systems",
        "kw": ["ui", "design", "components", "accessibility", "theme", "icons"],
    },
    {
        "id": "web3",
        "label": "Blockchain",
        "kw": ["blockchain", "ethereum", "solidity", "web3", "crypto"],
    },
]


def topic_phrase(t: dict[str, Any]) -> str:
    return f"An open-source project about {t['label'].lower()}: {', '.join(t['kw'])}."


def _all_interactions(processed: Path) -> InteractionData:
    """InteractionData whose train part holds every GitHub interaction (train + val + test)."""
    stats = load_json(processed / "stats.json")
    frames = [pd.read_parquet(processed / f"{p}.parquet") for p in ("train", "val", "test")]
    allf = pd.concat(frames, ignore_index=True)
    data = InteractionData.from_frames(
        allf, allf.iloc[:0], allf.iloc[:0], stats["n_users"], stats["n_items"], name="github_all"
    )
    data.processed_dir = str(processed)
    return data


def export_encoder(module: torch.nn.Module) -> dict[str, np.ndarray]:
    """SASRecModule weights in the layout `SASRecEncoder` expects (one attention head)."""
    if module.heads != 1:
        raise ValueError("the numpy encoder implements one attention head")
    t = lambda x: x.detach().cpu().numpy().astype(np.float32)  # noqa: E731
    w: dict[str, np.ndarray] = {
        "pos": t(module.pos_emb.weight),
        "out.w": t(module.out_norm.weight),
        "out.b": t(module.out_norm.bias),
        "n_blocks": np.array(len(module.attns)),
    }
    for i, (ln1, attn, ln2, ffn) in enumerate(
        zip(module.attn_norms, module.attns, module.ffn_norms, module.ffns, strict=True)
    ):
        w |= {
            f"b{i}.ln1.w": t(ln1.weight),
            f"b{i}.ln1.b": t(ln1.bias),
            f"b{i}.in.w": t(attn.in_proj_weight),
            f"b{i}.in.b": t(attn.in_proj_bias),
            f"b{i}.out.w": t(attn.out_proj.weight),
            f"b{i}.out.b": t(attn.out_proj.bias),
            f"b{i}.ln2.w": t(ln2.weight),
            f"b{i}.ln2.b": t(ln2.bias),
            f"b{i}.f1.w": t(ffn[0].weight),
            f"b{i}.f1.b": t(ffn[0].bias),
            f"b{i}.f2.w": t(ffn[3].weight),
            f"b{i}.f2.b": t(ffn[3].bias),
        }
    return w


def train_serving_model(processed: Path, epochs: int, seed: int = 0) -> tuple[TextSASRec, dict]:
    best = load_yaml("results/best/sasrec_text_mlp_github.yaml")["params"]
    config = {
        **best,
        "cold_negatives": False,
        "max_epochs": epochs,
        "patience": epochs,
        "seed": seed,
    }
    set_seed(seed)
    model = TextSASRec()
    info = model.fit(_all_interactions(processed), config, val_fn=None)
    model.module.eval()
    return model, {**config, "fit": {k: v for k, v in info.items() if k != "history"}}


def costar_neighbours(
    processed: Path, item_repo: np.ndarray, k: int = 10, shrink: float = 30.0
) -> dict:
    """Top-k co-starred repos per training repo (cosine with shrinkage on all interactions)."""
    data = _all_interactions(processed)
    x = data.train.astype(np.float32)
    co = (x.T @ x).toarray()
    np.fill_diagonal(co, 0)
    norm = np.sqrt(np.asarray(x.sum(0)).ravel())
    sim = co / (np.outer(norm, norm) + shrink)
    out = {}
    for i in range(sim.shape[0]):
        top = np.argsort(-sim[i])[:k]
        out[str(int(item_repo[i]))] = [
            [int(item_repo[j]), int(co[i, j])] for j in top if co[i, j] >= 2
        ]
    return out


def build_static(cfg: dict[str, Any], out: Path, epochs: int) -> None:
    processed, gdir = Path(cfg["processed_dir"]), Path(cfg["github_dir"])
    static = out / "static"
    static.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    details = pd.read_parquet(gdir / "details.parquet")
    details = details.loc[~details["is_archived"]].sort_values("repo_id").reset_index(drop=True)
    repo_set = pd.read_parquet(gdir / "scope" / "repo_set.parquet")[["repo_id", "group"]]
    act = (
        pd.concat(
            [
                pd.read_parquet(gdir / "scope" / f"activity_{w}.parquet")
                for w in ("train", "valtest")
            ]
        )
        .groupby("repo_id")[["n_contrib", "n_pr", "n_actors"]]
        .sum()
    )
    item_map = pd.read_parquet(processed / "item_map.parquet")
    training_ids = set(item_map["raw_id"])
    it = cfg["item_text"]
    texts = [
        item_text(r, it["readme_max_chars"], it["description_max_chars"])
        for _, r in details.iterrows()
    ]
    text_emb = cached_embeddings(texts, out / "cache" / "catalog", DEFAULT_ENCODER)

    model, mcfg = train_serving_model(processed, epochs)
    with torch.no_grad():
        item_vecs = model.module.item_emb.adapter(torch.from_numpy(text_emb)).numpy()
    np.savez(static / "model.npz", **export_encoder(model.module))
    np.save(static / "item_vecs.npy", item_vecs.astype(np.float32))
    np.save(static / "text_emb.npy", text_emb.astype(np.float16))

    groups = dict(zip(repo_set["repo_id"], repo_set["group"], strict=True))
    repos = []
    for r in details.itertuples():
        a = act.loc[r.repo_id] if r.repo_id in act.index else None
        repos.append(
            {
                "id": int(r.repo_id),
                "name": r.name,
                "description": scrub_text(r.description or "", 300),
                "language": r.language,
                "languages": list(r.languages),
                "topics": list(r.topics),
                "stars": int(r.stars_now),
                "created_at": r.created_at,
                "group": groups.get(r.repo_id),
                "trained": bool(r.repo_id in training_ids),
                "n_contrib": int(a["n_contrib"]) if a is not None else 0,
                "n_pr": int(a["n_pr"]) if a is not None else 0,
                "n_actors": int(a["n_actors"]) if a is not None else 0,
            }
        )
    (static / "repos.json").write_text(json.dumps(repos))

    topic_emb = encode_texts([topic_phrase(t) for t in TOPICS], DEFAULT_ENCODER)
    np.save(static / "topic_emb.npy", topic_emb.astype(np.float32))
    (static / "topics.json").write_text(json.dumps(TOPICS))
    costar = costar_neighbours(processed, item_map.sort_values("item")["raw_id"].to_numpy())
    (static / "costar.json").write_text(json.dumps(costar))
    meta = {
        "model": "text-SASRec (MLP adapter, warm-only softmax), numpy encoder",
        "config": mcfg,
        "encoder": DEFAULT_ENCODER,
        "data_window": [cfg["windows"]["train_start"], cfg["windows"]["test_end"]],
        "n_repos": len(repos),
        "n_training_repos": int(sum(r["trained"] for r in repos)),
        "git_commit": git_info()["commit"],
    }
    (static / "meta.json").write_text(json.dumps(meta, indent=2))
    # the numpy encoder must reproduce torch on real histories (catalog indices)
    from firstpr.serve.encoder import SASRecEncoder

    pos = {rid: i for i, rid in enumerate(details["repo_id"])}
    hist = _all_interactions(processed).train_histories
    item_repo = item_map.sort_values("item")["raw_id"].to_numpy()
    seqs = [np.array([pos[item_repo[i]] for i in h if item_repo[i] in pos]) for h in hist[:200]]
    err = check_encoder(model, SASRecEncoder.load(static / "model.npz"), item_vecs, seqs)
    meta["numpy_vs_torch_max_abs_diff"] = err
    (static / "meta.json").write_text(json.dumps(meta, indent=2))
    if err > 1e-3:
        raise RuntimeError(f"numpy encoder differs from torch by {err}")
    log.info(
        "static artifacts: %d repos (%d in training), encoder diff %.2e",
        len(repos),
        meta["n_training_repos"],
        err,
    )


def check_encoder(
    model: TextSASRec, encoder: Any, item_vecs: np.ndarray, seqs: list[np.ndarray]
) -> float:
    """Max |numpy - torch| over user vectors for the given catalog-index sequences."""
    from firstpr.models.sasrec import left_pad

    with torch.no_grad():
        pad = left_pad(seqs, model.max_len)
        table = torch.from_numpy(
            np.vstack([np.zeros((1, item_vecs.shape[1])), item_vecs]).astype(np.float32)
        )
        x = torch.from_numpy(pad)
        model.module.item_emb = torch.nn.Embedding.from_pretrained(table, padding_idx=0)
        ref = model.module(x)[:, -1, :].numpy()
    got = np.stack([encoder.encode(s, item_vecs) for s in seqs])
    return float(np.abs(ref - got).max())
