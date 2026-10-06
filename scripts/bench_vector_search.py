"""Exact numpy search vs Qdrant local mode for the API's vector sets.

Cases: repo text embeddings (bge, 384-d), repo model vectors (text-SASRec adapter space, 64-d),
open-issue title embeddings (bge, 384-d). For each: top-100 by dot product for 200 random query
vectors (seeded), p50 / p95 latency per query, recall@100 against exact search, build time.
-> results/analysis/vector_search_phase6.csv
"""

import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from firstpr.text.encoder import DEFAULT_ENCODER, cached_embeddings

K, N_QUERIES = 100, 200


def numpy_search(m: np.ndarray, q: np.ndarray) -> np.ndarray:
    s = m @ q
    top = np.argpartition(-s, K)[:K]
    return top[np.argsort(-s[top])]


def bench(name: str, m: np.ndarray, rng: np.random.Generator, tmp: Path) -> list[dict]:
    from qdrant_client import QdrantClient, models

    queries = m[rng.choice(len(m), N_QUERIES, replace=False)] + 0.01 * rng.normal(
        size=(N_QUERIES, m.shape[1])
    )
    queries = queries.astype(np.float32)
    rows = []

    def timed(fn) -> tuple[list[float], list[np.ndarray]]:
        times, outs = [], []
        for q in queries:
            t0 = time.perf_counter()
            outs.append(np.asarray(fn(q)))
            times.append((time.perf_counter() - t0) * 1000)
        return times, outs

    t_np, exact = timed(lambda q: numpy_search(m, q))
    rows.append(
        {
            "case": name,
            "engine": "numpy exact",
            "n": len(m),
            "dim": m.shape[1],
            "build_ms": 0.0,
            "p50_ms": np.percentile(t_np, 50),
            "p95_ms": np.percentile(t_np, 95),
            "recall@100": 1.0,
        }
    )
    for engine, location in [
        ("qdrant local (memory)", ":memory:"),
        ("qdrant local (disk)", str(tmp / name)),
    ]:
        client = (
            QdrantClient(location=location)
            if location == ":memory:"
            else QdrantClient(path=location)
        )
        t0 = time.perf_counter()
        client.recreate_collection(
            name, vectors_config=models.VectorParams(size=m.shape[1], distance=models.Distance.DOT)
        )
        client.upload_collection(name, vectors=m, ids=list(range(len(m))), batch_size=2048)
        build = (time.perf_counter() - t0) * 1000

        def q_fn(q: np.ndarray, c=client) -> list[int]:
            hits = c.query_points(name, query=q.tolist(), limit=K).points
            return [h.id for h in hits]

        t_q, outs = timed(q_fn)
        rec = np.mean([len(set(a) & set(b)) / K for a, b in zip(outs, exact, strict=True)])
        rows.append(
            {
                "case": name,
                "engine": engine,
                "n": len(m),
                "dim": m.shape[1],
                "build_ms": build,
                "p50_ms": np.percentile(t_q, 50),
                "p95_ms": np.percentile(t_q, 95),
                "recall@100": rec,
            }
        )
        client.close()
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--serving", default="data/serving")
    a = p.parse_args()
    s = Path(a.serving)
    rng = np.random.default_rng(0)
    text = np.load(s / "static" / "text_emb.npy").astype(np.float32)
    vecs = np.load(s / "static" / "item_vecs.npy").astype(np.float32)
    with gzip.open(s / "dynamic" / "issues.json.gz", "rt") as f:
        titles = [i["title"] for i in json.load(f)]
    issue_emb = cached_embeddings(titles, s / "cache" / "issue_titles", DEFAULT_ENCODER)
    tmp = s / "cache" / "qdrant_bench"
    tmp.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, m in [
        ("repos_text_384", text),
        ("repos_model_64", vecs),
        ("issues_text_384", issue_emb),
    ]:
        rows += bench(name, m, rng, tmp)
    df = pd.DataFrame(rows)
    out = Path("results/analysis/vector_search_phase6.csv")
    df.to_csv(out, index=False, float_format="%.4g")
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
