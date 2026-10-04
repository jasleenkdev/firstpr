"""Frozen sentence encoder for item (and LLM-generated) text, cached as .npy next to the data.

Default: BAAI/bge-small-en-v1.5 (33M parameters, 384-d), small enough to embed a few thousand
texts on CPU in minutes. Embeddings are L2-normalised. The encoder is never fine-tuned: all
learning happens in the adapters (UniSRec keeps BERT frozen the same way).
"""

import hashlib
import re
from pathlib import Path

import numpy as np

from firstpr.utils.logging import get_logger

log = get_logger(__name__)

DEFAULT_ENCODER = "BAAI/bge-small-en-v1.5"


def slug(model_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model_name.lower()).strip("-")


def encode_texts(texts: list[str], model_name: str, batch_size: int = 64) -> np.ndarray:
    from sentence_transformers import SentenceTransformer  # heavy import, only when encoding

    model = SentenceTransformer(model_name, device="cpu")
    emb = model.encode(
        texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
    )
    return np.asarray(emb, dtype=np.float32)


def cached_embeddings(
    texts: list[str], cache_path: str | Path, model_name: str = DEFAULT_ENCODER
) -> np.ndarray:
    """Encode `texts` once; the cache is keyed by the encoder and a hash of all texts."""
    cache_path = Path(cache_path)
    digest = hashlib.sha256("\x1e".join([model_name, *texts]).encode()).hexdigest()[:16]
    path = cache_path.with_name(f"{cache_path.stem}_{slug(model_name)}_{digest}.npy")
    if path.exists():
        return np.load(path)
    log.info("encoding %d texts with %s", len(texts), model_name)
    emb = encode_texts(texts, model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    return emb
