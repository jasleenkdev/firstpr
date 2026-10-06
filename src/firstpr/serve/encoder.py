"""Text-SASRec user encoder in numpy (inference only).

Reproduces `SASRecModule.forward` (one attention head, pre-LayerNorm blocks, left padding with
absolute positions, final LayerNorm) from exported weights, so the API needs neither torch nor
onnxruntime. Item vectors are precomputed: adapter(text embedding) for every catalog repo.
"""

from pathlib import Path

import numpy as np

EPS = 1e-5  # torch LayerNorm default


def _layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + EPS) * w + b


class SASRecEncoder:
    def __init__(self, weights: dict[str, np.ndarray]) -> None:
        self.w = {k: np.asarray(v, dtype=np.float32) for k, v in weights.items()}
        self.max_len, self.hidden = self.w["pos"].shape
        self.blocks = int(self.w["n_blocks"])

    @classmethod
    def load(cls, path: str | Path) -> "SASRecEncoder":
        with np.load(path) as z:
            return cls({k: z[k] for k in z.files})

    def encode(self, history: np.ndarray, item_vecs: np.ndarray) -> np.ndarray:
        """history: catalog indices, oldest -> newest -> user vector [hidden] (state after the
        most recent item). Scores for all items = item_vecs @ user vector."""
        h = np.asarray(history, dtype=np.int64)[-self.max_len :]
        L, H, w = self.max_len, self.hidden, self.w
        x = np.zeros((L, H), dtype=np.float32)
        pad = np.ones(L, dtype=bool)
        if len(h):
            x[-len(h) :] = item_vecs[h] * np.sqrt(H)
            pad[-len(h) :] = False
        x = (x + w["pos"]) * (~pad)[:, None]
        mask = np.triu(np.ones((L, L), dtype=bool), 1) | pad[None, :]
        mask &= ~np.eye(L, dtype=bool)
        for i in range(self.blocks):
            q = _layer_norm(x, w[f"b{i}.ln1.w"], w[f"b{i}.ln1.b"])
            qkv = q @ w[f"b{i}.in.w"].T + w[f"b{i}.in.b"]
            Q, K, V = qkv[:, :H], qkv[:, H : 2 * H], qkv[:, 2 * H :]
            s = (Q @ K.T) / np.sqrt(H)
            s = np.where(mask, -np.inf, s)
            s = np.exp(s - s.max(-1, keepdims=True))
            a = (s / s.sum(-1, keepdims=True)) @ V
            x = x + a @ w[f"b{i}.out.w"].T + w[f"b{i}.out.b"]
            f = _layer_norm(x, w[f"b{i}.ln2.w"], w[f"b{i}.ln2.b"])
            f = np.maximum(f @ w[f"b{i}.f1.w"].T + w[f"b{i}.f1.b"], 0)
            x = x + f @ w[f"b{i}.f2.w"].T + w[f"b{i}.f2.b"]
            x = x * (~pad)[:, None]
        return _layer_norm(x, w["out.w"], w["out.b"])[-1]
