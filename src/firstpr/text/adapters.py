"""Adapters from frozen text embeddings to a recommender's item space.

- linear: e = W t + b. The simplest transfer (ZESRec, Ding et al. 2021, maps pretrained text
  embeddings to the item space with a small network; the linear map is its minimal form).
- mlp: two layers with GELU. More capacity, still one shared mapping for all items.
- moe: UniSRec (Hou et al., KDD 2022). K experts, each a *parametric whitening*
  e_k = (t - b_k) W_k, mixed by a softmax gate g = softmax(t W_g):  e = sum_k g_k e_k.
  Whitening removes the anisotropy of sentence embeddings (all vectors sit in a narrow cone), and
  the experts let different item types use different projections.
- hybrid (KAR, Xi et al. 2024): the same gated-expert form applied to LLM knowledge embeddings;
  see `firstpr.models.sasrec_text`.
"""

import torch
import torch.nn.functional as F
from torch import nn


class LinearAdapter(nn.Module):
    def __init__(self, d_in: int, d_out: int, dropout: float) -> None:
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.lin = nn.Linear(d_in, d_out)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.lin(self.drop(t))


class MLPAdapter(nn.Module):
    def __init__(self, d_in: int, d_out: int, dropout: float, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_out),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)


class MoEAdapter(nn.Module):
    """UniSRec's MoE-enhanced adapter with parametric whitening experts."""

    def __init__(self, d_in: int, d_out: int, dropout: float, n_experts: int = 8) -> None:
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.bias = nn.Parameter(torch.zeros(n_experts, d_in))  # b_k (whitening centre)
        self.proj = nn.Parameter(torch.empty(n_experts, d_in, d_out))  # W_k
        nn.init.normal_(self.proj, std=0.02)
        self.gate = nn.Linear(d_in, n_experts, bias=False)
        nn.init.zeros_(self.gate.weight)  # uniform mixture at the start

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = self.drop(t)
        experts = torch.einsum("nkd,kdo->nko", x[:, None, :] - self.bias[None], self.proj)
        g = F.softmax(self.gate(x), dim=-1)  # [n, K]
        return (g[..., None] * experts).sum(1)


ADAPTERS = {"linear": LinearAdapter, "mlp": MLPAdapter, "moe": MoEAdapter}


def build_adapter(kind: str, d_in: int, d_out: int, dropout: float) -> nn.Module:
    if kind not in ADAPTERS:
        raise KeyError(f"unknown adapter {kind!r}; known: {sorted(ADAPTERS)}")
    return ADAPTERS[kind](d_in, d_out, dropout)
