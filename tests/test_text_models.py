import numpy as np
import pandas as pd
import torch

import firstpr.models.sasrec_text as st
from firstpr.data.dataset import InteractionData
from firstpr.eval.evaluator import Evaluator
from firstpr.models.sasrec_text import TextSASRec
from firstpr.text.adapters import MoEAdapter, build_adapter
from firstpr.utils.seed import set_seed

SMALL = {
    "max_len": 12,
    "hidden": 16,
    "blocks": 1,
    "heads": 1,
    "dropout": 0.0,
    "lr": 0.01,
    "batch_size": 32,
    "max_epochs": 30,
    "patience": 8,
    "seed": 0,
}


def test_adapter_shapes_and_moe_starts_as_expert_mean():
    t = torch.randn(7, 12)
    for kind in ("linear", "mlp", "moe"):
        assert build_adapter(kind, 12, 5, 0.0)(t).shape == (7, 5)
    moe = MoEAdapter(12, 5, 0.0, n_experts=3).eval()
    experts = torch.einsum("nkd,kdo->nko", t[:, None, :] - moe.bias[None], moe.proj)
    torch.testing.assert_close(moe(t), experts.mean(1))  # zero-initialised gate


def _clusters_with_cold_item() -> tuple[InteractionData, np.ndarray]:
    """Two user groups with disjoint item clusters (0-29, 30-59). Item 60 never appears in train
    but its text is close to cluster B's."""
    rng = np.random.default_rng(0)
    rows = []
    for u in range(200):
        cluster = np.arange(0, 30) if u < 130 else np.arange(30, 60)
        items = rng.choice(cluster, size=12, replace=False)
        rows += [(u, int(i), t) for t, i in enumerate(items)]
    df = pd.DataFrame(rows, columns=["user", "item", "timestamp"])
    df = df.sort_values(["user", "timestamp"])
    last = df.groupby("user").cumcount(ascending=False)
    test, val, train = df[last == 0], df[last == 1], df[last >= 2]
    data = InteractionData.from_frames(train, val, test, n_users=200, n_items=61)
    a, b = rng.normal(size=8), rng.normal(size=8)
    text = np.vstack(
        [a + 0.3 * rng.normal(size=(30, 8)), b + 0.3 * rng.normal(size=(30, 8)), b[None]]
    ).astype(np.float32)
    return data, text


def test_text_sasrec_personalises_and_scores_cold_items(monkeypatch):
    data, text = _clusters_with_cold_item()
    monkeypatch.setattr(st, "text_matrix", lambda d, file, key, n, enc: text)
    set_seed(0)
    ev = Evaluator(data, k=10)
    model = TextSASRec()
    model.fit(
        data,
        {**SMALL, "adapter": "moe"},
        val_fn=lambda m: ev.evaluate(m, "val")["overall"]["ndcg@10"],
    )
    users = np.arange(200)
    top = ev.topk(model, users, (data.train + data.val).tocsr())
    assert (top[:130] < 30).mean() > 0.9 and (top[130:] >= 30).mean() > 0.9
    scores = model.score(users)
    rank_cold = (scores > scores[:, [60]]).sum(1)  # rank of the cold item per user
    assert rank_cold[130:].mean() < rank_cold[:130].mean() - 10


def test_kar_paths_run(monkeypatch):
    data, text = _clusters_with_cold_item()
    pref = np.random.default_rng(1).normal(size=(200, 8)).astype(np.float32)
    monkeypatch.setattr(
        st, "text_matrix", lambda d, file, key, n, enc: pref if key == "user" else text
    )
    model = TextSASRec()
    cfg = {**SMALL, "adapter": "linear", "use_id": True, "item_knowledge": True}
    model.fit(data, {**cfg, "user_preference": True, "max_epochs": 2})
    assert model.score(np.arange(5)).shape == (5, 61)


def test_cold_negatives_false_keeps_cold_items_out_of_the_training_softmax(monkeypatch):
    data, text = _clusters_with_cold_item()
    monkeypatch.setattr(st, "text_matrix", lambda d, file, key, n, enc: text)
    adam = torch.optim.Adam
    snapshots: list[torch.Tensor] = []

    def snapshot_adam(params, **kw):  # copy the ID table as initialised, before training
        params = list(params)
        snapshots.append(next(p for p in params if p.shape == (62, 16)).detach().clone())
        return adam(params, **kw)

    monkeypatch.setattr(st.torch.optim, "Adam", snapshot_adam)
    cfg = {**SMALL, "adapter": "linear", "use_id": True, "max_epochs": 3}
    moved = {}
    for flag in (True, False):
        set_seed(0)
        model = TextSASRec()
        model.fit(data, {**cfg, "cold_negatives": flag})
        row = 60 + 1  # item 60 is cold; row 0 is padding
        w = model.module.item_emb.id_emb.weight[row].detach()
        moved[flag] = float((w - snapshots[-1][row]).abs().max())
    assert moved[True] > 0  # pushed down as a negative
    assert moved[False] == 0  # never in the softmax, never in an input sequence
