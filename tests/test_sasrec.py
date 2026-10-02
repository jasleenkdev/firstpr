import numpy as np
import pandas as pd
import pytest
import torch

from firstpr.data.dataset import InteractionData
from firstpr.eval.evaluator import Evaluator
from firstpr.models.popularity import Popularity
from firstpr.models.sasrec import SASRec, SASRecModule, left_pad
from firstpr.utils.seed import set_seed

SMALL = {
    "max_len": 12,
    "hidden": 16,
    "blocks": 1,
    "heads": 1,
    "dropout": 0.0,
    "lr": 0.01,
    "batch_size": 32,
    "loss": "ce",
    "max_epochs": 40,
    "patience": 8,
    "eval_every": 1,
    "seed": 0,
}


def test_left_pad():
    out = left_pad([np.array([4, 5, 6]), np.array([], dtype=np.int64), np.arange(10)], 4)
    np.testing.assert_array_equal(out[0], [0, 5, 6, 7])
    np.testing.assert_array_equal(out[1], [0, 0, 0, 0])
    np.testing.assert_array_equal(out[2], [7, 8, 9, 10])


def test_causal_no_future_leak_and_no_nan():
    torch.manual_seed(0)
    m = SASRecModule(n_items=20, max_len=6, hidden=8, blocks=2, heads=2, dropout=0.0).eval()
    seq = torch.tensor([[0, 0, 3, 4, 5, 6]])
    h = m(seq)
    assert torch.isfinite(h).all()
    changed = seq.clone()
    changed[0, -1] = 9  # changing the last item must not change earlier positions
    h2 = m(changed)
    torch.testing.assert_close(h[:, :-1], h2[:, :-1])
    assert not torch.allclose(h[:, -1], h2[:, -1])


def _sequential_data() -> InteractionData:
    """Each user walks a cycle of 30 items from a random start, forwards or backwards at random.
    The next item follows from the last two items, but the *set* of history items leaves two
    equally likely candidates (one past either end), so only the order resolves it."""
    rng = np.random.default_rng(0)
    rows = []
    for u in range(150):
        start, step = int(rng.integers(0, 30)), int(rng.choice([-1, 1]))
        items = [(start + step * t) % 30 for t in range(14)]
        rows += [(u, i, t) for t, i in enumerate(items)]
    df = pd.DataFrame(rows, columns=["user", "item", "timestamp"]).assign(
        seq=lambda d: d["timestamp"]
    )
    last = df.groupby("user").cumcount(ascending=False)
    return InteractionData.from_frames(
        df[last >= 2], df[last == 1], df[last == 0], n_users=150, n_items=30
    )


@pytest.mark.parametrize("loss", ["ce", "bce"])
def test_sasrec_learns_next_item(loss):
    set_seed(0)
    data = _sequential_data()
    ev = Evaluator(data, k=1)
    model = SASRec()
    model.fit(
        data, {**SMALL, "loss": loss}, val_fn=lambda m: ev.evaluate(m, "val")["overall"]["ndcg@1"]
    )
    pop = Popularity()
    pop.fit(data, {})
    assert ev.evaluate(model, "val")["overall"]["hit@1"] > 0.8
    assert ev.evaluate(pop, "val")["overall"]["hit@1"] < 0.2


def test_shuffle_history_destroys_order_signal():
    set_seed(0)
    data = _sequential_data()
    ev = Evaluator(data, k=1)
    model = SASRec()
    model.fit(
        data,
        {**SMALL, "shuffle_history": True},
        val_fn=lambda m: ev.evaluate(m, "val")["overall"]["ndcg@1"],
    )
    # without order the model can at best guess between the two window ends (~0.5)
    assert ev.evaluate(model, "val")["overall"]["hit@1"] < 0.7
