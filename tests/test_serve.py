"""Serving: numpy encoder == torch SASRec, ranking, privacy of the username path, API."""

import numpy as np
import torch

from firstpr.models.sasrec import SASRecModule, left_pad
from firstpr.serve.build import export_encoder
from firstpr.serve.encoder import SASRecEncoder


def test_numpy_encoder_matches_torch():
    torch.manual_seed(0)
    n, L, H = 30, 8, 16
    m = SASRecModule(n, L, H, blocks=2, heads=1, dropout=0.3).eval()
    with torch.no_grad():  # non-trivial norms and biases
        for p in m.parameters():
            p.add_(0.1 * torch.randn_like(p))
        m.item_emb.weight[0].zero_()
    enc = SASRecEncoder(export_encoder(m))
    vecs = m.item_emb.weight[1:].detach().numpy()
    seqs = [np.array([3, 1, 4]), np.arange(12) % n, np.array([], dtype=int), np.array([7])]
    with torch.no_grad():
        ref = m(torch.from_numpy(left_pad(seqs, L)))[:, -1, :].numpy()
    got = np.stack([enc.encode(s, vecs) for s in seqs])
    assert np.abs(ref - got).max() < 1e-5
