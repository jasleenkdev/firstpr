import math

import numpy as np
import pytest

from firstpr.eval import metrics as M


def _hits(topk, target_items, n_items=10):
    targets = np.zeros((len(topk), n_items), dtype=bool)
    for u, items in enumerate(target_items):
        targets[u, list(items)] = True
    return M.hits_matrix(np.array(topk), targets), targets.sum(axis=1)


def test_recall_ndcg_hit_hand_computed():
    # K=3, targets {1, 7, 9}; ranked list [3, 1, 7] -> hits at ranks 2 and 3
    hits, n = _hits([[3, 1, 7]], [{1, 7, 9}])
    assert list(hits[0]) == [False, True, True]
    assert M.recall_at_k(hits, n)[0] == pytest.approx(2 / 3)
    dcg = 1 / math.log2(3) + 1 / math.log2(4)
    idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)  # min(|targets|=3, K=3) ideal hits
    assert M.ndcg_at_k(hits, n)[0] == pytest.approx(dcg / idcg)
    assert M.hit_rate_at_k(hits)[0] == 1.0


def test_fewer_targets_than_k_idcg_uses_min():
    # K=4, one target at rank 2 -> NDCG = (1/log2 3) / 1; recall = 1
    hits, n = _hits([[0, 5, 2, 3]], [{5}])
    assert M.ndcg_at_k(hits, n)[0] == pytest.approx(1 / math.log2(3))
    assert M.recall_at_k(hits, n)[0] == 1.0


def test_more_targets_than_k_recall_denominator_is_targets():
    # K=2, 4 targets, both slots hit -> recall 2/4 (|targets| denominator), ndcg 1 (IDCG over K)
    hits, n = _hits([[1, 2]], [{1, 2, 3, 4}])
    assert M.recall_at_k(hits, n)[0] == 0.5
    assert M.ndcg_at_k(hits, n)[0] == pytest.approx(1.0)


def test_no_hits_and_all_hits_and_multiple_users():
    hits, n = _hits([[0, 1, 2], [4, 5, 6]], [{7}, {4, 5, 6}])
    np.testing.assert_allclose(M.recall_at_k(hits, n), [0.0, 1.0])
    np.testing.assert_allclose(M.ndcg_at_k(hits, n), [0.0, 1.0])
    np.testing.assert_allclose(M.hit_rate_at_k(hits), [0.0, 1.0])


def test_empty_slots_never_hit():
    hits, n = _hits([[2, -1, -1]], [{0, 2}])
    assert list(hits[0]) == [True, False, False]


def test_coverage_popularity_long_tail():
    topk = np.array([[0, 1], [1, 2], [0, -1]])
    pop = np.array([10.0, 4.0, 1.0, 0.0])
    tail = np.array([False, False, True, True])
    assert M.catalog_coverage(topk, n_items=4) == 0.75
    # filled slots: 0,1,1,2,0 -> pops 10,4,4,1,10
    assert M.average_popularity(topk, pop) == pytest.approx(29 / 5)
    assert M.long_tail_share(topk, tail) == pytest.approx(1 / 5)
