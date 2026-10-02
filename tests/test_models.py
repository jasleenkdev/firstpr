import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from firstpr.data.dataset import InteractionData
from firstpr.models.itemknn import ItemKNN, cosine_shrink_knn
from firstpr.models.popularity import Popularity
from firstpr.models.registry import build_model


def _tiny() -> InteractionData:
    # users x items (train):  u0: 0,1   u1: 0,1,2   u2: 1,2   u3: 3
    train = pd.DataFrame({"user": [0, 0, 1, 1, 1, 2, 2, 3], "item": [0, 1, 0, 1, 2, 1, 2, 3]})
    empty = pd.DataFrame({"user": [0], "item": [3]})
    return InteractionData.from_frames(train, empty, empty, n_users=4, n_items=4)


@pytest.mark.parametrize("name,config", [("popularity", {}), ("itemknn", {"k": 5, "shrink": 0})])
def test_models_return_full_score_matrix(toy_data, name, config):
    model = build_model(name)
    model.fit(toy_data, config)
    users = np.array([0, 3, 7])
    scores = model.score(users)
    assert scores.shape == (3, toy_data.n_items)
    assert np.isfinite(scores).all()


def test_popularity_ranks_by_train_count():
    data = _tiny()
    m = Popularity()
    m.fit(data, {})
    np.testing.assert_array_equal(m.score(np.array([0]))[0], [2, 3, 2, 1])


def test_cosine_shrink_matches_hand_computation():
    data = _tiny()
    s = cosine_shrink_knn(data.train, k=10, shrink=1.0).toarray()
    # |U_0|=2, |U_1|=3, |U_2|=2, co(0,1)=2, co(1,2)=2, co(0,2)=1
    assert s[0, 1] == pytest.approx(2 / (np.sqrt(2) * np.sqrt(3) + 1))
    assert s[0, 2] == pytest.approx(1 / (2 + 1))
    assert s[0, 3] == 0 and s[3].sum() == 0  # item 3 co-occurs with nothing
    np.testing.assert_allclose(np.diag(s), 0)
    np.testing.assert_allclose(s, s.T, rtol=1e-6)  # symmetric when k keeps everything


def test_topk_keeps_at_most_k_neighbours_per_item(toy_data):
    s = cosine_shrink_knn(toy_data.train, k=3, shrink=0.0)
    assert s.getnnz(axis=1).max() <= 3
    full = cosine_shrink_knn(toy_data.train, k=toy_data.n_items, shrink=0.0).toarray()
    for j in range(5):  # kept neighbours are the largest ones
        kept = s[j].toarray().ravel()
        assert kept.max() == pytest.approx(full[j].max())


def test_itemknn_score_is_sum_of_neighbour_similarities():
    data = _tiny()
    m = ItemKNN()
    m.fit(data, {"k": 10, "shrink": 0.0})
    s = cosine_shrink_knn(data.train, k=10, shrink=0.0).toarray()
    # u2 has items 1, 2 -> score(item 0) = sim(1,0) + sim(2,0)
    assert m.score(np.array([2]))[0, 0] == pytest.approx(s[0, 1] + s[0, 2])


def test_itemknn_blocked_equals_unblocked(toy_data):
    a = cosine_shrink_knn(toy_data.train, k=7, shrink=5.0, block_size=7)
    b = cosine_shrink_knn(toy_data.train, k=7, shrink=5.0, block_size=10_000)
    assert abs(a - b).max() < 1e-6
    assert isinstance(a, sp.csr_matrix)
