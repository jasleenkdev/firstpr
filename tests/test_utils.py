import numpy as np
import torch

from firstpr.utils.io import config_hash, load_json, save_json
from firstpr.utils.seed import set_seed


def test_set_seed_makes_numpy_and_torch_reproducible():
    set_seed(3)
    a = (np.random.rand(3), torch.rand(3))
    set_seed(3)
    b = (np.random.rand(3), torch.rand(3))
    np.testing.assert_array_equal(a[0], b[0])
    assert torch.equal(a[1], b[1])


def test_config_hash_ignores_key_order():
    assert config_hash({"a": 1, "b": [1, 2]}) == config_hash({"b": [1, 2], "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_json_roundtrip_with_numpy(tmp_path):
    save_json({"x": np.float32(1.5), "y": np.arange(3)}, tmp_path / "r.json")
    assert load_json(tmp_path / "r.json") == {"x": 1.5, "y": [0, 1, 2]}
