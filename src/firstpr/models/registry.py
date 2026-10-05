"""Model name (as used in configs/models/<name>.yaml) -> constructor."""

from collections.abc import Callable

from firstpr.models.base import Recommender
from firstpr.models.itemknn import ItemKNN
from firstpr.models.lightgcn import LightGCN
from firstpr.models.mf_bpr import MFBPR
from firstpr.models.ncf import NCF
from firstpr.models.popularity import Popularity
from firstpr.models.sasrec import SASRec
from firstpr.models.sasrec_text import TextSASRec
from firstpr.models.two_tower import TwoTower
from firstpr.recbole_bridge.model import RecBoleModel

MODELS: dict[str, Callable[[], Recommender]] = {
    "popularity": Popularity,
    "itemknn": ItemKNN,
    "mf_bpr": MFBPR,
    "mf_bpr_init001": MFBPR,  # init ablation: mf_bpr's tuned config with N(0, 0.01) init
    "ncf_gmf": lambda: NCF("gmf"),
    "ncf_mlp": lambda: NCF("mlp"),
    "ncf_neumf": lambda: NCF("neumf"),
    "two_tower": TwoTower,
    "two_tower_nologq": TwoTower,  # ablation: same model, logq: false (tuned separately)
    "sasrec": SASRec,
    "sasrec_bce": SASRec,  # ablation: sasrec's tuned config with the paper's BCE loss
    "sasrec_shuffled": SASRec,  # ablation: sasrec's tuned config on shuffled histories
    "recbole_bpr": lambda: RecBoleModel("BPR"),  # via recbole_bridge (isolated env)
    "lightgcn": LightGCN,
    "lightgcn_l1": LightGCN,  # layer ablation: lightgcn's tuned config with n_layers overridden
    "lightgcn_l2": LightGCN,
    "lightgcn_l4": LightGCN,
    "lightgcn_init001": LightGCN,  # init ablation: lightgcn's tuned config with N(0, 0.01) init
    "recbole_ngcf": lambda: RecBoleModel("NGCF"),
    "recbole_lightgcn": lambda: RecBoleModel("LightGCN"),  # reference check for our lightgcn
    # phase 4: SASRec over frozen text embeddings; adapter linear / mlp / moe, +- ID embedding
    **{f"sasrec_text_{a}{i}": TextSASRec for a in ("linear", "mlp", "moe") for i in ("", "_id")},
    "sasrec_text_kar_item": TextSASRec,
    "sasrec_text_kar": TextSASRec,
    "sasrec_text_moe_warm": TextSASRec,
    "sasrec_text_linear_id_warm": TextSASRec,
}


def build_model(name: str) -> Recommender:
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; known: {sorted(MODELS)}")
    return MODELS[name]()
