"""Write the FirstPR split as RecBole benchmark atomic files.

<out_dir>/<name>/<name>.{train,valid,test}.inter with columns
user_id:token, item_id:token, timestamp:float. Tokens are FirstPR integer ids, so scores can be
mapped back exactly. The timestamp field is a synthetic order key: train items get their position
in the user's history (`train_histories`, i.e. our tie-broken chronological order), val items come
after train, test items after val. RecBole therefore sees exactly our order, including ties.
"""

from pathlib import Path

import numpy as np

from firstpr.data.dataset import InteractionData

HEADER = "user_id:token\titem_id:token\ttimestamp:float\n"


def export_benchmark(data: InteractionData, out_dir: str | Path, name: str) -> Path:
    d = Path(out_dir) / name
    d.mkdir(parents=True, exist_ok=True)
    offsets = np.zeros(data.n_users, dtype=np.int64)

    with open(d / f"{name}.train.inter", "w") as f:
        f.write(HEADER)
        for u, hist in enumerate(data.train_histories):
            for t, i in enumerate(hist):
                f.write(f"{u}\t{i}\t{t}\n")
            offsets[u] = len(hist)

    for split, matrix in (("valid", data.val), ("test", data.test)):
        with open(d / f"{name}.{split}.inter", "w") as f:
            f.write(HEADER)
            for u in range(data.n_users):
                items = matrix.indices[matrix.indptr[u] : matrix.indptr[u + 1]]
                for k, i in enumerate(np.sort(items)):
                    f.write(f"{u}\t{i}\t{offsets[u] + k}\n")
                offsets[u] += len(items)
    return d
