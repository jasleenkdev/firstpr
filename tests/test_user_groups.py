import numpy as np

from firstpr.eval.user_groups import activity_groups, group_cis, group_table


def _runs(tmp_path, name, values, users):
    out = []
    for s, v in enumerate(values):
        np.savez(tmp_path / f"{name}_{s}.npz", users=users, ndcg_at_20=v)
        path, file = str(tmp_path / "x.json"), f"{name}_{s}.npz"
        out.append({"model": name, "_path": path, "per_user_file": file})
    return out


def test_activity_groups_are_quantiles_and_ordered():
    h = np.arange(1, 101)
    group, labels = activity_groups(h, 4)
    assert np.bincount(group).tolist() == [25, 25, 25, 25]
    assert labels[0] == "1-25" and labels[-1] == "76-100"
    assert (np.diff(group) >= 0).all()


def test_group_table_and_cis(tmp_path):
    users = np.arange(8)
    history_len = np.array([1, 2, 3, 4, 10, 20, 30, 40])
    a = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=float)  # strong on light users only
    b = np.zeros(8)
    runs = {"a": _runs(tmp_path, "a", [a, a], users), "b": _runs(tmp_path, "b", [b], users)}
    t = group_table(runs, history_len, "ndcg@20", 2)
    assert t.loc[(t.model == "a") & (t.group == 0), "ndcg@20"].item() == 1.0
    assert t.loc[(t.model == "a") & (t.group == 1), "ndcg@20"].item() == 0.0
    c = group_cis(runs, history_len, [("a", "b")], "ndcg@20", 2, 200, 0.95, 0)
    assert c.loc[c.group == 0, "ci_low"].item() == 1.0
    assert c.loc[c.group == 1, "diff"].item() == 0.0
