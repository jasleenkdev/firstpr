import pandas as pd

from firstpr.data.amazon import item_table, item_text, load_amazon_ratings


def test_load_ratings_date_cut_and_variant_dedupe(tmp_path):
    csv = tmp_path / "r.csv"
    pd.DataFrame(
        {
            "user_id": ["a", "a", "a", "b"],
            "parent_asin": ["x", "x", "y", "x"],
            "rating": [5.0, 3.0, 4.0, 5.0],
            "timestamp": [t * 1_000_000_000_000 for t in (1.6, 1.61, 1.5, 1.62)],
        }
    ).to_csv(csv, index=False)
    df = load_amazon_ratings(csv, "2020-01-01")
    assert len(df) == 2  # 2017 rating dropped, second rating of (a, x) dropped
    first = df[(df.user == "a") & (df.item == "x")].iloc[0]
    assert first.rating == 5.0 and first.timestamp == 1_600_000_000


def test_item_text_uses_only_page_fields_and_truncates():
    rec = {
        "title": "Digital  Caliper",
        "store": "Acme",
        "categories": ["Industrial & Scientific", "Measuring"],
        "features": ["0.01 mm resolution", "stainless", "battery", "case"],
        "description": ["word " * 100],
        "price": 9.99,
    }
    t = item_text(rec, max_features=2, max_description_chars=30)
    assert t.startswith("Title: Digital Caliper\nBrand: Acme\nCategory: Industrial & Scientific > ")
    assert "battery" not in t and "9.99" not in t
    assert t.endswith("...") and len(t.splitlines()[-1]) < 50


def test_item_table_keeps_id_order_and_handles_missing_meta():
    item_map = pd.DataFrame({"raw_id": ["p", "q"], "item": [0, 1]})
    out = item_table([{"parent_asin": "q", "title": "Q"}], item_map, 3, 100)
    assert out["item"].tolist() == [0, 1]
    assert out.loc[0, "text"] == "" and out.loc[1, "title"] == "Q"
