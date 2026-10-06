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


# ---- synthetic catalog -------------------------------------------------------------------------

import logging  # noqa: E402
from datetime import UTC, datetime  # noqa: E402

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from firstpr.serve import ranking  # noqa: E402
from firstpr.serve.app import create_app  # noqa: E402
from firstpr.serve.catalog import Catalog  # noqa: E402
from firstpr.serve.explain import Explainer, collect_facts  # noqa: E402
from firstpr.serve.github import Star, hash_username  # noqa: E402

NOW = datetime(2026, 10, 6, tzinfo=UTC)
TOPICS = [
    {"id": "web", "label": "Web development", "kw": ["web", "react"]},
    {"id": "ml", "label": "Machine learning", "kw": ["pytorch", "model"]},
]


def make_catalog(n: int = 12, d: int = 16) -> Catalog:
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    m = SASRecModule(n, 8, d, blocks=1, heads=1, dropout=0.0).eval()
    langs = ["Python", "TypeScript", "Rust"]
    repos = [
        {
            "id": 100 + i,
            "name": f"owner{i % 4}/repo{i}",
            "description": f"repo {i}",
            "language": langs[i % 3],
            "languages": [langs[i % 3]],
            "topics": ["web"] if i % 2 else ["pytorch"],
            "stars": 10 * i,
            "created_at": "2024-01-01T00:00:00Z",
            "group": "beginner",
            "trained": True,
            "n_contrib": i,
            "n_pr": i,
            "n_actors": 5 * i,
        }
        for i in range(n)
    ]
    text = rng.normal(size=(n, 8)).astype(np.float32)
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    cat = Catalog(
        repos=repos,
        item_vecs=m.item_emb.weight[1:].detach().numpy(),
        text_emb=text,
        encoder=SASRecEncoder(export_encoder(m)),
        topics=TOPICS,
        topic_emb=text[:2].copy(),
        costar={100: [(101, 5), (102, 3)], 101: [(100, 5)]},
        meta={"model": "test"},
    )
    cat.index = {r["id"]: i for i, r in enumerate(repos)}
    cat.issues = {
        r["id"]: [
            {
                "repo_id": r["id"],
                "number": k,
                "title": f"issue {k}",
                "labels": ["good first issue"],
                "created_at": "2026-09-20T00:00:00Z",
                "updated_at": "2026-10-01T00:00:00Z",
                "n_comments": 1,
                "n_assignees": int(k == 3),
                "difficulty": ["easy", "medium", "hard"][k % 3],
                "skills": ["documentation"] if k == 1 else [],
                "snippet": "",
            }
            for k in range(1, 5)
        ]
        for r in repos
        if r["id"] != 111  # one repo without issues
    }
    cat.features = {
        100: {"summary": "A web framework.", "skills": ["Python", "web"], "domain": "web"}
    }
    cat.fresh = [
        {
            "id": 999,
            "name": "newowner/fresh",
            "description": "a react web tool",
            "language": "TypeScript",
            "topics": ["react"],
            "stars": 50,
            "created_at": "2026-09-01T00:00:00Z",
            "issues": [
                {
                    "repo_id": 999,
                    "number": 7,
                    "title": "add docs",
                    "labels": ["good first issue"],
                    "created_at": "2026-10-01T00:00:00Z",
                    "updated_at": "2026-10-01T00:00:00Z",
                    "n_comments": 0,
                    "n_assignees": 0,
                    "difficulty": "easy",
                    "skills": [],
                    "snippet": "",
                }
            ],
        }
    ]
    cat.manifest = {"updated_at": "2026-10-06T00:00:00+00:00"}
    return cat


def stars(ids: list[int]) -> list[Star]:
    return [
        Star(i, f"owner{i % 4}/repo{i - 100}", "Python", ["web"], f"2026-01-{k + 1:02d}")
        for k, i in enumerate(ids)
    ]


def test_difficulty_target_and_match():
    assert ranking.target_difficulty(2) == "easy" and ranking.target_difficulty(12) == "hard"
    assert ranking.difficulty_match("easy", "easy") == 1.0
    assert ranking.difficulty_match("easy", "medium") > ranking.difficulty_match("hard", "medium")
    assert ranking.difficulty_match(None, "easy") == 0.5


def test_github_path_modes_and_constraints():
    cat = make_catalog()
    for hist, mode in [([], "text_profile"), ([100], "blend"), ([100, 101, 102, 103], "model")]:
        out = ranking.recommend_github(cat, stars(hist), hours=2, now=NOW)
        assert out["profile"]["retrieval"] == mode
        ids = [r["id"] for r in out["repos"]]
        assert 111 not in ids  # no open issues -> never recommended
        assert sum(r["signals"]["starred_by_you"] for r in out["repos"]) <= ranking.MAX_STARRED
        owners = [r["name"].split("/")[0] for r in out["repos"]]
        assert max(owners.count(o) for o in owners) <= ranking.MAX_PER_OWNER
        for r in out["repos"]:
            assert 1 <= len(r["issues"]) <= ranking.N_ISSUES
            assert all(i["url"].startswith("https://github.com/") for i in r["issues"])


def test_issue_ranking_prefers_easy_unclaimed_for_few_hours():
    cat = make_catalog()
    prof = ranking.Profile(skills={"documentation"}, hours=1)
    picked = ranking.pick_issues(cat, cat.repos[0], prof, NOW)
    assert picked[0]["number"] == 1  # same difficulty as issue 4, but matches the skill
    assert all(i["number"] != 3 for i in picked[:2])  # issue 3 is assigned


def test_onboarding_and_new_projects():
    cat = make_catalog()
    out = ranking.recommend_onboarding(cat, ["TypeScript"], ["web"], hours=3, now=NOW)
    assert out["repos"] and out["profile"]["retrieval"] == "onboarding"
    assert out["new_projects"][0]["name"] == "newowner/fresh"


def test_explanation_facts_are_grounded_and_fall_back():
    cat = make_catalog()
    facts = collect_facts(
        cat, 100, 1, co_starred=["owner1/repo1", "made/up"], skills=["Python", "Go"]
    )
    text = " ".join(facts)
    assert "owner1/repo1" in text and "made/up" not in text  # only real co-star neighbours
    assert "Skills you have that the project uses: Python." in text  # Go is not a repo skill
    assert collect_facts(cat, 12345, None, [], []) is None
    ex = Explainer(api_key=None, model="x")
    out = ex.explain(facts)
    assert out["source"] == "template" and out["text"]


def test_explainer_falls_back_on_rate_limit(monkeypatch):
    import requests as rq

    class R:
        status_code = 429
        headers = {"retry-after": "100"}

    monkeypatch.setattr(rq, "post", lambda *a, **k: R())
    ex = Explainer(api_key="k", model="x")
    assert ex.explain(["Repository: a.", "fact"])["source"] == "template"
    assert ex.blocked_until > 0


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ID_SALT", "test-salt")
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://firstpr.example")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    calls = []

    def fake_fetch(username, token):
        calls.append(username)
        return stars([100, 101, 102])

    app = create_app(catalog=make_catalog(), star_fetcher=fake_fetch)
    c = TestClient(app)
    c.calls = calls
    return c


def test_api_paths(client):
    assert client.get("/health").json()["status"] == "ok"
    opts = client.get("/options").json()
    assert opts["interests"] and opts["languages"]
    r = client.post("/recommend/github", json={"username": "Octo-Cat", "hours": 3})
    assert r.status_code == 200 and r.json()["repos"]
    assert "fetch_stars" in r.json()["timings_ms"]
    r2 = client.post("/recommend/github", json={"username": "octo-cat"})  # cached by hash
    assert r2.json()["timings_ms"]["stars_cached"] is True and len(client.calls) == 1
    r3 = client.post(
        "/recommend/onboarding", json={"languages": ["Python"], "interests": ["ml"], "hours": 5}
    )
    assert r3.status_code == 200
    assert client.post("/recommend/onboarding", json={}).status_code == 422
    e = client.post("/explain", json={"repo_id": 100, "issue_number": 1, "skills": ["Python"]})
    assert e.status_code == 200 and e.json()["source"] == "template"
    assert client.post("/explain", json={"repo_id": 1}).status_code == 404


def test_invalid_username_rejected_before_any_request(client):
    r = client.post("/recommend/github", json={"username": "bad name;rm"})
    assert r.status_code == 422 and client.calls == []


def test_username_never_in_logs_or_responses(client, caplog):
    secret = "Unique-Login-42"
    with caplog.at_level(logging.DEBUG):
        r = client.post("/recommend/github", json={"username": secret})
    assert secret.lower() not in r.text.lower()
    assert secret.lower() not in caplog.text.lower()
    h = hash_username(secret, b"test-salt")
    assert h not in caplog.text  # not even the hash is logged


def test_cors_allows_only_configured_origin(client):
    ok = client.options(
        "/recommend/onboarding",
        headers={"Origin": "https://firstpr.example", "Access-Control-Request-Method": "POST"},
    )
    assert ok.headers.get("access-control-allow-origin") == "https://firstpr.example"
    bad = client.options(
        "/recommend/onboarding",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in bad.headers


def test_own_repos_dropped_from_stars(monkeypatch):
    import requests as rq

    from firstpr.serve import github as gh

    class R:
        status_code = 200

        def json(self):
            def mk(o, i):
                return {
                    "starred_at": "2026-01-01",
                    "repo": {
                        "id": i,
                        "full_name": f"{o}/r{i}",
                        "owner": {"login": o},
                        "language": None,
                        "topics": [],
                    },
                }

            return [mk("Me", 1), mk("other", 2)]

        def raise_for_status(self):
            pass

    monkeypatch.setattr(rq, "get", lambda *a, **k: R())
    got = gh.fetch_stars("me", token=None, max_pages=1)
    assert [s.full_name for s in got] == ["other/r2"]
    with pytest.raises(gh.InvalidUsername):
        gh.fetch_stars("-bad-", token=None)
