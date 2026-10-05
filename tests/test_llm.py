import numpy as np

from firstpr.llm.client import LLMClient
from firstpr.llm.kar import user_prompt
from firstpr.llm.rerank import borda, format_prompt, parse_ranking, rerank_user


def test_parse_ranking_dedupes_and_appends_missing():
    order, missing = parse_ranking("[3] > [1] > [3] > [9] > [x]", 4)
    assert order == [2, 0, 1, 3] and missing == 2


def test_borda_combines_rankings():
    assert borda([[0, 1, 2], [1, 0, 2], [1, 2, 0]], 3).tolist() == [1, 0, 2]


def test_prompt_has_no_ids_and_shows_scores_only_when_asked():
    p = format_prompt(["Old caliper"], ["A", "B"], None, 50)
    assert "[1] A" in p and "score" not in p
    assert "(recommender score: 0.25)" in format_prompt(["x"], ["A"], [0.25], 50)
    assert user_prompt(["a", "b", "c"], 2, 50).count("\n2. ") == 1  # only the last 2 items


def test_rerank_user_undoes_presentation_order():
    titles = [f"item{i}" for i in range(5)]

    def oracle(prompt: str) -> str:  # always prefers higher item numbers, wherever shown
        lines = [ln for ln in prompt.splitlines() if ln.startswith("[")]
        shown = [int(ln.split("item")[1]) for ln in lines]
        return " > ".join(f"[{shown.index(i) + 1}]" for i in sorted(shown, reverse=True))

    ranked, missing = rerank_user(
        oracle, ["h"], np.array([0, 1, 2, 3, 4]), titles, None, 3, np.random.default_rng(0), 50
    )
    assert ranked.tolist() == [4, 3, 2, 1, 0] and missing == 0


def test_client_cache_avoids_second_call(tmp_path, monkeypatch):
    c = LLMClient("m", cache_dir=tmp_path)
    calls = []
    monkeypatch.setattr(c, "_request", lambda p, o: calls.append(p) or "out")
    assert c.generate("hi", {"temperature": 0}) == "out"
    assert c.generate("hi", {"temperature": 0}) == "out"
    assert len(calls) == 1
    c2 = LLMClient("m", cache_dir=tmp_path)  # cache persists on disk
    assert c2.cached("hi", {"temperature": 0}) == "out"


def test_groq_invalid_json_returns_raw_generation(tmp_path, monkeypatch):
    import requests

    class Resp:
        status_code = 400
        headers: dict = {}

        def json(self):
            return {"error": {"code": "json_validate_failed", "failed_generation": "{broken"}}

    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setattr(requests, "post", lambda *a, **k: Resp())
    c = LLMClient("m", backend="groq", cache_dir=tmp_path)
    assert c.generate("p", {"json": True}) == "{broken"
