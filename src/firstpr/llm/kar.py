"""KAR-style knowledge generation (Xi et al. 2024, "Towards Open-World Recommendation with
Knowledge Augmentation from Large Language Models").

Two prompts, both grounded in the input only:
- item factual knowledge: from the product listing (title, brand, category, features,
  description), what the product is, what it is used for, and who would buy it;
- user preference reasoning: from the titles of the user's past products (oldest first), what
  the user is interested in and is likely to want next.

The user prompt uses the train history *without its last item* (see models/sasrec_text.py:
the last train item is then a target the LLM never saw) and contains product titles only, never
any user identifier.
"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from tqdm import tqdm

from firstpr.llm.client import LLMClient

ITEM_PROMPT = """You are given a product listing from an online store.

{text}

Using only the information in the listing, write 2-3 short sentences: what the product is, what \
it is used for, and what kind of buyer needs it. Do not add facts that are not in the listing. \
Answer with the sentences only."""

USER_PROMPT = """A customer bought these products, oldest first:
{titles}

Based only on this list, write 2-3 short sentences describing the customer's interests and \
needs, and the kinds of products they are likely to want next. Do not guess personal details. \
Answer with the sentences only."""


def truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + " ..."


def item_prompt(text: str) -> str:
    return ITEM_PROMPT.format(text=text)


def user_prompt(titles: list[str], max_items: int, max_title_chars: int) -> str:
    recent = titles[-max_items:]
    lines = "\n".join(f"{i + 1}. {truncate(t, max_title_chars)}" for i, t in enumerate(recent))
    return USER_PROMPT.format(titles=lines)


def generate_all(
    client: LLMClient,
    prompts: list[str],
    options: dict[str, Any],
    workers: int = 1,
    clean: Callable[[str], str] = str.strip,
) -> list[str]:
    """Responses in prompt order; cached prompts are free, the rest go to the backend."""
    out: list[str] = [""] * len(prompts)
    todo = []
    for i, p in enumerate(prompts):
        hit = client.cached(p, options)
        if hit is None:
            todo.append(i)
        else:
            out[i] = clean(hit)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {i: ex.submit(client.generate, prompts[i], options) for i in todo}
        for i in tqdm(todo, desc=f"llm {client.model}", disable=not todo):
            out[i] = clean(futures[i].result())
    return out
