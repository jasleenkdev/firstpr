"""Privacy boundary for GitHub data.

- Actor logins never leave BigQuery: owner and bot filters run in SQL, only numeric actor ids are
  downloaded, and they are replaced by a salted hash in memory before anything is written.
- `find_leaks` checks a frame against a set of forbidden raw values before it is written.
- `scrub_text` removes @mentions, emails and URLs from README / issue text before it reaches an
  LLM prompt (they can name people).
"""

import hashlib
import hmac
import re
from collections.abc import Iterable

import numpy as np
import pandas as pd

from firstpr.utils.env import require_env

HASH_HEX = 16  # 64-bit ids: collision probability ~1e-6 for 10M users
HASH_PATTERN = re.compile(rf"^[0-9a-f]{{{HASH_HEX}}}$")


def load_salt() -> bytes:
    return require_env("ID_SALT").encode()


def hash_ids(ids: Iterable[object], salt: bytes) -> np.ndarray:
    """HMAC-SHA256(salt, str(id)) truncated to HASH_HEX hex chars, one per input id."""
    return np.array(
        [hmac.new(salt, str(i).encode(), hashlib.sha256).hexdigest()[:HASH_HEX] for i in ids],
        dtype=object,
    )


def hash_column(df: pd.DataFrame, column: str, salt: bytes, out: str = "user") -> pd.DataFrame:
    """Replace a raw id column by its salted hash (hashing each distinct id once)."""
    codes, uniques = pd.factorize(df[column])
    hashed = hash_ids(uniques, salt)[codes]
    return df.drop(columns=column).assign(**{out: hashed})


def find_leaks(
    df: pd.DataFrame, forbidden: set[str], skip: Iterable[str] = (), text_only: bool = False
) -> list[str]:
    """Columns (other than `skip`) holding any value from `forbidden` (case-insensitive).
    `text_only` scans string columns only (for logins: some are all digits, like ids)."""
    lowered = {f.lower() for f in forbidden}
    bad = []
    for col in df.columns:
        if col in skip:
            continue
        if text_only and not (
            pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])
        ):
            continue
        values = df[col].dropna()
        if values.empty:
            continue
        as_text = values.astype(str).str.lower()
        if as_text.isin(lowered).any():
            bad.append(col)
    return bad


def assert_hashed(series: pd.Series) -> None:
    """Every value is a HASH_HEX-char lowercase hex string."""
    ok = series.astype(str).str.fullmatch(HASH_PATTERN.pattern)
    if not ok.all():
        raise ValueError(f"{int((~ok).sum())} values of {series.name!r} are not salted hashes")


_MENTION = re.compile(r"(?<![\w.])@[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_URL = re.compile(r"https?://\S*|www\.\S+")
_HTML = re.compile(r"<[^>]+>")
_CODE_FENCE = re.compile(r"```.*?```", re.S)


def scrub_text(text: str, max_chars: int | None = None) -> str:
    """Plain text without people identifiers (emails, @mentions), URLs, images, HTML or code
    blocks; whitespace collapsed; optionally cut at a word boundary."""
    t = _CODE_FENCE.sub(" ", text or "")
    t = _MD_IMAGE.sub(" ", t)
    t = _MD_LINK.sub(r"\1", t)
    t = _EMAIL.sub("[email]", t)  # before URLs and mentions: emails contain '@'
    t = _URL.sub(" ", t)
    t = _HTML.sub(" ", t)
    t = _MENTION.sub("@user", t)
    t = " ".join(t.split())
    if max_chars is not None and len(t) > max_chars:
        t = t[:max_chars].rsplit(" ", 1)[0] + " ..."
    return t
