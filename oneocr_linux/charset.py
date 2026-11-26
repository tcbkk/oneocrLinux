"""Token table used by the OneOCR recognizers.

The recognizers are CTC heads. The first 95 tokens are plain ASCII
(``chr(i + 32)``); everything above is a private ordering, so Chinese must come
from a recovered table (`data/charset.json`, built by ``tools/build_charset.py``
by probing each glyph). Tokens outside the table fall back to ``chr(i + 32)``.
"""

from __future__ import annotations

import json
import os

FIRST_CODEPOINT = 32

_TABLE = None


def _table():
    global _TABLE
    if _TABLE is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "charset.json")
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            _TABLE = {int(k): v for k, v in data.get("tokens", {}).items()}
        except (OSError, ValueError):
            _TABLE = {}
    return _TABLE


def token_to_char(token: int) -> str:
    """Map a class index to its character. Raises ValueError for the blank index."""
    if token < 0:
        raise ValueError("negative token id")
    if token < 95:  # ASCII block is exact, never trust the probe table here
        return chr(token + FIRST_CODEPOINT)
    ch = _table().get(token)
    if ch is not None:
        return ch
    return chr(token + FIRST_CODEPOINT)


def decode_ctc(tokens, blank: int) -> str:
    """Greedy CTC decode: drop blanks, collapse runs, map ids to characters."""
    table = _table()
    out = []
    prev = -1
    for t in tokens:
        t = int(t)
        if t != prev and t != blank:
            ch = chr(t + FIRST_CODEPOINT) if t < 95 else table.get(t)
            if ch is None:
                if t + FIRST_CODEPOINT <= 0x10FFFF:
                    ch = chr(t + FIRST_CODEPOINT)
                else:
                    ch = ""
            if ch:
                out.append(ch)
        prev = t
    return "".join(out)