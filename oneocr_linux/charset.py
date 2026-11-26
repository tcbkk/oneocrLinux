"""Per-head token tables used by the OneOCR recognizers.

Every recognizer head has its own private ordering above token 95 (ASCII).
Tables are recovered by probing each glyph (``tools/build_charset_multi.py``)
and stored as ``data/charset_<classes>.json``; the 32632 CJK head also loads
the larger Chinese table in ``data/charset.json``.
"""

from __future__ import annotations

import json
import os
import re

FIRST_CODEPOINT = 32
CJK_HEAD = 32632
# right-to-left heads: the recognizer emits visual order, we reverse to logical
RTL_HEADS = {221}

_CACHE = {}


def _load_file(name):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", name)
    try:
        with open(path, encoding="utf-8") as f:
            return {int(k): v for k, v in json.load(f).get("tokens", {}).items()}
    except (OSError, ValueError):
        return {}


def table_for(classes: int):
    if classes in _CACHE:
        return _CACHE[classes]
    merged = {}
    if classes == CJK_HEAD:
        merged.update(_load_file("charset.json"))
    merged.update(_load_file("charset_%d.json" % classes))
    _CACHE[classes] = merged
    return merged


def token_to_char(token: int, classes: int = CJK_HEAD) -> str:
    if token < 0:
        raise ValueError("negative token id")
    if token < 95:  # ASCII block is exact in every head
        return chr(token + FIRST_CODEPOINT)
    ch = table_for(classes).get(token)
    if ch is not None:
        return ch
    return chr(token + FIRST_CODEPOINT)


def decode_ctc(tokens, blank: int, classes: int = CJK_HEAD) -> str:
    """Greedy CTC decode; RTL heads are reversed back to logical order.

    Collapsing runs happens in time order first (CTC semantics); only the
    resulting character sequence is mirrored for RTL scripts.
    """
    table = table_for(classes)
    out = []
    prev = -1
    for t in tokens:
        t = int(t)
        if t != prev and t != blank:
            if t < 95:
                ch = chr(t + FIRST_CODEPOINT)
            else:
                ch = table.get(t, "?")
            if ch:
                out.append(ch)
        prev = t
    text = "".join(out)
    if classes in RTL_HEADS:
        # RTL: mirror the whole run, then restore the digit runs (digits stay LTR)
        text = text[::-1]
        text = re.sub(r"[0-9\u0660-\u0669\u06f0-\u06f9]{2,}",
                      lambda m: m.group()[::-1], text)
    return text