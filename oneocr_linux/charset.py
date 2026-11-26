"""Per-head token tables used by the OneOCR recognizers."""

from __future__ import annotations

import json
import os

FIRST_CODEPOINT = 32
CJK_HEAD = 32632
# right-to-left heads: visual order must be mirrored back to logical order
RTL_HEADS = {221}
# Arabic-Indic digit tokens in head 221 (visual order == ٠..٩)
AR_DIGITS = {149: "٠", 150: "١", 151: "٢", 152: "٣", 153: "٤",
             154: "٥", 155: "٦", 156: "٧", 157: "٨", 158: "٩"}

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
    if classes in RTL_HEADS:
        merged.update(AR_DIGITS)
    _CACHE[classes] = merged
    return merged


def token_to_char(token: int, classes: int = CJK_HEAD) -> str:
    if token < 0:
        raise ValueError("negative token id")
    if token < 95:
        return chr(token + FIRST_CODEPOINT)
    ch = table_for(classes).get(token)
    if ch is not None:
        return ch
    return chr(token + FIRST_CODEPOINT)


def _bidi_tokens(tokens, classes):
    """Mirror the token stream, but keep each run of digit tokens forward.

    Arabic runs right-to-left, so the visual token order is reversed; a run of
    digits however is written left-to-right and must not be mirrored inside.
    """
    toks = list(reversed(tokens))
    out = []
    i = 0
    while i < len(toks):
        if classes == 221 and toks[i] in AR_DIGITS:
            j = i
            while j < len(toks) and toks[j] in AR_DIGITS:
                j += 1
            out.extend(reversed(toks[i:j]))
            i = j
        else:
            out.append(toks[i])
            i += 1
    return out


def decode_ctc(tokens, blank: int, classes: int = CJK_HEAD) -> str:
    # 1) CTC collapse in time order
    col = []
    prev = -1
    for t in tokens:
        t = int(t)
        if t != prev and t != blank:
            col.append(t)
        prev = t
    # 2) bidi reorder
    if classes in RTL_HEADS:
        col = _bidi_tokens(col, classes)
    # 3) map tokens
    table = table_for(classes)
    out = []
    for t in col:
        if t < 95:
            out.append(chr(t + FIRST_CODEPOINT))
        else:
            out.append(table.get(t, "?"))
    return "".join(out)