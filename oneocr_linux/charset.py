"""Token table used by the OneOCR recognizers.

The recognizers are CTC heads over a dense code-point table: token ``i`` maps to
``chr(i + 32)`` and the last index is the CTC blank.  This was verified against
real recognizer outputs, e.g. "Hello 123" decodes to ids
``40 69 76 76 79 17 18 19`` == ``H e l l o 1 2 3``.

The table covers U+0020..U+7F97 for the 32632-class head, which includes ASCII,
Latin-1, Greek, Cyrillic, kana and the CJK block up to U+7F97.
"""

from __future__ import annotations

FIRST_CODEPOINT = 32


def token_to_char(token: int) -> str:
    """Map a class index to its character. Raises ValueError for the blank index."""
    if token < 0:
        raise ValueError("negative token id")
    return chr(token + FIRST_CODEPOINT)


def decode_ctc(tokens, blank: int) -> str:
    """Greedy CTC decode: drop blanks, collapse runs, map ids to characters."""
    out = []
    prev = -1
    for t in tokens:
        t = int(t)
        if t != prev and t != blank:
            if t + FIRST_CODEPOINT <= 0x10FFFF:
                out.append(chr(t + FIRST_CODEPOINT))
        prev = t
    return "".join(out)