import sys

from oneocr_linux.charset import decode_ctc, token_to_char


def test_ascii_mapping():
    assert token_to_char(40) == "H"
    assert token_to_char(69) == "e"
    assert token_to_char(17) == "1"


def test_decode_hello():
    ids = [32631, 40, 32631, 69, 32631, 76, 32631, 76, 79, 32631, 17, 18, 19]
    assert decode_ctc(ids, 32631) == "Hello123"


def test_collapse_repeats_and_blanks():
    # token 0 maps to chr(32) == " "; repeats collapse, blank ids drop
    assert decode_ctc([40, 40, 40, 0, 0], 999) == "H "
    assert decode_ctc([32631, 32631], 32631) == ""


if __name__ == "__main__":
    sys.exit(0)