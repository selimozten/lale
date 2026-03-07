"""Tests for quality filtering logic."""

from lale.generate.filters import check_format, check_length, content_hash


def _make_example(user: str = "Merhaba", assistant: str = "Merhaba, size nasil yardimci olabilirim?"):
    return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]}


def test_check_format_valid():
    assert check_format(_make_example()) is True


def test_check_format_missing_role():
    assert check_format({"messages": [{"content": "hi"}, {"role": "assistant", "content": "ok"}]}) is False


def test_check_format_single_message():
    assert check_format({"messages": [{"role": "user", "content": "hi"}]}) is False


def test_check_format_empty_content():
    assert check_format({"messages": [{"role": "user", "content": ""}, {"role": "assistant", "content": "ok"}]}) is False


def test_check_length_valid():
    assert check_length(_make_example(
        user="Bu bir test mesajidir ve yeterli uzunluktadir",
        assistant="Bu da yeterli uzunlukta bir yanit mesajidir",
    )) is True


def test_check_length_too_short():
    assert check_length(_make_example(user="kisa", assistant="ok")) is False


def test_content_hash_deterministic():
    ex = _make_example()
    assert content_hash(ex) == content_hash(ex)


def test_content_hash_different():
    ex1 = _make_example(user="soru bir")
    ex2 = _make_example(user="soru iki")
    assert content_hash(ex1) != content_hash(ex2)
