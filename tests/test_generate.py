"""Tests for data generation parsing logic."""

from lale.generate.generate_data import parse_response, load_template, CATEGORIES, TEMPLATES_DIR


def test_parse_valid_json_array():
    raw = '[{"messages": [{"role": "user", "content": "Merhaba"}, {"role": "assistant", "content": "Merhaba!"}]}]'
    examples = parse_response(raw, "general")
    assert len(examples) == 1
    assert examples[0].category == "general"
    assert examples[0].messages[0]["role"] == "user"


def test_parse_markdown_wrapped():
    raw = '```json\n[{"messages": [{"role": "user", "content": "Test"}, {"role": "assistant", "content": "Yanit"}]}]\n```'
    examples = parse_response(raw, "reasoning")
    assert len(examples) == 1


def test_parse_invalid_json():
    examples = parse_response("this is not json", "general")
    assert examples == []


def test_parse_missing_messages():
    raw = '[{"foo": "bar"}]'
    examples = parse_response(raw, "general")
    assert examples == []


def test_all_templates_exist():
    for category in CATEGORIES:
        path = TEMPLATES_DIR / f"{category}.txt"
        assert path.exists(), f"Missing template: {path}"


def test_load_template():
    text = load_template("general")
    assert len(text) > 0
    assert "Turkce" in text
