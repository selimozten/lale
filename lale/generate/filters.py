"""Quality filtering for generated instruction data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from langdetect import detect
from pydantic import BaseModel
from rapidfuzz import fuzz
from tqdm import tqdm

MIN_CONTENT_LENGTH = 20
MAX_CONTENT_LENGTH = 8000
FUZZY_DEDUP_THRESHOLD = 90


class FilterStats(BaseModel):
    total: int = 0
    passed: int = 0
    rejected_short: int = 0
    rejected_long: int = 0
    rejected_language: int = 0
    rejected_format: int = 0
    rejected_duplicate: int = 0


def _extract_text(example: dict[str, Any]) -> str:
    """Extract all text content from an example for analysis."""
    parts: list[str] = []
    for msg in example.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
    return " ".join(parts)


def check_format(example: dict[str, Any]) -> bool:
    """Validate the structure of an example."""
    messages = example.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    for msg in messages:
        if not isinstance(msg, dict):
            return False
        if "role" not in msg or "content" not in msg:
            return False
        if msg["role"] not in ("user", "assistant", "system"):
            return False
        if not isinstance(msg["content"], str) or not msg["content"].strip():
            return False
    return True


def check_length(example: dict[str, Any]) -> bool:
    """Check that content is within acceptable length bounds."""
    for msg in example.get("messages", []):
        content = msg.get("content", "")
        if len(content) < MIN_CONTENT_LENGTH:
            return False
        if len(content) > MAX_CONTENT_LENGTH:
            return False
    return True


def check_language(example: dict[str, Any]) -> bool:
    """Check that the primary language is Turkish.

    Only checks user and assistant messages (skips system messages, which
    may legitimately be in English).
    """
    parts: list[str] = []
    for msg in example.get("messages", []):
        if msg.get("role") in ("user", "assistant"):
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(content)
    text = " ".join(parts)
    if len(text) < 30:
        return True  # Too short to detect reliably, let it pass
    try:
        lang = detect(text)
        return lang == "tr"
    except Exception:
        return True  # On detection failure, let it pass


def content_hash(example: dict[str, Any]) -> str:
    """Compute a hash of the user message for exact deduplication."""
    user_msgs = [
        m["content"] for m in example.get("messages", []) if m.get("role") == "user"
    ]
    key = "||".join(user_msgs)
    return hashlib.sha256(key.encode()).hexdigest()


def filter_dataset(
    input_path: Path,
    output_path: Path,
    skip_language_check: bool = False,
) -> FilterStats:
    """Filter a JSONL dataset, writing passing examples to output."""
    stats = FilterStats()
    seen_hashes: set[str] = set()
    seen_texts: list[str] = []

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open(encoding="utf-8") as fin:
        lines = fin.readlines()

    with output_path.open("w", encoding="utf-8") as fout:
        for line in tqdm(lines, desc=f"Filtering {input_path.name}"):
            line = line.strip()
            if not line:
                continue

            stats.total += 1

            try:
                example = json.loads(line)
            except json.JSONDecodeError:
                stats.rejected_format += 1
                continue

            # Format check
            if not check_format(example):
                stats.rejected_format += 1
                continue

            # Length check
            if not check_length(example):
                content_lens = [len(m.get("content", "")) for m in example.get("messages", [])]
                if any(l < MIN_CONTENT_LENGTH for l in content_lens):
                    stats.rejected_short += 1
                else:
                    stats.rejected_long += 1
                continue

            # Language check
            if not skip_language_check and not check_language(example):
                stats.rejected_language += 1
                continue

            # Exact dedup
            h = content_hash(example)
            if h in seen_hashes:
                stats.rejected_duplicate += 1
                continue
            seen_hashes.add(h)

            # Fuzzy dedup
            user_text = _extract_text(example)
            is_dup = False
            for prev in seen_texts:
                if fuzz.ratio(user_text, prev) > FUZZY_DEDUP_THRESHOLD:
                    is_dup = True
                    break
            if is_dup:
                stats.rejected_duplicate += 1
                continue
            seen_texts.append(user_text)

            fout.write(json.dumps(example, ensure_ascii=False) + "\n")
            stats.passed += 1

    return stats
