"""Compare evaluation results across multiple models."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel


class ComparisonRow(BaseModel):
    model: str
    scores: dict[str, float | str]


def load_results(results_dir: Path) -> list[ComparisonRow]:
    """Load all evaluation result files from a directory."""
    rows: list[ComparisonRow] = []
    for path in sorted(results_dir.glob("*.json")):
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        rows.append(ComparisonRow(
            model=data.get("model", path.stem),
            scores=data.get("scores", {}),
        ))
    return rows


def make_table(rows: list[ComparisonRow]) -> str:
    """Generate a markdown comparison table."""
    if not rows:
        return "No results found."

    # Collect all score keys
    all_keys: list[str] = []
    for row in rows:
        for key in row.scores:
            if key not in all_keys:
                all_keys.append(key)

    # Header
    header = "| Model | " + " | ".join(all_keys) + " |"
    separator = "|-------|" + "|".join("------" for _ in all_keys) + "|"

    # Rows
    lines = [header, separator]
    for row in rows:
        cells = [row.model]
        for key in all_keys:
            val = row.scores.get(key, "-")
            if isinstance(val, float):
                cells.append(f"{val:.4f}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def compare(results_dir: Path, output_path: Path | None = None) -> str:
    """Load results, build comparison table, optionally save."""
    rows = load_results(results_dir)
    table = make_table(rows)

    print(table)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(table, encoding="utf-8")
        print(f"\nTable saved to: {output_path}")

    return table
