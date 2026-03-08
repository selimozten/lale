"""Run data generation across all categories with limited concurrency."""

from __future__ import annotations

import multiprocessing
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lale.generate.generate_data import CATEGORIES, GenerationConfig, generate

CONCURRENCY = 2
NUM_EXAMPLES = 10_000
BATCH_SIZE = 15


def run_category(category: str) -> None:
    config = GenerationConfig(
        category=category,
        num_examples=NUM_EXAMPLES,
        batch_size=BATCH_SIZE,
        output_dir=Path("data/raw"),
        api_key=os.environ["BEDROCK_API_KEY"],
    )
    generate(config)


if __name__ == "__main__":
    if "BEDROCK_API_KEY" not in os.environ:
        print("Set BEDROCK_API_KEY env var")
        sys.exit(1)

    print(f"Generating {NUM_EXAMPLES} examples x {len(CATEGORIES)} categories")
    print(f"Concurrency: {CONCURRENCY}, batch size: {BATCH_SIZE}")

    with multiprocessing.Pool(processes=CONCURRENCY) as pool:
        pool.map(run_category, CATEGORIES)

    print("\nAll categories complete.")
