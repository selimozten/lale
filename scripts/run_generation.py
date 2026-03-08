"""Run data generation across all categories sequentially (each uses 20 concurrent workers)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lale.generate.generate_data import CATEGORIES, GenerationConfig, generate

NUM_EXAMPLES = 10_000


if __name__ == "__main__":
    if "BEDROCK_API_KEY" not in os.environ:
        print("Set BEDROCK_API_KEY env var")
        sys.exit(1)

    print(f"Generating {NUM_EXAMPLES} examples x {len(CATEGORIES)} categories")

    for category in CATEGORIES:
        config = GenerationConfig(
            category=category,
            num_examples=NUM_EXAMPLES,
            output_dir=Path("data/raw"),
            api_key=os.environ["BEDROCK_API_KEY"],
        )
        generate(config)

    print("\nAll categories complete.")
