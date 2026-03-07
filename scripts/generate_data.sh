#!/usr/bin/env bash
set -euo pipefail

# Generate Turkish instruction data across all categories
# Usage: ./scripts/generate_data.sh [num_examples_per_category]

NUM="${1:-10000}"

CATEGORIES=(general reasoning tool_use finance legal code translation)

for cat in "${CATEGORIES[@]}"; do
    echo "=== Generating: $cat ($NUM examples) ==="
    lale generate --category "$cat" --num-examples "$NUM"
done

echo "=== All categories done ==="
echo "Filtering..."
lale prepare --input data/raw --output data/filtered
echo "Done."
