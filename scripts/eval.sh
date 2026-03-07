#!/usr/bin/env bash
set -euo pipefail

# Evaluate model on terazi
# Usage: ./scripts/eval.sh <model_path>

MODEL="${1:?Usage: $0 <model_path>}"

echo "=== Evaluating: $MODEL ==="
lale eval --model "$MODEL" --benchmark terazi
echo "=== Comparing all results ==="
lale compare --results-dir results
echo "=== Done ==="
