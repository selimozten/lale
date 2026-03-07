#!/usr/bin/env bash
set -euo pipefail

# Train lale model
# Usage: ./scripts/train.sh [config_path]

CONFIG="${1:-lale/train/config.yaml}"

echo "=== Training lale ==="
echo "Config: $CONFIG"
lale train --config "$CONFIG"
echo "=== Done ==="
