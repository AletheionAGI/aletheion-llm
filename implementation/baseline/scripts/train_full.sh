#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "🚀 Training baseline model with default configuration..."
python train.py --config config/default.yaml

echo "✅ Training complete!"
