#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "🚀 Training small baseline model..."
python train.py --config config/small.yaml

echo "✅ Training complete!"
