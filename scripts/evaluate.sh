#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <checkpoint_path>"
  exit 1
fi

cd "$(dirname "$0")/.."

python eval.py --checkpoint "$1"
