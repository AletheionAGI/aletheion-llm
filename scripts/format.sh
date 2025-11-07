#!/usr/bin/env bash
# Format code with Black and isort

set -e

echo "==> Formatting code with Black..."
black .

echo ""
echo "==> Sorting imports with isort..."
isort .

echo ""
echo "✅ Code formatting complete!"
