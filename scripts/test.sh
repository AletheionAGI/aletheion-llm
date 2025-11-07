#!/usr/bin/env bash
# Run tests with coverage

set -e

# Parse command line arguments
PYTEST_ARGS="${@:---cov=src --cov-report=term-missing --cov-report=html}"

echo "==> Running tests with pytest..."
echo "Arguments: $PYTEST_ARGS"
echo ""

pytest $PYTEST_ARGS

echo ""
echo "✅ Tests complete!"
echo ""
echo "Coverage report saved to htmlcov/index.html"
echo "Open it with: open htmlcov/index.html  (macOS) or xdg-open htmlcov/index.html  (Linux)"
