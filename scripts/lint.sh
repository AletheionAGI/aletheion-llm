#!/usr/bin/env bash
# Run all linters and code quality checks

set -e

EXIT_CODE=0

echo "==> Running Ruff linter..."
if ! ruff check .; then
    echo "❌ Ruff found issues"
    EXIT_CODE=1
fi

echo ""
echo "==> Checking code formatting with Black..."
if ! black --check .; then
    echo "❌ Black formatting issues found"
    echo "Run './scripts/format.sh' to fix formatting"
    EXIT_CODE=1
fi

echo ""
echo "==> Checking import sorting with isort..."
if ! isort --check .; then
    echo "❌ isort found issues"
    echo "Run './scripts/format.sh' to fix import sorting"
    EXIT_CODE=1
fi

echo ""
echo "==> Running type checker (mypy)..."
if ! mypy src/ --ignore-missing-imports; then
    echo "⚠️  Type checking issues found (non-blocking)"
    # Don't fail on type errors for now
fi

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ All linters passed!"
else
    echo "❌ Some linters failed. Please fix the issues above."
fi

exit $EXIT_CODE
