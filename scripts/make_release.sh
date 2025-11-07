#!/usr/bin/env bash
# Create a new release
# Usage: ./scripts/make_release.sh <version>
# Example: ./scripts/make_release.sh 0.2.0

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if version argument is provided
if [ -z "$1" ]; then
    echo -e "${RED}Error: Version number required${NC}"
    echo "Usage: ./scripts/make_release.sh <version>"
    echo "Example: ./scripts/make_release.sh 0.2.0"
    exit 1
fi

VERSION=$1
TAG="v${VERSION}"

echo -e "${GREEN}==> Creating release ${TAG}${NC}"
echo ""

# Check if we're on main branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo -e "${YELLOW}Warning: You are on branch '${CURRENT_BRANCH}', not 'main'${NC}"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo -e "${RED}Error: You have uncommitted changes${NC}"
    git status --short
    exit 1
fi

# Check if tag already exists
if git rev-parse "$TAG" >/dev/null 2>&1; then
    echo -e "${RED}Error: Tag $TAG already exists${NC}"
    exit 1
fi

echo -e "${GREEN}Step 1: Updating version in pyproject.toml${NC}"
# Update version in pyproject.toml
sed -i.bak "s/^version = \".*\"/version = \"${VERSION}\"/" pyproject.toml
rm pyproject.toml.bak 2>/dev/null || true

echo -e "${GREEN}Step 2: Checking CHANGELOG.md${NC}"
# Check if CHANGELOG has entry for this version
if ! grep -q "## \[${VERSION}\]" CHANGELOG.md; then
    echo -e "${YELLOW}Warning: CHANGELOG.md does not contain entry for version ${VERSION}${NC}"
    echo "Please update CHANGELOG.md before continuing"
    read -p "Open CHANGELOG.md now? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ${EDITOR:-nano} CHANGELOG.md
    fi
    read -p "Continue with release? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        # Revert version change
        git checkout pyproject.toml
        exit 1
    fi
fi

echo -e "${GREEN}Step 3: Running tests${NC}"
./scripts/test.sh || {
    echo -e "${RED}Tests failed! Aborting release.${NC}"
    git checkout pyproject.toml
    exit 1
}

echo -e "${GREEN}Step 4: Running linters${NC}"
./scripts/lint.sh || {
    echo -e "${RED}Linting failed! Aborting release.${NC}"
    git checkout pyproject.toml
    exit 1
}

echo -e "${GREEN}Step 5: Building package${NC}"
python -m build || {
    echo -e "${RED}Build failed! Aborting release.${NC}"
    git checkout pyproject.toml
    exit 1
}

echo -e "${GREEN}Step 6: Checking package with twine${NC}"
twine check dist/* || {
    echo -e "${RED}Package check failed! Aborting release.${NC}"
    git checkout pyproject.toml
    exit 1
}

echo ""
echo -e "${GREEN}All checks passed!${NC}"
echo ""
echo "The following will be done:"
echo "  1. Commit version bump to pyproject.toml"
echo "  2. Create and push tag ${TAG}"
echo "  3. GitHub Actions will automatically:"
echo "     - Build and test the package"
echo "     - Publish to PyPI"
echo "     - Create GitHub release"
echo ""
read -p "Proceed with release? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Release cancelled"
    git checkout pyproject.toml
    exit 1
fi

echo -e "${GREEN}Step 7: Committing version bump${NC}"
git add pyproject.toml
git commit -m "chore: bump version to ${VERSION}"

echo -e "${GREEN}Step 8: Creating and pushing tag${NC}"
git tag -a "$TAG" -m "Release ${VERSION}"
git push origin main
git push origin "$TAG"

echo ""
echo -e "${GREEN}✅ Release ${TAG} initiated!${NC}"
echo ""
echo "Monitor the release workflow at:"
echo "  https://github.com/AletheionAGI/aletheion-llm/actions"
echo ""
echo "Once complete, the release will be available at:"
echo "  - PyPI: https://pypi.org/project/aletheion-llm/${VERSION}/"
echo "  - GitHub: https://github.com/AletheionAGI/aletheion-llm/releases/tag/${TAG}"
