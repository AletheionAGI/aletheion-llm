# Pull Request

## Description

<!-- Provide a brief description of your changes -->

## Type of Change

<!-- Mark the relevant option with an [x] -->

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Code refactoring
- [ ] Performance improvement
- [ ] Test addition/improvement
- [ ] CI/CD update
- [ ] Other (please describe):

## Related Issues

<!-- Link to related issues using # followed by issue number -->

Fixes #
Relates to #

## Motivation and Context

<!-- Why is this change required? What problem does it solve? -->

## Changes Made

<!-- Provide a detailed list of changes -->

- Change 1
- Change 2
- Change 3

## Testing Performed

<!-- Describe the tests you ran to verify your changes -->

### Test Environment
- **OS**:
- **Python version**:
- **PyTorch version**:

### Test Cases
- [ ] All existing tests pass
- [ ] Added new tests for new functionality
- [ ] Tested manually with the following scenarios:
  1. Scenario 1
  2. Scenario 2

### Test Commands
```bash
# Commands used for testing
pytest tests/
```

### Test Results
```
# Paste relevant test output
```

## Performance Impact

<!-- If applicable, describe any performance implications -->

- [ ] No performance impact
- [ ] Performance improved
- [ ] Performance degraded (please explain why this is acceptable)
- [ ] Not applicable

**Benchmarks** (if applicable):
```
Before: ...
After: ...
```

## Breaking Changes

<!-- List any breaking changes and migration instructions -->

- [ ] No breaking changes
- [ ] Breaking changes (described below)

**Breaking Changes Details**:
<!-- Describe what breaks and how users should migrate -->

**Migration Guide**:
```python
# Before
...

# After
...
```

## Documentation

- [ ] Documentation updated (if needed)
- [ ] Docstrings added/updated for public APIs
- [ ] README updated (if needed)
- [ ] CHANGELOG.md updated
- [ ] Examples added/updated (if applicable)
- [ ] No documentation needed

## Code Quality Checklist

<!-- Ensure all items are checked before requesting review -->

- [ ] Code follows the project's style guidelines (Black, Ruff, isort)
- [ ] Pre-commit hooks pass (`pre-commit run --all-files`)
- [ ] All tests pass locally (`pytest`)
- [ ] Type hints added where appropriate
- [ ] Comments added for complex logic
- [ ] No unnecessary print statements or debug code
- [ ] Dependencies updated in pyproject.toml (if added)
- [ ] Commit messages follow Conventional Commits format

## Review Checklist

<!-- For reviewers -->

- [ ] Code is readable and well-documented
- [ ] Tests adequately cover new functionality
- [ ] No security vulnerabilities introduced
- [ ] Performance implications considered
- [ ] Breaking changes properly documented
- [ ] Follows project architecture and patterns

## Screenshots/Outputs

<!-- If applicable, add screenshots or example outputs -->

## Additional Notes

<!-- Any additional information that reviewers should know -->

## Reviewer Guidelines

Please review:
1. **Functionality**: Does it work as intended?
2. **Tests**: Are there adequate tests with good coverage?
3. **Documentation**: Is it properly documented?
4. **Code Quality**: Does it follow style guidelines?
5. **Performance**: Are there performance implications?
6. **Security**: Are there security considerations?

---

**By submitting this pull request, I confirm that:**
- [ ] My contribution is licensed under the AGPL-3.0-or-later license
- [ ] I have read and agree to the project's [Code of Conduct](../CODE_OF_CONDUCT.md)
- [ ] I have followed the [Contributing Guidelines](../CONTRIBUTING.md)
- [ ] I retain copyright to my contributions

---

Thank you for contributing to Aletheion LLM! 🎉
