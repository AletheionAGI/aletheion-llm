---
name: Bug Report
about: Create a report to help us improve Aletheion LLM
title: '[BUG] '
labels: bug
assignees: ''

---

## Bug Description

A clear and concise description of what the bug is.

## To Reproduce

Steps to reproduce the behavior:
1. Install with '...'
2. Run command '....'
3. Use configuration '....'
4. See error

## Expected Behavior

A clear and concise description of what you expected to happen.

## Actual Behavior

What actually happened instead.

## Minimal Reproducible Example

```python
# Please provide a minimal code example that reproduces the issue
import torch
from aletheion_llm import ...

# Your code here
```

## Error Message/Stack Trace

```
Paste the full error message and stack trace here
```

## Environment

Please complete the following information:

- **OS**: [e.g., Ubuntu 22.04, macOS 13.0, Windows 11]
- **Python version**: [output of `python --version`]
- **Aletheion LLM version**: [output of `pip show aletheion-llm`]
- **PyTorch version**: [output of `python -c "import torch; print(torch.__version__)"`]
- **CUDA version** (if applicable): [output of `nvcc --version` or `nvidia-smi`]
- **Installation method**: [pip, conda, source]

**Dependency versions**:
```bash
# Output of: pip list | grep -E "(torch|transformers|aletheion)"
```

## Additional Context

Add any other context about the problem here:
- Does this happen consistently or intermittently?
- Did this work in a previous version?
- Are there any workarounds?
- Screenshots or logs (if applicable)

## Configuration Files

If relevant, attach or paste your configuration files:

```yaml
# config.yaml or relevant configuration
```

## Possible Solution

If you have suggestions on how to fix this bug, please share them here.

## Checklist

Before submitting, please check:

- [ ] I have searched existing issues to ensure this isn't a duplicate
- [ ] I have provided a minimal reproducible example
- [ ] I have included my environment details
- [ ] I have included the full error message/stack trace
- [ ] I have checked the documentation and FAQ
- [ ] I am using the latest version of Aletheion LLM (or have noted my version)

## Priority/Severity

- [ ] Critical - Prevents all usage
- [ ] High - Major feature broken
- [ ] Medium - Feature partially broken
- [ ] Low - Minor issue or cosmetic

---

**Thank you for helping improve Aletheion LLM!**

For security vulnerabilities, please do NOT open a public issue. Instead, report them privately to security@alethea.tech. See [SECURITY.md](../../SECURITY.md) for details.
