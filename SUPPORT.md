# Support

Thank you for using Aletheion LLM! This document provides information about how to get help and support for this project.

## Getting Help

There are several ways to get help with Aletheion LLM:

### Documentation

Start with our comprehensive documentation:

* **[Getting Started Guide](docs/getting-started.md)** - Installation and quickstart
* **[Architecture Documentation](docs/architecture.md)** - Understanding Q1/Q2 gates and Height
* **[API Reference](docs/api.md)** - Complete API documentation
* **[Examples](examples/)** - Code examples and tutorials
* **[Training Guide](TRAINING_GUIDE.md)** - How to train models

### GitHub Issues

For bugs, feature requests, and technical questions:

1. **Search existing issues** first to see if your question has been answered
2. **Create a new issue** using the appropriate template:
   * [Bug Report](.github/ISSUE_TEMPLATE/bug_report.md) - For reporting bugs
   * [Feature Request](.github/ISSUE_TEMPLATE/feature_request.md) - For suggesting new features

**Please provide**:
* A clear, descriptive title
* Detailed description of the issue or question
* Steps to reproduce (for bugs)
* Your environment (OS, Python version, dependency versions)
* Code samples or error messages (use code blocks)

### Discussions

For general questions, ideas, and community interaction:

* **GitHub Discussions** (coming soon) - For broader questions and discussions
* **Research questions** - Related to the epistemic uncertainty architecture
* **Use cases** - Share how you're using Aletheion LLM
* **Ideas and suggestions** - Brainstorm new features or improvements

## What to Include in Support Requests

To help us help you more effectively, please include:

### For Bug Reports

* **Python version**: Output of `python --version`
* **Package version**: `pip show aletheion-llm`
* **Operating System**: e.g., Ubuntu 22.04, macOS 13.0, Windows 11
* **Dependency versions**: `pip list` or your `requirements.txt`
* **Minimal reproducible example**: Smallest code that reproduces the issue
* **Expected behavior**: What you expected to happen
* **Actual behavior**: What actually happened
* **Error messages**: Full stack trace (use code blocks)
* **Configuration**: Any relevant config files or settings

### For Feature Requests

* **Use case**: What problem are you trying to solve?
* **Proposed solution**: How would you like it to work?
* **Alternatives considered**: What workarounds have you tried?
* **Additional context**: Screenshots, diagrams, references

## Response Times

This is an open-source project maintained by volunteers. Response times may vary:

* **Critical security issues**: Within 72 hours (report to security@alethea.tech)
* **Bugs**: Usually within 1 week
* **Feature requests**: Reviewed during planning cycles
* **General questions**: Best effort, usually within 1-2 weeks

## Commercial Support

For commercial support, enterprise deployments, or custom development:

* **Email**: contact@alethea.tech
* **Commercial licensing**: See [LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md)

We offer:
* Priority support and bug fixes
* Custom feature development
* Training and consultation
* Integration assistance
* Production deployment support

## Community Guidelines

When seeking support, please:

* **Be respectful** - Follow our [Code of Conduct](CODE_OF_CONDUCT.md)
* **Be clear and concise** - Help us understand your issue quickly
* **Be patient** - Maintainers are volunteers with limited time
* **Search first** - Check existing issues and documentation
* **Give back** - Help others when you can

## Self-Help Resources

Before asking for help, try these resources:

### Common Issues

1. **Import errors**
   ```bash
   pip install -e .  # Install in development mode
   pip install -e .[dev]  # Include development dependencies
   ```

2. **CUDA/GPU issues**
   * Verify PyTorch installation: `python -c "import torch; print(torch.cuda.is_available())"`
   * Check CUDA compatibility with your PyTorch version

3. **Out of memory errors**
   * Reduce batch size in config files
   * Use gradient accumulation
   * Enable mixed precision training

4. **Poor calibration**
   * Ensure VARO loss is enabled (`varo_weight > 0`)
   * Check Q1/Q2 gate thresholds in config
   * Verify training convergence

### Debugging Tips

* **Enable verbose logging**:
  ```python
  import logging
  logging.basicConfig(level=logging.DEBUG)
  ```

* **Test with minimal example**:
  ```python
  from aletheion_llm import EpSoftmax
  # Test basic functionality
  ```

* **Check dependencies**:
  ```bash
  pip check  # Verify dependency compatibility
  ```

### Useful Commands

```bash
# Run tests
pytest tests/

# Check code style
black --check src/
ruff check src/

# View coverage
pytest --cov=aletheion_llm --cov-report=html
```

## Contributing

If you'd like to contribute to Aletheion LLM:

* See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines
* Check [ROADMAP.md](ROADMAP.md) for planned features
* Look for issues tagged `good first issue`

## Contact Information

* **General inquiries**: contact@alethea.tech
* **Security issues**: security@alethea.tech
* **GitHub Issues**: https://github.com/AletheionAGI/aletheion-llm/issues
* **Repository**: https://github.com/AletheionAGI/aletheion-llm

## Additional Resources

* **Research Paper**: See [paper/](paper/) directory
* **Blog posts**: (coming soon)
* **Tutorials**: Check [examples/](examples/) directory
* **Benchmarks**: See [docs/benchmarks.md](docs/benchmarks.md)

---

**Last Updated**: 2025-01-07

Thank you for being part of the Aletheion LLM community!
