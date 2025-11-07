# Contributing to Aletheion

Thank you for your interest in contributing to Aletheion! This guide will help you get started with contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Project Structure](#project-structure)
- [Contact](#contact)

---

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to contact@alethea.tech.

Key principles:
- Be respectful and considerate
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Respect differing viewpoints and experiences

For full details, see [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Getting Started

### Prerequisites

- **Python 3.10 or higher** (3.11 or 3.12 recommended)
- **PyTorch 2.0 or higher**
- **Git** for version control
- **pipx** (optional but recommended for installing pre-commit)
- Basic understanding of transformers and language models

**Recommended Tools**:
- `make` for running convenience scripts
- `pre-commit` for automated code quality checks
- GPU with CUDA support (for training/experiments)

### Finding Something to Work On

1. Check the [Issues](https://github.com/AletheionAGI/aletheion-llm/issues) page for open issues
2. Look for issues labeled `good first issue` or `help wanted`
3. Comment on the issue to let others know you're working on it
4. If you have a new idea, open an issue first to discuss it

---

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/aletheion-llm.git
cd aletheion-llm
```

### 2. Set Up Upstream Remote

```bash
git remote add upstream https://github.com/AletheionAGI/aletheion-llm.git
git fetch upstream
```

### 3. Install Development Dependencies

```bash
# Install the package in editable mode with all dependencies
pip install -e ".[dev,docs]"

# Or install just development dependencies
pip install -e ".[dev]"
```

### 4. Set Up Pre-Commit Hooks

Pre-commit hooks automatically check code quality before commits:

```bash
# Install pre-commit hooks
pre-commit install

# (Optional) Run hooks on all files to verify setup
pre-commit run --all-files
```

This will automatically run:
- Code formatting (Black, isort)
- Linting (Ruff)
- Type checking (mypy)
- Security checks (bandit)
- File checks (trailing whitespace, etc.)

### 5. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

---

## How to Contribute

### Types of Contributions

We welcome various types of contributions:

#### 🐛 Bug Fixes
- Fix bugs in existing code
- Improve error handling
- Address edge cases

#### ✨ New Features
- Implement new epistemic uncertainty mechanisms
- Add evaluation metrics
- Create visualization tools
- Improve training algorithms

#### 📖 Documentation
- Improve existing documentation
- Add code examples
- Write tutorials
- Fix typos and clarify explanations

#### 🧪 Tests
- Add unit tests
- Create integration tests
- Improve test coverage
- Add benchmark tests

#### 🎨 Code Quality
- Refactor code for clarity
- Optimize performance
- Improve type hints
- Enhance code organization

---

## Code Style

We follow standard Python conventions with some project-specific guidelines:

### General Guidelines

- **PEP 8**: Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- **Line Length**: Maximum **100 characters** (configured in pyproject.toml)
- **Type Hints**: Use type hints for function signatures when practical
- **Docstrings**: Use Google-style docstrings for public APIs
- **Conventional Commits**: Use [Conventional Commits](https://www.conventionalcommits.org/) format

### Code Formatting

We use [Black](https://github.com/psf/black) for code formatting (line-length=100):

```bash
# Format your code
black .

# Check formatting
black --check .
```

### Import Sorting

We use [isort](https://pycqa.github.io/isort/) for import sorting (Black-compatible):

```bash
# Sort imports
isort .

# Check import sorting
isort --check .
```

### Linting

We use [Ruff](https://docs.astral.sh/ruff/) for fast, comprehensive linting:

```bash
# Run linter (with auto-fix)
ruff check --fix .

# Check without fixing
ruff check .
```

Ruff replaces flake8, isort checking, and more with a single fast tool.

### Type Checking

We use [mypy](http://mypy-lang.org/) for static type checking:

```bash
# Run type checker
mypy src/

# Check specific file
mypy src/aletheion/gates.py
```

### All-in-One Scripts

Use our convenience scripts for common tasks:

```bash
# Format code (Black + isort)
./scripts/format.sh

# Run all linters
./scripts/lint.sh

# Run tests with coverage
./scripts/test.sh
```

Or simply rely on **pre-commit hooks** which run automatically before each commit!

### Example Function

```python
import torch
from typing import Optional, Tuple

def compute_epistemic_uncertainty(
    logits: torch.Tensor,
    temperature: float = 1.0,
    epsilon: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute epistemic uncertainty from model logits.

    Args:
        logits: Model output logits of shape (batch, vocab_size)
        temperature: Temperature for softmax scaling. Default: 1.0
        epsilon: Small constant for numerical stability. Default: 1e-8

    Returns:
        uncertainty: Epistemic uncertainty scores of shape (batch,)
        probs: Probability distribution of shape (batch, vocab_size)

    Example:
        >>> logits = torch.randn(2, 1000)
        >>> uncertainty, probs = compute_epistemic_uncertainty(logits)
        >>> uncertainty.shape
        torch.Size([2])
    """
    probs = torch.softmax(logits / temperature, dim=-1)
    uncertainty = -torch.sum(probs * torch.log(probs + epsilon), dim=-1)
    return uncertainty, probs
```

---

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_model.py

# Run with coverage
pytest --cov=src --cov-report=html

# Run tests in parallel
pytest -n auto
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use descriptive test names
- Include docstrings explaining what the test does

### Example Test

```python
import pytest
import torch
from src.aletheion.gates import LocalUncertaintyGate

def test_local_uncertainty_gate_forward():
    """Test LocalUncertaintyGate forward pass with valid input."""
    batch_size = 2
    seq_len = 10
    d_model = 512
    vocab_size = 1000

    gate = LocalUncertaintyGate(d_model=d_model, vocab_size=vocab_size)
    hidden_states = torch.randn(batch_size, seq_len, d_model)

    output = gate(hidden_states)

    assert output.shape == (batch_size, seq_len, vocab_size)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

def test_local_uncertainty_gate_temperature():
    """Test LocalUncertaintyGate with different temperatures."""
    gate = LocalUncertaintyGate(d_model=512, vocab_size=1000)
    hidden_states = torch.randn(2, 10, 512)

    # Higher temperature should produce smoother distributions
    output_low_temp = gate(hidden_states, temperature=0.5)
    output_high_temp = gate(hidden_states, temperature=2.0)

    entropy_low = -torch.sum(output_low_temp * torch.log(output_low_temp + 1e-8))
    entropy_high = -torch.sum(output_high_temp * torch.log(output_high_temp + 1e-8))

    assert entropy_high > entropy_low
```

---

## Pull Request Process

### 1. Prepare Your Changes

```bash
# Make sure you're on your feature branch
git checkout feature/your-feature-name

# Make your changes
# ...

# Format code
black .
isort .

# Run tests
pytest

# Run linter
flake8 src/ tests/ examples/
```

### 2. Commit Your Changes

We follow [Conventional Commits](https://www.conventionalcommits.org/) format for clear, structured commit messages:

**Format**: `<type>(<scope>): <description>`

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `perf`: Performance improvements
- `ci`: CI/CD changes

**Examples**:

```bash
# Feature with scope
git commit -m "feat(gates): add temperature parameter to Q1/Q2 gates"

# Bug fix
git commit -m "fix(training): resolve VARO loss numerical instability"

# Documentation
git commit -m "docs: add calibration tutorial to README"

# Multi-line with body
git commit -m "feat(visualization): add uncertainty heatmap

- Implement uncertainty heatmap visualization
- Add support for multiple layers
- Include example usage in docs
- Add tests for visualization functions
"
```

**Tips**:
- Use imperative mood ("add" not "added" or "adds")
- Capitalize first letter after type/scope
- No period at end of subject line
- Wrap body at 72 characters

### 3. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 4. Create Pull Request

1. Go to the [Aletheion repository](https://github.com/AletheionAGI/aletheion-llm)
2. Click "New Pull Request"
3. Select your fork and branch
4. Fill out the PR template with:
   - Description of changes
   - Related issue(s)
   - Testing performed
   - Screenshots (if applicable)

### 5. Code Review

- Address review comments promptly
- Make requested changes
- Push updates to the same branch
- Request re-review when ready

### PR Checklist

Before submitting, ensure:

- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] New tests added for new functionality
- [ ] Documentation updated (if needed)
- [ ] Commit messages are clear and descriptive
- [ ] No merge conflicts
- [ ] PR description is complete

---

## Project Structure

Understanding the project structure will help you contribute effectively:

```
aletheion-llm/
├── src/                      # Core library code
│   ├── model.py             # Baseline transformer
│   ├── attention.py         # Attention mechanisms
│   ├── tokenizer.py         # Tokenization
│   ├── utils.py             # Utilities
│   └── aletheion/           # Epistemic uncertainty components
│       ├── gates.py         # Q₁ and Q₂ gates
│       ├── loss.py          # VARO loss
│       ├── model.py         # Aletheion transformer
│       └── pyramidal_*.py   # Pyramidal implementations
│
├── examples/                 # Usage examples (good starting point!)
│   ├── train.py
│   ├── train_aletheion.py
│   └── eval.py
│
├── experiments/              # Research experiments
│   └── level1/
│
├── tests/                    # Test suite
│   ├── test_model.py
│   └── aletheion/
│
├── docs/                     # Documentation
├── config/                   # YAML configurations
└── data/                     # Dataset handling
```

### Key Components

- **Gates** (`src/aletheion/gates.py`): Epistemic uncertainty gates (Q₁, Q₂)
- **Loss** (`src/aletheion/loss.py`): VARO loss functions
- **Model** (`src/aletheion/model.py`): Main Aletheion transformer
- **Pyramidal** (`src/aletheion/pyramidal_*.py`): Pyramidal architecture variants

---

## Development Tips

### Running Quick Experiments

```bash
# Train a small model for quick testing
python examples/train.py --config config/small.yaml --output outputs/test/

# Quick evaluation
python examples/quick_eval.py --checkpoint outputs/test/checkpoint_latest.pt
```

### Debugging

```bash
# Run with Python debugger
python -m pdb examples/train.py --config config/small.yaml

# Use PyTorch autograd anomaly detection
PYTORCH_AUTOGRAD_ANOMALY_DETECTION=1 python examples/train.py
```

### Profiling

```bash
# Profile training
python -m cProfile -o profile.stats examples/train.py
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"
```

---

## Documentation

### Building Documentation

Documentation improvements are always welcome! To build docs locally:

```bash
# Coming soon: Sphinx documentation
```

### Writing Documentation

- Use clear, concise language
- Include code examples
- Add diagrams where helpful
- Link to related documentation

---

## Questions?

If you have questions about contributing:

1. Check existing [documentation](docs/)
2. Search [issues](https://github.com/AletheionAGI/aletheion-llm/issues)
3. Review [SUPPORT.md](SUPPORT.md) for help resources
4. Ask in a new issue
5. Contact: [contact@alethea.tech](mailto:contact@alethea.tech)

---

## Additional Resources

Before contributing, you may find these documents helpful:

- **[ROADMAP.md](ROADMAP.md)** - Project roadmap and planned features
- **[GOVERNANCE.md](GOVERNANCE.md)** - Project governance and decision-making
- **[SECURITY.md](SECURITY.md)** - Security policy and reporting vulnerabilities
- **[SUPPORT.md](SUPPORT.md)** - Getting help and support
- **[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)** - Community standards
- **[CHANGELOG.md](CHANGELOG.md)** - Version history and changes

---

## License

By contributing to Aletheion LLM, you agree that your contributions will be licensed under the same [AGPL-3.0-or-later license](LICENSE) as the project.

**Important**: All contributors retain copyright to their contributions. By submitting a pull request, you grant the project maintainers a license to use your contribution under the project's license terms.

For commercial licensing inquiries, contact [contact@alethea.tech](mailto:contact@alethea.tech) or see [LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md).

---

## Thank You!

Your contributions help advance the field of uncertainty quantification in language models. Every contribution, no matter how small, makes a difference!

**Happy coding! 🚀**
