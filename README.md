# Aletheion: Epistemic Uncertainty for Large Language Models

<div align="center">

**Implementation of fractally-applied epistemic softmax for calibrated, uncertainty-aware language models**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-AGPL--3.0%20or%20Commercial-blue.svg)](LICENSE-AGPL.md)
[![Status](https://img.shields.io/badge/status-active%20research-yellow.svg)](https://github.com/AletheionAGI/aletheion-llm)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Tests](https://img.shields.io/badge/tests-pytest-blue.svg)](https://docs.pytest.org/)

[Features](#features) • [Installation](#installation) • [Quick Start](#quick-start) • [Documentation](#documentation) • [Citation](#citation)

</div>

---

## Table of Contents

- [Philosophical Foundations](#philosophical-foundations)
- [Overview](#overview)
- [Features](#features)
- [Background](#background)
- [Development Quickstart](#development-quickstart)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Usage Examples](#api-usage-examples)
- [Docker Usage](#docker-usage)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Development Workflow](#development-workflow)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Author's Note

This project was developed independently in Brazil, grounded in the belief that scientific progress should transcend geography, language, and institutional boundaries.

Aletheion is an open research effort built on transparency, reproducibility, and epistemic humility — values that matter more than prestige or affiliation.

All constructive collaboration is welcome. The goal is not recognition, but understanding.

---

## Philosophical Foundations

The epistemic framework behind Aletheion was first explored in 
"Demotéica" (Maya Rahto), a philosophical novel about ideological 
possession and epistemic authenticity.

For readers interested in the conceptual origins of Q1/Q2 gates 
and the "apex delusion" problem, the book is available on Amazon:

PT-BR: https://www.amazon.com.br/Demot%C3%A9ica-Completo-Maya-Rahto-ebook/dp/B09YN3MBQW

EN: https://www.amazon.com/dp/B0F2GDCGK5

*Note: Reading the book is not required to use Aletheion, but 
provides deeper context on the philosophical motivations.*

## Overview

Large language models hallucinate, contradict themselves, and rarely express calibrated uncertainty. **Aletheion** addresses this fundamental challenge by replacing traditional softmax operations with **epistemic softmax**—a gating mechanism that factors uncertainty into every decision.

### Key Innovation

Aletheion introduces **Pyramidal Epistemology**, a fractal architecture that applies uncertainty quantification at multiple levels:
- **Q₁ (Local Uncertainty Gate):** Token-level uncertainty estimation
- **Q₂ (Cross-Context Gate):** Context-aware uncertainty propagation
- **VARO Loss:** Variational Approximation to Rational Objectives

### ✅ Current Implementation Status

| Level | Description | Status | Details |
|-------|-------------|--------|---------|
| **Level 0** | Baseline Transformer | ✅ **Complete** | Fully operational baseline |
| **Level 1** | Output Gates (Q₁/Q₂/VARO) | ✅ **Complete** | Production-ready, ready for validation |
| **Level 2** | Attention + Output Gates | ⏳ **Partial** | Pyramidal variants available |
| **Level 3** | Full Fractal Architecture | 🔜 **Planned** | Future work |

**⚡ Latest:** Level 1 implementation complete! All core epistemic components (Q₁, Q₂, VARO loss, epistemic softmax) are fully implemented and tested. Ready for experimental validation.

---

## Features

✨ **Epistemic Uncertainty Quantification**
- Local uncertainty gates (Q₁) for token-level decisions
- Cross-context gates (Q₂) for semantic coherence
- Fractal architecture for multi-scale uncertainty

📊 **Improved Calibration**
- Expected Calibration Error (ECE) improvements of 20-40%
- Reduced hallucination rates
- Better abstention on out-of-distribution inputs

🔧 **Modular Architecture**
- Drop-in replacement for standard transformers
- Compatible with HuggingFace transformers
- Configurable via YAML files

🧪 **Comprehensive Testing**
- TruthfulQA benchmark integration
- Out-of-domain evaluation suite
- Calibration metrics and visualization tools

📖 **Research-Ready**
- Full experimental framework
- Reproducible training scripts
- Detailed documentation and papers

---

## Background

Large language models suffer from overconfidence and lack of uncertainty awareness. Aletheion addresses this by implementing a hierarchical approach to epistemic uncertainty:

1. **Local Uncertainty (Q₁):** Captures token-level uncertainty in predictions
2. **Cross-Context Uncertainty (Q₂):** Models semantic coherence across context
3. **Fractal Application:** Applies uncertainty principles at multiple architectural levels

This repository implements a progressive architecture across multiple levels:
- **Level 1:** Output-only gating (✅ **Fully Implemented & Production-Ready**)
- **Level 2:** Attention-level gating (⏳ Pyramidal variants available, integration pending)
- **Level 3:** Full fractal architecture (🔜 Planned for future releases)

**Theoretical Foundation:**
- [The Quality of Truth](https://github.com/AletheionAGI/aletheion-llm/blob/main/paper/en/main.pdf) - Philosophical framework (2021)
- Aletheion Research Paper - See [`paper/`](paper/) directory

---

## Development Quickstart

Get started with Aletheion development in minutes:

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/AletheionAGI/aletheion-llm.git
cd aletheion-llm

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### 2. Verify Installation

```bash
# Run tests to ensure everything is working
./scripts/test.sh

# Check code quality
./scripts/lint.sh
```

### 3. Quick Development Commands

```bash
# Format code (Black + isort)
./scripts/format.sh

# Run linters (Ruff + Black + isort + mypy)
./scripts/lint.sh

# Run tests with coverage
./scripts/test.sh

# Run specific tests
pytest tests/test_model.py -v

# Train a small model for testing
python examples/train_aletheion.py --config config/small.yaml
```

### 4. Development with GPU

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Train with GPU
python examples/train_aletheion.py --config config/aletheion_level1.yaml
```

### 5. Interactive Development

```bash
# Start Python REPL with Aletheion loaded
python -c "from src.aletheion.model import AletheionTransformer; print('Ready!')"

# Or use IPython for better experience
ipython
>>> from src.aletheion.gates import LocalUncertaintyGate, CrossContextGate
>>> from src.aletheion.model import AletheionTransformer
```

**Next Steps:**
- Read the [API Usage Examples](#api-usage-examples) to learn the API
- Check [Development Workflow](#development-workflow) for contribution guidelines
- See [Docker Usage](#docker-usage) for containerized development

---

## Installation

### From Source (Recommended for Development)

```bash
# Clone the repository
git clone https://github.com/AletheionAGI/aletheion-llm.git
cd aletheion-llm

# Install in editable mode with dependencies
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### From Requirements (Quick Start)

```bash
pip install -r requirements.txt
```

### System Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)
- 8GB+ RAM (16GB+ recommended)

---

## Quick Start

### 1. Train a Baseline Model

```bash
python examples/train.py --config config/small.yaml --output outputs/baseline/
```

### 2. Train an Aletheion Model

```bash
python examples/train_aletheion.py --config config/aletheion_level1.yaml --output outputs/aletheion/
```

### 3. Compare Baseline vs Aletheion

```bash
python experiments/level1/compare_baseline_aletheion.py
```

### 4. Evaluate on TruthfulQA

```bash
python experiments/level1/test_truthfulqa.py --checkpoint outputs/aletheion/checkpoint_final.pt
```

### 5. Generate Text with Uncertainty

```bash
python examples/generate.py --checkpoint outputs/aletheion/checkpoint_final.pt --prompt "Your prompt here"
```

For more examples and tutorials, see the [`examples/`](examples/) directory.

---

## API Usage Examples

Learn how to use Aletheion programmatically in your own projects:

### Basic Usage: Creating an Aletheion Model

```python
import torch
from src.aletheion.model import AletheionTransformer

# Create an Aletheion model with epistemic uncertainty
model = AletheionTransformer(
    vocab_size=50257,      # GPT-2 vocabulary
    d_model=512,           # Hidden dimension
    n_layers=6,            # Number of transformer layers
    n_heads=8,             # Number of attention heads
    d_ff=2048,             # Feed-forward dimension
    max_seq_len=512,       # Maximum sequence length
    dropout=0.1,           # Dropout probability
    # Epistemic parameters
    q1_threshold=0.7,      # Local uncertainty threshold
    q2_threshold=0.7,      # Cross-context threshold
    base_temperature=1.0,  # Base softmax temperature
    n_consensus_heads=4    # Heads for Q2 consensus
)

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Using Epistemic Gates (Q1 and Q2)

```python
from src.aletheion.gates import LocalUncertaintyGate, CrossContextGate, epistemic_softmax

# Create epistemic gates
d_model = 512
q1_gate = LocalUncertaintyGate(d_model=d_model, dropout=0.1)
q2_gate = CrossContextGate(d_model=d_model, n_heads=4, dropout=0.1)

# Example input: batch=2, sequence=32, hidden=512
batch_size, seq_len = 2, 32
context = torch.randn(batch_size, seq_len, d_model)
logits = torch.randn(batch_size, seq_len, 50257)  # vocab_size=50257

# Compute epistemic softmax with uncertainty
probs, uncertainty = epistemic_softmax(
    logits=logits,
    context=context,
    q1_gate=q1_gate,
    q2_gate=q2_gate,
    base_temperature=1.0,
    confidence_threshold=0.7
)

print(f"Output probabilities shape: {probs.shape}")      # (2, 32, 50257)
print(f"Uncertainty scores shape: {uncertainty.shape}")  # (2, 32, 1)
print(f"Mean uncertainty: {uncertainty.mean().item():.3f}")
```

### Training with VARO Loss

```python
from src.aletheion.loss import VaroLoss
import torch.nn.functional as F

# Create VARO loss function
varo_loss = VaroLoss(
    lambda_varo=0.1,           # Weight for uncertainty regularization
    u_star_method='head_variance',  # Method for target uncertainty
    min_entropy=0.1            # Minimum gate entropy
)

# Training step
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Forward pass
input_ids = torch.randint(0, 50257, (2, 32))  # (batch, seq_len)
outputs = model(input_ids)

# Compute VARO loss
loss_dict = varo_loss(
    logits=outputs['logits'],
    targets=input_ids,
    q1_values=outputs['q1'],
    q2_values=outputs['q2'],
    attention_weights=outputs.get('attention_weights')  # Optional
)

# Backward pass
loss = loss_dict['loss']
loss.backward()
optimizer.step()
optimizer.zero_grad()

print(f"Total loss: {loss.item():.4f}")
print(f"CE loss: {loss_dict['ce_loss'].item():.4f}")
print(f"Uncertainty loss: {loss_dict['uncertainty_loss'].item():.4f}")
```

### Text Generation with Uncertainty

```python
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')

# Generate text with uncertainty awareness
model.eval()
prompt = "The future of artificial intelligence is"
input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

with torch.no_grad():
    generated = model.generate(
        input_ids,
        max_length=50,
        temperature=1.0,
        top_k=50,
        do_sample=True
    )

    # Get uncertainty for generated tokens
    outputs = model(generated)
    uncertainty = 1.0 - (outputs['q1'] * outputs['q2'])

# Decode and display results
generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
mean_uncertainty = uncertainty.mean().item()

print(f"Generated: {generated_text}")
print(f"Mean uncertainty: {mean_uncertainty:.3f}")
```

### Evaluating Calibration

```python
from experiments.level1.test_truthfulqa import evaluate_calibration
import numpy as np

# Evaluate model calibration on validation set
model.eval()
predictions = []
uncertainties = []
targets = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids)
        probs = torch.softmax(outputs['logits'], dim=-1)
        uncertainty = 1.0 - (outputs['q1'] * outputs['q2'])

        # Get predictions
        pred_probs, pred_ids = probs.max(dim=-1)

        predictions.extend(pred_probs.cpu().numpy())
        uncertainties.extend(uncertainty.squeeze(-1).cpu().numpy())
        targets.extend((pred_ids == labels).cpu().numpy())

# Compute Expected Calibration Error (ECE)
predictions = np.array(predictions)
targets = np.array(targets)
n_bins = 10

ece = 0.0
for i in range(n_bins):
    bin_lower = i / n_bins
    bin_upper = (i + 1) / n_bins
    in_bin = (predictions >= bin_lower) & (predictions < bin_upper)

    if in_bin.sum() > 0:
        accuracy = targets[in_bin].mean()
        confidence = predictions[in_bin].mean()
        ece += np.abs(accuracy - confidence) * in_bin.mean()

print(f"Expected Calibration Error: {ece:.4f}")
```

### Comparing Baseline vs Aletheion

```python
from src.model import BaselineTransformer
from src.aletheion.model import AletheionTransformer

# Create both models with same architecture
config = {
    'vocab_size': 50257,
    'd_model': 512,
    'n_layers': 6,
    'n_heads': 8,
    'd_ff': 2048,
    'max_seq_len': 512,
    'dropout': 0.1
}

baseline_model = BaselineTransformer(**config).to(device)
aletheion_model = AletheionTransformer(
    **config,
    q1_threshold=0.7,
    q2_threshold=0.7,
    base_temperature=1.0,
    n_consensus_heads=4
).to(device)

# Compare parameter counts
baseline_params = sum(p.numel() for p in baseline_model.parameters())
aletheion_params = sum(p.numel() for p in aletheion_model.parameters())
overhead = (aletheion_params - baseline_params) / baseline_params * 100

print(f"Baseline parameters: {baseline_params:,}")
print(f"Aletheion parameters: {aletheion_params:,}")
print(f"Parameter overhead: {overhead:.2f}%")  # Expected: ~2%
```

### Using Pyramidal Models (Advanced)

```python
from src.aletheion.pyramidal_q1q2_model import PyramidalQ1Q2Transformer

# Create pyramidal model with multi-level epistemic gates
pyramidal_model = PyramidalQ1Q2Transformer(
    vocab_size=50257,
    d_model=512,
    n_layers=6,
    n_heads=8,
    d_ff=2048,
    max_seq_len=512,
    dropout=0.1,
    # Pyramidal-specific parameters
    q1_threshold=0.7,
    q2_threshold=0.7,
    base_temperature=1.0,
    n_consensus_heads=4
).to(device)

# Forward pass returns hierarchical uncertainty
outputs = pyramidal_model(input_ids)

# Access different levels of uncertainty
print(f"Q1 (local): {outputs['q1'].mean():.3f}")
print(f"Q2 (cross-context): {outputs['q2'].mean():.3f}")
print(f"Combined uncertainty: {(1 - outputs['q1'] * outputs['q2']).mean():.3f}")
```

### Configuration via YAML

```python
from src import load_config

# Load configuration from YAML file
config = load_config('config/aletheion_level1.yaml')

# Create model from config
model = AletheionTransformer(
    vocab_size=config['model']['vocab_size'],
    d_model=config['model']['d_model'],
    n_layers=config['model']['n_layers'],
    n_heads=config['model']['n_heads'],
    d_ff=config['model']['d_ff'],
    max_seq_len=config['model']['max_seq_len'],
    dropout=config['model']['dropout'],
    # Epistemic params from config
    q1_threshold=config['model']['epistemic']['q1_threshold'],
    q2_threshold=config['model']['epistemic']['q2_threshold'],
    base_temperature=config['model']['epistemic']['base_temperature'],
    n_consensus_heads=config['model']['epistemic']['n_consensus_heads']
).to(device)

print(f"Loaded config: {config['logging']['run_name']}")
```

**More Examples:**
- See [`examples/`](examples/) for complete training and evaluation scripts
- Check [`experiments/level1/`](experiments/level1/) for research experiments
- Read [`docs/ALETHEION_LEVEL1_README.md`](docs/ALETHEION_LEVEL1_README.md) for architectural details

---

## Docker Usage

Aletheion provides Docker support for reproducible development and deployment environments.

### Quick Start with Docker

```bash
# Note: Docker support is planned for future releases
# Current development uses local Python environment
# See Development Quickstart section above

# Build custom Docker image (when available)
docker build -t aletheion-llm:latest .

# Run training in container
docker run --gpus all -v $(pwd)/outputs:/workspace/outputs \
  aletheion-llm:latest python examples/train_aletheion.py --config config/aletheion_level1.yaml

# Run tests in container
docker run aletheion-llm:latest ./scripts/test.sh
```

### Docker Compose (Planned)

```yaml
# docker-compose.yml (example for future implementation)
version: '3.8'

services:
  train:
    build: .
    volumes:
      - ./outputs:/workspace/outputs
      - ./config:/workspace/config
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: python examples/train_aletheion.py --config config/aletheion_level1.yaml

  jupyter:
    build: .
    ports:
      - "8888:8888"
    volumes:
      - ./:/workspace
    command: jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

### GPU Support

```bash
# Check GPU availability in Docker
docker run --gpus all aletheion-llm:latest \
  python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Train with specific GPU
docker run --gpus '"device=0"' -v $(pwd)/outputs:/workspace/outputs \
  aletheion-llm:latest python examples/train_aletheion.py
```

**Note:** Full Docker and docker-compose support is planned for future releases. For now, use the local development environment described in [Development Quickstart](#development-quickstart).

---

## Project Structure

```
aletheion-llm/
├── src/                      # Core library
│   ├── model.py             # Baseline transformer
│   ├── attention.py         # Attention mechanisms
│   └── aletheion/           # Epistemic uncertainty components
│       ├── gates.py         # Q₁ and Q₂ gates
│       ├── loss.py          # VARO loss functions
│       ├── model.py         # Aletheion transformer
│       └── pyramidal_*.py   # Pyramidal implementations
│
├── examples/                 # Usage examples
│   ├── train.py             # Baseline training
│   ├── train_aletheion.py   # Aletheion training
│   ├── eval.py              # Evaluation
│   └── generate.py          # Text generation
│
├── experiments/              # Research experiments
│   └── level1/              # Level 1 experiments
│       ├── compare_*.py     # Comparison scripts
│       ├── test_*.py        # Testing scripts
│       └── visualize_*.py   # Visualization tools
│
├── tests/                    # Unit and integration tests
│   ├── test_model.py
│   ├── test_attention.py
│   └── aletheion/           # Aletheion-specific tests
│
├── config/                   # Training configurations
│   ├── default.yaml
│   ├── small.yaml
│   ├── medium.yaml
│   └── aletheion_level1.yaml
│
├── docs/                     # Documentation
│   ├── README.md            # Documentation index
│   ├── ALETHEION_LEVEL1_README.md
│   ├── PYRAMIDAL_EPISTEMOLOGY_README.md
│   └── *.md                 # Technical docs
│
├── paper/                    # Research papers
│   └── en/                  # English version
│       ├── main.pdf
│       └── main.tex
│
├── scripts/                  # Utility scripts
│   ├── train_*.sh
│   └── test_*.sh
│
├── data/                     # Dataset utilities
│   ├── dataset.py
│   └── prepare.py
│
└── audit/                    # Quality assurance
    └── AUDIT_REPORT.md
```

---

## Documentation

Comprehensive documentation is available in the [`docs/`](docs/) directory:

### Core Documentation
- **[Documentation Index](docs/README.md)** - Complete documentation overview
- **[Level 1 Implementation](docs/ALETHEION_LEVEL1_README.md)** - Detailed Level 1 architecture and quick start
- **[Pyramidal Epistemology](docs/PYRAMIDAL_EPISTEMOLOGY_README.md)** - Theoretical framework and geometric structure
- **[Implementation Notes](docs/IMPLEMENTATION_NOTES.md)** - Design decisions and technical details
- **[Quantitative Metrics Analysis](docs/QUANTITATIVE_METRICS_ANALYSIS.md)** - Complete experimental validation results

### Technical Deep Dives
- [LLM Fundamentals](docs/llm-fundamentals.md) - Fundamentals of Large Language Models
- [LLM Failures](docs/llm-failures.md) - Analysis of common LLM failure modes
- [Attention Mechanisms](docs/attention-mechanisms.md) - Deep dive into attention mechanisms
- [Training Strategy](docs/training-strategy.md) - Training strategies and best practices
- [Aletheion Integration](docs/aletheion-integration.md) - How to integrate Aletheion into existing models
- [Fractal Approach](docs/aletheion-fractal-approach.md) - Fractal architecture approach

### Evaluation & Testing
- [TruthfulQA Setup](docs/TRUTHFULQA_SETUP.md) - TruthfulQA benchmark setup and usage
- [Calibration Fixes](docs/BUGFIX_CALIBRATION.md) - Documentation of calibration bug fixes
- [Training Scripts Comparison](docs/TRAINING_SCRIPTS_COMPARISON.md) - Comparison of training approaches

### Research & Reports
- [Project Report](docs/Report.md) - Comprehensive project report
- [Pyramidal Q1Q2 Fractal](docs/PYRAMIDAL_Q1Q2_FRACTAL.md) - Complete fractal implementation guide
- [Remaining Limitations](docs/remaining-limitations.md) - Known limitations and future work

### Additional Resources
- [Changelog](docs/CHANGELOG.md) - Project version history and changes
- [Contributing Guide](CONTRIBUTING.md) - How to contribute to the project

---

## Development Workflow

Complete guide to contributing code to Aletheion:

### 1. Setting Up Your Development Environment

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/aletheion-llm.git
cd aletheion-llm

# Add upstream remote
git remote add upstream https://github.com/AletheionAGI/aletheion-llm.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install with all development dependencies
pip install -e ".[dev,docs]"

# Install pre-commit hooks
pre-commit install
```

### 2. Creating a Feature Branch

```bash
# Fetch latest changes from upstream
git fetch upstream
git checkout main
git merge upstream/main

# Create a feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/bug-description
```

### 3. Making Changes

```bash
# Make your changes to the code
# Add tests for new functionality in tests/

# Run formatters (will auto-fix issues)
./scripts/format.sh

# Run linters (will check for issues)
./scripts/lint.sh

# Run tests with coverage
./scripts/test.sh

# Run specific test file
pytest tests/test_model.py -v

# Run tests matching a pattern
pytest -k "test_epistemic" -v
```

### 4. Pre-Commit Hooks

Pre-commit hooks run automatically before each commit:

```bash
# Hooks will run on staged files
git add .
git commit -m "feat: add new feature"

# If hooks fail, they'll auto-fix what they can
# Review changes and commit again
git add .
git commit -m "feat: add new feature"

# Skip hooks only if absolutely necessary (not recommended)
git commit --no-verify -m "feat: urgent fix"
```

**Configured hooks:**
- **Black**: Code formatting (auto-fixes)
- **isort**: Import sorting (auto-fixes)
- **Ruff**: Linting (reports issues)
- **Trailing whitespace**: Removes trailing spaces (auto-fixes)
- **End of file**: Ensures newline at EOF (auto-fixes)

### 5. Writing Tests

All new code should include tests:

```python
# tests/test_my_feature.py
import pytest
import torch
from src.aletheion.gates import LocalUncertaintyGate

def test_local_uncertainty_gate():
    """Test Q1 gate produces valid uncertainty scores."""
    d_model = 512
    batch_size = 2
    seq_len = 32

    gate = LocalUncertaintyGate(d_model=d_model)
    context = torch.randn(batch_size, seq_len, d_model)

    # Forward pass
    q1 = gate(context)

    # Assertions
    assert q1.shape == (batch_size, seq_len, 1)
    assert (q1 >= 0).all() and (q1 <= 1).all()
    assert not torch.isnan(q1).any()

@pytest.mark.slow
def test_training_convergence():
    """Test that model loss decreases over training."""
    # Long-running integration test
    # Marked as 'slow' so it can be skipped with: pytest -m "not slow"
    pass
```

**Test commands:**
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run only fast tests (skip slow integration tests)
pytest -m "not slow"

# Run only unit tests
pytest -m unit

# Run with verbose output
pytest -v

# Run tests in parallel (faster)
pytest -n auto
```

### 6. Updating Documentation

When adding features, update documentation:

```bash
# Update relevant docs in docs/
# If adding new API, document it in the code with docstrings

# Example docstring format:
"""Short description.

Longer description with more details about the function,
its purpose, and how it fits into the larger system.

Args:
    param1: Description of param1
    param2: Description of param2

Returns:
    Description of return value

Example:
    >>> model = AletheionTransformer(...)
    >>> output = model(input_ids)
    >>> print(output['logits'].shape)
"""

# Build docs locally (if MkDocs configured)
mkdocs serve
# View at http://localhost:8000
```

### 7. Committing Your Changes

We use [Conventional Commits](https://www.conventionalcommits.org/):

```bash
# Commit format: <type>(<scope>): <description>

# Types:
# feat:     New feature
# fix:      Bug fix
# docs:     Documentation changes
# style:    Code style changes (formatting, etc.)
# refactor: Code refactoring
# test:     Adding or updating tests
# chore:    Maintenance tasks
# perf:     Performance improvements

# Examples:
git commit -m "feat(gates): add support for dynamic thresholds"
git commit -m "fix(loss): correct VARO loss computation for edge cases"
git commit -m "docs: update API usage examples in README"
git commit -m "test(model): add integration tests for pyramidal architecture"
git commit -m "refactor(attention): simplify cross-attention implementation"
```

### 8. Pushing and Creating a Pull Request

```bash
# Push your branch to your fork
git push origin feature/your-feature-name

# Go to GitHub and create a Pull Request
# Fill in the PR template with:
# - Description of changes
# - Related issues (if any)
# - Checklist items completed
```

### 9. Code Review Process

**After submitting your PR:**

1. **Automated Checks**: CI/CD will run tests, linters, and checks
2. **Code Review**: Maintainers will review your code
3. **Address Feedback**: Make requested changes and push updates
4. **Approval**: Once approved, maintainers will merge your PR

**Responding to feedback:**
```bash
# Make requested changes
# Commit with conventional format
git add .
git commit -m "fix: address review feedback"
git push origin feature/your-feature-name
```

### 10. Keeping Your Branch Updated

```bash
# Fetch latest changes from upstream
git fetch upstream

# Rebase your branch on latest main (preferred)
git rebase upstream/main

# Or merge (if rebase causes conflicts)
git merge upstream/main

# Push updated branch (may need force push after rebase)
git push origin feature/your-feature-name --force-with-lease
```

### Development Best Practices

**Code Quality:**
- ✅ Follow PEP 8 style guide (enforced by Black and Ruff)
- ✅ Write type hints for all functions
- ✅ Keep functions focused and small (<50 lines ideally)
- ✅ Use meaningful variable and function names
- ✅ Add docstrings to all public functions and classes

**Testing:**
- ✅ Aim for >80% code coverage
- ✅ Test edge cases and error conditions
- ✅ Use fixtures for common test setup
- ✅ Mock external dependencies
- ✅ Keep tests fast (use `@pytest.mark.slow` for slow tests)

**Git Workflow:**
- ✅ Keep commits atomic (one logical change per commit)
- ✅ Write clear commit messages
- ✅ Rebase on main before creating PR
- ✅ Squash fixup commits before merging
- ✅ Never force push to main

**Documentation:**
- ✅ Update README for user-facing changes
- ✅ Add docstrings with examples
- ✅ Update CHANGELOG.md
- ✅ Include usage examples for new features

### Quick Reference Commands

```bash
# Daily workflow
./scripts/format.sh          # Format code
./scripts/lint.sh            # Check code quality
./scripts/test.sh            # Run tests
git add .                    # Stage changes
git commit                   # Commit (pre-commit hooks run)
git push                     # Push to remote

# Debugging
pytest -v --pdb              # Drop into debugger on failure
pytest --lf                  # Run last failed tests
pytest -x                    # Stop on first failure

# Performance
pytest --durations=10        # Show 10 slowest tests
python -m cProfile script.py # Profile Python script

# Documentation
mkdocs serve                 # Preview docs locally
mkdocs build                 # Build docs
```

---

## Results

### Level 1 (Output-Only Gating) - ✅ Validated

**Training Status:** Complete (60,000 steps on WikiText-2)

**Key Achievement: 89% ECE Reduction**
- Baseline transformer exhibits the classic "Skynet problem": as capability increases (perplexity ↓), calibration degrades (ECE ↑)
- Aletheion Level 1 maintains excellent calibration while achieving comparable language modeling performance

### Experimental Validation

**Final Metrics Comparison:**

| Metric | Baseline (Level 0) | Aletheion Level 1 | Improvement |
|--------|-------------------|-------------------|-------------|
| **ECE** (↓) | 0.104 (poor) | **0.011 (excellent)** | **-89%** ✓ |
| **Brier Score** (↓) | ~0.88 | ~0.87-0.88 | Comparable |
| **Perplexity** (↓) | ~230-250 | ~250-300 | Comparable (-8%) |
| **Calibration Quality** | Poor (>0.10) | Excellent (<0.05) | Excellent ✓ |
| **Parameters** | 100% | ~102% | +2% overhead |

**Training Dynamics:**
- **Baseline:** ECE increases 10× during training (0.01 → 0.104) - the "Skynet problem"
- **Aletheion:** ECE remains excellent throughout training (~0.01-0.02) - epistemic equilibrium maintained

**Pyramidal Architecture Metrics:**
- **Height Convergence:** 0.1 → 0.95 (approaching truth apex at 1.0)
- **Base Stability:** 0.98-0.99 (exceptional equilibrium across Memory, Pain, Choice, Exploration forces)
- **Q₁/Q₂ Gates:** Converged to optimal mid-range uncertainty (0.42-0.47)
- **Adaptive Metalearning:** Model exhibited sophisticated epistemic exploration cycles

For detailed quantitative analysis, see [`docs/QUANTITATIVE_METRICS_ANALYSIS.md`](docs/QUANTITATIVE_METRICS_ANALYSIS.md)

### Benchmarks

| Metric | Baseline | Aletheion L1 | Improvement |
|--------|----------|--------------|-------------|
| ECE (↓) | 0.104 | 0.011 | **-89%** |
| Brier Score (↓) | 0.88 | 0.87 | Comparable |
| Perplexity (↓) | 230-250 | 250-300 | Comparable |
| Parameters | 100% | 102% | +2% |

---

## Troubleshooting

Common issues and their solutions:

### Installation Issues

#### **Problem: `pip install -e .` fails with dependency conflicts**

```bash
# Solution 1: Create a fresh virtual environment
python -m venv venv_new
source venv_new/bin/activate  # Windows: venv_new\Scripts\activate
pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"

# Solution 2: Install PyTorch first (especially for GPU)
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -e .

# Solution 3: Use conda instead
conda create -n aletheion python=3.10
conda activate aletheion
pip install -e ".[dev]"
```

#### **Problem: `ModuleNotFoundError: No module named 'src'`**

```bash
# Solution: Install in editable mode
pip install -e .

# Or add src to PYTHONPATH temporarily
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/Mac
set PYTHONPATH=%PYTHONPATH%;%CD%          # Windows
```

### CUDA/GPU Issues

#### **Problem: `torch.cuda.is_available()` returns False**

```bash
# Check CUDA installation
nvidia-smi  # Should show GPU info

# Reinstall PyTorch with correct CUDA version
# For CUDA 11.8:
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121

# For CPU-only (no GPU):
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

#### **Problem: Out of Memory (OOM) errors during training**

```python
# Solution 1: Reduce batch size in config
# Edit config/aletheion_level1.yaml:
training:
  batch_size: 16  # Reduce from 32
  gradient_accumulation_steps: 2  # Double to maintain effective batch size

# Solution 2: Enable mixed precision training
system:
  mixed_precision: true  # Reduces memory usage by ~50%

# Solution 3: Reduce model size
model:
  d_model: 256      # Reduce from 512
  n_layers: 4       # Reduce from 6
  d_ff: 1024        # Reduce from 2048
```

#### **Problem: CUDA out of memory during evaluation**

```python
# Solution: Use smaller batch size for evaluation
# In your evaluation script:
with torch.no_grad():  # Don't forget this!
    for batch in DataLoader(val_dataset, batch_size=8):  # Smaller batch
        outputs = model(batch['input_ids'])
```

### Training Issues

#### **Problem: Loss becomes NaN**

```bash
# Common causes and solutions:

# 1. Learning rate too high
# Edit config YAML:
training:
  learning_rate: 1.0e-4  # Reduce from 3.0e-4

# 2. Gradient explosion
training:
  grad_clip_norm: 0.5  # Reduce from 1.0

# 3. Numerical instability in gates
model:
  epistemic:
    base_temperature: 1.0  # Don't set too low
    lambda_varo: 0.01      # Reduce from 0.1
```

#### **Problem: Model not learning (loss not decreasing)**

```bash
# Checklist:
# 1. Check learning rate
training:
  learning_rate: 3.0e-4  # Not too low (e.g., 1e-6)
  warmup_steps: 2000     # Ensure warmup

# 2. Verify data is loaded correctly
python -c "
from data.dataset import load_wikitext_dataset
train_ds, _, _, _ = load_wikitext_dataset()
print(f'Dataset size: {len(train_ds)}')
print(f'Sample: {train_ds[0]}')
"

# 3. Check if model is in training mode
# In your training loop:
model.train()  # Not model.eval()!

# 4. Verify gradients are flowing
# Add to training loop:
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.abs().mean()}")
```

### Testing Issues

#### **Problem: Tests fail with import errors**

```bash
# Solution 1: Install package in editable mode
pip install -e .

# Solution 2: Run tests from repository root
cd /path/to/aletheion-llm
pytest tests/

# Solution 3: Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest tests/
```

#### **Problem: Pre-commit hooks failing**

```bash
# View what failed
pre-commit run --all-files

# Common fixes:

# Black/isort formatting
./scripts/format.sh
git add .

# Ruff linting issues
ruff check . --fix
git add .

# Temporarily skip hooks (not recommended)
git commit --no-verify -m "message"

# Update hooks to latest version
pre-commit autoupdate
```

### Data Issues

#### **Problem: WikiText dataset download fails**

```bash
# Solution 1: Download manually
python -c "
from datasets import load_dataset
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
print('Downloaded successfully!')
"

# Solution 2: Use cache
export HF_DATASETS_CACHE="/path/to/cache"
python examples/train.py

# Solution 3: Use offline mode if already downloaded
export HF_DATASETS_OFFLINE=1
```

#### **Problem: Tokenizer errors**

```bash
# Solution: Explicitly download tokenizer
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')
print('Tokenizer loaded successfully!')
"
```

### Performance Issues

#### **Problem: Training is very slow**

```bash
# Solutions:

# 1. Enable compilation (PyTorch 2.0+)
system:
  compile: true  # Can provide 30-50% speedup

# 2. Use more CPU workers for data loading
data:
  num_workers: 8  # Adjust based on CPU cores

# 3. Enable mixed precision
system:
  mixed_precision: true

# 4. Use flash attention (if available)
model:
  use_flash_attention: true  # Requires flash-attn package

# Install flash attention:
pip install flash-attn --no-build-isolation
```

### Common Error Messages

#### **`RuntimeError: Expected all tensors to be on the same device`**

```python
# Solution: Move all tensors to same device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
input_ids = input_ids.to(device)  # Move inputs too!
```

#### **`AttributeError: 'dict' object has no attribute 'logits'`**

```python
# Model returns a dict, not an object
outputs = model(input_ids)
logits = outputs['logits']  # Not outputs.logits
```

#### **`ValueError: Target size must be the same as input size`**

```python
# Common in loss computation
# Ensure targets match logits shape
logits = outputs['logits']  # (batch, seq_len, vocab_size)
targets = input_ids[:, 1:]  # Shift targets
logits = logits[:, :-1]     # Shift logits
# Now: logits.shape[:-1] == targets.shape
```

### Getting Help

If you can't resolve your issue:

1. **Search existing issues**: [GitHub Issues](https://github.com/AletheionAGI/aletheion-llm/issues)
2. **Check documentation**: See [`docs/`](docs/) directory
3. **Create a new issue**: Include:
   - Python version: `python --version`
   - PyTorch version: `python -c "import torch; print(torch.__version__)"`
   - CUDA version: `nvidia-smi` output
   - Full error traceback
   - Minimal code to reproduce the issue

---

## FAQ

### General Questions

**Q: What is Aletheion?**

A: Aletheion is a research project that adds epistemic uncertainty quantification to large language models. It uses special "gates" (Q₁ and Q₂) to estimate when the model is uncertain and adjusts its predictions accordingly, leading to better calibration and fewer hallucinations.

**Q: Is Aletheion ready for production use?**

A: No, Aletheion is currently in active research (Level 1 implementation). It's suitable for:
- ✅ Research and experimentation
- ✅ Academic projects
- ✅ Proof-of-concept applications
- ❌ Production deployments (not yet)

**Q: How does Aletheion relate to existing models like GPT-2/3/4?**

A: Aletheion is not a specific model but an architectural modification that can be applied to any transformer. Think of it as a "plugin" that adds uncertainty awareness. You can train an Aletheion version of any baseline transformer model.

### Technical Questions

**Q: What's the difference between Level 1, 2, and 3?**

A:
- **Level 1** (current): Epistemic gates at output layer only
  - Fastest to train
  - ~2% parameter overhead
  - Good calibration improvement
- **Level 2** (planned): Gates at attention level
  - Moderate training cost
  - ~5-10% parameter overhead
  - Better uncertainty estimation
- **Level 3** (planned): Full fractal architecture with gates everywhere
  - Most expensive to train
  - ~15-20% parameter overhead
  - Best uncertainty quantification

**Q: When should I use Aletheion instead of a baseline transformer?**

A: Use Aletheion when:
- ✅ You need calibrated confidence scores
- ✅ You want to detect and abstain on uncertain inputs
- ✅ You care about reducing hallucinations
- ✅ You need uncertainty-aware generation
- ❌ You only care about raw perplexity (baseline may be better)
- ❌ You can't afford any parameter overhead
- ❌ You need the absolute fastest inference

**Q: What are Q₁ and Q₂ gates?**

A:
- **Q₁ (Local Uncertainty Gate)**: Estimates uncertainty based on local context at each position. Answers: "Do I have enough evidence for this token?"
- **Q₂ (Cross-Context Gate)**: Estimates agreement across different contexts. Answers: "Do different parts of the model agree on this prediction?"
- **Combined**: Final confidence = Q₁ × Q₂

**Q: What is VARO loss?**

A: VARO (Variational Approximation to Rational Objectives) is the loss function for training Aletheion models:
```
L = L_CE + λ * ||u - u*||²
```
Where:
- `L_CE` is standard cross-entropy loss
- `u` is predicted uncertainty (1 - Q₁ × Q₂)
- `u*` is target uncertainty (from data/model statistics)
- `λ` controls trade-off (typically 0.1)

### Performance & Requirements

**Q: What hardware do I need to run Aletheion?**

A:
- **Minimum**: 8GB RAM, CPU-only (very slow training)
- **Recommended**: 16GB RAM, NVIDIA GPU with 8GB+ VRAM
- **Optimal**: 32GB+ RAM, NVIDIA GPU with 16GB+ VRAM (A100, RTX 3090, etc.)

**Q: How much slower is Aletheion compared to baseline?**

A:
- **Training**: ~5-10% slower (mostly due to gate computations)
- **Inference**: ~3-5% slower
- **Memory**: ~2% more GPU memory

**Q: Can I use Aletheion with my existing trained model?**

A: Not directly. You need to:
1. Start with same architecture
2. Add epistemic gates (Q₁, Q₂)
3. Train from scratch with VARO loss

Fine-tuning from a baseline checkpoint is possible but not recommended for best results.

### Usage Questions

**Q: How do I interpret uncertainty scores?**

A:
- **Uncertainty ∈ [0, 1]**: (1 - Q₁ × Q₂)
- **Low uncertainty (~0.0-0.3)**: Model is confident
- **Medium uncertainty (~0.3-0.7)**: Model is uncertain
- **High uncertainty (~0.7-1.0)**: Model should abstain

**Q: Can I use Aletheion with HuggingFace Transformers?**

A: Partial compatibility. The model follows HuggingFace conventions but isn't fully integrated yet. Planned for future releases.

**Q: How do I deploy an Aletheion model?**

A:
```python
# Save trained model
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config
}, 'aletheion_model.pt')

# Load for inference
checkpoint = torch.load('aletheion_model.pt')
model = AletheionTransformer(**checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Research Questions

**Q: Can I cite Aletheion in my research?**

A: Yes! See [Citation](#citation) section for BibTeX format.

**Q: What datasets work well with Aletheion?**

A:
- ✅ **WikiText**: Standard language modeling benchmark (default)
- ✅ **TruthfulQA**: Tests for truthfulness and calibration
- ✅ **Custom text datasets**: Any tokenized text data
- ⚠️ **Structured data**: Not recommended (designed for text)

**Q: How does Aletheion compare to Bayesian approaches?**

A: Aletheion is computationally cheaper than full Bayesian methods (e.g., Monte Carlo Dropout, ensemble methods) while still providing uncertainty estimates. Trade-off:
- **Bayesian**: More principled uncertainty, expensive
- **Aletheion**: Fast uncertainty estimation, less principled but practical

**Q: Can I extend Aletheion with my own gates?**

A: Yes! The gate architecture is modular:
```python
from src.aletheion.gates import LocalUncertaintyGate
import torch.nn as nn

class MyCustomGate(nn.Module):
    def forward(self, context):
        # Your custom uncertainty estimation
        return uncertainty_score  # Must be in [0, 1]
```

### Licensing & Commercial Use

**Q: Can I use Aletheion commercially?**

A: Yes, but you need a commercial license. Aletheion is dual-licensed:
- **AGPL-3.0**: Free for open-source and research
- **Commercial**: Required for closed-source/proprietary use

Contact [contact@alethea.tech](mailto:contact@alethea.tech) for commercial licensing.

**Q: What if I train my own Aletheion model?**

A: The AGPL license still applies to the model if trained using Aletheion code. Commercial license required for proprietary deployments.

---

## Roadmap

### Completed ✅
- [x] Baseline transformer implementation
- [x] Level 1 epistemic gates (Q₁, Q₂, VARO)
- [x] Pyramidal architecture framework
- [x] TruthfulQA integration
- [x] Comprehensive test suite
- [x] Documentation and papers

### In Progress 🔄
- [ ] Level 1 validation results (50% complete)
- [ ] Performance optimization
- [ ] Extended benchmarking

### Planned 🔜
- [ ] Level 2: Attention-level gates
- [ ] Level 3: Full fractal architecture
- [ ] HuggingFace Hub integration
- [ ] Pre-trained model releases
- [ ] Paper submission (NeurIPS/ICML)
- [ ] API and web demo

---

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:
- Code style and standards
- Testing requirements
- Pull request process
- Development setup

### Quick Contribution Guide

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/aletheion-llm.git

# Create a feature branch
git checkout -b feature/your-feature-name

# Make your changes and add tests
pytest tests/

# Submit a pull request
```

---

## Citation

If you use Aletheion in your research, please cite:

```bibtex
@software{aletheion2024,
  title = {Aletheion: Epistemic Uncertainty for Large Language Models},
  author = {Muniz, Felipe M.},
  year = {2024},
  url = {https://github.com/AletheionAGI/aletheion-llm},
  version = {0.1.0},
  license = {AGPL-3.0-or-later}
}
```

For the theoretical framework:

```bibtex
@article{muniz2021quality,
  title = {The Quality of Truth},
  author = {Muniz, Felipe M.},
  year = {2021},
  note = {Philosophical framework for epistemic uncertainty}
}
```

---

## License

AletheionGuard-Pypi is dual-licensed under:

- **GNU AGPL-3.0** – for open source and non-commercial use.
- **Aletheion Commercial License** – for proprietary or commercial use.

Commercial use **requires a paid license** from AlethionAGI.
For details, contact [contact@aletheionagi.com](mailto:contact@aletheionagi.com).


---

## Contact

📧 **Email:** [contact@alethea.tech](mailto:contact@alethea.tech)
💬 **Discord:** .lacivo
🐛 **Issues:** [GitHub Issues](https://github.com/AletheionAGI/aletheion-llm/issues)
🌐 **Website:** Coming soon

---

## Acknowledgments

This research builds upon decades of work in uncertainty quantification, Bayesian deep learning, and language model calibration. Special thanks to the open-source community and researchers advancing AI safety.

---

<div align="center">

**⚠️ Note:** This is active research. Results are preliminary and subject to change as experiments complete.

Made with ❤️ by the Aletheion team

[⬆ Back to Top](#aletheion-epistemic-uncertainty-for-large-language-models)

</div>
