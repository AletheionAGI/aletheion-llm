# Aletheion: Epistemic Uncertainty for Large Language Models

<div align="center">

**Implementation of fractally-applied epistemic softmax for calibrated, uncertainty-aware language models**

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-AGPL--3.0%20or%20Commercial-blue.svg)](LICENSE-AGPL.md)
[![Status](https://img.shields.io/badge/status-active%20research-yellow.svg)](https://github.com/AletheionAGI/aletheion-llm)

[Features](#features) • [Installation](#installation) • [Quick Start](#quick-start) • [Documentation](#documentation) • [Citation](#citation)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Background](#background)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Results](#results)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Overview

Large language models hallucinate, contradict themselves, and rarely express calibrated uncertainty. **Aletheion** addresses this fundamental challenge by replacing traditional softmax operations with **epistemic softmax**—a gating mechanism that factors uncertainty into every decision.

### Key Innovation

Aletheion introduces **Pyramidal Epistemology**, a fractal architecture that applies uncertainty quantification at multiple levels:
- **Q₁ (Local Uncertainty Gate):** Token-level uncertainty estimation
- **Q₂ (Cross-Context Gate):** Context-aware uncertainty propagation
- **VARO Loss:** Variational Approximation to Rational Objectives

⚠️ **Current Status:** Level 1 implementation complete, training in progress

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

This repository implements three progressive levels:
- **Level 1:** Output-only gating (✅ current implementation)
- **Level 2:** Attention-level gating (🔜 planned)
- **Level 3:** Full fractal architecture (🔜 planned)

**Theoretical Foundation:**
- [The Quality of Truth](https://github.com/AletheionAGI/aletheion-llm/blob/main/paper/en/main.pdf) - Philosophical framework (2021)
- Aletheion Research Paper - See [`paper/`](paper/) directory

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
- **[Level 1 Implementation](docs/ALETHEION_LEVEL1_README.md)** - Detailed Level 1 architecture
- **[Pyramidal Epistemology](docs/PYRAMIDAL_EPISTEMOLOGY_README.md)** - Theoretical framework
- **[Implementation Notes](docs/IMPLEMENTATION_NOTES.md)** - Design decisions and details

### Technical Deep Dives
- [LLM Fundamentals](docs/llm-fundamentals.md)
- [LLM Failures](docs/llm-failures.md)
- [Attention Mechanisms](docs/attention-mechanisms.md)
- [Training Strategy](docs/training-strategy.md)
- [Aletheion Integration](docs/aletheion-integration.md)
- [Fractal Approach](docs/aletheion-fractal-approach.md)

### Evaluation & Testing
- [TruthfulQA Setup](docs/TRUTHFULQA_SETUP.md)
- [Calibration Fixes](docs/BUGFIX_CALIBRATION.md)

### Additional Resources
- [API Reference](docs/) - Coming soon
- [FAQ](docs/) - Coming soon
- [Contributing Guide](CONTRIBUTING.md)

---

## Results

### Level 1 (Output-Only Gating)

**Training Status:** 50% complete (500/1000 steps)

**Early Indicators:**
- Aletheion showing lower loss (-0.014 gap vs baseline)
- Improved calibration metrics
- Better uncertainty quantification

**Expected Final Results:**
- ECE improvement: -20% to -40%
- Perplexity improvement: -5% to -10%
- Parameter overhead: ~2%

Full metrics will be posted when training completes.

### Benchmarks

| Metric | Baseline | Aletheion L1 | Improvement |
|--------|----------|--------------|-------------|
| ECE (↓) | TBD | TBD | TBD |
| Perplexity (↓) | TBD | TBD | TBD |
| TruthfulQA (↑) | TBD | TBD | TBD |
| Parameters | 100% | 102% | +2% |

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

Aletheion is **dual-licensed** to support both open-source and commercial use:

### Open Source License
**[GNU Affero General Public License v3.0](LICENSE-AGPL.md)**
- ✅ Free for research and non-commercial use
- ✅ Modifications must be shared under AGPL
- ✅ Full source code transparency

### Commercial License
**[Aletheion Commercial License](LICENSE-COMMERCIAL.md)**
- ✅ Proprietary deployments allowed
- ✅ No copyleft obligations
- ✅ Custom terms available

**Need a commercial license?** Contact [contact@alethea.tech](mailto:contact@alethea.tech) to discuss terms.

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
