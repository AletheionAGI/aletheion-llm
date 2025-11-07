# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Professional project structure and configuration
- Comprehensive code quality tooling (Ruff, Black, mypy, pre-commit)
- Community and governance documentation (CODE_OF_CONDUCT, SECURITY, SUPPORT, GOVERNANCE)
- Project roadmap with versioned milestones
- Cross-platform development configuration (.gitignore, .gitattributes, .editorconfig)
- CLI entry point configuration for `aletheion` command

### Changed
- Package name updated to `aletheion-llm` for PyPI distribution
- Python requirement raised to 3.10+
- Black line length standardized to 100 characters
- Enhanced development dependencies with testing and documentation tools

### Documentation
- CODE_OF_CONDUCT.md using Contributor Covenant v2.1
- SECURITY.md with vulnerability reporting process
- SUPPORT.md with comprehensive help resources
- GOVERNANCE.md defining project roles and decision-making
- ROADMAP.md with detailed version planning through 1.0
- Professional CHANGELOG following Keep a Changelog format

## [0.1.0] - 2024-11-04

### Added
- Initial project structure and baseline transformer implementation
- **Level 1**: Epistemic gates (Q₁, Q₂) for output-only gating
- **Pyramidal Architecture**: Multi-level uncertainty with Height coordinate
- **Fractal Q1/Q2**: Complete fractal implementation (Level 3)
- VARO (Variance-Adjusted Regularized Output) loss functions
- Core Aletheion architecture components in `src/aletheion/`
- Training scripts for baseline and Aletheion comparison
- Unit tests for epistemic components (gates, integration, pyramidal)
- Calibration metrics: ECE (Expected Calibration Error), Brier score
- Project documentation and comprehensive README
- Dual licensing (AGPL-3.0 and Commercial)
- Research paper v2 with detailed epistemic softmax theory
  - Epistemic softmax primitive definition
  - Fractal architecture levels description
  - VARO training methodology
  - Theoretical guarantees and proofs
  - Experimental design and ARC-AGI discussion
- Example scripts for training, evaluation, and generation
- TruthfulQA benchmark integration
- Abstention and out-of-domain testing capabilities
- Visualization tools for epistemic uncertainty
- WikiText-2 dataset integration

### Technical Details
- Epistemic uncertainty quantification via Q₁ (aleatoric) and Q₂ (epistemic) gates
- Height coordinate derived from Q₁×Q₂ for calibration
- Three implementation levels:
  - **Level 1**: Output-layer epistemic gating (~2% parameter overhead)
  - **Level 2**: Attention-level pyramidal gating
  - **Level 3**: Full fractal architecture across all layers
- VARO training procedure for uncertainty-aware optimization
- Parameter overhead: ~2% for Level 1 implementation
- Compatible with standard transformer architectures

### Documentation
- Comprehensive README with quick start guide
- Project structure documentation
- Theoretical foundation references
- Citation information (CITATION.cff)
- License information (AGPL-3.0 and Commercial options)
- Implementation notes and design decisions
- Training guides and best practices

### Dependencies
- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- Python >= 3.8 (now 3.10+ in unreleased)

## [0.0.1] - 2024-11-01

### Added
- Initial repository setup
- Basic project scaffolding
- License files (AGPL-3.0 and Commercial)
- Git repository initialization

---

## Types of Changes

This changelog uses the following categories:
- `Added` for new features
- `Changed` for changes in existing functionality
- `Deprecated` for soon-to-be removed features
- `Removed` for now removed features
- `Fixed` for any bug fixes
- `Security` for vulnerability fixes

---

## Links

- [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
- [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
- [Repository](https://github.com/AletheionAGI/aletheion-llm)
- [Issues](https://github.com/AletheionAGI/aletheion-llm/issues)
- [Releases](https://github.com/AletheionAGI/aletheion-llm/releases)

[Unreleased]: https://github.com/AletheionAGI/aletheion-llm/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/AletheionAGI/aletheion-llm/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/AletheionAGI/aletheion-llm/releases/tag/v0.0.1
