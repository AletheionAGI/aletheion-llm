# Changelog

All notable changes to the Aletheion project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### In Progress
- Level 1 training (50% complete, 500/1000 steps)
- Early results showing Aletheion pulling ahead on loss (-0.014 gap)

## [0.1.0] - 2024-11-04

### Added
- Initial project structure and baseline transformer implementation
- Level 1 epistemic gates (Q₁, Q₂, VARO) for output-only gating
- Core Aletheion architecture components in `src/aletheion/`
- Training scripts for baseline and Aletheion comparison
- Unit tests for epistemic components
- Project documentation and README
- Dual licensing (AGPL-3.0 and Commercial)
- Paper v2 with detailed epistemic softmax theory
  - Epistemic softmax primitive definition
  - Fractal architecture levels description
  - VARO training methodology
  - Theoretical guarantees and proofs
  - Experimental design and ARC-AGI discussion

### Technical Details
- Implemented epistemic uncertainty quantification (Q₁, Q₂)
- VARO (Variance-Adjusted Regularized Output) training procedure
- Baseline transformer with standard architecture
- Level 1 architecture with output-layer epistemic gating
- Parameter overhead: ~2%

### Documentation
- README with quick start guide
- Project structure documentation
- Theoretical foundation references
- Citation information
- License information

## [0.0.1] - 2024-11-01

### Added
- Initial repository setup
- Basic project scaffolding
- License files (AGPL-3.0 and Commercial)

---

## Release Notes

### Version 0.1.0 - Initial Public Release

This is the first public release of Aletheion, featuring the Level 1 implementation with output-only epistemic gating. The system is currently in training, with preliminary results showing promise in reducing hallucination and improving calibration.

**Status:** Training in progress (50% complete)

**Expected Results:**
- ECE improvement: -20% to -40%
- Perplexity improvement: -5% to -10%
- Parameter overhead: ~2%

**Known Limitations:**
- Level 1 only (output gating)
- Training incomplete
- Results preliminary

**Roadmap:**
- Complete Level 1 validation
- Implement Level 2 (attention-level gating)
- Implement Level 3 (full fractal architecture)
- Submit paper to NeurIPS/ICML

---

## Future Versions

### [0.2.0] - Planned
- Completed Level 1 training and validation results
- Full benchmark suite (perplexity, ECE, Brier score)
- TruthfulQA evaluation results
- Improved documentation and examples

### [0.3.0] - Planned
- Level 2 implementation (attention-level epistemic gates)
- Multi-level uncertainty propagation
- Enhanced calibration metrics

### [1.0.0] - Planned
- Level 3 implementation (full fractal architecture)
- Complete validation across all levels
- Production-ready release
- Paper publication

---

## Contributing

We welcome contributions! Please see CONTRIBUTING.md (coming soon) for guidelines.

## Contact

- Email: contact@alethea.tech
- Discord: .lacivo
- Issues: https://github.com/AletheionAGI/aletheion-llm/issues
