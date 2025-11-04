# Aletheion: Epistemic Uncertainty for Large Language Models

Implementation of fractally-applied epistemic softmax for calibrated, 
uncertainty-aware language models.

⚠️ **Status:** Level 1 training in progress (50% complete)

![Python](https://img.shields.io/badge/python-3.10-blue)
![License](https://img.shields.io/badge/license-AGPL--3.0-blue)
![Status](https://img.shields.io/badge/status-training-yellow)

## Background

Large language models hallucinate, contradict themselves, and rarely 
express calibrated uncertainty. Aletheion addresses this by replacing 
softmax operations with **epistemic softmax**—a gating mechanism that 
factors uncertainty into every decision.

This repository implements three progressive levels:
- **Level 1:** Output-only gating (current)
- **Level 2:** Attention-level gating (planned)
- **Level 3:** Full fractal architecture (planned)

**Theoretical foundation:**
- [The Quality of Truth](link) - Philosophical framework (2021)
- Aletheion preprint - Coming soon

---

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Train baseline
python train.py --config config/default.yaml

# Train Aletheion Level 1
python experiments/level1/compare_baseline_aletheion.py
```

---

## Project Structure
```
aletheion-llm/
├── src/              # Baseline transformer
├── src/aletheion/    # Epistemic components (Q₁, Q₂, VARO)
├── experiments/      # Training scripts
│   └── level1/       # Level 1 experiments
├── paper/            # Theoretical papers
├── tests/            # Unit tests
└── config/           # Training configs
```

---

## Results

### Level 1 (Output-Only)

**Training:** 50% complete (500/1000 steps)

**Early indicators:** Aletheion pulling ahead on loss (-0.014 gap)

**Full metrics:** Will be posted when training completes (~3h)

**Expected:**
- ECE improvement: -20% to -40%
- Perplexity improvement: -5% to -10%
- Parameter overhead: ~2%

---

## Roadmap

- [x] Baseline transformer implementation
- [x] Level 1 epistemic gates (Q₁, Q₂, VARO)
- [ ] Level 1 validation results (in progress)
- [ ] Level 2: Attention-level gates
- [ ] Level 3: Full fractal architecture
- [ ] Paper submission (NeurIPS/ICML)

---

## Citation
```bibtex
@misc{aletheion2024,
  title={Aletheion: Epistemic Uncertainty for Large Language Models},
  author={[Your Name]},
  year={2024},
  note={In preparation}
}
```

---

## License

AGPL-3.0 - See LICENSE file

---

## Contact

- Discord: [.lacivo]
- Email: [contact@alethea.tech]
- Issues: Use GitHub issues for bugs/questions

---

**Note:** This is active research. Results are preliminary and 
subject to change as experiments complete.