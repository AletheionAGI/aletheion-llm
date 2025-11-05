---
title: Aletheion LLM - Epistemic Uncertainty
emoji: 🗡️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: agpl-3.0
---

# 🗡️ Aletheion: Epistemic Uncertainty for Large Language Models

Interactive demo of the **Aletheion Pyramidal Model** - a language model with built-in epistemic uncertainty quantification.

## What is Aletheion?

Aletheion addresses a fundamental challenge in AI: **models that know when they don't know**.

Large language models hallucinate, contradict themselves, and rarely express calibrated uncertainty. Aletheion replaces traditional softmax operations with **epistemic softmax** - a gating mechanism that factors uncertainty into every token prediction.

### Key Innovation: Pyramidal Epistemology

Aletheion uses a **5-vertex geometric structure** to represent epistemic states:

```
        Truth (Apex = 1.0)
           ▲
          /|\
         / | \
        /  |  \
       /   |   \
      /____|____\
     Base Forces:
     - Memory
     - Pain (Loss)
     - Choice (Entropy)
     - Exploration
```

The model learns to **climb toward truth** by balancing these forces, producing:
- Better calibrated predictions
- Explicit uncertainty quantification
- Reduced hallucination rates

## Benchmark Results

**Expected Calibration Error (ECE):**
- **Baseline GPT-2:** 0.104
- **Aletheion Pyramidal:** 0.011
- **Improvement:** 89% reduction (10x better calibration)

**What this means:** Aletheion's confidence scores accurately reflect its true accuracy, unlike baseline models that are systematically overconfident.

## Epistemic Uncertainty Metrics

When you generate text with Aletheion, you get:

- **Height**: Proximity to truth (0 = uncertain, 1 = certain)
- **Base Stability**: Consistency of epistemic forces
- **Uncertainty**: 1 - Height (explicit uncertainty quantification)
- **Confidence**: Height × Base Stability (combined certainty measure)
- **ECE**: Expected Calibration Error (alignment of confidence with accuracy)

## Architecture

This demo showcases the **Pyramidal architecture** (Level 1 implementation):
- Output-only epistemic gates
- 4 base forces: Memory, Pain, Choice, Exploration
- Height-based temperature modulation
- VARO loss (Variational Approximation to Rational Objectives)

**Note:** The Q1Q2 variant (Level 2) with attention-level gates is coming soon.

## Model Details

- **Base Architecture:** GPT-2 style transformer (6 layers, 512d model, 8 heads)
- **Vocab Size:** 50,257 (GPT-2 tokenizer)
- **Parameters:** ~42M (baseline) + ~2% overhead for epistemic gates
- **Training Data:** TinyStories dataset
- **License:** AGPL-3.0 (open source) or Commercial

## How to Use

1. Enter a text prompt
2. Adjust generation parameters (temperature, top-k, top-p)
3. Click "Generate"
4. View the generated text and uncertainty metrics

**Interpretation Tips:**
- **High Height** (>0.8): Model is confident and likely accurate
- **Low Height** (<0.5): Model is uncertain, predictions may be unreliable
- **High Base Stability**: Consistent epistemic state
- **Low ECE**: Well-calibrated predictions

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

## Links

- 📄 **Paper:** [Geometry of Knowing](./assets/paper.pdf) (included in Space)
- 💻 **GitHub:** [AletheionAGI/aletheion-llm](https://github.com/AletheionAGI/aletheion-llm)
- 📊 **Training Curves:** See `assets/training_curves.png`
- 📧 **Contact:** contact@alethea.tech

## Technical Details

**Loss Function:**
```
L_total = L_CE + λ_base * L_base + λ_height * L_height
```

Where:
- `L_CE`: Cross-entropy loss
- `L_base`: Base stability regularization (encourages consistent forces)
- `L_height`: Height calibration (aligns height with prediction accuracy)

**Base Forces:**
- **Memory (w₁)**: Reliance on learned patterns
- **Pain (w₂)**: Sensitivity to loss/error
- **Choice (w₃)**: Exploration of alternatives (entropy)
- **Exploration (w₄)**: Deviation from norm (KL divergence)

**Height Computation:**
```
height = w₁ * memory + w₂ * pain + w₃ * choice + w₄ * exploration
where: w₁ + w₂ + w₃ + w₄ = 1 (normalized base forces)
```

## Limitations & Future Work

**Current Limitations:**
- Demo uses randomly initialized or early checkpoint weights
- Limited to short sequences (512 tokens max)
- CPU inference only (slower than GPU)
- Simplified ECE estimation in demo

**Coming Soon:**
- Q1Q2 variant with attention-level gates
- Full fractal architecture (Level 3)
- Pre-trained model releases
- Extended benchmarks (TruthfulQA, MMLU, etc.)

## License

**AGPL-3.0** for open source use. Commercial licenses available.

For commercial use, contact: contact@alethea.tech

---

**Made with ❤️ by the Aletheion team**

*Climbing toward truth, one token at a time.*
