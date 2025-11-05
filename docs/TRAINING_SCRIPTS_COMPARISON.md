# Training Scripts Technical Comparison & Project Architecture

**Document Version:** 2.0
**Date:** 2025-11-05
**Author:** Aletheion Team

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Project Architecture Overview](#project-architecture-overview)
3. [Directory Structure](#directory-structure)
4. [Architecture Comparison](#architecture-comparison)
5. [Key Components & Modules](#key-components--modules)
6. [Loss Function Analysis](#loss-function-analysis)
7. [Epistemic Metrics](#epistemic-metrics)
8. [Collapse Detection System](#collapse-detection-system)
9. [Dataset and Training Configuration](#dataset-and-training-configuration)
10. [Logging and Monitoring](#logging-and-monitoring)
11. [Data Flow & Dependencies](#data-flow--dependencies)
12. [Configuration System](#configuration-system)
13. [Strengths and Weaknesses Analysis](#strengths-and-weaknesses-analysis)
14. [Performance Comparison Matrix](#performance-comparison-matrix)
15. [Technical Recommendations](#technical-recommendations)
16. [Testing & Evaluation Framework](#testing--evaluation-framework)
17. [Code Quality and Maintainability](#code-quality-and-maintainability)
18. [Future Improvements](#future-improvements)
19. [Conclusion](#conclusion)
20. [Key Files Reference](#key-files-reference)
21. [Appendix A: Quick Reference Commands](#appendix-a-quick-reference-commands)

---

## Executive Summary

This document provides a comprehensive technical comparison of three training scripts in the Aletheion LLM project:

1. **train_baseline.py** - Standard GPT-2 architecture without epistemic modeling
2. **train_pyramidal.py** - Pyramidal architecture with basic epistemic gates
3. **train_pyramidal_q1q2.py** - Advanced pyramidal architecture with Q1/Q2 uncertainty decomposition

### Key Findings

| Model | Architecture | Parameters | Uncertainty Modeling | Best Use Case |
|-------|--------------|------------|---------------------|---------------|
| **Baseline** | GPT-2 | ~45M | None | Performance benchmark |
| **Pyramidal** | Aletheion Pyramidal | ~31M | Basic (height, base, 4 forces) | Production-ready epistemic modeling |
| **Pyramidal Q1Q2** | Aletheion Q1/Q2 | ~13M | Advanced (Q1, Q2, fractal) | Research & uncertainty analysis |

---

## Project Architecture Overview

The **aletheion-llm** project implements a progressive series of transformer-based language models with epistemic uncertainty quantification. The architecture is organized into three main training approaches of increasing sophistication:

1. **Baseline**: Standard GPT-2 architecture without epistemic gates
2. **Pyramidal**: Geometric epistemic uncertainty with 4 interpretable forces
3. **Q1Q2**: Decomposed uncertainty (aleatoric vs epistemic) with fractal meta-level

This architecture enables systematic comparison between:
- **Performance** (baseline)
- **Interpretability** (pyramidal)
- **Decomposition** (Q1Q2)

All three approaches can be trained, evaluated, and compared using the same dataset (WikiText-2) and evaluation metrics, allowing for rigorous scientific comparison of epistemic uncertainty quantification methods in language models.

---

## Directory Structure

```
/home/user/aletheion-llm/
├── src/                           # Core library code (3,686 lines)
│   ├── model.py                   # Baseline transformer implementation
│   ├── attention.py               # CausalSelfAttention implementation
│   ├── tokenizer.py               # Tokenization utilities
│   ├── utils.py                   # Helper functions (config, seeding, scheduling)
│   └── aletheion/                 # Epistemic uncertainty components
│       ├── gates.py               # Q₁ (LocalUncertaintyGate) & Q₂ (CrossContextGate)
│       ├── loss.py                # VARO loss functions
│       ├── model.py               # AletheionTransformer (Level 1)
│       ├── pyramid.py             # PyramidalEpistemicGates & base architecture
│       ├── pyramidal_model.py     # AletheionPyramidalTransformer
│       ├── pyramid_q1q2_fractal.py # Advanced Q1/Q2/Fractal components
│       └── pyramidal_q1q2_model.py # AletheionPyramidalQ1Q2Transformer
│
├── experiments/level1/            # Training & evaluation scripts
│   ├── train_baseline.py          # Baseline GPT-2 (no epistemic gates)
│   ├── train_pyramidal.py         # Pyramidal architecture (4 force weights)
│   ├── train_pyramidal_q1q2.py    # Q1/Q2/Fractal decomposition (advanced)
│   ├── compare_baseline_aletheion.py
│   ├── compare_pyramidal.py
│   ├── compare_models.py
│   ├── test_truthfulqa.py
│   ├── test_out_of_domain.py
│   ├── test_abstention.py
│   └── visualize_epistemic.py
│
├── examples/                      # Usage examples
│   ├── train.py                   # Baseline training
│   ├── train_aletheion.py         # Aletheion Level 1 training
│   ├── eval.py
│   ├── generate.py
│   ├── quick_eval.py
│   └── test_calibration_fix.py
│
├── config/                        # YAML configuration files
│   ├── default.yaml               # Default baseline config
│   ├── small.yaml                 # Small model config
│   ├── medium.yaml                # Medium model config
│   └── aletheion_level1.yaml      # Aletheion Level 1 config
│
├── data/                          # Dataset handling
│   ├── dataset.py                 # TextDataset, load_wikitext_dataset()
│   └── prepare.py                 # Data preparation utilities
│
├── tests/                         # Unit & integration tests
│   ├── test_model.py
│   ├── test_attention.py
│   ├── test_training.py
│   └── aletheion/
│       ├── test_gates.py
│       ├── test_integration.py
│       └── test_pyramidal_q1q2.py
│
├── docs/                          # Documentation (18 markdown files)
│   ├── ALETHEION_LEVEL1_README.md
│   ├── PYRAMIDAL_EPISTEMOLOGY_README.md
│   ├── TRAINING_SCRIPTS_COMPARISON.md
│   ├── PYRAMIDAL_Q1Q2_FRACTAL.md
│   └── ... (implementation notes, guides, technical docs)
│
├── outputs/                       # Training outputs
│   ├── baseline/                  # Baseline model checkpoints & history
│   ├── pyramidal/                 # Pyramidal model checkpoints & history
│   └── comparison/                # Comparison results
│
├── scripts/                       # Shell scripts for training
├── paper/                         # Research papers & theoretical framework
└── audit/                         # Quality assurance reports
```

---

## Architecture Comparison

### Model Configuration

| Parameter | Baseline | Pyramidal | Pyramidal Q1Q2 |
|-----------|----------|-----------|----------------|
| **Model Class** | `GPT2LMHeadModel` | `AletheionPyramidalTransformer` | `AletheionPyramidalQ1Q2Transformer` |
| **d_model** | 512 | 512 | 256 |
| **n_layers** | 6 | 6 | 4 |
| **n_heads** | 8 | 8 | 4 |
| **d_ff** | 2048 | 2048 | 1024 |
| **max_seq_len** | 512 | 512 | 256 |
| **Total Parameters** | ~45M | ~31M | ~13M |
| **Dropout** | 0.1 | 0.1 | 0.1 |

**Location References:**
- Baseline: `experiments/level1/train_baseline.py:73-84`
- Pyramidal: `experiments/level1/train_pyramidal.py:68-83`
- Pyramidal Q1Q2: `experiments/level1/train_pyramidal_q1q2.py:367-382`

### Architectural Features

#### Baseline (train_baseline.py)
- **Architecture:** Standard GPT-2 transformer
- **Loss Function:** Simple cross-entropy
- **Special Features:** None (pure language modeling)
- **Epistemic Modeling:** ❌ None

#### Pyramidal (train_pyramidal.py)
- **Architecture:** Pyramidal transformer with epistemic gates
- **Loss Function:** PyramidalVAROLoss (CE + base + height regularization)
- **Special Features:**
  - Height progression tracking
  - Base stability monitoring
  - 4 force weights (Memory, Pain, Choice, Exploration)
  - Temperature modulation
- **Epistemic Modeling:** ✅ Basic (single uncertainty measure)

#### Pyramidal Q1Q2 (train_pyramidal_q1q2.py)
- **Architecture:** Pyramidal transformer with Q1/Q2 decomposition
- **Loss Function:** Extended VARO loss (CE + base + Q1 + Q2 + fractal + height)
- **Special Features:**
  - Q1 (aleatoric uncertainty) with variance
  - Q2 (epistemic uncertainty) with variance
  - Fractal meta-epistemic layer
  - Height derived from Q1, Q2, base_stability
  - Comprehensive collapse detection system
- **Epistemic Modeling:** ✅ Advanced (3-layer uncertainty decomposition)

---

## Key Components & Modules

### A. Baseline Transformer (`src/model.py`)

**Class:** `BaselineTransformer`

**Architecture:** Decoder-only transformer (GPT-2 style)

**Components:**
- Token & positional embeddings
- 6 transformer blocks (`TransformerBlock`)
- `CausalSelfAttention` (masked self-attention)
- FeedForward networks (MLP with GELU)
- Output linear projection for logits

**Key Parameters:**
- `vocab_size=50257`
- `d_model=512`
- `n_layers=6`
- `n_heads=8`
- `d_ff=2048`
- `max_seq_len=512`
- **Total:** ~45M parameters

---

### B. Epistemic Gates (`src/aletheion/gates.py`)

Two core uncertainty gates:

#### 1. **LocalUncertaintyGate (Q₁)**
- Maps context → [0,1] confidence score
- Architecture: `Linear(d_model→1) + Sigmoid`
- Estimates local evidence quality

#### 2. **CrossContextGate (Q₂)**
- Multi-head attention for cross-context consensus
- Aggregates information across attention heads
- Also outputs [0,1] confidence

#### 3. **epistemic_softmax()**
- Algorithm 1 from the paper
- Applies temperature modulation based on Q₁ & Q₂
- Replaces standard softmax for gated distributions

---

### C. Loss Functions (`src/aletheion/loss.py`)

Three progressive loss implementations:

#### 1. **VaroLoss** (VARO = Variance-Adjusted Ranking Optimization)
```
L_total = L_CE + λ * ||u - u*||²
```
- `u` = predicted uncertainty
- `u*` = target uncertainty
- **Used by:** `AletheionTransformer` (Level 1)

#### 2. **PyramidalVAROLoss**
```
L_total = L_CE + λ_base * L_base + λ_height * L_height
```
- Adds base stability & height calibration terms
- **Used by:** `AletheionPyramidalTransformer`

#### 3. **PyramidalVAROLossWithQ1Q2**
```
L_total = L_CE + λ_base * L_base + λ_Q1 * L_Q1 + λ_Q2 * L_Q2 + λ_fractal * L_fractal + λ_height * L_height
```
- Complete decomposition with 6 components
- **Used by:** `AletheionPyramidalQ1Q2Transformer`

---

### D. Pyramidal Architecture (`src/aletheion/pyramid.py`)

#### **PyramidalEpistemicGates**
5-vertex geometric structure:
- **Base vertices:** Memory, Pain, Choice, Exploration (4 forces)
- **Apex:** Truth = 1.0 (constant attractor)
- **Height:** proximity to truth

#### **PyramidalTemperatureModulator**
Scales softmax temperature based on height:
- High height (confident) → lower temperature → sharper distribution
- Low height (uncertain) → higher temperature → smoother distribution

---

## Loss Function Analysis

### Loss Components

| Component | Baseline | Pyramidal | Pyramidal Q1Q2 |
|-----------|----------|-----------|----------------|
| **Cross-Entropy (L_CE)** | ✅ | ✅ | ✅ |
| **Base Stability (L_base)** | ❌ | ✅ (λ=0.005) | ✅ (λ=0.001) |
| **Height Calibration (L_height)** | ❌ | ✅ (λ=0.02) | ✅ (λ=0.002) |
| **Q1 Calibration (L_Q1)** | ❌ | ❌ | ✅ (λ=0.0015) |
| **Q2 Calibration (L_Q2)** | ❌ | ❌ | ✅ (λ=0.002) |
| **Fractal Regularization (L_fractal)** | ❌ | ❌ | ✅ (λ=0.0005) |

**Total Loss Formulas:**

**Baseline:**
```
L_total = L_CE
```

**Pyramidal:**
```
L_total = L_CE + λ_base * L_base + λ_height * L_height
```

**Pyramidal Q1Q2:**
```
L_total = L_CE + λ_base * L_base + λ_Q1 * L_Q1 + λ_Q2 * L_Q2 + λ_fractal * L_fractal + λ_height * L_height
```

### Lambda Values Philosophy

The Q1Q2 model uses **10x smaller** lambda values to ensure that:
1. **L_CE dominates** - Primary objective remains language modeling
2. **Epistemic terms guide** - Uncertainty modeling provides gentle regularization
3. **Collapse prevention** - Small regularization prevents degenerate solutions

---

## Epistemic Metrics

### Tracked Metrics

| Metric Category | Baseline | Pyramidal | Pyramidal Q1Q2 |
|----------------|----------|-----------|----------------|
| **Loss & Perplexity** | ✅ | ✅ | ✅ |
| **Calibration (ECE, Brier)** | ✅ | ✅ | ❌ (not implemented) |
| **Height Progression** | ❌ | ✅ | ✅ |
| **Base Stability** | ❌ | ✅ | ✅ |
| **Force Weights** | ❌ | ✅ (4 weights) | ❌ |
| **Q1 (Aleatoric)** | ❌ | ❌ | ✅ (mean, var, min, max, range, entropy) |
| **Q2 (Epistemic)** | ❌ | ❌ | ✅ (mean, var, min, max, range, entropy) |
| **Fractal Uncertainty** | ❌ | ❌ | ✅ |
| **Collapse Detection** | ❌ | ⚠️ Limited | ✅ Comprehensive |

### Pyramidal Metrics Details

#### train_pyramidal.py (experiments/level1/train_pyramidal.py:169-183)
```python
metrics = {
    "loss", "ce_loss", "base_loss", "height_loss",
    "mean_height", "target_height", "base_stability",
    "Q1_mean", "Q2_mean",  # Basic Q1/Q2 from gates
    "lambda_base", "lambda_height",
    "w_memory", "w_pain", "w_choice", "w_exploration",  # 4 force weights
    "uncertainty"  # General uncertainty
}
```

#### train_pyramidal_q1q2.py (experiments/level1/train_pyramidal_q1q2.py:491-501)
```python
# From model.get_pyramidal_stats()
stats = {
    "Q1_mean", "Q1_var",  # Aleatoric uncertainty + variance
    "Q2_mean", "Q2_var",  # Epistemic uncertainty + variance
    "height_mean", "height_var",  # Height derived from Q1, Q2, base
    "base_stability_mean",
    "fractal_mean", "fractal_var",  # Meta-epistemic layer
}

# Additional distribution analysis (lines 508-558)
distribution_metrics = {
    "Q1_min", "Q1_max", "Q1_range", "Q1_target_mean",
    "Q2_min", "Q2_max", "Q2_range", "Q2_target_mean",
    "Q1_entropy", "Q2_entropy", "height_entropy"
}
```

---

## Collapse Detection System

### Detection Mechanisms

| Model | Collapse Detection | Warnings |
|-------|-------------------|----------|
| **Baseline** | ❌ None | N/A |
| **Pyramidal** | ⚠️ Basic | Height > 0.95 (overconfidence) |
| **Pyramidal Q1Q2** | ✅ Comprehensive | 6 collapse signals + entropy checks |

### Q1Q2 Collapse Signals (experiments/level1/train_pyramidal_q1q2.py:106-162)

The Q1Q2 model implements a sophisticated collapse detection system:

```python
def compute_collapse_signals(pyramid_outputs: dict) -> dict:
    """
    Detects 6 types of collapse:

    1. Q1_collapse: Q1 < 0.05 or Q1 > 0.90 (aleatoric uncertainty collapsed)
    2. Q2_collapse: Q2 < 0.05 or Q2 > 0.90 (epistemic uncertainty collapsed)
    3. height_collapse: height > 0.95 (overconfident predictions)
    4. fractal_explosion: fractal > 0.8 (meta-uncertainty explosion)
    5. Q1_saturated: entropy(Q1) < 0.1 (Q1 distribution saturated)
    6. Q2_saturated: entropy(Q2) < 0.1 (Q2 distribution degenerate)
    """
```

**Healthy Ranges (Q1Q2):**
- Q1_mean ∈ [0.2, 0.4]
- Q2_mean ∈ [0.3, 0.6]
- height ∈ [0.5, 0.7]
- fractal ∈ [0.1, 0.3]
- Q1_entropy > 0.3
- Q2_entropy > 0.3

---

## Dataset and Training Configuration

### Dataset Comparison

| Aspect | Baseline | Pyramidal | Pyramidal Q1Q2 |
|--------|----------|-----------|----------------|
| **Dataset** | WikiText-2 | WikiText-2 | TinyStories |
| **Purpose** | Language modeling benchmark | Language modeling benchmark | Fast experimentation |
| **Max Sequence Length** | 512 | 512 | 256 |
| **Tokenizer** | GPT2Tokenizer | GPT2Tokenizer | Custom (TinyStories) |
| **Cache Dir** | `.cache/wikitext` | `.cache/wikitext` | `data/tinystories` |

### Training Hyperparameters

| Parameter | Baseline | Pyramidal | Pyramidal Q1Q2 |
|-----------|----------|-----------|----------------|
| **Default Steps** | 2,000 | 10,000 | 5,000 |
| **Batch Size** | 4 | 4 | 32 |
| **Gradient Accumulation** | 1 | 1 | 1 |
| **Learning Rate** | 3e-4 | 3e-4 | 3e-4 |
| **Weight Decay** | N/A | N/A | 0.01 |
| **Warmup Steps** | N/A | N/A | 500 |
| **Gradient Clipping** | 1.0 | 1.0 | 1.0 |
| **Eval Interval** | 500 | 500 | 100 |
| **Save Interval** | 2,000 | 2,000 | 500 |
| **Num Workers** | 0 | 0 | 4 |

### Memory Optimization

All three scripts support:
- ✅ Gradient checkpointing (`--gradient-checkpointing`) - ~40% memory reduction
- ✅ Mixed precision FP16 (`--fp16`) - ~50% memory reduction
- ✅ Gradient accumulation for effective larger batch sizes
- ✅ Aggressive memory cleanup (periodic `torch.cuda.empty_cache()`)

---

## Logging and Monitoring

### Logging Systems

| Feature | Baseline | Pyramidal | Pyramidal Q1Q2 |
|---------|----------|-----------|----------------|
| **Framework** | Matplotlib | Matplotlib | TensorBoard |
| **History JSON** | ✅ | ✅ | ❌ (uses TB) |
| **Training Curves** | ✅ (4 plots) | ✅ (6 plots) | ✅ (TensorBoard) |
| **Real-time Metrics** | Progress bar | Progress bar | Progress bar + detailed prints |
| **Distribution Analysis** | ❌ | ❌ | ✅ (Q1/Q2 min/max/range) |
| **Collapse Warnings** | ❌ | ⚠️ Basic | ✅ Real-time warnings |

### Visualization Output

**Baseline (train_baseline.py:278-326):**
- 4 plots: Loss, Perplexity, ECE, Brier Score
- Saved to: `{output_dir}/training_curves.png`

**Pyramidal (train_pyramidal.py:380-452):**
- 6 plots:
  1. Loss curves (train + eval)
  2. Height progression (with collapse threshold)
  3. Base stability
  4. Force weights (4 vertices)
  5. Loss components (CE, base, height)
  6. Uncertainty
- Saved to: `{output_dir}/training_curves.png`

**Pyramidal Q1Q2 (train_pyramidal_q1q2.py:311):**
- TensorBoard logs: `experiments/level1/runs/{experiment_name}/tensorboard/`
- Tracks: Loss, LR, all pyramid metrics, collapse signals, distribution metrics

---

## Data Flow & Dependencies

### Training Pipeline Data Flow

```
WikiText-2 Dataset
        ↓
load_wikitext_dataset() [data/dataset.py]
        ↓
TextDataset (tokenized, padded)
        ↓
DataLoader (batch creation with collate_fn)
        ↓
Batch: {input_ids: [batch_size, seq_len], labels: [batch_size, seq_len]}
        ↓
┌─────────────────────────────────────────────────────────────┐
│ train_baseline.py                                           │
├─────────────────────────────────────────────────────────────┤
│ Model: GPT2LMHeadModel                                      │
│ Forward: input_ids → logits [batch, seq_len, vocab_size]   │
│ Loss: CrossEntropyLoss(logits, labels)                     │
│ Backward: gradient update                                   │
└─────────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────────┐
│ train_pyramidal.py                                          │
├─────────────────────────────────────────────────────────────┤
│ Model: AletheionPyramidalTransformer                        │
│ Forward:                                                     │
│  ├─ Transformer layers → hidden [batch, seq_len, d_model]  │
│  ├─ Output projection → logits [batch, seq_len, vocab]     │
│  ├─ PyramidalEpistemicGates:                                │
│  │  ├─ Compute 4 base forces [batch, seq_len, 4]          │
│  │  ├─ Compute height [batch, seq_len, 1]                 │
│  │  └─ Modulate temperature based on height               │
│  └─ Return: logits + pyramid metrics                        │
│ Loss: PyramidalVAROLoss                                      │
│  ├─ L_CE = CE(logits, labels)                              │
│  ├─ L_base = ||base_stability - target||²                  │
│  ├─ L_height = ||height - target_height||²                 │
│  └─ L_total = L_CE + λ_base*L_base + λ_height*L_height    │
│ Backward: update model + gates                              │
└─────────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────────┐
│ train_pyramidal_q1q2.py                                     │
├─────────────────────────────────────────────────────────────┤
│ Model: AletheionPyramidalQ1Q2Transformer                    │
│ Forward:                                                     │
│  ├─ Transformer layers → hidden [batch, seq_len, d_model]  │
│  ├─ Output projection → logits [batch, seq_len, vocab]     │
│  ├─ Q1 Gate (aleatoric):                                    │
│  │  ├─ Estimate evidence quality                           │
│  │  ├─ Track variance                                      │
│  │  └─ Check entropy (>0.3 = no collapse)                 │
│  ├─ Q2 Gate (epistemic):                                    │
│  │  ├─ Cross-context consensus                             │
│  │  ├─ Track variance                                      │
│  │  └─ Check entropy (>0.3 = no collapse)                 │
│  ├─ Fractal:                                                │
│  │  ├─ Meta-epistemic layer                                │
│  │  └─ Combine Q1/Q2 uncertainties                         │
│  ├─ Compute height from Q1, Q2, base_stability             │
│  ├─ Modulate temperature                                    │
│  └─ Return: logits + comprehensive metrics                  │
│ Loss: PyramidalVAROLossWithQ1Q2                             │
│  ├─ L_CE = CE(logits, labels)                              │
│  ├─ L_base = ||base_stability - target||²                  │
│  ├─ L_Q1 = ||Q1 - target_Q1||²                             │
│  ├─ L_Q2 = ||Q2 - target_Q2||²                             │
│  ├─ L_fractal = ||fractal - target_fractal||²              │
│  ├─ L_height = ||height - target_height||²                 │
│  └─ L_total = L_CE + λ*all_terms (small lambdas)          │
│ Backward + collapse detection                               │
└─────────────────────────────────────────────────────────────┘
```

### Key Dependencies

```
Model Dependencies:
├─ BaselineTransformer
│  ├─ CausalSelfAttention (src/attention.py)
│  └─ FeedForward (src/model.py)
│
├─ AletheionTransformer (extends BaselineTransformer)
│  ├─ LocalUncertaintyGate (Q₁)
│  ├─ CrossContextGate (Q₂)
│  └─ epistemic_softmax
│
├─ AletheionPyramidalTransformer (extends BaselineTransformer)
│  ├─ PyramidalEpistemicGates
│  │  ├─ LocalUncertaintyGate (Q₁)
│  │  ├─ CrossContextGate (Q₂)
│  │  └─ Pyramidal force calculations
│  └─ PyramidalTemperatureModulator
│
└─ AletheionPyramidalQ1Q2Transformer (extends BaselineTransformer)
   ├─ PyramidalEpistemicGatesWithQ1Q2
   ├─ EpistemicMultiHeadAttention
   └─ Fractal layer

Loss Dependencies:
├─ VaroLoss → LocalUncertaintyGate + CrossContextGate
├─ PyramidalVAROLoss → PyramidalEpistemicGates
└─ PyramidalVAROLossWithQ1Q2 → PyramidalEpistemicGatesWithQ1Q2

Data Dependencies:
├─ load_wikitext_dataset() → datasets library
├─ TextDataset → torch.utils.data.Dataset
├─ collate_fn → pad_sequence
└─ DataLoader → batch creation
```

---

## Configuration System

### Configuration Inheritance

All configs inherit from `/config/default.yaml` and override specific sections:

```yaml
# default.yaml structure (used as base)
model:
  vocab_size: 50257
  d_model: 512
  n_layers: 6
  n_heads: 8
  d_ff: 2048

training:
  batch_size: 32
  learning_rate: 3.0e-4
  max_steps: 100000

# aletheion_level1.yaml extends default.yaml with:
model:
  epistemic:
    q1_threshold: 0.7
    q2_threshold: 0.7
    lambda_varo: 0.1
    u_star_method: head_variance
```

---

## Strengths and Weaknesses Analysis

### Baseline GPT-2

#### ✅ Strengths
1. **Simplicity** - Easy to understand, debug, and deploy
2. **Performance Benchmark** - Provides clean baseline for comparison
3. **Proven Architecture** - GPT-2 is well-tested and robust
4. **Fast Training** - No epistemic overhead (2,000 default steps)
5. **Calibration Metrics** - Tracks ECE and Brier score for comparison
6. **Largest Model** - 45M parameters for maximum capacity
7. **Production Ready** - HuggingFace GPT2LMHeadModel, mature ecosystem

#### ❌ Weaknesses
1. **No Uncertainty Modeling** - Cannot estimate prediction confidence
2. **No Epistemic Awareness** - Treats all errors equally
3. **Poor Calibration** - Standard softmax overconfident by default
4. **No Collapse Detection** - Cannot detect degenerate training
5. **Limited Introspection** - Only loss/perplexity, no internal metrics
6. **No Research Value** - Standard architecture, no novelty

**Best Use Cases:**
- Performance benchmarking
- Ablation studies (to prove epistemic modeling value)
- Production deployment where uncertainty is not needed
- Quick prototyping and sanity checks

---

### Pyramidal Transformer

#### ✅ Strengths
1. **Production-Ready Epistemic Modeling** - Balanced complexity/utility
2. **4-Force Interpretation** - Memory, Pain, Choice, Exploration weights
3. **Height/Base Metaphor** - Intuitive pyramid visualization
4. **WikiText-2 Benchmark** - Standard dataset for comparison
5. **Proven Loss Function** - PyramidalVAROLoss with good defaults
6. **Moderate Model Size** - 31M params, good balance
7. **Matplotlib Plots** - Self-contained visualization
8. **Temperature Modulation** - Adaptive sharpening based on uncertainty
9. **Multiple Height Methods** - error_based, entropy_based, loss_based

#### ❌ Weaknesses
1. **Limited Uncertainty Decomposition** - Single uncertainty measure
2. **Basic Collapse Detection** - Only height threshold check
3. **No Distribution Analysis** - Cannot track Q1/Q2 min/max/range
4. **4-Force Overhead** - Additional parameters may not be necessary
5. **No Real-time Warnings** - Collapse discovered post-training
6. **No TensorBoard** - Less convenient for long experiments
7. **Conservative Training** - 10,000 steps default (slower iteration)

**Best Use Cases:**
- Production deployment with epistemic modeling
- Experiments requiring interpretable force weights
- Research on pyramidal height dynamics
- Moderate-scale uncertainty quantification
- Teaching/explaining epistemic AI concepts

---

### Pyramidal Q1Q2 Transformer

#### ✅ Strengths
1. **Advanced Uncertainty Decomposition** - Separates Q1 (aleatoric) and Q2 (epistemic)
2. **Comprehensive Collapse Detection** - 6 signals + entropy monitoring
3. **Distribution Analysis** - Tracks min/max/range/entropy for Q1/Q2
4. **Real-time Warnings** - Detects collapse during training
5. **Fractal Meta-Layer** - Third-order epistemic modeling
6. **TensorBoard Integration** - Rich visualization and monitoring
7. **Research-Oriented** - Designed for epistemic architecture experiments
8. **Fast Iteration** - Smaller model (13M) + shorter sequences (256)
9. **Aggressive Monitoring** - Every 50 steps prints detailed diagnostics
10. **Theoretical Soundness** - Proper Q1/Q2 separation aligns with UQ theory

#### ❌ Weaknesses
1. **TinyStories Dataset** - Less standard than WikiText-2 (NOW FIXED in this PR)
2. **Smaller Model** - 13M params, less capacity than baseline/pyramidal
3. **Shorter Sequences** - 256 vs 512, less context
4. **Complex Loss Function** - 6 components, harder to debug
5. **No Force Weights** - Lost 4-force interpretability from pyramidal
6. **No Calibration Metrics** - ECE/Brier not implemented
7. **Research Code Quality** - More experimental, less production-ready
8. **Hyperparameter Sensitivity** - 6 lambdas to tune (λ_base, λ_Q1, λ_Q2, λ_fractal, λ_height, λ_temp)
9. **TensorBoard Dependency** - Requires separate viewer

**Best Use Cases:**
- Research on uncertainty decomposition (aleatoric vs epistemic)
- Experiments requiring collapse detection
- Fast prototyping of epistemic architectures
- Analysis of Q1/Q2 dynamics and distributions
- Meta-epistemic studies (fractal layer)
- Teaching advanced uncertainty quantification

---

## Performance Comparison Matrix

| Dimension | Baseline | Pyramidal | Pyramidal Q1Q2 |
|-----------|----------|-----------|----------------|
| **Training Speed** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Model Capacity** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Uncertainty Modeling** | ⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Interpretability** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Production Readiness** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| **Research Value** | ⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Monitoring/Debugging** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Collapse Prevention** | ⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Calibration** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| **Code Complexity** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |

**Legend:** ⭐ = Poor, ⭐⭐⭐ = Average, ⭐⭐⭐⭐⭐ = Excellent

---

## Technical Recommendations

### When to Use Each Model

| Scenario | Recommended Model | Rationale |
|----------|------------------|-----------|
| **Initial benchmarking** | Baseline | Establishes performance ceiling without epistemic overhead |
| **Production deployment** | Pyramidal | Best balance of uncertainty modeling and complexity |
| **Uncertainty research** | Pyramidal Q1Q2 | Most advanced epistemic features and monitoring |
| **Teaching/demos** | Pyramidal | 4-force weights most intuitive |
| **Fast prototyping** | Pyramidal Q1Q2 | Smaller model, faster training |
| **Calibration studies** | Baseline or Pyramidal | Both track ECE/Brier |
| **Collapse analysis** | Pyramidal Q1Q2 | Comprehensive detection system |
| **Large-scale training** | Baseline or Pyramidal | Larger models, more capacity |

### Migration Path

**Recommended Development Flow:**

```
1. Baseline (train_baseline.py)
   ↓ Establish performance baseline

2. Pyramidal (train_pyramidal.py)
   ↓ Validate epistemic modeling helps

3. Pyramidal Q1Q2 (train_pyramidal_q1q2.py)
   ↓ Research Q1/Q2 decomposition

4. Production (Pyramidal or custom)
   ↓ Deploy best architecture for use case
```

### Hyperparameter Tuning Guidelines

#### Baseline
- **lr:** 3e-4 (standard GPT-2)
- **batch_size:** Increase for stability (8-32)
- **steps:** 2,000-10,000 depending on convergence

#### Pyramidal
- **lambda_base:** 0.001-0.01 (higher = stronger base stability)
- **lambda_height:** 0.01-0.05 (higher = more height control)
- **lr:** 3e-4 (same as baseline)
- **height_method:** Start with `error_based`, try `entropy_based` if collapse occurs

#### Pyramidal Q1Q2
- **lambda_base:** 0.0005-0.002 (small, let L_CE dominate)
- **lambda_Q1:** 0.001-0.003 (calibrate aleatoric uncertainty)
- **lambda_Q2:** 0.001-0.005 (calibrate epistemic uncertainty)
- **lambda_fractal:** 0.0001-0.001 (gentle meta-regularization)
- **lambda_height:** 0.001-0.005 (derived from Q1/Q2)
- **lr:** 3e-4 with 500 warmup steps
- **Monitor:** Q1/Q2 distributions every 50 steps, adjust if collapsed

---

## Testing & Evaluation Framework

| Script | Purpose | Metrics |
|--------|---------|---------|
| `compare_baseline_aletheion.py` | Compare baseline vs Aletheion L1 | Loss, perplexity, ECE, Brier |
| `compare_pyramidal.py` | Compare pyramidal variants | Height, base, forces, Q1/Q2 |
| `compare_models.py` | Multi-model comparison | All metrics across all models |
| `test_truthfulqa.py` | TruthfulQA benchmark | Accuracy, truthfulness, informativeness |
| `test_out_of_domain.py` | OOD robustness | Abstention rates, calibration on unseen data |
| `test_abstention.py` | Uncertainty-based abstention | Rejection curves, coverage-accuracy tradeoff |
| `visualize_epistemic.py` | Visualization tools | Heatmaps, distributions, progression plots |

---

## Code Quality and Maintainability

| Aspect | Baseline | Pyramidal | Pyramidal Q1Q2 |
|--------|----------|-----------|----------------|
| **Documentation** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Type Hints** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Error Handling** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Memory Management** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Modularity** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Testability** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Dependencies** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

All three scripts demonstrate:
- ✅ Clear docstrings
- ✅ Aggressive memory cleanup (prevent OOM)
- ✅ Resume from checkpoint support
- ✅ Config saving (JSON)
- ✅ Progress bars (tqdm)
- ✅ Memory monitoring
- ✅ Gradient accumulation
- ✅ Mixed precision (FP16)
- ✅ Gradient checkpointing

---

## Future Improvements

### All Models
- [ ] Add learning rate schedulers (cosine, linear decay)
- [ ] Implement early stopping
- [ ] Add validation loss-based checkpointing (save best model)
- [ ] Support for multi-GPU training (DDP)
- [ ] Integrated hyperparameter search (Optuna)
- [ ] Automated testing suite

### Pyramidal Models
- [ ] Add ECE/Brier to Q1Q2 (currently only in baseline/pyramidal)
- [ ] Unified logging (make all use TensorBoard OR Matplotlib)
- [ ] Standardize checkpoint naming across models
- [ ] Cross-model evaluation script (compare all three on same test set)
- [ ] Uncertainty visualization tools (Q1 vs Q2 scatter plots)

### Q1Q2 Specific
- [ ] **WikiText-2 support** ✅ DONE (this PR)
- [ ] Implement force weights alongside Q1/Q2
- [ ] Add calibration metrics (ECE/Brier)
- [ ] Production-ready mode (less verbose logging)
- [ ] Automatic lambda tuning based on collapse signals
- [ ] Q1/Q2 distribution visualization in TensorBoard

---

## Conclusion

The three training scripts represent different points in the complexity-capability tradeoff:

1. **Baseline** excels at simplicity and serves as the performance ceiling
2. **Pyramidal** provides production-ready epistemic modeling with interpretable force weights
3. **Pyramidal Q1Q2** pushes the research frontier with advanced uncertainty decomposition

**Key Insight:** The progression from Baseline → Pyramidal → Q1Q2 demonstrates increasing epistemic sophistication at the cost of complexity. Choose based on your use case:
- Need a benchmark? → **Baseline**
- Building a product? → **Pyramidal**
- Doing research? → **Pyramidal Q1Q2**

**Next Steps:**
1. ✅ Standardize Q1Q2 to use WikiText-2 (this PR)
2. Add cross-model evaluation script
3. Publish comparative results on standard benchmarks
4. Ablation study on lambda values
5. Production deployment guide for pyramidal models

---

## Key Files Reference

| File Path | Purpose | Key Components |
|-----------|---------|-----------------|
| `/src/model.py` | Baseline transformer | BaselineTransformer, TransformerBlock, FeedForward |
| `/src/attention.py` | Attention mechanism | CausalSelfAttention (masked) |
| `/src/aletheion/gates.py` | Epistemic gates | LocalUncertaintyGate (Q₁), CrossContextGate (Q₂), epistemic_softmax |
| `/src/aletheion/loss.py` | Loss functions | VaroLoss, PyramidalVAROLoss, PyramidalVAROLossWithQ1Q2 |
| `/src/aletheion/model.py` | Level 1 model | AletheionTransformer (Q₁+Q₂ at output) |
| `/src/aletheion/pyramidal_model.py` | Pyramidal model | AletheionPyramidalTransformer (4 forces + height) |
| `/src/aletheion/pyramid.py` | Pyramidal geometry | PyramidalEpistemicGates, compute_pyramidal_metrics |
| `/src/aletheion/pyramidal_q1q2_model.py` | Q1/Q2 model | AletheionPyramidalQ1Q2Transformer |
| `/src/aletheion/pyramid_q1q2_fractal.py` | Advanced gates | PyramidalEpistemicGatesWithQ1Q2, EpistemicMultiHeadAttention |
| `/data/dataset.py` | Data loading | TextDataset, load_wikitext_dataset (WikiText-2) |
| `/data/prepare.py` | Data prep | Dataset preprocessing utilities |
| `/config/default.yaml` | Base config | Shared hyperparameters |
| `/config/aletheion_level1.yaml` | L1 config | Epistemic hyperparameters |
| `/experiments/level1/train_baseline.py` | Baseline training | Script for baseline model |
| `/experiments/level1/train_pyramidal.py` | Pyramidal training | Pyramidal model training |
| `/experiments/level1/train_pyramidal_q1q2.py` | Q1/Q2 training | Complete Q1/Q2/Fractal training |
| `/examples/train_aletheion.py` | Example training | Using config-based training |

---

## Appendix A: Quick Reference Commands

### Train Baseline
```bash
# Standard training
python experiments/level1/train_baseline.py --steps 2000 --output outputs/baseline

# With memory optimization
python experiments/level1/train_baseline.py --steps 2000 --fp16 --gradient-checkpointing
```

### Train Pyramidal
```bash
# Standard training
python experiments/level1/train_pyramidal.py --steps 10000 --output outputs/pyramidal

# With custom lambdas
python experiments/level1/train_pyramidal.py --steps 10000 \
    --lambda-base 0.01 --lambda-height 0.02 --height-method error_based
```

### Train Pyramidal Q1Q2
```bash
# Standard training (TinyStories - old)
python experiments/level1/train_pyramidal_q1q2.py --max_steps 5000 \
    --experiment_name pyramidal_q1q2_v1

# With WikiText-2 (new in this PR)
python experiments/level1/train_pyramidal_q1q2.py --max_steps 5000 \
    --data_dir .cache/wikitext --experiment_name pyramidal_q1q2_wikitext

# Custom lambdas
python experiments/level1/train_pyramidal_q1q2.py --max_steps 5000 \
    --lambda_Q1 0.0015 --lambda_Q2 0.002 --lambda_fractal 0.0005
```

---

**Document Changelog:**
- v1.0 (2025-11-05): Initial comprehensive comparison (docs/ version)
- v2.0 (2025-11-05): Consolidated version combining technical comparison + project architecture
- Future: Will update with experimental results and ablation studies
