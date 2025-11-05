# Training Scripts Comparison & Project Architecture

## Table of Contents
1. [Project Architecture Overview](#project-architecture-overview)
2. [Directory Structure](#directory-structure)
3. [Key Components & Modules](#key-components--modules)
4. [Training Approaches Comparison](#training-approaches-comparison)
5. [Data Flow & Dependencies](#data-flow--dependencies)
6. [Configuration System](#configuration-system)
7. [Testing & Evaluation Framework](#testing--evaluation-framework)

---

## Project Architecture Overview

The **aletheion-llm** project implements a progressive series of transformer-based language models with epistemic uncertainty quantification. The architecture is organized into three main training approaches of increasing sophistication:

1. **Baseline**: Standard GPT-2 architecture without epistemic gates
2. **Pyramidal**: Geometric epistemic uncertainty with 4 interpretable forces
3. **Q1Q2**: Decomposed uncertainty (aleatoric vs epistemic) with fractal meta-level

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

## Training Approaches Comparison

### Training Script Hierarchy

#### **1. train_baseline.py**
```
Model: GPT2LMHeadModel (HuggingFace)
Parameters: ~45M
Loss: Cross-Entropy only
Metrics: Loss, Perplexity, ECE
```

**Purpose:** Establishes performance baseline without epistemic uncertainty

**Key Features:**
- Standard GPT-2 architecture
- No epistemic gates
- Pure language modeling objective

---

#### **2. train_pyramidal.py**
```
Model: AletheionPyramidalTransformer
Parameters: ~31M (smaller model)
Loss: L_CE + λ_base*L_base + λ_height*L_height
```

**Epistemic Features:**
- Height progression (should be [0.5, 0.7])
- Base stability (should be >0.7)
- 4 force weights (Memory, Pain, Choice, Exploration)
- Temperature modulation

**Metrics:** All baseline metrics + height, base, force weights

**Lambda Values:** `λ_base=0.005`, `λ_height=0.02` (moderate regularization)

---

#### **3. train_pyramidal_q1q2.py**
```
Model: AletheionPyramidalQ1Q2Transformer
Parameters: ~13M (smaller still)
Loss: L_CE + λ_base*L_base + λ_Q1*L_Q1 + λ_Q2*L_Q2 + λ_fractal*L_fractal + λ_height*L_height
```

**Epistemic Features:**
- **Q1 (Aleatoric):** uncertainty from evidence variance
- **Q2 (Epistemic):** uncertainty from model disagreement
- **Fractal:** meta-epistemic layer for Q1/Q2 uncertainty
- **Height:** derived from Q1, Q2, base_stability
- Comprehensive collapse detection
- **Expected ranges:** Q1∈[0.2,0.4], Q2∈[0.3,0.6], height∈[0.5,0.7]

**Metrics:** Most comprehensive - entropy, variance, min/max/range for all gates

**Lambda Values:** `λ_Q1=0.0015`, `λ_Q2=0.002`, `λ_fractal=0.0005`, `λ_base=0.001`, `λ_height=0.002`
- **10x smaller** to ensure L_CE dominates while epistemic terms guide gently

---

### Comparison Table

| Feature | Baseline | Pyramidal | Q1Q2 |
|---------|----------|-----------|------|
| **Model** | GPT2LMHeadModel | AletheionPyramidalTransformer | AletheionPyramidalQ1Q2Transformer |
| **Parameters** | ~45M | ~31M | ~13M |
| **Epistemic Gates** | None | Height + 4 forces | Q1 + Q2 + Fractal + Height |
| **Loss Components** | 1 (CE) | 3 (CE + base + height) | 6 (CE + base + Q1 + Q2 + fractal + height) |
| **Lambda Strategy** | None | Moderate (0.005-0.02) | Small (0.0005-0.002) |
| **Interpretability** | Low | High (4 named forces) | Medium (decomposed uncertainty) |
| **Complexity** | Low | Medium | High |
| **Purpose** | Performance baseline | Interpretable uncertainty | Decomposed uncertainty analysis |

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

---

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

## Key Architectural Insights

### Progression of Complexity

1. **Baseline**: Pure language modeling (no uncertainty)
2. **Pyramidal**: Single-metric uncertainty (height) with 4 interpretable forces
3. **Q1Q2**: Decomposed uncertainty (aleatoric vs epistemic) with fractal meta-level

### Parameter Efficiency

- **Baseline**: ~45M params (no overhead)
- **Pyramidal**: ~31M params (smaller architecture for comparison)
- **Q1Q2**: ~13M params (even smaller for ablation studies)

### Epistemic Sophistication

- **Baseline**: None
- **Pyramidal**: Single uncertainty + 4 forces (interpretable)
- **Q1Q2**: Dual uncertainty with variance + fractal layer + collapse detection

### Regularization Philosophy

- Uses YAML-based hyperparameter configuration
- Small lambda values (0.0005-0.02) to ensure L_CE dominance
- Comprehensive metrics tracking for debugging

---

## Summary

This architecture enables systematic comparison between:
- **Performance** (baseline)
- **Interpretability** (pyramidal)
- **Decomposition** (Q1Q2)

All three approaches can be trained, evaluated, and compared using the same dataset (WikiText-2) and evaluation metrics, allowing for rigorous scientific comparison of epistemic uncertainty quantification methods in language models.

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
