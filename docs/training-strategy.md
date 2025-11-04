# Training Strategy: Aletheion-Native LLM

## Table of Contents
1. [Overview](#overview)
2. [Four-Phase Training Curriculum](#four-phase-training-curriculum)
3. [Phase 1: Base Pre-Training](#phase-1-base-pre-training)
4. [Phase 2: Q₁ Head Training](#phase-2-q1-head-training)
5. [Phase 3: Q₂ Head Training](#phase-3-q2-head-training)
6. [Phase 4: VARO Integration](#phase-4-varo-integration)
7. [Joint Fine-Tuning](#joint-fine-tuning)
8. [Hyperparameter Guide](#hyperparameter-guide)
9. [Evaluation Metrics](#evaluation-metrics)
10. [References](#references)

---

## Overview

Training an **Aletheion-native LLM** requires a **phased curriculum** that:
1. Develops strong base language modeling capabilities
2. Learns to detect coherence failures (Q₁)
3. Learns to detect semantic drift (Q₂)
4. Integrates variance anti-resonance (VARO) regularization
5. Jointly optimizes all components

This document provides a **complete training recipe** from scratch to deployment.

---

## Four-Phase Training Curriculum

### High-Level Strategy

```
Phase 1: Base Pre-Training (Standard LLM)
  ↓
  Freeze LLM, train Q₁ head
  ↓
Phase 2: Q₁ Head Training (Coherence Detection)
  ↓
  Freeze Q₁, train Q₂ head
  ↓
Phase 3: Q₂ Head Training (Drift Detection)
  ↓
  Unfreeze all, add VARO
  ↓
Phase 4: VARO Integration + Joint Fine-Tuning
  ↓
  Aletheion-Native LLM ✓
```

**Why phased?**
- **Stability**: Training all components simultaneously can be unstable
- **Interpretability**: Isolate each component's contribution
- **Efficiency**: Reuse pre-trained LLM checkpoints
- **Flexibility**: Can skip phases if fine-tuning existing model

### Timeline (1B Parameter Model)

| Phase | Objective | Tokens | GPU-Hours | Checkpoint |
|-------|-----------|--------|-----------|------------|
| 1 | Base LLM | 100B | 500-1000 | `base.pt` |
| 2 | Q₁ head | 10B | 50-100 | `base+q1.pt` |
| 3 | Q₂ head | 10B | 50-100 | `base+q1+q2.pt` |
| 4 | VARO + joint | 20B | 100-200 | `aletheion.pt` |
| **Total** | | **140B** | **700-1400** | |

---

## Phase 1: Base Pre-Training

### Objective

Train a **standard autoregressive LLM** to minimize next-token prediction loss:

$$
\mathcal{L}_{\text{LM}} = -\frac{1}{T} \sum_{t=1}^{T} \log P(x_t \mid x_{<t}; \theta)
$$

This phase is **identical to standard LLM training** (GPT, LLaMA, etc.).

### Architecture

```python
from torch import nn

class BaseLLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_seq_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids):
        # Standard forward pass (see llm-fundamentals.md)
        pass
```

### Training Configuration

**Model**: 1B parameters
- $d_{\text{model}} = 2048$
- $n_{\text{heads}} = 16$
- $d_{\text{ff}} = 8192$
- $n_{\text{layers}} = 18$
- $V = 50{,}000$

**Data**: Web corpus (e.g., C4, The Pile)
- **100B tokens** (100 epochs on 1B tokens)

**Optimizer**: AdamW
- Learning rate: $6 \times 10^{-4}$ (peak)
- Warmup: 4000 steps
- Decay: Cosine to $6 \times 10^{-5}$
- $\beta_1 = 0.9$, $\beta_2 = 0.95$
- Weight decay: $0.1$

**Batch size**:
- Tokens per batch: 524,288 (0.5M)
- Sequence length: 1024
- Effective batch: 512 sequences

**Training loop**:

```python
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

model = BaseLLM(vocab_size=50000, d_model=2048, n_heads=16, d_ff=8192, n_layers=18, max_seq_len=1024)
optimizer = AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1)
scheduler = CosineAnnealingLR(optimizer, T_max=200000, eta_min=6e-5)

for step, batch in enumerate(dataloader):
    input_ids = batch['input_ids']  # (B, T)
    target_ids = batch['target_ids']  # (B, T), shifted by 1

    # Forward
    logits = model(input_ids)
    loss = F.cross_entropy(
        logits.view(-1, model.vocab_size),
        target_ids.view(-1),
        ignore_index=-100
    )

    # Backward
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    if step % 1000 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")

# Save checkpoint
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'step': step,
}, 'checkpoints/base.pt')
```

### Validation

**Perplexity** on held-out set:

$$
\text{PPL} = \exp\left(-\frac{1}{T} \sum_{t=1}^{T} \log P(x_t \mid x_{<t})\right)
$$

Target: PPL < 20 for good quality.

---

## Phase 2: Q₁ Head Training (Coherence Detection)

### Objective

Train a **binary classifier** $Q_1: \mathbb{R}^{d_{\text{model}}} \to [0, 1]$ to predict:

$$
Q_1(\mathbf{h}_t) \approx P(\text{next token is coherent} \mid x_{<t})
$$

### Data Generation

Create **positive and negative examples**:

**Positive examples** ($y = 1$): Natural text
- Sample from training corpus
- $\mathbf{h}_t = \text{LLM}(x_{1:t-1})$
- Label: $y = 1$ (coherent)

**Negative examples** ($y = 0$): Perturbed text

**Method 1: Random token replacement**
```python
def create_negative_random(tokens, vocab_size, replace_prob=0.3):
    """Replace some tokens with random ones."""
    mask = torch.rand(tokens.shape) < replace_prob
    random_tokens = torch.randint(0, vocab_size, tokens.shape)
    corrupted = torch.where(mask, random_tokens, tokens)
    return corrupted
```

**Method 2: Adversarial tokens**
```python
@torch.no_grad()
def create_negative_adversarial(model, tokens, position):
    """Replace token at position with least-likely token."""
    logits = model(tokens[:position])[-1]  # Logits for position
    worst_token = torch.argmin(logits)
    tokens[position] = worst_token
    return tokens
```

**Method 3: Semantic shift**
```python
def create_negative_semantic(tokens, embedding_model):
    """Replace token with semantically distant token."""
    # Use embedding similarity to find dissimilar replacement
    # (Requires pre-computed embedding matrix)
    pass
```

### Architecture

Add Q₁ head to frozen LLM:

```python
class LLM_with_Q1(nn.Module):
    def __init__(self, base_llm):
        super().__init__()
        self.base_llm = base_llm
        # Freeze base LLM
        for param in self.base_llm.parameters():
            param.requires_grad = False

        # Q₁ head
        self.q1_head = nn.Sequential(
            nn.Linear(base_llm.d_model, base_llm.d_model // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(base_llm.d_model // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_ids, return_hidden=False):
        # Get hidden states from frozen LLM
        with torch.no_grad():
            h = self.base_llm.get_hidden_states(input_ids)  # (B, T, d)

        # Compute Q₁
        q1_scores = self.q1_head(h).squeeze(-1)  # (B, T)

        if return_hidden:
            return q1_scores, h
        return q1_scores
```

### Loss Function

Binary cross-entropy:

$$
\mathcal{L}_{Q_1} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log Q_1(\mathbf{h}_i) + (1 - y_i) \log (1 - Q_1(\mathbf{h}_i)) \right]
$$

### Training Loop

```python
model_q1 = LLM_with_Q1(base_llm=model)  # model from Phase 1
optimizer = AdamW(model_q1.q1_head.parameters(), lr=1e-4)

for step, batch in enumerate(dataloader):
    # Generate positive and negative examples
    pos_tokens = batch['tokens']
    neg_tokens = create_negative_random(pos_tokens, vocab_size=50000)

    # Concatenate
    all_tokens = torch.cat([pos_tokens, neg_tokens], dim=0)
    labels = torch.cat([
        torch.ones(pos_tokens.shape[0]),
        torch.zeros(neg_tokens.shape[0])
    ], dim=0)

    # Forward
    q1_scores = model_q1(all_tokens)  # (2B, T)
    q1_last = q1_scores[:, -1]  # (2B,) - score at last position

    # Loss
    loss = F.binary_cross_entropy(q1_last, labels)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 1000 == 0:
        print(f"Q1 Step {step}, Loss: {loss.item():.4f}")

# Save
torch.save(model_q1.state_dict(), 'checkpoints/base+q1.pt')
```

### Validation

**Accuracy** on held-out positive/negative examples:

$$
\text{Acc} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[Q_1(\mathbf{h}_i) > 0.5 \iff y_i = 1]
$$

Target: Acc > 85%.

**Calibration**: Plot Q₁ scores vs. actual coherence.

---

## Phase 3: Q₂ Head Training (Drift Detection)

### Objective

Train a **drift detector** $Q_2: \mathbb{R}^{d_{\text{model}}} \to [0, 1]$ to predict:

$$
Q_2(\mathbf{h}_t) \approx P(\text{generation is on-topic} \mid x_{<t})
$$

### Data Generation

Create examples of **drift**:

**Method 1: Topic shift in generated text**

```python
@torch.no_grad()
def generate_with_drift(model, prompt, drift_at_step=10, drift_prompt="Meanwhile, in a different topic:"):
    """Generate text, inject topic shift."""
    tokens = prompt.clone()

    for step in range(20):
        logits = model(tokens)
        next_token = torch.argmax(logits[-1])
        tokens = torch.cat([tokens, next_token.unsqueeze(0)])

        if step == drift_at_step:
            # Inject drift
            drift_tokens = tokenizer.encode(drift_prompt)
            tokens = torch.cat([tokens, drift_tokens])

    # Label: 1 before drift, 0 after
    labels = torch.cat([
        torch.ones(drift_at_step),
        torch.zeros(len(tokens) - drift_at_step)
    ])

    return tokens, labels
```

**Method 2: Embedding distance**

```python
def label_drift_by_embedding(hidden_states, threshold=2.0):
    """Label steps where embedding drifts from initial state."""
    h0 = hidden_states[0]  # Initial embedding
    distances = torch.norm(hidden_states - h0, dim=-1)  # (T,)
    labels = (distances < threshold).float()  # 1 if on-topic, 0 if drifted
    return labels
```

**Method 3: Perplexity spike**

$$
\text{drift}_t = \mathbb{1}\left[\text{PPL}(x_{t:t+k} \mid x_{<t}) > \tau_{\text{PPL}}\right]
$$

### Architecture

Add Q₂ head (freeze LLM + Q₁):

```python
class LLM_with_Q1_Q2(nn.Module):
    def __init__(self, model_with_q1):
        super().__init__()
        self.model = model_with_q1
        # Freeze everything except Q₂
        for param in self.model.parameters():
            param.requires_grad = False

        # Q₂ head
        self.q2_head = nn.Sequential(
            nn.Linear(self.model.base_llm.d_model, self.model.base_llm.d_model // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.model.base_llm.d_model // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_ids):
        q1_scores, h = self.model(input_ids, return_hidden=True)
        q2_scores = self.q2_head(h).squeeze(-1)
        return q1_scores, q2_scores
```

### Loss Function

$$
\mathcal{L}_{Q_2} = -\frac{1}{N} \sum_{i=1}^{N} \left[ d_i \log Q_2(\mathbf{h}_i) + (1 - d_i) \log (1 - Q_2(\mathbf{h}_i)) \right]
$$

where $d_i = 1$ if on-topic, $0$ if drifted.

### Training Loop

```python
model_q2 = LLM_with_Q1_Q2(model_q1)
optimizer = AdamW(model_q2.q2_head.parameters(), lr=1e-4)

for step, batch in enumerate(dataloader):
    tokens, drift_labels = generate_with_drift(model, batch['prompt'])

    # Forward
    _, q2_scores = model_q2(tokens)  # (T,)

    # Loss (per-timestep)
    loss = F.binary_cross_entropy(q2_scores, drift_labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 1000 == 0:
        print(f"Q2 Step {step}, Loss: {loss.item():.4f}")

torch.save(model_q2.state_dict(), 'checkpoints/base+q1+q2.pt')
```

### Validation

**Drift detection accuracy**: Measure True Positive Rate (TPR) and False Positive Rate (FPR) on drift events.

**F1 Score**:

$$
F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

Target: F₁ > 0.75.

---

## Phase 4: VARO Integration + Joint Fine-Tuning

### Objective

1. Add **VARO regularization** to penalize variance anti-resonance
2. **Jointly fine-tune** all components (LLM + Q₁ + Q₂ + VARO)
3. Optimize **total Aletheion loss**

### VARO Implementation

```python
def compute_varo_loss(hidden_states, window=10):
    """
    Compute VARO regularization loss.

    Args:
        hidden_states: (batch_size, seq_len, d_model)
        window: lookback window for variance

    Returns:
        varo_loss: scalar
    """
    B, T, D = hidden_states.shape
    var_t = hidden_states.var(dim=-1)  # (B, T)

    varo_scores = []
    for t in range(window, T):
        var_window = var_t[:, t-window:t].mean(dim=1)  # (B,)
        varo_t = torch.abs(var_t[:, t] - var_window) / (var_window + 1e-8)
        varo_scores.append(varo_t)

    varo_scores = torch.stack(varo_scores, dim=1)  # (B, T-window)
    varo_loss = (varo_scores ** 2).mean()
    return varo_loss
```

### Total Loss

$$
\boxed{
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{LM}} + \lambda_1 \mathcal{L}_{Q_1} + \lambda_2 \mathcal{L}_{Q_2} + \lambda_3 \mathcal{L}_{\text{VARO}}
}
$$

### Training Configuration

**Unfreeze all parameters**:
```python
for param in model.parameters():
    param.requires_grad = True
```

**Optimizer**: AdamW with **differential learning rates**:
- Base LLM: $1 \times 10^{-5}$ (small, to preserve pre-training)
- Q₁ head: $5 \times 10^{-5}$
- Q₂ head: $5 \times 10^{-5}$

```python
optimizer = AdamW([
    {'params': model.base_llm.parameters(), 'lr': 1e-5},
    {'params': model.q1_head.parameters(), 'lr': 5e-5},
    {'params': model.q2_head.parameters(), 'lr': 5e-5},
], weight_decay=0.01)
```

**Loss weights** (curriculum):

$$
\lambda_1(t) = 0.05 \cdot \left(1 - e^{-t / 5000}\right)
$$

$$
\lambda_2(t) = 0.02 \cdot \left(1 - e^{-t / 5000}\right)
$$

$$
\lambda_3(t) = 0.005 \cdot \left(1 - e^{-t / 10000}\right)
$$

Start with $\lambda_i \approx 0$, gradually increase.

### Training Loop

```python
from aletheion_llm import AletheionLLM  # Full implementation from aletheion-integration.md

model = AletheionLLM.from_checkpoint('checkpoints/base+q1+q2.pt')
optimizer = AdamW([...])  # Differential LR

lambda_1, lambda_2, lambda_3 = 0.0, 0.0, 0.0  # Start at 0

for step, batch in enumerate(dataloader):
    input_ids = batch['input_ids']
    target_ids = batch['target_ids']
    q1_labels = batch.get('q1_labels', None)
    q2_labels = batch.get('q2_labels', None)

    # Curriculum: gradually increase lambda
    lambda_1 = 0.05 * (1 - np.exp(-step / 5000))
    lambda_2 = 0.02 * (1 - np.exp(-step / 5000))
    lambda_3 = 0.005 * (1 - np.exp(-step / 10000))

    # Forward + loss
    loss, loss_dict = model.compute_loss(
        input_ids, target_ids,
        Q1_labels=q1_labels,
        Q2_labels=q2_labels,
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        lambda_3=lambda_3,
    )

    # Backward
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    if step % 1000 == 0:
        print(f"Step {step}, Total: {loss_dict['total']:.4f}, "
              f"LM: {loss_dict['lm']:.4f}, Q1: {loss_dict['q1']:.4f}, "
              f"Q2: {loss_dict['q2']:.4f}, VARO: {loss_dict['varo']:.4f}")

# Save final model
torch.save(model.state_dict(), 'checkpoints/aletheion.pt')
```

### Validation

**Multi-metric evaluation**:
1. **Perplexity**: Should remain close to base LLM (not degrade)
2. **Q₁ accuracy**: Coherence detection on test set
3. **Q₂ F1**: Drift detection performance
4. **VARO scores**: Average VARO on stable vs. unstable generations
5. **Gating rate**: Fraction of generations that pass all gates

---

## Joint Fine-Tuning: Advanced Techniques

### 1. Adversarial Training for Q Heads

Generate **hard negatives** by fooling current Q₁/Q₂:

```python
def generate_adversarial_negatives(model, tokens, q_head='q1'):
    """Generate tokens that fool Q₁ or Q₂."""
    tokens.requires_grad = True

    for _ in range(10):  # Iterative optimization
        q1, q2 = model(tokens)
        target_q = q1 if q_head == 'q1' else q2

        # Maximize Q score on corrupted example (adversarial)
        loss = -target_q.mean()
        loss.backward()

        # Perturb tokens in embedding space
        with torch.no_grad():
            tokens.grad = tokens.grad / (tokens.grad.norm() + 1e-8)
            tokens = tokens + 0.1 * tokens.grad
            tokens.grad.zero_()

    return tokens.detach()
```

### 2. Online Data Generation

Instead of pre-generating Q labels, **generate during training**:

```python
for step, batch in enumerate(dataloader):
    # Generate negative examples on-the-fly
    pos_tokens = batch['tokens']
    neg_tokens = create_negative_random(pos_tokens)

    # Train Q₁
    q1_scores = model.q1_head(model.base_llm.get_hidden_states(pos_tokens))
    # ... (as before)
```

### 3. Multi-Task Learning

Train on **auxiliary tasks** to improve representations:
- **Entailment**: Predict if sentence B follows from A
- **Similarity**: Predict semantic similarity scores
- **Factuality**: Predict if statement is factually correct (requires external KB)

---

## Hyperparameter Guide

### Critical Hyperparameters

| Hyperparameter | Symbol | Recommended Range | Default | Notes |
|----------------|--------|-------------------|---------|-------|
| **Q₁ threshold** | $Q_1^{\min}$ | 0.5 - 0.8 | 0.7 | Higher = stricter coherence |
| **Q₂ threshold** | $Q_2^{\min}$ | 0.4 - 0.7 | 0.6 | Higher = stricter drift control |
| **VARO threshold** | $\text{VARO}^{\max}$ | 0.3 - 0.6 | 0.5 | Lower = stricter stability |
| **Q₁ loss weight** | $\lambda_1$ | 0.01 - 0.1 | 0.05 | Balance with LM loss |
| **Q₂ loss weight** | $\lambda_2$ | 0.005 - 0.05 | 0.02 | Drift is rarer than incoherence |
| **VARO loss weight** | $\lambda_3$ | 0.001 - 0.01 | 0.005 | Regularization strength |

### Threshold Selection Strategy

**Validation-based tuning**:

1. Train model with default thresholds
2. Evaluate on validation set with **varying thresholds**
3. Plot **precision-recall curves** for Q₁ and Q₂
4. Choose thresholds that maximize **F₁ score**

```python
def find_optimal_threshold(model, val_dataloader):
    """Find optimal Q₁, Q₂, VARO thresholds."""
    q1_scores, q2_scores, varo_scores = [], [], []
    labels_q1, labels_q2 = [], []

    for batch in val_dataloader:
        q1, q2, varo = model(batch['tokens'])
        q1_scores.append(q1.cpu())
        q2_scores.append(q2.cpu())
        varo_scores.append(varo.cpu())
        labels_q1.append(batch['q1_label'].cpu())
        labels_q2.append(batch['q2_label'].cpu())

    q1_scores = torch.cat(q1_scores)
    labels_q1 = torch.cat(labels_q1)

    # Sweep thresholds
    best_f1, best_threshold = 0.0, 0.5
    for threshold in np.linspace(0.5, 0.9, 40):
        preds = (q1_scores > threshold).float()
        f1 = f1_score(labels_q1, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"Optimal Q₁ threshold: {best_threshold:.3f} (F1: {best_f1:.3f})")
    return best_threshold
```

### Loss Weight Scheduling

**Exponential warmup**:

$$
\lambda_i(t) = \lambda_i^{\max} \cdot \left(1 - e^{-t / \tau_i}\right)
$$

**Linear warmup**:

$$
\lambda_i(t) = \lambda_i^{\max} \cdot \min\left(1, \frac{t}{T_{\text{warmup}}}\right)
$$

**Adaptive weighting** (based on loss magnitudes):

$$
\lambda_i(t) = \frac{\mathcal{L}_{\text{LM}}(t)}{\mathcal{L}_{Q_i}(t) + \epsilon}
$$

Ensures all loss components have similar magnitude.

---

## Evaluation Metrics

### 1. Base LLM Quality

**Perplexity**:

$$
\text{PPL} = \exp\left(-\frac{1}{T} \sum_{t=1}^{T} \log P(x_t \mid x_{<t})\right)
$$

Target: Within 5% of baseline LLM (should not degrade).

### 2. Coherence Detection (Q₁)

**Accuracy**: $\frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$

**Precision**: $\frac{\text{TP}}{\text{TP} + \text{FP}}$ (fraction of flagged samples actually incoherent)

**Recall**: $\frac{\text{TP}}{\text{TP} + \text{FN}}$ (fraction of incoherent samples caught)

**F₁ Score**: $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$

Target: F₁ > 0.80

### 3. Drift Detection (Q₂)

Same metrics as Q₁, but on **drift events**.

Target: F₁ > 0.75

### 4. VARO Effectiveness

**Stability score**: Fraction of generations with VARO < threshold.

$$
\text{Stability} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\text{VARO}(x^{(i)}) \leq \text{VARO}^{\max}]
$$

Target: Stability > 0.90

### 5. Gating Rate

**Pass rate**: Fraction of tokens that pass all gates.

$$
\text{Pass rate} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[Q_1(x_i) \geq Q_1^{\min} \land Q_2(x_i) \geq Q_2^{\min} \land \text{VARO}(x_i) \leq \text{VARO}^{\max}]
$$

- **Too high** (>95%): Gates may be too lenient
- **Too low** (<70%): Gates may be too strict, blocking valid generation

Target: 75% - 90%

### 6. Downstream Task Performance

Evaluate on **real-world tasks**:
- **Question Answering**: Accuracy with abstention option
- **Summarization**: ROUGE scores + hallucination rate
- **Code Generation**: Pass@k with syntax/logic checks

**Key metric**: **Accuracy under abstention**

$$
\text{Acc}_{\text{abstain}} = \frac{\text{Correct}}{\text{Correct} + \text{Incorrect}}
$$

(Exclude abstentions from denominator)

Target: Higher accuracy than base LLM, even if coverage is lower.

---

## References

### Training Methodologies

1. **Scaling Laws for Neural Language Models** (Kaplan et al., 2020)
   Compute-optimal training strategies.

2. **Curriculum Learning** (Bengio et al., 2009)
   Phased training approaches.

3. **Multi-Task Learning in Deep Neural Networks** (Ruder, 2017)
   Joint optimization techniques.

### Uncertainty Quantification

4. **Uncertainty Estimation in Deep Learning** (Gal, 2016)
   Dropout-based uncertainty.

5. **Calibrating Neural Networks** (Guo et al., 2017)
   Temperature scaling, Platt scaling.

### Adversarial Training

6. **Explaining and Harnessing Adversarial Examples** (Goodfellow et al., 2015)
   FGSM and adversarial robustness.

### Cross-References

- [← Aletheion Integration](./aletheion-integration.md) - Architecture details
- [Remaining Limitations →](./remaining-limitations.md) - What training doesn't solve

---

**Next**: [Remaining Limitations →](./remaining-limitations.md)
