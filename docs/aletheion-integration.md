# Aletheion Integration: Where and How

## Table of Contents
1. [Overview](#overview)
2. [Critical Integration Point: The Softmax Layer](#critical-integration-point)
3. [Three Integration Levels](#three-integration-levels)
4. [Modified Loss Function](#modified-loss-function)
5. [PyTorch Implementation](#pytorch-implementation)
6. [Before/After Comparison](#before-after-comparison)
7. [References](#references)

---

## Overview

The **Aletheion framework** augments standard autoregressive LLMs with **epistemic gating** to detect and prevent unreliable generation. This document explains:

1. **WHERE** in the LLM architecture Aletheion integrates
2. **HOW** the integration modifies forward pass and training
3. **WHY** this specific integration point is optimal

**Key insight**: Aletheion operates at the **softmax layer** — the exact point where implicit embeddings become explicit token probabilities.

---

## Critical Integration Point: The Softmax Layer

### Standard LLM Generation

Recall from [llm-fundamentals.md](./llm-fundamentals.md) that token generation proceeds as:

$$
\begin{align}
\text{Context: } & x_{<t} = (x_1, x_2, \ldots, x_{t-1}) \\
\text{Embedding: } & \mathbf{h}_t^{(L)} = \text{Transformer}(x_{<t}) \in \mathbb{R}^{d_{\text{model}}} \\
\text{Logits: } & \mathbf{z}_t = W_{\text{out}} \mathbf{h}_t^{(L)} + \mathbf{b}_{\text{out}} \in \mathbb{R}^V \\
\text{Probabilities: } & P(x_t \mid x_{<t}) = \text{softmax}(\mathbf{z}_t) \in \Delta^V \\
\text{Sample: } & x_t \sim P(\cdot \mid x_{<t})
\end{align}
$$

### Aletheion Integration Point

**Aletheion injects epistemic quality scores** computed from hidden states:

$$
\begin{align}
Q_1(x_{<t}) &= \sigma\left( \mathbf{w}_1^T \mathbf{h}_t^{(L)} + b_1 \right) \quad &\text{(Coherence)} \\
Q_2(x_{<t}) &= \sigma\left( \mathbf{w}_2^T \mathbf{h}_t^{(L)} + b_2 \right) \quad &\text{(Drift)} \\
\text{VARO}(x_{<t}) &= f_{\text{VARO}}(\mathbf{h}_t^{(L)}, \mathbf{h}_{<t}^{(L)}) \quad &\text{(Anti-resonance)}
\end{align}
$$

where:
- $Q_1, Q_2 \in [0, 1]$ are **scalar quality scores**
- $\sigma(\cdot)$ is the sigmoid function
- VARO analyzes **variance anti-resonance** in hidden states

**Critical decision**: Apply gating **before** sampling:

$$
\boxed{
\text{IF } Q_1(x_{<t}) < Q_1^{\min} \text{ OR } Q_2(x_{<t}) < Q_2^{\min} \text{ OR } \text{VARO}(x_{<t}) > \text{VARO}^{\max}
}
$$

$$
\boxed{
\text{THEN } \text{reject generation, trigger fallback (e.g., abstention, retrieval, symbolic solver)}
}
$$

### Why This Point?

The softmax layer is the **boundary** between:

- **Implicit representations** (embedding space $\mathbb{R}^{d_{\text{model}}}$)
- **Explicit predictions** (probability distribution over tokens)

```
     Embedding Space              Softmax Boundary        Token Space
   (Implicit/Neural)          (ALETHEION INTEGRATION)    (Explicit/Symbolic)
          ↓                              ↓                      ↓
   h_t ∈ ℝ^d_model  ────────→   z_t = W·h_t  ────────→   P(x_t) = softmax(z_t)
                                      ↑
                                      │
                          ┌───────────┴───────────┐
                          │   Q₁(h_t), Q₂(h_t)    │
                          │   VARO(h_t, h_{<t})   │
                          │                       │
                          │   IF Q₁ < Q₁_min OR   │
                          │      Q₂ < Q₂_min OR   │
                          │      VARO > VARO_max  │
                          │   THEN                │
                          │      GATE / ABSTAIN   │
                          └───────────────────────┘
```

**Advantages**:
1. **Access to rich representations**: $\mathbf{h}_t^{(L)}$ contains full context information
2. **Pre-commitment**: Detect issues **before** sampling irreversible token
3. **Minimal disruption**: Can be added to pre-trained models
4. **Interpretable**: Q scores are scalar, human-interpretable

---

## Three Integration Levels

Aletheion can be integrated at **three levels** of increasing depth.

### Implementation Status Summary

| Level | Status | Location | Description |
|-------|--------|----------|-------------|
| **Level 1** | ✅ **Fully Implemented** | `src/aletheion/model.py` | Output-only gates (production-ready) |
| **Level 2** | ⏳ **Partial** | `src/aletheion/pyramidal_*.py` | Attention-level gates (pyramidal variants) |
| **Level 3** | 🔜 **Planned** | Future work | Deep VARO integration in hidden states |

### Level 1: Gating at Output (Post-Softmax) ✅ IMPLEMENTED

**Architecture**: Compute Q scores after softmax, gate the sampled token.

$$
\begin{align}
\mathbf{z}_t &= W_{\text{out}} \mathbf{h}_t^{(L)} \\
P(x_t) &= \text{softmax}(\mathbf{z}_t) \\
\tilde{x}_t &\sim P(x_t) \quad \text{(sample)} \\
Q_1, Q_2 &= f_Q(\mathbf{h}_t^{(L)}) \\
x_t &= \begin{cases}
\tilde{x}_t & \text{if } Q_1 \geq Q_1^{\min}, Q_2 \geq Q_2^{\min} \\
\texttt{<UNK>} & \text{otherwise (abstain)}
\end{cases}
\end{align}
$$

**Pros**:
- Simple to implement
- Works with **any pre-trained model** (just add Q heads)
- No retraining of base LLM required

**Cons**:
- Token already sampled (wasteful computation)
- Cannot modify probabilities, only accept/reject
- Less precise control

**Use case**: Quick integration with existing models.

### Level 2: Gating at Logits (Pre-Softmax) ⏳ PARTIAL IMPLEMENTATION

**Architecture**: Compute Q scores, then **modify logits** before softmax.

$$
\begin{align}
\mathbf{z}_t &= W_{\text{out}} \mathbf{h}_t^{(L)} \\
Q_1, Q_2, \text{VARO} &= f_Q(\mathbf{h}_t^{(L)}, \mathbf{h}_{<t}^{(L)}) \\
\text{gate} &= \mathbb{1}[Q_1 \geq Q_1^{\min}] \cdot \mathbb{1}[Q_2 \geq Q_2^{\min}] \cdot \mathbb{1}[\text{VARO} \leq \text{VARO}^{\max}] \\
\tilde{\mathbf{z}}_t &= \begin{cases}
\mathbf{z}_t & \text{if gate} = 1 \\
\mathbf{z}_{\texttt{<UNK>}} & \text{if gate} = 0 \quad \text{(force abstention)}
\end{cases} \\
P(x_t) &= \text{softmax}(\tilde{\mathbf{z}}_t)
\end{align}
$$

**Variant**: Soft gating with **adaptive temperature**:

$$
\tilde{\mathbf{z}}_t = \mathbf{z}_t \cdot \left( \alpha \cdot Q_1 \cdot Q_2 \cdot (1 - \beta \cdot \text{VARO}) \right)
$$

where $\alpha, \beta$ are hyperparameters.

**Pros**:
- **Fine-grained control**: Can modify token probabilities
- **Efficient**: Single forward pass
- **Differentiable**: Can train Q heads end-to-end with LLM
- **Calibrated**: Soft gating adjusts confidence rather than hard reject

**Cons**:
- Requires joint training (or fine-tuning)
- Slightly more complex implementation

**Use case**: Production deployment with quality-controlled generation.

### Level 3: Deep Integration (VARO in Hidden States) 🔜 PLANNED

**Architecture**: Inject VARO regularization **inside transformer layers**.

$$
\begin{align}
\mathbf{a}^{(\ell)} &= \mathbf{H}^{(\ell-1)} + \text{MultiHead}(\mathbf{H}^{(\ell-1)}) \\
\text{VARO}^{(\ell)} &= \text{variance-anti-resonance}(\mathbf{a}^{(\ell)}, \mathbf{a}^{(<\ell)}) \\
\lambda_{\text{reg}}^{(\ell)} &= \gamma \cdot \text{VARO}^{(\ell)} \quad \text{(adaptive regularization)} \\
\mathbf{h}^{(\ell)} &= \mathbf{a}^{(\ell)} + \text{FFN}(\mathbf{a}^{(\ell)}) \cdot (1 - \lambda_{\text{reg}}^{(\ell)})
\end{align}
$$

**Effect**: Suppress layer activations when VARO detects instability.

**Pros**:
- **Preventive**: Stops bad generations before final layer
- **Architectural**: Fundamentally changes how model processes information
- **Potentially more powerful**: Early intervention

**Cons**:
- Requires **training from scratch** or extensive fine-tuning
- Complex to implement and debug
- May interfere with pre-trained representations

**Use case**: Research, next-generation Aletheion-native models.

---

## Modified Loss Function

### Standard LLM Loss

$$
\mathcal{L}_{\text{LLM}} = -\frac{1}{T} \sum_{t=1}^{T} \log P(x_t \mid x_{<t}; \theta)
$$

### Aletheion Multi-Objective Loss

$$
\boxed{
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{LLM}} + \lambda_1 \cdot \mathcal{L}_{Q_1} + \lambda_2 \cdot \mathcal{L}_{Q_2} + \lambda_3 \cdot \mathcal{L}_{\text{VARO}}
}
$$

### Component Losses

#### 1. Q₁ Coherence Loss

**Goal**: Train Q₁ to predict **local coherence** (is next token plausible given context?).

**Training signal**: Use **negative sampling** or **perturbed examples**.

$$
\mathcal{L}_{Q_1} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log Q_1(x_{<t}^{(i)}) + (1 - y_i) \log (1 - Q_1(x_{<t}^{(i)})) \right]
$$

where:
- $y_i = 1$ if $x_t^{(i)}$ is the **true next token** (coherent)
- $y_i = 0$ if $x_t^{(i)}$ is a **random/adversarial token** (incoherent)

**Practical implementation**:

```python
def coherence_loss(Q1_scores, labels):
    """
    Binary cross-entropy for coherence prediction.

    Args:
        Q1_scores: (batch_size,) - Q₁ predictions in [0, 1]
        labels: (batch_size,) - 1 if coherent, 0 if not

    Returns:
        loss: scalar
    """
    return F.binary_cross_entropy(Q1_scores, labels)
```

**Data augmentation** for negative examples:
- Random token replacement: $x_t \leftarrow x_{\text{random}} \sim \text{Uniform}(\mathcal{V})$
- Adversarial perturbation: $x_t \leftarrow \arg\min_{v} P(v \mid x_{<t})$ (least likely token)
- Context shuffle: $x_{<t} \leftarrow \text{shuffle}(x_{<t})$

#### 2. Q₂ Drift Loss

**Goal**: Train Q₂ to detect **semantic drift** (is generation straying from original intent?).

**Training signal**: Use **multi-step generation** and label drift points.

$$
\mathcal{L}_{Q_2} = -\frac{1}{N} \sum_{i=1}^{N} \left[ d_i \log Q_2(x_{<t}^{(i)}) + (1 - d_i) \log (1 - Q_2(x_{<t}^{(i)})) \right]
$$

where:
- $d_i = 1$ if generation at step $t$ is **on-topic**
- $d_i = 0$ if generation has **drifted** from original query

**Measuring drift** (heuristics):
1. **Embedding distance**: $\|\mathbf{h}_t - \mathbf{h}_0\| > \tau_{\text{drift}}$
2. **Perplexity increase**: $\text{PPL}(x_{t:t+k} \mid x_{<t}) > \tau_{\text{PPL}}$
3. **Human annotation**: Label drift points in long-form generation

#### 3. VARO Anti-Resonance Loss

**Goal**: Penalize **variance anti-resonance** in hidden state trajectories.

**Definition**: VARO occurs when variance oscillates instead of stabilizing:

$$
\text{VARO}_t = \left| \text{Var}(\mathbf{h}_t^{(L)}) - \text{Var}(\mathbf{h}_{t-k:t-1}^{(L)}) \right|
$$

**Loss**: Penalize high VARO:

$$
\mathcal{L}_{\text{VARO}} = \frac{1}{T} \sum_{t=k}^{T} \max(0, \text{VARO}_t - \tau_{\text{VARO}})^2
$$

Encourage **smooth variance dynamics**:

$$
\text{Var}(\mathbf{h}_t) \approx \text{Var}(\mathbf{h}_{t-1}) \quad \text{(stability)}
$$

### Hyperparameter Tuning

**Loss weights** $\lambda_1, \lambda_2, \lambda_3$:

| Component | Weight | Typical Range | Priority |
|-----------|--------|---------------|----------|
| $\mathcal{L}_{\text{LLM}}$ | 1.0 (baseline) | - | High (primary task) |
| $\mathcal{L}_{Q_1}$ | $\lambda_1$ | 0.01 - 0.1 | Medium (local coherence) |
| $\mathcal{L}_{Q_2}$ | $\lambda_2$ | 0.005 - 0.05 | Medium (drift detection) |
| $\mathcal{L}_{\text{VARO}}$ | $\lambda_3$ | 0.001 - 0.01 | Low (regularization) |

**Adaptive weighting** (curriculum learning):

$$
\lambda_i(t) = \lambda_i^{\max} \cdot \left(1 - e^{-t / \tau_i}\right)
$$

Start with $\lambda_i \approx 0$ (pure LLM training), gradually increase.

---

## PyTorch Implementation

### Complete AletheionLLM Class

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class AletheionLLM(nn.Module):
    """
    Aletheion-augmented LLM with epistemic gating.

    Architecture:
        - Base transformer LLM
        - Q₁ head (coherence)
        - Q₂ head (drift)
        - VARO computation
        - Gating at logits (Level 2 integration)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        max_seq_len: int,
        Q1_min: float = 0.7,
        Q2_min: float = 0.6,
        VARO_max: float = 0.5,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.Q1_min = Q1_min
        self.Q2_min = Q2_min
        self.VARO_max = VARO_max

        # Base LLM (same as TransformerLM from llm-fundamentals.md)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_embedding.weight

        # Aletheion components
        self.Q1_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

        self.Q2_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

        # VARO requires looking at past hidden states
        self.varo_window = 10  # Look back 10 steps

    def forward(
        self,
        input_ids: torch.Tensor,
        return_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with Aletheion gating.

        Args:
            input_ids: (batch_size, seq_len)
            return_hidden_states: if True, return h^(L)

        Returns:
            logits: (batch_size, seq_len, vocab_size)
            Q1_scores: (batch_size, seq_len)
            Q2_scores: (batch_size, seq_len)
            VARO_scores: (batch_size, seq_len)
            hidden_states: (batch_size, seq_len, d_model) [optional]
        """
        batch_size, seq_len = input_ids.shape

        # Causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len)
        causal_mask = causal_mask.to(input_ids.device)

        # Embeddings
        token_emb = self.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)
        x = token_emb + pos_emb

        # Transformer
        h = self.transformer(x, mask=causal_mask, is_causal=True)  # (B, T, d)
        h = self.ln_f(h)

        # Logits (before gating)
        logits = self.lm_head(h)  # (B, T, V)

        # Compute Q scores
        Q1_scores = self.Q1_head(h).squeeze(-1)  # (B, T)
        Q2_scores = self.Q2_head(h).squeeze(-1)  # (B, T)

        # Compute VARO
        VARO_scores = self.compute_varo(h)  # (B, T)

        # Apply gating (Level 2: modify logits)
        gated_logits = self.apply_gating(logits, Q1_scores, Q2_scores, VARO_scores)

        outputs = (gated_logits, Q1_scores, Q2_scores, VARO_scores)
        if return_hidden_states:
            outputs = outputs + (h,)

        return outputs

    def compute_varo(self, h: torch.Tensor) -> torch.Tensor:
        """
        Compute variance anti-resonance score.

        Args:
            h: (batch_size, seq_len, d_model) - hidden states

        Returns:
            varo: (batch_size, seq_len) - VARO scores in [0, 1]
        """
        batch_size, seq_len, d_model = h.shape

        # Compute variance at each timestep
        var_t = h.var(dim=-1)  # (B, T)

        # Compute moving average of variance
        varo_scores = torch.zeros_like(var_t)
        for t in range(self.varo_window, seq_len):
            var_window = var_t[:, t - self.varo_window:t].mean(dim=1)  # (B,)
            varo_scores[:, t] = torch.abs(var_t[:, t] - var_window) / (var_window + 1e-8)

        # Normalize to [0, 1]
        varo_scores = torch.sigmoid(varo_scores - 1.0)  # Shift so 0 → 0.27, 1 → 0.73

        return varo_scores

    def apply_gating(
        self,
        logits: torch.Tensor,
        Q1: torch.Tensor,
        Q2: torch.Tensor,
        VARO: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply epistemic gating to logits.

        Args:
            logits: (batch_size, seq_len, vocab_size)
            Q1, Q2, VARO: (batch_size, seq_len) - quality scores

        Returns:
            gated_logits: (batch_size, seq_len, vocab_size)
        """
        # Compute gating mask
        gate = (
            (Q1 >= self.Q1_min) &
            (Q2 >= self.Q2_min) &
            (VARO <= self.VARO_max)
        ).float()  # (B, T)

        # Soft gating: scale logits by quality
        quality = Q1 * Q2 * (1 - VARO)  # (B, T)
        quality = quality.unsqueeze(-1)  # (B, T, 1)

        # Option 1: Multiplicative gating
        gated_logits = logits * quality

        # Option 2: Additive gating (penalize low quality)
        # gated_logits = logits + (quality - 0.5) * 10.0

        # Option 3: Hard gating (force <UNK> if gate == 0)
        # UNK_id = 0  # Assume <UNK> is token 0
        # mask = gate.unsqueeze(-1)  # (B, T, 1)
        # gated_logits = logits * mask
        # gated_logits[:, :, UNK_id] += (1 - gate) * 100.0  # Force <UNK>

        return gated_logits

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        Q1_labels: Optional[torch.Tensor] = None,
        Q2_labels: Optional[torch.Tensor] = None,
        lambda_1: float = 0.05,
        lambda_2: float = 0.02,
        lambda_3: float = 0.005,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute total Aletheion loss.

        Args:
            input_ids: (batch_size, seq_len)
            target_ids: (batch_size, seq_len)
            Q1_labels: (batch_size, seq_len) - coherence labels (0 or 1)
            Q2_labels: (batch_size, seq_len) - drift labels (0 or 1)
            lambda_1, lambda_2, lambda_3: loss weights

        Returns:
            total_loss: scalar
            loss_dict: dictionary of individual losses
        """
        logits, Q1, Q2, VARO = self.forward(input_ids)

        # LLM loss (cross-entropy)
        logits_flat = logits.view(-1, self.vocab_size)
        targets_flat = target_ids.view(-1)
        loss_lm = F.cross_entropy(logits_flat, targets_flat, ignore_index=-100)

        # Q1 loss (if labels provided)
        if Q1_labels is not None:
            loss_q1 = F.binary_cross_entropy(Q1, Q1_labels.float())
        else:
            loss_q1 = torch.tensor(0.0, device=logits.device)

        # Q2 loss (if labels provided)
        if Q2_labels is not None:
            loss_q2 = F.binary_cross_entropy(Q2, Q2_labels.float())
        else:
            loss_q2 = torch.tensor(0.0, device=logits.device)

        # VARO loss (regularization)
        loss_varo = (VARO ** 2).mean()  # Penalize high VARO

        # Total loss
        total_loss = (
            loss_lm +
            lambda_1 * loss_q1 +
            lambda_2 * loss_q2 +
            lambda_3 * loss_varo
        )

        loss_dict = {
            'total': total_loss.item(),
            'lm': loss_lm.item(),
            'q1': loss_q1.item(),
            'q2': loss_q2.item(),
            'varo': loss_varo.item(),
        }

        return total_loss, loss_dict

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        abstain_token_id: int = 0,  # <UNK>
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Autoregressive generation with Aletheion gating.

        Args:
            prompt_ids: (batch_size, prompt_len)
            max_new_tokens: max tokens to generate
            temperature: sampling temperature
            top_k: if set, sample from top-k
            abstain_token_id: token to emit when gating fails

        Returns:
            generated_ids: (batch_size, prompt_len + generated_len)
            gate_history: (batch_size, generated_len) - 1 if passed gate, 0 if failed
        """
        self.eval()
        input_ids = prompt_ids.clone()
        batch_size = input_ids.shape[0]
        gate_history = []

        for _ in range(max_new_tokens):
            # Forward pass
            logits, Q1, Q2, VARO = self.forward(input_ids)
            logits = logits[:, -1, :] / temperature  # (B, V)

            # Check gating
            Q1_last = Q1[:, -1]  # (B,)
            Q2_last = Q2[:, -1]  # (B,)
            VARO_last = VARO[:, -1]  # (B,)

            gate = (
                (Q1_last >= self.Q1_min) &
                (Q2_last >= self.Q2_min) &
                (VARO_last <= self.VARO_max)
            )  # (B,)
            gate_history.append(gate.cpu())

            # Sample
            if top_k is not None:
                top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                logits[logits < top_k_logits[:, -1:]] = -float('inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Apply gating: if gate fails, emit abstain token
            next_token = torch.where(
                gate.unsqueeze(-1),
                next_token,
                torch.full_like(next_token, abstain_token_id)
            )

            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)

        gate_history = torch.stack(gate_history, dim=1)  # (B, T)
        return input_ids, gate_history
```

### Usage Example

```python
# Initialize model
model = AletheionLLM(
    vocab_size=50000,
    d_model=768,
    n_heads=12,
    d_ff=3072,
    n_layers=12,
    max_seq_len=1024,
    Q1_min=0.7,
    Q2_min=0.6,
    VARO_max=0.5,
)

# Training
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for batch in dataloader:
    input_ids = batch['input_ids']
    target_ids = batch['target_ids']
    Q1_labels = batch.get('Q1_labels', None)  # Optional
    Q2_labels = batch.get('Q2_labels', None)  # Optional

    loss, loss_dict = model.compute_loss(
        input_ids, target_ids, Q1_labels, Q2_labels,
        lambda_1=0.05, lambda_2=0.02, lambda_3=0.005
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Loss: {loss_dict}")

# Generation
prompt = torch.tensor([[1, 2, 3, 4]])  # Example token IDs
generated, gates = model.generate(
    prompt,
    max_new_tokens=50,
    temperature=0.8,
    top_k=40
)

print("Generated:", generated)
print("Gate pass rate:", gates.float().mean().item())
```

---

## Before/After Comparison

### Standard LLM (Before Aletheion)

```
Input: "What is the capital of Australia?"

Forward Pass:
  h_t = Transformer(input)           → ℝ^768
  z_t = W_out · h_t                  → ℝ^50000
  P(x_t) = softmax(z_t)              → Δ^50000

Sample:
  x_t ~ P(x_t)
  → "Sydney" (WRONG, but P("Sydney") = 0.73)

Output: "Sydney"
Confidence: 0.73
Correctness: ✗
```

**Problems**:
- No coherence check
- No drift detection
- Overconfident hallucination

### Aletheion LLM (After Integration)

```
Input: "What is the capital of Australia?"

Forward Pass:
  h_t = Transformer(input)           → ℝ^768
  z_t = W_out · h_t                  → ℝ^50000

  [ALETHEION GATING]
  Q₁(h_t) = 0.45  ← LOW! (embedding inconsistency detected)
  Q₂(h_t) = 0.82  ← OK
  VARO(h_t, h_{<t}) = 0.31  ← OK

  Gate = (Q₁ >= 0.7) ∧ (Q₂ >= 0.6) ∧ (VARO <= 0.5)
       = (0.45 >= 0.7) ∧ (0.82 >= 0.6) ∧ (0.31 <= 0.5)
       = False ∧ True ∧ True
       = False  ← REJECT!

  Fallback: Emit <UNK> or trigger retrieval

Output: "<UNK> [Low confidence: Q₁=0.45. Consider web search.]"
Confidence: Abstained
Correctness: ✓ (avoided hallucination!)
```

**Improvements**:
- Detected embedding space inconsistency (Q₁)
- Prevented hallucination
- Can trigger retrieval or human oversight

### Side-by-Side Diagram

```
┌───────────────────────────────────────────────────────────────────────────┐
│                         STANDARD LLM                                       │
├───────────────────────────────────────────────────────────────────────────┤
│  Input → Embedding → Transformer → Logits → Softmax → Sample → Output    │
│                                                  ↓                         │
│                                            P("Sydney") = 0.73              │
│                                                  ↓                         │
│                                            Output: "Sydney" ✗              │
└───────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│                        ALETHEION LLM                                       │
├───────────────────────────────────────────────────────────────────────────┤
│  Input → Embedding → Transformer → h^(L) ──┬→ Logits → [GATED] → Output  │
│                                             │                              │
│                                             ├→ Q₁ = 0.45 ← FAIL!          │
│                                             ├→ Q₂ = 0.82 ← PASS           │
│                                             └→ VARO = 0.31 ← PASS         │
│                                                     ↓                      │
│                                         Gate = Q₁ ∧ Q₂ ∧ VARO = False     │
│                                                     ↓                      │
│                                         Output: <UNK> or Retrieval ✓      │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## References

### Aletheion Framework

1. **Q-Systems Repository**: https://github.com/AletheiQAGI/q-systems
   Formal definitions of Q₁, Q₂, VARO.

2. **Variance Anti-Resonance in Neural Networks** (Hypothetical)
   Theoretical foundation for VARO.

3. **Epistemic Uncertainty in Deep Learning** (Gal & Ghahramani, 2016)
   Related work on uncertainty quantification.

### Gating Mechanisms

4. **Mixture of Experts** (Shazeer et al., 2017)
   Gating in neural architectures.

5. **Adaptive Computation Time** (Graves, 2016)
   Dynamic computation based on confidence.

### Cross-References

- [← LLM Failures](./llm-failures.md) - Why gating is necessary
- [Training Strategy →](./training-strategy.md) - How to train Aletheion LLMs
- [Remaining Limitations →](./remaining-limitations.md) - What Aletheion doesn't solve

---

**Next**: [Training Strategy →](./training-strategy.md)
