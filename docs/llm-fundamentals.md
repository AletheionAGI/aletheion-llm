# LLM Fundamentals: Mathematical Foundations

## Table of Contents
1. [Formal Definition](#formal-definition)
2. [Transformer Architecture](#transformer-architecture)
3. [Training Objective](#training-objective)
4. [Inference Process](#inference-process)
5. [Parameter Count Analysis](#parameter-count-analysis)
6. [References](#references)

---

## 1. Formal Definition

A **Large Language Model (LLM)** is a neural network that models the conditional probability distribution over sequences of tokens.

### 1.1 Core Probability Model

Given a vocabulary $\mathcal{V}$ of size $|\mathcal{V}| = V$ and a sequence of tokens $\mathbf{x} = (x_1, x_2, \ldots, x_T)$ where $x_t \in \mathcal{V}$, an LLM models:

$$
P(\mathbf{x}; \theta) = \prod_{t=1}^{T} P(x_t \mid x_{<t}; \theta)
$$

where:
- $x_{<t} = (x_1, x_2, \ldots, x_{t-1})$ is the **context** (all previous tokens)
- $\theta$ represents all **learnable parameters** of the neural network
- Each conditional $P(x_t \mid x_{<t}; \theta)$ is a categorical distribution over $\mathcal{V}$

### 1.2 Mathematical Components

The conditional probability is computed as:

$$
P(x_t \mid x_{<t}; \theta) = \text{softmax}(\mathbf{z}_t)_{x_t}
$$

where the **logit vector** $\mathbf{z}_t \in \mathbb{R}^V$ is computed by:

$$
\mathbf{z}_t = W_{\text{out}} \cdot \mathbf{h}_t^{(L)} + \mathbf{b}_{\text{out}}
$$

and $\mathbf{h}_t^{(L)} \in \mathbb{R}^{d_{\text{model}}}$ is the final hidden state from the $L$-th transformer layer.

---

## 2. Transformer Architecture

The standard transformer decoder architecture consists of the following components stacked in sequence:

```
Input Tokens (x₁, x₂, ..., xₜ)
        ↓
    Embedding Layer
        ↓
  Positional Encoding
        ↓
   ┌─────────────────┐
   │ Transformer     │
   │ Layer 1         │  ← Multi-head Self-Attention + Feed-Forward
   ├─────────────────┤
   │ Transformer     │
   │ Layer 2         │
   ├─────────────────┤
   │     ...         │
   ├─────────────────┤
   │ Transformer     │
   │ Layer L         │
   └─────────────────┘
        ↓
   Layer Norm
        ↓
  Output Projection (W_out)
        ↓
     Softmax
        ↓
  P(xₜ₊₁ | x≤ₜ)
```

### 2.1 Embedding Layer

**Token Embedding**: Maps discrete tokens to continuous vectors:

$$
\mathbf{e}_t = E[x_t] \in \mathbb{R}^{d_{\text{model}}}
$$

where $E \in \mathbb{R}^{V \times d_{\text{model}}}$ is the embedding matrix.

**Positional Encoding**: Adds position information using sinusoidal functions:

$$
\text{PE}(t, 2i) = \sin\left(\frac{t}{10000^{2i/d_{\text{model}}}}\right)
$$

$$
\text{PE}(t, 2i+1) = \cos\left(\frac{t}{10000^{2i/d_{\text{model}}}}\right)
$$

Combined input to first layer:

$$
\mathbf{h}_t^{(0)} = \mathbf{e}_t + \text{PE}(t)
$$

### 2.2 Multi-Head Self-Attention

Each attention head $h$ computes:

$$
\text{head}_h = \text{Attention}(Q_h, K_h, V_h)
$$

where:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

**Query, Key, Value projections**:

$$
Q_h = \mathbf{H}^{(\ell-1)} W_Q^h, \quad K_h = \mathbf{H}^{(\ell-1)} W_K^h, \quad V_h = \mathbf{H}^{(\ell-1)} W_V^h
$$

where $W_Q^h, W_K^h, W_V^h \in \mathbb{R}^{d_{\text{model}} \times d_k}$ and $d_k = d_{\text{model}} / n_{\text{heads}}$.

**Causal masking**: For autoregressive modeling, attention scores are masked:

$$
\text{Attention}(Q, K, V)_{i,j} =
\begin{cases}
\text{softmax}\left(\frac{q_i \cdot k_j}{\sqrt{d_k}}\right) v_j & \text{if } j \leq i \\
0 & \text{if } j > i
\end{cases}
$$

This ensures token $t$ only attends to tokens $\{1, 2, \ldots, t\}$ (no future information).

**Multi-head output**:

$$
\text{MultiHead}(\mathbf{H}^{(\ell-1)}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_{n_{\text{heads}}}) W_O
$$

where $W_O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$.

### 2.3 Feed-Forward Network (FFN)

Each position is processed independently by a two-layer MLP:

$$
\text{FFN}(\mathbf{h}) = W_2 \cdot \text{ReLU}(W_1 \cdot \mathbf{h} + \mathbf{b}_1) + \mathbf{b}_2
$$

where:
- $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$ (typically $d_{\text{ff}} = 4 \cdot d_{\text{model}}$)
- $W_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$

**Modern variant**: Many recent models use **SwiGLU** activation:

$$
\text{SwiGLU}(\mathbf{h}) = (\text{swish}(W_1 \mathbf{h}) \odot W_3 \mathbf{h}) W_2
$$

where $\text{swish}(x) = x \cdot \sigma(x)$ and $\odot$ is element-wise multiplication.

### 2.4 Complete Transformer Layer

$$
\begin{align}
\mathbf{a}^{(\ell)} &= \text{LayerNorm}(\mathbf{H}^{(\ell-1)} + \text{MultiHead}(\mathbf{H}^{(\ell-1)})) \\
\mathbf{H}^{(\ell)} &= \text{LayerNorm}(\mathbf{a}^{(\ell)} + \text{FFN}(\mathbf{a}^{(\ell)}))
\end{align}
$$

This is the **post-norm** variant. Modern architectures often use **pre-norm**:

$$
\begin{align}
\mathbf{a}^{(\ell)} &= \mathbf{H}^{(\ell-1)} + \text{MultiHead}(\text{LayerNorm}(\mathbf{H}^{(\ell-1)})) \\
\mathbf{H}^{(\ell)} &= \mathbf{a}^{(\ell)} + \text{FFN}(\text{LayerNorm}(\mathbf{a}^{(\ell)}))
\end{align}
$$

---

## 3. Training Objective

### 3.1 Cross-Entropy Loss

Given a training corpus $\mathcal{D} = \{\mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \ldots, \mathbf{x}^{(N)}\}$, we minimize the negative log-likelihood:

$$
\mathcal{L}(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T_i} \log P(x_t^{(i)} \mid x_{<t}^{(i)}; \theta)
$$

This is equivalent to **cross-entropy** between the true distribution (one-hot) and predicted distribution:

$$
\mathcal{L}_t = -\sum_{v \in \mathcal{V}} \mathbb{1}[x_t = v] \log P(v \mid x_{<t}; \theta)
$$

Since $x_t$ is a single token (one-hot), this simplifies to:

$$
\mathcal{L}_t = -\log P(x_t \mid x_{<t}; \theta) = -\log \text{softmax}(\mathbf{z}_t)_{x_t}
$$

### 3.2 Softmax and Numerical Stability

The softmax function is defined as:

$$
\text{softmax}(\mathbf{z})_i = \frac{\exp(z_i)}{\sum_{j=1}^{V} \exp(z_j)}
$$

**Numerical stability trick**:

$$
\text{softmax}(\mathbf{z})_i = \frac{\exp(z_i - z_{\max})}{\sum_{j=1}^{V} \exp(z_j - z_{\max})}
$$

where $z_{\max} = \max_j z_j$.

**Log-softmax** (used in cross-entropy):

$$
\log \text{softmax}(\mathbf{z})_i = z_i - \log \sum_{j=1}^{V} \exp(z_j)
$$

### 3.3 PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_seq_len):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Embedding layer
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output projection
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying (optional but common)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids):
        """
        Args:
            input_ids: (batch_size, seq_len) - token indices

        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape

        # Create causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len)

        # Embeddings
        token_emb = self.token_embedding(input_ids)  # (B, T, d_model)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)  # (1, T, d_model)

        x = token_emb + pos_emb

        # Transformer
        h = self.transformer(x, mask=causal_mask, is_causal=True)

        # Output
        h = self.ln_f(h)
        logits = self.lm_head(h)  # (B, T, V)

        return logits

    def compute_loss(self, input_ids, target_ids):
        """
        Args:
            input_ids: (batch_size, seq_len) - input tokens
            target_ids: (batch_size, seq_len) - target tokens (shifted by 1)

        Returns:
            loss: scalar cross-entropy loss
        """
        logits = self.forward(input_ids)  # (B, T, V)

        # Reshape for cross-entropy
        logits_flat = logits.view(-1, self.vocab_size)  # (B*T, V)
        targets_flat = target_ids.view(-1)  # (B*T,)

        # Cross-entropy loss
        loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=-100)

        return loss
```

### 3.4 Optimization

**Adam optimizer** with learning rate schedule:

$$
\text{lr}(t) = d_{\text{model}}^{-0.5} \cdot \min(t^{-0.5}, t \cdot \text{warmup}^{-1.5})
$$

Common hyperparameters:
- $\beta_1 = 0.9$, $\beta_2 = 0.98$ (Adam betas)
- $\epsilon = 10^{-9}$ (Adam epsilon)
- warmup steps: 4000-10000

---

## 4. Inference Process

### 4.1 Autoregressive Generation

Starting from a prompt $\mathbf{x}_{1:t_0}$, generate new tokens sequentially:

```
For t = t₀ + 1, t₀ + 2, ..., max_length:
    1. Compute logits: z_t = LLM(x₁, ..., x_{t-1})
    2. Apply softmax: P(· | x_{<t}) = softmax(z_t)
    3. Sample: x_t ~ P(· | x_{<t})
    4. Append x_t to sequence
    5. If x_t = <EOS>, break
```

### 4.2 Sampling Strategies

**Greedy decoding** (deterministic):

$$
x_t = \arg\max_{v \in \mathcal{V}} P(v \mid x_{<t}; \theta)
$$

**Temperature sampling** ($T > 0$):

$$
P_T(x_t = v \mid x_{<t}) = \frac{\exp(z_v / T)}{\sum_{v' \in \mathcal{V}} \exp(z_{v'} / T)}
$$

- $T < 1$: **sharper** distribution (more deterministic)
- $T > 1$: **flatter** distribution (more random)
- $T \to 0$: greedy decoding
- $T \to \infty$: uniform sampling

**Top-k sampling**: Sample from top-k highest probability tokens:

$$
x_t \sim P(v \mid x_{<t}), \quad v \in \text{topk}(\mathbf{z}_t, k)
$$

**Top-p (nucleus) sampling**: Sample from smallest set with cumulative probability $\geq p$:

$$
\mathcal{V}_p = \left\{ v : \sum_{v' \in \mathcal{V}_{\geq v}} P(v' \mid x_{<t}) \geq p \right\}
$$

where tokens are ranked by probability.

### 4.3 PyTorch Inference Code

```python
@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens=100, temperature=1.0, top_k=None, top_p=None):
    """
    Generate text autoregressively.

    Args:
        model: TransformerLM instance
        prompt_ids: (batch_size, prompt_len) - initial tokens
        max_new_tokens: maximum number of tokens to generate
        temperature: sampling temperature
        top_k: if set, sample from top-k tokens
        top_p: if set, sample from nucleus (top-p)

    Returns:
        generated_ids: (batch_size, prompt_len + max_new_tokens)
    """
    model.eval()
    input_ids = prompt_ids.clone()

    for _ in range(max_new_tokens):
        # Forward pass (only need logits for last position)
        logits = model(input_ids)  # (B, T, V)
        logits = logits[:, -1, :] / temperature  # (B, V)

        # Apply top-k filtering
        if top_k is not None:
            top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
            logits[logits < top_k_logits[:, -1:]] = -float('inf')

        # Apply top-p (nucleus) filtering
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = -float('inf')

        # Sample from distribution
        probs = F.softmax(logits, dim=-1)  # (B, V)
        next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

        # Append to sequence
        input_ids = torch.cat([input_ids, next_token], dim=1)

    return input_ids
```

---

## 5. Parameter Count Analysis

### 5.1 Component-wise Parameters

For a model with:
- Vocabulary size: $V$
- Model dimension: $d_{\text{model}}$
- FFN dimension: $d_{\text{ff}} = 4 \cdot d_{\text{model}}$
- Number of attention heads: $n_{\text{heads}}$
- Number of layers: $L$

**Embedding Layer**:

$$
N_{\text{embed}} = V \cdot d_{\text{model}}
$$

**Single Attention Layer** (per head has $d_k = d_{\text{model}} / n_{\text{heads}}$):

$$
\begin{align}
N_{\text{attn}} &= n_{\text{heads}} \cdot 3 \cdot d_{\text{model}} \cdot d_k + d_{\text{model}} \cdot d_{\text{model}} \\
&= 3 \cdot d_{\text{model}}^2 + d_{\text{model}}^2 \\
&= 4 \cdot d_{\text{model}}^2
\end{align}
$$

(Factor of 3 for Q, K, V projections; plus output projection $W_O$)

**Single FFN Layer**:

$$
\begin{align}
N_{\text{FFN}} &= d_{\text{model}} \cdot d_{\text{ff}} + d_{\text{ff}} \cdot d_{\text{model}} \\
&= 2 \cdot d_{\text{model}} \cdot d_{\text{ff}} \\
&= 8 \cdot d_{\text{model}}^2
\end{align}
$$

**Single Transformer Layer**:

$$
N_{\text{layer}} = N_{\text{attn}} + N_{\text{FFN}} = 4d_{\text{model}}^2 + 8d_{\text{model}}^2 = 12d_{\text{model}}^2
$$

**LayerNorm** (typically negligible):

$$
N_{\text{LN}} = 2 \cdot d_{\text{model}} \quad \text{(scale and bias)}
$$

**Output Projection** (often tied with embedding):

$$
N_{\text{out}} = d_{\text{model}} \cdot V
$$

**Total Parameters**:

$$
N_{\text{total}} = V \cdot d_{\text{model}} + L \cdot 12d_{\text{model}}^2 + V \cdot d_{\text{model}}
$$

With weight tying ($W_{\text{out}} = E^T$):

$$
N_{\text{total}} = V \cdot d_{\text{model}} + L \cdot 12d_{\text{model}}^2
$$

### 5.2 Example: 1B Parameter Model

Target: $N_{\text{total}} \approx 10^9$ parameters

**Typical configuration**:
- $V = 50{,}000$ (vocabulary)
- $d_{\text{model}} = 2048$
- $n_{\text{heads}} = 16$
- $d_{\text{ff}} = 4 \cdot d_{\text{model}} = 8192$
- $L = ?$ (solve for this)

**Embedding parameters**:

$$
N_{\text{embed}} = 50{,}000 \times 2048 = 102{,}400{,}000 \approx 102\text{M}
$$

**Per-layer parameters**:

$$
N_{\text{layer}} = 12 \times 2048^2 = 50{,}331{,}648 \approx 50.3\text{M}
$$

**Solve for $L$**:

$$
10^9 = 102{,}400{,}000 + L \times 50{,}331{,}648
$$

$$
L = \frac{10^9 - 102{,}400{,}000}{50{,}331{,}648} \approx 17.8 \approx 18 \text{ layers}
$$

**Verification**:

$$
N_{\text{total}} = 102.4\text{M} + 18 \times 50.3\text{M} = 102.4\text{M} + 905.4\text{M} = 1{,}007.8\text{M} \approx 1\text{B}
$$

### 5.3 Parameter Distribution

```
Component          Parameters    Percentage
────────────────────────────────────────────
Embedding          102.4M        10.2%
Transformer (18L)  905.4M        89.8%
  ├─ Attention     144.8M        14.4%
  └─ FFN           760.6M        75.5%
────────────────────────────────────────────
Total              1,007.8M      100.0%
```

**Key insight**: Most parameters are in FFN layers (~75%), not attention (~14%).

### 5.4 Comparison: Popular Models

| Model | $d_{\text{model}}$ | $n_{\text{heads}}$ | $d_{\text{ff}}$ | $L$ | Params |
|-------|-------------------|-------------------|----------------|-----|--------|
| GPT-2 Small | 768 | 12 | 3072 | 12 | 117M |
| GPT-2 Medium | 1024 | 16 | 4096 | 24 | 345M |
| GPT-2 Large | 1280 | 20 | 5120 | 36 | 774M |
| GPT-2 XL | 1600 | 25 | 6400 | 48 | 1.5B |
| LLaMA 7B | 4096 | 32 | 11008 | 32 | 7B |
| LLaMA 13B | 5120 | 40 | 13824 | 40 | 13B |

---

## 6. References

### Foundational Papers

1. **Attention Is All You Need** (Vaswani et al., 2017)
   Original transformer architecture.

2. **Language Models are Unsupervised Multitask Learners** (Radford et al., 2019)
   GPT-2: Scaling autoregressive LMs.

3. **Language Models are Few-Shot Learners** (Brown et al., 2020)
   GPT-3: Emergent abilities at scale.

4. **LLaMA: Open and Efficient Foundation Language Models** (Touvron et al., 2023)
   Modern efficient architecture choices.

### Key Implementation Details

- **Layer Normalization**: Ba et al. (2016)
- **SwiGLU Activation**: Shazeer (2020)
- **Rotary Position Embeddings (RoPE)**: Su et al. (2021)
- **Flash Attention**: Dao et al. (2022) - efficient attention computation

### Cross-References

- See [llm-failures.md](./llm-failures.md) for analysis of fundamental limitations
- See [aletheion-integration.md](./aletheion-integration.md) for how Aletheion modifies this architecture
- See [training-strategy.md](./training-strategy.md) for training Aletheion-augmented models

---

**Next**: [LLM Failure Modes →](./llm-failures.md)
