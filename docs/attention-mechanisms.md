# Attention Mechanisms in Aletheion LLM

## Table of Contents
1. [Single-Head Attention](#1-single-head-attention)
   1. [Mathematical Definition](#11-mathematical-definition)
   2. [Worked Example](#12-worked-example)
   3. [Attention Matrix Visualization](#13-attention-matrix-visualization)
   4. [Reference Python Implementation](#14-reference-python-implementation)
   5. [Limitations](#15-limitations)
2. [Multi-Head Attention (Standard LLM)](#2-multi-head-attention-standard-llm)
   1. [Why Multiple Heads?](#21-why-multiple-heads)
   2. [Mathematical Formulation](#22-mathematical-formulation)
   3. [Learned Pattern Diversity](#23-learned-pattern-diversity)
   4. [Aggregation and the Softmax Failure](#24-aggregation-and-the-softmax-failure)
   5. [Reference Python Implementation](#25-reference-python-implementation)
3. [Aletheion Multi-Head Attention (Epistemic Gating)](#3-aletheion-multi-head-attention-epistemic-gating)
   1. [Hypothesis A: Per-Head Epistemic Gating](#31-hypothesis-a-per-head-epistemic-gating)
   2. [Hypothesis B: Aggregation-Level Epistemic Gating](#32-hypothesis-b-aggregation-level-epistemic-gating)
   3. [Hypothesis C: Hierarchical Epistemic Gating](#33-hypothesis-c-hierarchical-epistemic-gating)
4. [Comparative Analysis](#4-comparative-analysis)
5. [Concrete Case Studies](#5-concrete-case-studies)

---

## 1. Single-Head Attention

### 1.1 Mathematical Definition

Let an input sequence be represented by the matrix $X \in \mathbb{R}^{T \times d_{\text{model}}}$, where $T$ is the number of tokens and $d_{\text{model}}$ is the embedding dimensionality. A single attention head is parameterized by three learned linear projections:

$$
W^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}, \quad W^K \in \mathbb{R}^{d_{\text{model}} \times d_k}, \quad W^V \in \mathbb{R}^{d_{\text{model}} \times d_v}.
$$

Queries, keys, and values are obtained by:

$$
Q = X W^Q, \quad K = X W^K, \quad V = X W^V.
$$

The **scaled dot-product attention** computes attention scores and weights as:

$$
S = \frac{Q K^\top}{\sqrt{d_k}} \in \mathbb{R}^{T \times T}, \quad A = \text{softmax}(S) \in \mathbb{R}^{T \times T},
$$

where the softmax is applied row-wise. The output for the head is:

$$
\text{Head}(Q, K, V) = A V.
$$

Intuitively, each row of $A$ is a probability distribution over other tokens indicating how much the current token attends to each context token.

### 1.2 Worked Example

Consider the sequence `The cat sleeps` with token embeddings in $\mathbb{R}^3$:

$$
X = \begin{bmatrix}
1 & 0 & 1 \\
0 & 1 & 1 \\
1 & 1 & 0
\end{bmatrix}.
$$

Let the projection matrices be:

$$
W^Q = \begin{bmatrix}1 & 0 \\ 0 & 1 \\ 1 & 1\end{bmatrix}, \quad
W^K = \begin{bmatrix}1 & 1 \\ 0 & 1 \\ 1 & 0\end{bmatrix}, \quad
W^V = \begin{bmatrix}1 & 0 & 1 \\ 1 & 1 & 0 \\ 0 & 1 & 1\end{bmatrix}.
$$

Then:

$$
Q = X W^Q = \begin{bmatrix}2 & 1 \\ 1 & 2 \\ 2 & 1\end{bmatrix}, \quad
K = X W^K = \begin{bmatrix}2 & 1 \\ 1 & 2 \\ 1 & 1\end{bmatrix}, \quad
V = X W^V = \begin{bmatrix}1 & 1 & 2 \\ 2 & 2 & 1 \\ 1 & 2 & 1\end{bmatrix}.
$$

Scaled dot-product scores (with $d_k = 2$):

$$
S = \frac{Q K^\top}{\sqrt{2}} = \frac{1}{\sqrt{2}}\begin{bmatrix}
(2,1)\cdot(2,1) & (2,1)\cdot(1,2) & (2,1)\cdot(1,1) \\
(1,2)\cdot(2,1) & (1,2)\cdot(1,2) & (1,2)\cdot(1,1) \\
(2,1)\cdot(2,1) & (2,1)\cdot(1,2) & (2,1)\cdot(1,1)
\end{bmatrix} = \frac{1}{\sqrt{2}}\begin{bmatrix}
5 & 4 & 3 \\
4 & 5 & 3 \\
5 & 4 & 3
\end{bmatrix}.
$$

Applying row-wise softmax gives:

$$
A = \text{softmax}(S) \approx \begin{bmatrix}
0.54 & 0.30 & 0.16 \\
0.30 & 0.54 & 0.16 \\
0.54 & 0.30 & 0.16
\end{bmatrix}.
$$

The head output is then:

$$
\text{Head}(Q, K, V) = A V \approx \begin{bmatrix}
1.54 & 1.58 & 1.44 \\
1.58 & 1.54 & 1.44 \\
1.54 & 1.58 & 1.44
\end{bmatrix}.
$$

### 1.3 Attention Matrix Visualization

A textual visualization of matrix $A$ helps to reason about the focus of each token:

```
Token →      The       cat       sleeps
The        [ 0.54 |  0.30 |  0.16 ]
cat        [ 0.30 |  0.54 |  0.16 ]
sleeps     [ 0.54 |  0.30 |  0.16 ]
```

The first row indicates that the token `The` mostly attends to itself (0.54), somewhat to `cat` (0.30), and weakly to `sleeps` (0.16).

### 1.4 Reference Python Implementation

```python
import torch
import torch.nn.functional as F

def single_head_attention(x, Wq, Wk, Wv):
    Q = x @ Wq
    K = x @ Wk
    V = x @ Wv
    scores = (Q @ K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)
    weights = F.softmax(scores, dim=-1)
    return weights @ V

# Example usage
x = torch.tensor([[1., 0., 1.], [0., 1., 1.], [1., 1., 0.]])
Wq = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
Wk = torch.tensor([[1., 1.], [0., 1.], [1., 0.]])
Wv = torch.tensor([[1., 0., 1.], [1., 1., 0.], [0., 1., 1.]])
head_output = single_head_attention(x, Wq, Wk, Wv)
```

### 1.5 Limitations

1. **Limited Expressivity**: A single head can model only one dominant interaction pattern, restricting the ability to simultaneously capture syntax, semantics, and long-range dependencies.
2. **Softmax Normalization**: Every row in $A$ must sum to 1, forcing the head to allocate probability mass even when all keys appear equally irrelevant.
3. **No Uncertainty Modeling**: The scalar probabilities convey focus but not epistemic confidence—there is no way to represent "I do not know." This limitation motivates multi-head extensions and the epistemic gating explored in Aletheion LLM.

---

## 2. Multi-Head Attention (Standard LLM)

### 2.1 Why Multiple Heads?

Multiple heads allow the model to project the same token embeddings into different representation subspaces. Each head specializes in different relational patterns: syntactic dependencies, semantic roles, positional offsets, coreference resolution, etc. The aggregation of diverse insights increases modeling capacity without a quadratic increase in compute.

### 2.2 Mathematical Formulation

For $h$ heads, each head $i \in \{1, \dots, h\}$ has projections:

$$
W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}, \quad
W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}, \quad
W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}.
$$

The per-head computations follow the single-head process:

$$
Q_i = X W_i^Q, \quad K_i = X W_i^K, \quad V_i = X W_i^V,
$$

$$
S_i = \frac{Q_i K_i^\top}{\sqrt{d_k}}, \quad A_i = \text{softmax}(S_i), \quad H_i = A_i V_i.
$$

The multi-head output concatenates and projects:

$$
\text{MHA}(X) = \left[H_1 \, \Vert \cdots \Vert \, H_h\right] W^O,
$$

where $W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$ is the output projection.

### 2.5 Reference Python Implementation

```python
import torch

class StandardMHA(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_k, d_v):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.Wq = torch.nn.Linear(d_model, num_heads * d_k)
        self.Wk = torch.nn.Linear(d_model, num_heads * d_k)
        self.Wv = torch.nn.Linear(d_model, num_heads * d_v)
        self.Wo = torch.nn.Linear(num_heads * d_v, d_model)

    def forward(self, x):
        B, T, _ = x.shape
        Q = self.Wq(x).view(B, T, self.num_heads, self.d_k)
        K = self.Wk(x).view(B, T, self.num_heads, self.d_k)
        V = self.Wv(x).view(B, T, self.num_heads, self.d_v)
        scores = torch.einsum("bthd,bshd->bhts", Q, K) / (self.d_k ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        heads = torch.einsum("bhts,bshd->bthd", weights, V)
        H = heads.reshape(B, T, self.num_heads * self.d_v)
        return self.Wo(H)
```

### 2.3 Learned Pattern Diversity

Empirically, different heads discover specialized functions:
- **Syntax Head**: Aligns verbs with subjects or modifiers.
- **Semantic Head**: Links tokens with similar meaning or topical relevance.
- **Coreference Head**: Connects pronouns to antecedents.
- **Long-Range Head**: Focuses on distant tokens, such as retrieving facts across paragraphs.

This diversity emerges because each head has independent parameters and gradients while sharing downstream loss.

### 2.4 Aggregation and the Softmax Failure

#### Per-Head Softmax Constraint

Each head $A_i$ is normalized independently via softmax, so every row sums to 1. Even when all scores in a row are low, softmax transforms them into a distribution. Consequently, $H_i$ always produces a weighted sum of values—even under high uncertainty.

#### Failure Example

Consider two heads ($h = 2$) attending over three tokens, with value vectors $V_1 = V_2 = \text{Identity}_{3}$ for clarity. Suppose the unnormalized scores for a target token are:

$$
S_1 = \begin{bmatrix}0.1 & 0.0 & 0.2\end{bmatrix}, \quad
S_2 = \begin{bmatrix}0.0 & 0.1 & 0.1\end{bmatrix}.
$$

Softmaxed attention weights become:

$$
A_1 = [0.36, 0.33, 0.31], \quad A_2 = [0.33, 0.36, 0.31].
$$

Both heads are nearly uniform—no token stands out. Yet the concatenated output $[H_1 \Vert H_2]$ equals the almost uniform convex combination of values. If the output projection is

$$
W^O = \begin{bmatrix}
2 & -1 & 0 & 1 & 1 & 0 \\
-1 & 2 & 1 & 0 & 1 & 0 \\
0 & 1 & 2 & -1 & 0 & 1
\end{bmatrix},
$$

then the final logits are biased by the linear combination, potentially producing a confident distribution such as $[5.1, 4.8, 4.6]$. The model appears certain despite the heads distributing weight uniformly.

#### Why This Causes Hallucination

1. **Forced Commitment**: Softmax enforces a pseudo-confidence, so the model must select some context. Downstream layers interpret non-zero activations as evidence, leading to overconfident predictions.
2. **Aggregation Amplification**: $W^O$ mixes head outputs linearly. Even small, noisy signals can align coherently after projection, creating spurious peaks in the logits.
3. **Lack of Epistemic Feedback**: There is no channel to express disagreement or uncertainty across heads. When all heads are uncertain, the system still produces sharp outputs, manifesting as hallucinated statements.

A numerical visualization highlights the mismatch:

```
Head 1 weights: [0.36, 0.33, 0.31]
Head 2 weights: [0.33, 0.36, 0.31]
Combined signal (after W^O): logits ≈ [5.1, 4.8, 4.6]
Softmax(logits): [0.46, 0.33, 0.21]  ← falsely confident
```

---

## 3. Aletheion Multi-Head Attention (Epistemic Gating)

Aletheion introduces **epistemic gates** that modulate attention flows based on estimated uncertainty. We explore three complementary hypotheses.

### 3.1 Hypothesis A: Per-Head Epistemic Gating

#### Formulation

Each head gains a multiplicative gate derived from its attention scores:

$$
\gamma_i = Q^{(1)}_{\text{gate}}(S_i) \in [0, 1]^{T \times 1},
$$

$$
\tilde{H}_i = \gamma_i \odot (A_i V_i),
$$

where $Q^{(1)}_{\text{gate}}$ is a learnable function (e.g., a two-layer perceptron with sigmoid output) that compresses per-token uncertainty. The gate is broadcast across value dimensions.

An equivalent expanded expression is:

$$
\tilde{H}_i = \left(Q^{(1)}_{\text{gate}}(S_i) \odot \text{softmax}(S_i)\right) V_i.
$$

#### Intuition

- **Cause**: When $S_i$ is diffuse, the gate can output values near zero.
- **Effect**: The head effectively disables itself, preventing noisy contributions to $W^O$.

#### Python Reference Implementation

```python
import torch
import torch.nn.functional as F

class HeadWithGate(torch.nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super().__init__()
        self.Wq = torch.nn.Linear(d_model, d_k)
        self.Wk = torch.nn.Linear(d_model, d_k)
        self.Wv = torch.nn.Linear(d_model, d_v)
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(d_k, d_k),
            torch.nn.ReLU(),
            torch.nn.Linear(d_k, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / Q.size(-1) ** 0.5
        weights = F.softmax(scores, dim=-1)
        gate = self.gate(scores)  # shape: (batch, T, 1)
        head = torch.matmul(weights, V)
        return gate * head
```

#### ASCII Flow Diagram

```
X ─▶ W_i^Q ─┐                ┌─▶ softmax ─┐
            │                │            │
            └─▶ score S_i ─▶ Q₁_gate ─┐  │
K ◀─ W_i^K ◀──────────────────────────┘  │
                                         ▼
V ◀─ W_i^V ◀───────────────▶ values ─▶ ⊙ ─▶ gated head
```

#### Pros
- Fine-grained: each head decides locally whether to contribute.
- Compatible with existing architectures by wrapping head outputs.
- Reduces hallucinations arising from individual head confusion.

#### Cons
- Requires additional parameters per head.
- Gate accuracy depends solely on local scores, possibly missing cross-head conflicts.

### 3.2 Hypothesis B: Aggregation-Level Epistemic Gating

#### Formulation

Gating is applied after concatenating all heads:

$$
H = [H_1 \Vert H_2 \Vert \cdots \Vert H_h],
$$

$$
\Gamma = Q^{(2)}_{\text{gate}}(H) \in [0, 1]^{T \times h d_v},
$$

$$
\tilde{O} = (\Gamma \odot H) W^O.
$$

Here $Q^{(2)}_{\text{gate}}$ can assess inter-head agreement by examining the joint representation.

#### Intuition

- **Cause**: Heads provide conflicting signals or low-energy activations.
- **Effect**: The gate attenuates the aggregated vector, reducing logit magnitude.

#### Python Reference Implementation

```python
import torch

class AggregationGateMHA(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_k, d_v):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.Wq = torch.nn.Linear(d_model, num_heads * d_k)
        self.Wk = torch.nn.Linear(d_model, num_heads * d_k)
        self.Wv = torch.nn.Linear(d_model, num_heads * d_v)
        self.Wo = torch.nn.Linear(num_heads * d_v, d_model)
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(num_heads * d_v, num_heads * d_v),
            torch.nn.GELU(),
            torch.nn.Linear(num_heads * d_v, num_heads * d_v),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        B, T, _ = x.shape
        Q = self.Wq(x).view(B, T, self.num_heads, self.d_k)
        K = self.Wk(x).view(B, T, self.num_heads, self.d_k)
        V = self.Wv(x).view(B, T, self.num_heads, self.d_v)
        scores = torch.einsum("bthd,bshd->bhts", Q, K) / (self.d_k ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        heads = torch.einsum("bhts,bshd->bthd", weights, V)
        H = heads.reshape(B, T, self.num_heads * self.d_v)
        gate = self.gate(H)
        return self.Wo(gate * H)
```

#### ASCII Flow Diagram

```
Head_1 ─┐
Head_2 ─┼─▶ concat ─▶ Q₂_gate ─▶ ⊙ ─▶ W^O ─▶ output
⋮       │
Head_h ─┘
```

#### Pros
- Captures consensus/disagreement among heads.
- Requires only one additional gating module.
- Learns to recognize contradictory patterns.

#### Cons
- Cannot silence a single problematic head; only post-hoc attenuation.
- Larger gating vector ($h d_v$) increases memory bandwidth.

### 3.3 Hypothesis C: Hierarchical Epistemic Gating

#### Formulation

Combine per-head and aggregation gating:

$$
\tilde{H}_i = Q^{(1)}_{\text{gate}}(S_i) \odot (A_i V_i), \quad i = 1,\dots,h,
$$

$$
\tilde{H} = [\tilde{H}_1 \Vert \cdots \Vert \tilde{H}_h], \quad
\tilde{O} = Q^{(2)}_{\text{gate}}(\tilde{H}) \odot \tilde{H}, \quad
\text{Output} = \tilde{O} W^O.
$$

#### Intuition

- **Cause**: Both local uncertainty and cross-head disagreement are evaluated.
- **Effect**: Provides a two-stage epistemic filter that can silence confused heads and dampen inconsistent aggregates.

#### Python Reference Implementation

```python
import torch

class HierarchicalGateMHA(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_k, d_v):
        super().__init__()
        self.heads = torch.nn.ModuleList([
            HeadWithGate(d_model, d_k, d_v) for _ in range(num_heads)
        ])
        self.Wo = torch.nn.Linear(num_heads * d_v, d_model)
        self.agg_gate = torch.nn.Sequential(
            torch.nn.Linear(num_heads * d_v, num_heads * d_v),
            torch.nn.ReLU(),
            torch.nn.Linear(num_heads * d_v, num_heads * d_v),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        gated_heads = [head(x) for head in self.heads]
        H = torch.cat(gated_heads, dim=-1)
        gate = self.agg_gate(H)
        return self.Wo(gate * H)
```

#### ASCII Flow Diagram

```
             ┌─────────────┐
X ─▶ Head_1 ─┤ Q₁_gate ⊙ ├─┐
X ─▶ Head_2 ─┤ Q₁_gate ⊙ ├─┼─▶ concat ─▶ Q₂_gate ─▶ ⊙ ─▶ W^O ─▶ output
⋮            └─────────────┘ │
X ─▶ Head_h ────────────────┘
```

#### Pros
- Maximum flexibility; can capture granular and global uncertainty patterns.
- Encourages heads to specialize while maintaining global coherence.

#### Cons
- Highest computational overhead due to dual gating networks.
- Requires careful regularization to avoid degenerate behavior (e.g., all gates closing).

---

## 4. Comparative Analysis

### 4.1 Capability and Cost Matrix

| Mechanism                     | Extra Params        | FLOP Overhead            | Uncertainty Captured                   | Pros                                         | Cons                                      |
|------------------------------|---------------------|--------------------------|----------------------------------------|----------------------------------------------|-------------------------------------------|
| Standard MHA                 | Baseline            | Baseline                 | None                                   | Efficient, well-understood                   | Overconfident when heads are uncertain    |
| Hypothesis A (Per-Head)      | $O(h d_k)$          | $\approx$ +10% per head | Local head uncertainty                 | Silences noisy heads                        | Misses cross-head conflicts               |
| Hypothesis B (Aggregation)   | $O(h d_v d_{\text{model}})$ | $\approx$ +5% total      | Inter-head disagreement                | Captures consensus, simple to integrate     | Still mixes uncertain heads pre-gating    |
| Hypothesis C (Hierarchical)  | Sum of A + B        | $\approx$ +15% total     | Both local and global                  | Rich epistemic representation                | Highest complexity and training cost      |

### 4.2 Detection Capability by Uncertainty Type

| Uncertainty Type          | Standard MHA | Hyp. A | Hyp. B | Hyp. C |
|---------------------------|--------------|--------|--------|--------|
| Local token ambiguity     | ✗            | ✓      | △      | ✓      |
| Head-to-head disagreement | ✗            | △      | ✓      | ✓      |
| Global knowledge gaps     | ✗            | △      | △      | ✓      |

### 4.3 Recommendation

1. **Implement Hypothesis A first**: Minimal architectural change, immediate mitigation of hallucinations caused by locally confused heads.
2. **Add Hypothesis B** if global disagreement remains frequent in evaluations.
3. **Adopt Hypothesis C** for high-stakes deployments where epistemic calibration outweighs compute cost.

---

## 5. Concrete Case Studies

We examine three scenarios across mechanisms. Assume logits are converted to confidence via softmax and that higher entropy implies lower confidence.

### Case 1: Clear Question (“What is the capital of France?”)

- **Ground Truth**: Paris.
- **Observations**:
  - **Standard MHA**: All heads align on the supporting token `Paris`; logits sharp (confidence 0.92).
  - **Hypothesis A**: Gates remain near 1; identical output to baseline but with explicit confirmation of low uncertainty.
  - **Hypothesis B**: Aggregation gate outputs ones due to agreement.
  - **Hypothesis C**: Both levels pass signals with minimal attenuation.
- **Outcome**: All methods provide high-confidence correct answer.

### Case 2: Ambiguous Question (“What is the bank near the river?”)

Let heads focus on different interpretations (`financial bank` vs `river bank`). Attention and gating effects:

| Mechanism | Dominant Heads | Gate Values | Logit Spread | Confidence |
|-----------|----------------|-------------|--------------|------------|
| Standard  | Head A → finance, Head B → geography | N/A         | 4.0 vs 3.8 | 0.55      |
| Hyp. A    | Head A gate 0.6, Head B gate 0.4     | Local gating lowers contributions | 3.2 vs 3.1 | 0.42 |
| Hyp. B    | Gates detect disagreement, reduce magnitude to 0.5 | Aggregation gate ≈ 0.5 | 1.8 vs 1.7 | 0.33 |
| Hyp. C    | Head gates 0.6/0.4, aggregation gate 0.5 | Two-level attenuation | 1.2 vs 1.1 | 0.29 |

The epistemic gates decrease confidence when interpretations conflict.

### Case 3: Fictitious Entity (“Who discovered the planet Xandor?”)

- **Standard MHA**: All heads spread attention uniformly yet produce logits [3.5, 3.4, 3.3]; softmax ≈ [0.36, 0.33, 0.31] suggesting weak but non-zero confidence, often leading to fabricated answers.
- **Hypothesis A**: Each head’s gate < 0.1 because scores are diffuse; output ≈ zero vector before $W^O$, yielding near-uniform logits and signaling ignorance.
- **Hypothesis B**: Heads emit conflicting low-energy vectors; aggregation gate outputs ≈ 0.1, shrinking logits and preventing hallucination.
- **Hypothesis C**: Combines both effects, producing the flattest logits; downstream calibration layers can map this to explicit “unknown.”

### Numerical Simulation Snapshot

```
Scenario: Fictitious entity

Standard MHA logits      : [3.50, 3.40, 3.30] → confidences [0.36, 0.33, 0.31]
Hypothesis A logits       : [0.40, 0.38, 0.37] → confidences [0.35, 0.34, 0.31]
Hypothesis B logits       : [0.20, 0.19, 0.18] → confidences [0.34, 0.33, 0.33]
Hypothesis C logits       : [0.05, 0.05, 0.05] → confidences [0.33, 0.33, 0.33]
```

### Implementation Considerations

- **Calibration**: Downstream calibration layers should interpret low-magnitude logits as epistemic uncertainty, enabling abstention.
- **Training Objective**: Consider auxiliary losses (e.g., entropy regularization) to encourage meaningful gating rather than trivial all-ones or all-zeros solutions.
- **Monitoring**: Track gate activations across evaluation sets to ensure desired behavior (e.g., average gate value vs. correctness).

---

**Summary**: Single-head attention captures limited relational structure, and standard multi-head attention increases expressivity but cannot express uncertainty due to per-head softmax normalization. Aletheion’s epistemic gating hypotheses introduce trainable mechanisms to detect and suppress uncertain signals, mitigating hallucinations while preserving high-confidence answers.
