# Remaining Limitations: Honest Assessment

## Table of Contents
1. [Overview](#overview)
2. [Limitation A: Q-Systems Still Operate in Embedding Space](#limitation-a)
3. [Limitation B: Gating Detects But Doesn't Improve Base Capability](#limitation-b)
4. [Limitation C: Thresholds Are Hyperparameters](#limitation-c)
5. [Limitation D: Computational Overhead](#limitation-d)
6. [Limitation E: Training Data Quality Dependency](#limitation-e)
7. [What Aletheion Does NOT Solve](#what-aletheion-does-not-solve)
8. [Future Work: Hybrid Symbolic-Neural Systems](#future-work)
9. [References](#references)

---

## Overview

**Aletheion is not a panacea.** While it addresses critical failure modes of standard LLMs (see [llm-failures.md](./llm-failures.md)), it has **fundamental limitations** that cannot be solved within the current paradigm.

This document provides an **honest, technically rigorous assessment** of:
1. What Aletheion does **NOT** solve
2. Remaining architectural limitations
3. Open problems for future research

**Philosophical stance**: Scientific integrity requires acknowledging limitations. Overselling capabilities harms the field.

---

## Limitation A: Q-Systems Still Operate in Embedding Space

### The Problem

Q₁ and Q₂ are **neural networks** that operate on hidden states $\mathbf{h}_t \in \mathbb{R}^{d_{\text{model}}}$:

$$
Q_1(\mathbf{h}_t) = \sigma\left( \mathbf{w}_1^T \mathbf{h}_t + b_1 \right)
$$

$$
Q_2(\mathbf{h}_t) = \sigma\left( \mathbf{w}_2^T \mathbf{h}_t + b_2 \right)
$$

**Critical issue**: $\mathbf{h}_t$ is an **implicit, continuous representation**. The Q-systems:
- Do **NOT** have access to discrete symbolic structures
- Do **NOT** perform formal logical inference
- Do **NOT** verify facts against ground truth

They are **learned pattern detectors**, not symbolic reasoners.

### Mathematical Explanation

Recall from [llm-failures.md](./llm-failures.md#failure-1) that LLMs learn:

$$
\phi: \text{Tokens}(C) \to \mathbb{R}^{d_{\text{model}}}
$$

where $C$ is a symbolic concept (e.g., "prime number", "valid proof").

**Q-systems inherit this limitation**:

$$
Q_1: \mathbb{R}^{d_{\text{model}}} \to [0, 1]
$$

There is **no inverse** to extract symbolic structure:

$$
\phi^{-1}: \mathbb{R}^{d_{\text{model}}} \to \text{Symbols}(C)
$$

**Consequence**: Q₁ and Q₂ can detect **statistical anomalies** in embeddings, but cannot:
- Verify logical consistency (e.g., "If P and P→Q, then Q")
- Check factual correctness (e.g., "Is Paris the capital of France?" → query database)
- Perform symbolic reasoning (e.g., "Is 17 prime?" → trial division)

### Concrete Example: Arithmetic

```python
# Standard LLM
prompt = "What is 123456 + 654321?"
response = llm.generate(prompt)
# Output: "777777" ✓ (correct by pattern)

# If the pattern breaks:
prompt = "What is 987654321 + 123456789?"
response = llm.generate(prompt)
# Output: "1111111100" ✗ (wrong)

# Aletheion LLM with Q₁
response, q1_score = aletheion_llm.generate(prompt)
# Q₁ detects: embedding inconsistency (Q₁ = 0.42 < 0.7)
# Output: "<UNK>" (abstains)

# But Aletheion does NOT compute the correct answer!
# It detects the LLM is uncertain, but has no symbolic calculator
```

**What's missing**: A **symbolic module** that can:

$$
\text{Verify}_{\text{arithmetic}}(a, b, c) = \mathbb{1}[a + b = c]
$$

### Why This Matters

Q-systems **reduce false positives** (hallucinations) but do **not increase true positives** (correct answers on hard problems).

```
┌─────────────────────────────────────────────────────────────┐
│                    Standard LLM                              │
├─────────────────────────────────────────────────────────────┤
│  Easy problems:  90% correct                                 │
│  Hard problems:  30% correct (70% wrong with high confidence)│
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   Aletheion LLM                              │
├─────────────────────────────────────────────────────────────┤
│  Easy problems:  90% correct (same as base)                  │
│  Hard problems:  30% correct, 60% abstain, 10% wrong         │
│                  ↑ Better: Avoids 60% of mistakes!           │
│                  ✗ But: Still can't solve the hard 70%       │
└─────────────────────────────────────────────────────────────┘
```

---

## Limitation B: Gating Detects But Doesn't Improve Base Capability

### The Problem

Aletheion's gating mechanism can **reject** unreliable generation, but cannot **fix** it.

$$
\text{Gate}(x_{<t}) = \begin{cases}
\text{PASS} & \text{if } Q_1, Q_2 \geq \text{threshold}, \text{VARO} \leq \text{threshold} \\
\text{REJECT} & \text{otherwise}
\end{cases}
$$

**When REJECT occurs**:
- Option 1: Emit `<UNK>` (abstention)
- Option 2: Trigger fallback (e.g., retrieval, human, symbolic solver)
- Option 3: Resample (but still from $P(x_t \mid x_{<t})$, same flawed distribution)

**None of these options make the base LLM better at the task.**

### Mathematical Explanation

The base LLM defines:

$$
P(x_t \mid x_{<t}; \theta)
$$

If this distribution is **fundamentally flawed** (e.g., assigns low probability to correct answer), then:

$$
\max_{x_t} P(x_t \mid x_{<t}) = \text{wrong answer}
$$

**Gating cannot fix this** because it operates **downstream** of the distribution:

```
LLM → P(x_t | x_{<t}) → [GATING] → Output
  ↑                         ↑
  Flawed distribution     Can only reject, not improve
```

### Concrete Example: Knowledge Gap

```python
# Question outside training data
prompt = "What is the population of the newly discovered exoplanet Kepler-1649c?"

# Standard LLM
response = llm.generate(prompt)
# Output: "Approximately 7 million" (hallucinated)

# Aletheion LLM
response, q1, q2 = aletheion_llm.generate(prompt)
# Q₁ = 0.38 (low coherence, model is uncertain)
# Output: "<UNK> [Low confidence, consider external search]"

# ✓ Aletheion avoided hallucination
# ✗ Aletheion still doesn't KNOW the answer
```

**What's needed**: External knowledge retrieval (RAG, database lookup, web search).

### Implication: Aletheion is a **Filter**, Not a **Solver**

```
┌──────────────────────────────────────────────────────────────┐
│                    CAPABILITY FRONTIER                        │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│   Standard LLM:  [────────────────── 70% ──────────────────] │
│                  ↑ Can solve                 ↑ Hallucinates  │
│                                                               │
│  Aletheion LLM:  [────── 70% ──────][ABSTAIN][─ 5% wrong ─] │
│                  ↑ Can solve         ↑ Detects uncertainty   │
│                                                               │
│  **Capability frontier unchanged at 70%!**                   │
│  Improvement: Reduced hallucinations from 30% → 5%           │
│  Limitation: Still can't solve the hard 30%                  │
└──────────────────────────────────────────────────────────────┘
```

---

## Limitation C: Thresholds Are Hyperparameters (Not Adaptive)

### The Problem

Aletheion requires setting **fixed thresholds**:

$$
Q_1^{\min}, \quad Q_2^{\min}, \quad \text{VARO}^{\max}
$$

These are **hyperparameters** chosen during training/validation, not **adaptive** to context.

**Issues**:
1. **Task-dependence**: Optimal thresholds vary by task
2. **Distribution shift**: Thresholds tuned on validation set may not generalize
3. **Context-dependence**: Some contexts are inherently harder (should have higher threshold)

### Mathematical Explanation

Ideally, thresholds should be **context-dependent**:

$$
Q_1^{\min}(x_{<t}), \quad Q_2^{\min}(x_{<t}), \quad \text{VARO}^{\max}(x_{<t})
$$

For example:
- **Easy question**: Lower threshold (tolerate more uncertainty)
- **Critical application** (medical, legal): Higher threshold (strict verification)

**Current Aletheion**: Uses **global thresholds**:

$$
Q_1^{\min} = 0.7 \quad \text{(fixed for all contexts)}
$$

**Consequence**: Suboptimal trade-off between **coverage** (fraction answered) and **accuracy**.

### Concrete Example: Task Variability

```python
# Task 1: Creative writing (allow low Q₁)
prompt1 = "Write a poem about clouds"
# Aletheion with Q₁_min = 0.7 may abstain unnecessarily

# Task 2: Medical diagnosis (require high Q₁)
prompt2 = "Is this rash a symptom of meningitis?"
# Aletheion with Q₁_min = 0.7 may still hallucinate (should be 0.9)
```

**What's needed**: **Meta-learning** to predict optimal thresholds:

$$
\theta^*(x_{<t}) = f_{\text{meta}}(x_{<t})
$$

### Calibration Drift

Thresholds are calibrated on a **validation set** at training time. If the **deployment distribution** differs:

$$
P_{\text{train}}(x) \neq P_{\text{deploy}}(x)
$$

Calibration degrades:

$$
\mathbb{E}_{x \sim P_{\text{deploy}}}[\text{Accuracy} \mid Q_1(x) = q] \neq q
$$

**Example**: Model trained on Wikipedia, deployed on Twitter (informal language, slang, typos) → Q₁ thresholds may be too strict.

---

## Limitation D: Computational Overhead

### The Problem

Aletheion adds **three additional forward passes** through neural networks:

1. Q₁ head: $\mathbf{w}_1^T \mathbf{h}_t$ (small MLP)
2. Q₂ head: $\mathbf{w}_2^T \mathbf{h}_t$ (small MLP)
3. VARO computation: Variance over $\mathbf{h}_{t-k:t}$ (requires storing history)

### Computational Cost Analysis

**Base LLM inference** (per token):

$$
\text{FLOPs}_{\text{LLM}} \approx 2 \times N_{\text{params}} \approx 2 \times 10^9 \quad \text{(for 1B model)}
$$

**Aletheion additions**:

$$
\begin{align}
\text{FLOPs}_{Q_1} &\approx 2 \times d_{\text{model}} \times (d_{\text{model}} / 4) \approx 10^6 \\
\text{FLOPs}_{Q_2} &\approx 2 \times d_{\text{model}} \times (d_{\text{model}} / 4) \approx 10^6 \\
\text{FLOPs}_{\text{VARO}} &\approx k \times d_{\text{model}} \approx 2 \times 10^4 \quad (k = 10)
\end{align}
$$

**Total overhead**:

$$
\frac{\text{FLOPs}_{\text{Aletheion}}}{\text{FLOPs}_{\text{LLM}}} \approx \frac{2 \times 10^6}{2 \times 10^9} = 0.1\% \quad \text{(negligible)}
$$

**Memory overhead** (storing hidden states for VARO):

$$
\text{Memory}_{\text{VARO}} = k \times d_{\text{model}} \times \text{sizeof(float32)} = 10 \times 2048 \times 4 = 81 \text{ KB per sequence}
$$

For batch size 32:

$$
\text{Total memory} = 32 \times 81 \text{ KB} = 2.6 \text{ MB} \quad \text{(acceptable)}
$$

**Conclusion**: Computational overhead is **minimal** (~0.1% FLOPs, ~2-5 MB memory).

**However**: If using **Level 3 integration** (VARO in all layers), overhead increases:

$$
\text{FLOPs}_{\text{VARO, all layers}} \approx L \times k \times d_{\text{model}} \approx 18 \times 10 \times 2048 \approx 3.7 \times 10^5
$$

Still <0.02% overhead, but memory grows linearly with layers.

---

## Limitation E: Training Data Quality Dependency

### The Problem

Q₁ and Q₂ are **supervised** (or semi-supervised) classifiers. They require:

1. **Labeled examples** of coherence/incoherence
2. **Labeled examples** of drift/no-drift
3. **High-quality labels** (noisy labels degrade performance)

**Challenges**:
- **Labeling is expensive**: Requires human annotation or heuristics
- **Heuristics are imperfect**: Negative sampling (random tokens) may not capture real failure modes
- **Adversarial examples**: Model may overfit to synthetic negatives, miss real-world failures

### Mathematical Explanation

Q-systems learn:

$$
Q_1: \mathbf{h}_t \to [0, 1], \quad \text{where } Q_1(\mathbf{h}_t) \approx P(y = 1 \mid \mathbf{h}_t)
$$

If training labels $\{(x_i, y_i)\}$ are **biased** or **incomplete**:

$$
\mathbb{E}_{(x, y) \sim P_{\text{train}}}[Q_1(x)] \neq \mathbb{E}_{(x, y) \sim P_{\text{real}}}[Q_1(x)]
$$

**Example**: If negative examples are **only** random token replacements, Q₁ learns:

$$
Q_1(\mathbf{h}_t) \approx P(\text{tokens are not random} \mid \mathbf{h}_t)
$$

But may fail to detect **subtle semantic incoherence** (e.g., factual errors, logical contradictions).

### Concrete Example: Adversarial Failure

```python
# Training: Q₁ learns to detect random token replacements
train_negative = "The capital of France is xyzabc."  # Random token
# Q₁ correctly assigns low score: Q₁ = 0.12

# Deployment: Subtle factual error
test_input = "The capital of France is Lyon."  # Plausible but wrong
# Q₁ may assign high score: Q₁ = 0.74 (missed error!)
```

**Why**: Embedding for "Lyon" is semantically similar to "Paris" (both French cities), so $\mathbf{h}_t$ looks coherent in embedding space.

**What's missing**: External **factual verification** (knowledge base lookup).

---

## What Aletheion Does NOT Solve

### 1. Knowledge Gaps

If information is **not in training data**, Aletheion cannot retrieve it:

- LLM: Hallucinates an answer
- Aletheion: Abstains (better, but still doesn't provide answer)

**Solution**: Retrieval-Augmented Generation (RAG).

### 2. Logical/Mathematical Reasoning

Aletheion cannot perform **symbolic reasoning**:

- LLM: Pattern-matches arithmetic, fails on large numbers
- Aletheion: Detects uncertainty, abstains
- Still cannot solve: $987654321 + 123456789 = ?$

**Solution**: Hybrid symbolic-neural system (call external calculator/theorem prover).

### 3. Multi-Step Planning

Aletheion detects drift **after the fact**, cannot **plan ahead**:

- LLM: Generates step-by-step solution, drifts at step 5
- Aletheion: Detects drift at step 5, abstains
- Cannot: Backtrack to step 4 and choose different path (no tree search)

**Solution**: Explicit planning/search algorithms (e.g., Monte Carlo Tree Search).

### 4. Adversarial Robustness

Q-systems are **neural networks**, subject to adversarial attacks:

$$
\mathbf{h}_{\text{adv}} = \mathbf{h}_t + \epsilon \cdot \text{sign}(\nabla_{\mathbf{h}} Q_1(\mathbf{h}_t))
$$

Carefully crafted $\mathbf{h}_{\text{adv}}$ can fool Q₁ to assign high score to incoherent text.

**Solution**: Adversarial training (but arms race).

### 5. Long-Horizon Dependency

VARO looks back $k=10$ steps (configurable). For **long-range dependencies** (e.g., 1000 tokens apart), VARO may not detect drift.

**Solution**: Hierarchical VARO (operate at multiple timescales).

---

## Future Work: Hybrid Symbolic-Neural Systems

### Vision: NeuroSymbolic Aletheion

```
┌─────────────────────────────────────────────────────────────────┐
│                    NEUROSYMBOLIC ALETHEION                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input → LLM → h_t → [Q₁, Q₂, VARO Gating] → ?                 │
│                              ↓                                   │
│                         ┌────┴─────┐                             │
│                         │   PASS?  │                             │
│                         └────┬─────┘                             │
│                              │                                   │
│                    ┌─────────┼─────────┐                         │
│                    │         │         │                         │
│                   YES       NO        UNCERTAIN                  │
│                    │         │         │                         │
│                    ▼         ▼         ▼                         │
│              Output    Abstain    Route to Symbolic Module      │
│                                         │                        │
│                                         ├→ Calculator (math)     │
│                                         ├→ Theorem Prover (logic)│
│                                         ├→ Knowledge Base (facts)│
│                                         └→ Search Engine (web)   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Research Directions

#### 1. Symbolic Augmentation

When Q-gating fails, **route to symbolic module**:

$$
\text{Output} = \begin{cases}
\text{LLM}(x_{<t}) & \text{if } Q_1, Q_2, \text{VARO} \text{ pass} \\
\text{Symbolic}(x_{<t}) & \text{if } \text{task is symbolic} \land \text{gate fails} \\
\text{Retrieval}(x_{<t}) & \text{if } \text{factual query} \land \text{gate fails} \\
\texttt{<UNK>} & \text{otherwise}
\end{cases}
$$

**Example**:

```python
def aletheion_generate_with_fallback(prompt):
    # Try LLM
    response, q1, q2, varo = aletheion_llm.generate(prompt)

    if q1 >= 0.7 and q2 >= 0.6 and varo <= 0.5:
        return response  # LLM passed gates

    # Detect task type
    if is_arithmetic(prompt):
        return symbolic_calculator(prompt)  # Call sympy, wolfram, etc.
    elif is_factual(prompt):
        return retrieve_from_kb(prompt)  # Query knowledge base
    else:
        return "<UNK>"  # Abstain
```

#### 2. Learnable Thresholds

Instead of fixed $Q_1^{\min}$, learn a **meta-model**:

$$
\theta^*(x_{<t}) = f_{\text{meta}}(x_{<t}; \phi)
$$

where $\phi$ are learned parameters. Train $f_{\text{meta}}$ to maximize:

$$
\max_\phi \mathbb{E}_{x} \left[ \text{Accuracy}(x) - \lambda \cdot \text{Abstention}(x) \right]
$$

**Implementation**: Small MLP that predicts optimal thresholds per example.

#### 3. Backtracking and Search

Integrate **tree search** for multi-step reasoning:

1. Generate $k$ candidate next tokens: $\{x_t^{(1)}, \ldots, x_t^{(k)}\}$
2. For each candidate, compute $Q_1(x_{<t} + x_t^{(i)})$
3. Select candidate with **highest Q₁**:

$$
x_t^* = \arg\max_{i} Q_1(x_{<t} + x_t^{(i)})
$$

4. If all candidates have $Q_1 < Q_1^{\min}$, backtrack.

**Result**: Beam search guided by epistemic quality.

#### 4. Formal Verification Layer

For **high-stakes applications** (medical, legal), add **formal verifier**:

$$
\text{Verify}(\text{claim}, \text{knowledge base}) \in \{\text{True}, \text{False}, \text{Unknown}\}
$$

**Example**: Medical diagnosis

```python
claim = "Patient has meningitis because of rash and fever."
verification = medical_knowledge_base.verify(claim)

if verification == True:
    return claim
elif verification == False:
    return "Incorrect diagnosis (contradicts medical KB)"
else:
    return "Uncertain (insufficient evidence)"
```

**Challenges**: Requires formalized knowledge bases (expensive to create).

#### 5. Uncertainty-Aware RLHF

Modify RLHF to **reward abstention** when uncertain:

$$
r(x, y) = \begin{cases}
+10 & \text{if } y \text{ is correct} \\
+5 & \text{if } y = \texttt{<UNK>} \text{ and model was uncertain} \\
-10 & \text{if } y \text{ is incorrect}
\end{cases}
$$

**Goal**: Train model to **prefer abstention over hallucination**.

---

## Summary: What Aletheion Achieves vs. What Remains

### ✅ Aletheion Achievements

1. **Reduced hallucinations**: Detects and prevents many overconfident errors
2. **Epistemic humility**: Model can abstain when uncertain
3. **Interpretable gating**: Q₁, Q₂, VARO scores are human-understandable
4. **Minimal overhead**: <1% computational cost
5. **Plug-and-play**: Can be added to pre-trained LLMs

### ❌ Remaining Limitations

1. **Still operates in embedding space**: No symbolic reasoning
2. **Detection, not solution**: Cannot solve problems beyond base LLM capability
3. **Fixed thresholds**: Not adaptive to context
4. **Data-dependent**: Requires quality training labels
5. **No knowledge retrieval**: Cannot access information outside training data
6. **No multi-step planning**: Cannot backtrack or search

### 🔬 Open Research Questions

1. How to integrate symbolic reasoning with neural LLMs?
2. Can we learn adaptive, context-dependent thresholds?
3. How to formalize "coherence" and "drift" beyond heuristics?
4. Can VARO be extended to capture long-range dependencies?
5. How to create large-scale labeled datasets for Q-training?

---

## References

### Limitations of Neural LLMs

1. **On the Dangers of Stochastic Parrots** (Bender et al., 2021)
   Fundamental critique of LLM limitations.

2. **Measuring Massive Multitask Language Understanding** (Hendrycks et al., 2021)
   MMLU benchmark showing knowledge gaps.

### NeuroSymbolic AI

3. **Neuro-Symbolic AI: The 3rd Wave** (Kautz, 2020)
   Vision for hybrid systems.

4. **Retrieval-Augmented Generation** (Lewis et al., 2020)
   Combining LLMs with external knowledge.

5. **Solving Math Word Problems with Program Synthesis** (Chiang & Chen, 2019)
   Symbolic execution for reasoning.

### Meta-Learning and Adaptation

6. **Model-Agnostic Meta-Learning (MAML)** (Finn et al., 2017)
   Learning to learn (adaptive thresholds).

7. **Adaptive Computation Time** (Graves, 2016)
   Dynamic resource allocation.

### Cross-References

- [← Training Strategy](./training-strategy.md) - How to train Aletheion
- [LLM Fundamentals](./llm-fundamentals.md) - Base architecture
- [LLM Failures](./llm-failures.md) - Why Aletheion is needed
- [Aletheion Integration](./aletheion-integration.md) - Where Aletheion hooks in

---

**Conclusion**: Aletheion is a **significant step forward** in epistemic reliability, but **not the final solution**. The path to truly reliable AI requires **hybrid neurosymbolic architectures** that combine the fluency of neural LLMs with the rigor of symbolic reasoning.

**Next steps**: Explore [Q-Systems repository](https://github.com/AletheionAGI/q-systems) for formal specifications and implementation.
