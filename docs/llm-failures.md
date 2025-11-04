# LLM Failure Modes: Critical Analysis

## Table of Contents
1. [Overview](#overview)
2. [Failure 1: Implicit Embedding Space](#failure-1-implicit-embedding-space)
3. [Failure 2: Stateless Processing](#failure-2-stateless-processing)
4. [Failure 3: No Metacognition](#failure-3-no-metacognition)
5. [Failure 4: No Post-Generation Verification](#failure-4-no-post-generation-verification)
6. [Failure 5: Overconfidence from RLHF](#failure-5-overconfidence-from-rlhf)
7. [Synthesis: Why These Failures Matter](#synthesis)
8. [References](#references)

---

## Overview

Despite impressive performance on many tasks, Large Language Models exhibit **fundamental architectural limitations** that cannot be solved by scaling alone. These failures stem from the core design of autoregressive transformers and their training paradigm.

This document provides a **mathematically rigorous analysis** of five critical failure modes, each with:
- Formal mathematical explanation
- Concrete code examples demonstrating the failure
- Implications for reliability and safety

---

## Failure 1: Implicit Embedding Space (No Symbolic Reasoning)

### 1.1 The Problem

LLMs operate entirely in **continuous embedding space** $\mathbb{R}^{d_{\text{model}}}$. They have no access to:
- Explicit symbolic representations
- Formal logical inference
- Structured knowledge graphs
- Ground truth verification mechanisms

### 1.2 Mathematical Explanation

Recall that at each layer $\ell$, the model transforms:

$$
\mathbf{h}_t^{(\ell)} = f_\ell(\mathbf{h}_t^{(\ell-1)}, \mathbf{h}_{<t}^{(\ell-1)}; \theta_\ell)
$$

where $\mathbf{h}_t^{(\ell)} \in \mathbb{R}^{d_{\text{model}}}$ is a **continuous vector**.

**The core issue**: The model approximates discrete concepts (like "prime number", "capital city", "valid proof") as **point clouds** in embedding space. There is no guarantee that:

1. **Consistency**: Similar embeddings represent semantically related concepts
2. **Compositionality**: Operations in embedding space correspond to logical operations
3. **Verifiability**: The model can check if its representation satisfies constraints

### 1.3 Formal Statement

Let $C$ be a symbolic concept (e.g., "x is a prime number"). The LLM learns an implicit embedding function:

$$
\phi: \text{Tokens}(C) \to \mathbb{R}^{d_{\text{model}}}
$$

But there is **no inverse** $\phi^{-1}$ to extract the symbolic representation, and **no oracle** to verify:

$$
\text{Verify}(C, x) \stackrel{?}{=} \text{True}
$$

Instead, the model must learn verification purely from training data patterns:

$$
P(\text{"yes"} \mid \text{"Is } x \text{ prime?"}) \approx \text{softmax}(\mathbf{z}_t)_{\text{"yes"}}
$$

This is fundamentally a **pattern matching** problem, not logical reasoning.

### 1.4 Code Example: Arithmetic Failure

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def test_arithmetic(model, tokenizer, problem):
    """Test LLM on arithmetic problems."""
    input_ids = tokenizer.encode(problem, return_tensors='pt')
    output_ids = model.generate(input_ids, max_length=input_ids.shape[1] + 10)
    return tokenizer.decode(output_ids[0])

# Simple problems (likely in training data)
print(test_arithmetic(model, tokenizer, "2 + 2 = "))
# Output: "2 + 2 = 4" ✓

# Slightly harder (might work)
print(test_arithmetic(model, tokenizer, "123 + 456 = "))
# Output: "123 + 456 = 579" ✓

# Large numbers (outside training distribution)
print(test_arithmetic(model, tokenizer, "87654321 + 12345679 = "))
# Output: "87654321 + 12345679 = 999999997" ✗ (correct: 100000000)

# Multi-step reasoning
print(test_arithmetic(model, tokenizer, "If x = 17, what is (x^2 - 1) / (x - 1)? "))
# Output: "If x = 17, what is (x^2 - 1) / (x - 1)? 16"
# Likely correct by pattern, but no proof it used algebra (x^2-1)/(x-1) = x+1
```

**Why this fails**:
- The model learns $f: \text{arithmetic\_string} \to \text{answer\_string}$ as a **statistical pattern**
- No internal symbol manipulator: $(a + b) \bmod 10$, carry bits, etc.
- Performance degrades on **out-of-distribution** numbers

### 1.5 Consequence: Hallucinated "Facts"

```python
def test_factual_knowledge(model, tokenizer, question):
    """Test factual recall."""
    input_ids = tokenizer.encode(question, return_tensors='pt')
    output_ids = model.generate(input_ids, max_length=input_ids.shape[1] + 20)
    return tokenizer.decode(output_ids[0])

# Common fact (in training data)
print(test_factual_knowledge(model, tokenizer, "The capital of France is "))
# Output: "The capital of France is Paris" ✓

# Obscure fact (maybe not in training data)
print(test_factual_knowledge(model, tokenizer, "The population of Tuvalu in 2023 is "))
# Output: "The population of Tuvalu in 2023 is approximately 11,000"
# Actual: ~11,300 (close, but no verification)

# Counterfactual (contradiction)
print(test_factual_knowledge(model, tokenizer,
    "In the alternate universe where France's capital is Lyon, the president lives in "))
# Output: "In the alternate universe where France's capital is Lyon, the president lives in Paris"
# Model fails to track counterfactual reasoning
```

**The model has no knowledge base** to check against. It only has:

$$
P(\text{answer} \mid \text{question}) \propto \exp(\mathbf{w}_{\text{answer}}^T \mathbf{h}_{\text{final}})
$$

---

## Failure 2: Stateless Processing (No Symbolic Memory)

### 2.1 The Problem

Each forward pass processes the **entire context** from scratch. There is no:
- Persistent memory across generations
- Explicit belief state
- Scratchpad for intermediate computations
- Working memory separate from context

### 2.2 Mathematical Explanation

At generation step $t$, the model computes:

$$
\mathbf{h}_t = \text{Transformer}(x_1, x_2, \ldots, x_{t-1})
$$

**Key insight**: The hidden states $\{\mathbf{h}_1, \ldots, \mathbf{h}_{t-1}\}$ from the previous forward pass are **discarded**. The model must:

1. Re-process the entire sequence $x_{1:t-1}$ (expensive)
2. Re-derive any intermediate computations (no memory)

There is no explicit **state update** mechanism like:

$$
\mathbf{s}_t = g(\mathbf{s}_{t-1}, x_t) \quad \text{(e.g., RNN/LSTM)}
$$

Instead, the context $x_{1:t-1}$ serves as the **implicit state**, but:
- **Limited capacity**: $O(t \cdot d_{\text{model}})$ grows linearly
- **No structure**: Cannot distinguish "working memory" from "long-term facts"
- **No updates**: Cannot revise beliefs without regenerating

### 2.3 Code Example: Multi-Step Reasoning

```python
def test_multi_step_reasoning(model, tokenizer, problem):
    """Test multi-step logical reasoning."""
    input_ids = tokenizer.encode(problem, return_tensors='pt')
    output_ids = model.generate(
        input_ids,
        max_length=input_ids.shape[1] + 100,
        do_sample=False
    )
    return tokenizer.decode(output_ids[0])

problem = """
Premise 1: All humans are mortal.
Premise 2: Socrates is a human.
Question: Is Socrates mortal? Think step by step.
"""

response = test_multi_step_reasoning(model, tokenizer, problem)
print(response)

# Typical output:
# "Premise 1: All humans are mortal.
#  Premise 2: Socrates is a human.
#  Question: Is Socrates mortal? Think step by step.
#
#  Yes, Socrates is mortal because..."

# Notice: No explicit reasoning trace!
# The model does not maintain a "belief state" like:
# State_0: {Premise1: ∀x (Human(x) → Mortal(x)), Premise2: Human(Socrates)}
# State_1: {Infer: Human(Socrates) ∧ (Human(Socrates) → Mortal(Socrates))}
# State_2: {Conclude: Mortal(Socrates)}
```

**What's missing**: A symbolic reasoner would maintain:

$$
\begin{align}
\text{KB}_0 &= \{\forall x. \text{Human}(x) \Rightarrow \text{Mortal}(x), \text{Human}(\text{Socrates})\} \\
\text{KB}_1 &= \text{KB}_0 \cup \{\text{Mortal}(\text{Socrates})\} \quad \text{(by Modus Ponens)}
\end{align}
$$

LLMs approximate this via **autoregressive chaining**:

$$
P(x_t \mid x_{<t}) \approx P(\text{reasoning\_step}_t \mid \text{reasoning\_step}_{<t})
$$

But:
1. **No backtracking**: If step $k$ is wrong, cannot revise without regenerating
2. **No tree search**: Cannot explore multiple reasoning paths
3. **No verification**: Cannot check if chain is logically valid

### 2.4 Consequence: Context Length Limitations

```python
# Simulate long context reasoning
long_problem = """
Fact 1: The key is in the red box.
Fact 2: The red box is in the kitchen.
Fact 3: The kitchen is on the second floor.
""" + "\n".join([f"Distractor {i}: The {color} {item} is in the {place}."
                  for i, (color, item, place) in enumerate([
                      ("blue", "pen", "drawer"),
                      ("green", "book", "shelf"),
                      # ... 1000 more distractors ...
                  ])]) + """
Question: Where is the key?
"""

# Model must maintain "key -> red box -> kitchen -> second floor"
# across 1000+ distractor sentences
# In practice: performance degrades as context fills up
```

**Fundamental limit**: Attention is $O(T^2)$ and embeddings are $O(T \cdot d)$, so:

$$
\text{Computational cost} = O(T^2 \cdot d_{\text{model}})
$$

and context capacity saturates at $T_{\max}$ (e.g., 4096, 8192, 32768 tokens).

---

## Failure 3: No Metacognition (Doesn't Know What It Doesn't Know)

### 3.1 The Problem

The softmax output $P(x_t \mid x_{<t}; \theta)$ represents **confidence**, but this is:
- Uncalibrated (high probability ≠ high accuracy)
- Unaware of epistemic uncertainty
- No mechanism to say "I don't know"

### 3.2 Mathematical Explanation

The model outputs a probability distribution:

$$
P(x_t = v \mid x_{<t}; \theta) = \frac{\exp(z_v)}{\sum_{v'} \exp(z_{v'})}
$$

**Entropy** measures uncertainty:

$$
H(P) = -\sum_{v \in \mathcal{V}} P(v \mid x_{<t}) \log P(v \mid x_{<t})
$$

- **High entropy**: $H(P) \approx \log V$ (uniform, maximum uncertainty)
- **Low entropy**: $H(P) \approx 0$ (peaked, confident)

**The problem**: Low entropy does **not** imply correctness!

#### Example: Confidently Wrong

```
Context: "The capital of Australia is Syd"
Logits: z_{"Sydney"} = 15.2, z_{"Canberra"} = 8.1, z_others ≈ -5
Probabilities: P("Sydney") = 0.9998, P("Canberra") = 0.0002

Model is HIGHLY confident, but WRONG!
```

This happens because:

$$
P(\text{token} \mid \text{context}) \neq P(\text{token is factually correct} \mid \text{context})
$$

The model learns to predict **plausible continuations** from training data, not **verified facts**.

### 3.3 Calibration Analysis

**Expected Calibration Error (ECE)**:

$$
\text{ECE} = \sum_{i=1}^{N_{\text{bins}}} \frac{|B_i|}{N} \left| \text{acc}(B_i) - \text{conf}(B_i) \right|
$$

where:
- $B_i$ = bin of predictions with confidence in $[(i-1)/N_{\text{bins}}, i/N_{\text{bins}})$
- $\text{acc}(B_i)$ = accuracy in bin $i$
- $\text{conf}(B_i)$ = average confidence in bin $i$

**Ideally**: $\text{ECE} \approx 0$ (confidence matches accuracy)

**In practice**: LLMs have ECE $\approx 0.1$ to $0.3$ (poorly calibrated)

### 3.4 Code Example: Overconfident Hallucinations

```python
import torch.nn.functional as F

def get_confidence_and_correctness(model, tokenizer, prompt, correct_answer):
    """
    Measure model confidence on a factual question.
    """
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, -1, :]  # Last token logits
        probs = F.softmax(logits, dim=-1)

    # Get top prediction
    top_token_id = torch.argmax(probs).item()
    top_token = tokenizer.decode([top_token_id])
    top_prob = probs[top_token_id].item()

    # Check correctness
    correct_token_id = tokenizer.encode(correct_answer)[0]
    correct_prob = probs[correct_token_id].item()

    is_correct = (top_token_id == correct_token_id)

    return {
        'predicted': top_token,
        'confidence': top_prob,
        'correct': is_correct,
        'correct_prob': correct_prob
    }

# Test on factual questions
questions = [
    ("The capital of Australia is ", "Canberra"),
    ("The tallest mountain on Earth is ", "Everest"),
    ("The chemical symbol for gold is ", "Au"),
]

for prompt, correct in questions:
    result = get_confidence_and_correctness(model, tokenizer, prompt, correct)
    print(f"Q: {prompt}")
    print(f"   Predicted: {result['predicted']} (confidence: {result['confidence']:.4f})")
    print(f"   Correct: {result['correct']} (true answer prob: {result['correct_prob']:.4f})")
    print()

# Typical output:
# Q: The capital of Australia is
#    Predicted: Sydney (confidence: 0.8734)  ← WRONG but confident!
#    Correct: False (true answer prob: 0.0521)
```

### 3.5 No Epistemic Uncertainty

The model cannot distinguish:
1. **Aleatoric uncertainty**: Multiple valid answers ("What's a good restaurant?")
2. **Epistemic uncertainty**: Unknown information ("What is the population of Mars colony in 2157?")

Both cases yield probability distributions $P(x_t \mid x_{<t})$, but:

- **Aleatoric**: $P$ has high entropy (many valid options)
- **Epistemic**: $P$ has low entropy but is WRONG (out-of-distribution query)

**A calibrated model would output**:

$$
P(\text{"I don't know"} \mid x_{<t}) > P(\text{specific\_answer} \mid x_{<t})
$$

But LLMs are trained to **always predict a token**, never abstain!

---

## Failure 4: No Post-Generation Verification

### 4.1 The Problem

Once a token $x_t$ is generated and appended to the context, it is **treated as ground truth** for subsequent predictions:

$$
P(x_{t+1} \mid x_1, \ldots, x_t; \theta)
$$

Even if $x_t$ is **wrong**, the model:
- Cannot detect the error
- Cannot backtrack
- Cannot revise its generation

This is the **autoregressive curse**.

### 4.2 Mathematical Explanation

Generation proceeds as a **Markov chain**:

$$
x_t \sim P(\cdot \mid x_{<t}; \theta), \quad t = 1, 2, \ldots, T
$$

The joint probability is:

$$
P(x_{1:T}) = \prod_{t=1}^{T} P(x_t \mid x_{<t})
$$

**Key insight**: If $x_k$ is incorrect, all subsequent tokens $x_{k+1}, \ldots, x_T$ are **conditioned on the error**.

$$
P(x_t \mid x_{<t}) = P(x_t \mid x_1, \ldots, x_{k-1}, \underbrace{x_k}_{\text{ERROR}}, x_{k+1}, \ldots, x_{t-1})
$$

The model has **no mechanism** to:

1. **Detect**: Identify that $x_k$ violates constraints
2. **Reject**: Resample $x_k$ from $P(\cdot \mid x_{<k})$
3. **Revise**: Rewrite $x_{k:t}$ to fix the error

### 4.3 Code Example: Error Cascades

```python
def demonstrate_error_cascade(model, tokenizer, prompt, inject_error_at=5):
    """
    Generate text, then inject an error and observe cascade.
    """
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Generate normally up to inject_error_at tokens
    output_ids = model.generate(
        input_ids,
        max_length=input_ids.shape[1] + inject_error_at,
        do_sample=False
    )

    print("=== Normal generation ===")
    print(tokenizer.decode(output_ids[0]))
    print()

    # Inject an error (replace token at position)
    error_token_id = tokenizer.encode("ERRORERROR")[0]
    output_ids[0, input_ids.shape[1] + inject_error_at // 2] = error_token_id

    # Continue generation from error
    continued_ids = model.generate(
        output_ids,
        max_length=output_ids.shape[1] + inject_error_at,
        do_sample=False
    )

    print("=== After injecting error ===")
    print(tokenizer.decode(continued_ids[0]))
    print()

prompt = "The process of photosynthesis involves"
demonstrate_error_cascade(model, tokenizer, prompt)

# Typical output:
# === Normal generation ===
# The process of photosynthesis involves converting light energy into chemical energy.
#
# === After injecting error ===
# The process of ERRORERROR involves... [model tries to make sense of nonsense]
# → Generation becomes incoherent after error!
```

### 4.4 Consequence: No Self-Correction

Consider multi-step problem solving:

```
Step 1: Calculate 17 + 25 = 42  ← CORRECT
Step 2: Multiply by 3: 42 × 3 = 124  ← ERROR (should be 126)
Step 3: Subtract 10: 124 - 10 = 114  ← Correct arithmetic, but propagated error
```

A human would:
1. Check step 2: $42 \times 3 = 126 \neq 124$
2. Revise: "Wait, let me recalculate..."
3. Correct: $42 \times 3 = 126$, then $126 - 10 = 116$

An LLM:
1. Generates "124" with $P(\text{"124"} \mid \text{context}) = 0.43$ (not highest!)
2. **But** token is sampled, so it's committed
3. All subsequent steps use "124" as truth

**No verification layer**:

$$
\text{Verify}(x_t, x_{<t}) \stackrel{?}{=} \text{True}
$$

---

## Failure 5: Overconfidence from RLHF

### 5.1 The Problem

Modern LLMs are trained with **Reinforcement Learning from Human Feedback (RLHF)**, which:
- Optimizes for **human preference** (helpfulness, harmlessness, honesty)
- Encourages **confident, coherent responses**
- Penalizes **hedging, uncertainty, or refusal**

This exacerbates overconfidence.

### 5.2 Mathematical Explanation

After pre-training with loss $\mathcal{L}_{\text{LM}}$, RLHF fine-tunes via:

$$
\mathcal{L}_{\text{RLHF}} = -\mathbb{E}_{x \sim P_\theta} \left[ r_\phi(x, y) - \beta \cdot D_{\text{KL}}(P_\theta \| P_{\text{ref}}) \right]
$$

where:
- $r_\phi(x, y)$: **reward model** (trained on human preferences)
- $P_{\text{ref}}$: pre-trained model (reference)
- $\beta$: KL penalty (prevent drift from pre-trained distribution)

**The issue**: Reward model $r_\phi$ is trained on **comparisons**:

$$
r_\phi(x, y_1) > r_\phi(x, y_2) \iff \text{humans prefer } y_1 \text{ over } y_2
$$

Humans prefer:
- **Confident** answers over hedged ones
- **Detailed** explanations over "I don't know"
- **Coherent** narratives over admitting uncertainty

**Result**: The policy $P_\theta$ learns:

$$
\arg\max_\theta \mathbb{E}_{x, y \sim P_\theta} [r_\phi(x, y)] \implies \text{confident, detailed responses}
$$

Even when the model is **uncertain**!

### 5.3 Code Example: Before vs. After RLHF

```python
# Simulate pre-RLHF (base model) vs. post-RLHF (instruct model)

def test_base_vs_instruct(prompt):
    """
    Compare base model and RLHF-tuned model on uncertain question.
    """
    # Base model (pre-training only)
    base_model = GPT2LMHeadModel.from_pretrained('gpt2')
    base_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Instruct model (RLHF-tuned)
    # instruct_model = load("gpt-3.5-turbo")  # Hypothetical

    # Test on uncertain question
    uncertain_prompt = "What is the population of Mars colony in 2157?"

    # Base model response (more uncertain)
    base_ids = base_tokenizer.encode(uncertain_prompt, return_tensors='pt')
    base_output = base_model.generate(base_ids, max_length=100, do_sample=True, temperature=1.0)
    base_response = base_tokenizer.decode(base_output[0])
    print("Base model:", base_response)
    # Typical: "What is the population of Mars colony in 2157? [rambling, incoherent]"

    # Instruct model response (more confident, but fabricated!)
    # instruct_response = instruct_model.generate(uncertain_prompt)
    # print("Instruct model:", instruct_response)
    # Typical: "The population of Mars colony in 2157 is projected to be around
    #           2.3 million, based on current colonization plans..."
    #           ← Fabricated with high confidence!

test_base_vs_instruct("What is the population of Mars colony in 2157?")
```

### 5.4 Preference Optimization Amplifies Overconfidence

**DPO (Direct Preference Optimization)** loss:

$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\left( \beta \log \frac{P_\theta(y_w \mid x)}{P_{\text{ref}}(y_w \mid x)} - \beta \log \frac{P_\theta(y_l \mid x)}{P_{\text{ref}}(y_l \mid x)} \right) \right]
$$

where:
- $y_w$: **preferred** (winning) response
- $y_l$: **dispreferred** (losing) response

If humans consistently prefer confident answers, the model learns:

$$
P_\theta(\text{confident\_answer} \mid x) \gg P_\theta(\text{"I don't know"} \mid x)
$$

**Consequence**: Model hallucinates with high confidence rather than admitting uncertainty.

### 5.5 Calibration Degradation

Studies show:
- **Pre-training**: ECE $\approx 0.15$
- **After RLHF**: ECE $\approx 0.25$ to $0.35$ (worse calibration!)

RLHF increases **superficial helpfulness** at the cost of **epistemic humility**.

---

## Synthesis: Why These Failures Matter

### Compounding Effects

These five failures are **not independent**. They compound:

```
┌─────────────────────────────────────────────────────────────┐
│  Implicit Embeddings (Failure 1)                            │
│  → No symbolic reasoning                                     │
│  → Cannot formally verify facts                              │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  Stateless Processing (Failure 2)                           │
│  → No persistent memory                                      │
│  → Cannot track belief state                                 │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  No Metacognition (Failure 3)                               │
│  → Doesn't know when uncertain                               │
│  → Overconfident predictions                                 │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  No Verification (Failure 4)                                │
│  → Errors propagate autoregressively                         │
│  → No self-correction                                        │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  RLHF Overconfidence (Failure 5)                            │
│  → Trained to avoid uncertainty                              │
│  → Hallucinations packaged as facts                          │
└─────────────────────────────────────────────────────────────┘
                 │
                 ▼
       ╔═══════════════════════╗
       ║ UNRELIABLE GENERATION ║
       ╚═══════════════════════╝
```

### Critical Implication

Standard LLMs are **fluent but not rigorous**. They excel at:
- Pattern completion
- Surface coherence
- Mimicking human text

But fail at:
- Logical consistency
- Factual grounding
- Self-verification
- Epistemic humility

### Why This Matters for Aletheion

These five failures motivate the **Aletheion architecture**:

1. **Q₁ (Coherence)**: Detects implicit embedding inconsistencies
2. **Q₂ (Drift)**: Monitors stateless accumulation of errors
3. **VARO (Anti-resonance)**: Prevents overconfident cascades
4. **Gating**: Provides post-generation verification layer
5. **Adaptive thresholds**: Restores epistemic humility

See [aletheion-integration.md](./aletheion-integration.md) for how Aletheion addresses these failures.

---

## References

### Failure Mode Research

1. **On the Dangers of Stochastic Parrots** (Bender et al., 2021)
   Systematic analysis of LLM limitations.

2. **Hallucination in Neural NLG** (Ji et al., 2023)
   Survey of hallucination types and causes.

3. **Language Models (Mostly) Know What They Know** (Kadavath et al., 2022)
   Calibration analysis of LLMs.

4. **Training Verifiers to Solve Math Word Problems** (Cobbe et al., 2021)
   Demonstrates need for external verification.

### RLHF and Overconfidence

5. **Training Language Models to Follow Instructions with Human Feedback** (Ouyang et al., 2022)
   InstructGPT / ChatGPT methodology.

6. **Direct Preference Optimization** (Rafailov et al., 2023)
   Simplified RLHF without reward model.

7. **Calibration of Pre-trained Transformers** (Desai & Durrett, 2020)
   Analysis of confidence calibration.

### Cross-References

- [← LLM Fundamentals](./llm-fundamentals.md) - Mathematical foundation
- [Aletheion Integration →](./aletheion-integration.md) - How Aletheion addresses these failures

---

**Next**: [Aletheion Integration →](./aletheion-integration.md)
