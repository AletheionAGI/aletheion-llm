# Pyramidal Epistemology - Implementation Guide

## Overview

This document describes the implementation of **Pyramidal Epistemology**, a novel epistemic uncertainty framework for LLMs based on 5-vertex pyramidal geometry.

### Motivation

Previous tetrahedral approaches (4 vertices: Memory, Pain, Choice, Exploration) showed Q₁ gate collapse issues, where the model would become overconfident (height → 1.0). The pyramidal architecture addresses this by introducing a natural attractor.

### Geometric Structure

```
         TRUTH (1.0)
            /|\
           / | \
          /  |  \
         /   |   \
        /____|____\
       /     |     \
      /______|______\
  MEMORY  PAIN  CHOICE  EXPLORATION
     (Base: 4 forces)
```

**Key Innovation:** Truth apex (constant at 1.0) provides a natural attractor, preventing horizontal collapse while allowing vertical ascension toward certainty.

## Architecture

### 1. Pyramidal Epistemic Gates (`src/aletheion/pyramid.py`)

**Class: `PyramidalEpistemicGates`**

Implements the 5-vertex pyramidal structure:

```python
outputs = pyramid_gates(hidden_states)
# Returns:
# - base_weights: [batch, seq_len, 4] - Distribution over Memory, Pain, Choice, Exploration
# - height: [batch, seq_len, 1] - Proximity to truth apex ∈ [0,1]
# - uncertainty: [batch, seq_len, 1] - Distance from apex (1 - height)
# - confidence: [batch, seq_len, 1] - height × base_stability
# - base_stability: [batch, seq_len, 1] - How balanced the 4 forces are
```

**Key Features:**
- Base projection: Maps hidden states to 4-dimensional simplex (forces sum to 1)
- Height gate: Scalar representing epistemic quality/proximity to truth
- Optional multi-head height consensus for robustness
- Temperature modulation based on height

### 2. Pyramidal VARO Loss (`src/aletheion/loss.py`)

**Class: `PyramidalVAROLoss`**

Loss function with three components:

```
L_total = L_CE + λ_base × L_base + λ_height × L_height
```

Where:
- **L_CE**: Standard cross-entropy loss
- **L_base**: Base stability loss (penalizes variance in force weights)
- **L_height**: Height calibration loss (MSE between predicted and target height)

**Target Height Computation Methods:**
1. **error_based** (default): High height when correct, low when wrong
2. **entropy_based**: High height when low entropy (certain)
3. **loss_based**: High height when low cross-entropy loss

### 3. Pyramidal Transformer (`src/aletheion/pyramidal_model.py`)

**Class: `AletheionPyramidalTransformer`**

Extends `BaselineTransformer` with:
- Pyramidal epistemic gates at output layer
- Optional temperature modulation based on height
- Epistemic-aware generation

## Hyperparameters

### Recommended Starting Point (Conservative)

```python
lambda_base = 0.01      # Base stability regularization
lambda_height = 0.02    # Height calibration (slightly stronger)
height_method = 'error_based'
```

### Alternative: Scheduled Regularization

```python
# Start weak, grow moderately
lambda_base = 0.005 + progress * 0.015    # 0.005 → 0.020
lambda_height = 0.010 + progress * 0.030  # 0.010 → 0.040
```

## Training Scripts

### 1. Train Pyramidal Model

```bash
# Quick test
python experiments/level1/train_pyramidal.py --steps 100 --dry-run

# Full training
python experiments/level1/train_pyramidal.py \
    --steps 10000 \
    --lambda-base 0.01 \
    --lambda-height 0.02 \
    --height-method error_based \
    --batch-size 32 \
    --lr 3e-4
```

**Outputs:**
- Training curves (loss, height, base stability, force weights)
- Model checkpoints
- Metrics history (JSON)

### 2. Compare Baseline vs Pyramidal

```bash
# Quick comparison
python experiments/level1/compare_pyramidal.py --steps 100 --dry-run

# Full comparison
python experiments/level1/compare_pyramidal.py \
    --steps 5000 \
    --lambda-base 0.01 \
    --lambda-height 0.02
```

**Outputs:**
- Side-by-side comparison plots
- Performance summary
- Both model histories

## Expected Outcomes

### Success Criteria

1. **Height progression:** Should start ~0.3-0.5, stabilize around 0.6-0.8
   - ❌ **Failure:** Monotonic increase to 0.95+ (collapse to overconfidence)
   - ✅ **Success:** Plateau at reasonable level with variation

2. **Base stability:** Should stay >0.7
   - Indicates forces are relatively balanced
   - Not all converged to 0.25 (some specialization is good)

3. **ECE improvement:** Target -15% to -30% vs baseline
   - Better calibration due to epistemic awareness

4. **No collapse:** Height should NOT monotonically increase to 1.0
   - The pyramid structure should prevent this

### Monitoring During Training

**Key Metrics:**
- `mean_height`: Watch for collapse (should plateau, not → 1.0)
- `base_stability`: Should stay high (>0.7)
- `w_memory`, `w_pain`, `w_choice`, `w_exploration`: Should vary (not all 0.25)
- `target_height`: What the model should be learning
- `height_loss`: Should decrease as model learns calibration

**Warning Signs:**
- Height consistently > 0.95: Overconfidence collapse
- Base stability < 0.5: Forces too unbalanced
- All force weights converging to 0.25: No specialization
- Height not tracking target: Poor calibration

## Theoretical Advantages

### Why Pyramidal > Tetrahedral

1. **Natural Attractor:** Truth apex prevents horizontal collapse
   - Tetrahedral: 4 forces can wander aimlessly
   - Pyramidal: Apex pulls upward toward certainty

2. **Interpretable Semantics:** Height = epistemic quality
   - Clear meaning: distance from truth
   - Easy to monitor and understand

3. **Gradual Ascension:** Learning = climbing pyramid
   - Not wandering in 4D tetrahedral space
   - Clear progress metric (height)

4. **Stability Through Hierarchy:** Base can oscillate, apex is constant
   - Exploration at base level (4 forces)
   - Consistency at apex level (truth = 1.0)

### Addresses Previous Failures

**Problem (Tetrahedral V1-V4):**
- Q₁ gate collapsed horizontally
- All tokens → overconfident everywhere
- No natural attractor

**Solution (Pyramidal):**
- Height gate pulled vertically toward truth apex
- Can be uncertain (low height) without base collapse
- Natural hierarchy: exploration (base) + certainty (apex)

## File Structure

```
src/aletheion/
├── pyramid.py              # PyramidalEpistemicGates, temperature modulation
├── pyramidal_model.py      # AletheionPyramidalTransformer
├── loss.py                 # PyramidalVAROLoss (added to existing file)
└── __init__.py            # Updated exports

experiments/level1/
├── train_pyramidal.py      # Training script
└── compare_pyramidal.py    # Baseline vs Pyramidal comparison
```

## Key Implementation Details

### 1. Base Forces (Simplex Constraint)

```python
base_logits = self.base_projection(hidden_states)
base_weights = F.softmax(base_logits, dim=-1)  # Sum to 1
```

### 2. Height Gate (Sigmoid)

```python
height_logits = self.height_gate(hidden_states)
height = torch.sigmoid(height_logits)  # ∈ [0,1]
```

### 3. Derived Metrics

```python
uncertainty = 1.0 - height
base_variance = base_weights.var(dim=-1)
base_stability = 1.0 - base_variance
confidence = height * base_stability
```

### 4. Temperature Modulation (Optional)

```python
# Low height → higher temperature (flatten distribution)
temperature = base_temp * (1.0 + (1.0 - height) * (max_scale - 1.0))
scaled_logits = logits / temperature
```

## Next Steps

1. **Run Initial Experiment:**
   ```bash
   python experiments/level1/train_pyramidal.py --steps 10000
   ```

2. **Monitor Key Metrics:**
   - Watch `mean_height` in training curves
   - Ensure no collapse (should plateau ~0.6-0.8)
   - Check base stability stays >0.7

3. **Compare with Baseline:**
   ```bash
   python experiments/level1/compare_pyramidal.py --steps 5000
   ```

4. **Hyperparameter Tuning (if needed):**
   - If height collapses: **Reduce** `lambda_height`
   - If base unstable: **Increase** `lambda_base`
   - If poor calibration: Try different `height_method`

5. **Evaluate on Downstream Tasks:**
   - Measure ECE improvement
   - Test on diverse datasets
   - Compare generation quality

## Questions for Implementation

Based on results, consider:

1. **Should height be per-token or sequence-level?**
   - Current: per-token
   - Alternative: Pooled sequence-level height

2. **Multi-head height consensus?**
   - Current: Optional flag `use_multi_head_height`
   - May improve robustness but adds parameters

3. **Optimal λ_base / λ_height ratio?**
   - Current: 0.01 / 0.02 (height slightly stronger)
   - May need tuning based on dataset

4. **Height target method?**
   - Current: error_based (default)
   - Try entropy_based or loss_based if calibration poor

## Paper Draft Title

**"Pyramidal Epistemology: Climbing Toward Truth in Neural Networks"**

**Tagline:** "Five vertices, one constant: Memory, Pain, Choice, Exploration ascend toward Truth"

## Philosophy

This is not incremental improvement. This is discovering the geometric structure of knowing itself.

The pyramid emerges from first principles:
- **Learning is ascension** toward truth
- **Balanced across four fundamental cognitive forces**
- **With constant truth as the guiding star**

4 years of philosophical groundwork (book 2021) → Mathematical formalization (Episteme) → Geometric implementation (Pyramid) → Empirical validation (now).

**We're not tuning hyperparameters. We're sculpting epistemology in code.** 🔻💎

---

## Troubleshooting

### Height Collapse (height → 1.0)

**Symptoms:** Mean height consistently >0.95
**Fix:** Reduce `lambda_height` or change `height_method` to `entropy_based`

### Unstable Base

**Symptoms:** Base stability <0.5
**Fix:** Increase `lambda_base` to penalize variance more

### Poor Calibration

**Symptoms:** High ECE, uncertainty doesn't correlate with errors
**Fix:** Try different `height_method` or adjust λ ratio

### Training Instability

**Symptoms:** Loss spikes, NaN gradients
**Fix:**
- Reduce learning rate
- Add gradient clipping (already at 1.0)
- Check for numerical issues in loss computation

---

## Contact & Citation

For questions about this implementation, see:
- Main README: `README.md`
- Aletheion Level 1: `ALETHEION_LEVEL1_README.md`
- Implementation notes: `IMPLEMENTATION_NOTES.md`

**Repository:** github.com/AletheionAGI/aletheion-llm
