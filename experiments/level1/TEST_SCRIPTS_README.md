# Test Scripts for Pyramidal Value Propositions

This directory contains four comprehensive test scripts to validate the value propositions of the Pyramidal Epistemology architecture.

## Scripts Overview

### 1. `compare_models.py` - Model Comparison

Compare baseline vs pyramidal models side-by-side.

**Usage:**
```bash
python experiments/level1/compare_models.py \
    --baseline outputs/baseline/final \
    --pyramidal outputs/pyramidal/final \
    --output outputs/comparison
```

**Outputs:**
- `results.json`: Raw numerical results
- `report.md`: Markdown summary report
- `calibration_plots.png`: Reliability diagrams
- `perplexity_ece_scatter.png`: Perplexity vs ECE trade-off

**Key Metrics:**
- Perplexity comparison
- Expected Calibration Error (ECE)
- Brier score
- Statistical significance tests (bootstrap)

---

### 2. `test_out_of_domain.py` - Out-of-Domain Testing

Test calibration maintenance under domain shift.

**Hypothesis:** Pyramidal maintains calibration better on OOD data.

**Usage:**
```bash
# Test on WikiText-2
python experiments/level1/test_out_of_domain.py \
    --model outputs/pyramidal/final \
    --test-dataset wikitext2 \
    --output outputs/ood_test

# Test on custom text file
python experiments/level1/test_out_of_domain.py \
    --model outputs/pyramidal/final \
    --test-dataset custom \
    --custom-text-file data/custom.txt \
    --output outputs/ood_test
```

**Outputs:**
- `ood_results.json`: Raw numerical results
- `ood_report.md`: Domain shift analysis
- `ood_calibration.png`: In-domain vs OOD calibration
- `ood_uncertainty.png`: Uncertainty distribution shift (pyramidal only)

**Key Analysis:**
- Calibration degradation metrics
- Height/uncertainty shift on OOD data
- ECE change under domain shift

---

### 3. `visualize_epistemic.py` - Epistemic Visualization

Visualize height, Q1, Q2 evolution on specific examples.

**Usage:**
```bash
# With custom examples
python experiments/level1/visualize_epistemic.py \
    --model outputs/pyramidal/final \
    --examples examples.txt \
    --output outputs/epistemic_viz

# With default examples (validation set)
python experiments/level1/visualize_epistemic.py \
    --model outputs/pyramidal/final \
    --output outputs/epistemic_viz
```

**Example file format** (`examples.txt`):
```
The capital of France is Paris.
Quantum mechanics describes the behavior of matter and energy at
Once upon a time, there was a
```

**Outputs:**
- `height_progression.png`: Height evolution per example
- `q1_vs_q2_scatter.png`: Q1 vs Q2 scatter plot (colored by height)
- `force_weights_heatmap.png`: Force weights visualization
- `uncertainty_vs_error.png`: Uncertainty vs error correlation
- `knows_vs_doesnt_know.md`: Examples with detailed analysis
- `analyzed_examples.json`: Raw data

**Key Insights:**
- When model knows vs doesn't know
- Q1 (prediction quality) vs Q2 (confidence calibration)
- Force weight dynamics (memory, pain, choice, exploration)
- Uncertainty-error correlation

---

### 4. `test_abstention.py` - Selective Abstention

Test selective abstention using height as uncertainty signal.

**Usage:**
```bash
python experiments/level1/test_abstention.py \
    --model outputs/pyramidal/final \
    --threshold 0.3 \
    --output outputs/abstention_test
```

**Outputs:**
- `abstention_results.json`: Raw numerical results
- `abstention_report.md`: Summary with recommendations
- `abstention_curves.png`: Trade-off curves
- `abstention_examples.md`: Refused predictions with analysis

**Key Metrics:**
- Precision/Recall at different height thresholds
- Coverage vs accuracy trade-off
- Optimal threshold recommendation (e.g., for 80% coverage)
- F1 score optimization

**Use Cases:**
- Production systems requiring high accuracy (accept lower coverage)
- Safety-critical applications (abstain on uncertain predictions)
- Human-in-the-loop workflows (flag uncertain cases)

---

## Common Options

All scripts support:
- `--seed`: Random seed (default: 42)
- `--batch-size`: Batch size (default: 4)
- `--max-batches`: Max evaluation batches (default: 100)

## Example Workflow

### Step 1: Train models
```bash
# Train baseline
python experiments/level1/train_baseline.py \
    --steps 2000 \
    --output-dir outputs/baseline

# Train pyramidal
python experiments/level1/train_pyramidal.py \
    --steps 2000 \
    --output-dir outputs/pyramidal
```

### Step 2: Compare models
```bash
python experiments/level1/compare_models.py \
    --baseline outputs/baseline/final \
    --pyramidal outputs/pyramidal/final \
    --output outputs/comparison
```

### Step 3: Test OOD calibration
```bash
python experiments/level1/test_out_of_domain.py \
    --model outputs/pyramidal/final \
    --test-dataset wikitext2 \
    --output outputs/ood_test
```

### Step 4: Visualize epistemic metrics
```bash
python experiments/level1/visualize_epistemic.py \
    --model outputs/pyramidal/final \
    --output outputs/epistemic_viz
```

### Step 5: Test abstention
```bash
python experiments/level1/test_abstention.py \
    --model outputs/pyramidal/final \
    --threshold 0.3 \
    --output outputs/abstention_test
```

---

## Expected Results

### Pyramidal Value Propositions

1. **Better Calibration** (compare_models.py)
   - Lower ECE than baseline
   - Better Brier score
   - Maintained or improved perplexity

2. **OOD Robustness** (test_out_of_domain.py)
   - Calibration maintained under domain shift
   - Uncertainty increases appropriately on OOD data
   - Height metric signals distributional shift

3. **Epistemic Transparency** (visualize_epistemic.py)
   - Clear separation between "knows" and "doesn't know"
   - Q1/Q2 provide interpretable uncertainty signals
   - Force weights reveal decision-making dynamics

4. **Selective Prediction** (test_abstention.py)
   - Height enables effective abstention
   - Accuracy-coverage trade-off control
   - Optimal thresholds for different use cases

---

## Troubleshooting

### Import errors
Make sure you're running from the project root:
```bash
cd /path/to/aletheion-llm
python experiments/level1/compare_models.py ...
```

### Memory issues
Reduce batch size or max batches:
```bash
python experiments/level1/compare_models.py \
    --batch-size 2 \
    --max-batches 50 \
    ...
```

### Missing models
Train models first:
```bash
python experiments/level1/train_baseline.py --steps 2000
python experiments/level1/train_pyramidal.py --steps 2000
```

---

## Citation

If you use these test scripts, please cite:

```bibtex
@article{aletheion2024,
  title={Pyramidal Epistemology: Climbing Toward Truth in Neural Networks},
  author={AletheionAGI},
  year={2024}
}
```
