# How to Progress: Implementation Guide
**Date:** 2025-11-08
**Version:** 1.0
**Status:** Active Development Guide

---

## 📚 Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Task 1: Level 1 Validation (50% → 100%)](#task-1-level-1-validation-50--100)
4. [Task 2: Performance Optimization (0% → 100%)](#task-2-performance-optimization-0--100)
5. [Task 3: Extended Benchmarking (25% → 100%)](#task-3-extended-benchmarking-25--100)
6. [Troubleshooting](#troubleshooting)
7. [Checklist](#checklist)

---

## Overview

This document provides **step-by-step instructions** for completing the three "In Progress" tasks from the README.md roadmap:

- ✅ **Task 1:** Level 1 validation results (50% complete → 100%)
- ✅ **Task 2:** Performance optimization (0% → 100%)
- ✅ **Task 3:** Extended benchmarking (25% → 100%)

Each task is broken down into **actionable subtasks** with commands, code snippets, and expected outputs.

---

## Prerequisites

### System Requirements

```bash
# Verify your environment
python --version          # Python 3.10+
python -c "import torch; print(torch.__version__)"  # PyTorch 2.0+
python -c "import torch; print(torch.cuda.is_available())"  # Should be True

# Verify installation
pip list | grep -E "torch|transformers|datasets"

# Verify project installation
python -c "from src.aletheion.model import AletheionTransformer; print('✓ OK')"
```

### Compute Resources

**Required:**
- GPU with ≥ 8GB VRAM (preferably 16GB+)
- ~100GB disk space for datasets and checkpoints
- ~80 GPU-days total compute (can parallelize)

**Recommended:**
- Multiple GPUs for parallel runs
- Fast SSD for dataset loading
- Good internet connection for dataset downloads

### Setup Workspace

```bash
# Navigate to project root
cd /path/to/aletheion-llm

# Create directories for outputs
mkdir -p outputs/validation/{seed_42,seed_123,seed_456,seed_789,seed_999}
mkdir -p outputs/benchmarks/{wikitext103,penn_treebank,pile}
mkdir -p outputs/ablations/{varo_sweep,threshold_sweep,height_sweep,base_sweep}
mkdir -p outputs/visualizations/{reliability_diagrams,uncertainty_dist,gate_heatmaps}
mkdir -p outputs/profiling/

# Verify directory structure
tree outputs/ -L 2
```

---

## Task 1: Level 1 Validation (50% → 100%)

**Goal:** Complete statistical validation of Level 1 Aletheion architecture

**Current Status:** 50% (initial training complete, need reproducibility and evaluation)

**Estimated Time:** 3 weeks

**Estimated Compute:** ~50 GPU-days

---

### Step 1.1: Multi-Seed Validation (Critical!)

**Purpose:** Validate that results are reproducible and not a lucky random seed.

**Time:** 5 GPU-days (can parallelize)

#### 1.1.1 Launch Training Runs

```bash
# Seed 1: 42
CUDA_VISIBLE_DEVICES=0 python experiments/level1/train_pyramidal_q1q2.py \
  --config config/aletheion_level1.yaml \
  --output outputs/validation/seed_42 \
  --seed 42 \
  --wandb_run_name "aletheion_level1_seed42" \
  > logs/seed_42.log 2>&1 &

# Seed 2: 123
CUDA_VISIBLE_DEVICES=1 python experiments/level1/train_pyramidal_q1q2.py \
  --config config/aletheion_level1.yaml \
  --output outputs/validation/seed_123 \
  --seed 123 \
  --wandb_run_name "aletheion_level1_seed123" \
  > logs/seed_123.log 2>&1 &

# Seed 3: 456
CUDA_VISIBLE_DEVICES=2 python experiments/level1/train_pyramidal_q1q2.py \
  --config config/aletheion_level1.yaml \
  --output outputs/validation/seed_456 \
  --seed 456 \
  --wandb_run_name "aletheion_level1_seed456" \
  > logs/seed_456.log 2>&1 &

# Seed 4: 789
CUDA_VISIBLE_DEVICES=3 python experiments/level1/train_pyramidal_q1q2.py \
  --config config/aletheion_level1.yaml \
  --output outputs/validation/seed_789 \
  --seed 789 \
  --wandb_run_name "aletheion_level1_seed789" \
  > logs/seed_789.log 2>&1 &

# Seed 5: 999
CUDA_VISIBLE_DEVICES=4 python experiments/level1/train_pyramidal_q1q2.py \
  --config config/aletheion_level1.yaml \
  --output outputs/validation/seed_999 \
  --seed 999 \
  --wandb_run_name "aletheion_level1_seed999" \
  > logs/seed_999.log 2>&1 &

# Monitor progress
tail -f logs/seed_42.log
watch -n 30 'nvidia-smi'
```

**If you only have 1 GPU:** Run sequentially:
```bash
for seed in 42 123 456 789 999; do
  echo "Training seed $seed..."
  python experiments/level1/train_pyramidal_q1q2.py \
    --config config/aletheion_level1.yaml \
    --output outputs/validation/seed_$seed \
    --seed $seed \
    --wandb_run_name "aletheion_level1_seed$seed"
done
```

#### 1.1.2 Create Analysis Script

Create `scripts/analyze_multiseed.py`:

```python
#!/usr/bin/env python3
"""Analyze multi-seed validation results."""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_metrics(seed_dir: Path) -> dict:
    """Load final metrics from a seed run."""
    metrics_file = seed_dir / "metrics.json"
    if not metrics_file.exists():
        raise FileNotFoundError(f"No metrics found for {seed_dir}")

    with open(metrics_file, 'r') as f:
        return json.load(f)

def analyze_seeds(output_dir: Path, seeds: list[int]) -> pd.DataFrame:
    """Aggregate metrics across seeds."""
    results = []

    for seed in seeds:
        seed_dir = output_dir / f"seed_{seed}"
        try:
            metrics = load_metrics(seed_dir)
            metrics['seed'] = seed
            results.append(metrics)
        except FileNotFoundError:
            print(f"⚠️  Warning: Seed {seed} not found, skipping")

    df = pd.DataFrame(results)
    return df

def compute_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean ± std for all metrics."""
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != 'seed']

    stats = pd.DataFrame({
        'metric': numeric_cols,
        'mean': [df[col].mean() for col in numeric_cols],
        'std': [df[col].std() for col in numeric_cols],
        'min': [df[col].min() for col in numeric_cols],
        'max': [df[col].max() for col in numeric_cols],
    })

    stats['ci_95'] = 1.96 * stats['std'] / np.sqrt(len(df))
    stats['cv'] = (stats['std'] / stats['mean']) * 100  # Coefficient of variation

    return stats

def visualize_variance(df: pd.DataFrame, output_path: Path):
    """Create visualizations of metric variance."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # ECE distribution
    axes[0, 0].hist(df['ece'], bins=10, edgecolor='black')
    axes[0, 0].axvline(df['ece'].mean(), color='red', linestyle='--',
                       label=f"Mean: {df['ece'].mean():.4f}")
    axes[0, 0].set_xlabel('ECE')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('ECE Distribution Across Seeds')
    axes[0, 0].legend()

    # Perplexity distribution
    axes[0, 1].hist(df['perplexity'], bins=10, edgecolor='black')
    axes[0, 1].axvline(df['perplexity'].mean(), color='red', linestyle='--',
                       label=f"Mean: {df['perplexity'].mean():.1f}")
    axes[0, 1].set_xlabel('Perplexity')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Perplexity Distribution Across Seeds')
    axes[0, 1].legend()

    # Q1/Q2 mean values
    if 'q1_mean' in df.columns and 'q2_mean' in df.columns:
        axes[1, 0].scatter(df['q1_mean'], df['q2_mean'], s=100)
        axes[1, 0].set_xlabel('Q₁ Mean')
        axes[1, 0].set_ylabel('Q₂ Mean')
        axes[1, 0].set_title('Q₁ vs Q₂ Convergence')
        axes[1, 0].grid(True)

    # Height values
    if 'height_mean' in df.columns:
        axes[1, 1].hist(df['height_mean'], bins=10, edgecolor='black')
        axes[1, 1].axvline(df['height_mean'].mean(), color='red', linestyle='--',
                          label=f"Mean: {df['height_mean'].mean():.3f}")
        axes[1, 1].set_xlabel('Pyramidal Height')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Height Distribution Across Seeds')
        axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization to {output_path}")

def main():
    output_dir = Path("outputs/validation")
    seeds = [42, 123, 456, 789, 999]

    print("=" * 60)
    print("MULTI-SEED VALIDATION ANALYSIS")
    print("=" * 60)

    # Load all results
    df = analyze_seeds(output_dir, seeds)
    print(f"\n✓ Loaded {len(df)} seed results\n")

    # Compute statistics
    stats = compute_statistics(df)

    # Display results
    print("SUMMARY STATISTICS:")
    print("-" * 60)
    print(stats.to_string(index=False))
    print("-" * 60)

    # Check reproducibility
    print("\nREPRODUCIBILITY ASSESSMENT:")
    print("-" * 60)

    ece_cv = stats[stats['metric'] == 'ece']['cv'].values[0]
    perplexity_cv = stats[stats['metric'] == 'perplexity']['cv'].values[0]

    print(f"ECE Coefficient of Variation: {ece_cv:.2f}%")
    if ece_cv < 10:
        print("  ✓ EXCELLENT reproducibility (CV < 10%)")
    elif ece_cv < 20:
        print("  ✓ GOOD reproducibility (CV < 20%)")
    else:
        print("  ⚠️  HIGH variance (CV > 20%)")

    print(f"\nPerplexity Coefficient of Variation: {perplexity_cv:.2f}%")
    if perplexity_cv < 5:
        print("  ✓ EXCELLENT reproducibility (CV < 5%)")
    elif perplexity_cv < 10:
        print("  ✓ GOOD reproducibility (CV < 10%)")
    else:
        print("  ⚠️  HIGH variance (CV > 10%)")

    # Save results
    stats.to_csv(output_dir / "multiseed_statistics.csv", index=False)
    df.to_csv(output_dir / "multiseed_raw_data.csv", index=False)

    # Generate visualizations
    visualize_variance(df, output_dir / "multiseed_variance.png")

    print("\n" + "=" * 60)
    print("✓ Analysis complete!")
    print(f"  - Statistics: {output_dir}/multiseed_statistics.csv")
    print(f"  - Raw data: {output_dir}/multiseed_raw_data.csv")
    print(f"  - Visualization: {output_dir}/multiseed_variance.png")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

#### 1.1.3 Run Analysis

```bash
# Make executable
chmod +x scripts/analyze_multiseed.py

# Run analysis
python scripts/analyze_multiseed.py

# Expected output:
# ✓ EXCELLENT reproducibility (ECE CV < 10%)
# ✓ GOOD reproducibility (Perplexity CV < 10%)
```

#### 1.1.4 Success Criteria

- ✅ ECE mean ≈ 0.011 (± 0.002 acceptable)
- ✅ ECE coefficient of variation < 20%
- ✅ Perplexity coefficient of variation < 10%
- ✅ All 5 seeds converge without NaN/divergence

---

### Step 1.2: Reliability Diagrams

**Purpose:** Visualize calibration quality (predicted confidence vs actual accuracy)

**Time:** 2-3 days

#### 1.2.1 Create Reliability Diagram Script

Create `scripts/generate_reliability_diagram.py`:

```python
#!/usr/bin/env python3
"""Generate reliability diagrams for calibration visualization."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple
from src.aletheion.model import AletheionTransformer
from src.model import BaselineTransformer
from data.dataset import load_wikitext_dataset
from torch.utils.data import DataLoader

def compute_calibration_curve(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute calibration curve for reliability diagram."""

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Find predictions in this bin
        in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        if i == n_bins - 1:  # Last bin includes upper boundary
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)

        bin_count = in_bin.sum()

        if bin_count > 0:
            bin_conf = confidences[in_bin].mean()
            bin_acc = accuracies[in_bin].mean()
        else:
            bin_conf = (bin_lower + bin_upper) / 2
            bin_acc = 0.0

        bin_confidences.append(bin_conf)
        bin_accuracies.append(bin_acc)
        bin_counts.append(bin_count)

    return (
        np.array(bin_confidences),
        np.array(bin_accuracies),
        np.array(bin_counts)
    )

def evaluate_model_calibration(
    model,
    data_loader,
    device: str = 'cuda'
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate model and collect confidence/accuracy pairs."""

    model.eval()
    confidences = []
    accuracies = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids)
            logits = outputs['logits']

            # Get predictions
            probs = torch.softmax(logits, dim=-1)
            pred_conf, pred_ids = probs.max(dim=-1)

            # Compute accuracy
            correct = (pred_ids == labels).float()

            # Collect
            confidences.extend(pred_conf.cpu().numpy().flatten())
            accuracies.extend(correct.cpu().numpy().flatten())

    return np.array(confidences), np.array(accuracies)

def plot_reliability_diagram(
    baseline_conf, baseline_acc,
    aletheion_conf, aletheion_acc,
    output_path: Path,
    n_bins: int = 10
):
    """Create reliability diagram comparing baseline vs Aletheion."""

    # Compute calibration curves
    baseline_bin_conf, baseline_bin_acc, baseline_counts = \
        compute_calibration_curve(baseline_conf, baseline_acc, n_bins)

    aletheion_bin_conf, aletheion_bin_acc, aletheion_counts = \
        compute_calibration_curve(aletheion_conf, aletheion_acc, n_bins)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')

    # Baseline
    ax.plot(baseline_bin_conf, baseline_bin_acc, 'o-',
            linewidth=2, markersize=8, label='Baseline', color='C0')

    # Aletheion
    ax.plot(aletheion_bin_conf, aletheion_bin_acc, 's-',
            linewidth=2, markersize=8, label='Aletheion Level 1', color='C1')

    # Fill gap (calibration error)
    ax.fill_between([0, 1], [0, 1], [0, 1], alpha=0.1, color='gray')

    # Styling
    ax.set_xlabel('Confidence', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title('Reliability Diagram: Calibration Comparison', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Add ECE values
    baseline_ece = np.abs(baseline_bin_acc - baseline_bin_conf).mean()
    aletheion_ece = np.abs(aletheion_bin_acc - aletheion_bin_conf).mean()

    ax.text(0.05, 0.95,
            f"Baseline ECE: {baseline_ece:.4f}\nAletheion ECE: {aletheion_ece:.4f}",
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved reliability diagram to {output_path}")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load validation dataset
    _, val_dataset, _, tokenizer = load_wikitext_dataset()
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print("Loading models...")

    # Load baseline model
    baseline_checkpoint = torch.load("outputs/baseline/checkpoint_final.pt")
    baseline_model = BaselineTransformer(**baseline_checkpoint['config'])
    baseline_model.load_state_dict(baseline_checkpoint['model_state_dict'])
    baseline_model = baseline_model.to(device)

    # Load Aletheion model
    aletheion_checkpoint = torch.load("outputs/aletheion/checkpoint_final.pt")
    aletheion_model = AletheionTransformer(**aletheion_checkpoint['config'])
    aletheion_model.load_state_dict(aletheion_checkpoint['model_state_dict'])
    aletheion_model = aletheion_model.to(device)

    print("Evaluating baseline...")
    baseline_conf, baseline_acc = evaluate_model_calibration(
        baseline_model, val_loader, device
    )

    print("Evaluating Aletheion...")
    aletheion_conf, aletheion_acc = evaluate_model_calibration(
        aletheion_model, val_loader, device
    )

    print("Generating reliability diagram...")
    plot_reliability_diagram(
        baseline_conf, baseline_acc,
        aletheion_conf, aletheion_acc,
        Path("outputs/visualizations/reliability_diagrams/calibration_comparison.png")
    )

    print("✓ Done!")

if __name__ == "__main__":
    main()
```

#### 1.2.2 Run Reliability Diagram Generation

```bash
# Create output directory
mkdir -p outputs/visualizations/reliability_diagrams/

# Run script
python scripts/generate_reliability_diagram.py

# Expected output:
# ✓ Saved reliability diagram to outputs/visualizations/reliability_diagrams/calibration_comparison.png
```

#### 1.2.3 Interpretation

**Good calibration:** Points lie close to diagonal line
**Poor calibration:** Points deviate far from diagonal
**Expected:** Aletheion points closer to diagonal than baseline

---

### Step 1.3: TruthfulQA Evaluation

**Purpose:** Test truthfulness and calibration on adversarial questions

**Time:** 3-4 days

#### 1.3.1 Setup TruthfulQA

```bash
# Install TruthfulQA dependencies
pip install datasets

# Download dataset
python -c "from datasets import load_dataset; load_dataset('truthful_qa', 'generation')"
```

#### 1.3.2 Run Evaluation

```bash
# Evaluate baseline
python experiments/level1/test_truthfulqa.py \
  --checkpoint outputs/baseline/checkpoint_final.pt \
  --model_type baseline \
  --output outputs/benchmarks/truthfulqa_baseline.json

# Evaluate Aletheion
python experiments/level1/test_truthfulqa.py \
  --checkpoint outputs/aletheion/checkpoint_final.pt \
  --model_type aletheion \
  --output outputs/benchmarks/truthfulqa_aletheion.json
```

#### 1.3.3 Analyze Results

```python
# Compare results
import json

baseline = json.load(open("outputs/benchmarks/truthfulqa_baseline.json"))
aletheion = json.load(open("outputs/benchmarks/truthfulqa_aletheion.json"))

print(f"Baseline truthfulness: {baseline['truthfulness']:.3f}")
print(f"Aletheion truthfulness: {aletheion['truthfulness']:.3f}")
print(f"Improvement: {(aletheion['truthfulness'] - baseline['truthfulness']):.3f}")
```

---

### Step 1.4: Case Study Analysis

**Purpose:** Understand when/why uncertainty is high or low

**Time:** 2-3 days

#### 1.4.1 Extract High/Low Uncertainty Examples

Create `scripts/extract_uncertainty_cases.py`:

```python
#!/usr/bin/env python3
"""Extract and analyze high/low uncertainty prediction cases."""

import torch
from pathlib import Path
from src.aletheion.model import AletheionTransformer
from data.dataset import load_wikitext_dataset
from transformers import AutoTokenizer

def analyze_predictions(model, tokenizer, device='cuda', n_examples=20):
    """Find high and low uncertainty examples."""

    model.eval()

    # Load validation data
    _, val_dataset, _, _ = load_wikitext_dataset()

    high_uncertainty_cases = []
    low_uncertainty_cases = []

    with torch.no_grad():
        for i in range(min(len(val_dataset), 1000)):
            sample = val_dataset[i]
            input_ids = sample['input_ids'].unsqueeze(0).to(device)

            outputs = model(input_ids)

            # Compute uncertainty
            q1 = outputs['q1'].mean().item()
            q2 = outputs['q2'].mean().item()
            uncertainty = 1.0 - (q1 * q2)

            # Get prediction
            logits = outputs['logits']
            pred_token = logits.argmax(dim=-1)[0, -1].item()
            pred_text = tokenizer.decode([pred_token])

            # Get context
            context = tokenizer.decode(input_ids[0])

            case = {
                'context': context,
                'prediction': pred_text,
                'q1': q1,
                'q2': q2,
                'uncertainty': uncertainty
            }

            if uncertainty > 0.7:
                high_uncertainty_cases.append(case)
            elif uncertainty < 0.3:
                low_uncertainty_cases.append(case)

    # Sort and limit
    high_uncertainty_cases.sort(key=lambda x: x['uncertainty'], reverse=True)
    low_uncertainty_cases.sort(key=lambda x: x['uncertainty'])

    return high_uncertainty_cases[:n_examples], low_uncertainty_cases[:n_examples]

def save_case_study(high_cases, low_cases, output_file):
    """Save case study to markdown file."""

    with open(output_file, 'w') as f:
        f.write("# Uncertainty Case Study Analysis\n\n")

        f.write("## High Uncertainty Examples (Model is Uncertain)\n\n")
        for i, case in enumerate(high_cases, 1):
            f.write(f"### Example {i}\n")
            f.write(f"**Uncertainty:** {case['uncertainty']:.3f} (Q₁={case['q1']:.3f}, Q₂={case['q2']:.3f})\n\n")
            f.write(f"**Context:** {case['context']}\n\n")
            f.write(f"**Prediction:** {case['prediction']}\n\n")
            f.write("---\n\n")

        f.write("## Low Uncertainty Examples (Model is Confident)\n\n")
        for i, case in enumerate(low_cases, 1):
            f.write(f"### Example {i}\n")
            f.write(f"**Uncertainty:** {case['uncertainty']:.3f} (Q₁={case['q1']:.3f}, Q₂={case['q2']:.3f})\n\n")
            f.write(f"**Context:** {case['context']}\n\n")
            f.write(f"**Prediction:** {case['prediction']}\n\n")
            f.write("---\n\n")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    checkpoint = torch.load("outputs/aletheion/checkpoint_final.pt")
    model = AletheionTransformer(**checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    print("Analyzing predictions...")
    high_cases, low_cases = analyze_predictions(model, tokenizer, device)

    print(f"Found {len(high_cases)} high uncertainty cases")
    print(f"Found {len(low_cases)} low uncertainty cases")

    # Save
    output_file = Path("outputs/visualizations/uncertainty_case_study.md")
    save_case_study(high_cases, low_cases, output_file)

    print(f"✓ Saved case study to {output_file}")

if __name__ == "__main__":
    main()
```

#### 1.4.2 Run Analysis

```bash
python scripts/extract_uncertainty_cases.py
```

#### 1.4.3 Manual Review

Review `outputs/visualizations/uncertainty_case_study.md` and verify:
- High uncertainty examples make sense (ambiguous contexts, rare words, etc.)
- Low uncertainty examples make sense (common patterns, deterministic contexts)

---

### Step 1.5: Out-of-Domain Testing

**Purpose:** Verify uncertainty increases on unfamiliar domains

**Time:** 2-3 days

#### 1.5.1 Run OOD Tests

```bash
# Test on code
python experiments/level1/test_out_of_domain.py \
  --checkpoint outputs/aletheion/checkpoint_final.pt \
  --domain code \
  --output outputs/benchmarks/ood_code.json

# Test on math
python experiments/level1/test_out_of_domain.py \
  --checkpoint outputs/aletheion/checkpoint_final.pt \
  --domain math \
  --output outputs/benchmarks/ood_math.json

# Test on scientific text
python experiments/level1/test_out_of_domain.py \
  --checkpoint outputs/aletheion/checkpoint_final.pt \
  --domain scientific \
  --output outputs/benchmarks/ood_scientific.json
```

#### 1.5.2 Analyze OOD Results

Expected: Uncertainty should be higher on OOD domains than on WikiText-2

```python
import json

wikitext_uncertainty = 0.53  # From training (1 - 0.47 * 0.47)

code_results = json.load(open("outputs/benchmarks/ood_code.json"))
math_results = json.load(open("outputs/benchmarks/ood_math.json"))
sci_results = json.load(open("outputs/benchmarks/ood_scientific.json"))

print(f"WikiText-2 uncertainty: {wikitext_uncertainty:.3f}")
print(f"Code uncertainty: {code_results['mean_uncertainty']:.3f}")
print(f"Math uncertainty: {math_results['mean_uncertainty']:.3f}")
print(f"Scientific uncertainty: {sci_results['mean_uncertainty']:.3f}")

# All OOD should be > WikiText-2
assert code_results['mean_uncertainty'] > wikitext_uncertainty
assert math_results['mean_uncertainty'] > wikitext_uncertainty
assert sci_results['mean_uncertainty'] > wikitext_uncertainty

print("✓ OOD uncertainty correctly increased!")
```

---

### Step 1.6: Final Integration

#### 1.6.1 Update Documentation

```bash
# Update QUANTITATIVE_METRICS_ANALYSIS.md with multi-seed results
# Add section 11: Multi-Seed Validation

# Update README.md roadmap
# Change: - [ ] Level 1 validation results (50% complete)
# To:     - [x] Level 1 validation results (100% complete) ✅

# Update CHANGELOG.md
# Add validation completion announcement
```

#### 1.6.2 Create Summary Report

Create `docs/LEVEL1_VALIDATION_COMPLETE.md` summarizing all results.

#### 1.6.3 Commit and Push

```bash
git add docs/progress/ docs/QUANTITATIVE_METRICS_ANALYSIS.md docs/LEVEL1_VALIDATION_COMPLETE.md README.md CHANGELOG.md
git commit -m "docs: complete Level 1 validation (100%)"
git push -u origin claude/validation-performance-optimization-011CUuW85bYr3XMCN6pG4Pca
```

---

## Task 2: Performance Optimization (0% → 100%)

**Goal:** Optimize computational performance without degrading calibration

**Current Status:** 0% (not started, waiting for validation)

**Estimated Time:** 2 weeks

**Estimated Compute:** ~5 GPU-days

**Prerequisites:** Task 1 must be at least 75% complete

---

### Step 2.1: Profiling

**Purpose:** Identify computational bottlenecks

**Time:** 2-3 days

#### 2.1.1 Setup Profiling Tools

```bash
# Install profiling tools
pip install py-spy
pip install torch-tb-profiler

# Verify installation
python -c "import torch.profiler; print('✓ OK')"
```

#### 2.1.2 Profile Training Loop

Create `scripts/profile_training.py`:

```python
#!/usr/bin/env python3
"""Profile Aletheion training to identify bottlenecks."""

import torch
from torch.profiler import profile, ProfilerActivity, record_function
from src.aletheion.model import AletheionTransformer
from src.aletheion.loss import VaroLoss
from data.dataset import load_wikitext_dataset
from torch.utils.data import DataLoader

def profile_training_step():
    """Profile a single training step."""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create model
    model = AletheionTransformer(
        vocab_size=50257,
        d_model=512,
        n_layers=6,
        n_heads=8,
        d_ff=2048,
        max_seq_len=512,
        dropout=0.1,
        q1_threshold=0.7,
        q2_threshold=0.7
    ).to(device)

    # Create loss
    criterion = VaroLoss(lambda_varo=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Load data
    train_dataset, _, _, _ = load_wikitext_dataset()
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Get one batch
    batch = next(iter(train_loader))
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)

    # Warmup
    for _ in range(5):
        outputs = model(input_ids)
        loss_dict = criterion(outputs['logits'], labels, outputs['q1'], outputs['q2'])
        loss_dict['loss'].backward()
        optimizer.step()
        optimizer.zero_grad()

    # Profile
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with record_function("forward"):
            outputs = model(input_ids)

        with record_function("loss"):
            loss_dict = criterion(outputs['logits'], labels, outputs['q1'], outputs['q2'])

        with record_function("backward"):
            loss_dict['loss'].backward()

        with record_function("optimizer"):
            optimizer.step()
            optimizer.zero_grad()

    # Print report
    print("\n" + "=" * 80)
    print("PROFILING REPORT: Training Step")
    print("=" * 80)

    print("\nTop 10 CPU Operations:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    print("\nTop 10 CUDA Operations:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    print("\nTop 10 Memory Operations:")
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))

    # Export trace
    prof.export_chrome_trace("outputs/profiling/training_trace.json")
    print("\n✓ Exported trace to outputs/profiling/training_trace.json")
    print("  View at: chrome://tracing")

if __name__ == "__main__":
    profile_training_step()
```

#### 2.1.3 Run Profiling

```bash
mkdir -p outputs/profiling/
python scripts/profile_training.py

# Analyze trace in Chrome
# 1. Open Chrome browser
# 2. Navigate to chrome://tracing
# 3. Load outputs/profiling/training_trace.json
```

#### 2.1.4 Identify Bottlenecks

Look for operations taking > 5% of total time:
- Attention computation
- Q₁/Q₂ gate forward pass
- VARO loss computation
- Matrix multiplications

Document findings in `outputs/profiling/bottleneck_analysis.md`

---

### Step 2.2: Implement Optimizations

**Purpose:** Apply optimizations to identified bottlenecks

**Time:** 4-5 days

#### 2.2.1 Torch Compile

```python
# Add to src/aletheion/model.py

class AletheionTransformer(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # ... existing code ...

        # Mark for compilation (PyTorch 2.0+)
        self._compiled = False

    def compile(self, mode="reduce-overhead"):
        """Compile model for faster inference."""
        if not self._compiled:
            import torch
            if hasattr(torch, 'compile'):
                self = torch.compile(self, mode=mode)
                self._compiled = True
        return self
```

Usage:
```python
model = AletheionTransformer(...)
model = model.compile()  # Compile for speedup
```

#### 2.2.2 Optimize Gates

Review `src/aletheion/gates.py` for redundant operations:

```python
# Before (inefficient)
def forward(self, context):
    features = self.fc1(context)
    features = self.relu(features)
    features = self.dropout(features)
    gate = self.fc2(features)
    gate = torch.sigmoid(gate)
    return gate

# After (optimized - fuse operations)
def forward(self, context):
    # Fuse fc1 + relu
    features = torch.relu(self.fc1(context))
    features = self.dropout(features)
    # Use in-place sigmoid
    return torch.sigmoid(self.fc2(features))
```

#### 2.2.3 Mixed Precision Training

```python
# Add to training scripts
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    optimizer.zero_grad()

    with autocast():  # Mixed precision context
        outputs = model(input_ids)
        loss_dict = criterion(...)
        loss = loss_dict['loss']

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

#### 2.2.4 Gradient Checkpointing

```python
# Add to model config
from torch.utils.checkpoint import checkpoint

class AletheionTransformer(nn.Module):
    def __init__(self, ..., use_gradient_checkpointing=False):
        self.use_gradient_checkpointing = use_gradient_checkpointing

    def forward(self, input_ids):
        if self.use_gradient_checkpointing and self.training:
            # Checkpoint transformer layers
            for layer in self.layers:
                hidden = checkpoint(layer, hidden)
        else:
            # Normal forward
            for layer in self.layers:
                hidden = layer(hidden)
```

---

### Step 2.3: Benchmark Optimizations

**Purpose:** Measure speedup and verify no calibration regression

**Time:** 2-3 days

#### 2.3.1 Benchmark Script

Create `scripts/benchmark_performance.py`:

```python
#!/usr/bin/env python3
"""Benchmark training and inference performance."""

import torch
import time
import numpy as np
from src.aletheion.model import AletheionTransformer
from src.model import BaselineTransformer
from data.dataset import load_wikitext_dataset
from torch.utils.data import DataLoader

def benchmark_model(model, data_loader, device='cuda', n_steps=100):
    """Benchmark model throughput."""

    model.train()
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Warmup
    for i, batch in enumerate(data_loader):
        if i >= 10:
            break
        input_ids = batch['input_ids'].to(device)
        outputs = model(input_ids)
        loss = outputs['logits'].mean()  # Dummy loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    tokens_processed = 0

    for i, batch in enumerate(data_loader):
        if i >= n_steps:
            break

        input_ids = batch['input_ids'].to(device)
        batch_size, seq_len = input_ids.shape

        outputs = model(input_ids)
        loss = outputs['logits'].mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        tokens_processed += batch_size * seq_len

    torch.cuda.synchronize()
    end_time = time.time()

    elapsed = end_time - start_time
    throughput = tokens_processed / elapsed

    return {
        'throughput': throughput,
        'elapsed': elapsed,
        'tokens_processed': tokens_processed
    }

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load dataset
    train_dataset, _, _, _ = load_wikitext_dataset()
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    print("=" * 80)
    print("PERFORMANCE BENCHMARK")
    print("=" * 80)

    # Baseline
    print("\nBenchmarking Baseline Transformer...")
    baseline_model = BaselineTransformer(
        vocab_size=50257, d_model=512, n_layers=6, n_heads=8, d_ff=2048
    )
    baseline_results = benchmark_model(baseline_model, train_loader, device)

    # Aletheion (unoptimized)
    print("Benchmarking Aletheion (unoptimized)...")
    aletheion_model = AletheionTransformer(
        vocab_size=50257, d_model=512, n_layers=6, n_heads=8, d_ff=2048
    )
    aletheion_results = benchmark_model(aletheion_model, train_loader, device)

    # Aletheion (optimized)
    print("Benchmarking Aletheion (optimized)...")
    aletheion_opt_model = AletheionTransformer(
        vocab_size=50257, d_model=512, n_layers=6, n_heads=8, d_ff=2048
    )
    aletheion_opt_model = aletheion_opt_model.compile()  # Enable optimizations
    aletheion_opt_results = benchmark_model(aletheion_opt_model, train_loader, device)

    # Report
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(f"\nBaseline Throughput: {baseline_results['throughput']:.0f} tokens/sec")
    print(f"Aletheion (unoptimized): {aletheion_results['throughput']:.0f} tokens/sec")
    print(f"Aletheion (optimized): {aletheion_opt_results['throughput']:.0f} tokens/sec")

    overhead_unopt = (baseline_results['throughput'] - aletheion_results['throughput']) / baseline_results['throughput'] * 100
    overhead_opt = (baseline_results['throughput'] - aletheion_opt_results['throughput']) / baseline_results['throughput'] * 100
    speedup = (aletheion_opt_results['throughput'] - aletheion_results['throughput']) / aletheion_results['throughput'] * 100

    print(f"\nOverhead (unoptimized): {overhead_unopt:.1f}%")
    print(f"Overhead (optimized): {overhead_opt:.1f}%")
    print(f"Speedup from optimizations: {speedup:.1f}%")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
```

#### 2.3.2 Run Benchmark

```bash
python scripts/benchmark_performance.py

# Expected output:
# Baseline: ~5000 tokens/sec
# Aletheion (unopt): ~4500 tokens/sec (10% overhead)
# Aletheion (opt): ~4800 tokens/sec (4% overhead)
# Speedup: ~7% from optimizations
```

#### 2.3.3 Verify No Calibration Regression

Train for 1000 steps with optimizations enabled:

```bash
python experiments/level1/train_pyramidal_q1q2.py \
  --config config/aletheion_level1.yaml \
  --output outputs/optimized_validation \
  --use_compile \
  --use_mixed_precision \
  --max_steps 1000

# Check ECE hasn't regressed
# Should still be ~0.01-0.02
```

---

### Step 2.4: Document Optimizations

Create `docs/PERFORMANCE_OPTIMIZATION_REPORT.md` documenting:
- Bottlenecks identified
- Optimizations applied
- Speedup achieved
- Verification that calibration maintained

---

## Task 3: Extended Benchmarking (25% → 100%)

**Goal:** Validate on diverse datasets and perform ablation studies

**Current Status:** 25% (WikiText-2 complete)

**Estimated Time:** 6 weeks

**Estimated Compute:** ~30 GPU-days

---

### Step 3.1: Additional Datasets

**Purpose:** Test generalization beyond WikiText-2

**Time:** 5-7 days

#### 3.1.1 WikiText-103

```bash
# Download WikiText-103
python -c "from datasets import load_dataset; load_dataset('wikitext', 'wikitext-103-v1')"

# Train on WikiText-103
python experiments/level1/train_pyramidal_q1q2.py \
  --config config/aletheion_level1.yaml \
  --dataset wikitext-103 \
  --output outputs/benchmarks/wikitext103 \
  --max_steps 60000
```

#### 3.1.2 Penn Treebank

```bash
# Train on Penn Treebank
python experiments/level1/train_pyramidal_q1q2.py \
  --config config/aletheion_level1.yaml \
  --dataset penn_treebank \
  --output outputs/benchmarks/penn_treebank \
  --max_steps 60000
```

#### 3.1.3 The Pile (subset)

```bash
# Train on The Pile subset
python experiments/level1/train_pyramidal_q1q2.py \
  --config config/aletheion_level1.yaml \
  --dataset pile_subset \
  --output outputs/benchmarks/pile \
  --max_steps 60000
```

#### 3.1.4 Compare Results

Create summary table in `docs/EXTENDED_BENCHMARKS_REPORT.md`:

| Dataset | Baseline ECE | Aletheion ECE | Improvement |
|---------|--------------|---------------|-------------|
| WikiText-2 | 0.104 | 0.011 | 89% |
| WikiText-103 | ? | ? | ? |
| Penn Treebank | ? | ? | ? |
| The Pile | ? | ? | ? |

---

### Step 3.2: Ablation Studies

**Purpose:** Understand hyperparameter sensitivity

**Time:** 8-10 days

#### 3.2.1 VARO Loss Weight Sweep

```bash
# λ_varo ∈ {0.05, 0.1, 0.15, 0.2}
for lambda in 0.05 0.1 0.15 0.2; do
  python experiments/level1/train_pyramidal_q1q2.py \
    --config config/aletheion_level1.yaml \
    --lambda_varo $lambda \
    --output outputs/ablations/varo_sweep/lambda_$lambda \
    --max_steps 20000
done
```

#### 3.2.2 Q₁/Q₂ Threshold Sweep

```bash
# τ ∈ {0.5, 0.6, 0.7, 0.8, 0.9}
for thresh in 0.5 0.6 0.7 0.8 0.9; do
  python experiments/level1/train_pyramidal_q1q2.py \
    --config config/aletheion_level1.yaml \
    --q1_threshold $thresh \
    --q2_threshold $thresh \
    --output outputs/ablations/threshold_sweep/thresh_$thresh \
    --max_steps 20000
done
```

#### 3.2.3 Analyze Ablations

Create `scripts/analyze_ablations.py`:

```python
#!/usr/bin/env python3
"""Analyze ablation study results."""

import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_varo_sweep():
    """Analyze VARO loss weight sweep."""

    results = []
    for lambda_val in [0.05, 0.1, 0.15, 0.2]:
        metrics_file = Path(f"outputs/ablations/varo_sweep/lambda_{lambda_val}/metrics.json")
        with open(metrics_file) as f:
            metrics = json.load(f)
        metrics['lambda_varo'] = lambda_val
        results.append(metrics)

    df = pd.DataFrame(results)

    # Plot ECE vs lambda
    plt.figure(figsize=(8, 6))
    plt.plot(df['lambda_varo'], df['ece'], 'o-', linewidth=2, markersize=8)
    plt.xlabel('λ_varo', fontsize=12)
    plt.ylabel('ECE', fontsize=12)
    plt.title('Ablation: VARO Loss Weight', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig("outputs/ablations/varo_sweep_plot.png", dpi=300)

    print("VARO Loss Weight Ablation:")
    print(df[['lambda_varo', 'ece', 'perplexity']].to_string(index=False))

    # Find optimal
    optimal_idx = df['ece'].argmin()
    optimal_lambda = df.iloc[optimal_idx]['lambda_varo']
    print(f"\n✓ Optimal λ_varo: {optimal_lambda}")

def analyze_threshold_sweep():
    """Analyze Q₁/Q₂ threshold sweep."""

    results = []
    for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
        metrics_file = Path(f"outputs/ablations/threshold_sweep/thresh_{thresh}/metrics.json")
        with open(metrics_file) as f:
            metrics = json.load(f)
        metrics['threshold'] = thresh
        results.append(metrics)

    df = pd.DataFrame(results)

    # Plot ECE vs threshold
    plt.figure(figsize=(8, 6))
    plt.plot(df['threshold'], df['ece'], 's-', linewidth=2, markersize=8)
    plt.xlabel('Q₁/Q₂ Threshold', fontsize=12)
    plt.ylabel('ECE', fontsize=12)
    plt.title('Ablation: Uncertainty Threshold', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig("outputs/ablations/threshold_sweep_plot.png", dpi=300)

    print("\n\nThreshold Ablation:")
    print(df[['threshold', 'ece', 'perplexity']].to_string(index=False))

    # Find optimal
    optimal_idx = df['ece'].argmin()
    optimal_thresh = df.iloc[optimal_idx]['threshold']
    print(f"\n✓ Optimal threshold: {optimal_thresh}")

if __name__ == "__main__":
    analyze_varo_sweep()
    analyze_threshold_sweep()
```

Run:
```bash
python scripts/analyze_ablations.py
```

---

### Step 3.3: Advanced Metrics

**Purpose:** Compute additional calibration and uncertainty metrics

**Time:** 3-4 days

#### 3.3.1 Selective Accuracy

Measure accuracy when model is confident (various thresholds):

```python
def compute_selective_accuracy(confidences, accuracies, thresholds):
    """Compute accuracy at various confidence thresholds."""

    results = []
    for thresh in thresholds:
        mask = confidences >= thresh
        if mask.sum() > 0:
            sel_acc = accuracies[mask].mean()
            coverage = mask.mean()
        else:
            sel_acc = 0.0
            coverage = 0.0

        results.append({
            'threshold': thresh,
            'selective_accuracy': sel_acc,
            'coverage': coverage
        })

    return pd.DataFrame(results)

# Usage
thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
sel_acc_df = compute_selective_accuracy(confidences, accuracies, thresholds)
print(sel_acc_df)
```

#### 3.3.2 AUC-ROC for Uncertainty

Measure if uncertainty predicts correctness:

```python
from sklearn.metrics import roc_auc_score

# Uncertainty as predictor of errors
uncertainties = 1.0 - confidences
errors = 1 - accuracies  # 1 if wrong, 0 if correct

auc_roc = roc_auc_score(errors, uncertainties)
print(f"AUC-ROC (uncertainty predicts errors): {auc_roc:.3f}")
# Expected: > 0.7 (good), > 0.8 (excellent)
```

---

### Step 3.4: Comparative Baselines

**Purpose:** Compare against other uncertainty methods

**Time:** 4-5 days

#### 3.4.1 Implement MC Dropout

```python
class MCDropoutTransformer(BaselineTransformer):
    """Baseline with Monte Carlo Dropout for uncertainty."""

    def mc_forward(self, input_ids, n_samples=10):
        """Forward pass with MC dropout."""
        self.train()  # Enable dropout even at test time

        logits_samples = []
        for _ in range(n_samples):
            outputs = self(input_ids)
            logits_samples.append(outputs['logits'])

        # Stack and compute statistics
        logits_stack = torch.stack(logits_samples)  # (n_samples, batch, seq, vocab)
        logits_mean = logits_stack.mean(dim=0)
        logits_var = logits_stack.var(dim=0)

        # Uncertainty from variance
        uncertainty = logits_var.mean(dim=-1, keepdim=True)

        return {
            'logits': logits_mean,
            'uncertainty': uncertainty
        }
```

#### 3.4.2 Benchmark Comparison

Compare:
- **Aletheion:** Q₁/Q₂ gates (this work)
- **MC Dropout:** Multiple forward passes
- **Deep Ensembles:** Multiple models
- **Temperature Scaling:** Post-hoc calibration

Metrics:
- ECE (calibration quality)
- Inference time (efficiency)
- Memory usage
- Computational cost

Document in `docs/COMPARATIVE_BASELINES_REPORT.md`

---

### Step 3.5: Final Documentation

Create comprehensive report: `docs/EXTENDED_BENCHMARKING_COMPLETE.md`

Include:
- All dataset results
- Ablation study findings
- Advanced metrics
- Comparative baseline results
- Recommendations for hyperparameters

---

## Troubleshooting

### Common Issues

#### Issue: Out of Memory (OOM)

**Solution:**
```bash
# Reduce batch size
# Edit config/aletheion_level1.yaml:
training:
  batch_size: 16  # Down from 32
  gradient_accumulation_steps: 2  # Maintain effective batch size
```

#### Issue: Training diverges (NaN loss)

**Solution:**
```bash
# Reduce learning rate
training:
  learning_rate: 1e-4  # Down from 3e-4

# Increase gradient clipping
training:
  grad_clip_norm: 0.5  # Down from 1.0
```

#### Issue: Multi-seed runs show high variance

**Solution:**
- Run more seeds (10+ instead of 5)
- Increase training steps (100k instead of 60k)
- Average results and report confidence intervals

#### Issue: Profiling script fails

**Solution:**
```bash
# Install profiling tools
pip install torch-tb-profiler

# Use simpler profiling
import cProfile
cProfile.run('train_step()', 'profile.stats')
```

---

## Checklist

Use this checklist to track progress:

### Task 1: Level 1 Validation

- [ ] Multi-seed validation (5 runs complete)
- [ ] Statistical analysis (mean ± std computed)
- [ ] Reliability diagrams generated
- [ ] TruthfulQA evaluation complete
- [ ] Case study analysis done
- [ ] Out-of-domain testing done
- [ ] Results integrated into docs
- [ ] README.md updated

### Task 2: Performance Optimization

- [ ] Profiling complete (bottlenecks identified)
- [ ] Torch compile implemented
- [ ] Gate optimizations applied
- [ ] Mixed precision enabled
- [ ] Gradient checkpointing added (optional)
- [ ] Performance benchmark run
- [ ] Calibration verified (no regression)
- [ ] Optimization report written

### Task 3: Extended Benchmarking

- [ ] WikiText-103 evaluation
- [ ] Penn Treebank evaluation
- [ ] The Pile evaluation
- [ ] VARO loss ablation
- [ ] Threshold ablation
- [ ] Height/base loss ablations
- [ ] Selective accuracy computed
- [ ] AUC-ROC computed
- [ ] MC Dropout comparison
- [ ] Deep Ensembles comparison
- [ ] Temperature scaling comparison
- [ ] Extended benchmarking report written

### Final Steps

- [ ] Update README.md roadmap (all items 100%)
- [ ] Update CHANGELOG.md
- [ ] Integrate results into paper
- [ ] Create summary visualizations
- [ ] Prepare for publication
- [ ] Commit and push all changes

---

## Timeline Summary

| Week | Tasks | Deliverables |
|------|-------|--------------|
| 1 | Multi-seed validation, reliability diagrams | Validation plots, stats |
| 2 | TruthfulQA, case studies, OOD testing | Evaluation reports |
| 3 | Final validation integration | Level 1 complete (100%) |
| 4 | Profiling, optimization implementation | Optimized model |
| 5 | Performance benchmarking | Optimization report |
| 6-8 | Additional datasets, ablations | Extended benchmarks |
| 9-10 | Comparative baselines, final docs | All tasks complete |

**Total Estimated Time:** 10 weeks
**Total Estimated Compute:** ~80 GPU-days

---

## Questions?

If you encounter issues not covered here:

1. Check existing documentation in `docs/`
2. Review similar experiments in `experiments/level1/`
3. Search issues: https://github.com/AletheionAGI/aletheion-llm/issues
4. Ask for help: contact@alethea.tech or Discord (.lacivo)

---

**Document Version:** 1.0
**Last Updated:** 2025-11-08
**Status:** Active Development Guide
**Maintainer:** Aletheion Research Team
