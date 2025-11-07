"""Test Selective Abstention.

This script tests selective abstention on ambiguous cases using height as
an uncertainty signal.

The model can "refuse to answer" when height is below a threshold.

Outputs:
    - Precision/Recall at different height thresholds
    - "Refused to answer" examples with Q1, Q2 breakdown
    - Optimal threshold recommendation
    - Trade-off curves

Usage:
    python experiments/level1/test_abstention.py \
        --model outputs/pyramidal/final \
        --threshold 0.3 \
        --output outputs/abstention_test
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import load_wikitext_dataset
from src import get_device, set_seed
from src.aletheion.pyramidal_model import AletheionPyramidalTransformer


def collate_fn(batch):
    """Pad variable length sequences."""
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    return {"input_ids": input_ids_padded, "labels": labels_padded}


@torch.no_grad()
def collect_predictions(
    model: AletheionPyramidalTransformer,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 100,
) -> dict:
    """Collect predictions with epistemic metrics."""
    model.eval()

    all_heights = []
    all_confidences = []
    all_correctness = []
    all_tokens = []
    all_predictions = []

    for batch_idx, batch in enumerate(tqdm(loader, desc="Collecting predictions", leave=False)):
        if batch_idx >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, labels=labels, return_pyramid_state=True)
        logits = outputs.logits

        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Get probabilities and predictions
        probs = F.softmax(shift_logits, dim=-1)
        confidence, predictions = probs.max(dim=-1)

        # Get height
        height = outputs.pyramid["height"][..., :-1, :].squeeze(-1)

        # Compute correctness
        valid_mask = shift_labels != -100
        if valid_mask.any():
            correct = (predictions == shift_labels).float()

            # Collect data
            all_heights.extend(height[valid_mask].cpu().numpy())
            all_confidences.extend(confidence[valid_mask].cpu().numpy())
            all_correctness.extend(correct[valid_mask].cpu().numpy())
            all_tokens.extend(shift_labels[valid_mask].cpu().numpy())
            all_predictions.extend(predictions[valid_mask].cpu().numpy())

    return {
        "heights": np.array(all_heights),
        "confidences": np.array(all_confidences),
        "correctness": np.array(all_correctness),
        "tokens": np.array(all_tokens),
        "predictions": np.array(all_predictions),
    }


def compute_abstention_metrics(
    correctness: np.ndarray, heights: np.ndarray, threshold: float
) -> dict:
    """Compute abstention metrics at a given threshold."""
    # Model abstains when height < threshold
    abstain_mask = heights < threshold
    answer_mask = ~abstain_mask

    # Of the predictions where model answered
    accuracy_when_answered = correctness[answer_mask].mean() if answer_mask.sum() > 0 else 0.0

    # Coverage: fraction of predictions answered
    coverage = answer_mask.mean()

    # Of the predictions where model abstained
    accuracy_when_abstained = correctness[abstain_mask].mean() if abstain_mask.sum() > 0 else 0.0

    # Precision: accuracy when model answers
    precision = accuracy_when_answered

    # Recall: fraction of correct predictions among all correct predictions
    recall = correctness[answer_mask].sum() / correctness.sum() if correctness.sum() > 0 else 0.0

    # F1 score
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {
        "threshold": threshold,
        "coverage": coverage,
        "accuracy_when_answered": accuracy_when_answered,
        "accuracy_when_abstained": accuracy_when_abstained,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_answered": answer_mask.sum(),
        "n_abstained": abstain_mask.sum(),
    }


def find_optimal_threshold(
    correctness: np.ndarray, heights: np.ndarray, target_coverage: float = 0.8
) -> tuple[float, dict]:
    """Find optimal threshold for a target coverage."""
    thresholds = np.linspace(0, 1, 100)
    best_threshold = 0.5
    best_accuracy = 0.0
    best_metrics = None

    for threshold in thresholds:
        metrics = compute_abstention_metrics(correctness, heights, threshold)

        # Check if coverage is close to target
        if (
            abs(metrics["coverage"] - target_coverage) < 0.05
            and metrics["accuracy_when_answered"] > best_accuracy
        ):
            best_threshold = threshold
            best_accuracy = metrics["accuracy_when_answered"]
            best_metrics = metrics

    return best_threshold, best_metrics


def plot_abstention_curves(correctness: np.ndarray, heights: np.ndarray, save_path: Path):
    """Plot abstention trade-off curves."""
    thresholds = np.linspace(0, 1, 50)

    coverages = []
    accuracies_answered = []
    accuracies_abstained = []
    precisions = []
    recalls = []
    f1s = []

    for threshold in thresholds:
        metrics = compute_abstention_metrics(correctness, heights, threshold)
        coverages.append(metrics["coverage"])
        accuracies_answered.append(metrics["accuracy_when_answered"])
        accuracies_abstained.append(metrics["accuracy_when_abstained"])
        precisions.append(metrics["precision"])
        recalls.append(metrics["recall"])
        f1s.append(metrics["f1"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Coverage vs Accuracy
    ax = axes[0, 0]
    ax.plot(coverages, accuracies_answered, "b-", label="Accuracy when answered", linewidth=2)
    ax.plot(
        coverages,
        accuracies_abstained,
        "r--",
        label="Accuracy when abstained",
        linewidth=2,
        alpha=0.7,
    )
    ax.axhline(
        y=correctness.mean(), color="gray", linestyle="--", alpha=0.5, label="Overall accuracy"
    )
    ax.set_xlabel("Coverage (fraction answered)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Selective Prediction: Coverage vs Accuracy Trade-off")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Threshold vs metrics
    ax = axes[0, 1]
    ax.plot(thresholds, accuracies_answered, "b-", label="Accuracy", linewidth=2)
    ax.plot(thresholds, coverages, "g-", label="Coverage", linewidth=2)
    ax.plot(thresholds, f1s, "orange", label="F1 Score", linewidth=2)
    ax.set_xlabel("Height Threshold")
    ax.set_ylabel("Metric Value")
    ax.set_title("Threshold vs Performance Metrics")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Precision-Recall curve
    ax = axes[1, 0]
    ax.plot(recalls, precisions, "purple", linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Trade-off")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Height distribution for correct vs incorrect
    ax = axes[1, 1]
    correct_mask = correctness == 1
    incorrect_mask = correctness == 0
    ax.hist(heights[correct_mask], bins=50, alpha=0.6, label="Correct", density=True, color="green")
    ax.hist(
        heights[incorrect_mask], bins=50, alpha=0.6, label="Incorrect", density=True, color="red"
    )
    ax.set_xlabel("Height")
    ax.set_ylabel("Density")
    ax.set_title("Height Distribution: Correct vs Incorrect")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Saved abstention curves to {save_path}")
    plt.close()


def create_abstention_examples(
    data: dict, tokenizer, threshold: float, save_path: Path, n_examples: int = 20
):
    """Create examples of refused predictions."""
    abstain_mask = data["heights"] < threshold
    abstain_indices = np.where(abstain_mask)[0]

    # Sort by height (most uncertain first)
    sorted_indices = abstain_indices[np.argsort(data["heights"][abstain_indices])]

    report = f"""# Abstention Examples (Threshold = {threshold:.3f})

## Model refused to answer on these predictions

Below are examples where the model's height was below {threshold:.3f},
indicating high uncertainty.

"""

    # Show some examples
    n_show = min(n_examples, len(sorted_indices))
    for i in range(n_show):
        idx = sorted_indices[i]

        token = data["tokens"][idx]
        prediction = data["predictions"][idx]
        height = data["heights"][idx]
        confidence = data["confidences"][idx]
        correct = data["correctness"][idx]

        token_str = tokenizer.decode([token])
        pred_str = tokenizer.decode([prediction])

        correctness_str = "✓ CORRECT" if correct else "✗ INCORRECT"

        report += f"""
### Example {i + 1}

- **True Token**: `{token_str}`
- **Predicted Token**: `{pred_str}`
- **Height**: {height:.4f}
- **Confidence**: {confidence:.4f}
- **Correctness**: {correctness_str}
- **Uncertainty**: {1 - height:.4f}

"""

    # Statistics
    if abstain_mask.sum() > 0:
        report += f"""
---

## Statistics for Abstained Predictions

- **Total abstained**: {abstain_mask.sum()} / {len(data['heights'])} ({abstain_mask.mean() * 100:.2f}%)
- **Accuracy when abstained**: {data['correctness'][abstain_mask].mean():.4f}
- **Mean height when abstained**: {data['heights'][abstain_mask].mean():.4f}
- **Mean confidence when abstained**: {data['confidences'][abstain_mask].mean():.4f}

---

## Statistics for Answered Predictions

- **Total answered**: {(~abstain_mask).sum()} / {len(data['heights'])} ({(~abstain_mask).mean() * 100:.2f}%)
- **Accuracy when answered**: {data['correctness'][~abstain_mask].mean():.4f}
- **Mean height when answered**: {data['heights'][~abstain_mask].mean():.4f}
- **Mean confidence when answered**: {data['confidences'][~abstain_mask].mean():.4f}

---

## Conclusion

By abstaining on predictions with height < {threshold:.3f}, the model:
- Maintains **{data['correctness'][~abstain_mask].mean():.4f}** accuracy on answered predictions
- Covers **{(~abstain_mask).mean() * 100:.2f}%** of all predictions
- Correctly identifies uncertain predictions (accuracy on abstained: {data['correctness'][abstain_mask].mean():.4f})
"""

    with open(save_path, "w") as f:
        f.write(report)

    print(f"  Saved abstention examples to {save_path}")


def create_summary_report(
    data: dict,
    threshold: float,
    metrics: dict,
    optimal_threshold: float,
    optimal_metrics: dict,
    save_path: Path,
):
    """Create summary report."""
    baseline_accuracy = data["correctness"].mean()

    report = f"""# Selective Abstention Test Report

## Overview

This report analyzes the model's ability to selectively abstain on uncertain
predictions using the height metric as an uncertainty signal.

## Baseline Performance (No Abstention)

- **Overall Accuracy**: {baseline_accuracy:.4f}
- **Total Predictions**: {len(data['correctness'])}

---

## Abstention at Threshold = {threshold:.3f}

### Performance Metrics

- **Coverage**: {metrics['coverage']:.4f} ({metrics['n_answered']} / {len(data['correctness'])} predictions)
- **Accuracy when answered**: {metrics['accuracy_when_answered']:.4f}
- **Accuracy when abstained**: {metrics['accuracy_when_abstained']:.4f}
- **Precision**: {metrics['precision']:.4f}
- **Recall**: {metrics['recall']:.4f}
- **F1 Score**: {metrics['f1']:.4f}

### Improvement

- **Accuracy improvement**: {metrics['accuracy_when_answered'] - baseline_accuracy:+.4f}
- **Relative improvement**: {(metrics['accuracy_when_answered'] - baseline_accuracy) / baseline_accuracy * 100:+.2f}%

---

## Optimal Threshold (80% Coverage)

To maintain 80% coverage while maximizing accuracy:

- **Optimal Threshold**: {optimal_threshold:.4f}
- **Coverage**: {optimal_metrics['coverage']:.4f}
- **Accuracy when answered**: {optimal_metrics['accuracy_when_answered']:.4f}
- **Accuracy improvement**: {optimal_metrics['accuracy_when_answered'] - baseline_accuracy:+.4f}

---

## Analysis

### Height as Uncertainty Signal

The height metric serves as an effective uncertainty signal:

1. **Separation**: Correct predictions have higher mean height ({data['heights'][data['correctness'] == 1].mean():.4f}) than incorrect ({data['heights'][data['correctness'] == 0].mean():.4f})

2. **Calibration**: By abstaining on low-height predictions, accuracy improves from {baseline_accuracy:.4f} to {metrics['accuracy_when_answered']:.4f}

3. **Trade-off**: At {metrics['coverage']:.0%} coverage, we gain {(metrics['accuracy_when_answered'] - baseline_accuracy) * 100:+.2f}% accuracy

### Recommendations

"""

    if metrics["accuracy_when_answered"] > baseline_accuracy:
        report += f"""
✓ **Recommended to use selective abstention with threshold ≈ {optimal_threshold:.3f}**

This allows the model to:
- Maintain high accuracy on answered predictions
- Signal uncertainty on ambiguous cases
- Provide a confidence score for downstream applications
"""
    else:
        report += """
✗ Height metric does not provide sufficient separation for abstention.

Consider:
- Training with stronger calibration objectives
- Using additional uncertainty signals (e.g., Q1, Q2)
- Ensemble methods for uncertainty estimation
"""

    with open(save_path, "w") as f:
        f.write(report)

    print(f"  Saved summary report to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Test selective abstention")
    parser.add_argument("--model", type=str, required=True, help="Path to pyramidal model")
    parser.add_argument(
        "--threshold", type=float, default=0.3, help="Height threshold for abstention"
    )
    parser.add_argument(
        "--output", type=str, default="outputs/abstention_test", help="Output directory"
    )
    parser.add_argument("--max-batches", type=int, default=100, help="Max evaluation batches")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = get_device()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SELECTIVE ABSTENTION TEST")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Threshold: {args.threshold}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print()

    # Load data
    print("Loading WikiText-2...")
    _, val_dataset, _, tokenizer = load_wikitext_dataset(
        max_length=512, cache_dir=".cache/wikitext"
    )

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    # Load model
    print("Loading pyramidal model...")
    model = AletheionPyramidalTransformer.from_pretrained(args.model).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Collect predictions
    print("\nCollecting predictions...")
    data = collect_predictions(model, val_loader, device, args.max_batches)
    print(f"  Collected {len(data['correctness'])} predictions")
    print(f"  Overall accuracy: {data['correctness'].mean():.4f}")
    print(f"  Mean height: {data['heights'].mean():.4f}")

    # Compute metrics at specified threshold
    print(f"\nComputing metrics at threshold = {args.threshold}...")
    metrics = compute_abstention_metrics(data["correctness"], data["heights"], args.threshold)
    print(f"  Coverage: {metrics['coverage']:.4f}")
    print(f"  Accuracy when answered: {metrics['accuracy_when_answered']:.4f}")
    print(f"  Accuracy when abstained: {metrics['accuracy_when_abstained']:.4f}")

    # Find optimal threshold
    print("\nFinding optimal threshold (80% coverage)...")
    optimal_threshold, optimal_metrics = find_optimal_threshold(
        data["correctness"], data["heights"], target_coverage=0.8
    )
    if optimal_metrics:
        print(f"  Optimal threshold: {optimal_threshold:.4f}")
        print(f"  Coverage: {optimal_metrics['coverage']:.4f}")
        print(f"  Accuracy: {optimal_metrics['accuracy_when_answered']:.4f}")
    else:
        print("  Could not find optimal threshold")
        optimal_threshold = 0.5
        optimal_metrics = compute_abstention_metrics(
            data["correctness"], data["heights"], optimal_threshold
        )

    # Create visualizations
    print("\nCreating visualizations...")
    plot_abstention_curves(
        data["correctness"], data["heights"], output_dir / "abstention_curves.png"
    )

    # Create abstention examples
    create_abstention_examples(
        data, tokenizer, args.threshold, output_dir / "abstention_examples.md", n_examples=20
    )

    # Create summary report
    create_summary_report(
        data,
        args.threshold,
        metrics,
        optimal_threshold,
        optimal_metrics,
        output_dir / "abstention_report.md",
    )

    # Save results
    print("\nSaving results...")
    results = {
        "baseline_accuracy": float(data["correctness"].mean()),
        "threshold": args.threshold,
        "metrics": {
            k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
            for k, v in metrics.items()
        },
        "optimal_threshold": float(optimal_threshold),
        "optimal_metrics": {
            k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
            for k, v in optimal_metrics.items()
        },
    }

    with open(output_dir / "abstention_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved results to {output_dir / 'abstention_results.json'}")

    print("\n" + "=" * 80)
    print("ABSTENTION TEST COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print("\nFiles generated:")
    print("  - abstention_curves.png: Trade-off curves")
    print("  - abstention_examples.md: Refused predictions with analysis")
    print("  - abstention_report.md: Summary report with recommendations")
    print("  - abstention_results.json: Raw numerical results")


if __name__ == "__main__":
    main()
