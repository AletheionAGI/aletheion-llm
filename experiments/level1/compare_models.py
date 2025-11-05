"""Compare Baseline vs Pyramidal Models.

This script loads trained models and compares them side-by-side:
    - Perplexity comparison
    - Calibration metrics (ECE, Brier score)
    - Calibration plots (reliability diagrams)
    - Statistical significance tests
    - Perplexity vs ECE scatter plots

Usage:
    python experiments/level1/compare_models.py \
        --baseline outputs/baseline/final \
        --pyramidal outputs/pyramidal/final \
        --output outputs/comparison
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import math

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

from transformers import GPT2LMHeadModel
from src import get_device, set_seed
from src.aletheion.pyramidal_model import AletheionPyramidalTransformer
from src.aletheion.loss import compute_calibration_metrics
from data.dataset import load_wikitext_dataset


def collate_fn(batch):
    """Pad variable length sequences."""
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    return {"input_ids": input_ids_padded, "labels": labels_padded}


@torch.no_grad()
def evaluate_baseline(
    model: GPT2LMHeadModel,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 100
) -> Dict:
    """Evaluate baseline model."""
    model.eval()

    all_losses = []
    all_confidences = []
    all_correctness = []
    all_probs = []
    all_targets = []

    for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluating Baseline", leave=False)):
        if batch_idx >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, labels=labels)
        logits = outputs.logits

        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Get probabilities and predictions
        probs = F.softmax(shift_logits, dim=-1)
        confidence, predictions = probs.max(dim=-1)

        # Compute correctness
        valid_mask = (shift_labels != -100)
        if valid_mask.any():
            correct = (predictions == shift_labels).float()

            # Collect metrics
            all_losses.append(outputs.loss.item())
            all_confidences.extend(confidence[valid_mask].cpu().numpy())
            all_correctness.extend(correct[valid_mask].cpu().numpy())

            # Sample for calibration (to avoid memory issues)
            n_valid = valid_mask.sum().item()
            n_sample = min(n_valid, 1000)
            if n_sample > 0:
                valid_indices = torch.where(valid_mask.view(-1))[0]
                sampled_indices = valid_indices[torch.randperm(len(valid_indices))[:n_sample]]

                probs_flat = probs.view(-1, probs.size(-1))
                labels_flat = shift_labels.view(-1)

                all_probs.append(probs_flat[sampled_indices].cpu())
                all_targets.append(labels_flat[sampled_indices].cpu())

    # Aggregate metrics
    avg_loss = np.mean(all_losses)
    perplexity = math.exp(avg_loss)

    # Calibration metrics
    all_probs_cat = torch.cat(all_probs)
    all_targets_cat = torch.cat(all_targets)
    dummy_uncertainty = torch.zeros(len(all_targets_cat), 1)

    cal_metrics = compute_calibration_metrics(
        all_probs_cat,
        all_targets_cat,
        dummy_uncertainty,
        n_bins=10
    )

    return {
        'perplexity': perplexity,
        'loss': avg_loss,
        'ece': cal_metrics['ece'],
        'brier_score': cal_metrics['brier_score'],
        'confidences': np.array(all_confidences),
        'correctness': np.array(all_correctness),
        'probs': all_probs_cat,
        'targets': all_targets_cat
    }


@torch.no_grad()
def evaluate_pyramidal(
    model: AletheionPyramidalTransformer,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 100
) -> Dict:
    """Evaluate pyramidal model."""
    model.eval()

    all_losses = []
    all_confidences = []
    all_correctness = []
    all_probs = []
    all_targets = []
    all_heights = []
    all_uncertainties = []

    for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluating Pyramidal", leave=False)):
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

        # Compute correctness
        valid_mask = (shift_labels != -100)
        if valid_mask.any():
            correct = (predictions == shift_labels).float()

            # Get pyramidal metrics
            height = outputs.pyramid['height'][..., :-1, :].squeeze(-1)

            # Collect metrics
            all_losses.append(outputs.loss.item())
            all_confidences.extend(confidence[valid_mask].cpu().numpy())
            all_correctness.extend(correct[valid_mask].cpu().numpy())
            all_heights.extend(height[valid_mask].cpu().numpy())

            # Uncertainty is 1 - height
            uncertainty = 1.0 - height
            all_uncertainties.extend(uncertainty[valid_mask].cpu().numpy())

            # Sample for calibration
            n_valid = valid_mask.sum().item()
            n_sample = min(n_valid, 1000)
            if n_sample > 0:
                valid_indices = torch.where(valid_mask.view(-1))[0]
                sampled_indices = valid_indices[torch.randperm(len(valid_indices))[:n_sample]]

                probs_flat = probs.view(-1, probs.size(-1))
                labels_flat = shift_labels.view(-1)

                all_probs.append(probs_flat[sampled_indices].cpu())
                all_targets.append(labels_flat[sampled_indices].cpu())

    # Aggregate metrics
    avg_loss = np.mean(all_losses)
    perplexity = math.exp(avg_loss)

    # Calibration metrics
    all_probs_cat = torch.cat(all_probs)
    all_targets_cat = torch.cat(all_targets)
    dummy_uncertainty = torch.zeros(len(all_targets_cat), 1)

    cal_metrics = compute_calibration_metrics(
        all_probs_cat,
        all_targets_cat,
        dummy_uncertainty,
        n_bins=10
    )

    return {
        'perplexity': perplexity,
        'loss': avg_loss,
        'ece': cal_metrics['ece'],
        'brier_score': cal_metrics['brier_score'],
        'confidences': np.array(all_confidences),
        'correctness': np.array(all_correctness),
        'heights': np.array(all_heights),
        'uncertainties': np.array(all_uncertainties),
        'probs': all_probs_cat,
        'targets': all_targets_cat
    }


def plot_calibration_comparison(
    baseline_results: Dict,
    pyramidal_results: Dict,
    save_path: Path
):
    """Plot reliability diagrams for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (results, name) in enumerate([
        (baseline_results, 'Baseline'),
        (pyramidal_results, 'Pyramidal')
    ]):
        ax = axes[idx]

        # Compute calibration curve
        confidences = results['confidences']
        correctness = results['correctness']

        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        accuracies = []
        avg_confidences = []
        counts = []

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            if in_bin.sum() > 0:
                accuracies.append(correctness[in_bin].mean())
                avg_confidences.append(confidences[in_bin].mean())
                counts.append(in_bin.sum())
            else:
                accuracies.append(0)
                avg_confidences.append((bin_lower + bin_upper) / 2)
                counts.append(0)

        # Plot reliability diagram
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', alpha=0.5)
        ax.plot(avg_confidences, accuracies, 'o-', label=f'{name} (ECE={results["ece"]:.4f})', linewidth=2)

        # Add bars for counts
        ax2 = ax.twinx()
        ax2.bar(avg_confidences, counts, alpha=0.2, width=0.08, color='gray')
        ax2.set_ylabel('Count', alpha=0.5)

        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{name} Calibration')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved calibration plots to {save_path}")
    plt.close()


def plot_perplexity_ece_scatter(
    baseline_results: Dict,
    pyramidal_results: Dict,
    save_path: Path
):
    """Plot perplexity vs ECE scatter."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot both models
    ax.scatter(
        baseline_results['ece'],
        baseline_results['perplexity'],
        s=200,
        alpha=0.7,
        label='Baseline',
        marker='o'
    )
    ax.scatter(
        pyramidal_results['ece'],
        pyramidal_results['perplexity'],
        s=200,
        alpha=0.7,
        label='Pyramidal',
        marker='s'
    )

    ax.set_xlabel('Expected Calibration Error (ECE)')
    ax.set_ylabel('Perplexity')
    ax.set_title('Perplexity vs Calibration Trade-off')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add annotations
    ax.annotate(
        f'PPL={baseline_results["perplexity"]:.2f}\nECE={baseline_results["ece"]:.4f}',
        xy=(baseline_results['ece'], baseline_results['perplexity']),
        xytext=(10, 10),
        textcoords='offset points',
        fontsize=9
    )
    ax.annotate(
        f'PPL={pyramidal_results["perplexity"]:.2f}\nECE={pyramidal_results["ece"]:.4f}',
        xy=(pyramidal_results['ece'], pyramidal_results['perplexity']),
        xytext=(10, -20),
        textcoords='offset points',
        fontsize=9
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved scatter plot to {save_path}")
    plt.close()


def statistical_significance_test(
    baseline_results: Dict,
    pyramidal_results: Dict
) -> Dict:
    """Perform statistical significance tests."""
    results = {}

    # Test ECE difference
    # Use bootstrap to estimate confidence interval
    n_bootstrap = 1000
    baseline_eces = []
    pyramidal_eces = []

    for _ in range(n_bootstrap):
        # Resample baseline
        indices = np.random.choice(
            len(baseline_results['confidences']),
            size=len(baseline_results['confidences']),
            replace=True
        )
        conf = baseline_results['confidences'][indices]
        corr = baseline_results['correctness'][indices]

        # Compute ECE
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            in_bin = (conf >= bin_boundaries[i]) & (conf < bin_boundaries[i + 1])
            if in_bin.sum() > 0:
                accuracy = corr[in_bin].mean()
                confidence = conf[in_bin].mean()
                ece += (in_bin.sum() / len(conf)) * abs(accuracy - confidence)
        baseline_eces.append(ece)

        # Resample pyramidal
        indices = np.random.choice(
            len(pyramidal_results['confidences']),
            size=len(pyramidal_results['confidences']),
            replace=True
        )
        conf = pyramidal_results['confidences'][indices]
        corr = pyramidal_results['correctness'][indices]

        ece = 0.0
        for i in range(n_bins):
            in_bin = (conf >= bin_boundaries[i]) & (conf < bin_boundaries[i + 1])
            if in_bin.sum() > 0:
                accuracy = corr[in_bin].mean()
                confidence = conf[in_bin].mean()
                ece += (in_bin.sum() / len(conf)) * abs(accuracy - confidence)
        pyramidal_eces.append(ece)

    baseline_eces = np.array(baseline_eces)
    pyramidal_eces = np.array(pyramidal_eces)
    ece_diff = baseline_eces - pyramidal_eces

    results['ece_improvement'] = {
        'mean': ece_diff.mean(),
        'std': ece_diff.std(),
        'ci_95': (np.percentile(ece_diff, 2.5), np.percentile(ece_diff, 97.5)),
        'p_value': (ece_diff <= 0).mean()  # Probability pyramidal is not better
    }

    # Test accuracy difference
    baseline_acc = baseline_results['correctness'].mean()
    pyramidal_acc = pyramidal_results['correctness'].mean()

    # Paired t-test (approximate, assuming independence)
    t_stat, p_value = stats.ttest_ind(
        baseline_results['correctness'],
        pyramidal_results['correctness']
    )

    results['accuracy'] = {
        'baseline': baseline_acc,
        'pyramidal': pyramidal_acc,
        'difference': pyramidal_acc - baseline_acc,
        't_statistic': t_stat,
        'p_value': p_value
    }

    return results


def create_summary_report(
    baseline_results: Dict,
    pyramidal_results: Dict,
    stat_results: Dict,
    save_path: Path
):
    """Create markdown summary report."""
    report = f"""# Model Comparison Report

## Summary Statistics

### Baseline Model
- **Perplexity**: {baseline_results['perplexity']:.2f}
- **Loss**: {baseline_results['loss']:.4f}
- **ECE**: {baseline_results['ece']:.4f}
- **Brier Score**: {baseline_results['brier_score']:.4f}
- **Accuracy**: {baseline_results['correctness'].mean():.4f}

### Pyramidal Model
- **Perplexity**: {pyramidal_results['perplexity']:.2f}
- **Loss**: {pyramidal_results['loss']:.4f}
- **ECE**: {pyramidal_results['ece']:.4f}
- **Brier Score**: {pyramidal_results['brier_score']:.4f}
- **Accuracy**: {pyramidal_results['correctness'].mean():.4f}
- **Mean Height**: {pyramidal_results['heights'].mean():.4f}
- **Mean Uncertainty**: {pyramidal_results['uncertainties'].mean():.4f}

## Improvements

### Perplexity
- **Change**: {pyramidal_results['perplexity'] - baseline_results['perplexity']:+.2f}
- **Relative**: {100 * (pyramidal_results['perplexity'] - baseline_results['perplexity']) / baseline_results['perplexity']:+.2f}%

### Calibration (ECE)
- **Change**: {pyramidal_results['ece'] - baseline_results['ece']:+.4f}
- **Relative**: {100 * (pyramidal_results['ece'] - baseline_results['ece']) / baseline_results['ece']:+.2f}%

### Brier Score
- **Change**: {pyramidal_results['brier_score'] - baseline_results['brier_score']:+.4f}
- **Relative**: {100 * (pyramidal_results['brier_score'] - baseline_results['brier_score']) / baseline_results['brier_score']:+.2f}%

## Statistical Significance

### ECE Improvement
- **Mean Improvement**: {stat_results['ece_improvement']['mean']:.4f}
- **95% CI**: [{stat_results['ece_improvement']['ci_95'][0]:.4f}, {stat_results['ece_improvement']['ci_95'][1]:.4f}]
- **P-value**: {stat_results['ece_improvement']['p_value']:.4f}
- **Significant?**: {'Yes' if stat_results['ece_improvement']['p_value'] < 0.05 else 'No'}

### Accuracy Difference
- **Baseline Accuracy**: {stat_results['accuracy']['baseline']:.4f}
- **Pyramidal Accuracy**: {stat_results['accuracy']['pyramidal']:.4f}
- **Difference**: {stat_results['accuracy']['difference']:+.4f}
- **T-statistic**: {stat_results['accuracy']['t_statistic']:.4f}
- **P-value**: {stat_results['accuracy']['p_value']:.4f}

## Conclusion

"""

    # Add conclusion
    if pyramidal_results['ece'] < baseline_results['ece']:
        report += "✓ **Pyramidal model shows better calibration (lower ECE).**\n\n"
    else:
        report += "✗ Baseline model shows better calibration.\n\n"

    if pyramidal_results['perplexity'] < baseline_results['perplexity']:
        report += "✓ **Pyramidal model shows better perplexity.**\n\n"
    else:
        report += "✗ Baseline model shows better perplexity.\n\n"

    if stat_results['ece_improvement']['p_value'] < 0.05:
        report += "✓ **ECE improvement is statistically significant (p < 0.05).**\n\n"
    else:
        report += "✗ ECE improvement is not statistically significant.\n\n"

    with open(save_path, 'w') as f:
        f.write(report)

    print(f"  Saved report to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare baseline and pyramidal models')
    parser.add_argument('--baseline', type=str, required=True, help='Path to baseline model')
    parser.add_argument('--pyramidal', type=str, required=True, help='Path to pyramidal model')
    parser.add_argument('--output', type=str, default='outputs/comparison', help='Output directory')
    parser.add_argument('--max-batches', type=int, default=100, help='Max evaluation batches')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = get_device()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("COMPARING BASELINE VS PYRAMIDAL MODELS")
    print("=" * 80)
    print(f"Baseline: {args.baseline}")
    print(f"Pyramidal: {args.pyramidal}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print()

    # Load data
    print("Loading WikiText-2...")
    _, val_dataset, _, tokenizer = load_wikitext_dataset(
        max_length=512,
        cache_dir='.cache/wikitext'
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # Load baseline model
    print("\nLoading baseline model...")
    baseline_model = GPT2LMHeadModel.from_pretrained(args.baseline).to(device)
    print(f"  Parameters: {sum(p.numel() for p in baseline_model.parameters()) / 1e6:.1f}M")

    # Load pyramidal model
    print("\nLoading pyramidal model...")
    pyramidal_model = AletheionPyramidalTransformer.from_pretrained(args.pyramidal).to(device)
    print(f"  Parameters: {sum(p.numel() for p in pyramidal_model.parameters()) / 1e6:.1f}M")

    # Evaluate baseline
    print("\nEvaluating baseline model...")
    baseline_results = evaluate_baseline(baseline_model, val_loader, device, args.max_batches)
    print(f"  Perplexity: {baseline_results['perplexity']:.2f}")
    print(f"  ECE: {baseline_results['ece']:.4f}")
    print(f"  Brier Score: {baseline_results['brier_score']:.4f}")

    # Evaluate pyramidal
    print("\nEvaluating pyramidal model...")
    pyramidal_results = evaluate_pyramidal(pyramidal_model, val_loader, device, args.max_batches)
    print(f"  Perplexity: {pyramidal_results['perplexity']:.2f}")
    print(f"  ECE: {pyramidal_results['ece']:.4f}")
    print(f"  Brier Score: {pyramidal_results['brier_score']:.4f}")
    print(f"  Mean Height: {pyramidal_results['heights'].mean():.4f}")

    # Statistical tests
    print("\nRunning statistical significance tests...")
    stat_results = statistical_significance_test(baseline_results, pyramidal_results)

    # Save results
    print("\nSaving results...")

    # Save raw data
    results_data = {
        'baseline': {
            'perplexity': float(baseline_results['perplexity']),
            'loss': float(baseline_results['loss']),
            'ece': float(baseline_results['ece']),
            'brier_score': float(baseline_results['brier_score']),
            'accuracy': float(baseline_results['correctness'].mean())
        },
        'pyramidal': {
            'perplexity': float(pyramidal_results['perplexity']),
            'loss': float(pyramidal_results['loss']),
            'ece': float(pyramidal_results['ece']),
            'brier_score': float(pyramidal_results['brier_score']),
            'accuracy': float(pyramidal_results['correctness'].mean()),
            'mean_height': float(pyramidal_results['heights'].mean()),
            'mean_uncertainty': float(pyramidal_results['uncertainties'].mean())
        },
        'statistical_tests': {
            'ece_improvement': {
                'mean': float(stat_results['ece_improvement']['mean']),
                'std': float(stat_results['ece_improvement']['std']),
                'ci_95': [float(x) for x in stat_results['ece_improvement']['ci_95']],
                'p_value': float(stat_results['ece_improvement']['p_value'])
            },
            'accuracy': {
                'baseline': float(stat_results['accuracy']['baseline']),
                'pyramidal': float(stat_results['accuracy']['pyramidal']),
                'difference': float(stat_results['accuracy']['difference']),
                't_statistic': float(stat_results['accuracy']['t_statistic']),
                'p_value': float(stat_results['accuracy']['p_value'])
            }
        }
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"  Saved results to {output_dir / 'results.json'}")

    # Plot calibration diagrams
    plot_calibration_comparison(
        baseline_results,
        pyramidal_results,
        output_dir / 'calibration_plots.png'
    )

    # Plot perplexity vs ECE
    plot_perplexity_ece_scatter(
        baseline_results,
        pyramidal_results,
        output_dir / 'perplexity_ece_scatter.png'
    )

    # Create summary report
    create_summary_report(
        baseline_results,
        pyramidal_results,
        stat_results,
        output_dir / 'report.md'
    )

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print("\nFiles generated:")
    print(f"  - results.json: Raw numerical results")
    print(f"  - report.md: Markdown summary report")
    print(f"  - calibration_plots.png: Reliability diagrams")
    print(f"  - perplexity_ece_scatter.png: Perplexity vs ECE trade-off")


if __name__ == '__main__':
    main()
