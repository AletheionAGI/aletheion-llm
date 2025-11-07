"""Test Out-of-Domain Calibration.

This script tests how well pyramidal models maintain calibration on
out-of-domain data compared to baseline models.

Hypothesis: Pyramidal maintains calibration better on OOD data.

Supported datasets:
    - wikitext2 (in-domain baseline)
    - wikitext103
    - custom text file

Usage:
    python experiments/level1/test_out_of_domain.py \
        --model outputs/pyramidal/final \
        --test-dataset wikitext103 \
        --output outputs/ood_test

    python experiments/level1/test_out_of_domain.py \
        --model outputs/pyramidal/final \
        --test-dataset custom \
        --custom-text-file data/custom.txt \
        --output outputs/ood_test
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import GPT2LMHeadModel

from data.dataset import load_wikitext_dataset
from src import get_device, set_seed
from src.aletheion.loss import compute_calibration_metrics
from src.aletheion.pyramidal_model import AletheionPyramidalTransformer


class CustomTextDataset(Dataset):
    """Dataset for custom text files."""

    def __init__(self, text_file: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Read text file
        with open(text_file, encoding="utf-8") as f:
            text = f.read()

        # Split into chunks
        self.examples = []
        tokens = tokenizer.encode(text)

        for i in range(0, len(tokens) - max_length, max_length):
            chunk = tokens[i : i + max_length]
            if len(chunk) == max_length:
                self.examples.append(chunk)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens = self.examples[idx]
        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "labels": torch.tensor(tokens, dtype=torch.long),
        }


def collate_fn(batch):
    """Pad variable length sequences."""
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    return {"input_ids": input_ids_padded, "labels": labels_padded}


def load_test_dataset(
    dataset_name: str, tokenizer, max_length: int = 512, custom_text_file: str = None
):
    """Load test dataset."""
    if dataset_name == "wikitext2":
        _, _, test_dataset, _ = load_wikitext_dataset(
            max_length=max_length, cache_dir=".cache/wikitext"
        )
        return test_dataset

    elif dataset_name == "custom":
        if custom_text_file is None:
            raise ValueError("Must provide --custom-text-file for custom dataset")
        return CustomTextDataset(custom_text_file, tokenizer, max_length)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


@torch.no_grad()
def evaluate_model_ood(
    model,
    loader: DataLoader,
    device: torch.device,
    is_pyramidal: bool = False,
    max_batches: int = 100,
) -> dict:
    """Evaluate model on OOD data."""
    model.eval()

    all_losses = []
    all_confidences = []
    all_correctness = []
    all_entropies = []
    all_probs = []
    all_targets = []

    # Pyramidal-specific metrics
    all_heights = []
    all_uncertainties = []

    for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluating OOD", leave=False)):
        if batch_idx >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        if is_pyramidal:
            outputs = model(input_ids, labels=labels, return_pyramid_state=True)
        else:
            outputs = model(input_ids, labels=labels)

        logits = outputs.logits

        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Get probabilities and predictions
        probs = F.softmax(shift_logits, dim=-1)
        confidence, predictions = probs.max(dim=-1)

        # Compute entropy
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)

        # Compute correctness
        valid_mask = shift_labels != -100
        if valid_mask.any():
            correct = (predictions == shift_labels).float()

            # Collect metrics
            all_losses.append(outputs.loss.item())
            all_confidences.extend(confidence[valid_mask].cpu().numpy())
            all_correctness.extend(correct[valid_mask].cpu().numpy())
            all_entropies.extend(entropy[valid_mask].cpu().numpy())

            # Pyramidal-specific metrics
            if is_pyramidal:
                height = outputs.pyramid["height"][..., :-1, :].squeeze(-1)
                all_heights.extend(height[valid_mask].cpu().numpy())
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
        all_probs_cat, all_targets_cat, dummy_uncertainty, n_bins=10
    )

    results = {
        "perplexity": perplexity,
        "loss": avg_loss,
        "ece": cal_metrics["ece"],
        "brier_score": cal_metrics["brier_score"],
        "accuracy": np.mean(all_correctness),
        "mean_confidence": np.mean(all_confidences),
        "mean_entropy": np.mean(all_entropies),
        "confidences": np.array(all_confidences),
        "correctness": np.array(all_correctness),
        "entropies": np.array(all_entropies),
    }

    if is_pyramidal:
        results["mean_height"] = np.mean(all_heights)
        results["mean_uncertainty"] = np.mean(all_uncertainties)
        results["heights"] = np.array(all_heights)
        results["uncertainties"] = np.array(all_uncertainties)

    return results


def plot_ood_calibration(
    in_domain_results: dict, ood_results: dict, save_path: Path, model_name: str = "Model"
):
    """Plot calibration comparison for in-domain vs OOD."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (results, name) in enumerate(
        [(in_domain_results, "In-Domain"), (ood_results, "Out-of-Domain")]
    ):
        ax = axes[idx]

        # Compute calibration curve
        confidences = results["confidences"]
        correctness = results["correctness"]

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
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", alpha=0.5)
        ax.plot(
            avg_confidences,
            accuracies,
            "o-",
            label=f'{name}\nECE={results["ece"]:.4f}',
            linewidth=2,
        )

        # Add bars for counts
        ax2 = ax.twinx()
        ax2.bar(avg_confidences, counts, alpha=0.2, width=0.08, color="gray")
        ax2.set_ylabel("Count", alpha=0.5)

        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{model_name} - {name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Saved OOD calibration plots to {save_path}")
    plt.close()


def plot_uncertainty_distribution(in_domain_results: dict, ood_results: dict, save_path: Path):
    """Plot uncertainty distribution for pyramidal model."""
    if "uncertainties" not in in_domain_results or "uncertainties" not in ood_results:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Uncertainty histogram
    ax = axes[0]
    ax.hist(in_domain_results["uncertainties"], bins=50, alpha=0.6, label="In-Domain", density=True)
    ax.hist(ood_results["uncertainties"], bins=50, alpha=0.6, label="Out-of-Domain", density=True)
    ax.set_xlabel("Uncertainty (1 - Height)")
    ax.set_ylabel("Density")
    ax.set_title("Uncertainty Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Height vs correctness
    ax = axes[1]

    # In-domain
    in_correct = in_domain_results["correctness"] == 1
    in_incorrect = in_domain_results["correctness"] == 0
    ax.scatter(
        in_domain_results["heights"][in_correct],
        np.ones(in_correct.sum()),
        alpha=0.1,
        s=1,
        label="In-Domain Correct",
        color="green",
    )
    ax.scatter(
        in_domain_results["heights"][in_incorrect],
        np.zeros(in_incorrect.sum()),
        alpha=0.1,
        s=1,
        label="In-Domain Incorrect",
        color="red",
    )

    # OOD
    ood_correct = ood_results["correctness"] == 1
    ood_incorrect = ood_results["correctness"] == 0
    ax.scatter(
        ood_results["heights"][ood_correct],
        np.ones(ood_correct.sum()) + 0.1,
        alpha=0.1,
        s=1,
        label="OOD Correct",
        color="blue",
        marker="x",
    )
    ax.scatter(
        ood_results["heights"][ood_incorrect],
        np.zeros(ood_incorrect.sum()) - 0.1,
        alpha=0.1,
        s=1,
        label="OOD Incorrect",
        color="orange",
        marker="x",
    )

    ax.set_xlabel("Height")
    ax.set_ylabel("Correctness")
    ax.set_title("Height vs Correctness")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Saved uncertainty plots to {save_path}")
    plt.close()


def create_ood_report(
    in_domain_results: dict,
    ood_results: dict,
    save_path: Path,
    model_name: str,
    dataset_name: str,
    is_pyramidal: bool,
):
    """Create OOD evaluation report."""
    report = f"""# Out-of-Domain Evaluation Report

## Model: {model_name}
## Test Dataset: {dataset_name}

## In-Domain Performance (WikiText-2 validation)

- **Perplexity**: {in_domain_results['perplexity']:.2f}
- **Loss**: {in_domain_results['loss']:.4f}
- **ECE**: {in_domain_results['ece']:.4f}
- **Brier Score**: {in_domain_results['brier_score']:.4f}
- **Accuracy**: {in_domain_results['accuracy']:.4f}
- **Mean Confidence**: {in_domain_results['mean_confidence']:.4f}
- **Mean Entropy**: {in_domain_results['mean_entropy']:.4f}
"""

    if is_pyramidal:
        report += f"""- **Mean Height**: {in_domain_results['mean_height']:.4f}
- **Mean Uncertainty**: {in_domain_results['mean_uncertainty']:.4f}
"""

    report += f"""
## Out-of-Domain Performance ({dataset_name})

- **Perplexity**: {ood_results['perplexity']:.2f}
- **Loss**: {ood_results['loss']:.4f}
- **ECE**: {ood_results['ece']:.4f}
- **Brier Score**: {ood_results['brier_score']:.4f}
- **Accuracy**: {ood_results['accuracy']:.4f}
- **Mean Confidence**: {ood_results['mean_confidence']:.4f}
- **Mean Entropy**: {ood_results['mean_entropy']:.4f}
"""

    if is_pyramidal:
        report += f"""- **Mean Height**: {ood_results['mean_height']:.4f}
- **Mean Uncertainty**: {ood_results['mean_uncertainty']:.4f}
"""

    # Compute degradation
    ppl_degradation = (
        (ood_results["perplexity"] - in_domain_results["perplexity"])
        / in_domain_results["perplexity"]
        * 100
    )
    ece_degradation = (
        (ood_results["ece"] - in_domain_results["ece"]) / in_domain_results["ece"] * 100
    )

    report += f"""
## Domain Shift Analysis

### Perplexity Degradation
- **Change**: {ood_results['perplexity'] - in_domain_results['perplexity']:+.2f}
- **Relative**: {ppl_degradation:+.2f}%

### Calibration Degradation (ECE)
- **Change**: {ood_results['ece'] - in_domain_results['ece']:+.4f}
- **Relative**: {ece_degradation:+.2f}%
- **Interpretation**: {'Calibration maintained well' if abs(ece_degradation) < 20 else 'Significant calibration drift'}

### Brier Score Degradation
- **Change**: {ood_results['brier_score'] - in_domain_results['brier_score']:+.4f}
- **Relative**: {(ood_results['brier_score'] - in_domain_results['brier_score']) / in_domain_results['brier_score'] * 100:+.2f}%

### Accuracy Degradation
- **Change**: {ood_results['accuracy'] - in_domain_results['accuracy']:+.4f}
- **Relative**: {(ood_results['accuracy'] - in_domain_results['accuracy']) / in_domain_results['accuracy'] * 100:+.2f}%
"""

    if is_pyramidal:
        report += f"""
### Epistemic Metrics

#### Height Shift
- **In-Domain Mean**: {in_domain_results['mean_height']:.4f}
- **OOD Mean**: {ood_results['mean_height']:.4f}
- **Change**: {ood_results['mean_height'] - in_domain_results['mean_height']:+.4f}
- **Interpretation**: {'Model appropriately reduces height (confidence) on OOD data' if ood_results['mean_height'] < in_domain_results['mean_height'] else 'Model maintains or increases height on OOD data (unexpected)'}

#### Uncertainty Shift
- **In-Domain Mean**: {in_domain_results['mean_uncertainty']:.4f}
- **OOD Mean**: {ood_results['mean_uncertainty']:.4f}
- **Change**: {ood_results['mean_uncertainty'] - in_domain_results['mean_uncertainty']:+.4f}
- **Interpretation**: {'✓ Uncertainty appropriately increases on OOD data' if ood_results['mean_uncertainty'] > in_domain_results['mean_uncertainty'] else '✗ Uncertainty does not increase on OOD data'}
"""

    report += """
## Conclusion

"""

    # Add conclusion
    if abs(ece_degradation) < 20:
        report += "✓ **Model maintains good calibration under domain shift.**\n\n"
    else:
        report += "✗ **Calibration degrades significantly under domain shift.**\n\n"

    if is_pyramidal and ood_results["mean_uncertainty"] > in_domain_results["mean_uncertainty"]:
        report += "✓ **Pyramidal model appropriately increases uncertainty on OOD data.**\n\n"

    with open(save_path, "w") as f:
        f.write(report)

    print(f"  Saved OOD report to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Test model on out-of-domain data")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument(
        "--model-type",
        type=str,
        default="pyramidal",
        choices=["baseline", "pyramidal"],
        help="Model type",
    )
    parser.add_argument("--test-dataset", type=str, default="wikitext2", help="Test dataset name")
    parser.add_argument(
        "--custom-text-file",
        type=str,
        default=None,
        help="Path to custom text file (for custom dataset)",
    )
    parser.add_argument("--output", type=str, default="outputs/ood_test", help="Output directory")
    parser.add_argument("--max-batches", type=int, default=100, help="Max evaluation batches")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = get_device()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    is_pyramidal = args.model_type == "pyramidal"

    print("=" * 80)
    print("OUT-OF-DOMAIN CALIBRATION TEST")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Model Type: {args.model_type}")
    print(f"Test Dataset: {args.test_dataset}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    _, _, _, tokenizer = load_wikitext_dataset(max_length=512, cache_dir=".cache/wikitext")

    # Load model
    print(f"\nLoading {args.model_type} model...")
    if is_pyramidal:
        model = AletheionPyramidalTransformer.from_pretrained(args.model).to(device)
    else:
        model = GPT2LMHeadModel.from_pretrained(args.model).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Load in-domain data (WikiText-2 validation)
    print("\nLoading in-domain data (WikiText-2)...")
    _, val_dataset, _, _ = load_wikitext_dataset(max_length=512, cache_dir=".cache/wikitext")
    in_domain_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    # Load OOD data
    print(f"\nLoading out-of-domain data ({args.test_dataset})...")
    ood_dataset = load_test_dataset(
        args.test_dataset, tokenizer, max_length=512, custom_text_file=args.custom_text_file
    )
    ood_loader = DataLoader(
        ood_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    # Evaluate in-domain
    print("\nEvaluating on in-domain data...")
    in_domain_results = evaluate_model_ood(
        model, in_domain_loader, device, is_pyramidal, args.max_batches
    )
    print(f"  Perplexity: {in_domain_results['perplexity']:.2f}")
    print(f"  ECE: {in_domain_results['ece']:.4f}")
    if is_pyramidal:
        print(f"  Mean Height: {in_domain_results['mean_height']:.4f}")

    # Evaluate OOD
    print(f"\nEvaluating on out-of-domain data ({args.test_dataset})...")
    ood_results = evaluate_model_ood(model, ood_loader, device, is_pyramidal, args.max_batches)
    print(f"  Perplexity: {ood_results['perplexity']:.2f}")
    print(f"  ECE: {ood_results['ece']:.4f}")
    if is_pyramidal:
        print(f"  Mean Height: {ood_results['mean_height']:.4f}")

    # Save results
    print("\nSaving results...")

    results_data = {
        "in_domain": {
            "perplexity": float(in_domain_results["perplexity"]),
            "loss": float(in_domain_results["loss"]),
            "ece": float(in_domain_results["ece"]),
            "brier_score": float(in_domain_results["brier_score"]),
            "accuracy": float(in_domain_results["accuracy"]),
            "mean_confidence": float(in_domain_results["mean_confidence"]),
            "mean_entropy": float(in_domain_results["mean_entropy"]),
        },
        "out_of_domain": {
            "dataset": args.test_dataset,
            "perplexity": float(ood_results["perplexity"]),
            "loss": float(ood_results["loss"]),
            "ece": float(ood_results["ece"]),
            "brier_score": float(ood_results["brier_score"]),
            "accuracy": float(ood_results["accuracy"]),
            "mean_confidence": float(ood_results["mean_confidence"]),
            "mean_entropy": float(ood_results["mean_entropy"]),
        },
    }

    if is_pyramidal:
        results_data["in_domain"]["mean_height"] = float(in_domain_results["mean_height"])
        results_data["in_domain"]["mean_uncertainty"] = float(in_domain_results["mean_uncertainty"])
        results_data["out_of_domain"]["mean_height"] = float(ood_results["mean_height"])
        results_data["out_of_domain"]["mean_uncertainty"] = float(ood_results["mean_uncertainty"])

    with open(output_dir / "ood_results.json", "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"  Saved results to {output_dir / 'ood_results.json'}")

    # Plot calibration
    plot_ood_calibration(
        in_domain_results,
        ood_results,
        output_dir / "ood_calibration.png",
        model_name=args.model_type.capitalize(),
    )

    # Plot uncertainty (pyramidal only)
    if is_pyramidal:
        plot_uncertainty_distribution(
            in_domain_results, ood_results, output_dir / "ood_uncertainty.png"
        )

    # Create report
    create_ood_report(
        in_domain_results,
        ood_results,
        output_dir / "ood_report.md",
        args.model_type.capitalize(),
        args.test_dataset,
        is_pyramidal,
    )

    print("\n" + "=" * 80)
    print("OOD TEST COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
