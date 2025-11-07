"""Visualize Epistemic Metrics.

This script visualizes height, Q1, Q2 evolution on specific examples
to understand when the model knows vs doesn't know.

Outputs:
    - Per-example height progression
    - Q1 vs Q2 scatter plot
    - Force weights heatmap
    - "When model knows vs doesn't know" examples

Usage:
    python experiments/level1/visualize_epistemic.py \
        --model outputs/pyramidal/final \
        --examples examples.txt \
        --output outputs/epistemic_viz

    # Without examples file (uses validation set)
    python experiments/level1/visualize_epistemic.py \
        --model outputs/pyramidal/final \
        --output outputs/epistemic_viz
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
from data.dataset import load_wikitext_dataset
from src import get_device, set_seed
from src.aletheion.pyramidal_model import AletheionPyramidalTransformer
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


def collate_fn(batch):
    """Pad variable length sequences."""
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    return {"input_ids": input_ids_padded, "labels": labels_padded}


def load_examples(examples_file: str | None, _tokenizer) -> list[str]:
    """Load example texts."""
    if examples_file is None:
        # Use some default examples
        return [
            "The capital of France is Paris.",
            "Quantum mechanics describes the behavior of matter and energy at",
            "Once upon a time, there was a",
            "The answer to life, the universe, and everything is",
            "In machine learning, overfitting occurs when",
        ]

    with open(examples_file, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


@torch.no_grad()
def analyze_example(
    model: AletheionPyramidalTransformer, text: str, tokenizer, device: torch.device
) -> dict:
    """Analyze a single example and extract epistemic metrics."""
    model.eval()

    # Tokenize
    tokens = tokenizer.encode(text)
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
    labels = input_ids.clone()

    # Forward pass
    outputs = model(input_ids, labels=labels, return_pyramid_state=True)

    # Get logits and pyramid state
    logits = outputs.logits
    pyramid = outputs.pyramid

    # Shift for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous().squeeze(0)
    shift_labels = labels[..., 1:].contiguous().squeeze(0)

    # Get predictions
    probs = F.softmax(shift_logits, dim=-1)
    confidence, predictions = probs.max(dim=-1)

    # Compute correctness and Q1/Q2
    correct = (predictions == shift_labels).float()

    # Q1: Prediction quality (accuracy)
    Q1 = correct

    # Q2: Confidence calibration quality
    Q2 = correct * confidence + (1 - correct) * (1 - confidence)

    # Get pyramid metrics
    height = pyramid["height"][0, :-1, 0]
    base_stability = pyramid["base_stability"][0, :-1, 0]

    # Get force weights
    w_memory = pyramid["w_memory"][0, :-1, 0]
    w_pain = pyramid["w_pain"][0, :-1, 0]
    w_choice = pyramid["w_choice"][0, :-1, 0]
    w_exploration = pyramid["w_exploration"][0, :-1, 0]

    # Decode tokens
    token_strs = [tokenizer.decode([t]) for t in shift_labels.cpu().numpy()]

    return {
        "text": text,
        "tokens": token_strs,
        "predictions": [tokenizer.decode([p]) for p in predictions.cpu().numpy()],
        "correct": correct.cpu().numpy(),
        "confidence": confidence.cpu().numpy(),
        "Q1": Q1.cpu().numpy(),
        "Q2": Q2.cpu().numpy(),
        "height": height.cpu().numpy(),
        "base_stability": base_stability.cpu().numpy(),
        "w_memory": w_memory.cpu().numpy(),
        "w_pain": w_pain.cpu().numpy(),
        "w_choice": w_choice.cpu().numpy(),
        "w_exploration": w_exploration.cpu().numpy(),
    }


def plot_height_progression(examples: list[dict], save_path: Path):
    """Plot height progression for examples."""
    fig, axes = plt.subplots(len(examples), 1, figsize=(12, 4 * len(examples)))

    if len(examples) == 1:
        axes = [axes]

    for idx, example in enumerate(examples):
        ax = axes[idx]

        positions = np.arange(len(example["height"]))

        # Plot height
        ax.plot(positions, example["height"], "b-", label="Height", linewidth=2)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Uncertain")

        # Color-code by correctness
        for i, correct in enumerate(example["correct"]):
            if correct:
                ax.axvspan(i - 0.5, i + 0.5, alpha=0.1, color="green")
            else:
                ax.axvspan(i - 0.5, i + 0.5, alpha=0.1, color="red")

        # Add token labels
        ax.set_xticks(positions)
        ax.set_xticklabels(example["tokens"], rotation=45, ha="right", fontsize=8)

        ax.set_ylabel("Height")
        ax.set_title(f'Example {idx + 1}: "{example["text"][:50]}..."')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Saved height progression to {save_path}")
    plt.close()


def plot_q1_q2_scatter(examples: list[dict], save_path: Path):
    """Plot Q1 vs Q2 scatter plot."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Collect all Q1, Q2 values
    all_Q1 = []
    all_Q2 = []
    all_heights = []

    for example in examples:
        all_Q1.extend(example["Q1"])
        all_Q2.extend(example["Q2"])
        all_heights.extend(example["height"])

    all_Q1 = np.array(all_Q1)
    all_Q2 = np.array(all_Q2)
    all_heights = np.array(all_heights)

    # Create scatter plot colored by height
    scatter = ax.scatter(
        all_Q1, all_Q2, c=all_heights, cmap="viridis", alpha=0.5, s=20, vmin=0, vmax=1
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Height", rotation=270, labelpad=20)

    # Add diagonal line (ideal Q1 = Q2)
    ax.plot([0, 1], [0, 1], "r--", alpha=0.5, label="Ideal (Q1 = Q2)")

    # Add quadrants
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.3)

    # Add quadrant labels
    ax.text(
        0.75,
        0.75,
        "Correct & Calibrated",
        ha="center",
        va="center",
        fontsize=10,
        alpha=0.5,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7},
    )
    ax.text(
        0.25,
        0.25,
        "Wrong & Calibrated",
        ha="center",
        va="center",
        fontsize=10,
        alpha=0.5,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7},
    )
    ax.text(
        0.75,
        0.25,
        "Correct but Underconfident",
        ha="center",
        va="center",
        fontsize=10,
        alpha=0.5,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7},
    )
    ax.text(
        0.25,
        0.75,
        "Wrong but Overconfident",
        ha="center",
        va="center",
        fontsize=10,
        alpha=0.5,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7},
    )

    ax.set_xlabel("Q1 (Prediction Quality)")
    ax.set_ylabel("Q2 (Confidence Calibration)")
    ax.set_title("Q1 vs Q2 (colored by Height)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Saved Q1 vs Q2 scatter to {save_path}")
    plt.close()


def plot_force_weights_heatmap(examples: list[dict], save_path: Path):
    """Plot force weights heatmap."""
    fig, axes = plt.subplots(len(examples), 1, figsize=(14, 3 * len(examples)))

    if len(examples) == 1:
        axes = [axes]

    for idx, example in enumerate(examples):
        ax = axes[idx]

        # Create force weights matrix
        force_matrix = np.array(
            [example["w_memory"], example["w_pain"], example["w_choice"], example["w_exploration"]]
        )

        # Plot heatmap
        im = ax.imshow(force_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

        # Set ticks
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(["Memory", "Pain", "Choice", "Exploration"])
        ax.set_xticks(np.arange(len(example["tokens"])))
        ax.set_xticklabels(example["tokens"], rotation=45, ha="right", fontsize=8)

        ax.set_title(f'Example {idx + 1}: Force Weights - "{example["text"][:50]}..."')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Weight", rotation=270, labelpad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Saved force weights heatmap to {save_path}")
    plt.close()


def create_knows_vs_doesnt_know_examples(examples: list[dict], save_path: Path):
    """Create markdown report of when model knows vs doesn't know."""
    report = """# When Model Knows vs Doesn't Know

## Methodology

We analyze predictions based on:
- **Height**: Proximity to truth (0=uncertain, 1=certain)
- **Q1**: Prediction quality (0=wrong, 1=correct)
- **Q2**: Confidence calibration (high when confidence matches correctness)

---

"""

    for idx, example in enumerate(examples):
        report += f"\n## Example {idx + 1}: {example['text']}\n\n"

        # Find high-confidence correct predictions (model knows)
        knows_mask = (example["height"] > 0.7) & (example["correct"] == 1)
        knows_indices = np.where(knows_mask)[0]

        if len(knows_indices) > 0:
            report += "### ✓ Model Knows (High Height + Correct)\n\n"
            report += "| Token | Prediction | Height | Q1 | Q2 | Confidence |\n"
            report += "|-------|------------|--------|----|----|------------|\n"

            for i in knows_indices[:5]:  # Show first 5
                report += f"| `{example['tokens'][i]}` | `{example['predictions'][i]}` | "
                report += f"{example['height'][i]:.3f} | {example['Q1'][i]:.3f} | "
                report += f"{example['Q2'][i]:.3f} | {example['confidence'][i]:.3f} |\n"

            report += "\n"

        # Find low-confidence or incorrect predictions (model doesn't know)
        doesnt_know_mask = (example["height"] < 0.5) | (example["correct"] == 0)
        doesnt_know_indices = np.where(doesnt_know_mask)[0]

        if len(doesnt_know_indices) > 0:
            report += "### ✗ Model Doesn't Know (Low Height or Wrong)\n\n"
            report += "| Token | Prediction | Height | Q1 | Q2 | Confidence |\n"
            report += "|-------|------------|--------|----|----|------------|\n"

            for i in doesnt_know_indices[:5]:  # Show first 5
                report += f"| `{example['tokens'][i]}` | `{example['predictions'][i]}` | "
                report += f"{example['height'][i]:.3f} | {example['Q1'][i]:.3f} | "
                report += f"{example['Q2'][i]:.3f} | {example['confidence'][i]:.3f} |\n"

            report += "\n"

        # Summary statistics
        report += "### Summary\n\n"
        report += f"- **Mean Height**: {example['height'].mean():.3f}\n"
        report += f"- **Mean Q1**: {example['Q1'].mean():.3f}\n"
        report += f"- **Mean Q2**: {example['Q2'].mean():.3f}\n"
        report += f"- **Accuracy**: {example['correct'].mean():.3f}\n"
        report += f"- **Tokens where model knows**: {knows_mask.sum()} / {len(knows_mask)}\n"
        report += f"- **Tokens where model doesn't know**: {doesnt_know_mask.sum()} / {len(doesnt_know_mask)}\n"
        report += "\n---\n"

    with open(save_path, "w") as f:
        f.write(report)

    print(f"  Saved knows vs doesn't know examples to {save_path}")


def plot_uncertainty_vs_error(examples: list[dict], save_path: Path):
    """Plot uncertainty (1 - height) vs error."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Collect all data
    all_uncertainty = []
    all_error = []

    for example in examples:
        uncertainty = 1.0 - example["height"]
        error = 1.0 - example["correct"]
        all_uncertainty.extend(uncertainty)
        all_error.extend(error)

    all_uncertainty = np.array(all_uncertainty)
    all_error = np.array(all_error)

    # Create bins for uncertainty
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2

    mean_errors = []
    std_errors = []

    for i in range(n_bins):
        bin_mask = (all_uncertainty >= bin_boundaries[i]) & (
            all_uncertainty < bin_boundaries[i + 1]
        )
        if bin_mask.sum() > 0:
            mean_errors.append(all_error[bin_mask].mean())
            std_errors.append(all_error[bin_mask].std())
        else:
            mean_errors.append(0)
            std_errors.append(0)

    mean_errors = np.array(mean_errors)
    std_errors = np.array(std_errors)

    # Plot
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect correlation")
    ax.errorbar(
        bin_centers,
        mean_errors,
        yerr=std_errors,
        fmt="o-",
        label="Mean error ± std",
        capsize=5,
        linewidth=2,
    )

    ax.set_xlabel("Uncertainty (1 - Height)")
    ax.set_ylabel("Error Rate")
    ax.set_title("Uncertainty vs Error Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Compute correlation
    corr = np.corrcoef(all_uncertainty, all_error)[0, 1]
    ax.text(
        0.05,
        0.95,
        f"Correlation: {corr:.3f}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Saved uncertainty vs error plot to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize epistemic metrics")
    parser.add_argument("--model", type=str, required=True, help="Path to pyramidal model")
    parser.add_argument(
        "--examples", type=str, default=None, help="Path to examples file (one per line)"
    )
    parser.add_argument(
        "--output", type=str, default="outputs/epistemic_viz", help="Output directory"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = get_device()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("VISUALIZING EPISTEMIC METRICS")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Examples: {args.examples or 'Default examples'}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    _, _, _, tokenizer = load_wikitext_dataset(max_length=512, cache_dir=".cache/wikitext")

    # Load model
    print("Loading pyramidal model...")
    model = AletheionPyramidalTransformer.from_pretrained(args.model).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Load examples
    print("\nLoading examples...")
    example_texts = load_examples(args.examples, tokenizer)
    print(f"  Loaded {len(example_texts)} examples")

    # Analyze examples
    print("\nAnalyzing examples...")
    analyzed_examples = []
    for text in tqdm(example_texts):
        try:
            result = analyze_example(model, text, tokenizer, device)
            analyzed_examples.append(result)
        except Exception as e:
            print(f"  Warning: Failed to analyze '{text[:50]}...': {e}")

    print(f"  Successfully analyzed {len(analyzed_examples)} examples")

    if len(analyzed_examples) == 0:
        print("No examples to visualize. Exiting.")
        return

    # Create visualizations
    print("\nCreating visualizations...")

    # Height progression
    plot_height_progression(analyzed_examples, output_dir / "height_progression.png")

    # Q1 vs Q2 scatter
    plot_q1_q2_scatter(analyzed_examples, output_dir / "q1_vs_q2_scatter.png")

    # Force weights heatmap
    plot_force_weights_heatmap(analyzed_examples, output_dir / "force_weights_heatmap.png")

    # Uncertainty vs error
    plot_uncertainty_vs_error(analyzed_examples, output_dir / "uncertainty_vs_error.png")

    # Knows vs doesn't know examples
    create_knows_vs_doesnt_know_examples(analyzed_examples, output_dir / "knows_vs_doesnt_know.md")

    # Save raw data
    print("\nSaving raw data...")
    # Convert numpy arrays to lists for JSON serialization
    json_data = []
    for ex in analyzed_examples:
        json_ex = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in ex.items()}
        json_data.append(json_ex)

    with open(output_dir / "analyzed_examples.json", "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"  Saved raw data to {output_dir / 'analyzed_examples.json'}")

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print("\nFiles generated:")
    print("  - height_progression.png: Height evolution per example")
    print("  - q1_vs_q2_scatter.png: Q1 vs Q2 scatter plot")
    print("  - force_weights_heatmap.png: Force weights visualization")
    print("  - uncertainty_vs_error.png: Uncertainty vs error correlation")
    print("  - knows_vs_doesnt_know.md: Examples when model knows/doesn't know")
    print("  - analyzed_examples.json: Raw data")


if __name__ == "__main__":
    main()
