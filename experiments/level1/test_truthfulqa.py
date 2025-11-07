"""Test models on TruthfulQA dataset (out-of-domain evaluation).

This script evaluates baseline and pyramidal models on TruthfulQA to assess:
    - Model calibration on truthfulness task
    - Answer quality through log-likelihood scoring
    - Epistemic uncertainty behavior on QA task
    - Performance comparison between baseline and pyramidal approaches

Usage:
    python experiments/level1/test_truthfulqa.py \
        --baseline outputs/baseline/final \
        --pyramidal outputs/pyramidal/final \
        --output outputs/truthfulqa
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import gc
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from tqdm import tqdm
from transformers import GPT2LMHeadModel

from data.dataset import load_truthfulqa_dataset
from src import get_device, set_seed
from src.aletheion.pyramidal_model import AletheionPyramidalTransformer


def collate_fn_qa(batch):
    """Collate function for TruthfulQA that handles variable-length inputs."""
    pad_token_id = batch[0]["pad_token_id"].item()

    input_ids = [item["input_ids"] for item in batch]
    max_len = max(seq.size(0) for seq in input_ids)

    padded_ids = []
    for seq in input_ids:
        padding = torch.full((max_len - seq.size(0),), pad_token_id, dtype=torch.long)
        padded_ids.append(torch.cat([seq, padding], dim=0))

    return {
        "input_ids": torch.stack(padded_ids),
        "questions": [item["question"] for item in batch],
        "best_answers": [item["best_answers"] for item in batch],
        "correct_answers": [item["correct_answers"] for item in batch],
        "incorrect_answers": [item["incorrect_answers"] for item in batch],
    }


def compute_answer_log_likelihood(
    model: torch.nn.Module,
    tokenizer,
    question: str,
    answer: str,
    device: torch.device,
    is_pyramidal: bool = False,
) -> tuple[float, float]:
    """Compute log-likelihood of an answer given a question.

    Returns:
        Tuple of (log_likelihood, mean_uncertainty)
    """
    # Create prompt: "Q: [question]\nA: [answer]"
    prompt = f"Q: {question}\nA: {answer}"

    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    question_only_ids = tokenizer.encode(f"Q: {question}\nA:", return_tensors="pt").to(device)

    # Get the answer token positions
    answer_start_pos = question_only_ids.size(1)

    model.eval()
    with torch.no_grad():
        if is_pyramidal:
            outputs = model(input_ids)
            logits = outputs["logits"]
            uncertainty = outputs.get("uncertainty", torch.zeros(1).to(device))
        else:
            outputs = model(input_ids)
            logits = outputs.logits
            uncertainty = torch.tensor(0.0)

        # Compute log probabilities for answer tokens
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        target_ids = input_ids[:, 1:]

        # Extract log probs for answer tokens only
        answer_log_probs = []
        for i in range(answer_start_pos - 1, target_ids.size(1)):
            if i < log_probs.size(1):
                token_id = target_ids[0, i].item()
                token_log_prob = log_probs[0, i, token_id].item()
                answer_log_probs.append(token_log_prob)

        if len(answer_log_probs) == 0:
            return float("-inf"), uncertainty.mean().item()

        # Average log likelihood
        avg_log_likelihood = np.mean(answer_log_probs)

        return avg_log_likelihood, uncertainty.mean().item()


@torch.no_grad()
def evaluate_on_truthfulqa(
    model: torch.nn.Module,
    tokenizer,
    dataset,
    device: torch.device,
    is_pyramidal: bool = False,
    max_samples: int = 200,
    model_name: str = "Model",
) -> dict:
    """Evaluate model on TruthfulQA.

    For each question, we:
    1. Compute log-likelihood for correct answers
    2. Compute log-likelihood for incorrect answers
    3. Check if model assigns higher likelihood to correct vs incorrect
    4. Track calibration metrics
    """
    model.eval()

    results = {
        "truthful_choices": 0,  # Times correct answer had higher likelihood
        "total_questions": 0,
        "correct_answer_scores": [],
        "incorrect_answer_scores": [],
        "uncertainties": [],
        "question_results": [],
    }

    print(f"\n{'='*60}")
    print(f"Evaluating {model_name} on TruthfulQA")
    print(f"{'='*60}\n")

    for idx in tqdm(range(min(len(dataset), max_samples)), desc=f"Evaluating {model_name}"):
        item = dataset[idx]
        question = item["question"]
        best_answers = item["best_answers"]
        correct_answers = item["correct_answers"]
        incorrect_answers = item["incorrect_answers"]

        # Combine best and correct answers
        all_correct = list(set(best_answers + correct_answers))

        # Score correct answers
        correct_scores = []
        correct_uncertainties = []
        for ans in all_correct:
            score, unc = compute_answer_log_likelihood(
                model, tokenizer, question, ans, device, is_pyramidal
            )
            correct_scores.append(score)
            correct_uncertainties.append(unc)

        # Score incorrect answers
        incorrect_scores = []
        incorrect_uncertainties = []
        for ans in incorrect_answers:
            score, unc = compute_answer_log_likelihood(
                model, tokenizer, question, ans, device, is_pyramidal
            )
            incorrect_scores.append(score)
            incorrect_uncertainties.append(unc)

        # Determine if model made truthful choice
        best_correct_score = max(correct_scores) if correct_scores else float("-inf")
        best_incorrect_score = max(incorrect_scores) if incorrect_scores else float("-inf")

        is_truthful = best_correct_score > best_incorrect_score

        results["truthful_choices"] += int(is_truthful)
        results["total_questions"] += 1
        results["correct_answer_scores"].extend(correct_scores)
        results["incorrect_answer_scores"].extend(incorrect_scores)
        results["uncertainties"].extend(correct_uncertainties + incorrect_uncertainties)

        results["question_results"].append(
            {
                "question": question,
                "is_truthful": is_truthful,
                "best_correct_score": best_correct_score,
                "best_incorrect_score": best_incorrect_score,
                "margin": best_correct_score - best_incorrect_score,
            }
        )

        # Periodic cleanup
        if idx % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # Compute summary statistics
    truthfulness_rate = results["truthful_choices"] / results["total_questions"]

    results["summary"] = {
        "truthfulness_rate": truthfulness_rate,
        "mean_correct_score": np.mean(results["correct_answer_scores"]),
        "mean_incorrect_score": np.mean(results["incorrect_answer_scores"]),
        "mean_uncertainty": np.mean(results["uncertainties"]) if results["uncertainties"] else 0.0,
        "score_gap": np.mean(results["correct_answer_scores"])
        - np.mean(results["incorrect_answer_scores"]),
    }

    print(f"\n{model_name} Results:")
    print(f"  Truthfulness Rate: {truthfulness_rate:.2%}")
    print(f"  Mean Correct Score: {results['summary']['mean_correct_score']:.4f}")
    print(f"  Mean Incorrect Score: {results['summary']['mean_incorrect_score']:.4f}")
    print(f"  Score Gap: {results['summary']['score_gap']:.4f}")
    if is_pyramidal:
        print(f"  Mean Uncertainty: {results['summary']['mean_uncertainty']:.4f}")

    return results


def plot_score_distributions(baseline_results: dict, pyramidal_results: dict, output_dir: Path):
    """Plot distributions of correct vs incorrect answer scores."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (results, model_name) in enumerate(
        [(baseline_results, "Baseline"), (pyramidal_results, "Pyramidal")]
    ):
        ax = axes[idx]

        # Plot histograms
        correct_scores = results["correct_answer_scores"]
        incorrect_scores = results["incorrect_answer_scores"]

        ax.hist(correct_scores, bins=30, alpha=0.6, label="Correct Answers", color="green")
        ax.hist(incorrect_scores, bins=30, alpha=0.6, label="Incorrect Answers", color="red")

        ax.axvline(
            np.mean(correct_scores),
            color="darkgreen",
            linestyle="--",
            label=f"Mean Correct: {np.mean(correct_scores):.3f}",
        )
        ax.axvline(
            np.mean(incorrect_scores),
            color="darkred",
            linestyle="--",
            label=f"Mean Incorrect: {np.mean(incorrect_scores):.3f}",
        )

        ax.set_xlabel("Log-Likelihood Score")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{model_name} - Answer Score Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "score_distributions.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved score distributions to {output_dir / 'score_distributions.png'}")


def plot_truthfulness_comparison(baseline_results: dict, pyramidal_results: dict, output_dir: Path):
    """Create bar chart comparing truthfulness rates."""
    fig, ax = plt.subplots(figsize=(10, 6))

    models = ["Baseline", "Pyramidal"]
    truthfulness_rates = [
        baseline_results["summary"]["truthfulness_rate"],
        pyramidal_results["summary"]["truthfulness_rate"],
    ]

    colors = ["#3498db", "#e74c3c"]
    bars = ax.bar(models, truthfulness_rates, color=colors, alpha=0.7, edgecolor="black")

    # Add value labels on bars
    for bar, rate in zip(bars, truthfulness_rates):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{rate:.1%}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_ylabel("Truthfulness Rate (%)", fontsize=12)
    ax.set_title("TruthfulQA: Model Comparison", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "truthfulness_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved truthfulness comparison to {output_dir / 'truthfulness_comparison.png'}")


def plot_uncertainty_vs_correctness(pyramidal_results: dict, output_dir: Path):
    """Plot relationship between uncertainty and correctness for pyramidal model."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract data
    question_results = pyramidal_results["question_results"]
    margins = [q["margin"] for q in question_results]
    is_truthful = [q["is_truthful"] for q in question_results]

    # Separate by correctness
    truthful_margins = [m for m, t in zip(margins, is_truthful) if t]
    untruthful_margins = [m for m, t in zip(margins, is_truthful) if not t]

    # Plot
    ax.hist(truthful_margins, bins=30, alpha=0.6, label="Truthful", color="green")
    ax.hist(untruthful_margins, bins=30, alpha=0.6, label="Untruthful", color="red")

    ax.set_xlabel("Score Margin (Correct - Incorrect)")
    ax.set_ylabel("Frequency")
    ax.set_title("Pyramidal Model: Score Margin Distribution by Correctness")
    ax.axvline(0, color="black", linestyle="--", linewidth=2, label="Decision Boundary")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "uncertainty_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved uncertainty analysis to {output_dir / 'uncertainty_analysis.png'}")


def plot_sample_questions(
    baseline_results: dict, pyramidal_results: dict, output_dir: Path, n_samples: int = 10
):
    """Create visualization of sample questions and model predictions."""
    fig, axes = plt.subplots(n_samples, 2, figsize=(16, 4 * n_samples))

    for i in range(min(n_samples, len(baseline_results["question_results"]))):
        baseline_q = baseline_results["question_results"][i]
        pyramidal_q = pyramidal_results["question_results"][i]

        question = (
            baseline_q["question"][:100] + "..."
            if len(baseline_q["question"]) > 100
            else baseline_q["question"]
        )

        # Baseline
        ax_b = axes[i, 0] if n_samples > 1 else axes[0]
        ax_b.barh(
            [0, 1],
            [baseline_q["best_correct_score"], baseline_q["best_incorrect_score"]],
            color=["green", "red"],
            alpha=0.7,
        )
        ax_b.set_yticks([0, 1])
        ax_b.set_yticklabels(["Correct", "Incorrect"])
        ax_b.set_xlabel("Log-Likelihood")
        ax_b.set_title(f"Baseline\n{question}", fontsize=9)
        ax_b.axvline(0, color="black", linestyle="--", alpha=0.5)

        # Add checkmark or X
        symbol = "✓" if baseline_q["is_truthful"] else "✗"
        color = "green" if baseline_q["is_truthful"] else "red"
        ax_b.text(
            0.98,
            0.98,
            symbol,
            transform=ax_b.transAxes,
            fontsize=20,
            color=color,
            ha="right",
            va="top",
            fontweight="bold",
        )

        # Pyramidal
        ax_p = axes[i, 1] if n_samples > 1 else axes[1]
        ax_p.barh(
            [0, 1],
            [pyramidal_q["best_correct_score"], pyramidal_q["best_incorrect_score"]],
            color=["green", "red"],
            alpha=0.7,
        )
        ax_p.set_yticks([0, 1])
        ax_p.set_yticklabels(["Correct", "Incorrect"])
        ax_p.set_xlabel("Log-Likelihood")
        ax_p.set_title(f"Pyramidal\n{question}", fontsize=9)
        ax_p.axvline(0, color="black", linestyle="--", alpha=0.5)

        # Add checkmark or X
        symbol = "✓" if pyramidal_q["is_truthful"] else "✗"
        color = "green" if pyramidal_q["is_truthful"] else "red"
        ax_p.text(
            0.98,
            0.98,
            symbol,
            transform=ax_p.transAxes,
            fontsize=20,
            color=color,
            ha="right",
            va="top",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_dir / "sample_questions.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved sample questions to {output_dir / 'sample_questions.png'}")


def create_comprehensive_report(baseline_results: dict, pyramidal_results: dict, output_dir: Path):
    """Generate a comprehensive markdown report."""
    report = []
    report.append("# TruthfulQA Evaluation Report")
    report.append("\n## Overview")
    report.append("\nThis report presents the out-of-domain evaluation results on TruthfulQA,")
    report.append("comparing the Baseline and Pyramidal (Aletheion Level 1) models.")
    report.append("\n### Evaluation Methodology")
    report.append("\nFor each question in TruthfulQA:")
    report.append("1. We compute the log-likelihood of correct answers given the question")
    report.append("2. We compute the log-likelihood of incorrect answers given the question")
    report.append(
        "3. The model is considered 'truthful' if it assigns higher likelihood to correct answers"
    )
    report.append("4. We analyze the distribution of scores and uncertainties")

    report.append("\n## Results Summary")
    report.append("\n### Truthfulness Rates")
    report.append("\n| Model | Truthfulness Rate | Score Gap |")
    report.append("|-------|-------------------|-----------|")

    for name, results in [("Baseline", baseline_results), ("Pyramidal", pyramidal_results)]:
        rate = results["summary"]["truthfulness_rate"]
        gap = results["summary"]["score_gap"]
        report.append(f"| {name} | {rate:.2%} | {gap:.4f} |")

    report.append("\n### Score Statistics")
    report.append("\n| Model | Mean Correct Score | Mean Incorrect Score | Difference |")
    report.append("|-------|-------------------|---------------------|------------|")

    for name, results in [("Baseline", baseline_results), ("Pyramidal", pyramidal_results)]:
        correct = results["summary"]["mean_correct_score"]
        incorrect = results["summary"]["mean_incorrect_score"]
        diff = correct - incorrect
        report.append(f"| {name} | {correct:.4f} | {incorrect:.4f} | {diff:.4f} |")

    report.append("\n### Key Findings")

    improvement = (
        pyramidal_results["summary"]["truthfulness_rate"]
        - baseline_results["summary"]["truthfulness_rate"]
    )

    if improvement > 0:
        report.append(
            f"\n- **Pyramidal model shows {improvement:.1%} improvement** in truthfulness rate"
        )
    elif improvement < 0:
        report.append(f"\n- Baseline model performs {-improvement:.1%} better in truthfulness rate")
    else:
        report.append("\n- Both models show similar truthfulness rates")

    gap_improvement = (
        pyramidal_results["summary"]["score_gap"] - baseline_results["summary"]["score_gap"]
    )

    if gap_improvement > 0:
        report.append(
            f"- Pyramidal model has **{gap_improvement:.4f} larger score gap** between correct/incorrect"
        )

    if pyramidal_results["summary"]["mean_uncertainty"] > 0:
        report.append(
            f"- Pyramidal model average uncertainty: {pyramidal_results['summary']['mean_uncertainty']:.4f}"
        )

    report.append("\n## Visualizations")
    report.append("\n### Truthfulness Comparison")
    report.append("\n![Truthfulness Comparison](truthfulness_comparison.png)")

    report.append("\n### Score Distributions")
    report.append("\n![Score Distributions](score_distributions.png)")
    report.append("\nGreen: Correct answers | Red: Incorrect answers")

    report.append("\n### Uncertainty Analysis (Pyramidal)")
    report.append("\n![Uncertainty Analysis](uncertainty_analysis.png)")

    report.append("\n### Sample Questions")
    report.append("\n![Sample Questions](sample_questions.png)")
    report.append(
        "\n✓ = Truthful (correct answer scored higher) | ✗ = Untruthful (incorrect answer scored higher)"
    )

    report.append("\n## Statistical Analysis")

    # T-test on score gaps
    baseline_gaps = [q["margin"] for q in baseline_results["question_results"]]
    pyramidal_gaps = [q["margin"] for q in pyramidal_results["question_results"]]

    t_stat, p_value = stats.ttest_ind(pyramidal_gaps, baseline_gaps)

    report.append("\n### T-test on Score Margins")
    report.append(f"\n- t-statistic: {t_stat:.4f}")
    report.append(f"- p-value: {p_value:.4f}")

    if p_value < 0.05:
        report.append("- **Statistically significant difference** (p < 0.05)")
    else:
        report.append("- No statistically significant difference (p >= 0.05)")

    report.append("\n## Conclusion")
    report.append("\nThis evaluation demonstrates the models' behavior on an out-of-domain QA task")
    report.append(
        "focused on truthfulness and factual accuracy. The models were trained on WikiText-2"
    )
    report.append("and tested on TruthfulQA without any fine-tuning on QA tasks.")

    report_text = "\n".join(report)

    with open(output_dir / "truthfulqa_report.md", "w") as f:
        f.write(report_text)

    print(f"\n✓ Saved comprehensive report to {output_dir / 'truthfulqa_report.md'}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate models on TruthfulQA")
    parser.add_argument(
        "--baseline", type=str, required=True, help="Path to baseline model checkpoint"
    )
    parser.add_argument(
        "--pyramidal", type=str, required=True, help="Path to pyramidal model checkpoint"
    )
    parser.add_argument(
        "--output", type=str, default="outputs/truthfulqa", help="Output directory for results"
    )
    parser.add_argument(
        "--max-samples", type=int, default=200, help="Maximum number of questions to evaluate"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TruthfulQA Evaluation")
    print("=" * 60)
    print("\nConfiguration:")
    print(f"  Baseline: {args.baseline}")
    print(f"  Pyramidal: {args.pyramidal}")
    print(f"  Output: {args.output}")
    print(f"  Max Samples: {args.max_samples}")
    print(f"  Device: {device}")

    # Verify checkpoints exist
    baseline_path = Path(args.baseline)
    pyramidal_path = Path(args.pyramidal)

    if not baseline_path.exists():
        print(f"\n❌ ERROR: Baseline checkpoint not found at {args.baseline}")
        print("\nPlease train the baseline model first:")
        print("  python experiments/level1/train_baseline.py --output outputs/baseline")
        print("\nSee experiments/level1/TRUTHFULQA_README.md for details.")
        return

    if not pyramidal_path.exists():
        print(f"\n❌ ERROR: Pyramidal checkpoint not found at {args.pyramidal}")
        print("\nPlease train the pyramidal model first:")
        print("  python experiments/level1/train_pyramidal.py --output outputs/pyramidal")
        print("\nSee experiments/level1/TRUTHFULQA_README.md for details.")
        return

    # Check if model files exist
    baseline_has_model = any(
        (baseline_path / f).exists() for f in ["pytorch_model.bin", "model.safetensors", "model.pt"]
    )
    pyramidal_has_model = any(
        (pyramidal_path / f).exists()
        for f in ["pytorch_model.bin", "model.safetensors", "model.pt"]
    )

    if not baseline_has_model:
        print(f"\n⚠️  WARNING: No model weights found in {args.baseline}")
        print("   Expected files: pytorch_model.bin, model.safetensors, or model.pt")
        print("\nPlease ensure the model was trained and saved correctly.")
        return

    if not pyramidal_has_model:
        print(f"\n⚠️  WARNING: No model weights found in {args.pyramidal}")
        print("   Expected files: pytorch_model.bin, model.safetensors, or model.pt")
        print("\nPlease ensure the model was trained and saved correctly.")
        return

    # Load dataset
    print("\nLoading TruthfulQA dataset...")
    dataset, tokenizer = load_truthfulqa_dataset(
        tokenizer_name="gpt2",
        max_length=512,
    )
    print(f"✓ Loaded {len(dataset)} questions")

    # Load baseline model
    print("\nLoading baseline model...")
    try:
        baseline_model = GPT2LMHeadModel.from_pretrained(args.baseline).to(device)
        print("✓ Baseline model loaded")
    except Exception as e:
        print(f"\n❌ ERROR loading baseline model: {e}")
        print("\nPlease check that the checkpoint is valid and contains model weights.")
        return

    # Load pyramidal model
    print("\nLoading pyramidal model...")
    try:
        pyramidal_model = AletheionPyramidalTransformer.from_pretrained(args.pyramidal).to(device)
        print("✓ Pyramidal model loaded")
    except Exception as e:
        print(f"\n❌ ERROR loading pyramidal model: {e}")
        print("\nPlease check that the checkpoint is valid and contains model weights.")
        return

    # Evaluate baseline
    baseline_results = evaluate_on_truthfulqa(
        baseline_model,
        tokenizer,
        dataset,
        device,
        is_pyramidal=False,
        max_samples=args.max_samples,
        model_name="Baseline",
    )

    # Evaluate pyramidal
    pyramidal_results = evaluate_on_truthfulqa(
        pyramidal_model,
        tokenizer,
        dataset,
        device,
        is_pyramidal=True,
        max_samples=args.max_samples,
        model_name="Pyramidal",
    )

    # Save raw results
    print("\nSaving results...")
    with open(output_dir / "baseline_results.json", "w") as f:
        # Convert numpy types to Python types for JSON serialization
        baseline_save = {
            "summary": baseline_results["summary"],
            "truthful_choices": baseline_results["truthful_choices"],
            "total_questions": baseline_results["total_questions"],
        }
        json.dump(baseline_save, f, indent=2)

    with open(output_dir / "pyramidal_results.json", "w") as f:
        pyramidal_save = {
            "summary": pyramidal_results["summary"],
            "truthful_choices": pyramidal_results["truthful_choices"],
            "total_questions": pyramidal_results["total_questions"],
        }
        json.dump(pyramidal_save, f, indent=2)

    print("✓ Saved raw results")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_truthfulness_comparison(baseline_results, pyramidal_results, output_dir)
    plot_score_distributions(baseline_results, pyramidal_results, output_dir)
    plot_uncertainty_vs_correctness(pyramidal_results, output_dir)
    plot_sample_questions(baseline_results, pyramidal_results, output_dir, n_samples=10)

    # Generate comprehensive report
    print("\nGenerating comprehensive report...")
    create_comprehensive_report(baseline_results, pyramidal_results, output_dir)

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    print("  - truthfulqa_report.md (comprehensive report)")
    print("  - truthfulness_comparison.png")
    print("  - score_distributions.png")
    print("  - uncertainty_analysis.png")
    print("  - sample_questions.png")
    print("  - baseline_results.json")
    print("  - pyramidal_results.json")


if __name__ == "__main__":
    main()
