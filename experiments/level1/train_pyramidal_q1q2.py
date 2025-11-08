"""Training script for Pyramidal Q1/Q2/Fractal architecture.

This script trains the complete pyramidal epistemic model with:
- Q1 (aleatoric uncertainty) + variance
- Q2 (epistemic uncertainty) + variance
- Fractal meta-epistemic layer
- Height derived from Q1, Q2, base_stability
- Full VARO loss with all components

Monitors all epistemic metrics to detect collapse (Q1 → 0.88+)

Usage:
    # With WikiText-2 (default)
    python experiments/level1/train_pyramidal_q1q2.py \
        --lambda_Q1 0.0015 \
        --lambda_Q2 0.002 \
        --lambda_fractal 0.0005 \
        --max_steps 5000

    # With TinyStories (legacy)
    python experiments/level1/train_pyramidal_q1q2.py \
        --dataset tinystories \
        --data_dir data/tinystories \
        --max_steps 5000

Expected healthy behavior:
    Q1_mean ∈ [0.2, 0.4]
    Q2_mean ∈ [0.3, 0.6]
    height ∈ [0.5, 0.7]
    fractal ∈ [0.1, 0.3]
    Q1_entropy > 0.3 (no collapse)
    Q2_entropy > 0.3 (no collapse)
"""

import argparse
import json

# Add project root to path
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.dataset import load_wikitext_dataset
from src.aletheion.pyramidal_q1q2_model import AletheionPyramidalQ1Q2Transformer
from src.utils import get_device, set_seed


def collate_fn(batch):
    """Pad variable length sequences for WikiText-2."""
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {"input_ids": input_ids_padded, "labels": labels_padded}


def log_memory(step: int):
    """Log GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  [Step {step}] GPU: {allocated:.2f}GB alloc, {reserved:.2f}GB reserved")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Pyramidal Q1/Q2/Fractal model")

    # Model architecture
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=1024, help="Feedforward dimension")
    parser.add_argument("--max_seq_len", type=int, default=256, help="Max sequence length")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Pyramidal Q1/Q2 parameters (reduced by 10x to let L_CE dominate)
    parser.add_argument("--lambda_base", type=float, default=0.001, help="Base stability weight")
    parser.add_argument("--lambda_Q1", type=float, default=0.0015, help="Q1 calibration weight")
    parser.add_argument("--lambda_Q2", type=float, default=0.002, help="Q2 calibration weight")
    parser.add_argument(
        "--lambda_fractal", type=float, default=0.0005, help="Fractal regularization weight"
    )
    parser.add_argument(
        "--lambda_height", type=float, default=0.002, help="Height calibration weight"
    )
    parser.add_argument(
        "--use_multi_head_height", action="store_true", help="Use multi-head height"
    )
    parser.add_argument(
        "--max_temperature_scale", type=float, default=2.0, help="Max temperature scale"
    )

    # Training parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Micro batch size (per gradient accumulation step)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps (effective batch = batch_size * accum_steps)",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save memory (~40% reduction)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable mixed precision training (fp16) to save memory (~50% reduction)",
    )
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_steps", type=int, default=5000, help="Max training steps")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--eval_interval", type=int, default=100, help="Evaluation interval")
    parser.add_argument("--save_interval", type=int, default=500, help="Save checkpoint interval")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")

    # Data
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext",
        choices=["wikitext", "tinystories"],
        help="Dataset to use (default: wikitext)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=".cache/wikitext",
        help="Data directory (default: .cache/wikitext for WikiText-2, data/tinystories for TinyStories)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of data workers (default: 0 for WikiText-2)",
    )

    # Experiment
    parser.add_argument(
        "--experiment_name", type=str, default="pyramidal_q1q2_v1", help="Experiment name"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume training from checkpoint directory (e.g., experiments/level1/runs/exp/checkpoint_step_1000)",
    )

    return parser.parse_args()


def compute_collapse_signals(pyramid_outputs: dict) -> dict:
    """Compute signals indicating collapse.

    Returns warnings if:
    - Q1_mean → 0.0 or 0.9+ (collapse)
    - Q2_mean → 0.0 or 0.9+ (collapse)
    - height → 0.95+ (overconfidence)
    - fractal → 0.8+ (meta-uncertainty explosion)
    - Q1_entropy < 0.1 (saturated)
    - Q2_entropy < 0.1 (saturated)
    """
    signals = {}

    # Extract metrics
    Q1 = pyramid_outputs["Q1_mean"]
    Q2 = pyramid_outputs["Q2_mean"]
    height = pyramid_outputs["height"]
    fractal = pyramid_outputs["fractal_uncertainty"]

    # Binary entropy: H(p) = -p*log(p) - (1-p)*log(1-p)
    def binary_entropy(p):
        p = p.clamp(1e-8, 1 - 1e-8)
        return -(p * p.log() + (1 - p) * (1 - p).log()).mean()

    Q1_entropy = binary_entropy(Q1)
    Q2_entropy = binary_entropy(Q2)
    height_entropy = binary_entropy(height)

    # Collapse indicators
    Q1_mean = Q1.mean().item()
    Q2_mean = Q2.mean().item()
    height_mean = height.mean().item()
    fractal_mean = fractal.mean().item()

    signals["Q1_collapse"] = Q1_mean < 0.05 or Q1_mean > 0.90
    signals["Q2_collapse"] = Q2_mean < 0.05 or Q2_mean > 0.90
    signals["height_collapse"] = height_mean > 0.95
    signals["fractal_explosion"] = fractal_mean > 0.8
    signals["Q1_saturated"] = Q1_entropy.item() < 0.1
    signals["Q2_saturated"] = Q2_entropy.item() < 0.1

    # Overall health
    signals["any_collapse"] = any(
        [
            signals["Q1_collapse"],
            signals["Q2_collapse"],
            signals["height_collapse"],
            signals["fractal_explosion"],
            signals["Q1_saturated"],
            signals["Q2_saturated"],
        ]
    )

    # Add raw metrics
    signals["Q1_entropy"] = Q1_entropy.item()
    signals["Q2_entropy"] = Q2_entropy.item()
    signals["height_entropy"] = height_entropy.item()

    return signals


def train_step(
    model,
    batch,
    optimizer,
    scaler,
    device,
    grad_clip,
    accumulation_steps=1,
    is_accumulation_step=False,
    use_amp=False,
):
    """Single training step with optional gradient accumulation and mixed precision.

    Args:
        model: The model to train
        batch: Input batch
        optimizer: Optimizer
        scaler: GradScaler for mixed precision training
        device: Device to use
        grad_clip: Gradient clipping value
        accumulation_steps: Number of gradient accumulation steps
        is_accumulation_step: If True, don't zero grads or step optimizer
        use_amp: Whether to use automatic mixed precision
    """
    model.train()

    # Only zero grads at the start of accumulation
    if not is_accumulation_step:
        optimizer.zero_grad(set_to_none=True)

    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    # Forward pass with optional mixed precision
    if use_amp and scaler is not None:
        with torch.cuda.amp.autocast():
            outputs = model(input_ids, labels=labels, return_dict=True)
            loss = outputs.loss / accumulation_steps

            # Compute loss components for detailed logging
            if outputs.pyramid is not None:
                loss_dict = model.pyramid_loss_fn(outputs.logits, labels, outputs.pyramid)
            else:
                loss_dict = None
    else:
        outputs = model(input_ids, labels=labels, return_dict=True)
        loss = outputs.loss / accumulation_steps

        # Compute loss components for detailed logging
        if outputs.pyramid is not None:
            loss_dict = model.pyramid_loss_fn(outputs.logits, labels, outputs.pyramid)
        else:
            loss_dict = None

    # Backward pass with optional scaling
    if use_amp and scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    # Only clip and step at the end of accumulation
    if not is_accumulation_step:
        if use_amp and scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

    # Memory cleanup
    unscaled_loss = loss.item() * accumulation_steps
    pyramid_out = outputs.pyramid
    del loss, outputs, input_ids, labels
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return unscaled_loss, pyramid_out, loss_dict


@torch.no_grad()
def evaluate(model, dataloader, device, max_batches=10):
    """Evaluate model with memory-efficient computation."""
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    # Online statistics for pyramidal metrics
    pyramid_sums = {}
    pyramid_counts = 0

    # Accumulators for calibration metrics
    ece_sum = 0.0
    brier_sum = 0.0
    calibration_counts = 0

    # Accumulators for force weights
    force_sums = {
        "memory": 0.0,
        "pain": 0.0,
        "choice": 0.0,
        "exploration": 0.0,
        "base_stability": 0.0,
    }

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, labels=labels, return_dict=True)
        loss = outputs.loss

        # Accumulate loss
        batch_tokens = (labels != -100).sum().item()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens

        # Accumulate pyramidal metrics incrementally
        if outputs.pyramid is not None:
            for key, value in outputs.pyramid.items():
                if key not in pyramid_sums:
                    pyramid_sums[key] = 0.0
                pyramid_sums[key] += value.mean().item()
            pyramid_counts += 1

            # Compute calibration metrics for this batch
            try:
                loss_dict = model.pyramid_loss_fn(outputs.logits, labels, outputs.pyramid)
                if "ece" in loss_dict and "brier_score" in loss_dict:
                    ece_sum += loss_dict["ece"]
                    brier_sum += loss_dict["brier_score"]
                    calibration_counts += 1

                # Accumulate force weights
                if "mean_memory" in loss_dict:
                    force_sums["memory"] += loss_dict["mean_memory"]
                    force_sums["pain"] += loss_dict["mean_pain"]
                    force_sums["choice"] += loss_dict["mean_choice"]
                    force_sums["exploration"] += loss_dict["mean_exploration"]
                    force_sums["base_stability"] += loss_dict["base_stability"]
            except Exception:
                # Skip if calibration computation fails
                pass

        # Memory cleanup
        del outputs, input_ids, labels, loss

        # Periodic cache clear
        if i % 5 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc

    gc.collect()

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0

    # Average pyramidal metrics
    pyramid_metrics = {}
    if pyramid_counts > 0:
        for key, total in pyramid_sums.items():
            pyramid_metrics[key] = total / pyramid_counts

    # Add calibration metrics
    if calibration_counts > 0:
        pyramid_metrics["ece"] = ece_sum / calibration_counts
        pyramid_metrics["brier_score"] = brier_sum / calibration_counts

        # Add force weights
        for key in force_sums:
            pyramid_metrics[f"force_{key}"] = force_sums[key] / calibration_counts

    return avg_loss, pyramid_metrics


def generate_training_report(
    history: dict,
    final_metrics: dict,
    args: argparse.Namespace,
    save_dir: Path
) -> None:
    """Generate comprehensive Markdown training report.

    Creates a detailed report with:
    - Training configuration
    - Final metrics (ECE, Perplexity, Brier, Q1, Q2, etc.)
    - Training curves summary
    - Calibration metrics
    - Force weights analysis
    - Lambda values used
    """
    report_path = save_dir / f"TRAINING_REPORT_seed_{args.seed}.md"

    # Compute final statistics from history
    final_step = len(history["train_loss"])

    with open(report_path, "w") as f:
        # Header
        f.write(f"# Training Report: Pyramidal Q1/Q2 Model\n\n")
        f.write(f"**Experiment:** {args.experiment_name}\n")
        f.write(f"**Seed:** {args.seed}\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Total Steps:** {final_step}\n")
        f.write(f"**Dataset:** {args.dataset.upper()}\n\n")
        f.write("---\n\n")

        # Table of Contents
        f.write("## 📚 Table of Contents\n\n")
        f.write("1. [Configuration](#configuration)\n")
        f.write("2. [Final Metrics](#final-metrics)\n")
        f.write("3. [Calibration Metrics](#calibration-metrics)\n")
        f.write("4. [Q1/Q2 Analysis](#q1q2-analysis)\n")
        f.write("5. [Force Weights](#force-weights)\n")
        f.write("6. [Loss Components](#loss-components)\n")
        f.write("7. [Training Curves](#training-curves)\n")
        f.write("8. [Recommendations](#recommendations)\n\n")
        f.write("---\n\n")

        # 1. Configuration
        f.write("## Configuration\n\n")
        f.write("### Model Architecture\n\n")
        f.write(f"- **Model Dimension (d_model):** {args.d_model}\n")
        f.write(f"- **Number of Layers:** {args.n_layers}\n")
        f.write(f"- **Attention Heads:** {args.n_heads}\n")
        f.write(f"- **Feedforward Dimension:** {args.d_ff}\n")
        f.write(f"- **Max Sequence Length:** {args.max_seq_len}\n")
        f.write(f"- **Dropout Rate:** {args.dropout}\n\n")

        f.write("### Pyramidal Parameters (Lambda Values)\n\n")
        f.write(f"- **λ_base (Base Stability):** {args.lambda_base}\n")
        f.write(f"- **λ_Q1 (Aleatoric Calibration):** {args.lambda_Q1}\n")
        f.write(f"- **λ_Q2 (Epistemic Calibration):** {args.lambda_Q2}\n")
        f.write(f"- **λ_fractal (Fractal Regularization):** {args.lambda_fractal}\n")
        f.write(f"- **λ_height (Height Calibration):** {args.lambda_height}\n")
        f.write(f"- **Max Temperature Scale:** {args.max_temperature_scale}\n\n")

        f.write("### Training Configuration\n\n")
        effective_batch = args.batch_size * args.gradient_accumulation_steps
        f.write(f"- **Batch Size:** {args.batch_size} (effective: {effective_batch} with {args.gradient_accumulation_steps}x accumulation)\n")
        f.write(f"- **Learning Rate:** {args.learning_rate}\n")
        f.write(f"- **Weight Decay:** {args.weight_decay}\n")
        f.write(f"- **Max Steps:** {args.max_steps}\n")
        f.write(f"- **Warmup Steps:** {args.warmup_steps}\n")
        f.write(f"- **Gradient Clipping:** {args.grad_clip}\n")
        f.write(f"- **Gradient Checkpointing:** {args.gradient_checkpointing}\n")
        f.write(f"- **Mixed Precision (FP16):** {args.fp16}\n\n")
        f.write("---\n\n")

        # 2. Final Metrics
        f.write("## Final Metrics\n\n")

        if history["eval_loss"]:
            final_eval_loss = history["eval_loss"][-1]
            final_perplexity = history["eval_perplexity"][-1]
        else:
            final_eval_loss = history["train_loss"][-1]
            final_perplexity = np.exp(final_eval_loss)

        f.write("### Overall Performance\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| **Final Training Loss** | {history['train_loss'][-1]:.4f} |\n")
        f.write(f"| **Final Validation Loss** | {final_eval_loss:.4f} |\n")
        f.write(f"| **Final Perplexity** | {final_perplexity:.2f} |\n\n")

        # Q1/Q2 Final Values
        f.write("### Uncertainty Components\n\n")
        f.write(f"| Component | Mean | Min | Max | Range | Entropy |\n")
        f.write(f"|-----------|------|-----|-----|-------|----------|\n")

        q1_mean = history["Q1_mean"][-1] if history["Q1_mean"] else 0.0
        q1_min = history["Q1_min"][-1] if history["Q1_min"] else 0.0
        q1_max = history["Q1_max"][-1] if history["Q1_max"] else 0.0
        q1_range = history["Q1_range"][-1] if history["Q1_range"] else 0.0
        q1_entropy = history["Q1_entropy"][-1] if history["Q1_entropy"] else 0.0

        q2_mean = history["Q2_mean"][-1] if history["Q2_mean"] else 0.0
        q2_min = history["Q2_min"][-1] if history["Q2_min"] else 0.0
        q2_max = history["Q2_max"][-1] if history["Q2_max"] else 0.0
        q2_range = history["Q2_range"][-1] if history["Q2_range"] else 0.0
        q2_entropy = history["Q2_entropy"][-1] if history["Q2_entropy"] else 0.0

        height_mean = history["height_mean"][-1] if history["height_mean"] else 0.0
        fractal_mean = history["fractal_mean"][-1] if history["fractal_mean"] else 0.0

        f.write(f"| **Q1 (Aleatoric)** | {q1_mean:.4f} | {q1_min:.4f} | {q1_max:.4f} | {q1_range:.4f} | {q1_entropy:.4f} |\n")
        f.write(f"| **Q2 (Epistemic)** | {q2_mean:.4f} | {q2_min:.4f} | {q2_max:.4f} | {q2_range:.4f} | {q2_entropy:.4f} |\n")
        f.write(f"| **Height** | {height_mean:.4f} | - | - | - | - |\n")
        f.write(f"| **Fractal** | {fractal_mean:.4f} | - | - | - | - |\n\n")

        # Health Check
        f.write("### Health Status\n\n")
        f.write("| Check | Status | Value | Expected Range |\n")
        f.write("|-------|--------|-------|----------------|\n")

        q1_healthy = 0.2 <= q1_mean <= 0.4
        q2_healthy = 0.3 <= q2_mean <= 0.6
        height_healthy = 0.5 <= height_mean <= 0.7
        fractal_healthy = 0.1 <= fractal_mean <= 0.3
        q1_entropy_ok = q1_entropy > 0.3
        q2_entropy_ok = q2_entropy > 0.3

        f.write(f"| Q1 Mean | {'✅ HEALTHY' if q1_healthy else '⚠️ OUT OF RANGE'} | {q1_mean:.3f} | [0.2, 0.4] |\n")
        f.write(f"| Q2 Mean | {'✅ HEALTHY' if q2_healthy else '⚠️ OUT OF RANGE'} | {q2_mean:.3f} | [0.3, 0.6] |\n")
        f.write(f"| Height | {'✅ HEALTHY' if height_healthy else '⚠️ OUT OF RANGE'} | {height_mean:.3f} | [0.5, 0.7] |\n")
        f.write(f"| Fractal | {'✅ HEALTHY' if fractal_healthy else '⚠️ OUT OF RANGE'} | {fractal_mean:.3f} | [0.1, 0.3] |\n")
        f.write(f"| Q1 Entropy | {'✅ NO COLLAPSE' if q1_entropy_ok else '⚠️ COLLAPSED'} | {q1_entropy:.3f} | > 0.3 |\n")
        f.write(f"| Q2 Entropy | {'✅ NO COLLAPSE' if q2_entropy_ok else '⚠️ COLLAPSED'} | {q2_entropy:.3f} | > 0.3 |\n\n")

        f.write("---\n\n")

        # 3. Calibration Metrics
        f.write("## Calibration Metrics\n\n")

        final_ece = history["ece"][-1] if history["ece"] else 0.0
        final_brier = history["brier_score"][-1] if history["brier_score"] else 0.0

        f.write(f"| Metric | Value | Quality |\n")
        f.write(f"|--------|-------|----------|\n")
        f.write(f"| **ECE (Expected Calibration Error)** | {final_ece:.4f} | {'✅ EXCELLENT' if final_ece < 0.05 else '✅ GOOD' if final_ece < 0.1 else '⚠️ NEEDS IMPROVEMENT'} |\n")
        f.write(f"| **Brier Score** | {final_brier:.4f} | {'✅ EXCELLENT' if final_brier < 0.2 else '✅ GOOD' if final_brier < 0.3 else '⚠️ NEEDS IMPROVEMENT'} |\n\n")

        f.write("### Calibration Interpretation\n\n")
        f.write("- **ECE < 0.05:** Excellent calibration (model confidence matches accuracy)\n")
        f.write("- **ECE 0.05-0.10:** Good calibration\n")
        f.write("- **ECE > 0.10:** Poor calibration (needs tuning)\n\n")
        f.write("- **Brier Score < 0.2:** Excellent probabilistic predictions\n")
        f.write("- **Brier Score 0.2-0.3:** Good probabilistic predictions\n")
        f.write("- **Brier Score > 0.3:** Poor probabilistic predictions\n\n")
        f.write("---\n\n")

        # 4. Q1/Q2 Analysis
        f.write("## Q1/Q2 Analysis\n\n")

        f.write("### What are Q1 and Q2?\n\n")
        f.write("- **Q1 (Aleatoric Uncertainty):** Irreducible uncertainty due to noise in the data\n")
        f.write("- **Q2 (Epistemic Uncertainty):** Reducible uncertainty due to lack of knowledge\n\n")

        f.write("### Convergence Analysis\n\n")

        if history["Q1_target"]:
            q1_target = history["Q1_target"][-1]
            q2_target = history["Q2_target"][-1]
            q1_gap = abs(q1_mean - q1_target)
            q2_gap = abs(q2_mean - q2_target)

            f.write(f"| Component | Current | Target | Gap | Converged? |\n")
            f.write(f"|-----------|---------|--------|-----|------------|\n")
            f.write(f"| **Q1** | {q1_mean:.4f} | {q1_target:.4f} | {q1_gap:.4f} | {'✅ YES' if q1_gap < 0.05 else '⚠️ NO'} |\n")
            f.write(f"| **Q2** | {q2_mean:.4f} | {q2_target:.4f} | {q2_gap:.4f} | {'✅ YES' if q2_gap < 0.05 else '⚠️ NO'} |\n\n")

        # Distinctness check
        q1_q2_distinct = abs(q1_mean - q2_mean) > 0.05
        f.write("### Q1/Q2 Distinctness\n\n")
        f.write(f"- **Q1 Mean:** {q1_mean:.4f}\n")
        f.write(f"- **Q2 Mean:** {q2_mean:.4f}\n")
        f.write(f"- **Difference:** {abs(q1_mean - q2_mean):.4f}\n")
        f.write(f"- **Status:** {'✅ Q1 and Q2 are DISTINCT' if q1_q2_distinct else '⚠️ Q1 and Q2 are TOO SIMILAR'}\n\n")

        if not q1_q2_distinct:
            f.write("⚠️ **Warning:** Q1 and Q2 should be distinct. Consider adjusting λ_Q1 and λ_Q2.\n\n")

        f.write("---\n\n")

        # 5. Force Weights
        f.write("## Force Weights\n\n")

        f.write("### Cognitive Force Balance\n\n")

        w_memory = history["w_memory"][-1] if history["w_memory"] else 0.0
        w_pain = history["w_pain"][-1] if history["w_pain"] else 0.0
        w_choice = history["w_choice"][-1] if history["w_choice"] else 0.0
        w_exploration = history["w_exploration"][-1] if history["w_exploration"] else 0.0
        base_stability = history["base_stability"][-1] if history["base_stability"] else 0.0

        total_forces = w_memory + w_pain + w_choice + w_exploration

        f.write(f"| Force | Weight | Percentage | Balanced? |\n")
        f.write(f"|-------|--------|------------|------------|\n")
        f.write(f"| **Memory** | {w_memory:.4f} | {(w_memory/total_forces*100 if total_forces > 0 else 0):.1f}% | {'✅' if 0.15 <= w_memory <= 0.35 else '⚠️'} |\n")
        f.write(f"| **Pain** | {w_pain:.4f} | {(w_pain/total_forces*100 if total_forces > 0 else 0):.1f}% | {'✅' if 0.15 <= w_pain <= 0.35 else '⚠️'} |\n")
        f.write(f"| **Choice** | {w_choice:.4f} | {(w_choice/total_forces*100 if total_forces > 0 else 0):.1f}% | {'✅' if 0.15 <= w_choice <= 0.35 else '⚠️'} |\n")
        f.write(f"| **Exploration** | {w_exploration:.4f} | {(w_exploration/total_forces*100 if total_forces > 0 else 0):.1f}% | {'✅' if 0.15 <= w_exploration <= 0.35 else '⚠️'} |\n")
        f.write(f"| **Total** | {total_forces:.4f} | 100.0% | - |\n\n")

        f.write(f"### Base Stability\n\n")
        f.write(f"- **Value:** {base_stability:.4f}\n")
        f.write(f"- **Target:** > 0.7 (stable foundation)\n")
        f.write(f"- **Status:** {'✅ STABLE' if base_stability > 0.7 else '⚠️ UNSTABLE'}\n\n")

        f.write("---\n\n")

        # 6. Loss Components
        f.write("## Loss Components\n\n")

        final_ce = history["ce_loss"][-1] if history["ce_loss"] else 0.0
        final_base = history["base_loss"][-1] if history["base_loss"] else 0.0
        final_q1_loss = history["Q1_loss"][-1] if history["Q1_loss"] else 0.0
        final_q2_loss = history["Q2_loss"][-1] if history["Q2_loss"] else 0.0
        final_fractal_loss = history["fractal_loss"][-1] if history["fractal_loss"] else 0.0
        final_height_loss = history["height_loss"][-1] if history["height_loss"] else 0.0

        total_loss = final_ce + final_base + final_q1_loss + final_q2_loss + final_fractal_loss + final_height_loss

        f.write(f"| Component | Raw Value | Weighted Value | Percentage |\n")
        f.write(f"|-----------|-----------|----------------|-------------|\n")
        f.write(f"| **Cross-Entropy (CE)** | {final_ce:.6f} | {final_ce:.6f} | {(final_ce/total_loss*100 if total_loss > 0 else 0):.1f}% |\n")
        f.write(f"| **Base Stability** | {final_base:.6f} | {final_base * args.lambda_base:.6f} | {(final_base/total_loss*100 if total_loss > 0 else 0):.1f}% |\n")
        f.write(f"| **Q1 Calibration** | {final_q1_loss:.6f} | {final_q1_loss * args.lambda_Q1:.6f} | {(final_q1_loss/total_loss*100 if total_loss > 0 else 0):.1f}% |\n")
        f.write(f"| **Q2 Calibration** | {final_q2_loss:.6f} | {final_q2_loss * args.lambda_Q2:.6f} | {(final_q2_loss/total_loss*100 if total_loss > 0 else 0):.1f}% |\n")
        f.write(f"| **Fractal Regularization** | {final_fractal_loss:.6f} | {final_fractal_loss * args.lambda_fractal:.6f} | {(final_fractal_loss/total_loss*100 if total_loss > 0 else 0):.1f}% |\n")
        f.write(f"| **Height Calibration** | {final_height_loss:.6f} | {final_height_loss * args.lambda_height:.6f} | {(final_height_loss/total_loss*100 if total_loss > 0 else 0):.1f}% |\n\n")

        f.write("### Loss Balance\n\n")
        f.write("- **CE should dominate** (typically 85-95% of total loss)\n")
        f.write("- Pyramidal components should be small regularizers (5-15% total)\n")
        f.write("- If CE < 80%, lambda values may be too high\n\n")

        f.write("---\n\n")

        # 7. Training Curves
        f.write("## Training Curves\n\n")
        f.write(f"Training curves visualization saved to:\n\n")
        f.write(f"```\n{save_dir / 'training_curves.png'}\n```\n\n")
        f.write("The visualization includes:\n")
        f.write("- Loss curves (train/eval)\n")
        f.write("- Q1/Q2 progression with targets\n")
        f.write("- Height and Fractal uncertainty\n")
        f.write("- Base stability\n")
        f.write("- Force weights evolution\n")
        f.write("- Loss components breakdown\n")
        f.write("- Calibration metrics (ECE, Brier)\n")
        f.write("- Q1/Q2 entropy (collapse detection)\n")
        f.write("- Distribution ranges\n")
        f.write("- Evaluation perplexity\n\n")

        f.write("---\n\n")

        # 8. Recommendations
        f.write("## Recommendations\n\n")

        f.write("### Model Health\n\n")

        all_healthy = (q1_healthy and q2_healthy and height_healthy and
                      fractal_healthy and q1_entropy_ok and q2_entropy_ok)

        if all_healthy:
            f.write("✅ **All metrics are in healthy ranges!**\n\n")
            f.write("The model has converged successfully. You can:\n")
            f.write("- Proceed to evaluation on test set\n")
            f.write("- Run multi-seed validation for reproducibility\n")
            f.write("- Generate reliability diagrams\n")
            f.write("- Test on out-of-domain data\n\n")
        else:
            f.write("⚠️ **Some metrics are out of healthy ranges.**\n\n")
            f.write("**Recommendations:**\n\n")

            if not q1_healthy:
                f.write(f"- **Q1 Mean ({q1_mean:.3f}):** Adjust λ_Q1 (current: {args.lambda_Q1})\n")
                if q1_mean < 0.2:
                    f.write(f"  - Q1 too low → decrease λ_Q1 to allow more aleatoric uncertainty\n")
                else:
                    f.write(f"  - Q1 too high → increase λ_Q1 to reduce aleatoric uncertainty\n")

            if not q2_healthy:
                f.write(f"- **Q2 Mean ({q2_mean:.3f}):** Adjust λ_Q2 (current: {args.lambda_Q2})\n")
                if q2_mean < 0.3:
                    f.write(f"  - Q2 too low → decrease λ_Q2 to allow more epistemic uncertainty\n")
                else:
                    f.write(f"  - Q2 too high → increase λ_Q2 to reduce epistemic uncertainty\n")

            if not height_healthy:
                f.write(f"- **Height ({height_mean:.3f}):** Adjust λ_height (current: {args.lambda_height})\n")

            if not fractal_healthy:
                f.write(f"- **Fractal ({fractal_mean:.3f}):** Adjust λ_fractal (current: {args.lambda_fractal})\n")

            if not q1_entropy_ok:
                f.write(f"- **Q1 Entropy ({q1_entropy:.3f}):** Q1 may have collapsed! Lower λ_Q1\n")

            if not q2_entropy_ok:
                f.write(f"- **Q2 Entropy ({q2_entropy:.3f}):** Q2 may have collapsed! Lower λ_Q2\n")

            f.write("\n")

        f.write("### Calibration Quality\n\n")

        if final_ece < 0.05:
            f.write("✅ **Excellent calibration!** (ECE < 0.05)\n\n")
            f.write("The model's confidence matches its accuracy very well.\n\n")
        elif final_ece < 0.1:
            f.write("✅ **Good calibration** (ECE < 0.10)\n\n")
            f.write("The model is reasonably well-calibrated. Consider:\n")
            f.write("- Temperature scaling for further refinement\n")
            f.write("- More training steps\n\n")
        else:
            f.write("⚠️ **Poor calibration** (ECE > 0.10)\n\n")
            f.write("**Recommendations:**\n")
            f.write("- Increase λ_Q1 and λ_Q2 for better calibration\n")
            f.write("- Train for more steps\n")
            f.write("- Check if model has converged\n")
            f.write("- Apply temperature scaling post-training\n\n")

        f.write("### Next Steps\n\n")
        f.write("1. **Review training curves:** Check `training_curves.png` for convergence patterns\n")
        f.write("2. **Multi-seed validation:** Run with different seeds for reproducibility\n")
        f.write("3. **Evaluation:** Test on held-out test set\n")
        f.write("4. **Ablation studies:** Test sensitivity to hyperparameters\n")
        f.write("5. **Benchmarking:** Compare against baselines (MC Dropout, Deep Ensembles)\n\n")

        f.write("---\n\n")

        # Footer
        f.write("## Files Generated\n\n")
        f.write(f"- **Model checkpoint:** `{save_dir / 'final_model'}`\n")
        f.write(f"- **Training history:** `{save_dir / 'history.json'}`\n")
        f.write(f"- **Training curves:** `{save_dir / 'training_curves.png'}`\n")
        f.write(f"- **This report:** `{report_path}`\n")
        f.write(f"- **Configuration:** `{save_dir / 'config.json'}`\n\n")

        f.write("---\n\n")
        f.write(f"**Report Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Experiment:** {args.experiment_name}\n")
        f.write(f"**Seed:** {args.seed}\n")

    print(f"📄 Training report saved to {report_path}")
    return report_path


def plot_training_curves(history: dict, save_dir: Path):
    """Plot comprehensive training curves for Pyramidal Q1/Q2 model.

    Creates a detailed 4x3 grid showing:
    - Loss curves and perplexity
    - Q1/Q2 progression with targets
    - Height and Fractal uncertainty
    - Base stability and Force weights
    - Loss components breakdown
    - Calibration metrics (ECE, Brier)
    - Q1/Q2 entropy (collapse detection)
    - Q1/Q2 distributions (min/max ranges)
    """
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

    # 1. Loss curves
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history["train_loss"], label="Train Loss", alpha=0.7, linewidth=1.5)
    if "eval_loss" in history and history["eval_loss"]:
        eval_steps = np.linspace(0, len(history["train_loss"]), len(history["eval_loss"]))
        ax1.plot(
            eval_steps,
            history["eval_loss"],
            label="Eval Loss",
            marker="o",
            markersize=4,
            linewidth=2,
        )
    ax1.set_xlabel("Step", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.set_title("Loss Curves", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. Q1 progression
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(history["Q1_mean"], label="Q1 Mean", color="#2E86AB", linewidth=2, alpha=0.8)
    if "Q1_target" in history and any(history["Q1_target"]):
        ax2.plot(
            history["Q1_target"],
            label="Q1 Target",
            color="#06A77D",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
        )
    if "Q1_min" in history and "Q1_max" in history:
        ax2.fill_between(
            range(len(history["Q1_mean"])),
            history["Q1_min"],
            history["Q1_max"],
            alpha=0.2,
            color="#2E86AB",
            label="Q1 Range",
        )
    ax2.axhline(
        y=0.88, color="red", linestyle="--", alpha=0.5, linewidth=1, label="Collapse threshold"
    )
    ax2.set_xlabel("Step", fontsize=10)
    ax2.set_ylabel("Q1 (Aleatoric)", fontsize=10)
    ax2.set_title("Q1 Uncertainty Progression", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    # 3. Q2 progression
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(history["Q2_mean"], label="Q2 Mean", color="#A23B72", linewidth=2, alpha=0.8)
    if "Q2_target" in history and any(history["Q2_target"]):
        ax3.plot(
            history["Q2_target"],
            label="Q2 Target",
            color="#F18F01",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
        )
    if "Q2_min" in history and "Q2_max" in history:
        ax3.fill_between(
            range(len(history["Q2_mean"])),
            history["Q2_min"],
            history["Q2_max"],
            alpha=0.2,
            color="#A23B72",
            label="Q2 Range",
        )
    ax3.axhline(
        y=0.88, color="red", linestyle="--", alpha=0.5, linewidth=1, label="Collapse threshold"
    )
    ax3.set_xlabel("Step", fontsize=10)
    ax3.set_ylabel("Q2 (Epistemic)", fontsize=10)
    ax3.set_title("Q2 Uncertainty Progression", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])

    # 4. Height progression
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(history["height_mean"], label="Mean Height", color="#4361EE", linewidth=2, alpha=0.8)
    if "height_target" in history and any(history["height_target"]):
        ax4.plot(
            history["height_target"],
            label="Target Height",
            color="#06A77D",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
        )
    ax4.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, linewidth=1, label="Mid-pyramid")
    ax4.axhline(
        y=0.95, color="red", linestyle="--", alpha=0.5, linewidth=1, label="Collapse threshold"
    )
    ax4.set_xlabel("Step", fontsize=10)
    ax4.set_ylabel("Height", fontsize=10)
    ax4.set_title("Height Progression (Watch for Collapse!)", fontsize=12, fontweight="bold")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])

    # 5. Fractal uncertainty
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(
        history["fractal_mean"],
        label="Fractal Uncertainty",
        color="#F72585",
        linewidth=2,
        alpha=0.8,
    )
    ax5.axhline(
        y=0.8, color="red", linestyle="--", alpha=0.5, linewidth=1, label="Explosion threshold"
    )
    ax5.set_xlabel("Step", fontsize=10)
    ax5.set_ylabel("Fractal Meta-Epistemic", fontsize=10)
    ax5.set_title("Fractal Uncertainty (Meta-Level)", fontsize=12, fontweight="bold")
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([0, 1])

    # 6. Base stability
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(history["base_stability"], label="Base Stability", color="#7209B7", linewidth=2)
    ax6.axhline(y=0.7, color="green", linestyle="--", alpha=0.5, linewidth=1, label="Target >0.7")
    ax6.set_xlabel("Step", fontsize=10)
    ax6.set_ylabel("Stability", fontsize=10)
    ax6.set_title("Base Stability", fontsize=12, fontweight="bold")
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim([0, 1])

    # 7. Force weights (4 cognitive vertices)
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(history["w_memory"], label="Memory", alpha=0.8, linewidth=1.5, color="#06A77D")
    ax7.plot(history["w_pain"], label="Pain", alpha=0.8, linewidth=1.5, color="#F72585")
    ax7.plot(history["w_choice"], label="Choice", alpha=0.8, linewidth=1.5, color="#4361EE")
    ax7.plot(
        history["w_exploration"], label="Exploration", alpha=0.8, linewidth=1.5, color="#F18F01"
    )
    ax7.axhline(
        y=0.25, color="gray", linestyle="--", alpha=0.5, linewidth=1, label="Balanced (0.25)"
    )
    ax7.set_xlabel("Step", fontsize=10)
    ax7.set_ylabel("Weight", fontsize=10)
    ax7.set_title("Force Weights (4 Cognitive Vertices)", fontsize=12, fontweight="bold")
    ax7.legend(fontsize=8, ncol=2)
    ax7.grid(True, alpha=0.3)
    ax7.set_ylim([0, 1])

    # 8. Loss components
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(history["ce_loss"], label="CE Loss", alpha=0.8, linewidth=1.5, color="#2E86AB")
    if "base_loss" in history and any(history["base_loss"]):
        ax8.plot(history["base_loss"], label="Base Loss", alpha=0.8, linewidth=1.5, color="#7209B7")
    if "Q1_loss" in history and any(history["Q1_loss"]):
        ax8.plot(history["Q1_loss"], label="Q1 Loss", alpha=0.8, linewidth=1.5, color="#2E86AB")
    if "Q2_loss" in history and any(history["Q2_loss"]):
        ax8.plot(history["Q2_loss"], label="Q2 Loss", alpha=0.8, linewidth=1.5, color="#A23B72")
    if "fractal_loss" in history and any(history["fractal_loss"]):
        ax8.plot(
            history["fractal_loss"], label="Fractal Loss", alpha=0.8, linewidth=1.5, color="#F72585"
        )
    if "height_loss" in history and any(history["height_loss"]):
        ax8.plot(
            history["height_loss"], label="Height Loss", alpha=0.8, linewidth=1.5, color="#06A77D"
        )
    ax8.set_xlabel("Step", fontsize=10)
    ax8.set_ylabel("Loss", fontsize=10)
    ax8.set_title("Loss Components Breakdown", fontsize=12, fontweight="bold")
    ax8.legend(fontsize=7, ncol=2)
    ax8.grid(True, alpha=0.3)
    ax8.set_yscale("log")

    # 9. Calibration metrics (ECE, Brier Score)
    ax9 = fig.add_subplot(gs[2, 2])
    if "ece" in history and any(history["ece"]):
        ax9.plot(
            history["ece"],
            label="ECE (Expected Calibration Error)",
            alpha=0.8,
            linewidth=2,
            color="#F18F01",
        )
    if "brier_score" in history and any(history["brier_score"]):
        ax9_twin = ax9.twinx()
        ax9_twin.plot(
            history["brier_score"],
            label="Brier Score",
            alpha=0.8,
            linewidth=2,
            color="#A23B72",
            linestyle="--",
        )
        ax9_twin.set_ylabel("Brier Score", fontsize=10, color="#A23B72")
        ax9_twin.tick_params(axis="y", labelcolor="#A23B72")
        ax9_twin.legend(fontsize=9, loc="upper right")
    ax9.set_xlabel("Step", fontsize=10)
    ax9.set_ylabel("ECE", fontsize=10, color="#F18F01")
    ax9.tick_params(axis="y", labelcolor="#F18F01")
    ax9.set_title("Calibration Metrics", fontsize=12, fontweight="bold")
    ax9.legend(fontsize=9, loc="upper left")
    ax9.grid(True, alpha=0.3)

    # 10. Q1/Q2 Entropy (Collapse detection)
    ax10 = fig.add_subplot(gs[3, 0])
    if "Q1_entropy" in history and any(history["Q1_entropy"]):
        ax10.plot(
            history["Q1_entropy"], label="Q1 Entropy", alpha=0.8, linewidth=2, color="#2E86AB"
        )
    if "Q2_entropy" in history and any(history["Q2_entropy"]):
        ax10.plot(
            history["Q2_entropy"], label="Q2 Entropy", alpha=0.8, linewidth=2, color="#A23B72"
        )
    ax10.axhline(
        y=0.1, color="red", linestyle="--", alpha=0.5, linewidth=1, label="Saturation threshold"
    )
    ax10.set_xlabel("Step", fontsize=10)
    ax10.set_ylabel("Entropy", fontsize=10)
    ax10.set_title("Q1/Q2 Entropy (Collapse Detection)", fontsize=12, fontweight="bold")
    ax10.legend(fontsize=9)
    ax10.grid(True, alpha=0.3)
    ax10.set_ylim([0, 1])

    # 11. Q1/Q2 Ranges (Distribution width)
    ax11 = fig.add_subplot(gs[3, 1])
    if "Q1_range" in history and any(history["Q1_range"]):
        ax11.plot(
            history["Q1_range"], label="Q1 Range (max-min)", alpha=0.8, linewidth=2, color="#2E86AB"
        )
    if "Q2_range" in history and any(history["Q2_range"]):
        ax11.plot(
            history["Q2_range"], label="Q2 Range (max-min)", alpha=0.8, linewidth=2, color="#A23B72"
        )
    ax11.axhline(
        y=0.01, color="red", linestyle="--", alpha=0.5, linewidth=1, label="Collapse threshold"
    )
    ax11.set_xlabel("Step", fontsize=10)
    ax11.set_ylabel("Range", fontsize=10)
    ax11.set_title("Q1/Q2 Distribution Width", fontsize=12, fontweight="bold")
    ax11.legend(fontsize=9)
    ax11.grid(True, alpha=0.3)
    ax11.set_ylim([0, 1])

    # 12. Perplexity
    ax12 = fig.add_subplot(gs[3, 2])
    if "eval_perplexity" in history and history["eval_perplexity"]:
        eval_steps = np.linspace(0, len(history["train_loss"]), len(history["eval_perplexity"]))
        ax12.plot(
            eval_steps,
            history["eval_perplexity"],
            label="Eval Perplexity",
            marker="o",
            markersize=4,
            linewidth=2,
            color="#7209B7",
        )
    ax12.set_xlabel("Step", fontsize=10)
    ax12.set_ylabel("Perplexity", fontsize=10)
    ax12.set_title("Evaluation Perplexity", fontsize=12, fontweight="bold")
    ax12.legend(fontsize=9)
    ax12.grid(True, alpha=0.3)

    plt.suptitle("Pyramidal Q1/Q2/Fractal Training Curves", fontsize=16, fontweight="bold", y=0.995)
    plt.savefig(save_dir / "training_curves.png", dpi=300, bbox_inches="tight")
    print(f"📊 Training curves saved to {save_dir / 'training_curves.png'}")
    plt.close()


def main():
    args = parse_args()

    # Setup
    set_seed(args.seed)
    device = get_device()

    # Create experiment directory
    exp_dir = Path("experiments/level1/runs") / args.experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(exp_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Tensorboard
    writer = SummaryWriter(log_dir=str(exp_dir / "tensorboard"))

    effective_batch_size = args.batch_size * args.gradient_accumulation_steps

    print("=" * 80)
    print("🔻 PYRAMIDAL Q1/Q2/FRACTAL TRAINING")
    print("=" * 80)
    print(f"Experiment: {args.experiment_name}")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset.upper()}")
    print("\nTraining Configuration:")
    print(
        f"  Batch size: {args.batch_size} (effective: {effective_batch_size} with {args.gradient_accumulation_steps}x accumulation)"
    )
    if args.gradient_checkpointing:
        print("  Gradient checkpointing: enabled")
    if args.fp16:
        print("  Mixed precision (fp16): enabled")
    if args.resume_from:
        print(f"  Resuming from: {args.resume_from}")
    print("\nPyramidal Parameters:")
    print(f"  λ_base:    {args.lambda_base}")
    print(f"  λ_Q1:      {args.lambda_Q1}")
    print(f"  λ_Q2:      {args.lambda_Q2}")
    print(f"  λ_fractal: {args.lambda_fractal}")
    print(f"  λ_height:  {args.lambda_height}")
    print("=" * 80)

    # Load tokenizer and vocab_size based on dataset
    if args.dataset == "wikitext":
        print("\n📚 Loading WikiText-2...")
        train_dataset, val_dataset, test_dataset, tokenizer = load_wikitext_dataset(
            max_length=args.max_seq_len, cache_dir=args.data_dir
        )
        vocab_size = tokenizer.vocab_size
        print(f"   Loaded WikiText-2 with vocab size: {vocab_size}")
    else:  # tinystories
        print("\n📚 Loading TinyStories tokenizer...")
        import pickle

        with open(f"{args.data_dir}/tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        vocab_size = len(tokenizer)
        print(f"   Loaded TinyStories tokenizer with vocab size: {vocab_size}")

    # Create or load model
    start_step = 0
    if args.resume_from:
        print(f"\n🔄 Resuming from checkpoint: {args.resume_from}")
        model = AletheionPyramidalQ1Q2Transformer.from_pretrained(args.resume_from)
        model = model.to(device)

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            print("   ✓ Gradient checkpointing enabled")

        # Extract step number from checkpoint name
        checkpoint_name = Path(args.resume_from).name
        if checkpoint_name.startswith("checkpoint_step_"):
            start_step = int(checkpoint_name.split("_")[-1])
            print(f"   ✓ Resuming from step {start_step}")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Model parameters: {total_params:,} (trainable: {trainable_params:,})")
    else:
        print("\n🏗️  Creating pyramidal Q1/Q2 model...")
        model = AletheionPyramidalQ1Q2Transformer(
            vocab_size=vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            max_seq_len=args.max_seq_len,
            dropout=args.dropout,
            lambda_base=args.lambda_base,
            lambda_Q1=args.lambda_Q1,
            lambda_Q2=args.lambda_Q2,
            lambda_fractal=args.lambda_fractal,
            lambda_height=args.lambda_height,
            use_multi_head_height=args.use_multi_head_height,
            max_temperature_scale=args.max_temperature_scale,
        )
        model = model.to(device)

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            print("   ✓ Gradient checkpointing enabled (saves ~40% memory)")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Model parameters: {total_params:,} (trainable: {trainable_params:,})")

    # Note: WikiText-2 datasets already loaded above
    # TinyStories support has been removed

    # Create dataloaders with appropriate collate_fn
    if args.dataset == "wikitext":
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=False,  # WikiText-2 doesn't need pin_memory
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=False,
        )
    else:  # tinystories
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )

    # Learning rate scheduler
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / args.warmup_steps
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and torch.cuda.is_available())
    use_amp = args.fp16 and torch.cuda.is_available()
    if use_amp:
        print("   ✓ GradScaler initialized for mixed precision training")

    # Initialize training history
    history = {
        "train_loss": [],
        "ce_loss": [],
        "base_loss": [],
        "Q1_loss": [],
        "Q2_loss": [],
        "fractal_loss": [],
        "height_loss": [],
        "Q1_mean": [],
        "Q2_mean": [],
        "Q1_target": [],
        "Q2_target": [],
        "Q1_min": [],
        "Q1_max": [],
        "Q2_min": [],
        "Q2_max": [],
        "Q1_range": [],
        "Q2_range": [],
        "height_mean": [],
        "height_target": [],
        "fractal_mean": [],
        "base_stability": [],
        "w_memory": [],
        "w_pain": [],
        "w_choice": [],
        "w_exploration": [],
        "ece": [],
        "brier_score": [],
        "Q1_entropy": [],
        "Q2_entropy": [],
        "eval_loss": [],
        "eval_perplexity": [],
    }

    # Training loop
    global_step = start_step
    start_time = time.time()

    print(f"\n{'='*80}")
    print("Starting training...")
    print(f"{'='*80}\n")

    train_iter = iter(train_loader)

    while global_step < args.max_steps:
        # Accumulate gradients over multiple micro-batches
        accumulated_loss = 0.0

        for accum_step in range(args.gradient_accumulation_steps):
            # Get batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            # Train step (with gradient accumulation)
            is_accumulation_step = accum_step < args.gradient_accumulation_steps - 1
            loss, pyramid_outputs, loss_dict = train_step(
                model,
                batch,
                optimizer,
                scaler,
                device,
                args.grad_clip,
                accumulation_steps=args.gradient_accumulation_steps,
                is_accumulation_step=is_accumulation_step,
                use_amp=use_amp,
            )

            accumulated_loss += loss / args.gradient_accumulation_steps

        # Scheduler step after full accumulation
        scheduler.step()

        # Use accumulated loss for logging
        loss = accumulated_loss

        # Update history with training metrics
        history["train_loss"].append(loss)

        if pyramid_outputs is not None and loss_dict is not None:
            # Q1/Q2 metrics
            Q1 = pyramid_outputs["Q1_mean"]
            Q2 = pyramid_outputs["Q2_mean"]
            height = pyramid_outputs["height"]
            fractal = pyramid_outputs["fractal_uncertainty"]

            history["Q1_mean"].append(Q1.mean().item())
            history["Q2_mean"].append(Q2.mean().item())
            history["Q1_min"].append(Q1.min().item())
            history["Q1_max"].append(Q1.max().item())
            history["Q2_min"].append(Q2.min().item())
            history["Q2_max"].append(Q2.max().item())
            history["Q1_range"].append((Q1.max() - Q1.min()).item())
            history["Q2_range"].append((Q2.max() - Q2.min()).item())

            history["Q1_target"].append(loss_dict.get("target_Q1_mean", 0.0))
            history["Q2_target"].append(loss_dict.get("target_Q2_mean", 0.0))

            history["height_mean"].append(height.mean().item())
            history["height_target"].append(loss_dict.get("target_height_mean", 0.0))
            history["fractal_mean"].append(fractal.mean().item())

            # Base stability and forces
            history["base_stability"].append(loss_dict.get("base_stability", 0.0))
            history["w_memory"].append(loss_dict.get("mean_memory", 0.0))
            history["w_pain"].append(loss_dict.get("mean_pain", 0.0))
            history["w_choice"].append(loss_dict.get("mean_choice", 0.0))
            history["w_exploration"].append(loss_dict.get("mean_exploration", 0.0))

            # Loss components
            history["ce_loss"].append(loss_dict.get("ce_loss", 0.0))
            history["base_loss"].append(loss_dict.get("base_loss", 0.0))
            history["Q1_loss"].append(loss_dict.get("Q1_loss", 0.0))
            history["Q2_loss"].append(loss_dict.get("Q2_loss", 0.0))
            history["fractal_loss"].append(loss_dict.get("fractal_loss", 0.0))
            history["height_loss"].append(loss_dict.get("height_loss", 0.0))

            # Calibration
            history["ece"].append(loss_dict.get("ece", 0.0))
            history["brier_score"].append(loss_dict.get("brier_score", 0.0))

            # Entropy (collapse detection)
            def binary_entropy(p):
                p = p.clamp(1e-8, 1 - 1e-8)
                return -(p * p.log() + (1 - p) * (1 - p).log()).mean().item()

            history["Q1_entropy"].append(binary_entropy(Q1))
            history["Q2_entropy"].append(binary_entropy(Q2))
        else:
            # Fill with zeros if no pyramid outputs
            for key in [
                "Q1_mean",
                "Q2_mean",
                "Q1_min",
                "Q1_max",
                "Q2_min",
                "Q2_max",
                "Q1_range",
                "Q2_range",
                "Q1_target",
                "Q2_target",
                "height_mean",
                "height_target",
                "fractal_mean",
                "base_stability",
                "w_memory",
                "w_pain",
                "w_choice",
                "w_exploration",
                "ce_loss",
                "base_loss",
                "Q1_loss",
                "Q2_loss",
                "fractal_loss",
                "height_loss",
                "ece",
                "brier_score",
                "Q1_entropy",
                "Q2_entropy",
            ]:
                history[key].append(0.0)

        # Log training metrics
        if global_step % 10 == 0:
            lr = scheduler.get_last_lr()[0]
            writer.add_scalar("train/loss", loss, global_step)
            writer.add_scalar("train/lr", lr, global_step)

            # Log pyramidal metrics
            if pyramid_outputs is not None:
                stats = model.get_pyramidal_stats(pyramid_outputs)
                for key, value in stats.items():
                    writer.add_scalar(f"train/pyramid/{key}", value, global_step)

                # Log loss components and calibration metrics if available
                if loss_dict is not None:
                    # Loss components
                    writer.add_scalar("train/loss/ce", loss_dict.get("ce_loss", 0), global_step)
                    writer.add_scalar("train/loss/base", loss_dict.get("base_loss", 0), global_step)
                    writer.add_scalar("train/loss/Q1", loss_dict.get("Q1_loss", 0), global_step)
                    writer.add_scalar("train/loss/Q2", loss_dict.get("Q2_loss", 0), global_step)
                    writer.add_scalar(
                        "train/loss/fractal", loss_dict.get("fractal_loss", 0), global_step
                    )
                    writer.add_scalar(
                        "train/loss/height", loss_dict.get("height_loss", 0), global_step
                    )

                    # Calibration metrics (ECE, Brier)
                    writer.add_scalar("train/calibration/ece", loss_dict.get("ece", 0), global_step)
                    writer.add_scalar(
                        "train/calibration/brier_score",
                        loss_dict.get("brier_score", 0),
                        global_step,
                    )
                    writer.add_scalar(
                        "train/calibration/uncertainty_error_corr",
                        loss_dict.get("uncertainty_error_corr", 0),
                        global_step,
                    )

                    # Force weights (base cognitive forces)
                    writer.add_scalar(
                        "train/forces/memory", loss_dict.get("mean_memory", 0), global_step
                    )
                    writer.add_scalar(
                        "train/forces/pain", loss_dict.get("mean_pain", 0), global_step
                    )
                    writer.add_scalar(
                        "train/forces/choice", loss_dict.get("mean_choice", 0), global_step
                    )
                    writer.add_scalar(
                        "train/forces/exploration",
                        loss_dict.get("mean_exploration", 0),
                        global_step,
                    )
                    writer.add_scalar(
                        "train/forces/base_stability",
                        loss_dict.get("base_stability", 0),
                        global_step,
                    )

                # Check for collapse
                collapse_signals = compute_collapse_signals(pyramid_outputs)
                for key, value in collapse_signals.items():
                    if isinstance(value, bool):
                        writer.add_scalar(f"train/collapse/{key}", int(value), global_step)
                    else:
                        writer.add_scalar(f"train/collapse/{key}", value, global_step)

                # Print status every 50 steps with detailed Q1/Q2 distributions
                if global_step % 50 == 0:
                    elapsed = time.time() - start_time

                    # Extract Q1/Q2 tensors for distribution analysis
                    Q1 = pyramid_outputs["Q1_mean"]
                    Q2 = pyramid_outputs["Q2_mean"]

                    # Compute distributions
                    Q1_min = Q1.min().item()
                    Q1_max = Q1.max().item()
                    Q1_mean = Q1.mean().item()
                    Q2_min = Q2.min().item()
                    Q2_max = Q2.max().item()
                    Q2_mean = Q2.mean().item()

                    # Get target values from loss_dict if available
                    if loss_dict is not None:
                        Q1_target_mean = loss_dict.get("target_Q1_mean", 0.0)
                        Q2_target_mean = loss_dict.get("target_Q2_mean", 0.0)
                    else:
                        Q1_target_mean = 0.0
                        Q2_target_mean = 0.0

                    # Verification checks
                    Q1_collapsed = Q1_min == Q1_max or (Q1_max - Q1_min < 0.01)
                    Q2_collapsed = Q2_min == Q2_max or (Q2_max - Q2_min < 0.01)
                    Q1_Q2_distinct = abs(Q1_mean - Q2_mean) > 0.05

                    # Standard status line
                    print(
                        f"\nStep {global_step}/{args.max_steps} | "
                        f"Loss: {loss:.4f} | "
                        f"Q1: {stats['Q1_mean']:.3f} | "
                        f"Q2: {stats['Q2_mean']:.3f} | "
                        f"Height: {stats['height_mean']:.3f} | "
                        f"Fractal: {stats['fractal_mean']:.3f} | "
                        f"Time: {elapsed:.1f}s"
                    )

                    # Detailed Q1/Q2 distribution diagnostics
                    print(
                        f"  Q1 Distribution: min={Q1_min:.4f}, max={Q1_max:.4f}, mean={Q1_mean:.4f}, target={Q1_target_mean:.4f}"
                    )
                    print(
                        f"  Q2 Distribution: min={Q2_min:.4f}, max={Q2_max:.4f}, mean={Q2_mean:.4f}, target={Q2_target_mean:.4f}"
                    )

                    # Calibration metrics (ECE, Brier)
                    if loss_dict is not None:
                        ece = loss_dict.get("ece", 0)
                        brier = loss_dict.get("brier_score", 0)
                        print(f"  Calibration: ECE={ece:.4f}, Brier={brier:.4f}")

                    # Force weights (base cognitive forces)
                    if loss_dict is not None:
                        mem = loss_dict.get("mean_memory", 0)
                        pain = loss_dict.get("mean_pain", 0)
                        choice = loss_dict.get("mean_choice", 0)
                        explore = loss_dict.get("mean_exploration", 0)
                        base_stab = loss_dict.get("base_stability", 0)
                        print(
                            f"  Forces: Memory={mem:.3f}, Pain={pain:.3f}, Choice={choice:.3f}, Exploration={explore:.3f} (Stability={base_stab:.3f})"
                        )

                    # Verification status
                    print(
                        f"  Q1 Collapsed: {Q1_collapsed} | Q2 Collapsed: {Q2_collapsed} | Q1/Q2 Distinct: {Q1_Q2_distinct}"
                    )

                    # Log distribution metrics to tensorboard
                    writer.add_scalar("train/Q1/min", Q1_min, global_step)
                    writer.add_scalar("train/Q1/max", Q1_max, global_step)
                    writer.add_scalar("train/Q1/range", Q1_max - Q1_min, global_step)
                    writer.add_scalar("train/Q2/min", Q2_min, global_step)
                    writer.add_scalar("train/Q2/max", Q2_max, global_step)
                    writer.add_scalar("train/Q2/range", Q2_max - Q2_min, global_step)
                    if loss_dict is not None:
                        writer.add_scalar("train/Q1/target_mean", Q1_target_mean, global_step)
                        writer.add_scalar("train/Q2/target_mean", Q2_target_mean, global_step)

                    # Warning if collapse detected
                    if collapse_signals["any_collapse"]:
                        print("  ⚠️  WARNING: Collapse signals detected!")
                        for key, value in collapse_signals.items():
                            if isinstance(value, bool) and value:
                                print(f"     - {key}")

                    # Additional warnings for distribution issues
                    if Q1_collapsed:
                        print("  ⚠️  WARNING: Q1 has collapsed to a constant!")
                    if Q2_collapsed:
                        print("  ⚠️  WARNING: Q2 has collapsed to a constant!")
                    if not Q1_Q2_distinct:
                        print("  ⚠️  WARNING: Q1 and Q2 are not distinct!")

        # Aggressive memory cleanup
        if global_step % 5 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc

            gc.collect()

        # Evaluation
        if global_step % args.eval_interval == 0 and global_step > 0:
            print(f"\nEvaluating at step {global_step}...")
            val_loss, val_pyramid_metrics = evaluate(model, val_loader, device)

            # Update history with eval metrics
            history["eval_loss"].append(val_loss)
            history["eval_perplexity"].append(torch.exp(torch.tensor(val_loss)).item())

            writer.add_scalar("val/loss", val_loss, global_step)
            for key, value in val_pyramid_metrics.items():
                if key.startswith("force_"):
                    # Log force weights separately
                    writer.add_scalar(f"val/forces/{key[6:]}", value, global_step)
                elif key in ["ece", "brier_score"]:
                    # Log calibration metrics separately
                    writer.add_scalar(f"val/calibration/{key}", value, global_step)
                else:
                    writer.add_scalar(f"val/pyramid/{key}", value, global_step)

            print(f"Validation Loss: {val_loss:.4f}")
            print(
                f"Val Q1: {val_pyramid_metrics.get('Q1_mean', 0):.3f} | "
                f"Val Q2: {val_pyramid_metrics.get('Q2_mean', 0):.3f} | "
                f"Val Height: {val_pyramid_metrics.get('height_mean', 0):.3f}"
            )

            # Print calibration metrics
            if "ece" in val_pyramid_metrics:
                print(
                    f"Val Calibration: ECE={val_pyramid_metrics['ece']:.4f} | "
                    f"Brier={val_pyramid_metrics['brier_score']:.4f}"
                )

            # Print force weights
            if "force_memory" in val_pyramid_metrics:
                print(
                    f"Val Forces: Memory={val_pyramid_metrics['force_memory']:.3f} | "
                    f"Pain={val_pyramid_metrics['force_pain']:.3f} | "
                    f"Choice={val_pyramid_metrics['force_choice']:.3f} | "
                    f"Exploration={val_pyramid_metrics['force_exploration']:.3f} | "
                    f"Stability={val_pyramid_metrics['force_base_stability']:.3f}\n"
                )

            log_memory(global_step)

        # Save checkpoint
        if global_step % args.save_interval == 0 and global_step > 0:
            checkpoint_path = exp_dir / f"checkpoint_step_{global_step}"
            model.save_pretrained(str(checkpoint_path))
            print(f"💾 Checkpoint saved: {checkpoint_path}")

        global_step += 1

    # Memory cleanup before final evaluation
    print("\n🧹 Cleaning up memory before final evaluation...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc

    gc.collect()
    log_memory(global_step)

    # Final evaluation
    print("\n📊 Final evaluation...")
    val_loss, val_pyramid_metrics = evaluate(model, val_loader, device, max_batches=50)
    print(f"Final Validation Loss: {val_loss:.4f}")
    print("\nFinal Pyramidal Metrics:")
    for key, value in val_pyramid_metrics.items():
        print(f"  {key}: {value:.4f}")

    # Final save
    final_path = exp_dir / "final_model"
    model.save_pretrained(str(final_path))
    print(f"\n✅ Training complete! Final model saved to {final_path}")

    # Save training history
    history_path = exp_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"💾 Training history saved to {history_path}")

    # Plot training curves
    print("\n📊 Generating training curves...")
    plot_training_curves(history, exp_dir)

    # Generate comprehensive training report
    print("\n📄 Generating training report...")
    generate_training_report(history, val_pyramid_metrics, args, exp_dir)

    writer.close()


if __name__ == "__main__":
    main()
