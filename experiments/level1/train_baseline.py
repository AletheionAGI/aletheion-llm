"""Training script for baseline GPT-2 model.

This script trains a standard GPT-2 transformer on WikiText-2 WITHOUT epistemic gates.
It serves as a baseline for comparison with the Pyramidal Epistemology model.

Key differences from pyramidal training:
    - Standard GPT-2 architecture (no pyramidal structure)
    - Simple cross-entropy loss (no VARO, no height/base regularization)
    - No epistemic gates or uncertainty modeling
    - Still tracks calibration metrics (ECE) for comparison

Usage:
    python experiments/level1/train_baseline.py --num-epochs 10 --batch-size 32 --output outputs/baseline
    python experiments/level1/train_baseline.py --steps 100 --dry-run
    python experiments/level1/train_baseline.py --steps 2000 --output-dir outputs/baseline
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import argparse
import gc
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Config, GPT2LMHeadModel

from data.dataset import load_wikitext_dataset
from src import get_device, set_seed
from src.aletheion.loss import compute_calibration_metrics


def collate_fn(batch):
    """Pad variable length sequences."""
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


def create_baseline_model(
    vocab_size: int,
    device: torch.device,
    use_gradient_checkpointing: bool = False,
) -> GPT2LMHeadModel:
    """Create baseline GPT-2 model matching pyramidal architecture size.

    Architecture matches pyramidal model:
        - d_model=512, n_layers=6, n_heads=8, d_ff=2048
        - ~45M parameters (same as pyramidal)
    """
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=512,  # max_seq_len
        n_embd=512,  # d_model
        n_layer=6,  # n_layers
        n_head=8,  # n_heads
        n_inner=2048,  # d_ff (feedforward)
        resid_pdrop=0.1,  # dropout
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        activation_function="gelu_new",
    )

    model = GPT2LMHeadModel(config).to(device)

    # Enable gradient checkpointing if requested
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("   ✓ Gradient checkpointing enabled (saves ~40% memory)")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {n_params / 1e6:.1f}M")

    return model


def train_step(
    model: GPT2LMHeadModel,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    accumulation_steps: int = 1,
    is_accumulation_step: bool = False,
    scaler: GradScaler = None,
    use_amp: bool = False,
) -> dict[str, float]:
    """Perform one training step with optional gradient accumulation and mixed precision.

    Args:
        model: The model to train
        batch: Input batch
        optimizer: Optimizer
        device: Device to use
        accumulation_steps: Number of gradient accumulation steps
        is_accumulation_step: If True, don't zero grads or step optimizer
        scaler: GradScaler for mixed precision training
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
        with autocast():
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss / accumulation_steps
    else:
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss / accumulation_steps

    metrics = {
        "loss": loss.item() * accumulation_steps,  # Report unscaled loss
        "perplexity": (
            math.exp(loss.item() * accumulation_steps)
            if loss.item() * accumulation_steps < 20
            else float("inf")
        ),
    }

    # Backward pass with optional scaling
    if use_amp and scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    # Only clip and step at the end of accumulation
    if not is_accumulation_step:
        if use_amp and scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    # Memory cleanup
    del loss, outputs, input_ids, labels
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


@torch.no_grad()
def evaluate_model(
    model: GPT2LMHeadModel,
    loader: DataLoader,
    device: torch.device,
    compute_calibration: bool = True,
    max_eval_batches: int = 100,
) -> dict[str, float]:
    """Evaluate baseline model and compute metrics using online statistics."""
    model.eval()

    total_loss = 0.0
    total_batches = 0

    # Calibration sampling
    calibration_probs = []
    calibration_targets = []
    max_calibration_samples = 5000

    for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluating", leave=False)):
        if batch_idx >= max_eval_batches:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, labels=labels)

        # Accumulate loss
        total_loss += outputs.loss.item()
        total_batches += 1

        # Sample for calibration
        if compute_calibration and len(calibration_targets) < max_calibration_samples:
            logits = outputs.logits

            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Get probabilities
            probs = F.softmax(shift_logits, dim=-1)

            valid_mask = shift_labels != -100
            if valid_mask.any():
                # Sample randomly to stay under limit
                n_valid = valid_mask.sum().item()
                n_to_sample = min(n_valid, max_calibration_samples - len(calibration_targets))

                if n_to_sample > 0:
                    valid_indices = torch.where(valid_mask.view(-1))[0]
                    sampled_indices = valid_indices[
                        torch.randperm(len(valid_indices))[:n_to_sample]
                    ]

                    probs_flat = probs.view(-1, probs.size(-1))
                    labels_flat = shift_labels.view(-1)

                    calibration_probs.append(probs_flat[sampled_indices].cpu())
                    calibration_targets.append(labels_flat[sampled_indices].cpu())

            del shift_logits, shift_labels, probs, valid_mask

        # Memory cleanup
        del outputs, input_ids, labels

        # Periodic cache clear
        if batch_idx % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    avg_loss = total_loss / total_batches
    perplexity = math.exp(avg_loss)

    metrics = {
        "eval_loss": avg_loss,
        "eval_perplexity": perplexity,
        "eval_batches": total_batches,
    }

    # Calibration metrics
    if compute_calibration and calibration_probs and calibration_targets:
        all_probs_cat = torch.cat(calibration_probs)
        all_targets_cat = torch.cat(calibration_targets)

        # Dummy uncertainty for calibration computation
        dummy_uncertainty = torch.zeros(len(all_targets_cat), 1)

        cal_metrics = compute_calibration_metrics(
            all_probs_cat, all_targets_cat, dummy_uncertainty, n_bins=10
        )
        metrics.update(
            {
                "ece": cal_metrics["ece"],
                "brier_score": cal_metrics["brier_score"],
            }
        )

        # Cleanup
        del (
            all_probs_cat,
            all_targets_cat,
            dummy_uncertainty,
            calibration_probs,
            calibration_targets,
        )

    model.train()
    return metrics


def plot_training_curves(history: dict[str, list], save_dir: Path):
    """Plot training curves for baseline model."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss curves
    axes[0, 0].plot(history["train_loss"], label="Train Loss", alpha=0.7)
    if "eval_loss" in history and history["eval_loss"]:
        eval_steps = np.linspace(0, len(history["train_loss"]), len(history["eval_loss"]))
        axes[0, 0].plot(eval_steps, history["eval_loss"], label="Eval Loss", marker="o")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Loss Curves (Baseline GPT-2)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Perplexity
    if "eval_perplexity" in history and history["eval_perplexity"]:
        eval_steps = np.linspace(0, len(history["train_loss"]), len(history["eval_perplexity"]))
        axes[0, 1].plot(eval_steps, history["eval_perplexity"], marker="o", color="purple")
        axes[0, 1].set_xlabel("Step")
        axes[0, 1].set_ylabel("Perplexity")
        axes[0, 1].set_title("Evaluation Perplexity")
        axes[0, 1].grid(True, alpha=0.3)

    # ECE (Expected Calibration Error)
    if "eval_ece" in history and history["eval_ece"]:
        eval_steps = np.linspace(0, len(history["train_loss"]), len(history["eval_ece"]))
        axes[1, 0].plot(eval_steps, history["eval_ece"], marker="o", color="orange")
        axes[1, 0].set_xlabel("Step")
        axes[1, 0].set_ylabel("ECE")
        axes[1, 0].set_title("Expected Calibration Error")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0.1, color="red", linestyle="--", alpha=0.5, label="Poor calibration")
        axes[1, 0].legend()

    # Brier Score
    if "eval_brier" in history and history["eval_brier"]:
        eval_steps = np.linspace(0, len(history["train_loss"]), len(history["eval_brier"]))
        axes[1, 1].plot(eval_steps, history["eval_brier"], marker="o", color="green")
        axes[1, 1].set_xlabel("Step")
        axes[1, 1].set_ylabel("Brier Score")
        axes[1, 1].set_title("Brier Score (Lower is Better)")
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "training_curves.png", dpi=300, bbox_inches="tight")
    print(f"📊 Training curves saved to {save_dir / 'training_curves.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train baseline GPT-2 model (no epistemic gates)")

    # Training duration - support both steps and epochs
    train_group = parser.add_mutually_exclusive_group()
    train_group.add_argument("--steps", type=int, default=None, help="Number of training steps")
    train_group.add_argument(
        "--num-epochs", type=int, default=None, help="Number of training epochs"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Micro batch size (per gradient accumulation step)",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps (effective batch = batch_size * accum_steps)",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save memory (~40% reduction)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable mixed precision training (fp16) to save memory (~50% reduction)",
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--eval-interval", type=int, default=500, help="Evaluation interval")
    parser.add_argument("--save-interval", type=int, default=2000, help="Checkpoint save interval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Quick test run")
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume training from checkpoint directory (e.g., outputs/baseline/checkpoint-2000)",
    )

    # Output directory - support both --output and --output-dir
    parser.add_argument(
        "--output",
        "--output-dir",
        dest="output_dir",
        type=str,
        default="outputs/baseline",
        help="Output directory",
    )

    args = parser.parse_args()

    # Set default if neither steps nor num_epochs provided
    if args.steps is None and args.num_epochs is None:
        args.steps = 2000
        args.num_epochs = None

    # Setup
    set_seed(args.seed)
    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    effective_batch_size = args.batch_size * args.gradient_accumulation_steps

    print("📊 Training Baseline GPT-2 Model")
    if args.num_epochs is not None:
        print(f"   - Epochs: {args.num_epochs}")
    elif args.steps is not None:
        print(f"   - Steps: {args.steps}")
    print("   - Model: GPT-2 (no epistemic gates)")
    print(
        f"   - Batch size: {args.batch_size} (effective: {effective_batch_size} with {args.gradient_accumulation_steps}x accumulation)"
    )
    if args.gradient_checkpointing:
        print("   - Gradient checkpointing: enabled")
    if args.fp16:
        print("   - Mixed precision (fp16): enabled")
    if args.resume_from:
        print(f"   - Resuming from: {args.resume_from}")
    print(f"   - Output: {output_dir}")
    print(f"   - Device: {device}")

    # Load data
    print("\n📚 Loading WikiText-2...")
    train_dataset, val_dataset, test_dataset, tokenizer = load_wikitext_dataset(
        max_length=512, cache_dir=".cache/wikitext"
    )

    # Convert num_epochs to steps if specified
    if args.num_epochs is not None:
        steps_per_epoch = len(train_dataset) // args.batch_size
        args.steps = args.num_epochs * steps_per_epoch
        print(
            f"   Converting {args.num_epochs} epochs to {args.steps} steps ({steps_per_epoch} steps/epoch)"
        )

    # Adjust intervals for dry-run if needed
    if args.dry_run:
        args.eval_interval = min(args.eval_interval, args.steps // 4)
        args.save_interval = min(args.save_interval, args.steps)
        print(
            f"   Dry-run mode: adjusted intervals (eval={args.eval_interval}, save={args.save_interval})"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    # Create or load model
    start_step = 0
    history = {
        "train_loss": [],
        "train_perplexity": [],
        "eval_loss": [],
        "eval_perplexity": [],
        "eval_ece": [],
        "eval_brier": [],
    }

    if args.resume_from:
        print(f"\n🔄 Resuming from checkpoint: {args.resume_from}")
        model = GPT2LMHeadModel.from_pretrained(args.resume_from).to(device)

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            print("   ✓ Gradient checkpointing enabled")

        # Load history if available
        history_path = Path(args.resume_from).parent / "history.json"
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)
            print(f"   ✓ Loaded training history ({len(history['train_loss'])} steps)")

        # Extract step number from checkpoint name
        checkpoint_name = Path(args.resume_from).name
        if checkpoint_name.startswith("checkpoint-"):
            start_step = int(checkpoint_name.split("-")[1])
            print(f"   ✓ Resuming from step {start_step}")

        n_params = sum(p.numel() for p in model.parameters())
        print(f"   Model parameters: {n_params / 1e6:.1f}M")
    else:
        print("\n🏗️  Creating baseline GPT-2 model...")
        model = create_baseline_model(
            vocab_size=tokenizer.vocab_size,
            device=device,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Mixed precision scaler
    scaler = GradScaler() if args.fp16 and torch.cuda.is_available() else None
    use_amp = args.fp16 and torch.cuda.is_available()
    if use_amp:
        print("   ✓ GradScaler initialized for mixed precision training")

    # Training loop
    print("\n🚀 Starting training...")

    step = start_step
    train_iter = iter(train_loader)

    pbar = tqdm(total=args.steps, initial=start_step, desc="Training")

    while step < args.steps:
        # Accumulate gradients over multiple micro-batches
        accumulated_loss = 0.0
        accumulated_perplexity = 0.0

        for accum_step in range(args.gradient_accumulation_steps):
            # Get batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            # Train step (with gradient accumulation)
            is_accumulation_step = accum_step < args.gradient_accumulation_steps - 1
            metrics = train_step(
                model,
                batch,
                optimizer,
                device,
                accumulation_steps=args.gradient_accumulation_steps,
                is_accumulation_step=is_accumulation_step,
                scaler=scaler,
                use_amp=use_amp,
            )

            accumulated_loss += metrics["loss"] / args.gradient_accumulation_steps
            accumulated_perplexity += metrics["perplexity"] / args.gradient_accumulation_steps

        # Log metrics (after full accumulation)
        history["train_loss"].append(accumulated_loss)
        history["train_perplexity"].append(accumulated_perplexity)

        # Update progress bar
        pbar.set_postfix(
            {
                "loss": f"{accumulated_loss:.4f}",
                "ppl": f"{accumulated_perplexity:.2f}",
            }
        )
        pbar.update(1)

        step += 1

        # Aggressive memory cleanup
        if step % 5 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # Evaluation
        if step % args.eval_interval == 0:
            eval_metrics = evaluate_model(model, val_loader, device, max_eval_batches=100)
            history["eval_loss"].append(eval_metrics["eval_loss"])
            history["eval_perplexity"].append(eval_metrics["eval_perplexity"])

            if "ece" in eval_metrics:
                history["eval_ece"].append(eval_metrics["ece"])
            if "brier_score" in eval_metrics:
                history["eval_brier"].append(eval_metrics["brier_score"])

            print(f"\n📊 Step {step} Evaluation ({eval_metrics['eval_batches']} batches):")
            print(f"   - Loss: {eval_metrics['eval_loss']:.4f}")
            print(f"   - Perplexity: {eval_metrics['eval_perplexity']:.2f}")
            if "ece" in eval_metrics:
                print(f"   - ECE: {eval_metrics['ece']:.4f}")
            if "brier_score" in eval_metrics:
                print(f"   - Brier Score: {eval_metrics['brier_score']:.4f}")
            log_memory(step)

        # Save checkpoint
        if step % args.save_interval == 0:
            checkpoint_dir = output_dir / f"checkpoint-{step}"
            checkpoint_dir.mkdir(exist_ok=True)
            model.save_pretrained(str(checkpoint_dir))

    pbar.close()

    # Memory cleanup before final evaluation
    print("\n🧹 Cleaning up memory before final evaluation...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    log_memory(step)

    # Final evaluation
    print("\n📊 Final evaluation...")
    final_metrics = evaluate_model(model, val_loader, device, max_eval_batches=100)

    print("\n✅ Training complete!")
    print(f"   - Final loss: {final_metrics['eval_loss']:.4f}")
    print(
        f"   - Final perplexity: {final_metrics['eval_perplexity']:.2f} ({final_metrics['eval_batches']} batches)"
    )
    if "ece" in final_metrics:
        print(f"   - Final ECE: {final_metrics['ece']:.4f}")
    if "brier_score" in final_metrics:
        print(f"   - Final Brier Score: {final_metrics['brier_score']:.4f}")

    # Save final model
    final_dir = output_dir / "final"
    final_dir.mkdir(exist_ok=True)
    model.save_pretrained(str(final_dir))

    # Save history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Plot curves
    plot_training_curves(history, output_dir)

    print(f"\n💾 All outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
