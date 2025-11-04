"""Comparison script for Baseline vs Aletheion Level 1.

This script trains both models on WikiText-2 with identical hyperparameters
and compares their performance on:
    - Perplexity (train/val)
    - Expected Calibration Error (ECE)
    - Brier Score
    - Uncertainty-Error Correlation
    - Reliability diagram

Usage:
    python experiments/level1/compare_baseline_aletheion.py --steps 100 --dry-run
    python experiments/level1/compare_baseline_aletheion.py --steps 10000
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import os  # FIX: Added for checkpoint directory creation

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from src import BaselineTransformer, get_device, set_seed
from src.aletheion.model import AletheionTransformer
from src.aletheion.loss import VaroLoss, compute_calibration_metrics
from data.dataset import collate_fn, load_wikitext_dataset


def create_baseline_model(vocab_size: int, device: torch.device) -> BaselineTransformer:
    """Create baseline transformer."""
    return BaselineTransformer(
        vocab_size=vocab_size,
        d_model=512,
        n_layers=6,
        n_heads=8,
        d_ff=2048,
        max_seq_len=512,
        dropout=0.1,
        tie_weights=True,
        use_flash_attention=False
    ).to(device)


def create_aletheion_model(vocab_size: int, device: torch.device) -> AletheionTransformer:
    """Create Aletheion Level 1 transformer."""
    return AletheionTransformer(
        vocab_size=vocab_size,
        d_model=512,
        n_layers=6,
        n_heads=8,
        d_ff=2048,
        max_seq_len=512,
        dropout=0.1,
        tie_weights=True,
        use_flash_attention=False,
        q1_threshold=0.7,
        q2_threshold=0.7,
        base_temperature=1.0,
        n_consensus_heads=4
    ).to(device)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    checkpoint_dir: Path,
    model_name: str
) -> None:
    """FIX: Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer to save
        step: Current training step
        loss: Current loss value
        checkpoint_dir: Directory to save checkpoint
        model_name: Name of the model (baseline or aletheion)
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{model_name}_step{step}.pt"

    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)

    print(f"  💾 Saved {model_name} checkpoint to {checkpoint_path}")


def train_step(
    model: torch.nn.Module,
    batch: Dict,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    varo_loss: VaroLoss = None
) -> Tuple[float, Dict]:
    """Perform one training step.

    Returns:
        Tuple of (loss, stats_dict) where stats_dict contains optional statistics
    """
    model.train()
    optimizer.zero_grad()

    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    # === DEBUG CRÍTICO ===
    vocab_size = model.token_embedding.num_embeddings
    max_token = input_ids.max().item()
    min_token = input_ids.min().item()

    print(f"🔍 Vocab: {vocab_size} | Input range: [{min_token}, {max_token}]")

    if max_token >= vocab_size:
        print(f"❌ ERROR: Token ID {max_token} >= vocab_size {vocab_size}")
        print(f"   Batch shape: {input_ids.shape}")
        print(f"   Invalid count: {(input_ids >= vocab_size).sum().item()}")
        raise ValueError(f"Token {max_token} out of vocab range!")
    # === FIM DEBUG ===

    # Forward pass
    if isinstance(model, AletheionTransformer):
        outputs = model(input_ids, labels=labels, return_uncertainty=True)
    else:
        outputs = model(input_ids, labels=labels)

    # IMPROVED: Collect statistics for Aletheion model
    stats = {}
    if isinstance(model, AletheionTransformer):
        # Extract Q1, Q2, confidence from outputs if available
        if hasattr(outputs, 'q1') and outputs.q1 is not None:
            stats['q1_mean'] = outputs.q1.mean().item()
            stats['q1_std'] = outputs.q1.std().item()
        if hasattr(outputs, 'q2') and outputs.q2 is not None:
            stats['q2_mean'] = outputs.q2.mean().item()
            stats['q2_std'] = outputs.q2.std().item()
        if hasattr(outputs, 'confidence') and outputs.confidence is not None:
            stats['confidence_mean'] = outputs.confidence.mean().item()

    # Compute loss
    if isinstance(model, AletheionTransformer) and varo_loss is not None:
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_uncertainty = outputs.uncertainty[..., :-1, :].contiguous()

        loss_dict = varo_loss(
            logits=shift_logits,
            targets=shift_labels,
            uncertainty=shift_uncertainty
        )
        loss = loss_dict['loss']

        # IMPROVED: Add VARO loss components to stats
        stats['ce_loss'] = loss_dict.get('ce_loss', 0.0)
        stats['varo_loss'] = loss_dict.get('varo_loss', 0.0)
    else:
        loss = outputs.loss

    # Backward pass
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss.item(), stats



def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    compute_calibration: bool = True
) -> Dict[str, float]:
    """Evaluate model and compute metrics."""
    model.eval()
    total_loss = 0.0
    total_batches = 0

    all_probs = []
    all_targets = []
    all_uncertainties = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            try:
                outputs = model(
                    input_ids,
                    labels=labels,
                    return_uncertainty=True
                )
            except TypeError:
                # Baseline doesn't support return_uncertainty
                outputs = model(input_ids, labels=labels)

            total_loss += outputs.loss.item()
            total_batches += 1

            if compute_calibration:
                # Get predictions for last token in each sequence
                logits_last = outputs.logits[:, -1, :]
                probs_last = F.softmax(logits_last, dim=-1)
                targets_last = labels[:, -1]

                valid_mask = targets_last != -100
                if valid_mask.any():
                    all_probs.append(probs_last[valid_mask].cpu())
                    all_targets.append(targets_last[valid_mask].cpu())

                    if isinstance(model, AletheionTransformer):
                        uncertainty_last = outputs.uncertainty[:, -1, :]
                        all_uncertainties.append(uncertainty_last[valid_mask].cpu())

    # Compute basic metrics
    avg_loss = total_loss / max(1, total_batches)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    metrics = {
        "loss": avg_loss,
        "perplexity": perplexity
    }

    # Compute calibration metrics
    if compute_calibration and all_probs:
        all_probs = torch.cat(all_probs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        if isinstance(model, AletheionTransformer) and all_uncertainties:
            all_uncertainties = torch.cat(all_uncertainties, dim=0)
        else:
            all_uncertainties = torch.zeros(all_targets.size(0), 1)

        calib_metrics = compute_calibration_metrics(
            probs=all_probs,
            targets=all_targets,
            uncertainty=all_uncertainties,
            n_bins=10
        )
        metrics.update(calib_metrics)

        if isinstance(model, AletheionTransformer):
            metrics["uncertainty_mean"] = all_uncertainties.mean().item()

    model.train()
    return metrics


def plot_reliability_diagram(
    baseline_metrics: Dict,
    aletheion_metrics: Dict,
    save_path: Path
) -> None:
    """Plot reliability diagram comparing both models."""
    # This is a placeholder - would need to collect binned data during evaluation
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')

    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title('Reliability Diagram')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Reliability diagram saved to {save_path}")


def main(args):
    """Main comparison function."""
    # Setup
    set_seed(42)
    device = get_device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path("experiments/level1/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("BASELINE VS ALETHEION LEVEL 1 COMPARISON")
    print("=" * 80)

    # Load dataset
    print("\n📚 Loading WikiText-2 dataset...")
    train_ds, val_ds, test_ds, tokenizer= load_wikitext_dataset(
        tokenizer_name="gpt2",
        dataset_config="wikitext-2-raw-v1",
        max_length=512
    )
    vocab_size = tokenizer.vocab_size  # Atributo correto do tokenizer GPT-2
    print(f"📊 Vocab size: {vocab_size}")
    print(f"📊 Tokenizer: {tokenizer}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    print(f"   - Training samples: {len(train_ds)}")
    print(f"   - Validation samples: {len(val_ds)}")
    print(f"   - Vocabulary size: {vocab_size}")

    # Create models
    print("\n🏗️  Creating models...")
    baseline_model = create_baseline_model(vocab_size, device)
    aletheion_model = create_aletheion_model(vocab_size, device)

    baseline_params = sum(p.numel() for p in baseline_model.parameters())
    aletheion_params = sum(p.numel() for p in aletheion_model.parameters())
    param_overhead = (aletheion_params - baseline_params) / baseline_params * 100

    print(f"   - Baseline parameters: {baseline_params:,}")
    print(f"   - Aletheion parameters: {aletheion_params:,}")
    print(f"   - Parameter overhead: {param_overhead:.2f}%")

    # Create optimizers
    baseline_opt = torch.optim.AdamW(baseline_model.parameters(), lr=3e-4, weight_decay=0.1)
    aletheion_opt = torch.optim.AdamW(aletheion_model.parameters(), lr=3e-4, weight_decay=0.1)

    # Create VARO loss
    varo_loss = VaroLoss(lambda_varo=0.1, u_star_method='head_variance')

    # FIX: Create checkpoint directory
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"   - Checkpoint directory: {checkpoint_dir}")

    # Training
    print(f"\n🚀 Training for {args.steps} steps...")
    print()

    # FIX: Remove dry-run early exit - let it run but with fewer steps if needed
    actual_steps = 5 if args.dry_run else args.steps
    if args.dry_run:
        print("DRY RUN MODE: Running 5 steps for testing")
        print()

    train_iter = None  # Initialize train_iter

    for step in tqdm(range(actual_steps), desc="Training"):
        # Get batch
        try:
            batch = next(train_iter)
        except (StopIteration, TypeError):  # TypeError for when train_iter is None
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # Train both models
        baseline_loss, baseline_stats = train_step(baseline_model, batch, baseline_opt, device)
        aletheion_loss, aletheion_stats = train_step(aletheion_model, batch, aletheion_opt, device, varo_loss)

        # IMPROVED: Log progress with Q1, Q2 statistics every 50 steps
        if (step + 1) % 50 == 0:
            print(f"\nStep {step + 1}/{actual_steps}")
            print(f"  Baseline loss: {baseline_loss:.4f}")
            print(f"  Aletheion loss: {aletheion_loss:.4f}")

            # IMPROVED: Print Q1, Q2 statistics if available
            if 'q1_mean' in aletheion_stats:
                print(f"  Q₁: mean={aletheion_stats['q1_mean']:.3f}, std={aletheion_stats['q1_std']:.3f}")
            if 'q2_mean' in aletheion_stats:
                print(f"  Q₂: mean={aletheion_stats['q2_mean']:.3f}, std={aletheion_stats['q2_std']:.3f}")
            if 'confidence_mean' in aletheion_stats:
                print(f"  Confidence: {aletheion_stats['confidence_mean']:.3f}")
            if 'varo_loss' in aletheion_stats:
                print(f"  VARO loss: {aletheion_stats['varo_loss']:.4f}")

        # FIX: Save checkpoints every 100 steps
        if (step + 1) % 100 == 0 or (step + 1) == actual_steps:
            print(f"\n💾 Saving checkpoints at step {step + 1}...")
            save_checkpoint(
                baseline_model, baseline_opt, step + 1, baseline_loss,
                checkpoint_dir, "baseline"
            )
            save_checkpoint(
                aletheion_model, aletheion_opt, step + 1, aletheion_loss,
                checkpoint_dir, "aletheion"
            )

    # FIX: Save final models after training
    print("\n💾 Saving final models...")
    final_model_dir = results_dir / "final_models"
    final_model_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        'model_state_dict': baseline_model.state_dict(),
        'optimizer_state_dict': baseline_opt.state_dict(),
        'config': {
            'vocab_size': vocab_size,
            'd_model': 512,
            'n_layers': 6,
            'n_heads': 8,
            'd_ff': 2048,
        }
    }, final_model_dir / "baseline_final.pt")

    torch.save({
        'model_state_dict': aletheion_model.state_dict(),
        'optimizer_state_dict': aletheion_opt.state_dict(),
        'config': {
            'vocab_size': vocab_size,
            'd_model': 512,
            'n_layers': 6,
            'n_heads': 8,
            'd_ff': 2048,
            'q1_threshold': 0.7,
            'q2_threshold': 0.7,
        }
    }, final_model_dir / "aletheion_final.pt")

    print(f"  ✓ Final models saved to {final_model_dir}")

    # Final evaluation
    print("\n📊 Final Evaluation...")
    print("\nBaseline Model:")
    baseline_metrics = evaluate_model(baseline_model, val_loader, device)
    for key, value in baseline_metrics.items():
        print(f"  {key}: {value:.4f}")

    print("\nAletheion Level 1 Model:")
    aletheion_metrics = evaluate_model(aletheion_model, val_loader, device)
    for key, value in aletheion_metrics.items():
        print(f"  {key}: {value:.4f}")

    # Compute improvements
    print("\n📈 Improvements:")
    if 'ece' in baseline_metrics and 'ece' in aletheion_metrics:
        ece_improvement = (baseline_metrics['ece'] - aletheion_metrics['ece']) / baseline_metrics['ece'] * 100
        print(f"  ECE reduction: {ece_improvement:+.1f}%")

    if 'brier_score' in baseline_metrics and 'brier_score' in aletheion_metrics:
        brier_improvement = (baseline_metrics['brier_score'] - aletheion_metrics['brier_score']) / baseline_metrics['brier_score'] * 100
        print(f"  Brier score reduction: {brier_improvement:+.1f}%")

    # Save results
    results = {
        "baseline": baseline_metrics,
        "aletheion": aletheion_metrics,
        "config": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "seed": 42
        }
    }

    results_file = results_dir / f"comparison_steps{args.steps}.pt"
    torch.save(results, results_file)
    print(f"\n💾 Results saved to {results_file}")

    # Plot reliability diagram
    if not args.dry_run:
        plot_reliability_diagram(
            baseline_metrics,
            aletheion_metrics,
            results_dir / f"reliability_diagram_steps{args.steps}.png"
        )

    print("\n✅ Comparison complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Baseline vs Aletheion Level 1")
    parser.add_argument("--steps", type=int, default=1000, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--dry-run", action="store_true", help="Dry run (no actual training)")
    args = parser.parse_args()

    main(args)
