"""Training script for Pyramidal Epistemology model.

This script trains the AletheionPyramidalTransformer on WikiText-2 with pyramidal
epistemic gates. It tracks:
    - Height progression (should stabilize, not collapse to 1.0)
    - Base stability (should stay >0.7)
    - Individual force weights (Memory, Pain, Choice, Exploration)
    - Perplexity and calibration metrics

Usage:
    python experiments/level1/train_pyramidal.py --steps 100 --dry-run
    python experiments/level1/train_pyramidal.py --steps 10000 --lambda-base 0.01 --lambda-height 0.02
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import argparse
import os
import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from src import BaselineTransformer, get_device, set_seed
from src.aletheion.pyramidal_model import AletheionPyramidalTransformer
from src.aletheion.loss import PyramidalVAROLoss, compute_calibration_metrics
from data.dataset import load_wikitext_dataset


def collate_fn(batch):
    """Pad variable length sequences."""
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {"input_ids": input_ids_padded, "labels": labels_padded}


def create_pyramidal_model(
    vocab_size: int,
    device: torch.device,
    lambda_base: float = 0.01,
    lambda_height: float = 0.02,
    height_method: str = 'error_based',
    use_multi_head_height: bool = False,
    modulate_temperature: bool = True
) -> AletheionPyramidalTransformer:
    """Create pyramidal transformer."""
    return AletheionPyramidalTransformer(
        vocab_size=vocab_size,
        d_model=512,
        n_layers=6,
        n_heads=8,
        d_ff=2048,
        max_seq_len=512,
        dropout=0.1,
        tie_weights=True,
        use_flash_attention=False,
        lambda_base=lambda_base,
        lambda_height=lambda_height,
        height_method=height_method,
        use_multi_head_height=use_multi_head_height,
        modulate_temperature=modulate_temperature
    ).to(device)


def train_step(
    model: AletheionPyramidalTransformer,
    batch: Dict,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pyramid_loss: PyramidalVAROLoss
) -> Dict[str, float]:
    """Perform one training step and return diagnostics."""
    model.train()
    optimizer.zero_grad()

    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    # Forward pass with pyramidal state
    outputs = model(input_ids, labels=labels, return_pyramid_state=True)

    metrics: Dict[str, float] = {}

    # Compute pyramidal VARO loss
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Shift pyramid outputs
    shift_pyramid = {}
    for key, value in outputs.pyramid.items():
        shift_pyramid[key] = value[..., :-1, :].contiguous()

    loss_dict = pyramid_loss(
        logits=shift_logits,
        targets=shift_labels,
        pyramid_outputs=shift_pyramid
    )

    loss = loss_dict['loss']

    # Collect metrics
    metrics.update({
        "loss": loss.item(),
        "ce_loss": loss_dict['ce_loss'],
        "base_loss": loss_dict['base_loss'],
        "height_loss": loss_dict['height_loss'],
        "mean_height": loss_dict['mean_height'],
        "target_height": loss_dict['target_height_mean'],
        "base_stability": loss_dict['base_stability_mean'],
        "lambda_base": loss_dict['lambda_base'],
        "lambda_height": loss_dict['lambda_height'],
    })

    # Individual force weights
    if outputs.pyramid is not None:
        # Shift the mask to align with shifted pyramid outputs [..., :-1, :]
        valid_mask = (labels[..., 1:] != -100)
        metrics["w_memory"] = outputs.pyramid['w_memory'][..., :-1, :][valid_mask].mean().item()
        metrics["w_pain"] = outputs.pyramid['w_pain'][..., :-1, :][valid_mask].mean().item()
        metrics["w_choice"] = outputs.pyramid['w_choice'][..., :-1, :][valid_mask].mean().item()
        metrics["w_exploration"] = outputs.pyramid['w_exploration'][..., :-1, :][valid_mask].mean().item()
        metrics["uncertainty"] = outputs.pyramid['uncertainty'][..., :-1, :][valid_mask].mean().item()

    # Backward pass
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return metrics


def evaluate_model(
    model: AletheionPyramidalTransformer,
    loader: DataLoader,
    device: torch.device,
    compute_calibration: bool = True
) -> Dict[str, float]:
    """Evaluate model and compute metrics."""
    model.eval()
    total_loss = 0.0
    total_batches = 0

    # Pyramidal metrics
    all_heights = []
    all_uncertainties = []
    all_base_stabilities = []
    all_w_memory = []
    all_w_pain = []
    all_w_choice = []
    all_w_exploration = []

    # Calibration metrics
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, labels=labels, return_pyramid_state=True)

            total_loss += outputs.loss.item()
            total_batches += 1

            # Collect pyramidal metrics
            if outputs.pyramid is not None:
                valid_mask = (labels != -100)

                all_heights.append(outputs.pyramid['height'][valid_mask].cpu())
                all_uncertainties.append(outputs.pyramid['uncertainty'][valid_mask].cpu())
                all_base_stabilities.append(outputs.pyramid['base_stability'][valid_mask].cpu())
                all_w_memory.append(outputs.pyramid['w_memory'][valid_mask].cpu())
                all_w_pain.append(outputs.pyramid['w_pain'][valid_mask].cpu())
                all_w_choice.append(outputs.pyramid['w_choice'][valid_mask].cpu())
                all_w_exploration.append(outputs.pyramid['w_exploration'][valid_mask].cpu())

            # Collect for calibration
            if compute_calibration:
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                probs = F.softmax(shift_logits, dim=-1)

                valid_mask = (shift_labels != -100)
                if valid_mask.any():
                    all_probs.append(probs[valid_mask].cpu())
                    all_targets.append(shift_labels[valid_mask].cpu())

    avg_loss = total_loss / total_batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    metrics = {
        "eval_loss": avg_loss,
        "eval_perplexity": perplexity,
    }

    # Pyramidal metrics
    if all_heights:
        all_heights_cat = torch.cat(all_heights)
        all_uncertainties_cat = torch.cat(all_uncertainties)
        all_base_stabilities_cat = torch.cat(all_base_stabilities)

        metrics.update({
            "height_mean": all_heights_cat.mean().item(),
            "height_std": all_heights_cat.std().item(),
            "height_min": all_heights_cat.min().item(),
            "height_max": all_heights_cat.max().item(),
            "uncertainty_mean": all_uncertainties_cat.mean().item(),
            "base_stability_mean": all_base_stabilities_cat.mean().item(),
            "w_memory_mean": torch.cat(all_w_memory).mean().item(),
            "w_pain_mean": torch.cat(all_w_pain).mean().item(),
            "w_choice_mean": torch.cat(all_w_choice).mean().item(),
            "w_exploration_mean": torch.cat(all_w_exploration).mean().item(),
        })

    # Calibration metrics
    if compute_calibration and all_probs and all_targets:
        all_probs_cat = torch.cat(all_probs)
        all_targets_cat = torch.cat(all_targets)

        # Sample for calibration (too expensive to compute on all tokens)
        sample_size = min(10000, len(all_targets_cat))
        indices = torch.randperm(len(all_targets_cat))[:sample_size]

        # We don't have uncertainty at eval, so use dummy
        dummy_uncertainty = torch.zeros(sample_size, 1)

        cal_metrics = compute_calibration_metrics(
            all_probs_cat[indices],
            all_targets_cat[indices],
            dummy_uncertainty,
            n_bins=10
        )
        metrics.update({
            "ece": cal_metrics['ece'],
            "brier_score": cal_metrics['brier_score'],
        })

    return metrics


def plot_training_curves(history: Dict[str, list], save_dir: Path):
    """Plot training curves."""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))

    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train Loss', alpha=0.7)
    if 'eval_loss' in history and history['eval_loss']:
        eval_steps = np.linspace(0, len(history['train_loss']), len(history['eval_loss']))
        axes[0, 0].plot(eval_steps, history['eval_loss'], label='Eval Loss', marker='o')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Height progression
    axes[0, 1].plot(history['mean_height'], label='Mean Height', color='blue', alpha=0.7)
    axes[0, 1].plot(history['target_height'], label='Target Height', color='green', alpha=0.7)
    axes[0, 1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Mid-pyramid')
    axes[0, 1].axhline(y=0.95, color='red', linestyle='--', alpha=0.5, label='Collapse threshold')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Height')
    axes[0, 1].set_title('Height Progression (Watch for Collapse!)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])

    # Base stability
    axes[1, 0].plot(history['base_stability'], label='Base Stability', color='purple')
    axes[1, 0].axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Target >0.7')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Stability')
    axes[1, 0].set_title('Base Stability')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Force weights
    axes[1, 1].plot(history['w_memory'], label='Memory', alpha=0.7)
    axes[1, 1].plot(history['w_pain'], label='Pain', alpha=0.7)
    axes[1, 1].plot(history['w_choice'], label='Choice', alpha=0.7)
    axes[1, 1].plot(history['w_exploration'], label='Exploration', alpha=0.7)
    axes[1, 1].axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, label='Balanced (0.25)')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Weight')
    axes[1, 1].set_title('Force Weights (4 Vertices)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Loss components
    axes[2, 0].plot(history['ce_loss'], label='CE Loss', alpha=0.7)
    axes[2, 0].plot(history['base_loss'], label=f'Base Loss (λ={history["lambda_base"][-1]:.3f})', alpha=0.7)
    axes[2, 0].plot(history['height_loss'], label=f'Height Loss (λ={history["lambda_height"][-1]:.3f})', alpha=0.7)
    axes[2, 0].set_xlabel('Step')
    axes[2, 0].set_ylabel('Loss')
    axes[2, 0].set_title('Loss Components')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].set_yscale('log')

    # Uncertainty
    axes[2, 1].plot(history['uncertainty'], label='Mean Uncertainty', color='orange')
    axes[2, 1].set_xlabel('Step')
    axes[2, 1].set_ylabel('Uncertainty')
    axes[2, 1].set_title('Epistemic Uncertainty')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    print(f"📊 Training curves saved to {save_dir / 'training_curves.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train Pyramidal Epistemology model')
    parser.add_argument('--steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--lambda-base', type=float, default=0.01, help='Base stability weight')
    parser.add_argument('--lambda-height', type=float, default=0.02, help='Height calibration weight')
    parser.add_argument('--height-method', type=str, default='error_based',
                       choices=['error_based', 'entropy_based', 'loss_based'],
                       help='Method for computing target height')
    parser.add_argument('--multi-head-height', action='store_true', help='Use multi-head height consensus')
    parser.add_argument('--no-temp-modulation', action='store_true', help='Disable temperature modulation')
    parser.add_argument('--eval-interval', type=int, default=500, help='Evaluation interval')
    parser.add_argument('--save-interval', type=int, default=2000, help='Checkpoint save interval')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dry-run', action='store_true', help='Quick test run')
    parser.add_argument('--output-dir', type=str, default='outputs/pyramidal', help='Output directory')

    args = parser.parse_args()

    if args.dry_run:
        args.steps = 100
        args.eval_interval = 50
        args.save_interval = 100

    # Setup
    set_seed(args.seed)
    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"🔻 Training Pyramidal Epistemology Model")
    print(f"   - Steps: {args.steps}")
    print(f"   - λ_base: {args.lambda_base}")
    print(f"   - λ_height: {args.lambda_height}")
    print(f"   - Height method: {args.height_method}")
    print(f"   - Output: {output_dir}")
    print(f"   - Device: {device}")

    # Load data
    print("\n📚 Loading WikiText-2...")
    train_dataset, val_dataset, test_dataset, tokenizer = load_wikitext_dataset(
        max_length=512,
        cache_dir='.cache/wikitext'
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # Create model
    print("\n🏗️  Creating pyramidal model...")
    model = create_pyramidal_model(
        vocab_size=tokenizer.vocab_size,
        device=device,
        lambda_base=args.lambda_base,
        lambda_height=args.lambda_height,
        height_method=args.height_method,
        use_multi_head_height=args.multi_head_height,
        modulate_temperature=not args.no_temp_modulation
    )

    # Create loss
    pyramid_loss = PyramidalVAROLoss(
        lambda_base=args.lambda_base,
        lambda_height=args.lambda_height,
        height_method=args.height_method
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Training loop
    print("\n🚀 Starting training...")
    history = {
        'train_loss': [],
        'ce_loss': [],
        'base_loss': [],
        'height_loss': [],
        'mean_height': [],
        'target_height': [],
        'base_stability': [],
        'w_memory': [],
        'w_pain': [],
        'w_choice': [],
        'w_exploration': [],
        'uncertainty': [],
        'lambda_base': [],
        'lambda_height': [],
        'eval_loss': [],
        'eval_perplexity': []
    }

    step = 0
    train_iter = iter(train_loader)

    pbar = tqdm(total=args.steps, desc="Training")

    while step < args.steps:
        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # Train step
        metrics = train_step(model, batch, optimizer, device, pyramid_loss)

        # Log metrics
        for key in history.keys():
            if key in metrics:
                history[key].append(metrics[key])
            elif key.startswith('eval_'):
                pass  # Skip eval metrics during training
            else:
                history[key].append(0.0)

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{metrics['loss']:.4f}",
            'height': f"{metrics['mean_height']:.3f}",
            'base_stab': f"{metrics['base_stability']:.3f}"
        })
        pbar.update(1)

        step += 1

        # Evaluation
        if step % args.eval_interval == 0:
            eval_metrics = evaluate_model(model, val_loader, device)
            history['eval_loss'].append(eval_metrics['eval_loss'])
            history['eval_perplexity'].append(eval_metrics['eval_perplexity'])

            print(f"\n📊 Step {step} Evaluation:")
            print(f"   - Perplexity: {eval_metrics['eval_perplexity']:.2f}")
            print(f"   - Height: {eval_metrics.get('height_mean', 0):.3f} ± {eval_metrics.get('height_std', 0):.3f}")
            print(f"   - Base stability: {eval_metrics.get('base_stability_mean', 0):.3f}")
            if 'ece' in eval_metrics:
                print(f"   - ECE: {eval_metrics['ece']:.4f}")

        # Save checkpoint
        if step % args.save_interval == 0:
            checkpoint_dir = output_dir / f'checkpoint-{step}'
            model.save_pretrained(str(checkpoint_dir))

    pbar.close()

    # Final evaluation
    print("\n📊 Final evaluation...")
    final_metrics = evaluate_model(model, val_loader, device)

    print("\n✅ Training complete!")
    print(f"   - Final perplexity: {final_metrics['eval_perplexity']:.2f}")
    print(f"   - Final height: {final_metrics.get('height_mean', 0):.3f} ± {final_metrics.get('height_std', 0):.3f}")
    print(f"   - Final base stability: {final_metrics.get('base_stability_mean', 0):.3f}")
    if 'ece' in final_metrics:
        print(f"   - Final ECE: {final_metrics['ece']:.4f}")

    # Save final model
    model.save_pretrained(str(output_dir / 'final'))

    # Save history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Plot curves
    plot_training_curves(history, output_dir)

    print(f"\n💾 All outputs saved to {output_dir}")


if __name__ == '__main__':
    main()
