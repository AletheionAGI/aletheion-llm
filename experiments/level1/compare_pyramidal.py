"""Comparison script: Baseline vs Pyramidal Epistemology.

This script trains both models with identical hyperparameters and compares:
    - Perplexity (train/val)
    - Expected Calibration Error (ECE)
    - Height progression (pyramidal only)
    - Base stability (pyramidal only)

Usage:
    python experiments/level1/compare_pyramidal.py --steps 100 --dry-run
    python experiments/level1/compare_pyramidal.py --steps 10000
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import argparse
import os
import json
from pathlib import Path
from typing import Dict, Tuple

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


def train_baseline(
    model: BaselineTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    steps: int,
    eval_interval: int,
    lr: float = 3e-4
) -> Dict:
    """Train baseline model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = {'train_loss': [], 'eval_loss': [], 'eval_perplexity': []}

    step = 0
    train_iter = iter(train_loader)
    pbar = tqdm(total=steps, desc="Training Baseline")

    while step < steps:
        model.train()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, labels=labels)
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        history['train_loss'].append(loss.item())
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        pbar.update(1)
        step += 1

        if step % eval_interval == 0:
            model.eval()
            total_loss = 0.0
            total_batches = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Eval", leave=False):
                    input_ids = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device)
                    outputs = model(input_ids, labels=labels)
                    total_loss += outputs.loss.item()
                    total_batches += 1
            avg_loss = total_loss / total_batches
            perplexity = torch.exp(torch.tensor(avg_loss)).item()
            history['eval_loss'].append(avg_loss)
            history['eval_perplexity'].append(perplexity)
            print(f"\n   Baseline Step {step}: Perplexity={perplexity:.2f}")

    pbar.close()
    return history


def train_pyramidal(
    model: AletheionPyramidalTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    steps: int,
    eval_interval: int,
    pyramid_loss: PyramidalVAROLoss,
    lr: float = 3e-4
) -> Dict:
    """Train pyramidal model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = {
        'train_loss': [], 'eval_loss': [], 'eval_perplexity': [],
        'mean_height': [], 'target_height': [], 'base_stability': [],
        'w_memory': [], 'w_pain': [], 'w_choice': [], 'w_exploration': []
    }

    step = 0
    train_iter = iter(train_loader)
    pbar = tqdm(total=steps, desc="Training Pyramidal")

    while step < steps:
        model.train()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, labels=labels, return_pyramid_state=True)

        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_pyramid = {k: v[..., :-1, :].contiguous() for k, v in outputs.pyramid.items()}

        loss_dict = pyramid_loss(shift_logits, shift_labels, shift_pyramid)
        loss = loss_dict['loss']

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        history['train_loss'].append(loss.item())
        history['mean_height'].append(loss_dict['mean_height'])
        history['target_height'].append(loss_dict['target_height_mean'])
        history['base_stability'].append(loss_dict['base_stability_mean'])

        valid_mask = (labels != -100)
        history['w_memory'].append(outputs.pyramid['w_memory'][..., :-1, :][valid_mask].mean().item())
        history['w_pain'].append(outputs.pyramid['w_pain'][..., :-1, :][valid_mask].mean().item())
        history['w_choice'].append(outputs.pyramid['w_choice'][..., :-1, :][valid_mask].mean().item())
        history['w_exploration'].append(outputs.pyramid['w_exploration'][..., :-1, :][valid_mask].mean().item())

        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'height': f"{loss_dict['mean_height']:.3f}"})
        pbar.update(1)
        step += 1

        if step % eval_interval == 0:
            model.eval()
            total_loss = 0.0
            total_batches = 0
            all_heights = []
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Eval", leave=False):
                    input_ids = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device)
                    outputs = model(input_ids, labels=labels, return_pyramid_state=True)
                    total_loss += outputs.loss.item()
                    total_batches += 1
                    valid_mask = (labels != -100)
                    all_heights.append(outputs.pyramid['height'][valid_mask].mean().item())
            avg_loss = total_loss / total_batches
            perplexity = torch.exp(torch.tensor(avg_loss)).item()
            avg_height = np.mean(all_heights)
            history['eval_loss'].append(avg_loss)
            history['eval_perplexity'].append(perplexity)
            print(f"\n   Pyramidal Step {step}: Perplexity={perplexity:.2f}, Height={avg_height:.3f}")

    pbar.close()
    return history


def plot_comparison(baseline_hist: Dict, pyramidal_hist: Dict, save_dir: Path):
    """Plot comparison between baseline and pyramidal."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Training loss
    axes[0, 0].plot(baseline_hist['train_loss'], label='Baseline', alpha=0.7)
    axes[0, 0].plot(pyramidal_hist['train_loss'], label='Pyramidal', alpha=0.7)
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Validation perplexity
    if baseline_hist['eval_perplexity'] and pyramidal_hist['eval_perplexity']:
        steps_baseline = np.linspace(0, len(baseline_hist['train_loss']), len(baseline_hist['eval_perplexity']))
        steps_pyramidal = np.linspace(0, len(pyramidal_hist['train_loss']), len(pyramidal_hist['eval_perplexity']))
        axes[0, 1].plot(steps_baseline, baseline_hist['eval_perplexity'], label='Baseline', marker='o')
        axes[0, 1].plot(steps_pyramidal, pyramidal_hist['eval_perplexity'], label='Pyramidal', marker='s')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Perplexity')
        axes[0, 1].set_title('Validation Perplexity')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # Height progression (pyramidal only)
    axes[0, 2].plot(pyramidal_hist['mean_height'], label='Mean Height', color='blue', alpha=0.7)
    axes[0, 2].plot(pyramidal_hist['target_height'], label='Target Height', color='green', alpha=0.7)
    axes[0, 2].axhline(y=0.95, color='red', linestyle='--', alpha=0.5, label='Collapse threshold')
    axes[0, 2].set_xlabel('Step')
    axes[0, 2].set_ylabel('Height')
    axes[0, 2].set_title('Height Progression (Pyramidal)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_ylim([0, 1])

    # Base stability (pyramidal only)
    axes[1, 0].plot(pyramidal_hist['base_stability'], color='purple')
    axes[1, 0].axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Target >0.7')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Stability')
    axes[1, 0].set_title('Base Stability (Pyramidal)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Force weights (pyramidal only)
    axes[1, 1].plot(pyramidal_hist['w_memory'], label='Memory', alpha=0.7)
    axes[1, 1].plot(pyramidal_hist['w_pain'], label='Pain', alpha=0.7)
    axes[1, 1].plot(pyramidal_hist['w_choice'], label='Choice', alpha=0.7)
    axes[1, 1].plot(pyramidal_hist['w_exploration'], label='Exploration', alpha=0.7)
    axes[1, 1].axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, label='Balanced')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Weight')
    axes[1, 1].set_title('Force Weights (Pyramidal)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Performance summary
    final_baseline_ppl = baseline_hist['eval_perplexity'][-1] if baseline_hist['eval_perplexity'] else 0
    final_pyramidal_ppl = pyramidal_hist['eval_perplexity'][-1] if pyramidal_hist['eval_perplexity'] else 0
    improvement = ((final_baseline_ppl - final_pyramidal_ppl) / final_baseline_ppl * 100) if final_baseline_ppl > 0 else 0

    axes[1, 2].text(0.1, 0.8, f"Final Perplexity:", fontsize=12, weight='bold')
    axes[1, 2].text(0.1, 0.7, f"  Baseline: {final_baseline_ppl:.2f}", fontsize=11)
    axes[1, 2].text(0.1, 0.6, f"  Pyramidal: {final_pyramidal_ppl:.2f}", fontsize=11)
    axes[1, 2].text(0.1, 0.5, f"  Improvement: {improvement:+.2f}%", fontsize=11,
                   color='green' if improvement > 0 else 'red')
    axes[1, 2].text(0.1, 0.3, f"Final Height: {pyramidal_hist['mean_height'][-1]:.3f}", fontsize=11)
    axes[1, 2].text(0.1, 0.2, f"Final Base Stability: {pyramidal_hist['base_stability'][-1]:.3f}", fontsize=11)
    axes[1, 2].axis('off')
    axes[1, 2].set_title('Summary')

    plt.tight_layout()
    plt.savefig(save_dir / 'comparison.png', dpi=300, bbox_inches='tight')
    print(f"📊 Comparison plot saved to {save_dir / 'comparison.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare Baseline vs Pyramidal models')
    parser.add_argument('--steps', type=int, default=5000, help='Training steps per model')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--lambda-base', type=float, default=0.01, help='Base stability weight')
    parser.add_argument('--lambda-height', type=float, default=0.02, help='Height calibration weight')
    parser.add_argument('--eval-interval', type=int, default=500, help='Evaluation interval')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dry-run', action='store_true', help='Quick test run')
    parser.add_argument('--output-dir', type=str, default='outputs/comparison_pyramidal', help='Output directory')

    args = parser.parse_args()

    if args.dry_run:
        args.steps = 100
        args.eval_interval = 50

    # Setup
    set_seed(args.seed)
    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🔻 Comparing Baseline vs Pyramidal Epistemology")
    print(f"   - Steps: {args.steps}")
    print(f"   - Device: {device}")
    print(f"   - Output: {output_dir}")

    # Load data
    print("\n📚 Loading WikiText-2...")
    train_dataset, val_dataset, tokenizer = load_wikitext_dataset(max_length=512, cache_dir='.cache/wikitext')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Train baseline
    print("\n🏗️  Training Baseline model...")
    baseline_model = BaselineTransformer(
        vocab_size=tokenizer.vocab_size, d_model=512, n_layers=6, n_heads=8, d_ff=2048,
        max_seq_len=512, dropout=0.1, tie_weights=True
    ).to(device)
    baseline_hist = train_baseline(baseline_model, train_loader, val_loader, device, args.steps, args.eval_interval, args.lr)

    # Train pyramidal
    print("\n🔻 Training Pyramidal model...")
    pyramidal_model = AletheionPyramidalTransformer(
        vocab_size=tokenizer.vocab_size, d_model=512, n_layers=6, n_heads=8, d_ff=2048,
        max_seq_len=512, dropout=0.1, tie_weights=True,
        lambda_base=args.lambda_base, lambda_height=args.lambda_height
    ).to(device)
    pyramid_loss = PyramidalVAROLoss(lambda_base=args.lambda_base, lambda_height=args.lambda_height)
    pyramidal_hist = train_pyramidal(pyramidal_model, train_loader, val_loader, device, args.steps, args.eval_interval, pyramid_loss, args.lr)

    # Save results
    with open(output_dir / 'baseline_history.json', 'w') as f:
        json.dump(baseline_hist, f, indent=2)
    with open(output_dir / 'pyramidal_history.json', 'w') as f:
        json.dump(pyramidal_hist, f, indent=2)

    # Plot comparison
    plot_comparison(baseline_hist, pyramidal_hist, output_dir)

    # Print summary
    final_baseline_ppl = baseline_hist['eval_perplexity'][-1] if baseline_hist['eval_perplexity'] else 0
    final_pyramidal_ppl = pyramidal_hist['eval_perplexity'][-1] if pyramidal_hist['eval_perplexity'] else 0
    improvement = ((final_baseline_ppl - final_pyramidal_ppl) / final_baseline_ppl * 100) if final_baseline_ppl > 0 else 0

    print("\n✅ Comparison complete!")
    print(f"   - Baseline perplexity: {final_baseline_ppl:.2f}")
    print(f"   - Pyramidal perplexity: {final_pyramidal_ppl:.2f}")
    print(f"   - Improvement: {improvement:+.2f}%")
    print(f"   - Final height: {pyramidal_hist['mean_height'][-1]:.3f}")
    print(f"   - Final base stability: {pyramidal_hist['base_stability'][-1]:.3f}")
    print(f"\n💾 Results saved to {output_dir}")


if __name__ == '__main__':
    main()
