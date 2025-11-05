"""Training script for Pyramidal Q1/Q2/Fractal architecture.

This script trains the complete pyramidal epistemic model with:
- Q1 (aleatoric uncertainty) + variance
- Q2 (epistemic uncertainty) + variance
- Fractal meta-epistemic layer
- Height derived from Q1, Q2, base_stability
- Full VARO loss with all components

Monitors all epistemic metrics to detect collapse (Q1 → 0.88+)

Usage:
    python experiments/level1/train_pyramidal_q1q2.py \
        --lambda_Q1 0.015 \
        --lambda_Q2 0.020 \
        --lambda_fractal 0.005 \
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
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.aletheion.pyramidal_q1q2_model import AletheionPyramidalQ1Q2Transformer
from data.dataset import TinyStoriesDataset
from src.utils import get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Train Pyramidal Q1/Q2/Fractal model')

    # Model architecture
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=1024, help='Feedforward dimension')
    parser.add_argument('--max_seq_len', type=int, default=256, help='Max sequence length')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')

    # Pyramidal Q1/Q2 parameters
    parser.add_argument('--lambda_base', type=float, default=0.01, help='Base stability weight')
    parser.add_argument('--lambda_Q1', type=float, default=0.015, help='Q1 calibration weight')
    parser.add_argument('--lambda_Q2', type=float, default=0.020, help='Q2 calibration weight')
    parser.add_argument('--lambda_fractal', type=float, default=0.005, help='Fractal regularization weight')
    parser.add_argument('--lambda_height', type=float, default=0.02, help='Height calibration weight')
    parser.add_argument('--use_multi_head_height', action='store_true', help='Use multi-head height')
    parser.add_argument('--max_temperature_scale', type=float, default=2.0, help='Max temperature scale')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--max_steps', type=int, default=5000, help='Max training steps')
    parser.add_argument('--warmup_steps', type=int, default=500, help='Warmup steps')
    parser.add_argument('--eval_interval', type=int, default=100, help='Evaluation interval')
    parser.add_argument('--save_interval', type=int, default=500, help='Save checkpoint interval')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')

    # Data
    parser.add_argument('--data_dir', type=str, default='data/tinystories', help='Data directory')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data workers')

    # Experiment
    parser.add_argument('--experiment_name', type=str, default='pyramidal_q1q2_v1',
                       help='Experiment name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--resume_from', type=str, default=None, help='Resume from checkpoint')

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
    Q1 = pyramid_outputs['Q1_mean']
    Q2 = pyramid_outputs['Q2_mean']
    height = pyramid_outputs['height']
    fractal = pyramid_outputs['fractal_uncertainty']

    # Binary entropy: H(p) = -p*log(p) - (1-p)*log(1-p)
    def binary_entropy(p):
        p = p.clamp(1e-8, 1-1e-8)
        return -(p * p.log() + (1-p) * (1-p).log()).mean()

    Q1_entropy = binary_entropy(Q1)
    Q2_entropy = binary_entropy(Q2)
    height_entropy = binary_entropy(height)

    # Collapse indicators
    Q1_mean = Q1.mean().item()
    Q2_mean = Q2.mean().item()
    height_mean = height.mean().item()
    fractal_mean = fractal.mean().item()

    signals['Q1_collapse'] = Q1_mean < 0.05 or Q1_mean > 0.90
    signals['Q2_collapse'] = Q2_mean < 0.05 or Q2_mean > 0.90
    signals['height_collapse'] = height_mean > 0.95
    signals['fractal_explosion'] = fractal_mean > 0.8
    signals['Q1_saturated'] = Q1_entropy.item() < 0.1
    signals['Q2_saturated'] = Q2_entropy.item() < 0.1

    # Overall health
    signals['any_collapse'] = any([
        signals['Q1_collapse'],
        signals['Q2_collapse'],
        signals['height_collapse'],
        signals['fractal_explosion'],
        signals['Q1_saturated'],
        signals['Q2_saturated']
    ])

    # Add raw metrics
    signals['Q1_entropy'] = Q1_entropy.item()
    signals['Q2_entropy'] = Q2_entropy.item()
    signals['height_entropy'] = height_entropy.item()

    return signals


def train_step(model, batch, optimizer, scaler, device, grad_clip):
    """Single training step."""
    model.train()

    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)

    # Forward pass with mixed precision
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        outputs = model(input_ids, labels=labels, return_dict=True)
        loss = outputs.loss

    # Backward pass
    optimizer.zero_grad()
    if torch.cuda.is_available():
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    return loss.item(), outputs.pyramid


@torch.no_grad()
def evaluate(model, dataloader, device, max_batches=10):
    """Evaluate model."""
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    all_pyramid_outputs = []

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break

        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, labels=labels, return_dict=True)
        loss = outputs.loss

        # Accumulate
        batch_tokens = (labels != -100).sum().item()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens
        all_pyramid_outputs.append(outputs.pyramid)

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0

    # Aggregate pyramidal metrics
    pyramid_metrics = {}
    if all_pyramid_outputs and all_pyramid_outputs[0] is not None:
        # Average over batches
        for key in all_pyramid_outputs[0].keys():
            values = [p[key].mean().item() for p in all_pyramid_outputs]
            pyramid_metrics[key] = sum(values) / len(values)

    return avg_loss, pyramid_metrics


def main():
    args = parse_args()

    # Setup
    set_seed(args.seed)
    device = get_device()

    # Create experiment directory
    exp_dir = Path('experiments/level1/runs') / args.experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Tensorboard
    writer = SummaryWriter(log_dir=str(exp_dir / 'tensorboard'))

    print("=" * 80)
    print("🔻 PYRAMIDAL Q1/Q2/FRACTAL TRAINING")
    print("=" * 80)
    print(f"Experiment: {args.experiment_name}")
    print(f"Device: {device}")
    print(f"\nPyramidal Parameters:")
    print(f"  λ_base:    {args.lambda_base}")
    print(f"  λ_Q1:      {args.lambda_Q1}")
    print(f"  λ_Q2:      {args.lambda_Q2}")
    print(f"  λ_fractal: {args.lambda_fractal}")
    print(f"  λ_height:  {args.lambda_height}")
    print("=" * 80)

    # Load tokenizer
    import pickle
    with open('data/tinystories/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    vocab_size = len(tokenizer)

    # Create model
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
        max_temperature_scale=args.max_temperature_scale
    )
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,} (trainable: {trainable_params:,})")

    # Load datasets
    train_dataset = TinyStoriesDataset(
        data_dir=args.data_dir,
        split='train',
        max_length=args.max_seq_len
    )
    val_dataset = TinyStoriesDataset(
        data_dir=args.data_dir,
        split='validation',
        max_length=args.max_seq_len
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )

    # Learning rate scheduler
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / args.warmup_steps
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # Training loop
    global_step = 0
    start_time = time.time()

    print(f"\n{'='*80}")
    print("Starting training...")
    print(f"{'='*80}\n")

    while global_step < args.max_steps:
        for batch in train_loader:
            if global_step >= args.max_steps:
                break

            # Training step
            loss, pyramid_outputs = train_step(
                model, batch, optimizer, scaler, device, args.grad_clip
            )
            scheduler.step()

            # Log training metrics
            if global_step % 10 == 0:
                lr = scheduler.get_last_lr()[0]
                writer.add_scalar('train/loss', loss, global_step)
                writer.add_scalar('train/lr', lr, global_step)

                # Log pyramidal metrics
                if pyramid_outputs is not None:
                    stats = model.get_pyramidal_stats(pyramid_outputs)
                    for key, value in stats.items():
                        writer.add_scalar(f'train/pyramid/{key}', value, global_step)

                    # Check for collapse
                    collapse_signals = compute_collapse_signals(pyramid_outputs)
                    for key, value in collapse_signals.items():
                        if isinstance(value, bool):
                            writer.add_scalar(f'train/collapse/{key}', int(value), global_step)
                        else:
                            writer.add_scalar(f'train/collapse/{key}', value, global_step)

                    # Print status
                    if global_step % 50 == 0:
                        elapsed = time.time() - start_time
                        print(f"Step {global_step}/{args.max_steps} | "
                              f"Loss: {loss:.4f} | "
                              f"Q1: {stats['Q1_mean']:.3f} | "
                              f"Q2: {stats['Q2_mean']:.3f} | "
                              f"Height: {stats['height_mean']:.3f} | "
                              f"Fractal: {stats['fractal_mean']:.3f} | "
                              f"Time: {elapsed:.1f}s")

                        # Warning if collapse detected
                        if collapse_signals['any_collapse']:
                            print("⚠️  WARNING: Collapse signals detected!")
                            for key, value in collapse_signals.items():
                                if isinstance(value, bool) and value:
                                    print(f"   - {key}")

            # Evaluation
            if global_step % args.eval_interval == 0 and global_step > 0:
                print(f"\nEvaluating at step {global_step}...")
                val_loss, val_pyramid_metrics = evaluate(model, val_loader, device)

                writer.add_scalar('val/loss', val_loss, global_step)
                for key, value in val_pyramid_metrics.items():
                    writer.add_scalar(f'val/pyramid/{key}', value, global_step)

                print(f"Validation Loss: {val_loss:.4f}")
                print(f"Val Q1: {val_pyramid_metrics.get('Q1_mean', 0):.3f} | "
                      f"Val Q2: {val_pyramid_metrics.get('Q2_mean', 0):.3f} | "
                      f"Val Height: {val_pyramid_metrics.get('height_mean', 0):.3f}\n")

            # Save checkpoint
            if global_step % args.save_interval == 0 and global_step > 0:
                checkpoint_path = exp_dir / f'checkpoint_step_{global_step}'
                model.save_pretrained(str(checkpoint_path))
                print(f"💾 Checkpoint saved: {checkpoint_path}")

            global_step += 1

    # Final save
    final_path = exp_dir / 'final_model'
    model.save_pretrained(str(final_path))
    print(f"\n✅ Training complete! Final model saved to {final_path}")

    # Final evaluation
    print("\nFinal evaluation...")
    val_loss, val_pyramid_metrics = evaluate(model, val_loader, device, max_batches=50)
    print(f"Final Validation Loss: {val_loss:.4f}")
    print("\nFinal Pyramidal Metrics:")
    for key, value in val_pyramid_metrics.items():
        print(f"  {key}: {value:.4f}")

    writer.close()


if __name__ == '__main__':
    main()
