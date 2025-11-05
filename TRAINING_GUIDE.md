# Baseline Training Guide - Memory Optimizations

This guide shows how to use the improved baseline training script with memory optimizations.

## What's New

The training script now includes 4 major improvements to handle memory constraints:

1. **Lower default batch size** (32 → 8)
2. **Gradient accumulation** - Simulate large batches with smaller micro-batches
3. **Gradient checkpointing** - Trade compute for ~40% memory savings
4. **Checkpoint resumption** - Resume training from any checkpoint

## Quick Start

### Basic Training (Memory-Efficient)
```bash
python experiments/level1/train_baseline.py \
    --output outputs/baseline \
    --num-epochs 10 \
    --batch-size 8 \
    --gradient-checkpointing
```

### Simulate Your Original Batch Size (32)
Use gradient accumulation to get effective batch_size=32 while using only batch_size=8:
```bash
python experiments/level1/train_baseline.py \
    --output outputs/baseline \
    --num-epochs 10 \
    --batch-size 8 \
    --gradient-accumulation-steps 4 \
    --gradient-checkpointing
```
This uses 4x less memory while maintaining the same effective batch size!

### Resume from Checkpoint
If training gets killed again, resume from where you left off:
```bash
python experiments/level1/train_baseline.py \
    --output outputs/baseline \
    --num-epochs 10 \
    --batch-size 8 \
    --gradient-accumulation-steps 4 \
    --gradient-checkpointing \
    --resume-from outputs/baseline/checkpoint-2000
```

## New Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch-size` | 8 | Micro batch size (was 4 before, but 8 is better with gradient accumulation) |
| `--gradient-accumulation-steps` | 1 | Number of gradient accumulation steps. Effective batch = batch_size × accumulation_steps |
| `--gradient-checkpointing` | False | Enable gradient checkpointing (~40% memory reduction) |
| `--resume-from` | None | Path to checkpoint directory to resume from |

## Memory Optimization Strategy

### If you have ~4-6 GB GPU memory:
```bash
--batch-size 4 --gradient-accumulation-steps 8 --gradient-checkpointing
```
Effective batch size: 32

### If you have ~8-12 GB GPU memory:
```bash
--batch-size 8 --gradient-accumulation-steps 4 --gradient-checkpointing
```
Effective batch size: 32

### If you have ~16+ GB GPU memory:
```bash
--batch-size 16 --gradient-accumulation-steps 2 --gradient-checkpointing
```
Effective batch size: 32

## How It Works

### Gradient Accumulation
Instead of:
- Forward pass with batch_size=32 → backward pass → update weights

We do:
- Forward pass with batch_size=8 → backward pass (accumulate gradients)
- Forward pass with batch_size=8 → backward pass (accumulate gradients)
- Forward pass with batch_size=8 → backward pass (accumulate gradients)
- Forward pass with batch_size=8 → backward pass (accumulate gradients)
- Update weights with accumulated gradients

This gives the **same training dynamics** as batch_size=32 but uses **4x less memory**!

### Gradient Checkpointing
Normally, PyTorch stores all intermediate activations during forward pass for use in backward pass. Gradient checkpointing **recomputes** some activations during backward pass instead of storing them, trading:
- **Memory saved**: ~40% reduction
- **Cost**: ~20-30% slower training

For memory-constrained situations, this is a great trade-off!

### Checkpoint Resumption
Every 2000 steps, the script saves a checkpoint. If training crashes:
1. Find the latest checkpoint: `ls -lh outputs/baseline/checkpoint-*`
2. Resume with `--resume-from outputs/baseline/checkpoint-XXXX`

The script will:
- Load the model weights
- Load the training history
- Continue from the exact step where it left off

## Recommended Command for Your Setup

Based on your crash at step 499 with batch_size=32, try:

```bash
python experiments/level1/train_baseline.py \
    --output outputs/baseline \
    --num-epochs 10 \
    --batch-size 8 \
    --gradient-accumulation-steps 4 \
    --gradient-checkpointing \
    --save-interval 500
```

This will:
- ✅ Use 4x less memory than before
- ✅ Maintain effective batch_size=32
- ✅ Save checkpoints every 500 steps (so you lose less progress if killed)
- ✅ Enable gradient checkpointing for extra memory savings

## Troubleshooting

**Still getting killed?**
- Reduce `--batch-size` to 4 or even 2
- Increase `--gradient-accumulation-steps` to maintain effective batch size
- Make sure gradient checkpointing is enabled

**Training slower?**
- Gradient accumulation adds no overhead
- Gradient checkpointing adds ~20% overhead (worth it for memory!)
- If too slow, reduce accumulation steps and increase batch size

**Want to start fresh?**
- Remove the `--resume-from` flag
- Or delete the output directory: `rm -rf outputs/baseline`
