# Aletheion Examples

This directory contains example scripts demonstrating how to use Aletheion for training, evaluation, and inference.

## Training Scripts

### Baseline Training
- **[train.py](train.py)** - Train a baseline transformer model without epistemic uncertainty

```bash
python examples/train.py --config config/default.yaml --output outputs/baseline/
```

### Aletheion Training
- **[train_aletheion.py](train_aletheion.py)** - Train an Aletheion model with epistemic uncertainty gates

```bash
python examples/train_aletheion.py --config config/aletheion_level1.yaml --output outputs/aletheion/
```

## Evaluation Scripts

- **[eval.py](eval.py)** - Comprehensive evaluation of trained models
- **[quick_eval.py](quick_eval.py)** - Quick evaluation for development and testing

```bash
# Evaluate a trained model
python examples/eval.py --checkpoint outputs/aletheion/checkpoint_final.pt

# Quick evaluation
python examples/quick_eval.py --checkpoint outputs/aletheion/checkpoint_latest.pt
```

## Inference Scripts

- **[generate.py](generate.py)** - Generate text using trained models

```bash
python examples/generate.py --checkpoint outputs/aletheion/checkpoint_final.pt --prompt "Your prompt here"
```

## Testing Scripts

- **[test_calibration_fix.py](test_calibration_fix.py)** - Test calibration improvements

## Advanced Examples

For more advanced experiments including:
- Pyramidal training
- Baseline vs Aletheion comparisons
- TruthfulQA benchmarking
- Out-of-domain evaluation
- Epistemic visualization

See the [`experiments/level1/`](../experiments/level1/) directory.

## Quick Start

1. Install dependencies:
```bash
pip install -e .
```

2. Train a baseline model:
```bash
python examples/train.py --config config/small.yaml --output outputs/my_baseline/
```

3. Train an Aletheion model:
```bash
python examples/train_aletheion.py --config config/aletheion_level1.yaml --output outputs/my_aletheion/
```

4. Evaluate your model:
```bash
python examples/eval.py --checkpoint outputs/my_aletheion/checkpoint_final.pt
```

## Configuration

All training scripts support YAML configuration files. See the [`config/`](../config/) directory for examples:
- `config/small.yaml` - Small model for quick experiments
- `config/default.yaml` - Default configuration
- `config/medium.yaml` - Medium-sized model
- `config/aletheion_level1.yaml` - Aletheion Level 1 configuration

## Command Line Arguments

Most scripts support common arguments:
- `--config` - Path to YAML configuration file
- `--output` - Output directory for checkpoints and logs
- `--checkpoint` - Path to checkpoint for evaluation/inference
- `--num-epochs` - Number of training epochs
- `--batch-size` - Batch size for training
- `--learning-rate` - Learning rate

Run any script with `--help` to see all available options.
