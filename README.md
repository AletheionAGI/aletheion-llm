# Baseline Transformer

Baseline transformer implementation used as a control experiment for the Aletheion LLM project. The code in this
folder intentionally mirrors a standard GPT-style architecture **without** epistemic gating so that we can measure the
impact of the fractal gating mechanism in isolation.

## Features

- Clean PyTorch implementation of a decoder-only transformer
- Modular attention, feed-forward, and block definitions
- Config-driven training and evaluation flows
- Hugging Face dataset/tokenizer integration for WikiText out of the box
- Support for mixed precision, gradient accumulation, and cosine decay with warmup
- Optional Weights & Biases and TensorBoard logging
- Comprehensive unit tests for core components

## Project Layout

```
implementation/baseline/
├── README.md
├── requirements.txt
├── config/
│   ├── default.yaml
│   ├── medium.yaml
│   └── small.yaml
├── src/
│   ├── __init__.py
│   ├── attention.py
│   ├── model.py
│   ├── tokenizer.py
│   └── utils.py
├── data/
│   ├── __init__.py
│   ├── dataset.py
│   └── prepare.py
├── train.py
├── eval.py
├── generate.py
├── tests/
│   ├── test_attention.py
│   ├── test_model.py
│   └── test_training.py
└── scripts/
    ├── evaluate.sh
    ├── train_full.sh
    └── train_small.sh
```

## Quick Start

```bash
cd implementation/baseline
pip install -r requirements.txt
```

### Training

```bash
# Debug-friendly run
python train.py --config config/small.yaml

# Default configuration
python train.py --config config/default.yaml
```

### Evaluation

```bash
python eval.py --checkpoint checkpoints/best_model.pt
```

### Text Generation

```bash
python generate.py \
  --checkpoint checkpoints/best_model.pt \
  --prompt "The future of AI is" \
  --max_tokens 100
```

## Expected Baselines

| Model | Parameters | Dataset | Expected PPL |
|-------|------------|---------|--------------|
| Small | ~10M       | WikiText-2 | 35 - 40 |
| Medium | ~50M      | WikiText-103 | 25 - 30 |

The numbers above assume full training on the specified dataset splits with cosine decay and warmup. They provide a
sanity check that the baseline is operating as expected before introducing epistemic gates.

## Next Steps

1. Train the baseline to convergence on the desired dataset.
2. Record validation perplexity, training curves, and qualitative samples.
3. Introduce the epistemic gating mechanism on top of this foundation.

## License

This code inherits the license of the main repository.
