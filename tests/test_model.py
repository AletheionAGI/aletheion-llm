"""Tests for the baseline transformer model."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import BaselineTransformer  # noqa: E402


def test_model_forward_shapes():
    model = BaselineTransformer(vocab_size=100, d_model=64, n_layers=2, n_heads=4, max_seq_len=32)
    input_ids = torch.randint(0, 100, (2, 16))

    outputs = model(input_ids)
    assert outputs.logits.shape == (2, 16, 100)
    assert outputs.loss is None


def test_model_loss_computation():
    model = BaselineTransformer(vocab_size=50, d_model=64, n_layers=2, n_heads=4, max_seq_len=32)
    input_ids = torch.randint(0, 50, (2, 16))

    outputs = model(input_ids, labels=input_ids.clone())
    assert outputs.loss is not None
    assert outputs.loss.item() > 0


def test_generation_length():
    model = BaselineTransformer(vocab_size=100, d_model=64, n_layers=2, n_heads=4, max_seq_len=64)
    model.eval()

    input_ids = torch.randint(0, 100, (1, 8))
    generated = model.generate(input_ids, max_new_tokens=12, do_sample=False)

    assert generated.shape == (1, 20)
