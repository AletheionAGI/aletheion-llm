"""Training related tests."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import BaselineTransformer  # noqa: E402


def test_backward_pass_produces_finite_gradients():
    torch.manual_seed(0)
    model = BaselineTransformer(vocab_size=50, d_model=32, n_layers=2, n_heads=4, max_seq_len=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    input_ids = torch.randint(0, 50, (2, 16))
    outputs = model(input_ids, labels=input_ids.clone())

    outputs.loss.backward()

    grads_finite = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
    assert grads_finite and all(grads_finite)

    optimizer.step()
    optimizer.zero_grad()

    outputs_after = model(input_ids, labels=input_ids.clone())
    assert torch.isfinite(outputs_after.loss)
