# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Felipe Maya Muniz

"""Attention module tests."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.attention import CausalSelfAttention, MultiHeadAttention  # noqa: E402


def test_multi_head_attention_shapes():
    attn = MultiHeadAttention(d_model=32, n_heads=4, dropout=0.0)
    x = torch.randn(2, 10, 32)

    out, weights = attn(x, return_attention=True)
    assert out.shape == (2, 10, 32)
    assert weights.shape == (2, 4, 10, 10)


def test_causal_masking():
    attn = CausalSelfAttention(d_model=16, n_heads=4, max_seq_len=32, dropout=0.0)
    x = torch.randn(1, 5, 16)

    _, weights = attn(x, return_attention=True)
    upper_triangle = torch.triu(weights[0, 0], diagonal=1)
    assert torch.allclose(upper_triangle, torch.zeros_like(upper_triangle), atol=1e-5)


def test_flash_flag_graceful_fallback():
    attn = MultiHeadAttention(d_model=16, n_heads=4, use_flash=True)
    x = torch.randn(1, 4, 16)

    out = attn(x)
    assert out.shape == (1, 4, 16)
