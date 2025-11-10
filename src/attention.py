# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Felipe Maya Muniz

"""Multi-head attention layers for the baseline transformer."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention implementation.

    This module exposes a baseline self-attention operator without any epistemic
    gating. It optionally integrates FlashAttention when the package is
    available so that future experiments can toggle it directly from the
    configuration.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        use_flash: bool = False,
    ) -> None:
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.use_flash = use_flash

        # Combined query/key/value projection for efficiency.
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Flash attention is optional at runtime to keep the baseline portable.
        self.flash_attn = None
        if use_flash:
            try:  # pragma: no cover - import guarded by availability
                from flash_attn import flash_attn_func

                self.flash_attn = flash_attn_func
            except ImportError:  # pragma: no cover - informative warning only
                print("⚠️  Flash attention not available, using standard attention")

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run attention over the input sequence.

        Args:
            x: Input tensor of shape ``(batch, seq_len, d_model)``.
            mask: Optional attention mask where a value of ``0`` indicates
                positions that should be masked. Supported shapes are
                ``(batch, seq_len, seq_len)`` or ``(batch, 1, seq_len, seq_len)``.
            return_attention: Whether to return the attention weights along with
                the output.

        Returns:
            Either the output tensor of shape ``(batch, seq_len, d_model)`` or a
            tuple ``(output, attention_weights)`` when ``return_attention`` is
            ``True``.
        """

        batch_size, seq_len, _ = x.shape

        # (B, T, 3 * C)
        qkv = self.qkv_proj(x)
        # (B, T, 3, H, Dh)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.d_head)
        # Split into components: each is (B, T, H, Dh)
        q, k, v = qkv.unbind(dim=2)

        # Transpose to (B, H, T, Dh)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.flash_attn is not None and not return_attention:
            # FlashAttention expects (B, T, H, Dh)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            out = self.flash_attn(q, k, v, causal=True)
            out = out.reshape(batch_size, seq_len, self.d_model)
            attn_weights = None
        else:
            out, attn_weights = self._standard_attention(q, k, v, mask)

        out = self.out_proj(out)
        out = self.resid_dropout(out)

        if return_attention:
            return out, attn_weights
        return out

    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scaled dot-product attention used when FlashAttention is disabled."""

        # (B, H, T, T)
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = attn_weights @ v
        out = out.transpose(1, 2).contiguous().view(q.size(0), q.size(2), self.d_model)
        return out, attn_weights


class CausalSelfAttention(MultiHeadAttention):
    """Decoder-only causal self-attention with a persistent mask."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 2048,
        **kwargs: object,
    ) -> None:
        super().__init__(d_model=d_model, n_heads=n_heads, **kwargs)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool)).view(
                1, 1, max_seq_len, max_seq_len
            ),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.size(1)
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        return super().forward(x, mask=mask, **kwargs)
