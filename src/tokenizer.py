# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Felipe Maya Muniz

"""Tokenizer utilities."""

from __future__ import annotations

from transformers import GPT2TokenizerFast


def build_tokenizer(name: str = "gpt2", cache_dir: str | None = None) -> GPT2TokenizerFast:
    """Load and configure a GPT-2 tokenizer."""

    tokenizer = GPT2TokenizerFast.from_pretrained(name, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
