"""Tokenizer utilities."""

from __future__ import annotations

from typing import Optional

from transformers import GPT2TokenizerFast


def build_tokenizer(name: str = "gpt2", cache_dir: Optional[str] = None) -> GPT2TokenizerFast:
    """Load and configure a GPT-2 tokenizer."""

    tokenizer = GPT2TokenizerFast.from_pretrained(name, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
