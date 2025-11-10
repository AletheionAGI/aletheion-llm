# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Felipe Maya Muniz

"""Text generation utility for the baseline transformer."""

from __future__ import annotations

import argparse

import torch
from src import BaselineTransformer, get_device
from src.tokenizer import build_tokenizer


def generate_text(
    checkpoint_path: str,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int | None = 40,
    top_p: float | None = None,
) -> str:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]

    device = get_device(config["system"].get("device", "cuda"))

    tokenizer = build_tokenizer(config["data"].get("tokenizer_name", "gpt2"))
    model = BaselineTransformer(
        vocab_size=config["model"]["vocab_size"],
        d_model=config["model"]["d_model"],
        n_layers=config["model"]["n_layers"],
        n_heads=config["model"]["n_heads"],
        d_ff=config["model"]["d_ff"],
        max_seq_len=config["model"]["max_seq_len"],
        dropout=config["model"]["dropout"],
        tie_weights=config["model"].get("tie_weights", True),
        use_flash_attention=config["model"].get("use_flash_attention", False),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    generated_ids = model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=temperature > 0,
    )

    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using a trained baseline model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="The future of AI is")
    parser.add_argument("--max_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=None)
    args = parser.parse_args()

    output = generate_text(
        args.checkpoint,
        args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    print("🎯 Prompt:", args.prompt)
    print("🤖 Output:\n")
    print(output)
