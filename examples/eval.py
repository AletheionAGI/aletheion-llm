# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Felipe Maya Muniz

"""Evaluation script for the baseline transformer."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from data.dataset import collate_fn, load_wikitext_dataset
from src import BaselineTransformer, get_device
from torch.utils.data import DataLoader


def evaluate_checkpoint(checkpoint_path: Path) -> tuple[float, float]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]

    device = get_device(config["system"].get("device", "cuda"))

    _, val_dataset, test_dataset, tokenizer = load_wikitext_dataset(
        tokenizer_name=config["data"].get("tokenizer_name", "gpt2"),
        dataset_config=config["data"].get("dataset_config", "wikitext-2-raw-v1"),
        max_length=config["model"]["max_seq_len"],
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"].get("num_workers", 0),
        collate_fn=collate_fn,
    )

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

    mixed_precision = config["system"].get("mixed_precision", True) and device.type == "cuda"

    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            with torch.cuda.amp.autocast(enabled=mixed_precision):
                outputs = model(input_ids, labels=labels)
            total_loss += outputs.loss.item()
            total_batches += 1

    avg_loss = total_loss / max(1, total_batches)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    print("📊 Evaluation results")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")
    return avg_loss, perplexity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    evaluate_checkpoint(Path(args.checkpoint))
