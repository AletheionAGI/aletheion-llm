# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Felipe Maya Muniz

"""Utility helpers for the baseline implementation."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Seed Python and PyTorch RNGs for reproducibility."""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - optional hardware
        torch.cuda.manual_seed_all(seed)


def get_device(preferred: str = "cuda") -> torch.device:
    """Return the preferred device when available."""

    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    return torch.device("cpu")


def cosine_decay_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup followed by cosine decay."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def linear_decay_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup followed by linear decay to zero."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def constant_schedule(
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Return a scheduler that keeps the learning rate constant."""

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)


def count_parameters(module: torch.nn.Module, trainable_only: bool = False) -> int:
    """Count parameters in ``module``."""

    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())
