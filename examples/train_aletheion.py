"""Training script for Aletheion Level 1 model with VARO loss.

This extends the baseline train.py to support epistemic uncertainty training.
Key differences:
    - Uses AletheionTransformer instead of BaselineTransformer
    - Applies VARO loss (L_CE + λ * ||u - u*||²)
    - Logs uncertainty metrics (q1, q2, uncertainty)
    - Supports epistemic hyperparameters from config
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src import get_device, load_config, set_seed
from src.utils import (
    constant_schedule,
    cosine_decay_with_warmup,
    linear_decay_with_warmup,
)
from src.aletheion.model import AletheionTransformer
from src.aletheion.loss import VaroLoss
from data.dataset import collate_fn, load_wikitext_dataset

try:  # pragma: no cover - optional dependency
    import wandb
except Exception:  # pragma: no cover - fallback when wandb missing
    wandb = None


def build_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """Instantiate dataloaders for training and validation."""

    data_cfg = config["data"]
    model_cfg = config["model"]

    if data_cfg.get("dataset", "wikitext") == "wikitext":
        train_ds, val_ds, _, _ = load_wikitext_dataset(
            tokenizer_name=data_cfg.get("tokenizer_name", "gpt2"),
            dataset_config=data_cfg.get("dataset_config", "wikitext-2-raw-v1"),
            max_length=model_cfg["max_seq_len"],
        )
    else:
        raise NotImplementedError("Only the WikiText dataset is implemented")

    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 0),
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 0),
        collate_fn=collate_fn,
    )

    return train_loader, val_loader


def create_model(config: Dict, device: torch.device) -> AletheionTransformer:
    """Create Aletheion model with epistemic gates."""
    model_cfg = config["model"]
    epistemic_cfg = model_cfg.get("epistemic", {})

    model = AletheionTransformer(
        vocab_size=model_cfg["vocab_size"],
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        d_ff=model_cfg["d_ff"],
        max_seq_len=model_cfg["max_seq_len"],
        dropout=model_cfg["dropout"],
        tie_weights=model_cfg.get("tie_weights", True),
        use_flash_attention=model_cfg.get("use_flash_attention", False),
        # Epistemic parameters
        q1_threshold=epistemic_cfg.get("q1_threshold", 0.7),
        q2_threshold=epistemic_cfg.get("q2_threshold", 0.7),
        base_temperature=epistemic_cfg.get("base_temperature", 1.0),
        n_consensus_heads=epistemic_cfg.get("n_consensus_heads", 4)
    ).to(device)

    if config["system"].get("compile", False) and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[attr-defined]
    return model


def create_optimizer(config: Dict, model: AletheionTransformer) -> torch.optim.Optimizer:
    opt_cfg = config["optimizer"]
    training_cfg = config["training"]

    if opt_cfg.get("type", "adamw").lower() != "adamw":
        raise NotImplementedError("Only AdamW optimizer is supported")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg["learning_rate"],
        betas=tuple(opt_cfg.get("betas", (0.9, 0.95))),
        eps=opt_cfg.get("eps", 1e-8),
        weight_decay=training_cfg.get("weight_decay", 0.0),
    )
    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    training_cfg = config["training"]
    schedule = training_cfg.get("lr_schedule", "cosine").lower()
    warmup_steps = training_cfg.get("warmup_steps", 0)

    if schedule == "cosine":
        return cosine_decay_with_warmup(optimizer, warmup_steps, total_steps)
    if schedule == "linear":
        return linear_decay_with_warmup(optimizer, warmup_steps, total_steps)
    if schedule == "constant":
        return constant_schedule(optimizer)
    raise ValueError(f"Unknown lr_schedule: {schedule}")


def evaluate(
    model: AletheionTransformer,
    loader: DataLoader,
    device: torch.device,
    mixed_precision: bool,
    varo_loss: VaroLoss
) -> Dict[str, float]:
    """Evaluate model on validation set with uncertainty metrics."""
    model.eval()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_uncertainty_loss = 0.0
    total_uncertainty = 0.0
    total_q1 = 0.0
    total_q2 = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=mixed_precision):
                outputs = model(input_ids, labels=labels, return_uncertainty=True)

                # Compute VARO loss
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                shift_uncertainty = outputs.uncertainty[..., :-1, :].contiguous()

                loss_dict = varo_loss(
                    logits=shift_logits,
                    targets=shift_labels,
                    uncertainty=shift_uncertainty
                )

            total_loss += loss_dict['loss'].item()
            total_ce_loss += loss_dict['ce_loss'].item()
            total_uncertainty_loss += loss_dict['uncertainty_loss'].item()
            total_uncertainty += loss_dict['u_pred_mean']
            total_q1 += outputs.q1[..., :-1, :].mean().item()
            total_q2 += outputs.q2[..., :-1, :].mean().item()
            total_batches += 1

    model.train()

    return {
        "val_loss": total_loss / max(1, total_batches),
        "val_ce_loss": total_ce_loss / max(1, total_batches),
        "val_uncertainty_loss": total_uncertainty_loss / max(1, total_batches),
        "val_perplexity": torch.exp(torch.tensor(total_ce_loss / max(1, total_batches))).item(),
        "val_uncertainty_mean": total_uncertainty / max(1, total_batches),
        "val_q1_mean": total_q1 / max(1, total_batches),
        "val_q2_mean": total_q2 / max(1, total_batches),
    }


def main(config_path: str) -> None:
    config = load_config(config_path)

    set_seed(config["system"].get("seed", 42))
    device = get_device(config["system"].get("device", "cuda"))

    train_loader, val_loader = build_dataloaders(config)
    model = create_model(config, device)
    optimizer = create_optimizer(config, model)

    # Create VARO loss
    epistemic_cfg = config["model"].get("epistemic", {})
    varo_loss = VaroLoss(
        lambda_varo=epistemic_cfg.get("lambda_varo", 0.1),
        u_star_method=epistemic_cfg.get("u_star_method", "head_variance"),
        ignore_index=-100
    )

    max_steps = config["training"]["max_steps"]
    scheduler = create_scheduler(optimizer, config, max_steps)
    accumulation_steps = config["training"].get("gradient_accumulation_steps", 1)
    grad_clip = config["training"].get("grad_clip_norm", 1.0)
    mixed_precision = config["system"].get("mixed_precision", True) and device.type == "cuda"

    scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)

    writer = SummaryWriter(log_dir=config["logging"].get("log_dir", "./logs"))

    use_wandb = config["logging"].get("use_wandb", False) and wandb is not None
    if use_wandb:
        wandb.init(
            project=config["logging"].get("wandb_project", "aletheion-level1"),
            entity=config["logging"].get("wandb_entity"),
            config=config,
            name=config["logging"].get("run_name"),
        )

    global_step = 0
    best_val_loss = float("inf")
    best_checkpoint = Path(config["logging"].get("save_dir", "./checkpoints")) / "best_aletheion_model.pt"
    best_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    progress = tqdm(total=max_steps, desc="Training Aletheion Level 1")
    micro_step = 0

    while global_step < max_steps:
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=mixed_precision):
                outputs = model(input_ids, labels=labels, return_uncertainty=True)

                # Compute VARO loss
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                shift_uncertainty = outputs.uncertainty[..., :-1, :].contiguous()

                loss_dict = varo_loss(
                    logits=shift_logits,
                    targets=shift_labels,
                    uncertainty=shift_uncertainty
                )

                loss = loss_dict['loss'] / accumulation_steps

            scaler.scale(loss).backward()

            micro_step += 1

            if micro_step % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                global_step += 1
                progress.update(1)

                if global_step % config["training"].get("log_interval", 100) == 0:
                    # Log training metrics
                    writer.add_scalar("train/loss", loss_dict['loss'].item(), global_step)
                    writer.add_scalar("train/ce_loss", loss_dict['ce_loss'].item(), global_step)
                    writer.add_scalar("train/uncertainty_loss", loss_dict['uncertainty_loss'].item(), global_step)
                    writer.add_scalar("train/uncertainty_mean", loss_dict['u_pred_mean'], global_step)
                    writer.add_scalar("train/q1_mean", outputs.q1.mean().item(), global_step)
                    writer.add_scalar("train/q2_mean", outputs.q2.mean().item(), global_step)
                    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

                    if use_wandb:
                        wandb.log({
                            "train/loss": loss_dict['loss'].item(),
                            "train/ce_loss": loss_dict['ce_loss'].item(),
                            "train/uncertainty_loss": loss_dict['uncertainty_loss'].item(),
                            "train/uncertainty_mean": loss_dict['u_pred_mean'],
                            "train/q1_mean": outputs.q1.mean().item(),
                            "train/q2_mean": outputs.q2.mean().item(),
                            "train/lr": optimizer.param_groups[0]["lr"]
                        }, step=global_step)

                if global_step % config["training"].get("eval_interval", 1000) == 0:
                    val_metrics = evaluate(model, val_loader, device, mixed_precision, varo_loss)

                    # Log validation metrics
                    for key, value in val_metrics.items():
                        writer.add_scalar(f"{key}", value, global_step)

                    if use_wandb:
                        wandb.log(val_metrics, step=global_step)

                    print(f"\n[Step {global_step}] Val Loss: {val_metrics['val_loss']:.4f} | "
                          f"Perplexity: {val_metrics['val_perplexity']:.2f} | "
                          f"Uncertainty: {val_metrics['val_uncertainty_mean']:.3f}")

                    if val_metrics['val_loss'] < best_val_loss:
                        best_val_loss = val_metrics['val_loss']
                        torch.save(
                            {
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "config": config,
                                "global_step": global_step,
                                **val_metrics
                            },
                            best_checkpoint,
                        )

                if global_step % config["training"].get("save_interval", 5000) == 0:
                    checkpoint_path = (
                        Path(config["logging"].get("save_dir", "./checkpoints")) / f"aletheion_step_{global_step}.pt"
                    )
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "config": config,
                            "global_step": global_step,
                        },
                        checkpoint_path,
                    )

                if global_step >= max_steps:
                    break
        else:
            continue
        break

    progress.close()
    writer.close()
    if use_wandb:
        wandb.finish()

    print(f"🏁 Training finished. Best validation loss: {best_val_loss:.4f}")
    print(f"Best checkpoint saved to: {best_checkpoint}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Aletheion Level 1 transformer")
    parser.add_argument("--config", type=str, default="config/aletheion_level1.yaml")
    args = parser.parse_args()
    main(args.config)
