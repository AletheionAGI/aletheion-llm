"""Training entrypoint for the baseline transformer."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from data.dataset import collate_fn, load_wikitext_dataset
from src import BaselineTransformer, get_device, load_config, set_seed
from src.utils import (
    constant_schedule,
    cosine_decay_with_warmup,
    linear_decay_with_warmup,
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:  # pragma: no cover - optional dependency
    import wandb
except Exception:  # pragma: no cover - fallback when wandb missing
    wandb = None


def build_dataloaders(config: dict, device: torch.device) -> tuple[DataLoader, DataLoader]:
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
        raise NotImplementedError("Only the WikiText dataset is implemented in the baseline")

    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 0),
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 0),
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    return train_loader, val_loader


def create_model(config: dict, device: torch.device) -> BaselineTransformer:
    model_cfg = config["model"]
    model = BaselineTransformer(
        vocab_size=model_cfg["vocab_size"],
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        d_ff=model_cfg["d_ff"],
        max_seq_len=model_cfg["max_seq_len"],
        dropout=model_cfg["dropout"],
        tie_weights=model_cfg.get("tie_weights", True),
        use_flash_attention=model_cfg.get("use_flash_attention", False),
    ).to(device)

    if config["system"].get("compile", False) and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[attr-defined]
    return model


def create_optimizer(config: dict, model: BaselineTransformer) -> torch.optim.Optimizer:
    opt_cfg = config["optimizer"]
    training_cfg = config["training"]

    if opt_cfg.get("type", "adamw").lower() != "adamw":
        raise NotImplementedError("Only AdamW optimizer is supported in the baseline")

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
    config: dict,
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
    model: BaselineTransformer, loader: DataLoader, device: torch.device, mixed_precision: bool
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            with torch.cuda.amp.autocast(enabled=mixed_precision):
                outputs = model(input_ids, labels=labels)
                loss = outputs.loss
            total_loss += loss.item()
            total_batches += 1

    avg_loss = total_loss / max(1, total_batches)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    model.train()
    return avg_loss, perplexity


def main(config_path: str) -> None:
    config = load_config(config_path)

    set_seed(config["system"].get("seed", 42))
    device = get_device(config["system"].get("device", "cuda"))

    train_loader, val_loader = build_dataloaders(config, device)
    model = create_model(config, device)
    optimizer = create_optimizer(config, model)

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
            project=config["logging"].get("wandb_project", "aletheion-baseline"),
            entity=config["logging"].get("wandb_entity"),
            config=config,
            name=config["logging"].get("run_name"),
        )

    global_step = 0
    best_val_loss = float("inf")
    best_checkpoint = Path(config["logging"].get("save_dir", "./checkpoints")) / "best_model.pt"
    best_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    progress = tqdm(total=max_steps, desc="Training steps")
    micro_step = 0

    while global_step < max_steps:
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=mixed_precision):
                outputs = model(input_ids, labels=labels)
                loss = outputs.loss / accumulation_steps

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
                    loss_value = outputs.loss.item()
                    writer.add_scalar("train/loss", loss_value, global_step)
                    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                    if use_wandb:
                        wandb.log(
                            {"train/loss": loss_value, "train/lr": optimizer.param_groups[0]["lr"]},
                            step=global_step,
                        )

                if global_step % config["training"].get("eval_interval", 1000) == 0:
                    val_loss, val_ppl = evaluate(model, val_loader, device, mixed_precision)
                    writer.add_scalar("val/loss", val_loss, global_step)
                    writer.add_scalar("val/perplexity", val_ppl, global_step)
                    if use_wandb:
                        wandb.log(
                            {"val/loss": val_loss, "val/perplexity": val_ppl}, step=global_step
                        )

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(
                            {
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "config": config,
                                "global_step": global_step,
                                "val_loss": val_loss,
                                "val_perplexity": val_ppl,
                            },
                            best_checkpoint,
                        )

                if global_step % config["training"].get("save_interval", 5000) == 0:
                    checkpoint_path = (
                        Path(config["logging"].get("save_dir", "./checkpoints"))
                        / f"step_{global_step}.pt"
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
    parser = argparse.ArgumentParser(description="Train the baseline transformer")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    args = parser.parse_args()
    main(args.config)
