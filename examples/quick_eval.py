"""MEMORY-EFFICIENT evaluation - computes ECE incrementally."""

import sys

sys.path.insert(0, "/home/sapo/aletheion-llm")

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import load_wikitext_dataset
from src import BaselineTransformer, get_device
from src.aletheion.model import AletheionTransformer


def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    return {"input_ids": input_ids_padded, "labels": labels_padded}


class IncrementalECE:
    """Compute ECE without storing all predictions."""

    def __init__(self, n_bins=10):
        self.n_bins = n_bins
        self.bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_counts = torch.zeros(n_bins)
        self.bin_correct = torch.zeros(n_bins)
        self.bin_conf_sum = torch.zeros(n_bins)

    def update(self, probs, targets):
        """Update bins with new batch."""
        confidences, predictions = probs.max(dim=1)
        accuracies = predictions.eq(targets).float()

        for i in range(self.n_bins):
            in_bin = (confidences > self.bin_boundaries[i]) & (
                confidences <= self.bin_boundaries[i + 1]
            )
            self.bin_counts[i] += in_bin.sum().item()
            self.bin_correct[i] += accuracies[in_bin].sum().item()
            self.bin_conf_sum[i] += confidences[in_bin].sum().item()

    def compute(self):
        """Compute final ECE."""
        ece = 0.0
        total = self.bin_counts.sum()

        for i in range(self.n_bins):
            if self.bin_counts[i] > 0:
                acc = self.bin_correct[i] / self.bin_counts[i]
                conf = self.bin_conf_sum[i] / self.bin_counts[i]
                prop = self.bin_counts[i] / total
                ece += abs(conf - acc) * prop

        return ece.item()


def evaluate(model, loader, device):
    """Memory-efficient evaluation."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    ece_calculator = IncrementalECE(n_bins=10)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            try:
                outputs = model(input_ids, labels=labels, return_uncertainty=True)
            except Exception:
                outputs = model(input_ids, labels=labels)

            total_loss += outputs.loss.item()

            # Process batch immediately, don't accumulate
            probs_flat = F.softmax(outputs.logits, dim=-1).view(-1, outputs.logits.size(-1))
            targets_flat = labels.view(-1)
            valid = targets_flat != -100

            if valid.any():
                probs_valid = probs_flat[valid].cpu()
                targets_valid = targets_flat[valid].cpu()

                # Update ECE incrementally
                ece_calculator.update(probs_valid, targets_valid)

                # Update accuracy
                predictions = probs_valid.max(dim=1)[1]
                total_correct += predictions.eq(targets_valid).sum().item()
                total_samples += targets_valid.size(0)

                # Free memory immediately
                del probs_valid, targets_valid, probs_flat, targets_flat

            torch.cuda.empty_cache()

    return {
        "loss": total_loss / len(loader),
        "ece": ece_calculator.compute(),
        "accuracy": total_correct / total_samples if total_samples > 0 else 0.0,
    }


print("🔍 Loading...")
device = get_device()

# seq_len=128, batch=1
train_ds, val_ds, _, tokenizer = load_wikitext_dataset(tokenizer_name="gpt2", max_length=128)

val_loader = DataLoader(val_ds, batch_size=1, collate_fn=collate_fn)
print(f"📊 Evaluating {len(val_ds)} samples (FULL DATASET)")

# Load baseline
baseline = BaselineTransformer(
    vocab_size=tokenizer.vocab_size,
    d_model=512,
    n_heads=8,
    n_layers=6,
    d_ff=2048,
    max_seq_len=512,
    dropout=0.1,
).to(device)

ckpt = torch.load("checkpoints/baseline_final.pt", map_location=device)
baseline.load_state_dict(ckpt["model_state_dict"])
print("✅ Baseline loaded")

# Load aletheion
aletheion = AletheionTransformer(
    vocab_size=tokenizer.vocab_size,
    d_model=512,
    n_heads=8,
    n_layers=6,
    d_ff=2048,
    max_seq_len=512,
    dropout=0.1,
    q1_threshold=0.7,
    q2_threshold=0.7,
).to(device)

ckpt = torch.load("checkpoints/aletheion_final.pt", map_location=device)
aletheion.load_state_dict(ckpt["model_state_dict"])
print("✅ Aletheion loaded")

# Evaluate
print("\n" + "=" * 60)
print("BASELINE")
print("=" * 60)
baseline_metrics = evaluate(baseline, val_loader, device)
print(f"   ECE: {baseline_metrics['ece']:.4f}")
print(f"   Acc: {baseline_metrics['accuracy']:.4f}")
print(f"   Loss: {baseline_metrics['loss']:.4f}")

print("\n" + "=" * 60)
print("ALETHEION")
print("=" * 60)
aletheion_metrics = evaluate(aletheion, val_loader, device)
print(f"   ECE: {aletheion_metrics['ece']:.4f}")
print(f"   Acc: {aletheion_metrics['accuracy']:.4f}")
print(f"   Loss: {aletheion_metrics['loss']:.4f}")

print("\n" + "=" * 60)
ece_change = (baseline_metrics["ece"] - aletheion_metrics["ece"]) / baseline_metrics["ece"] * 100
loss_change = (
    (baseline_metrics["loss"] - aletheion_metrics["loss"]) / baseline_metrics["loss"] * 100
)

print(f"\n🎯 ECE CHANGE: {ece_change:+.1f}%")
print(f"🎯 LOSS CHANGE: {loss_change:+.1f}%")

if ece_change > 0:
    print("\n✅ ALETHEION BETTER CALIBRATED!")
else:
    print("\n❌ ALETHEION WORSE CALIBRATED")
