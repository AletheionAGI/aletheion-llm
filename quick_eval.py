"""Quick evaluation script for trained checkpoints - FIXED VERSION."""
import sys
from pathlib import Path
sys.path.insert(0, '/home/sapo/aletheion-llm')

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import numpy as np

from src import BaselineTransformer, get_device
from src.aletheion.model import AletheionTransformer
from data.dataset import load_wikitext_dataset

def collate_fn(batch):
    """Collate function for variable length sequences."""
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    
    return {
        'input_ids': input_ids_padded,
        'labels': labels_padded
    }

def compute_ece(probs, targets, n_bins=10):
    """Compute Expected Calibration Error."""
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(targets)
    
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece.item()

def evaluate(model, loader, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            try:
                outputs = model(input_ids, labels=labels, return_uncertainty=True)
            except TypeError:
                outputs = model(input_ids, labels=labels)
            
            total_loss += outputs.loss.item()
            
            # Get predictions for all tokens
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)

            # Flatten batch and sequence dimensions to align with targets
            probs_flat = probs.view(-1, probs.size(-1))
            targets_flat = labels.view(-1)

            # Filter out padding tokens (-100) from targets
            valid_mask = targets_flat != -100
            if valid_mask.any():
                all_probs.append(probs_flat[valid_mask].cpu())
                all_targets.append(targets_flat[valid_mask].cpu())
    
    # Compute metrics
    avg_loss = total_loss / len(loader)
    perplexity = np.exp(avg_loss)
    
    all_probs = torch.cat(all_probs)
    all_targets = torch.cat(all_targets)
    ece = compute_ece(all_probs, all_targets)
    
    # Compute accuracy
    _, predictions = all_probs.max(dim=1)
    accuracy = predictions.eq(all_targets).float().mean().item()
    
    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "ece": ece,
        "accuracy": accuracy
    }

def main():
    print("🔍 Loading dataset...")
    device = get_device()
    
    train_ds, val_ds, test_ds, tokenizer = load_wikitext_dataset(
        tokenizer_name="gpt2",
        max_length=256
    )
    
    # FIX: Add collate_fn for variable length sequences
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
    vocab_size = tokenizer.vocab_size
    
    print(f"📊 Vocab size: {vocab_size}")
    print(f"📊 Val samples: {len(val_ds)}")
    
    # Load models
    print("\n🔍 Loading baseline checkpoint...")
    baseline = BaselineTransformer(
        vocab_size=vocab_size,
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        max_seq_len=256,
        dropout=0.1
    ).to(device)
    
    # FIX: Load from nested dict structure
    try:
        baseline_ckpt = torch.load('checkpoints/baseline_final.pt', map_location=device)
        baseline.load_state_dict(baseline_ckpt['model_state_dict'])
        print("✅ Loaded baseline checkpoint")
    except Exception as e:
        print(f"⚠️  No baseline checkpoint found: {e}")
        print("Evaluating untrained model")
    
    print("\n🔍 Loading Aletheion checkpoint...")
    aletheion = AletheionTransformer(
        vocab_size=vocab_size,
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        max_seq_len=256,
        dropout=0.1,
        q1_threshold=0.7,
        q2_threshold=0.7
    ).to(device)
    
    # FIX: Load from nested dict structure
    try:
        aletheion_ckpt = torch.load('checkpoints/aletheion_final.pt', map_location=device)
        aletheion.load_state_dict(aletheion_ckpt['model_state_dict'])
        print("✅ Loaded Aletheion checkpoint")
    except Exception as e:
        print(f"⚠️  No Aletheion checkpoint found: {e}")
        print("Evaluating untrained model")
    
    # Evaluate
    print("\n" + "="*80)
    print("BASELINE EVALUATION")
    print("="*80)
    baseline_metrics = evaluate(baseline, val_loader, device)
    
    print("\n" + "="*80)
    print("ALETHEION EVALUATION")
    print("="*80)
    aletheion_metrics = evaluate(aletheion, val_loader, device)
    
    # Print results
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    print("\n📊 BASELINE:")
    print(f"   Loss: {baseline_metrics['loss']:.4f}")
    print(f"   Perplexity: {baseline_metrics['perplexity']:.2f}")
    print(f"   ECE: {baseline_metrics['ece']:.4f}")
    print(f"   Accuracy: {baseline_metrics['accuracy']:.4f}")
    
    print("\n📊 ALETHEION:")
    print(f"   Loss: {aletheion_metrics['loss']:.4f}")
    print(f"   Perplexity: {aletheion_metrics['perplexity']:.2f}")
    print(f"   ECE: {aletheion_metrics['ece']:.4f}")
    print(f"   Accuracy: {aletheion_metrics['accuracy']:.4f}")
    
    print("\n📊 IMPROVEMENTS:")
    loss_improvement = (baseline_metrics['loss'] - aletheion_metrics['loss']) / baseline_metrics['loss'] * 100
    ece_improvement = (baseline_metrics['ece'] - aletheion_metrics['ece']) / baseline_metrics['ece'] * 100
    
    print(f"   Loss: {loss_improvement:+.2f}%")
    print(f"   ECE: {ece_improvement:+.2f}%")
    print(f"   Perplexity: {(baseline_metrics['perplexity'] - aletheion_metrics['perplexity']) / baseline_metrics['perplexity'] * 100:+.2f}%")

if __name__ == "__main__":
    main()