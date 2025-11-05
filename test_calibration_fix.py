"""Test script to verify calibration metric fixes."""
import torch
import torch.nn.functional as F
from src.aletheion.loss import compute_calibration_metrics

# Create simple test case
print("🧪 Testing Calibration Metrics Fix\n")

# Example: 3 samples, 4 classes
batch_size = 3
num_classes = 4

# Create predictions (logits -> probs)
logits = torch.tensor([
    [2.0, 1.0, 0.5, 0.1],  # Confident in class 0
    [0.5, 2.5, 0.3, 0.2],  # Confident in class 1
    [0.1, 0.2, 0.3, 1.5],  # Confident in class 3
])
probs = F.softmax(logits, dim=-1)

# True labels
targets = torch.tensor([0, 1, 3])  # All correct predictions

# Dummy uncertainty (not used for ECE/Brier)
uncertainty = torch.zeros(batch_size, 1)

print(f"Probs shape: {probs.shape}")
print(f"Probs:\n{probs}\n")
print(f"Targets: {targets}\n")

# Compute metrics
metrics = compute_calibration_metrics(probs, targets, uncertainty, n_bins=10)

print("📊 Results:")
print(f"   ECE: {metrics['ece']:.4f}")
print(f"   Brier Score: {metrics['brier_score']:.4f}")
print(f"   Uncertainty-Error Correlation: {metrics['uncertainty_error_corr']:.4f}")

# Manual Brier Score calculation for verification
one_hot = F.one_hot(targets, num_classes=num_classes).float()
manual_brier = torch.mean(torch.sum((probs - one_hot) ** 2, dim=-1))
print(f"\n✅ Manual Brier calculation: {manual_brier.item():.4f}")

# Expected: Since all predictions are correct and confident,
# Brier score should be low (good) but not zero
# ECE should be low since confidence matches accuracy

print("\n📝 Expected behavior:")
print("   - Brier Score: Should be > 0 (not exactly 0)")
print("   - ECE: Should be low (predictions are correct and confident)")
print("   - All metrics should be reasonable, not suspiciously close to 0")
