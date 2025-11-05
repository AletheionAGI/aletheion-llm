# 🐛 Bug Fix: Brier Score Calculation Error

## Summary

Fixed critical bug in `src/aletheion/loss.py` where Brier Score was incorrectly computed, resulting in artificially low values (near 0).

## Problem

### Original Implementation (INCORRECT)
```python
# line 308 (old)
brier_score = torch.mean((probs - one_hot) ** 2)
```

**Issue**: This computed the mean over ALL dimensions (batch × vocab_size), diluting the value significantly. With vocab_size ≈ 50,000, the score was divided by 50k unnecessarily.

**Result**: Brier scores of ~0.0000 were reported, which is mathematically impossible for a non-perfect language model.

## Solution

### Fixed Implementation (CORRECT)
```python
# lines 306-311 (new)
# Compute Brier score
# Formula: Brier = (1/N) * Σ_i Σ_c (p_ic - y_ic)²
# where p_ic is predicted prob for class c, y_ic is one-hot target
one_hot = F.one_hot(targets, num_classes=probs.size(-1)).float()
# Sum over classes (dim=-1), then mean over batch (dim=0)
brier_score = torch.mean(torch.sum((probs - one_hot) ** 2, dim=-1))
```

**Correct Formula**:
1. Compute squared error for each (example, class) pair: `(p_ic - y_ic)²`
2. **Sum** over all classes (vocab_size) for each example
3. **Average** over batch size

## Impact

### Before Fix
- **Reported Brier Score**: 0.0000 (incorrect)
- **Reason**: Value diluted by vocab_size

### After Fix
- **Expected Brier Score**: 1.5-2.5 (typical for untrained/early LMs)
- **Expected Brier Score**: 0.3-0.8 (typical for well-trained LMs)
- **Interpretation**: Lower is better, but should never be exactly 0

## Verification Status

✅ **ECE (Expected Calibration Error)**: Already correct
- Uses proper binning (10 bins)
- Computes `|accuracy - confidence|` per bin
- Weighted by bin proportion

⚠️ **ECE value of 0.0084**: Suspiciously low for baseline GPT-2
- Typical ECE for LLMs: 0.05-0.15
- May indicate other calibration issue or very small eval set

❌ **Brier Score**: Fixed in this commit

## Action Required

🔄 **Re-run training** to get correct Brier scores:
```bash
python experiments/level1/train_baseline.py --steps 2000 --output-dir outputs/baseline
```

## Files Changed

- `src/aletheion/loss.py`: Fixed Brier score calculation in `compute_calibration_metrics()`

## References

- Brier Score formula: G. Brier (1950), "Verification of forecasts expressed in terms of probability"
- Proper scoring rule for multi-class: Sum over classes, mean over examples

---

**Commit**: Fix Brier score calculation bug in calibration metrics
**Date**: 2025-11-05
**Impact**: Critical - affects all calibration evaluations
