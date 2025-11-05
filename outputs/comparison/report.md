# Model Comparison Report

## Summary Statistics

### Baseline Model
- **Perplexity**: 349.89
- **Loss**: 5.8576
- **ECE**: 0.0092
- **Brier Score**: 0.9095
- **Accuracy**: 0.1986

### Pyramidal Model
- **Perplexity**: 403.90
- **Loss**: 6.0012
- **ECE**: 0.0145
- **Brier Score**: 0.9143
- **Accuracy**: 0.1894
- **Mean Height**: 0.2135
- **Mean Uncertainty**: 0.7865

## Improvements

### Perplexity
- **Change**: +54.01
- **Relative**: +15.44%

### Calibration (ECE)
- **Change**: +0.0053
- **Relative**: +57.67%

### Brier Score
- **Change**: +0.0048
- **Relative**: +0.53%

## Statistical Significance

### ECE Improvement
- **Mean Improvement**: -0.0045
- **95% CI**: [-0.0099, 0.0016]
- **P-value**: 0.9350
- **Significant?**: No

### Accuracy Difference
- **Baseline Accuracy**: 0.1986
- **Pyramidal Accuracy**: 0.1894
- **Difference**: -0.0091
- **T-statistic**: 2.2882
- **P-value**: 0.0221

## Conclusion

✗ Baseline model shows better calibration.

✗ Baseline model shows better perplexity.

✗ ECE improvement is not statistically significant.

