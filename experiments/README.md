# AletheionGuard — Experiments

This directory contains empirical runs and logs exploring epistemic calibration and pyramidal dynamics.

## 1. Purpose
These experiments aim to study the emergence of meta-calibration behaviors in the Aletheion architecture, focusing on:
- differentiation between aleatoric (Q₁) and epistemic (Q₂) uncertainty,
- the relationship between Height and Calibration Error (ECE/Brier),
- and stability of the fractal feedback loop across training steps.

## 2. Contents
Typical subfolders include:
- `logs/` – raw console outputs and metric dumps
- `checkpoints/` – model weights at key steps (e.g., 25k, 30k, 60k)
- `plots/` – visualizations of Q₁, Q₂, Height, ECE, and Forces
- `notebooks/` – analysis scripts or exploratory notebooks

## 3. Reproducibility
Each run should define:
- dataset or synthetic source used
- seed and hyperparameters (if different from defaults)
- commit hash or tag of the codebase

## 4. Observations (for reviewers)
Emergent behaviors are expected in the transition regime (20k–30k steps):
- self-calibration and meta-uncertainty feedbacks
- spontaneous reduction of overconfidence without explicit supervision
- bifurcation between "false humility" (Height→1 with Q₁,Q₂≈0) and "recognized humility" (Height→0)

Reviewers are encouraged to replicate the logs to observe these transitions.

## 5. Reproducibility Notes
If reproducing:
1. Run `train.py` with the same seed and dataset described above.
2. Validate using `evaluate.py` or the integrated validation loop.
3. Compare trajectories (Q₁, Q₂, Height, ECE, Fractal Stability).

> The emergent phenomena are not deterministic but statistically consistent across seeds.
