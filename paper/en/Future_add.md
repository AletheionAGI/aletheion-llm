### Training Dynamics: The Height Drift Problem

Without explicit Q1/Q2 supervision, Height coordinate
drifts toward apex (1.0), causing severe overconfidence:

Step    | Height | ECE   | Interpretation
--------|--------|-------|------------------
10.5k   | 0.624  | 0.011 | Optimal balance
23.5k   | 0.892  | 0.033 | Drift begins
34.5k   | 0.971  | 0.057 | Severe drift
41.0k   | 0.989  | 0.070 | Near-apex overconfidence

[FIGURA: Height vs ECE correlation R² > 0.95]

Critically, base stability remained near-perfect (0.99-1.00)
throughout, indicating that gate collapse was NOT the issue.
The problem is purely Height drift.

This motivates Q1Q2 architecture: explicit aleatoric (Q1)
and epistemic (Q2) gates provide supervision that anchors
Height coordinate, preventing drift.