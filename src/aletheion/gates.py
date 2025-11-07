"""Epistemic gates for uncertainty-aware softmax operations.

This module implements the core components of Aletheion Level 1:
- LocalUncertaintyGate (Q₁): Estimates local evidence quality
- CrossContextGate (Q₂): Estimates cross-context consensus
- epistemic_softmax: Algorithm 1 from the paper

References:
    Aletheion paper, Section 4 (Epistemic Softmax) and Section 5.1 (Level 1)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalUncertaintyGate(nn.Module):
    """Q₁ gate: Local uncertainty estimation.

    Maps context features to [0,1] where:
    - 1.0 = high confidence (sufficient local evidence)
    - 0.0 = low confidence (insufficient local evidence)

    Architecture: Linear(d_model -> 1) + Sigmoid

    Args:
        d_model: Hidden dimension of the transformer
        dropout: Dropout probability for regularization
    """

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model

        # Simple linear projection to scalar + sigmoid
        self.projection = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

        # Initialize bias to positive value so initial q1 ≈ 0.7 (confident but not saturated)
        nn.init.constant_(self.projection.bias, 0.8)
        nn.init.normal_(self.projection.weight, mean=0.0, std=0.02)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Compute local uncertainty gate.

        Args:
            context: Context features of shape (batch, seq_len, d_model) or (batch, d_model)

        Returns:
            q1: Local confidence scores in [0,1] of shape (batch, seq_len, 1) or (batch, 1)
        """
        # Validate input shape
        if context.dim() not in [2, 3]:
            raise ValueError(f"Expected context to be 2D or 3D, got shape {context.shape}")

        # Apply dropout for regularization
        context = self.dropout(context)

        # Project to scalar and apply sigmoid
        q1 = torch.sigmoid(self.projection(context))

        return q1


class CrossContextGate(nn.Module):
    """Q₂ gate: Cross-context consensus estimation.

    Estimates agreement across different contexts (e.g., attention heads, layers).
    Uses multi-head attention to aggregate information from sibling contexts.

    Architecture: Multi-head attention + mean pooling + Linear + Sigmoid

    Args:
        d_model: Hidden dimension
        n_heads: Number of attention heads for consensus estimation
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.d_head = d_model // n_heads

        # Query, Key, Value projections for cross-attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # Output projection
        self.out_proj = nn.Linear(d_model, 1)

        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

        # Initialize bias for confident initial behavior
        nn.init.constant_(self.out_proj.bias, 0.8)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Compute cross-context consensus gate.

        Args:
            context: Context features of shape (batch, seq_len, d_model)

        Returns:
            q2: Consensus scores in [0,1] of shape (batch, seq_len, 1)
        """
        batch_size, seq_len, _ = context.shape

        # Project to Q, K, V
        q = self.q_proj(context)  # (B, T, d_model)
        k = self.k_proj(context)  # (B, T, d_model)
        v = self.v_proj(context)  # (B, T, d_model)

        # Reshape for multi-head attention: (B, T, H, d_head)
        q = q.reshape(batch_size, seq_len, self.n_heads, self.d_head)
        k = k.reshape(batch_size, seq_len, self.n_heads, self.d_head)
        v = v.reshape(batch_size, seq_len, self.n_heads, self.d_head)

        # Transpose to (B, H, T, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        attn_scores = (q @ k.transpose(-2, -1)) / (self.d_head**0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        attn_output = attn_weights @ v  # (B, H, T, d_head)

        # Transpose back and reshape: (B, T, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)

        # Mean pooling across sequence dimension for global consensus
        consensus_features = attn_output.mean(dim=1, keepdim=True)  # (B, 1, d_model)

        # Expand to match sequence length
        consensus_features = consensus_features.expand(-1, seq_len, -1)  # (B, T, d_model)

        # Project to scalar and apply sigmoid
        q2 = torch.sigmoid(self.out_proj(self.dropout(consensus_features)))

        return q2


def epistemic_softmax(
    logits: torch.Tensor,
    context: torch.Tensor,
    q1_gate: LocalUncertaintyGate,
    q2_gate: CrossContextGate | None = None,
    base_temperature: float = 1.0,
    confidence_threshold: float = 0.7,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Epistemic softmax operator (Algorithm 1 from paper).

    Combines local and global uncertainty gates to produce a calibrated
    probability distribution that interpolates between confident (peaked)
    and uncertain (uniform) distributions based on epistemic confidence.

    Algorithm:
        1. Compute q1 = Q₁(context) - local evidence
        2. Compute q2 = Q₂(context) - cross-context consensus
        3. Confidence c = clip(q1 * q2, ε, 1)
        4. Adjust temperature: τ = τ₀/c if c < threshold else τ₀
        5. Compute p = softmax(logits/τ)
        6. Gate between peaked and uniform: p_gated = c·p + (1-c)·uniform
        7. Return (p_gated, uncertainty=1-c)

    Args:
        logits: Raw logits of shape (batch, seq_len, vocab_size) or (batch, vocab_size)
        context: Context features for gates, shape (batch, seq_len, d_model) or (batch, d_model)
        q1_gate: Local uncertainty gate module
        q2_gate: Cross-context consensus gate module (optional)
        base_temperature: Base temperature for softmax (τ₀)
        confidence_threshold: Threshold below which to increase temperature
        eps: Small constant for numerical stability

    Returns:
        p_gated: Gated probability distribution, same shape as logits
        uncertainty: Uncertainty scalar (1 - confidence), shape (batch, seq_len, 1) or (batch, 1)

    Example:
        >>> logits = torch.randn(2, 32, 1000)  # batch=2, seq=32, vocab=1000
        >>> context = torch.randn(2, 32, 512)   # batch=2, seq=32, d_model=512
        >>> q1 = LocalUncertaintyGate(d_model=512)
        >>> q2 = CrossContextGate(d_model=512, n_heads=4)
        >>> probs, uncertainty = epistemic_softmax(logits, context, q1, q2)
        >>> assert probs.shape == logits.shape
        >>> assert (probs.sum(dim=-1) - 1.0).abs().max() < 1e-5  # probabilities sum to 1
    """
    # Step 1: Compute q1 (local evidence)
    q1 = q1_gate(context)  # (B, T, 1) or (B, 1)

    # Step 2: Compute q2 (cross-context consensus) if available
    if q2_gate is not None:
        # Ensure context is 3D for Q2
        context_3d = context.unsqueeze(1) if context.dim() == 2 else context

        q2 = q2_gate(context_3d)  # (B, T, 1)

        # Match dimensions with q1
        if q1.dim() == 2 and q2.dim() == 3:
            q2 = q2.squeeze(1)  # (B, 1)
    else:
        # If no Q2 gate, assume full consensus (q2 = 1)
        q2 = torch.ones_like(q1)

    # Step 3: Compute epistemic confidence c = clip(q1 * q2, eps, 1)
    confidence = torch.clamp(q1 * q2, min=eps, max=1.0)

    # Step 4: Adjust temperature based on confidence
    # If confidence < threshold, increase temperature (flatten distribution)
    temperature = torch.where(
        confidence < confidence_threshold,
        base_temperature / confidence,
        torch.full_like(confidence, base_temperature),
    )

    # Expand temperature to match logits shape
    while temperature.dim() < logits.dim():
        temperature = temperature.unsqueeze(-1)

    # Step 5: Compute temperature-scaled softmax
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)

    # Step 6: Create uniform distribution
    vocab_size = logits.size(-1)
    uniform_probs = torch.full_like(probs, 1.0 / vocab_size)

    # Expand confidence to match probs shape
    confidence_expanded = confidence
    while confidence_expanded.dim() < probs.dim():
        confidence_expanded = confidence_expanded.unsqueeze(-1)

    # Step 7: Interpolate between peaked (confident) and uniform (uncertain)
    p_gated = confidence_expanded * probs + (1 - confidence_expanded) * uniform_probs

    # Step 8: Compute uncertainty (inverse of confidence)
    uncertainty = 1.0 - confidence

    return p_gated, uncertainty


def entropy_regularization(gate_outputs: torch.Tensor, min_entropy: float = 0.1) -> torch.Tensor:
    """Compute entropy regularization loss to prevent gate collapse.

    Encourages gates to maintain diversity and not collapse to always-on (≈1) or always-off (≈0).

    Args:
        gate_outputs: Gate values in [0,1] of shape (batch, ...)
        min_entropy: Minimum desired entropy threshold

    Returns:
        loss: Scalar penalty if entropy falls below threshold
    """
    # Treat gate output as binary distribution: p(on)=gate, p(off)=1-gate
    p_on = gate_outputs
    p_off = 1 - gate_outputs

    # Binary entropy: H = -p*log(p) - (1-p)*log(1-p)
    entropy = -(p_on * torch.log(p_on + 1e-8) + p_off * torch.log(p_off + 1e-8))
    mean_entropy = entropy.mean()

    # Penalize if entropy drops below threshold
    penalty = F.relu(min_entropy - mean_entropy)

    return penalty
