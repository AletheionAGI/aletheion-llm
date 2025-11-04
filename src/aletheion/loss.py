"""VARO (Variance-Adjusted Ranking Optimization) loss for epistemic training.

This module implements the loss function described in Section 6 of the Aletheion paper:
    L = L_CE + λ * ||u - u*||²

Where:
    - L_CE: Standard cross-entropy loss
    - u: Predicted uncertainty from epistemic gates
    - u*: Target uncertainty (computed from data ambiguity or head variance)
    - λ: Hyperparameter controlling uncertainty regularization strength

References:
    Aletheion paper, Section 6 (Training with VARO)
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class VaroLoss(nn.Module):
    """Variance-Adjusted Ranking Optimization loss.

    Combines cross-entropy with uncertainty regularization to train epistemic gates.

    Args:
        lambda_varo: Weight for uncertainty regularization term (λ in paper)
        u_star_method: Method for computing target uncertainty:
            - 'head_variance': Use variance across attention heads (Phase 0-1)
            - 'data_ambiguity': Use label ambiguity (Phase 2)
            - 'uniform': Use uniform uncertainty as baseline
        ignore_index: Index to ignore in cross-entropy (typically padding token)
    """

    def __init__(
        self,
        lambda_varo: float = 0.1,
        u_star_method: str = 'head_variance',
        ignore_index: int = -100
    ) -> None:
        super().__init__()
        self.lambda_varo = lambda_varo
        self.u_star_method = u_star_method
        self.ignore_index = ignore_index

        # Cross-entropy loss
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        uncertainty: torch.Tensor,
        u_star: Optional[torch.Tensor] = None,
        head_logits: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute VARO loss.

        Args:
            logits: Model logits of shape (batch, seq_len, vocab_size) or (batch, vocab_size)
            targets: Target labels of shape (batch, seq_len) or (batch,)
            uncertainty: Predicted uncertainty of shape (batch, seq_len, 1) or (batch, 1)
            u_star: Target uncertainty (optional, will be computed if not provided)
            head_logits: Logits from different attention heads for variance computation
                        Shape: (n_heads, batch, seq_len, vocab_size)

        Returns:
            Dictionary containing:
                - 'loss': Total loss (L_CE + λ * L_uncertainty)
                - 'ce_loss': Cross-entropy component
                - 'uncertainty_loss': Uncertainty regularization component
                - 'u_star_mean': Mean target uncertainty
                - 'u_pred_mean': Mean predicted uncertainty
        """
        # Reshape logits and targets for cross-entropy
        if logits.dim() == 3:
            batch, seq_len, vocab_size = logits.shape
            logits_flat = logits.reshape(-1, vocab_size)
            targets_flat = targets.reshape(-1)
        else:
            logits_flat = logits
            targets_flat = targets

        # Compute cross-entropy loss
        ce_loss = self.ce_loss(logits_flat, targets_flat)

        # Compute target uncertainty if not provided
        if u_star is None:
            u_star = self._compute_u_star(
                logits=logits,
                targets=targets,
                head_logits=head_logits
            )

        # Match shapes for uncertainty loss
        uncertainty_squeezed = uncertainty.squeeze(-1) if uncertainty.dim() > targets.dim() else uncertainty
        u_star_squeezed = u_star.squeeze(-1) if u_star.dim() > targets.dim() else u_star

        # Create mask for valid positions (ignore padding)
        if targets.dim() == 2:
            # Sequence targets
            valid_mask = (targets != self.ignore_index).float()
        else:
            # Single token targets
            valid_mask = torch.ones_like(targets, dtype=torch.float32)

        # Compute MSE between predicted and target uncertainty
        uncertainty_diff = (uncertainty_squeezed - u_star_squeezed) ** 2
        uncertainty_loss = (uncertainty_diff * valid_mask).sum() / (valid_mask.sum() + 1e-8)

        # Total VARO loss
        total_loss = ce_loss + self.lambda_varo * uncertainty_loss

        # Return detailed breakdown
        return {
            'loss': total_loss,
            'ce_loss': ce_loss,
            'uncertainty_loss': uncertainty_loss,
            'u_star_mean': (u_star_squeezed * valid_mask).sum() / (valid_mask.sum() + 1e-8),
            'u_pred_mean': (uncertainty_squeezed * valid_mask).sum() / (valid_mask.sum() + 1e-8)
        }

    def _compute_u_star(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        head_logits: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute target uncertainty u*.

        Implements different strategies from paper Section 6.1.1:
        - Method 2 (Head Variance): u* = σ²(heads) / (σ²(heads) + 1)
        - Fallback (Uniform): u* = 0.5 for all tokens

        Args:
            logits: Model logits
            targets: Target labels
            head_logits: Per-head logits for variance computation

        Returns:
            u_star: Target uncertainty of same shape as targets
        """
        if self.u_star_method == 'head_variance' and head_logits is not None:
            return self._compute_head_variance_uncertainty(head_logits)

        elif self.u_star_method == 'data_ambiguity':
            # For future implementation with multi-label data
            # Currently returns moderate uncertainty
            return torch.full_like(targets, 0.5, dtype=torch.float32)

        else:
            # Uniform uncertainty as baseline
            return torch.full_like(targets, 0.5, dtype=torch.float32)

    def _compute_head_variance_uncertainty(
        self,
        head_logits: torch.Tensor
    ) -> torch.Tensor:
        """Compute uncertainty from variance across attention heads.

        From paper Equation (14):
            u* = σ²(z_h) / (σ²(z_h) + 1)

        Where z_h are logits from different heads.

        Args:
            head_logits: Logits from H heads, shape (H, batch, seq_len, vocab_size)
                        or (H, batch, vocab_size)

        Returns:
            u_star: Normalized variance in [0, 1]
        """
        # Compute variance across heads (dim=0)
        variance = torch.var(head_logits, dim=0)  # (batch, seq_len, vocab_size) or (batch, vocab_size)

        # Average variance across vocabulary dimension
        mean_variance = variance.mean(dim=-1)  # (batch, seq_len) or (batch,)

        # Normalize to [0, 1]: u* = var / (var + 1)
        u_star = mean_variance / (mean_variance + 1.0)

        return u_star


def create_uncertainty_targets_from_token_frequency(
    tokens: torch.Tensor,
    vocab_size: int,
    token_counts: Optional[torch.Tensor] = None,
    rare_threshold: float = 0.01
) -> torch.Tensor:
    """Create uncertainty targets based on token rarity.

    Rare tokens should have higher uncertainty as the model has seen fewer examples.

    Args:
        tokens: Token IDs of shape (batch, seq_len)
        vocab_size: Size of vocabulary
        token_counts: Pre-computed token frequency counts of shape (vocab_size,)
                     If None, returns uniform uncertainty
        rare_threshold: Fraction of total tokens below which a token is considered rare

    Returns:
        u_star: Uncertainty targets of shape (batch, seq_len)
                High uncertainty (≈1.0) for rare tokens
                Low uncertainty (≈0.0) for common tokens
    """
    if token_counts is None:
        # Return moderate uncertainty if no frequency info available
        return torch.full_like(tokens, 0.5, dtype=torch.float32)

    # Normalize token counts to probabilities
    token_probs = token_counts / (token_counts.sum() + 1e-8)

    # Get probability for each token in the batch
    token_probs_batch = token_probs[tokens]  # (batch, seq_len)

    # Map probability to uncertainty:
    # - Rare tokens (low prob) → high uncertainty
    # - Common tokens (high prob) → low uncertainty
    # Using: u* = 1 - min(p / threshold, 1)
    u_star = 1.0 - torch.clamp(token_probs_batch / rare_threshold, 0.0, 1.0)

    return u_star


def create_uncertainty_targets_from_ambiguity(
    targets: torch.Tensor,
    valid_labels: Optional[Dict[int, list]] = None
) -> torch.Tensor:
    """Create uncertainty targets based on label ambiguity.

    From paper Equation (15):
        u* = 1 - 1/|Y|

    Where |Y| is the number of valid labels for a given example.

    Args:
        targets: Target labels of shape (batch, seq_len)
        valid_labels: Dictionary mapping position to list of valid labels
                     Example: {0: [1, 2], 1: [3]} means position 0 has 2 valid labels

    Returns:
        u_star: Uncertainty targets where ambiguous positions have higher uncertainty
    """
    if valid_labels is None:
        # No ambiguity information, return low uncertainty
        return torch.zeros_like(targets, dtype=torch.float32)

    u_star = torch.zeros_like(targets, dtype=torch.float32)

    for pos, labels in valid_labels.items():
        num_valid = len(labels)
        if num_valid > 1:
            # Ambiguous: u* = 1 - 1/num_valid
            u_star[pos] = 1.0 - (1.0 / num_valid)
        else:
            # Unambiguous: u* = 0
            u_star[pos] = 0.0

    return u_star


def compute_calibration_metrics(
    probs: torch.Tensor,
    targets: torch.Tensor,
    uncertainty: torch.Tensor,
    n_bins: int = 10
) -> Dict[str, float]:
    """Compute calibration metrics (ECE, Brier score, etc.).

    Args:
        probs: Predicted probabilities of shape (batch, vocab_size)
        targets: True labels of shape (batch,)
        uncertainty: Predicted uncertainty of shape (batch, 1)
        n_bins: Number of bins for ECE computation

    Returns:
        Dictionary containing:
            - 'ece': Expected Calibration Error
            - 'brier_score': Brier score
            - 'uncertainty_error_corr': Correlation between uncertainty and error
    """
    # Get predicted class and confidence
    confidence, predicted = torch.max(probs, dim=-1)
    correct = (predicted == targets).float()

    # Compute ECE (Expected Calibration Error)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidence >= bin_lower) & (confidence < bin_upper)
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin > 0:
            accuracy_in_bin = correct[in_bin].mean()
            avg_confidence_in_bin = confidence[in_bin].mean()
            ece += prop_in_bin * torch.abs(accuracy_in_bin - avg_confidence_in_bin)

    # Compute Brier score
    one_hot = F.one_hot(targets, num_classes=probs.size(-1)).float()
    brier_score = torch.mean((probs - one_hot) ** 2)

    # Compute correlation between uncertainty and error
    errors = (1.0 - correct).unsqueeze(-1)  # (batch, 1)
    uncertainty_squeezed = uncertainty.squeeze(-1) if uncertainty.dim() > 1 else uncertainty

    # Pearson correlation
    if errors.numel() > 1:
        uncertainty_error_corr = torch.corrcoef(
            torch.stack([uncertainty_squeezed, errors.squeeze()])
        )[0, 1].item()
    else:
        uncertainty_error_corr = 0.0

    return {
        'ece': ece.item(),
        'brier_score': brier_score.item(),
        'uncertainty_error_corr': uncertainty_error_corr
    }
