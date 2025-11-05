"""Pyramidal Epistemic Architecture with Q1/Q2/Fractal Integration.

This module implements the complete integration of:
1. Q1 (Aleatoric Uncertainty) - irreducible uncertainty
2. Q2 (Epistemic Uncertainty) - reducible uncertainty
3. Fractal System - meta-epistemic (uncertainty about uncertainty)
4. Pyramidal Architecture - 5 vertices (Memory, Pain, Choice, Exploration + Truth apex)

The key insight: Q1 and Q2 MODULATE the height (proximity to truth apex),
preventing the collapse observed in tetrahedral architecture.

References:
    - Aletheion Preprint v4.0
    - Geometry of Knowing
    - Pyramidal Epistemology Technical Report (Nov 2025)
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PyramidalEpistemicGatesWithQ1Q2(nn.Module):
    """Complete Pyramidal Architecture with Q1, Q2, Fractal.

    Architecture:
        BASE (4D simplex): Memory, Pain, Choice, Exploration
        Q1 (Aleatoric): Irreducible uncertainty gate
        Q2 (Epistemic): Reducible uncertainty gate
        HEIGHT (derived): Derived from Q1, Q2, base_stability
        FRACTAL: Meta-epistemic uncertainty (uncertainty about uncertainty)
        TRUTH (apex): Constant = 1.0 (attractor)

    The height is NOT independent - it's derived from:
        height = f(1-Q1, 1-Q2, base_stability)

    This prevents horizontal collapse by tying height to epistemic quality.

    Args:
        d_model: Hidden dimension of transformer
        n_heads: Number of attention heads
        dropout: Dropout probability
        use_multi_head_height: Whether to use multi-head consensus for height
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        use_multi_head_height: bool = False
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.use_multi_head_height = use_multi_head_height

        # === BASE: 4 cognitive forces ===
        self.base_projection = nn.Linear(d_model, 4)

        # === Q1: ALEATORIC UNCERTAINTY (irreducible) ===
        # Mean and variance (fractal layer)
        self.Q1_mean_gate = nn.Linear(d_model, 1)
        self.Q1_var_gate = nn.Linear(d_model, 1)

        # === Q2: EPISTEMIC UNCERTAINTY (reducible) ===
        # Mean and variance (fractal layer)
        self.Q2_mean_gate = nn.Linear(d_model, 1)
        self.Q2_var_gate = nn.Linear(d_model, 1)

        # === HEIGHT: Derived from Q1, Q2, base_stability ===
        # Maps [Q1_inv, Q2_inv, base_stability] → height
        self.height_combiner = nn.Linear(3, 1)

        # === FRACTAL: Meta-epistemic uncertainty ===
        self.fractal_gate = nn.Linear(d_model, 1)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # === INITIALIZATION ===
        # Base projection: slight bias toward balance
        nn.init.normal_(self.base_projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.base_projection.bias)

        # Q1 mean: start at ~0.3 (low aleatoric uncertainty)
        nn.init.normal_(self.Q1_mean_gate.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.Q1_mean_gate.bias, -0.5)  # sigmoid(-0.5) ≈ 0.38

        # Q1 variance: start small
        nn.init.normal_(self.Q1_var_gate.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.Q1_var_gate.bias, -1.0)  # softplus(-1) ≈ 0.31

        # Q2 mean: start at ~0.5 (moderate epistemic uncertainty)
        nn.init.normal_(self.Q2_mean_gate.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.Q2_mean_gate.bias, 0.0)  # sigmoid(0) = 0.5

        # Q2 variance: start small
        nn.init.normal_(self.Q2_var_gate.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.Q2_var_gate.bias, -1.0)

        # Height combiner: positive weights for inverted Q1, Q2
        nn.init.normal_(self.height_combiner.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.height_combiner.bias)

        # Fractal: start at ~0.2 (low meta-uncertainty)
        nn.init.normal_(self.fractal_gate.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.fractal_gate.bias, -1.0)  # sigmoid(-1) ≈ 0.27

    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through pyramidal gates with Q1/Q2/Fractal.

        Args:
            hidden_states: Transformer hidden states [batch, seq_len, d_model]

        Returns:
            Dictionary containing:
                - base_weights: [B, T, 4] - (memory, pain, choice, exploration)
                - w_memory, w_pain, w_choice, w_exploration: [B, T, 1] - individual forces
                - Q1_mean, Q1_var: [B, T, 1] - aleatoric uncertainty
                - Q2_mean, Q2_var: [B, T, 1] - epistemic uncertainty
                - height: [B, T, 1] - derived from Q1, Q2, base_stability
                - uncertainty: [B, T, 1] - 1 - height
                - fractal_uncertainty: [B, T, 1] - meta-epistemic
                - total_uncertainty: [B, T, 1] - Q1 + Q2_fractal
                - confidence: [B, T, 1] - height * (1 - fractal)
                - base_stability, base_variance: [B, T, 1]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Apply dropout
        hidden_dropped = self.dropout(hidden_states)

        # === 1. BASE WEIGHTS (4 forces, sum to 1) ===
        base_logits = self.base_projection(hidden_dropped)
        base_weights = F.softmax(base_logits, dim=-1)  # [B, T, 4]

        # Extract individual forces
        w_memory = base_weights[..., 0:1]
        w_pain = base_weights[..., 1:2]
        w_choice = base_weights[..., 2:3]
        w_exploration = base_weights[..., 3:4]

        # Base stability
        base_variance = base_weights.var(dim=-1, keepdim=True)
        base_stability = 1.0 - base_variance

        # === 2. Q1: ALEATORIC UNCERTAINTY (irreducible) ===
        Q1_mean = torch.sigmoid(self.Q1_mean_gate(hidden_dropped))
        Q1_var = F.softplus(self.Q1_var_gate(hidden_dropped))  # Fractal layer

        # === 3. Q2: EPISTEMIC UNCERTAINTY (reducible) ===
        Q2_mean = torch.sigmoid(self.Q2_mean_gate(hidden_dropped))
        Q2_var = F.softplus(self.Q2_var_gate(hidden_dropped))  # Fractal layer

        # === 4. HEIGHT: Derived from Q1, Q2, base_stability ===
        # High height when: low Q1 + low Q2 + high base_stability
        # This creates natural attractor toward Truth apex
        height_inputs = torch.cat([
            1.0 - Q1_mean,      # Invert: low aleatoric → high height
            1.0 - Q2_mean,      # Invert: low epistemic → high height
            base_stability      # High stability → high height
        ], dim=-1)

        height_logits = self.height_combiner(height_inputs)
        height = torch.sigmoid(height_logits)  # [B, T, 1] ∈ [0,1]

        # === 5. FRACTAL: Meta-epistemic uncertainty ===
        fractal_logits = self.fractal_gate(hidden_dropped)
        fractal_uncertainty = torch.sigmoid(fractal_logits)

        # === 6. TOTAL UNCERTAINTY ===
        # Q2 inflated by fractal (uncertainty about epistemic uncertainty)
        Q2_fractal = Q2_mean * (1.0 + fractal_uncertainty)
        total_uncertainty = Q1_mean + Q2_fractal

        # === 7. CONFIDENCE ===
        # High when near apex AND low meta-uncertainty
        confidence = height * (1.0 - fractal_uncertainty)

        return {
            # Base
            'base_weights': base_weights,
            'w_memory': w_memory,
            'w_pain': w_pain,
            'w_choice': w_choice,
            'w_exploration': w_exploration,
            'base_stability': base_stability,
            'base_variance': base_variance,

            # Q1 (Aleatoric)
            'Q1_mean': Q1_mean,
            'Q1_var': Q1_var,

            # Q2 (Epistemic)
            'Q2_mean': Q2_mean,
            'Q2_var': Q2_var,

            # Height (Derived)
            'height': height,
            'uncertainty': 1.0 - height,

            # Fractal
            'fractal_uncertainty': fractal_uncertainty,
            'total_uncertainty': total_uncertainty,

            # Confidence
            'confidence': confidence
        }


class EpistemicMultiHeadAttention(nn.Module):
    """Multi-head attention with epistemic softmax replacement.

    Instead of standard softmax over attention weights, uses epistemic_softmax
    that modulates distribution based on Q1, Q2 confidence.

    This is Level 2/3 fractal implementation: replacing ALL softmax calls.

    Args:
        d_model: Hidden dimension
        n_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.d_k = d_model // n_heads

        # Q, K, V projections
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        Q1_gate: nn.Module,
        Q2_gate: nn.Module,
        tau_thresh: float = 0.3
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with epistemic softmax.

        Args:
            x: Input hidden states [B, T, d_model]
            Q1_gate: Q1 mean gate module
            Q2_gate: Q2 mean gate module
            tau_thresh: Confidence threshold for temperature modulation

        Returns:
            output: Attention output [B, T, d_model]
            uncertainty: Uncertainty scalar [B, 1]
        """
        B, T, _ = x.shape

        # 1. Project to Q, K, V
        Q = self.W_Q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        # Shape: [B, n_heads, T, d_k]

        # 2. Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # Shape: [B, n_heads, T, T]

        # 3. EPISTEMIC SOFTMAX REPLACEMENT
        # Use first query token as context for gates
        context = Q[:, :, 0, :]  # [B, n_heads, d_k]
        context = context.reshape(B, -1)  # [B, n_heads * d_k]

        # Compute Q1, Q2
        q1 = torch.sigmoid(Q1_gate(context))  # [B, 1]
        q2 = torch.sigmoid(Q2_gate(context))  # [B, 1]
        c = torch.clamp(q1 * q2, min=1e-8, max=1.0)

        # Temperature modulation
        tau = torch.where(
            c < tau_thresh,
            1.0 / c,
            torch.ones_like(c)
        )
        tau = tau.view(B, 1, 1, 1)  # Broadcast to [B, n_heads, T, T]

        # Gated softmax
        p = F.softmax(scores / tau, dim=-1)
        u_uniform = 1.0 / T
        p_gated = c.view(B, 1, 1, 1) * p + (1 - c.view(B, 1, 1, 1)) * u_uniform

        # Uncertainty
        u = (1 - c).mean(dim=-1, keepdim=True)  # [B, 1]

        # 4. Apply gated attention
        attn_out = torch.matmul(self.dropout(p_gated), V)  # [B, n_heads, T, d_k]
        attn_out = attn_out.transpose(1, 2).reshape(B, T, -1)

        # 5. Output projection
        output = self.W_O(attn_out)

        return output, u


class PyramidalVAROLossWithQ1Q2(nn.Module):
    """Pyramidal VARO Loss with Q1, Q2, Fractal components.

    Total loss:
        L = L_CE + λ_base * L_base + λ_Q1 * L_Q1 + λ_Q2 * L_Q2
            + λ_fractal * L_fractal + λ_height * L_height

    Components:
        - L_CE: Cross-entropy (task loss)
        - L_base: Base stability (variance of 4 forces)
        - L_Q1: Q1 calibration (aleatoric uncertainty)
        - L_Q2: Q2 calibration (epistemic uncertainty)
        - L_fractal: Fractal regularization (prevent explosion)
        - L_height: Height calibration (derived from Q1, Q2)

    Args:
        lambda_base: Weight for base stability
        lambda_Q1: Weight for Q1 calibration
        lambda_Q2: Weight for Q2 calibration
        lambda_fractal: Weight for fractal regularization
        lambda_height: Weight for height calibration
        ignore_index: Padding token index to ignore
    """

    def __init__(
        self,
        lambda_base: float = 0.001,
        lambda_Q1: float = 0.0015,
        lambda_Q2: float = 0.002,
        lambda_fractal: float = 0.0005,
        lambda_height: float = 0.002,
        ignore_index: int = -100
    ) -> None:
        super().__init__()
        self.lambda_base = lambda_base
        self.lambda_Q1 = lambda_Q1
        self.lambda_Q2 = lambda_Q2
        self.lambda_fractal = lambda_fractal
        self.lambda_height = lambda_height
        self.ignore_index = ignore_index

    def compute_target_Q1(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute target Q1 (aleatoric uncertainty).

        Q1 should be HIGH when:
        - Intrinsic randomness in the prediction
        - Even the correct class has low probability

        Method: Use probability of correct class
        - High prob → low Q1 (low aleatoric)
        - Low prob → high Q1 (high aleatoric)

        Args:
            logits: Model logits [B, T, V] or [B*T, V]
            targets: Target labels [B, T] or [B*T]

        Returns:
            target_Q1: [B, T, 1] or [B*T, 1]
        """
        # Reshape if needed
        if logits.dim() == 3:
            B, T, V = logits.shape
            logits_flat = logits.reshape(-1, V)
            targets_flat = targets.reshape(-1)
            need_reshape = True
        else:
            logits_flat = logits
            targets_flat = targets
            need_reshape = False

        # Softmax probabilities
        probs = F.softmax(logits_flat, dim=-1)

        # Probability of correct class
        correct_probs = probs.gather(-1, targets_flat.unsqueeze(-1))

        # Q1 = 1 - correct_prob (high uncertainty when low prob)
        target_Q1 = 1.0 - correct_probs

        if need_reshape:
            target_Q1 = target_Q1.reshape(B, T, 1)

        return target_Q1

    def compute_target_Q2(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute target Q2 (epistemic uncertainty).

        Q2 should be HIGH when:
        - Model doesn't know (reducible with more data)
        - Prediction is wrong
        - High entropy in distribution

        Method: Combine correctness + entropy

        Args:
            logits: Model logits [B, T, V] or [B*T, V]
            targets: Target labels [B, T] or [B*T]

        Returns:
            target_Q2: [B, T, 1] or [B*T, 1]
        """
        # Reshape if needed
        if logits.dim() == 3:
            B, T, V = logits.shape
            logits_flat = logits.reshape(-1, V)
            targets_flat = targets.reshape(-1)
            need_reshape = True
        else:
            logits_flat = logits
            targets_flat = targets
            need_reshape = False

        # Softmax probabilities
        probs = F.softmax(logits_flat, dim=-1)

        # Method 1: Correctness
        confidence, predictions = probs.max(dim=-1)
        correct = predictions.eq(targets_flat).float()
        target_Q2_confidence = 1.0 - correct

        # Method 2: Entropy
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)
        max_entropy = math.log(probs.size(-1))
        target_Q2_entropy = entropy / max_entropy

        # Combine: average of both signals
        target_Q2 = (target_Q2_confidence + target_Q2_entropy) / 2.0
        target_Q2 = target_Q2.unsqueeze(-1)

        if need_reshape:
            target_Q2 = target_Q2.reshape(B, T, 1)

        return target_Q2

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        pyramid_outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute total pyramidal VARO loss with Q1/Q2/Fractal.

        Args:
            logits: Model logits [B, T, V]
            targets: Target labels [B, T]
            pyramid_outputs: Dictionary from PyramidalEpistemicGatesWithQ1Q2

        Returns:
            Dictionary with loss components and metrics
        """
        # Standard cross-entropy
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=self.ignore_index
        )

        # Extract pyramid state
        base_weights = pyramid_outputs['base_weights']
        Q1_mean = pyramid_outputs['Q1_mean']
        Q2_mean = pyramid_outputs['Q2_mean']
        Q1_var = pyramid_outputs['Q1_var']
        Q2_var = pyramid_outputs['Q2_var']
        height = pyramid_outputs['height']
        fractal_uncertainty = pyramid_outputs['fractal_uncertainty']

        # Valid mask (ignore padding)
        valid_mask = (targets != self.ignore_index).unsqueeze(-1)  # [B, T, 1]

        # === 1. BASE STABILITY LOSS ===
        base_variance = base_weights.var(dim=-1, keepdim=True)
        base_loss = (base_variance * valid_mask).sum() / (valid_mask.sum() + 1e-8)

        # === 2. Q1 CALIBRATION LOSS ===
        target_Q1 = self.compute_target_Q1(logits, targets)
        Q1_loss = F.mse_loss(
            Q1_mean * valid_mask,
            target_Q1 * valid_mask,
            reduction='sum'
        ) / (valid_mask.sum() + 1e-8)

        # === 3. Q2 CALIBRATION LOSS ===
        target_Q2 = self.compute_target_Q2(logits, targets)
        Q2_loss = F.mse_loss(
            Q2_mean * valid_mask,
            target_Q2 * valid_mask,
            reduction='sum'
        ) / (valid_mask.sum() + 1e-8)

        # === 4. FRACTAL REGULARIZATION ===
        # Penalize excessive meta-uncertainty
        fractal_loss = (fractal_uncertainty ** 2 * valid_mask).sum() / (valid_mask.sum() + 1e-8)

        # === 5. HEIGHT CALIBRATION ===
        # Height should reflect inverse of total uncertainty
        target_height = 1.0 - (Q1_mean + Q2_mean) / 2.0
        height_loss = F.mse_loss(
            height * valid_mask,
            target_height * valid_mask,
            reduction='sum'
        ) / (valid_mask.sum() + 1e-8)

        # === TOTAL LOSS ===
        total_loss = ce_loss \
                   + self.lambda_base * base_loss \
                   + self.lambda_Q1 * Q1_loss \
                   + self.lambda_Q2 * Q2_loss \
                   + self.lambda_fractal * fractal_loss \
                   + self.lambda_height * height_loss

        # Compute metrics for logging
        num_valid = valid_mask.sum() + 1e-8

        return {
            'loss': total_loss,
            'ce_loss': ce_loss.item(),
            'base_loss': base_loss.item(),
            'Q1_loss': Q1_loss.item(),
            'Q2_loss': Q2_loss.item(),
            'fractal_loss': fractal_loss.item(),
            'height_loss': height_loss.item(),

            # Metrics
            'mean_Q1': (Q1_mean * valid_mask).sum().item() / num_valid.item(),
            'mean_Q2': (Q2_mean * valid_mask).sum().item() / num_valid.item(),
            'mean_height': (height * valid_mask).sum().item() / num_valid.item(),
            'mean_fractal': (fractal_uncertainty * valid_mask).sum().item() / num_valid.item(),
            'target_Q1_mean': (target_Q1 * valid_mask).sum().item() / num_valid.item(),
            'target_Q2_mean': (target_Q2 * valid_mask).sum().item() / num_valid.item(),
            'target_height_mean': (target_height * valid_mask).sum().item() / num_valid.item(),

            # Lambdas
            'lambda_base': self.lambda_base,
            'lambda_Q1': self.lambda_Q1,
            'lambda_Q2': self.lambda_Q2,
            'lambda_fractal': self.lambda_fractal,
            'lambda_height': self.lambda_height
        }


def compute_pyramidal_q1q2_metrics(
    pyramid_outputs: Dict[str, torch.Tensor],
    valid_mask: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """Compute aggregate metrics from pyramidal Q1/Q2 outputs.

    Args:
        pyramid_outputs: Output from PyramidalEpistemicGatesWithQ1Q2
        valid_mask: Optional mask for valid tokens

    Returns:
        Dictionary with comprehensive metrics
    """
    if valid_mask is None:
        valid_mask = torch.ones_like(pyramid_outputs['height'], dtype=torch.bool)

    def masked_mean(tensor: torch.Tensor) -> float:
        masked = tensor[valid_mask]
        return masked.mean().item() if masked.numel() > 0 else 0.0

    def masked_std(tensor: torch.Tensor) -> float:
        masked = tensor[valid_mask]
        return masked.std().item() if masked.numel() > 1 else 0.0

    def masked_entropy(tensor: torch.Tensor) -> float:
        """Compute entropy treating values as probabilities."""
        masked = tensor[valid_mask]
        if masked.numel() == 0:
            return 0.0
        # Binary entropy: H(p) = -p*log(p) - (1-p)*log(1-p)
        p = masked.clamp(1e-8, 1-1e-8)
        entropy = -(p * p.log() + (1-p) * (1-p).log())
        return entropy.mean().item()

    metrics = {
        # Q1 metrics (Aleatoric)
        'Q1_mean': masked_mean(pyramid_outputs['Q1_mean']),
        'Q1_std': masked_std(pyramid_outputs['Q1_mean']),
        'Q1_entropy': masked_entropy(pyramid_outputs['Q1_mean']),
        'Q1_var_mean': masked_mean(pyramid_outputs['Q1_var']),

        # Q2 metrics (Epistemic)
        'Q2_mean': masked_mean(pyramid_outputs['Q2_mean']),
        'Q2_std': masked_std(pyramid_outputs['Q2_mean']),
        'Q2_entropy': masked_entropy(pyramid_outputs['Q2_mean']),
        'Q2_var_mean': masked_mean(pyramid_outputs['Q2_var']),

        # Height metrics
        'height_mean': masked_mean(pyramid_outputs['height']),
        'height_std': masked_std(pyramid_outputs['height']),
        'height_entropy': masked_entropy(pyramid_outputs['height']),

        # Fractal metrics
        'fractal_mean': masked_mean(pyramid_outputs['fractal_uncertainty']),
        'fractal_std': masked_std(pyramid_outputs['fractal_uncertainty']),

        # Total uncertainty
        'total_uncertainty_mean': masked_mean(pyramid_outputs['total_uncertainty']),

        # Confidence
        'confidence_mean': masked_mean(pyramid_outputs['confidence']),

        # Base metrics
        'base_stability_mean': masked_mean(pyramid_outputs['base_stability']),
        'w_memory_mean': masked_mean(pyramid_outputs['w_memory']),
        'w_pain_mean': masked_mean(pyramid_outputs['w_pain']),
        'w_choice_mean': masked_mean(pyramid_outputs['w_choice']),
        'w_exploration_mean': masked_mean(pyramid_outputs['w_exploration'])
    }

    return metrics
