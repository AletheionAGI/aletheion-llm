"""Pyramidal Epistemic Gates for Aletheion.

This module implements the Pyramidal Epistemology framework, which uses a 5-vertex
geometric structure to represent epistemic uncertainty:

    Base (4 vertices): Memory, Pain, Choice, Exploration
    Apex (1 vertex): Truth = 1.0 (constant attractor)

The AI learns by:
1. Balancing base forces (horizontal stability)
2. Climbing toward apex (vertical ascension to truth)

References:
    Pyramidal Epistemology: Climbing Toward Truth in Neural Networks
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PyramidalEpistemicGates(nn.Module):
    """Pyramidal epistemic architecture with 5-vertex geometry.

    Base (4 vertices): Memory, Pain, Choice, Exploration
    Apex (1 vertex): Truth = 1.0 (constant attractor)

    The model learns epistemic quality through:
    - Base weights: Distribution over 4 fundamental cognitive forces
    - Height: Scalar ∈ [0,1] representing proximity to truth/certainty

    Args:
        d_model: Hidden dimension of the transformer
        n_heads: Number of attention heads for multi-head height consensus
        dropout: Dropout probability for regularization
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

        # Base projection: 4 forces (Memory, Pain, Choice, Exploration)
        # Projects hidden state to 4-dimensional simplex (sums to 1)
        self.base_projection = nn.Linear(d_model, 4)

        # Q1 gate: Prediction quality (correctness proxy)
        # Will be computed from model predictions in forward pass

        # Q2 gate: Confidence calibration quality
        # Will be computed from confidence vs correctness in forward pass

        # Height combiner: Derives height from Q1, Q2, and base_stability
        # Input: [1-Q1, 1-Q2, base_stability] → scalar height
        self.height_combiner = nn.Linear(3, 1)

        # Optional: Multi-head height for consensus
        if use_multi_head_height:
            self.height_heads = nn.ModuleList([
                nn.Linear(d_model, 1) for _ in range(n_heads)
            ])

        self.dropout = nn.Dropout(dropout)

        # Initialize base projection with slight bias toward balance
        nn.init.normal_(self.base_projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.base_projection.bias)

        # Initialize height_combiner to start near base (low height ~0.12)
        # bias = -2.0 → sigmoid(-2) ≈ 0.12 (starts near base, not apex)
        nn.init.normal_(self.height_combiner.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.height_combiner.bias, -2.0)

    def forward(self, hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass through pyramidal gates.

        Args:
            hidden_states: Transformer hidden states of shape (batch, seq_len, d_model)

        Returns:
            Dictionary containing:
                - base_weights: [batch, seq_len, 4] - (memory, pain, choice, exploration)
                - w_memory: [batch, seq_len, 1] - memory force weight
                - w_pain: [batch, seq_len, 1] - pain force weight
                - w_choice: [batch, seq_len, 1] - choice force weight
                - w_exploration: [batch, seq_len, 1] - exploration force weight
                - height: [batch, seq_len, 1] - proximity to truth ∈ [0,1]
                - uncertainty: [batch, seq_len, 1] - 1 - height
                - confidence: [batch, seq_len, 1] - height × base_stability
                - base_stability: [batch, seq_len, 1] - 1 - base_variance
                - base_variance: [batch, seq_len, 1] - variance of base forces
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Apply dropout for regularization
        hidden_dropped = self.dropout(hidden_states)

        # === COMPUTE BASE WEIGHTS (4 forces, sum to 1) ===
        base_logits = self.base_projection(hidden_dropped)
        base_weights = F.softmax(base_logits, dim=-1)  # [batch, seq_len, 4]

        # Extract individual forces
        w_memory = base_weights[..., 0:1]
        w_pain = base_weights[..., 1:2]
        w_choice = base_weights[..., 2:3]
        w_exploration = base_weights[..., 3:4]

        # Base stability: how balanced are the 4 forces?
        base_variance = base_weights.var(dim=-1, keepdim=True)
        base_stability = 1.0 - base_variance

        # === COMPUTE HEIGHT (epistemic ascension toward truth) ===
        # Height is derived from hidden state features + base_stability
        # The height_combiner learns to produce heights that correlate with prediction quality (Q1)
        # and confidence calibration (Q2) through the training loss

        if self.use_multi_head_height and hasattr(self, 'height_heads'):
            # Multi-head height consensus
            head_heights = torch.stack([
                torch.sigmoid(head(hidden_dropped))
                for head in self.height_heads
            ], dim=-1)  # [batch, seq_len, 1, n_heads]

            # Compute mean and variance across heads
            height_mean = head_heights.mean(dim=-1, keepdim=True)  # [batch, seq_len, 1, 1]
            height_variance = head_heights.var(dim=-1, keepdim=True)  # [batch, seq_len, 1, 1]

            # Consensus-weighted height: reduce height when heads disagree
            height = height_mean * (1.0 - height_variance)
            height = height.squeeze(-1)  # [batch, seq_len, 1]
        else:
            # Create features for height prediction:
            # - hidden_mean: average activation (proxy for confidence)
            # - hidden_std: activation variance (proxy for uncertainty)
            # - base_stability: force balance (high when forces are balanced)
            hidden_mean = hidden_dropped.mean(dim=-1, keepdim=True)  # [batch, seq_len, 1]
            hidden_std = hidden_dropped.std(dim=-1, keepdim=True)    # [batch, seq_len, 1]

            # Normalize features to [0, 1] range for better stability
            hidden_mean_norm = torch.sigmoid(hidden_mean)
            hidden_std_norm = torch.sigmoid(hidden_std)

            # Combine features: [hidden_mean, hidden_std, base_stability]
            height_inputs = torch.cat([
                hidden_mean_norm,
                hidden_std_norm,
                base_stability
            ], dim=-1)  # [batch, seq_len, 3]

            # Height combiner: learns to map features → epistemic quality
            # Initialized with bias=-2.0 → sigmoid(-2)≈0.12 (starts near base)
            height_logits = self.height_combiner(height_inputs)
            height = torch.sigmoid(height_logits)  # [batch, seq_len, 1] ∈ [0,1]

        # === COMPUTE DERIVED METRICS ===

        # Uncertainty: distance from apex
        uncertainty = 1.0 - height

        # Confidence: high when near apex AND base is stable
        # This combines vertical position (height) with horizontal stability
        confidence = height * base_stability

        return {
            'base_weights': base_weights,
            'w_memory': w_memory,
            'w_pain': w_pain,
            'w_choice': w_choice,
            'w_exploration': w_exploration,
            'height': height,
            'uncertainty': uncertainty,
            'confidence': confidence,
            'base_stability': base_stability,
            'base_variance': base_variance
        }

    def get_pyramid_position(
        self,
        hidden_states: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Get the AI's position in the pyramid.

        Returns detailed geometric information about the epistemic state.

        Args:
            hidden_states: Transformer hidden states

        Returns:
            Dictionary with pyramid position metrics
        """
        outputs = self.forward(hidden_states)

        # Compute additional geometric metrics
        base_weights = outputs['base_weights']
        height = outputs['height']

        # Distance from apex (epistemic gap)
        apex_distance = 1.0 - height

        # Base entropy: how diverse are the forces?
        # High entropy = balanced exploration of all 4 forces
        # Low entropy = dominated by 1-2 forces
        base_entropy = -(base_weights * torch.log(base_weights + 1e-8)).sum(dim=-1, keepdim=True)
        max_entropy = torch.log(torch.tensor(4.0))  # log(4) for 4 categories
        base_entropy_normalized = base_entropy / max_entropy

        # Epistemic quality: high when near apex with balanced base
        quality = height * base_entropy_normalized

        return {
            **outputs,
            'apex_distance': apex_distance,
            'base_entropy': base_entropy_normalized,
            'epistemic_quality': quality
        }


class PyramidalTemperatureModulator(nn.Module):
    """Modulates softmax temperature based on pyramidal height.

    When height is low (uncertain, near base), increase temperature to flatten distribution.
    When height is high (certain, near apex), decrease temperature to sharpen distribution.

    Args:
        base_temperature: Base temperature for softmax
        max_temperature_scale: Maximum temperature multiplier when at base (height=0)
    """

    def __init__(
        self,
        base_temperature: float = 1.0,
        max_temperature_scale: float = 2.0
    ) -> None:
        super().__init__()
        self.base_temperature = base_temperature
        self.max_temperature_scale = max_temperature_scale

    def forward(
        self,
        logits: torch.Tensor,
        height: torch.Tensor
    ) -> torch.Tensor:
        """Apply height-modulated temperature to logits.

        Args:
            logits: Raw logits of shape (batch, seq_len, vocab_size)
            height: Height values of shape (batch, seq_len, 1)

        Returns:
            Temperature-scaled logits
        """
        # Compute temperature: increases as height decreases
        # temp = base_temp * (1 + (1 - height) * (max_scale - 1))
        temperature = self.base_temperature * (
            1.0 + (1.0 - height) * (self.max_temperature_scale - 1.0)
        )

        # Expand temperature to match logits shape
        while temperature.dim() < logits.dim():
            temperature = temperature.unsqueeze(-1)

        # Apply temperature scaling
        scaled_logits = logits / temperature

        return scaled_logits


def compute_pyramidal_metrics(
    pyramid_outputs: dict[str, torch.Tensor],
    valid_mask: torch.Tensor | None = None
) -> dict[str, float]:
    """Compute aggregate metrics from pyramidal outputs.

    Args:
        pyramid_outputs: Output dictionary from PyramidalEpistemicGates
        valid_mask: Optional mask for valid tokens (ignoring padding)

    Returns:
        Dictionary with aggregate metrics
    """
    if valid_mask is None:
        valid_mask = torch.ones_like(pyramid_outputs['height'], dtype=torch.bool)

    valid_mask = valid_mask.squeeze(-1) if valid_mask.dim() > pyramid_outputs['height'].dim() else valid_mask

    def masked_mean(tensor: torch.Tensor) -> float:
        """Compute mean over valid tokens."""
        masked = tensor[valid_mask]
        return masked.mean().item() if masked.numel() > 0 else 0.0

    def masked_std(tensor: torch.Tensor) -> float:
        """Compute std over valid tokens."""
        masked = tensor[valid_mask]
        return masked.std().item() if masked.numel() > 1 else 0.0

    metrics = {
        # Height metrics
        'height_mean': masked_mean(pyramid_outputs['height']),
        'height_std': masked_std(pyramid_outputs['height']),
        'height_min': pyramid_outputs['height'][valid_mask].min().item() if valid_mask.any() else 0.0,
        'height_max': pyramid_outputs['height'][valid_mask].max().item() if valid_mask.any() else 0.0,

        # Uncertainty metrics
        'uncertainty_mean': masked_mean(pyramid_outputs['uncertainty']),
        'uncertainty_std': masked_std(pyramid_outputs['uncertainty']),

        # Base stability
        'base_stability_mean': masked_mean(pyramid_outputs['base_stability']),
        'base_variance_mean': masked_mean(pyramid_outputs['base_variance']),

        # Individual forces
        'w_memory_mean': masked_mean(pyramid_outputs['w_memory']),
        'w_pain_mean': masked_mean(pyramid_outputs['w_pain']),
        'w_choice_mean': masked_mean(pyramid_outputs['w_choice']),
        'w_exploration_mean': masked_mean(pyramid_outputs['w_exploration']),

        # Confidence
        'confidence_mean': masked_mean(pyramid_outputs['confidence']),
    }

    return metrics
