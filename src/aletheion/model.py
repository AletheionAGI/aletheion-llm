"""Aletheion transformer with epistemic uncertainty quantification.

This module implements the Aletheion Level 1 architecture described in Section 5.1 of the paper.
It extends the baseline transformer by adding epistemic gates at the output layer.

Architecture changes:
    - Adds Q₁ gate (local uncertainty) at output
    - Adds Q₂ gate (cross-context consensus) at output
    - Replaces final softmax with epistemic_softmax
    - Returns uncertainty alongside logits

References:
    Aletheion paper, Section 5.1 (Level 1: Output-Only)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..model import BaselineTransformer, ModelOutput
from .gates import LocalUncertaintyGate, CrossContextGate, epistemic_softmax


@dataclass
class AletheionModelOutput(ModelOutput):
    """Extended output container with uncertainty information.

    Attributes:
        logits: Raw logits from the model
        loss: Optional loss value
        uncertainty: Epistemic uncertainty scalar in [0, 1]
        q1: Local uncertainty gate output
        q2: Cross-context consensus gate output
        probs_gated: Gated probability distribution (epistemic softmax output)
    """

    uncertainty: Optional[torch.Tensor] = None
    q1: Optional[torch.Tensor] = None
    q2: Optional[torch.Tensor] = None
    probs_gated: Optional[torch.Tensor] = None


class AletheionTransformer(BaselineTransformer):
    """Aletheion Level 1: Transformer with output-level epistemic gating.

    Extends BaselineTransformer by adding uncertainty quantification at the output layer.
    All other components (attention, FFN) remain unchanged for fair comparison.

    Args:
        vocab_size: Size of vocabulary
        d_model: Model hidden dimension
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        d_ff: Feedforward dimension
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
        tie_weights: Whether to tie input/output embeddings
        use_flash_attention: Whether to use FlashAttention
        q1_threshold: Confidence threshold for Q₁ gate
        q2_threshold: Confidence threshold for Q₂ gate
        base_temperature: Base temperature for epistemic softmax
        n_consensus_heads: Number of heads for Q₂ cross-attention
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        tie_weights: bool = True,
        use_flash_attention: bool = False,
        # Aletheion-specific parameters
        q1_threshold: float = 0.7,
        q2_threshold: float = 0.7,
        base_temperature: float = 1.0,
        n_consensus_heads: int = 4
    ) -> None:
        # Initialize baseline transformer
        super().__init__(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            tie_weights=tie_weights,
            use_flash_attention=use_flash_attention
        )

        # Store epistemic parameters
        self.q1_threshold = q1_threshold
        self.q2_threshold = q2_threshold
        self.base_temperature = base_temperature
        self.confidence_threshold = min(q1_threshold, q2_threshold)

        # Add epistemic gates
        self.q1_gate = LocalUncertaintyGate(d_model=d_model, dropout=dropout)
        self.q2_gate = CrossContextGate(
            d_model=d_model,
            n_heads=n_consensus_heads,
            dropout=dropout
        )

        print(f"🔮 Aletheion Level 1 initialized")
        print(f"   - Q₁ threshold: {q1_threshold}")
        print(f"   - Q₂ threshold: {q2_threshold}")
        print(f"   - Base temperature: {base_temperature}")
        print(f"   - Epistemic parameters: {self._count_epistemic_params():,}")

    def _count_epistemic_params(self) -> int:
        """Count parameters in epistemic gates."""
        q1_params = sum(p.numel() for p in self.q1_gate.parameters())
        q2_params = sum(p.numel() for p in self.q2_gate.parameters())
        return q1_params + q2_params

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        return_uncertainty: bool = True
    ) -> AletheionModelOutput | Dict[str, torch.Tensor]:
        """Forward pass with epistemic uncertainty quantification.

        Args:
            input_ids: Input token IDs of shape (batch, seq_len)
            labels: Optional target labels for loss computation
            return_dict: Whether to return ModelOutput object
            return_uncertainty: Whether to compute and return uncertainty

        Returns:
            AletheionModelOutput containing logits, loss, and uncertainty information
        """
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}"
            )

        # Standard transformer forward pass (same as baseline)
        token_emb = self.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)

        hidden_states = self.emb_dropout(token_emb + pos_emb)

        # Pass through transformer blocks
        for block in self.blocks:
            hidden_states = block(hidden_states)

        # Final layer norm
        hidden_states = self.ln_final(hidden_states)

        # Compute logits
        logits = self.lm_head(hidden_states)  # (batch, seq_len, vocab_size)

        # === EPISTEMIC SOFTMAX (Level 1 modification) ===
        uncertainty = None
        q1 = None
        q2 = None
        probs_gated = None

        if return_uncertainty:
            # Use final hidden states as context for gates
            context = hidden_states  # (batch, seq_len, d_model)

            # Apply epistemic softmax
            probs_gated, uncertainty = epistemic_softmax(
                logits=logits,
                context=context,
                q1_gate=self.q1_gate,
                q2_gate=self.q2_gate,
                base_temperature=self.base_temperature,
                confidence_threshold=self.confidence_threshold
            )

            # Store individual gate outputs for analysis
            q1 = self.q1_gate(context)
            q2 = self.q2_gate(context)

        # Compute loss (same as baseline, using raw logits)
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Standard cross-entropy (VARO loss is applied in training loop)
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )

        if return_dict:
            return AletheionModelOutput(
                logits=logits,
                loss=loss,
                uncertainty=uncertainty,
                q1=q1,
                q2=q2,
                probs_gated=probs_gated
            )

        return {
            "logits": logits,
            "loss": loss,
            "uncertainty": uncertainty,
            "q1": q1,
            "q2": q2,
            "probs_gated": probs_gated
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        use_epistemic: bool = True,
        uncertainty_threshold: float = 0.8
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate tokens with epistemic uncertainty-aware decoding.

        Args:
            input_ids: Input token IDs of shape (batch, seq_len)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (applied after epistemic adjustment)
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            do_sample: Whether to sample or use greedy decoding
            use_epistemic: Whether to use epistemic softmax for generation
            uncertainty_threshold: If uncertainty > threshold, increase sampling temperature

        Returns:
            Tuple of:
                - generated: Generated token IDs of shape (batch, seq_len + max_new_tokens)
                - uncertainties: Uncertainty values for each generated token (batch, max_new_tokens)
        """
        self.eval()

        generated = input_ids
        uncertainties = []

        for _ in range(max_new_tokens):
            # Truncate to max_seq_len
            idx_cond = (
                generated
                if generated.size(1) <= self.max_seq_len
                else generated[:, -self.max_seq_len :]
            )

            # Forward pass
            outputs = self(idx_cond, return_dict=True, return_uncertainty=use_epistemic)

            # Get logits for last position
            logits = outputs.logits[:, -1, :]  # (batch, vocab_size)

            # Get uncertainty for last position
            if use_epistemic and outputs.uncertainty is not None:
                uncertainty = outputs.uncertainty[:, -1, :].squeeze(-1)  # (batch,)
                uncertainties.append(uncertainty)

                # Adjust temperature based on uncertainty
                # High uncertainty → higher temperature (more exploration)
                adjusted_temp = torch.where(
                    uncertainty > uncertainty_threshold,
                    temperature * (1.0 + uncertainty),
                    torch.full_like(uncertainty, temperature)
                )

                # Apply adjusted temperature per sample
                for b in range(logits.size(0)):
                    logits[b] = logits[b] / adjusted_temp[b]
            else:
                logits = logits / temperature
                uncertainties.append(torch.zeros(logits.size(0), device=logits.device))

            # Apply top-k filtering
            if top_k is not None:
                top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_values[:, [-1]]] = float("-inf")

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    nn.functional.softmax(sorted_logits, dim=-1), dim=-1
                )

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            # Sample or take argmax
            probs = nn.functional.softmax(logits, dim=-1)
            if do_sample:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)

        # Stack uncertainties
        uncertainties_tensor = torch.stack(uncertainties, dim=1)  # (batch, max_new_tokens)

        return generated, uncertainties_tensor

    def get_uncertainty_stats(self, uncertainty: torch.Tensor) -> Dict[str, float]:
        """Compute statistics about uncertainty values.

        Args:
            uncertainty: Uncertainty tensor of shape (batch, seq_len, 1)

        Returns:
            Dictionary with mean, std, min, max uncertainty
        """
        u_flat = uncertainty.flatten()
        return {
            "uncertainty_mean": u_flat.mean().item(),
            "uncertainty_std": u_flat.std().item(),
            "uncertainty_min": u_flat.min().item(),
            "uncertainty_max": u_flat.max().item()
        }
