# PROPRIETARY AND CONFIDENTIAL
# Copyright (c) 2024-2025 AletheionAGI
# Unauthorized copying prohibited
#
# LEVEL 3 PROPRIETARY ARCHITECTURE
# This file contains proprietary pyramidal epistemology implementations
# that are confidential and not covered by AGPL-3.0

"""Aletheion transformer with Pyramidal Epistemology.

This module implements the Pyramidal Epistemology architecture, which extends
the baseline transformer with a 5-vertex geometric structure:
    - Base: Memory, Pain, Choice, Exploration (4 vertices)
    - Apex: Truth = 1.0 (constant attractor)

Architecture changes:
    - Adds pyramidal epistemic gates at the output layer
    - Modulates temperature based on height (proximity to truth)
    - Returns height, base forces, and derived metrics

References:
    "Pyramidal Epistemology: Climbing Toward Truth in Neural Networks"
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..model import BaselineTransformer, ModelOutput
from .pyramid import (
    PyramidalEpistemicGates,
    PyramidalTemperatureModulator,
    compute_pyramidal_metrics,
)


@dataclass
class PyramidalModelOutput(ModelOutput):
    """Extended output container with pyramidal epistemic information.

    Attributes:
        logits: Raw logits from the model
        loss: Optional loss value
        pyramid: Dictionary containing pyramidal outputs:
            - base_weights: [batch, seq_len, 4] - (memory, pain, choice, exploration)
            - height: [batch, seq_len, 1] - proximity to truth
            - uncertainty: [batch, seq_len, 1] - 1 - height
            - confidence: [batch, seq_len, 1] - height × base_stability
            - base_stability: [batch, seq_len, 1] - 1 - base_variance
            - w_memory, w_pain, w_choice, w_exploration: Individual force weights
    """

    pyramid: dict[str, torch.Tensor] | None = None


class AletheionPyramidalTransformer(BaselineTransformer):
    """Aletheion with Pyramidal Epistemic Architecture.

    Extends BaselineTransformer by adding pyramidal epistemic gates that represent
    the AI's position in a 5-vertex pyramid:
        - Base forces: Memory, Pain, Choice, Exploration
        - Height: Proximity to Truth apex

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
        lambda_base: Weight for base stability regularization
        lambda_height: Weight for height calibration
        height_method: Method for computing target height ('error_based', 'entropy_based', 'loss_based')
        use_multi_head_height: Whether to use multi-head consensus for height
        modulate_temperature: Whether to modulate softmax temperature based on height
        max_temperature_scale: Maximum temperature scale when height=0 (uncertain)
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
        # Pyramidal-specific parameters
        lambda_base: float = 0.01,
        lambda_height: float = 0.02,
        height_method: str = "error_based",
        use_multi_head_height: bool = False,
        modulate_temperature: bool = True,
        max_temperature_scale: float = 2.0,
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
            use_flash_attention=use_flash_attention,
        )

        # Store pyramidal parameters
        self.lambda_base = lambda_base
        self.lambda_height = lambda_height
        self.height_method = height_method
        self.modulate_temperature = modulate_temperature

        # Add pyramidal epistemic gates
        self.pyramid_gates = PyramidalEpistemicGates(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            use_multi_head_height=use_multi_head_height,
        )

        # Optional temperature modulator
        if modulate_temperature:
            self.temp_modulator = PyramidalTemperatureModulator(
                base_temperature=1.0, max_temperature_scale=max_temperature_scale
            )

        print("🔻 Pyramidal Epistemology initialized")
        print(f"   - λ_base: {lambda_base}")
        print(f"   - λ_height: {lambda_height}")
        print(f"   - Height method: {height_method}")
        print(f"   - Multi-head height: {use_multi_head_height}")
        print(f"   - Temperature modulation: {modulate_temperature}")
        print(f"   - Pyramidal parameters: {self._count_pyramidal_params():,}")

    def _count_pyramidal_params(self) -> int:
        """Count parameters in pyramidal gates."""
        return sum(p.numel() for p in self.pyramid_gates.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        return_dict: bool = True,
        return_pyramid_state: bool = True,
    ) -> PyramidalModelOutput | dict[str, torch.Tensor]:
        """Forward pass with pyramidal epistemic computation.

        Args:
            input_ids: Input token IDs of shape (batch, seq_len)
            labels: Optional target labels for loss computation
            return_dict: Whether to return ModelOutput object
            return_pyramid_state: Whether to compute and return pyramidal state

        Returns:
            PyramidalModelOutput containing logits, loss, and pyramidal information
        """
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}")

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

        # === PYRAMIDAL EPISTEMIC COMPUTATION ===
        pyramid_outputs = None
        if return_pyramid_state:
            pyramid_outputs = self.pyramid_gates(hidden_states)

        # Compute logits with optional temperature modulation
        if self.modulate_temperature and pyramid_outputs is not None:
            # Modulate temperature based on height
            height = pyramid_outputs["height"]
            logits = self.lm_head(hidden_states)
            logits = self.temp_modulator(logits, height)
        else:
            # Standard logits
            logits = self.lm_head(hidden_states)

        # Compute loss (cross-entropy only; pyramidal loss applied in training loop)
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Standard cross-entropy
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if return_dict:
            return PyramidalModelOutput(logits=logits, loss=loss, pyramid=pyramid_outputs)

        return {"logits": logits, "loss": loss, "pyramid": pyramid_outputs}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        do_sample: bool = True,
        use_pyramid: bool = True,
        height_threshold: float = 0.5,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Generate tokens with pyramidal epistemic-aware decoding.

        Args:
            input_ids: Input token IDs of shape (batch, seq_len)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Base sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            do_sample: Whether to sample or use greedy decoding
            use_pyramid: Whether to use pyramidal epistemic state for generation
            height_threshold: If height < threshold, increase sampling temperature

        Returns:
            Tuple of:
                - generated: Generated token IDs of shape (batch, seq_len + max_new_tokens)
                - pyramid_history: Dictionary with pyramidal metrics over generation
        """
        self.eval()

        generated = input_ids
        heights = []
        uncertainties = []
        base_stabilities = []

        for _ in range(max_new_tokens):
            # Truncate to max_seq_len
            idx_cond = (
                generated
                if generated.size(1) <= self.max_seq_len
                else generated[:, -self.max_seq_len :]
            )

            # Forward pass
            outputs = self(idx_cond, return_dict=True, return_pyramid_state=use_pyramid)

            # Get logits for last position
            logits = outputs.logits[:, -1, :]  # (batch, vocab_size)

            # Get pyramidal state for last position
            if use_pyramid and outputs.pyramid is not None:
                height = outputs.pyramid["height"][:, -1, :].squeeze(-1)  # (batch,)
                uncertainty = outputs.pyramid["uncertainty"][:, -1, :].squeeze(-1)
                base_stability = outputs.pyramid["base_stability"][:, -1, :].squeeze(-1)

                heights.append(height)
                uncertainties.append(uncertainty)
                base_stabilities.append(base_stability)

                # Adjust temperature based on height
                # Low height → higher temperature (more exploration)
                adjusted_temp = torch.where(
                    height < height_threshold,
                    temperature * (1.0 + (1.0 - height)),
                    torch.full_like(height, temperature),
                )

                # Apply adjusted temperature per sample
                for b in range(logits.size(0)):
                    logits[b] = logits[b] / adjusted_temp[b]
            else:
                logits = logits / temperature

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

        # Stack pyramidal history
        pyramid_history = {}
        if heights:
            pyramid_history["heights"] = torch.stack(heights, dim=1)  # (batch, max_new_tokens)
            pyramid_history["uncertainties"] = torch.stack(uncertainties, dim=1)
            pyramid_history["base_stabilities"] = torch.stack(base_stabilities, dim=1)

        return generated, pyramid_history

    def get_pyramidal_stats(self, pyramid_outputs: dict[str, torch.Tensor]) -> dict[str, float]:
        """Compute statistics about pyramidal state.

        Args:
            pyramid_outputs: Dictionary from pyramid_gates

        Returns:
            Dictionary with pyramidal metrics
        """
        return compute_pyramidal_metrics(pyramid_outputs)

    def save_pretrained(self, save_dir: str) -> None:
        """Save model checkpoint including pyramidal gates.

        Args:
            save_dir: Directory to save checkpoint
        """
        import os

        os.makedirs(save_dir, exist_ok=True)

        # Get config from first attention block
        first_block = self.blocks[0]
        n_heads = first_block.attn.n_heads
        d_ff = first_block.ffn.fc1.out_features

        # Save full state dict (includes all parameters)
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": {
                "vocab_size": self.token_embedding.num_embeddings,
                "d_model": self.token_embedding.embedding_dim,
                "n_layers": len(self.blocks),
                "n_heads": n_heads,
                "d_ff": d_ff,
                "max_seq_len": self.pos_embedding.num_embeddings,
                "dropout": 0.1,
                "tie_weights": self.token_embedding.weight.data_ptr()
                == self.lm_head.weight.data_ptr(),
                "use_flash_attention": False,
                "lambda_base": self.lambda_base,
                "lambda_height": self.lambda_height,
                "height_method": self.height_method,
                "use_multi_head_height": self.pyramid_gates.use_multi_head_height,
                "modulate_temperature": self.modulate_temperature,
            },
        }

        torch.save(checkpoint, os.path.join(save_dir, "pytorch_model.bin"))
        print(f"✅ Pyramidal model saved to {save_dir}")

    @classmethod
    def load_pretrained(cls, load_dir: str, device: str = "cpu") -> AletheionPyramidalTransformer:
        """Load model checkpoint including pyramidal gates.

        Args:
            load_dir: Directory containing checkpoint
            device: Device to load model on

        Returns:
            Loaded AletheionPyramidalTransformer instance
        """
        import os

        checkpoint_path = os.path.join(load_dir, "pytorch_model.bin")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint["config"]

        # Instantiate model with saved config
        model = cls(**config)

        # Load weights
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        print(f"✅ Pyramidal model loaded from {load_dir}")
        return model
