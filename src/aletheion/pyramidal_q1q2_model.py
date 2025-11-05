"""Aletheion Pyramidal Transformer with Q1/Q2/Fractal Integration.

This module implements the complete transformer architecture with:
- Pyramidal Epistemic Gates (Q1, Q2, Fractal, Height)
- Optional Epistemic Attention (fractal softmax replacement)
- Temperature modulation based on pyramidal state
- Full VARO loss with Q1/Q2 calibration

References:
    - Pyramidal Epistemology Technical Report (Nov 2025)
    - Aletheion Preprint v4.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from ..model import BaselineTransformer, ModelOutput
from .pyramid_q1q2_fractal import (
    PyramidalEpistemicGatesWithQ1Q2,
    PyramidalVAROLossWithQ1Q2,
    compute_pyramidal_q1q2_metrics
)


@dataclass
class PyramidalQ1Q2ModelOutput(ModelOutput):
    """Extended output with Q1/Q2/Fractal pyramidal information.

    Attributes:
        logits: Raw logits from model
        loss: Optional loss value
        pyramid: Dictionary containing:
            - Q1_mean, Q1_var: Aleatoric uncertainty
            - Q2_mean, Q2_var: Epistemic uncertainty
            - height: Proximity to truth apex (derived from Q1, Q2)
            - fractal_uncertainty: Meta-epistemic uncertainty
            - base_weights: 4 cognitive forces
            - confidence: Overall epistemic confidence
    """

    pyramid: Optional[Dict[str, torch.Tensor]] = None


class AletheionPyramidalQ1Q2Transformer(BaselineTransformer):
    """Aletheion with Pyramidal Q1/Q2/Fractal Architecture.

    This is the complete integration of:
    1. Pyramidal base (Memory, Pain, Choice, Exploration)
    2. Q1 (aleatoric uncertainty) + variance (fractal)
    3. Q2 (epistemic uncertainty) + variance (fractal)
    4. Height derived from Q1, Q2, base_stability
    5. Fractal meta-epistemic layer
    6. Truth apex (constant = 1.0)

    The architecture prevents Q1 collapse (observed in tetrahedral version)
    by making height a derived quantity with Truth apex as natural attractor.

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
        lambda_Q1: Weight for Q1 calibration
        lambda_Q2: Weight for Q2 calibration
        lambda_fractal: Weight for fractal regularization
        lambda_height: Weight for height calibration
        use_multi_head_height: Whether to use multi-head consensus for height
        modulate_temperature: Whether to modulate softmax temperature
        max_temperature_scale: Maximum temperature scale when uncertain
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
        # Pyramidal Q1/Q2 parameters (reduced by 10x to let L_CE dominate)
        lambda_base: float = 0.001,
        lambda_Q1: float = 0.0015,
        lambda_Q2: float = 0.002,
        lambda_fractal: float = 0.0005,
        lambda_height: float = 0.002,
        use_multi_head_height: bool = False,
        modulate_temperature: bool = True,
        max_temperature_scale: float = 2.0
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

        # Store pyramidal parameters
        self.lambda_base = lambda_base
        self.lambda_Q1 = lambda_Q1
        self.lambda_Q2 = lambda_Q2
        self.lambda_fractal = lambda_fractal
        self.lambda_height = lambda_height
        self.modulate_temperature = modulate_temperature
        self.max_temperature_scale = max_temperature_scale

        # Add pyramidal Q1/Q2/Fractal gates
        self.pyramid_gates = PyramidalEpistemicGatesWithQ1Q2(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            use_multi_head_height=use_multi_head_height
        )

        # Pyramidal VARO loss
        self.pyramid_loss_fn = PyramidalVAROLossWithQ1Q2(
            lambda_base=lambda_base,
            lambda_Q1=lambda_Q1,
            lambda_Q2=lambda_Q2,
            lambda_fractal=lambda_fractal,
            lambda_height=lambda_height,
            ignore_index=-100
        )

        print(f"🔻 Pyramidal Q1/Q2/Fractal Architecture initialized")
        print(f"   - λ_base: {lambda_base}")
        print(f"   - λ_Q1: {lambda_Q1}")
        print(f"   - λ_Q2: {lambda_Q2}")
        print(f"   - λ_fractal: {lambda_fractal}")
        print(f"   - λ_height: {lambda_height}")
        print(f"   - Multi-head height: {use_multi_head_height}")
        print(f"   - Temperature modulation: {modulate_temperature}")
        print(f"   - Pyramidal parameters: {self._count_pyramidal_params():,}")

    def _count_pyramidal_params(self) -> int:
        """Count parameters in pyramidal gates."""
        return sum(p.numel() for p in self.pyramid_gates.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        return_pyramid_state: bool = True
    ) -> PyramidalQ1Q2ModelOutput | Dict[str, torch.Tensor]:
        """Forward pass with pyramidal Q1/Q2/Fractal computation.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            labels: Optional target labels for loss computation
            return_dict: Whether to return ModelOutput object
            return_pyramid_state: Whether to compute pyramidal state

        Returns:
            PyramidalQ1Q2ModelOutput with logits, loss, and pyramid state
        """
        batch_size, seq_len = input_ids.shape

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}"
            )

        # Standard transformer forward pass
        token_emb = self.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)

        hidden_states = self.emb_dropout(token_emb + pos_emb)

        # Pass through transformer blocks
        for block in self.blocks:
            hidden_states = block(hidden_states)

        # Final layer norm
        hidden_states = self.ln_final(hidden_states)

        # === PYRAMIDAL Q1/Q2/FRACTAL COMPUTATION ===
        pyramid_outputs = None
        if return_pyramid_state:
            pyramid_outputs = self.pyramid_gates(hidden_states)

        # Compute logits with optional temperature modulation
        if self.modulate_temperature and pyramid_outputs is not None:
            # Modulate temperature based on height and fractal uncertainty
            height = pyramid_outputs['height']
            fractal = pyramid_outputs['fractal_uncertainty']

            # Temperature increases with uncertainty AND meta-uncertainty
            temperature = 1.0 + (1.0 - height + fractal) * (self.max_temperature_scale - 1.0)

            # Apply to logits
            logits = self.lm_head(hidden_states)
            logits = logits / temperature
        else:
            logits = self.lm_head(hidden_states)

        # Compute loss
        loss = None
        if labels is not None:
            if pyramid_outputs is not None:
                # Use pyramidal VARO loss
                loss_dict = self.pyramid_loss_fn(logits, labels, pyramid_outputs)
                loss = loss_dict['loss']
            else:
                # Fallback to standard cross-entropy
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100
                )

        if return_dict:
            return PyramidalQ1Q2ModelOutput(
                logits=logits,
                loss=loss,
                pyramid=pyramid_outputs
            )

        return {
            'logits': logits,
            'loss': loss,
            'pyramid': pyramid_outputs
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
        use_pyramid: bool = True,
        Q1_threshold: float = 0.5,
        Q2_threshold: float = 0.5
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Generate tokens with Q1/Q2-aware decoding.

        Adjusts sampling based on:
        - Q1 (aleatoric): If high, increase randomness
        - Q2 (epistemic): If high, increase exploration

        Args:
            input_ids: Input token IDs [batch, seq_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Base sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            do_sample: Whether to sample or greedy
            use_pyramid: Whether to use pyramidal state
            Q1_threshold: Threshold for Q1-based temperature adjustment
            Q2_threshold: Threshold for Q2-based exploration

        Returns:
            Tuple of:
                - generated: Generated token IDs
                - pyramid_history: Q1, Q2, height, fractal over generation
        """
        self.eval()

        generated = input_ids
        Q1_history = []
        Q2_history = []
        height_history = []
        fractal_history = []

        for _ in range(max_new_tokens):
            # Truncate to max_seq_len
            idx_cond = (
                generated
                if generated.size(1) <= self.max_seq_len
                else generated[:, -self.max_seq_len:]
            )

            # Forward pass
            outputs = self(idx_cond, return_dict=True, return_pyramid_state=use_pyramid)

            # Get logits for last position
            logits = outputs.logits[:, -1, :]  # [batch, vocab_size]

            # Get pyramidal state
            if use_pyramid and outputs.pyramid is not None:
                Q1 = outputs.pyramid['Q1_mean'][:, -1, :].squeeze(-1)  # [batch]
                Q2 = outputs.pyramid['Q2_mean'][:, -1, :].squeeze(-1)
                height = outputs.pyramid['height'][:, -1, :].squeeze(-1)
                fractal = outputs.pyramid['fractal_uncertainty'][:, -1, :].squeeze(-1)

                Q1_history.append(Q1)
                Q2_history.append(Q2)
                height_history.append(height)
                fractal_history.append(fractal)

                # Adjust temperature based on Q1, Q2
                # High Q1 (aleatoric) → increase randomness
                # High Q2 (epistemic) → increase exploration
                adjusted_temp = temperature * (1.0 + Q1 + Q2)

                # Apply per-sample temperature
                for b in range(logits.size(0)):
                    logits[b] = logits[b] / adjusted_temp[b]
            else:
                logits = logits / temperature

            # Apply top-k
            if top_k is not None:
                top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_values[:, [-1]]] = float('-inf')

            # Apply top-p
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
                logits[indices_to_remove] = float('-inf')

            # Sample
            probs = nn.functional.softmax(logits, dim=-1)
            if do_sample:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)

            # Append
            generated = torch.cat([generated, next_token], dim=1)

        # Stack history
        pyramid_history = {}
        if Q1_history:
            pyramid_history['Q1_mean'] = torch.stack(Q1_history, dim=1)
            pyramid_history['Q2_mean'] = torch.stack(Q2_history, dim=1)
            pyramid_history['heights'] = torch.stack(height_history, dim=1)
            pyramid_history['fractal_uncertainty'] = torch.stack(fractal_history, dim=1)

        return generated, pyramid_history

    def get_pyramidal_stats(
        self,
        pyramid_outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute statistics about pyramidal Q1/Q2 state.

        Args:
            pyramid_outputs: Dictionary from pyramid_gates

        Returns:
            Dictionary with comprehensive metrics
        """
        return compute_pyramidal_q1q2_metrics(pyramid_outputs)

    def save_pretrained(self, save_dir: str) -> None:
        """Save model checkpoint.

        Args:
            save_dir: Directory to save checkpoint
        """
        import os
        os.makedirs(save_dir, exist_ok=True)

        # Get config
        first_block = self.blocks[0]
        n_heads = first_block.attn.n_heads
        d_ff = first_block.ffn.fc1.out_features

        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': {
                'vocab_size': self.token_embedding.num_embeddings,
                'd_model': self.token_embedding.embedding_dim,
                'n_layers': len(self.blocks),
                'n_heads': n_heads,
                'd_ff': d_ff,
                'max_seq_len': self.pos_embedding.num_embeddings,
                'dropout': 0.1,
                'tie_weights': self.token_embedding.weight.data_ptr() == self.lm_head.weight.data_ptr(),
                'use_flash_attention': False,
                'lambda_base': self.lambda_base,
                'lambda_Q1': self.lambda_Q1,
                'lambda_Q2': self.lambda_Q2,
                'lambda_fractal': self.lambda_fractal,
                'lambda_height': self.lambda_height,
                'use_multi_head_height': self.pyramid_gates.use_multi_head_height,
                'modulate_temperature': self.modulate_temperature,
                'max_temperature_scale': self.max_temperature_scale
            }
        }

        torch.save(checkpoint, os.path.join(save_dir, 'pytorch_model.bin'))
        print(f"✅ Pyramidal Q1/Q2 model saved to {save_dir}")

    @classmethod
    def load_pretrained(
        cls,
        load_dir: str,
        device: str = 'cpu'
    ) -> 'AletheionPyramidalQ1Q2Transformer':
        """Load model checkpoint.

        Args:
            load_dir: Directory containing checkpoint
            device: Device to load on

        Returns:
            Loaded model
        """
        import os
        checkpoint_path = os.path.join(load_dir, 'pytorch_model.bin')

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint['config']

        # Instantiate model
        model = cls(**config)

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        print(f"✅ Pyramidal Q1/Q2 model loaded from {load_dir}")
        return model
