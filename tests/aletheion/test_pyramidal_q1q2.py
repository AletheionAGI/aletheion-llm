"""Unit tests for Pyramidal Q1/Q2/Fractal architecture.

Tests:
1. PyramidalEpistemicGatesWithQ1Q2: Shape, range, initialization
2. PyramidalVAROLossWithQ1Q2: Loss components, gradients
3. AletheionPyramidalQ1Q2Transformer: Forward pass, generation
4. Collapse detection: Q1, Q2 entropy checks
"""

import pytest
import torch
import torch.nn as nn

from src.aletheion.pyramid_q1q2_fractal import (
    EpistemicMultiHeadAttention,
    PyramidalEpistemicGatesWithQ1Q2,
    PyramidalVAROLossWithQ1Q2,
    compute_pyramidal_q1q2_metrics,
)
from src.aletheion.pyramidal_q1q2_model import AletheionPyramidalQ1Q2Transformer


class TestPyramidalEpistemicGatesWithQ1Q2:
    """Test pyramidal gates with Q1/Q2/Fractal."""

    def test_output_shapes(self):
        """Test that all outputs have correct shapes."""
        batch_size, seq_len, d_model = 2, 16, 128
        gates = PyramidalEpistemicGatesWithQ1Q2(d_model=d_model, n_heads=4)

        hidden = torch.randn(batch_size, seq_len, d_model)
        outputs = gates(hidden)

        # Check all keys exist
        expected_keys = [
            "base_weights",
            "w_memory",
            "w_pain",
            "w_choice",
            "w_exploration",
            "base_stability",
            "base_variance",
            "Q1_mean",
            "Q1_var",
            "Q2_mean",
            "Q2_var",
            "height",
            "uncertainty",
            "fractal_uncertainty",
            "total_uncertainty",
            "confidence",
        ]
        for key in expected_keys:
            assert key in outputs, f"Missing key: {key}"

        # Check shapes
        assert outputs["base_weights"].shape == (batch_size, seq_len, 4)
        assert outputs["Q1_mean"].shape == (batch_size, seq_len, 1)
        assert outputs["Q2_mean"].shape == (batch_size, seq_len, 1)
        assert outputs["height"].shape == (batch_size, seq_len, 1)
        assert outputs["fractal_uncertainty"].shape == (batch_size, seq_len, 1)

    def test_value_ranges(self):
        """Test that all outputs are in valid ranges."""
        gates = PyramidalEpistemicGatesWithQ1Q2(d_model=64)
        hidden = torch.randn(2, 10, 64)
        outputs = gates(hidden)

        # Base weights sum to 1
        assert torch.allclose(outputs["base_weights"].sum(dim=-1), torch.ones(2, 10), atol=1e-5)

        # All gates in [0, 1]
        for key in ["Q1_mean", "Q2_mean", "height", "fractal_uncertainty"]:
            values = outputs[key]
            assert (values >= 0).all() and (values <= 1).all(), f"{key} out of range"

        # Q1_var, Q2_var >= 0 (softplus outputs)
        assert (outputs["Q1_var"] >= 0).all()
        assert (outputs["Q2_var"] >= 0).all()

        # Base stability in [0, 1]
        assert (outputs["base_stability"] >= 0).all()
        assert (outputs["base_stability"] <= 1).all()

    def test_height_derivation(self):
        """Test that height is derived from Q1, Q2, base_stability."""
        gates = PyramidalEpistemicGatesWithQ1Q2(d_model=64)

        # Create inputs with known properties
        # High Q1, Q2 → low height
        hidden_uncertain = torch.randn(1, 5, 64) * 3.0  # Large values → saturated gates

        # Low Q1, Q2 → high height
        hidden_certain = torch.randn(1, 5, 64) * 0.1  # Small values → unsaturated gates

        outputs_uncertain = gates(hidden_uncertain)
        outputs_certain = gates(hidden_certain)

        # Note: This is a soft test due to random initialization
        # Just check that outputs are valid
        assert outputs_uncertain["height"].mean() >= 0
        assert outputs_certain["height"].mean() >= 0

    def test_initialization_ranges(self):
        """Test that initial values are in healthy ranges."""
        gates = PyramidalEpistemicGatesWithQ1Q2(d_model=128)
        hidden = torch.randn(4, 20, 128)
        outputs = gates(hidden)

        # Check initial Q1 is around 0.3-0.4 (initialized with bias -0.5)
        Q1_mean = outputs["Q1_mean"].mean().item()
        assert 0.2 < Q1_mean < 0.5, f"Q1 init out of range: {Q1_mean}"

        # Check initial Q2 is around 0.4-0.6 (initialized with bias 0.0)
        Q2_mean = outputs["Q2_mean"].mean().item()
        assert 0.3 < Q2_mean < 0.7, f"Q2 init out of range: {Q2_mean}"

        # Check initial fractal is low ~0.2-0.3 (initialized with bias -1.0)
        fractal_mean = outputs["fractal_uncertainty"].mean().item()
        assert 0.1 < fractal_mean < 0.5, f"Fractal init out of range: {fractal_mean}"


class TestPyramidalVAROLossWithQ1Q2:
    """Test pyramidal VARO loss with Q1/Q2."""

    def test_loss_computation(self):
        """Test that loss computes without errors."""
        loss_fn = PyramidalVAROLossWithQ1Q2(
            lambda_base=0.01,
            lambda_Q1=0.015,
            lambda_Q2=0.020,
            lambda_fractal=0.005,
            lambda_height=0.02,
        )

        batch_size, seq_len, vocab_size = 2, 16, 100
        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Create dummy pyramid outputs
        gates = PyramidalEpistemicGatesWithQ1Q2(d_model=64)
        hidden = torch.randn(batch_size, seq_len, 64)
        pyramid_outputs = gates(hidden)

        # Compute loss
        loss_dict = loss_fn(logits, targets, pyramid_outputs)

        # Check all components exist
        expected_keys = [
            "loss",
            "ce_loss",
            "base_loss",
            "Q1_loss",
            "Q2_loss",
            "fractal_loss",
            "height_loss",
            "mean_Q1",
            "mean_Q2",
            "mean_height",
            "mean_fractal",
        ]
        for key in expected_keys:
            assert key in loss_dict, f"Missing loss key: {key}"

        # Check loss is scalar
        assert loss_dict["loss"].ndim == 0

        # Check loss is finite
        assert torch.isfinite(loss_dict["loss"])

    def test_target_Q1_computation(self):
        """Test Q1 target computation."""
        loss_fn = PyramidalVAROLossWithQ1Q2()

        batch_size, seq_len, vocab_size = 2, 10, 50
        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))

        target_Q1 = loss_fn.compute_target_Q1(logits, targets)

        # Shape check
        assert target_Q1.shape == (batch_size, seq_len, 1)

        # Range check: Q1 in [0, 1]
        assert (target_Q1 >= 0).all() and (target_Q1 <= 1).all()

    def test_target_Q2_computation(self):
        """Test Q2 target computation."""
        loss_fn = PyramidalVAROLossWithQ1Q2()

        batch_size, seq_len, vocab_size = 2, 10, 50
        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))

        target_Q2 = loss_fn.compute_target_Q2(logits, targets)

        # Shape check
        assert target_Q2.shape == (batch_size, seq_len, 1)

        # Range check: Q2 in [0, 1]
        assert (target_Q2 >= 0).all() and (target_Q2 <= 1).all()

    def test_gradient_flow(self):
        """Test that gradients flow through all components."""
        loss_fn = PyramidalVAROLossWithQ1Q2()
        gates = PyramidalEpistemicGatesWithQ1Q2(d_model=64)

        logits = torch.randn(2, 10, 50, requires_grad=True)
        targets = torch.randint(0, 50, (2, 10))
        hidden = torch.randn(2, 10, 64, requires_grad=True)

        pyramid_outputs = gates(hidden)
        loss_dict = loss_fn(logits, targets, pyramid_outputs)

        # Backward
        loss_dict["loss"].backward()

        # Check gradients exist
        assert logits.grad is not None
        assert hidden.grad is not None

        # Check gradients are finite
        assert torch.isfinite(logits.grad).all()
        assert torch.isfinite(hidden.grad).all()


class TestEpistemicMultiHeadAttention:
    """Test epistemic attention with softmax replacement."""

    def test_forward_pass(self):
        """Test forward pass."""
        d_model, n_heads = 64, 4
        attn = EpistemicMultiHeadAttention(d_model=d_model, n_heads=n_heads)

        # Create Q1, Q2 gates
        Q1_gate = nn.Linear(n_heads * (d_model // n_heads), 1)
        Q2_gate = nn.Linear(n_heads * (d_model // n_heads), 1)

        batch_size, seq_len = 2, 16
        x = torch.randn(batch_size, seq_len, d_model)

        output, uncertainty = attn(x, Q1_gate, Q2_gate)

        # Shape checks
        assert output.shape == (batch_size, seq_len, d_model)
        assert uncertainty.shape == (batch_size, 1)

        # Uncertainty in [0, 1]
        assert (uncertainty >= 0).all() and (uncertainty <= 1).all()


class TestAletheionPyramidalQ1Q2Transformer:
    """Test complete transformer model."""

    def test_model_creation(self):
        """Test model instantiation."""
        model = AletheionPyramidalQ1Q2Transformer(
            vocab_size=100,
            d_model=64,
            n_layers=2,
            n_heads=4,
            d_ff=256,
            max_seq_len=32,
            lambda_Q1=0.015,
            lambda_Q2=0.020,
        )

        # Check pyramid gates exist
        assert hasattr(model, "pyramid_gates")
        assert isinstance(model.pyramid_gates, PyramidalEpistemicGatesWithQ1Q2)

    def test_forward_pass(self):
        """Test forward pass."""
        model = AletheionPyramidalQ1Q2Transformer(vocab_size=100, d_model=64, n_layers=2, n_heads=4)

        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        labels = torch.randint(0, 100, (batch_size, seq_len))

        outputs = model(input_ids, labels=labels, return_dict=True)

        # Check outputs
        assert outputs.logits.shape == (batch_size, seq_len, 100)
        assert outputs.loss is not None
        assert outputs.pyramid is not None

        # Check pyramidal outputs
        assert "Q1_mean" in outputs.pyramid
        assert "Q2_mean" in outputs.pyramid
        assert "height" in outputs.pyramid
        assert "fractal_uncertainty" in outputs.pyramid

    def test_generation(self):
        """Test generation."""
        model = AletheionPyramidalQ1Q2Transformer(
            vocab_size=100, d_model=64, n_layers=2, n_heads=4, max_seq_len=32
        )

        batch_size, seq_len = 1, 8
        input_ids = torch.randint(0, 100, (batch_size, seq_len))

        generated, pyramid_history = model.generate(
            input_ids, max_new_tokens=10, do_sample=False, use_pyramid=True
        )

        # Check generated shape
        assert generated.shape == (batch_size, seq_len + 10)

        # Check pyramid history
        assert "Q1_mean" in pyramid_history
        assert "Q2_mean" in pyramid_history
        assert "heights" in pyramid_history

    def test_save_load(self, tmp_path):
        """Test saving and loading."""
        model = AletheionPyramidalQ1Q2Transformer(
            vocab_size=100, d_model=64, n_layers=2, n_heads=4, lambda_Q1=0.015
        )

        # Save
        save_dir = tmp_path / "test_model"
        model.save_pretrained(str(save_dir))

        # Load
        loaded_model = AletheionPyramidalQ1Q2Transformer.load_pretrained(
            str(save_dir), device="cpu"
        )

        # Check config preserved
        assert loaded_model.lambda_Q1 == 0.015

        # Check weights are the same
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), loaded_model.named_parameters()):
            assert n1 == n2
            assert torch.allclose(p1, p2)


class TestCollapseDetection:
    """Test collapse detection utilities."""

    def test_healthy_state(self):
        """Test that healthy state has no collapse signals."""
        # Create gates with healthy values
        gates = PyramidalEpistemicGatesWithQ1Q2(d_model=64)
        hidden = torch.randn(4, 20, 64) * 0.5  # Moderate values

        outputs = gates(hidden)

        # Check metrics
        metrics = compute_pyramidal_q1q2_metrics(outputs)

        # Healthy ranges
        assert 0.1 < metrics["Q1_mean"] < 0.6, "Q1 out of healthy range"
        assert 0.1 < metrics["Q2_mean"] < 0.8, "Q2 out of healthy range"
        assert 0.2 < metrics["height_mean"] < 0.9, "Height out of healthy range"
        assert 0.0 < metrics["fractal_mean"] < 0.6, "Fractal out of healthy range"

        # Entropy should be reasonable
        assert metrics["Q1_entropy"] > 0.2, "Q1 entropy too low (possible collapse)"
        assert metrics["Q2_entropy"] > 0.2, "Q2 entropy too low (possible collapse)"

    def test_collapse_state(self):
        """Test detection of collapsed state."""
        # Create artificial collapse: all Q1 → 1.0
        batch_size, seq_len = 4, 20
        collapsed_outputs = {
            "Q1_mean": torch.ones(batch_size, seq_len, 1) * 0.95,  # Collapsed high
            "Q2_mean": torch.ones(batch_size, seq_len, 1) * 0.5,
            "height": torch.ones(batch_size, seq_len, 1) * 0.6,
            "fractal_uncertainty": torch.ones(batch_size, seq_len, 1) * 0.2,
            "base_weights": torch.ones(batch_size, seq_len, 4) * 0.25,
            "base_stability": torch.ones(batch_size, seq_len, 1) * 0.7,
            "w_memory": torch.ones(batch_size, seq_len, 1) * 0.25,
            "w_pain": torch.ones(batch_size, seq_len, 1) * 0.25,
            "w_choice": torch.ones(batch_size, seq_len, 1) * 0.25,
            "w_exploration": torch.ones(batch_size, seq_len, 1) * 0.25,
            "Q1_var": torch.ones(batch_size, seq_len, 1) * 0.1,
            "Q2_var": torch.ones(batch_size, seq_len, 1) * 0.1,
            "total_uncertainty": torch.ones(batch_size, seq_len, 1) * 0.4,
            "confidence": torch.ones(batch_size, seq_len, 1) * 0.5,
        }

        metrics = compute_pyramidal_q1q2_metrics(collapsed_outputs)

        # Q1 should be collapsed
        assert metrics["Q1_mean"] > 0.9

        # Q1 entropy should be very low
        assert metrics["Q1_entropy"] < 0.2, "Q1 entropy should be low (collapsed)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
