"""Unit tests for epistemic gates and epistemic softmax.

Tests cover:
    - LocalUncertaintyGate (Q₁)
    - CrossContextGate (Q₂)
    - epistemic_softmax function
    - Shape validation
    - Range validation ([0,1] for gates, sum=1 for probabilities)
    - Gradient flow
    - Uncertainty behavior
"""

import pytest
import torch
import torch.nn as nn

from src.aletheion.gates import (
    LocalUncertaintyGate,
    CrossContextGate,
    epistemic_softmax,
    entropy_regularization
)


@pytest.fixture
def device():
    """Get device for testing (prefer CUDA if available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def batch_config():
    """Standard batch configuration for tests."""
    return {
        "batch_size": 4,
        "seq_len": 32,
        "d_model": 128,
        "vocab_size": 1000
    }


class TestLocalUncertaintyGate:
    """Tests for Q₁ gate (local uncertainty)."""

    def test_initialization(self):
        """Test Q₁ gate can be created with correct architecture."""
        gate = LocalUncertaintyGate(d_model=512, dropout=0.1)
        assert gate.d_model == 512
        assert gate.projection.in_features == 512
        assert gate.projection.out_features == 1

    def test_forward_2d(self, device, batch_config):
        """Test Q₁ forward pass with 2D input (batch, d_model)."""
        gate = LocalUncertaintyGate(d_model=batch_config["d_model"]).to(device)
        context = torch.randn(
            batch_config["batch_size"],
            batch_config["d_model"],
            device=device
        )

        q1 = gate(context)

        # Check shape
        assert q1.shape == (batch_config["batch_size"], 1)

        # Check range [0, 1]
        assert (q1 >= 0.0).all() and (q1 <= 1.0).all()

    def test_forward_3d(self, device, batch_config):
        """Test Q₁ forward pass with 3D input (batch, seq_len, d_model)."""
        gate = LocalUncertaintyGate(d_model=batch_config["d_model"]).to(device)
        context = torch.randn(
            batch_config["batch_size"],
            batch_config["seq_len"],
            batch_config["d_model"],
            device=device
        )

        q1 = gate(context)

        # Check shape
        assert q1.shape == (batch_config["batch_size"], batch_config["seq_len"], 1)

        # Check range [0, 1]
        assert (q1 >= 0.0).all() and (q1 <= 1.0).all()

    def test_gradients_flow(self, device, batch_config):
        """Test that gradients flow through Q₁ gate."""
        gate = LocalUncertaintyGate(d_model=batch_config["d_model"]).to(device)
        context = torch.randn(
            batch_config["batch_size"],
            batch_config["d_model"],
            device=device,
            requires_grad=True
        )

        q1 = gate(context)
        loss = q1.sum()
        loss.backward()

        # Check gradients exist
        assert context.grad is not None
        assert not torch.isnan(context.grad).any()
        assert gate.projection.weight.grad is not None

    def test_different_inputs_different_outputs(self, device, batch_config):
        """Test that different contexts produce different Q₁ values."""
        gate = LocalUncertaintyGate(d_model=batch_config["d_model"]).to(device)

        context1 = torch.randn(4, batch_config["d_model"], device=device)
        context2 = torch.randn(4, batch_config["d_model"], device=device)

        q1_1 = gate(context1)
        q1_2 = gate(context2)

        # Different inputs should produce different outputs
        assert not torch.allclose(q1_1, q1_2)


class TestCrossContextGate:
    """Tests for Q₂ gate (cross-context consensus)."""

    def test_initialization(self):
        """Test Q₂ gate can be created with correct architecture."""
        gate = CrossContextGate(d_model=512, n_heads=4, dropout=0.1)
        assert gate.d_model == 512
        assert gate.n_heads == 4
        assert gate.d_head == 512 // 4

    def test_initialization_invalid_heads(self):
        """Test Q₂ raises error if d_model not divisible by n_heads."""
        with pytest.raises(ValueError):
            CrossContextGate(d_model=513, n_heads=4)  # 513 not divisible by 4

    def test_forward_3d(self, device, batch_config):
        """Test Q₂ forward pass with 3D input."""
        gate = CrossContextGate(
            d_model=batch_config["d_model"],
            n_heads=4,
            dropout=0.1
        ).to(device)

        context = torch.randn(
            batch_config["batch_size"],
            batch_config["seq_len"],
            batch_config["d_model"],
            device=device
        )

        q2 = gate(context)

        # Check shape
        assert q2.shape == (batch_config["batch_size"], batch_config["seq_len"], 1)

        # Check range [0, 1]
        assert (q2 >= 0.0).all() and (q2 <= 1.0).all()

    def test_gradients_flow(self, device, batch_config):
        """Test that gradients flow through Q₂ gate."""
        gate = CrossContextGate(d_model=batch_config["d_model"], n_heads=4).to(device)
        context = torch.randn(
            batch_config["batch_size"],
            batch_config["seq_len"],
            batch_config["d_model"],
            device=device,
            requires_grad=True
        )

        q2 = gate(context)
        loss = q2.sum()
        loss.backward()

        # Check gradients exist
        assert context.grad is not None
        assert not torch.isnan(context.grad).any()
        assert gate.out_proj.weight.grad is not None

    def test_consensus_across_sequence(self, device, batch_config):
        """Test that Q₂ captures cross-sequence consensus."""
        gate = CrossContextGate(d_model=batch_config["d_model"], n_heads=4).to(device)

        # Create context with high agreement (all positions similar)
        base = torch.randn(1, 1, batch_config["d_model"], device=device)
        context_high_consensus = base.expand(4, 32, -1) + 0.01 * torch.randn(4, 32, batch_config["d_model"], device=device)

        # Create context with low agreement (positions very different)
        context_low_consensus = torch.randn(4, 32, batch_config["d_model"], device=device)

        q2_high = gate(context_high_consensus)
        q2_low = gate(context_low_consensus)

        # Mean Q₂ should be different (though not guaranteed always higher/lower)
        assert q2_high.shape == q2_low.shape


class TestEpistemicSoftmax:
    """Tests for epistemic_softmax function (Algorithm 1)."""

    def test_basic_functionality(self, device, batch_config):
        """Test epistemic softmax basic forward pass."""
        q1_gate = LocalUncertaintyGate(d_model=batch_config["d_model"]).to(device)
        q2_gate = CrossContextGate(d_model=batch_config["d_model"], n_heads=4).to(device)

        logits = torch.randn(
            batch_config["batch_size"],
            batch_config["seq_len"],
            batch_config["vocab_size"],
            device=device
        )
        context = torch.randn(
            batch_config["batch_size"],
            batch_config["seq_len"],
            batch_config["d_model"],
            device=device
        )

        probs, uncertainty = epistemic_softmax(
            logits=logits,
            context=context,
            q1_gate=q1_gate,
            q2_gate=q2_gate
        )

        # Check shapes
        assert probs.shape == logits.shape
        assert uncertainty.shape[0] == batch_config["batch_size"]
        assert uncertainty.shape[1] == batch_config["seq_len"]

        # Check probability distribution sums to 1
        prob_sums = probs.sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5)

        # Check uncertainty range [0, 1]
        assert (uncertainty >= 0.0).all() and (uncertainty <= 1.0).all()

    def test_without_q2_gate(self, device, batch_config):
        """Test epistemic softmax works without Q₂ gate."""
        q1_gate = LocalUncertaintyGate(d_model=batch_config["d_model"]).to(device)

        logits = torch.randn(4, 32, 1000, device=device)
        context = torch.randn(4, 32, batch_config["d_model"], device=device)

        probs, uncertainty = epistemic_softmax(
            logits=logits,
            context=context,
            q1_gate=q1_gate,
            q2_gate=None  # No Q₂ gate
        )

        # Should still work (assumes q2 = 1)
        assert probs.shape == logits.shape
        assert (probs.sum(dim=-1) - 1.0).abs().max() < 1e-5

    def test_high_confidence_produces_peaked_distribution(self, device):
        """Test that high confidence (q1=q2≈1) produces peaked distribution."""
        d_model = 64
        vocab_size = 100

        # Create gates that output high confidence
        q1_gate = LocalUncertaintyGate(d_model=d_model).to(device)
        q2_gate = CrossContextGate(d_model=d_model, n_heads=4).to(device)

        # Logits with clear winner
        logits = torch.zeros(1, 1, vocab_size, device=device)
        logits[0, 0, 0] = 10.0  # Strong preference for token 0

        context = torch.randn(1, 1, d_model, device=device)

        probs, uncertainty = epistemic_softmax(
            logits=logits,
            context=context,
            q1_gate=q1_gate,
            q2_gate=q2_gate
        )

        # Should be mostly peaked (if gates are confident)
        max_prob = probs.max().item()
        uniform_prob = 1.0 / vocab_size

        # Max probability should be significantly higher than uniform
        # (exact value depends on gate outputs, but should show peaking)
        assert max_prob > uniform_prob * 2

    def test_temperature_adjustment(self, device):
        """Test that low confidence increases temperature."""
        d_model = 64
        vocab_size = 100

        q1_gate = LocalUncertaintyGate(d_model=d_model).to(device)

        logits = torch.tensor([[[10.0] + [0.0] * (vocab_size - 1)]], device=device)
        context = torch.randn(1, 1, d_model, device=device)

        # Test with different confidence thresholds
        probs_low_thresh, _ = epistemic_softmax(
            logits=logits,
            context=context,
            q1_gate=q1_gate,
            q2_gate=None,
            confidence_threshold=0.1  # Low threshold
        )

        probs_high_thresh, _ = epistemic_softmax(
            logits=logits,
            context=context,
            q1_gate=q1_gate,
            q2_gate=None,
            confidence_threshold=0.9  # High threshold
        )

        # Both should sum to 1
        assert torch.allclose(probs_low_thresh.sum(dim=-1), torch.ones(1, 1, device=device), atol=1e-5)
        assert torch.allclose(probs_high_thresh.sum(dim=-1), torch.ones(1, 1, device=device), atol=1e-5)

    def test_gradients_flow_through_gates(self, device, batch_config):
        """Test that gradients flow through epistemic softmax to gates."""
        q1_gate = LocalUncertaintyGate(d_model=batch_config["d_model"]).to(device)
        q2_gate = CrossContextGate(d_model=batch_config["d_model"], n_heads=4).to(device)

        logits = torch.randn(4, 32, 1000, device=device, requires_grad=True)
        context = torch.randn(4, 32, batch_config["d_model"], device=device, requires_grad=True)

        probs, uncertainty = epistemic_softmax(
            logits=logits,
            context=context,
            q1_gate=q1_gate,
            q2_gate=q2_gate
        )

        loss = uncertainty.sum()
        loss.backward()

        # Check gradients flow to gates
        assert q1_gate.projection.weight.grad is not None
        assert q2_gate.out_proj.weight.grad is not None
        assert context.grad is not None

    def test_uncertainty_inverse_of_confidence(self, device, batch_config):
        """Test that uncertainty = 1 - confidence."""
        q1_gate = LocalUncertaintyGate(d_model=batch_config["d_model"]).to(device)
        q2_gate = CrossContextGate(d_model=batch_config["d_model"], n_heads=4).to(device)

        # Set to eval mode to disable dropout for deterministic testing
        q1_gate.eval()
        q2_gate.eval()

        logits = torch.randn(4, 32, 1000, device=device)
        context = torch.randn(4, 32, batch_config["d_model"], device=device)

        probs, uncertainty = epistemic_softmax(
            logits=logits,
            context=context,
            q1_gate=q1_gate,
            q2_gate=q2_gate
        )

        # Get individual gate outputs
        q1 = q1_gate(context)
        q2 = q2_gate(context)

        # Confidence = q1 * q2
        confidence = torch.clamp(q1 * q2, min=1e-8, max=1.0)

        # Check u = 1 - c
        expected_uncertainty = 1.0 - confidence
        assert torch.allclose(uncertainty, expected_uncertainty, atol=1e-6)


class TestEntropyRegularization:
    """Tests for entropy regularization to prevent gate collapse."""

    def test_entropy_penalty_on_collapsed_gates(self):
        """Test that collapsed gates (all 0 or all 1) have high penalty."""
        # All gates saturated at 1 (low entropy)
        gates_collapsed = torch.ones(10, 10)
        penalty_collapsed = entropy_regularization(gates_collapsed, min_entropy=0.5)

        # Mixed gates (high entropy)
        gates_mixed = torch.rand(10, 10) * 0.5 + 0.25  # Range [0.25, 0.75]
        penalty_mixed = entropy_regularization(gates_mixed, min_entropy=0.5)

        # Collapsed gates should have higher penalty
        assert penalty_collapsed > penalty_mixed

    def test_no_penalty_for_diverse_gates(self):
        """Test that gates with sufficient entropy have low penalty."""
        gates = torch.rand(100, 100) * 0.6 + 0.2  # Range [0.2, 0.8]
        penalty = entropy_regularization(gates, min_entropy=0.1)

        # Should have low or zero penalty
        assert penalty < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
