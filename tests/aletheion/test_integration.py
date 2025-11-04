"""Integration tests for Aletheion Level 1 end-to-end workflow.

Tests the complete pipeline:
    1. Load WikiText-2 dataset (small subset)
    2. Instantiate AletheionTransformer
    3. Train for a few steps with VARO loss
    4. Validate that loss decreases
    5. Validate that q1, q2 are in valid range
    6. Save and load checkpoint
    7. Test generation with uncertainty
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import torch
from pathlib import Path
import tempfile

from src import set_seed, get_device
from src.aletheion.model import AletheionTransformer
from src.aletheion.loss import VaroLoss
from data.dataset import load_wikitext_dataset, collate_fn
from torch.utils.data import DataLoader, Subset


@pytest.fixture
def device():
    """Get device for testing."""
    return get_device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def small_dataset():
    """Load a small subset of WikiText-2 for fast testing."""
    train_ds, val_ds, tokenizer, vocab_size = load_wikitext_dataset(
        tokenizer_name="gpt2",
        dataset_config="wikitext-2-raw-v1",
        max_length=128  # Shorter sequences for faster testing
    )

    # Use only first 10 samples
    train_subset = Subset(train_ds, range(min(10, len(train_ds))))
    val_subset = Subset(val_ds, range(min(10, len(val_ds))))

    return train_subset, val_subset, vocab_size, tokenizer


class TestAletheionIntegration:
    """Integration tests for complete Aletheion workflow."""

    def test_model_instantiation(self, device):
        """Test that Aletheion model can be instantiated."""
        model = AletheionTransformer(
            vocab_size=1000,
            d_model=256,
            n_layers=2,
            n_heads=4,
            d_ff=1024,
            max_seq_len=128,
            dropout=0.1,
            q1_threshold=0.7,
            q2_threshold=0.7,
            base_temperature=1.0,
            n_consensus_heads=2
        ).to(device)

        assert model is not None
        assert hasattr(model, 'q1_gate')
        assert hasattr(model, 'q2_gate')

    def test_forward_pass(self, device):
        """Test forward pass through Aletheion model."""
        set_seed(42)

        model = AletheionTransformer(
            vocab_size=1000,
            d_model=256,
            n_layers=2,
            n_heads=4,
            d_ff=1024,
            max_seq_len=128
        ).to(device)

        # Create dummy input
        input_ids = torch.randint(0, 1000, (4, 32), device=device)
        labels = torch.randint(0, 1000, (4, 32), device=device)

        # Forward pass
        outputs = model(input_ids, labels=labels, return_uncertainty=True)

        # Check outputs
        assert outputs.logits.shape == (4, 32, 1000)
        assert outputs.loss is not None
        assert outputs.uncertainty is not None
        assert outputs.q1 is not None
        assert outputs.q2 is not None

        # Check ranges
        assert (outputs.uncertainty >= 0.0).all() and (outputs.uncertainty <= 1.0).all()
        assert (outputs.q1 >= 0.0).all() and (outputs.q1 <= 1.0).all()
        assert (outputs.q2 >= 0.0).all() and (outputs.q2 <= 1.0).all()

    def test_training_loop(self, device, small_dataset):
        """Test training loop with VARO loss."""
        set_seed(42)

        train_subset, val_subset, vocab_size, tokenizer = small_dataset

        # Create model
        model = AletheionTransformer(
            vocab_size=vocab_size,
            d_model=256,
            n_layers=2,
            n_heads=4,
            d_ff=1024,
            max_seq_len=128,
            dropout=0.1
        ).to(device)

        # Create optimizer and loss
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        varo_loss = VaroLoss(lambda_varo=0.1, u_star_method='head_variance')

        # Create dataloader
        train_loader = DataLoader(
            train_subset,
            batch_size=2,
            shuffle=True,
            collate_fn=collate_fn
        )

        # Train for 5 steps
        model.train()
        losses = []

        for step, batch in enumerate(train_loader):
            if step >= 5:
                break

            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(input_ids, labels=labels, return_uncertainty=True)

            # Compute VARO loss
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_uncertainty = outputs.uncertainty[..., :-1, :].contiguous()

            loss_dict = varo_loss(
                logits=shift_logits,
                targets=shift_labels,
                uncertainty=shift_uncertainty
            )

            loss = loss_dict['loss']
            losses.append(loss.item())

            # Backward pass
            loss.backward()
            optimizer.step()

        # Loss should be finite
        assert all(torch.isfinite(torch.tensor(l)) for l in losses)

        # Loss should generally decrease (with some tolerance for small dataset)
        # Check that final loss is not worse than first loss by more than 20%
        assert losses[-1] < losses[0] * 1.2

    def test_checkpoint_save_load(self, device):
        """Test saving and loading Aletheion checkpoint."""
        set_seed(42)
        
        # Create model
        model = AletheionTransformer(
            vocab_size=1000,
            d_model=256,
            n_layers=2,
            n_heads=4,
            d_ff=1024,
            max_seq_len=128
        ).to(device)
        
        model.eval()  # ADICIONAR ESTA LINHA
        
        # Generate test input
        input_ids = torch.randint(0, 1000, (2, 16), device=device)
        
        # Get output from original model
        with torch.no_grad():
            output1 = model(input_ids, return_uncertainty=True)
        
        # Save and load using proper methods
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save checkpoint
            model.save_pretrained(tmpdir)
            
            # Load checkpoint
            model2 = AletheionTransformer.load_pretrained(tmpdir, device=str(device))
            model2.eval()  # ADICIONAR ESTA LINHA
            
            # Get output from loaded model
            with torch.no_grad():
                output2 = model2(input_ids, return_uncertainty=True)
            
            # Compare outputs
            assert torch.allclose(output1.logits, output2.logits, atol=1e-5), \
                "Loaded model produces different logits"
            assert torch.allclose(output1.uncertainty, output2.uncertainty, atol=1e-5), \
                "Loaded model produces different uncertainty"

    def test_generation_with_uncertainty(self, device):
        """Test generation with uncertainty-aware decoding."""
        set_seed(42)

        model = AletheionTransformer(
            vocab_size=1000,
            d_model=256,
            n_layers=2,
            n_heads=4,
            d_ff=1024,
            max_seq_len=128
        ).to(device)

        # Generate tokens
        input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=device)

        generated, uncertainties = model.generate(
            input_ids=input_ids,
            max_new_tokens=10,
            temperature=1.0,
            use_epistemic=True,
            uncertainty_threshold=0.8
        )

        # Check output shapes
        assert generated.shape == (1, 15)  # 5 input + 10 generated
        assert uncertainties.shape == (1, 10)  # 10 generated tokens

        # Check uncertainty range
        assert (uncertainties >= 0.0).all() and (uncertainties <= 1.0).all()

    def test_uncertainty_stats(self, device):
        """Test uncertainty statistics computation."""
        model = AletheionTransformer(
            vocab_size=1000,
            d_model=256,
            n_layers=2,
            n_heads=4,
            d_ff=1024,
            max_seq_len=128
        ).to(device)

        input_ids = torch.randint(0, 1000, (4, 32), device=device)
        outputs = model(input_ids, return_uncertainty=True)

        stats = model.get_uncertainty_stats(outputs.uncertainty)

        # Check that stats are computed
        assert 'uncertainty_mean' in stats
        assert 'uncertainty_std' in stats
        assert 'uncertainty_min' in stats
        assert 'uncertainty_max' in stats

        # Check that stats are in valid range
        assert 0.0 <= stats['uncertainty_mean'] <= 1.0
        assert 0.0 <= stats['uncertainty_min'] <= 1.0
        assert 0.0 <= stats['uncertainty_max'] <= 1.0

    def test_varo_loss_components(self, device):
        """Test that VARO loss components are computed correctly."""
        set_seed(42)

        varo_loss = VaroLoss(lambda_varo=0.1, u_star_method='head_variance')

        # Create dummy data
        batch_size, seq_len, vocab_size = 4, 32, 1000
        logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        uncertainty = torch.rand(batch_size, seq_len, 1, device=device)

        # Compute loss
        loss_dict = varo_loss(
            logits=logits,
            targets=targets,
            uncertainty=uncertainty
        )

        # Check all components are present
        assert 'loss' in loss_dict
        assert 'ce_loss' in loss_dict
        assert 'uncertainty_loss' in loss_dict
        assert 'u_star_mean' in loss_dict
        assert 'u_pred_mean' in loss_dict

        # Check that total loss = CE + λ * uncertainty_loss (approximately)
        expected_total = loss_dict['ce_loss'] + 0.1 * loss_dict['uncertainty_loss']
        assert torch.isclose(
            loss_dict['loss'],
            torch.tensor(expected_total, device=device),
            atol=1e-5
        )

    def test_eval_mode(self, device):
        """Test that model behaves correctly in eval mode."""
        set_seed(42)

        model = AletheionTransformer(
            vocab_size=1000,
            d_model=256,
            n_layers=2,
            n_heads=4,
            d_ff=1024,
            max_seq_len=128
        ).to(device)

        input_ids = torch.randint(0, 1000, (2, 16), device=device)

        # Get output in train mode
        model.train()
        with torch.no_grad():
            output_train = model(input_ids, return_uncertainty=True)

        # Get output in eval mode
        model.eval()
        with torch.no_grad():
            output_eval = model(input_ids, return_uncertainty=True)

        # Outputs should be similar (dropout is the main difference)
        # Just check shapes are correct
        assert output_train.logits.shape == output_eval.logits.shape
        assert output_train.uncertainty.shape == output_eval.uncertainty.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
