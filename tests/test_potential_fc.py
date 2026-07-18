import pytest
import torch
from quantum_plumbing.layers import PotentialFCLayer


class TestPotentialFCLayer:
    """
    Unit tests for PotentialFCLayer.
    """

    def test_initialization(self):
        """Test that layer initializes correctly."""
        layer = PotentialFCLayer(in_features=10, out_features=5, num_potentials=4)

        assert layer.in_features == 10
        assert layer.out_features == 5
        assert layer.num_potentials == 4
        assert layer.weight_potentials.shape == (4, 5, 10)
        assert layer.bias_potentials.shape == (4, 5)

    def test_forward_shape(self):
        """Test that forward pass produces correct shapes."""
        layer = PotentialFCLayer(in_features=20, out_features=10, num_potentials=8)

        x = torch.randn(32, 20)  # batch_size=32, in_features=20
        output, H = layer(x)

        # Output should be (batch_size, out_features)
        assert output.shape == (32, 10), f"Expected (32, 10), got {output.shape}"

        # H should be (num_potentials, batch_size, out_features)
        assert H.shape == (8, 32, 10), f"Expected (8, 32, 10), got {H.shape}"

    def test_h_properties(self):
        """Test that H has correct properties."""
        layer = PotentialFCLayer(10, 5, num_potentials=4)
        x = torch.randn(16, 10)
        output, H = layer(x)

        # H should have metadata
        assert hasattr(H, "_is_potential")
        assert H._is_potential is True

        assert hasattr(H, "_scores")
        assert H._scores.shape == (4, 16)

        # Scores should sum to 1 (softmax)
        scores_sum = torch.sum(H._scores, dim=0)
        assert torch.allclose(scores_sum, torch.ones(16))

    def test_with_bias_false(self):
        """Test layer without bias."""
        layer = PotentialFCLayer(
            in_features=10, out_features=5, num_potentials=4, bias=False
        )

        assert layer.bias_potentials is None

        x = torch.randn(16, 10)
        output, H = layer(x)

        assert output.shape == (16, 5)
        assert H.shape == (4, 16, 5)

    def test_gradient_flow(self):
        """Test that gradients flow correctly."""
        layer = PotentialFCLayer(10, 5, num_potentials=4)
        x = torch.randn(8, 10, requires_grad=True)

        output, H = layer(x)
        loss = output.sum()
        loss.backward()

        # Check that gradients are computed
        assert layer.weight_potentials.grad is not None
        assert layer.bias_potentials.grad is not None
        assert x.grad is not None

    def test_h_flow_concept(self):
        """Test that H can be passed to next layer."""
        layer1 = PotentialFCLayer(20, 10, num_potentials=4)
        layer2 = PotentialFCLayer(10, 5, num_potentials=4)

        x = torch.randn(8, 20)

        # First layer
        out1, H1 = layer1(x)
        assert out1.shape == (8, 10)
        assert H1.shape == (4, 8, 10)

        # Second layer receives output and H context
        out2, H2 = layer2(out1, prev_H=H1)
        assert out2.shape == (8, 5)
        assert H2.shape == (4, 8, 5)

    def test_prev_h_changes_output(self):
        """Different previous H tensors should change the actualized result."""
        torch.manual_seed(0)
        layer = PotentialFCLayer(10, 5, num_potentials=4)
        x = torch.randn(8, 10)
        prev_h_a = torch.randn(4, 8, 10)
        prev_h_b = torch.randn(4, 8, 10)

        out_a, H_a = layer(x, prev_H=prev_h_a)
        out_b, H_b = layer(x, prev_H=prev_h_b)

        assert not torch.allclose(out_a, out_b)
        assert not torch.allclose(H_a, H_b)

    def test_prev_h_can_be_disabled(self):
        """use_prev_h=False should ignore previous H context."""
        torch.manual_seed(0)
        layer = PotentialFCLayer(10, 5, num_potentials=4, use_prev_h=False)
        x = torch.randn(8, 10)
        prev_h_a = torch.randn(4, 8, 10)
        prev_h_b = torch.randn(4, 8, 10)

        out_a, H_a = layer(x, prev_H=prev_h_a)
        out_b, H_b = layer(x, prev_H=prev_h_b)

        assert torch.allclose(out_a, out_b)
        assert torch.allclose(H_a, H_b)

    def test_prev_h_shape_validation(self):
        """prev_H shape must align with layer expectations."""
        layer = PotentialFCLayer(10, 5, num_potentials=4)
        x = torch.randn(8, 10)
        with pytest.raises(ValueError, match="num_potentials"):
            layer(x, prev_H=torch.randn(3, 8, 10))
        with pytest.raises(ValueError, match="in_features"):
            layer(x, prev_H=torch.randn(4, 8, 9))

    def test_batch_size_one(self):
        """Test with batch size 1."""
        layer = PotentialFCLayer(10, 5, num_potentials=4)
        x = torch.randn(1, 10)

        output, H = layer(x)

        assert output.shape == (1, 5)
        assert H.shape == (4, 1, 5)

    def test_deterministic_with_seed(self):
        """Test reproducibility with seed."""
        torch.manual_seed(42)
        layer1 = PotentialFCLayer(10, 5, num_potentials=4)
        x = torch.randn(8, 10)
        output1, H1 = layer1(x)

        torch.manual_seed(42)
        layer2 = PotentialFCLayer(10, 5, num_potentials=4)
        output2, H2 = layer2(x)

        # Same seed should give same initialization
        assert torch.allclose(output1, output2)
        assert torch.allclose(H1, H2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
