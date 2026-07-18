import pytest
import torch

from quantum_plumbing.layers import (
    PotentialActivation,
    PotentialBatchNorm,
    PotentialDropout,
    PotentialFCLayer,
)


class TestPotentialBatchNormContracts:
    def test_train_shapes_and_running_stats(self):
        bn = PotentialBatchNorm(num_features=6)
        x = torch.randn(5, 6)
        H = torch.randn(4, 5, 6)

        x_norm, H_norm = bn(x, H)

        assert x_norm.shape == x.shape
        assert H_norm.shape == H.shape
        assert bn.running_mean.shape == (6,)
        assert bn.running_var.shape == (6,)

    def test_eval_normalizes_h_with_running_stats(self):
        bn = PotentialBatchNorm(num_features=6)
        x = torch.randn(5, 6)
        H = torch.randn(4, 5, 6)

        bn.train()
        bn(x, H)

        bn.eval()
        with torch.no_grad():
            _x_eval, H_eval = bn(x, H)

        assert not torch.allclose(H_eval, H)

    def test_invalid_shapes_raise(self):
        bn = PotentialBatchNorm(num_features=6)
        with pytest.raises(ValueError):
            bn(torch.randn(5, 6, 1), torch.randn(4, 5, 6))
        with pytest.raises(ValueError):
            bn(torch.randn(5, 6), torch.randn(4, 5))
        with pytest.raises(ValueError):
            bn(torch.randn(5, 7), torch.randn(4, 5, 7))


class TestPotentialDropoutContracts:
    def test_feature_mask_applied_to_x_and_h(self):
        layer = PotentialDropout(p=0.5)
        layer.train()

        x = torch.randn(8, 10)
        H = torch.randn(4, 8, 10)

        x_out, H_out = layer(x, H)
        mask = H_out._mask

        assert mask.shape == (10,)
        assert torch.all(x_out[:, mask == 0] == 0)
        assert torch.all(H_out[:, :, mask == 0] == 0)

    def test_invalid_shapes_raise(self):
        layer = PotentialDropout(p=0.2)
        with pytest.raises(ValueError):
            layer(torch.randn(8, 10, 1), torch.randn(4, 8, 10))
        with pytest.raises(ValueError):
            layer(torch.randn(8, 10), torch.randn(4, 8))


class TestPotentialActivationContracts:
    def test_invalid_shapes_raise(self):
        layer = PotentialActivation("relu")
        with pytest.raises(ValueError):
            layer(torch.randn(8, 10, 1), torch.randn(4, 8, 10))
        with pytest.raises(ValueError):
            layer(torch.randn(8, 10), torch.randn(4, 8))


class TestPotentialFCLayerContracts:
    def test_feature_validation_raises(self):
        layer = PotentialFCLayer(10, 5, num_potentials=4)
        with pytest.raises(ValueError):
            layer(torch.randn(8, 9))

    def test_prev_h_batch_validation_raises(self):
        layer = PotentialFCLayer(10, 5, num_potentials=4)
        with pytest.raises(ValueError):
            layer(torch.randn(8, 10), prev_H=torch.randn(4, 7, 10))
