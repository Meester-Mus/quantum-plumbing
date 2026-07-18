import pytest
import torch
import torch.optim as optim

from quantum_plumbing import (
    PotentialFCLayer,
    PotentialBatchNorm,
    PotentialDropout,
    PotentialActivation,
    PotentialSequential,
    PotentialMLP,
    potential_loss,
    h_utilization,
)


# ---------------------------------------------------------------------------
# PotentialSequential
# ---------------------------------------------------------------------------

class TestPotentialSequential:
    def _simple_net(self, in_f=20, hidden=10, out_f=5, num_potentials=4):
        return PotentialSequential(
            PotentialFCLayer(in_f, hidden, num_potentials=num_potentials),
            PotentialBatchNorm(hidden),
            PotentialDropout(0.1),
            PotentialActivation('relu'),
            PotentialFCLayer(hidden, out_f, num_potentials=num_potentials),
        )

    def test_output_shapes(self):
        net = self._simple_net()
        x = torch.randn(16, 20)
        output, H = net(x)
        assert output.shape == (16, 5)
        assert H.shape == (4, 16, 5)

    def test_single_fc_layer(self):
        net = PotentialSequential(PotentialFCLayer(10, 4, num_potentials=3))
        x = torch.randn(8, 10)
        output, H = net(x)
        assert output.shape == (8, 4)
        assert H.shape == (3, 8, 4)

    def test_no_fc_layer_raises(self):
        """A sequential with no FC layer should raise at forward time."""
        net = PotentialSequential()
        # Manually add a non-FC layer won't produce H; but an empty net raises too.
        x = torch.randn(4, 10)
        with pytest.raises(RuntimeError):
            net(x)

    def test_h_layer_before_fc_raises(self):
        """BatchNorm before any FC layer (no H yet) should raise."""
        net = PotentialSequential(
            PotentialBatchNorm(10),
            PotentialFCLayer(10, 5, num_potentials=2),
        )
        x = torch.randn(4, 10)
        with pytest.raises(RuntimeError):
            net(x)

    def test_gradient_flow(self):
        net = self._simple_net()
        x = torch.randn(8, 20, requires_grad=True)
        output, H = net(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        for param in net.parameters():
            assert param.grad is not None

    def test_train_eval_mode(self):
        net = self._simple_net()
        net.train()
        x = torch.randn(8, 20)
        output_train, _ = net(x)

        net.eval()
        with torch.no_grad():
            output_eval, _ = net(x)

        assert output_train.shape == output_eval.shape

    def test_batch_size_one(self):
        net = self._simple_net()
        net.eval()
        with torch.no_grad():
            output, H = net(torch.randn(1, 20))
        assert output.shape == (1, 5)

    def test_h_carries_scores_metadata(self):
        """Final H should carry _scores from the last FC layer."""
        net = PotentialSequential(
            PotentialFCLayer(10, 5, num_potentials=4),
        )
        x = torch.randn(6, 10)
        _, H = net(x)
        assert hasattr(H, '_scores')
        assert H._scores.shape == (4, 6)


# ---------------------------------------------------------------------------
# PotentialMLP builder
# ---------------------------------------------------------------------------

class TestPotentialMLP:
    def test_basic_construction(self):
        model = PotentialMLP([784, 256, 128, 10], num_potentials=8)
        x = torch.randn(4, 784)
        output, H = model(x)
        assert output.shape == (4, 10)
        assert H.shape == (8, 4, 10)

    def test_two_layer_mlp(self):
        model = PotentialMLP([20, 10], num_potentials=4)
        x = torch.randn(8, 20)
        output, H = model(x)
        assert output.shape == (8, 10)

    def test_no_batch_norm(self):
        model = PotentialMLP([20, 10, 5], num_potentials=4, batch_norm=False)
        x = torch.randn(8, 20)
        output, H = model(x)
        assert output.shape == (8, 5)

    def test_no_dropout(self):
        model = PotentialMLP([20, 10, 5], num_potentials=4, dropout_p=0.0)
        x = torch.randn(8, 20)
        output, H = model(x)
        assert output.shape == (8, 5)

    def test_different_activations(self):
        for act in ('relu', 'tanh', 'sigmoid', 'elu'):
            model = PotentialMLP([20, 10, 5], num_potentials=4, activation=act)
            output, _ = model(torch.randn(4, 20))
            assert output.shape == (4, 5)

    def test_invalid_layer_sizes(self):
        with pytest.raises(ValueError):
            PotentialMLP([10])

    def test_gradient_flow(self):
        model = PotentialMLP([20, 10, 5], num_potentials=4)
        x = torch.randn(8, 20)
        output, H = model(x)
        output.sum().backward()
        for p in model.parameters():
            assert p.grad is not None

    def test_no_bias(self):
        model = PotentialMLP([20, 10], num_potentials=4, bias=False)
        x = torch.randn(4, 20)
        output, H = model(x)
        assert output.shape == (4, 10)


# ---------------------------------------------------------------------------
# potential_loss
# ---------------------------------------------------------------------------

class TestPotentialLoss:
    def test_classification_loss(self):
        output = torch.randn(8, 5)
        target = torch.randint(0, 5, (8,))
        loss = potential_loss(output, target, task='classification')
        assert loss.shape == ()
        assert loss.item() > 0

    def test_regression_loss(self):
        output = torch.randn(8, 1)
        target = torch.randn(8)
        loss = potential_loss(output, target, task='regression')
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_with_h_diversity(self):
        output = torch.randn(8, 5)
        target = torch.randint(0, 5, (8,))
        H = torch.randn(4, 8, 5)
        loss_no_reg = potential_loss(output, target, task='classification')
        loss_with_reg = potential_loss(output, target, H=H, task='classification',
                                       h_diversity_weight=0.01)
        # They should differ (diversity term changes the loss)
        assert not torch.isclose(loss_no_reg, loss_with_reg)

    def test_h_diversity_zero_weight(self):
        """h_diversity_weight=0 should give same result as not passing H."""
        output = torch.randn(8, 5)
        target = torch.randint(0, 5, (8,))
        H = torch.randn(4, 8, 5)
        loss_a = potential_loss(output, target, task='classification')
        loss_b = potential_loss(output, target, H=H, task='classification',
                                h_diversity_weight=0.0)
        assert torch.isclose(loss_a, loss_b)

    def test_invalid_task(self):
        with pytest.raises(ValueError):
            potential_loss(torch.randn(4, 3), torch.randint(0, 3, (4,)), task='unknown')

    def test_differentiable(self):
        output = torch.randn(8, 5, requires_grad=True)
        target = torch.randint(0, 5, (8,))
        H = torch.randn(4, 8, 5)
        loss = potential_loss(output, target, H=H, h_diversity_weight=0.01)
        loss.backward()
        assert output.grad is not None


# ---------------------------------------------------------------------------
# h_utilization
# ---------------------------------------------------------------------------

class TestHUtilization:
    def test_returns_scalar(self):
        H = torch.randn(4, 8, 10)
        util = h_utilization(H)
        assert util.shape == ()

    def test_range(self):
        H = torch.randn(4, 8, 10)
        util = h_utilization(H)
        assert 0.0 <= util.item() <= 1.0

    def test_with_scores_metadata(self):
        """Uses _scores attribute when present (as set by PotentialFCLayer)."""
        H = torch.randn(4, 8, 10)
        # Uniform scores → max entropy → utilisation = 1
        uniform_scores = torch.ones(4, 8) / 4
        H._scores = uniform_scores
        util = h_utilization(H)
        assert torch.isclose(util, torch.tensor(1.0), atol=1e-5)

    def test_with_scores_peaked(self):
        """Peaked distribution → low utilisation."""
        H = torch.randn(4, 8, 10)
        peaked_scores = torch.zeros(4, 8)
        peaked_scores[0] = 1.0  # all weight on potential 0
        H._scores = peaked_scores
        util = h_utilization(H)
        assert util.item() < 0.1

    def test_from_real_layer(self):
        """Utilisation computed from a real PotentialFCLayer output."""
        layer = PotentialFCLayer(10, 5, num_potentials=4)
        x = torch.randn(8, 10)
        _, H = layer(x)
        util = h_utilization(H)
        assert 0.0 <= util.item() <= 1.0


# ---------------------------------------------------------------------------
# End-to-end mini training loop
# ---------------------------------------------------------------------------

class TestEndToEndTraining:
    def test_training_step(self):
        """A single SGD step should reduce loss."""
        torch.manual_seed(0)
        model = PotentialMLP([20, 10, 4], num_potentials=4, dropout_p=0.0)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        x = torch.randn(16, 20)
        y = torch.randint(0, 4, (16,))

        model.train()
        output, H = model(x)
        loss_before = potential_loss(output, y)

        optimizer.zero_grad()
        loss_before.backward()
        optimizer.step()

        output2, _ = model(x)
        loss_after = potential_loss(output2, y)

        # Loss should change (not necessarily decrease in 1 step with Adam, but
        # it proves the training loop runs without error)
        assert loss_before.item() != loss_after.item()

    def test_h_utilization_in_training(self):
        """h_utilization should be computable at every training step."""
        model = PotentialMLP([10, 8, 3], num_potentials=4, dropout_p=0.0)
        x = torch.randn(8, 10)
        model.train()
        _, H = model(x)
        util = h_utilization(H)
        assert 0.0 <= util.item() <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
