"""Tests for the quantum hardware interface (QuantumHScorer, QuantumPotentialFCLayer).

All tests are skipped automatically when Qiskit is not installed.
"""
import numpy as np
import pytest
import torch

# Skip the entire module if Qiskit is not installed.
pytest.importorskip("qiskit", reason="qiskit not installed – skipping quantum interface tests")

from quantum_plumbing.quantum_interface import QuantumHScorer, QuantumPotentialFCLayer
from quantum_plumbing.layers import PotentialFCLayer


# ---------------------------------------------------------------------------
# QuantumHScorer
# ---------------------------------------------------------------------------


class TestQuantumHScorer:
    """Unit tests for QuantumHScorer."""

    def test_invalid_num_potentials_raises(self):
        """num_potentials < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="num_potentials"):
            QuantumHScorer(num_potentials=0)

    def test_init_2_potentials(self):
        """2 potentials → 1 qubit."""
        scorer = QuantumHScorer(num_potentials=2)
        assert scorer._num_potentials == 2
        assert scorer._n_qubits == 1
        assert scorer._state_size == 2

    def test_init_4_potentials(self):
        """4 potentials → 2 qubits."""
        scorer = QuantumHScorer(num_potentials=4)
        assert scorer._n_qubits == 2
        assert scorer._state_size == 4

    def test_init_8_potentials(self):
        """8 potentials → 3 qubits."""
        scorer = QuantumHScorer(num_potentials=8)
        assert scorer._n_qubits == 3
        assert scorer._state_size == 8

    def test_n_qubits_non_power_of_two(self):
        """5 potentials → ceil(log2(5))=3 qubits (state_size=8)."""
        scorer = QuantumHScorer(num_potentials=5)
        assert scorer._n_qubits == 3
        assert scorer._state_size == 8

    def test_score_output_shape(self):
        scorer = QuantumHScorer(num_potentials=4)
        H = torch.randn(4, 8, 10)
        scores = scorer.score(H)
        assert scores.shape == (4, 8)

    def test_scores_sum_to_one(self):
        scorer = QuantumHScorer(num_potentials=4)
        H = torch.randn(4, 8, 10)
        scores = scorer.score(H)
        col_sums = scores.sum(dim=0)
        assert torch.allclose(col_sums, torch.ones(8), atol=1e-5)

    def test_scores_non_negative(self):
        scorer = QuantumHScorer(num_potentials=4)
        H = torch.randn(4, 8, 10)
        scores = scorer.score(H)
        assert (scores >= -1e-9).all(), "scores must be non-negative"

    def test_degenerate_all_zero_h(self):
        """All-zero H should give uniform scores."""
        scorer = QuantumHScorer(num_potentials=4)
        H = torch.zeros(4, 4, 5)
        scores = scorer.score(H)
        assert scores.shape == (4, 4)
        col_sums = scores.sum(dim=0)
        assert torch.allclose(col_sums, torch.ones(4), atol=1e-5)
        # Each potential should have equal score
        expected = torch.full((4, 4), 0.25)
        assert torch.allclose(scores, expected, atol=1e-5)

    def test_different_interference_depths_give_different_scores(self):
        """Deeper interference should change the score distribution."""
        torch.manual_seed(0)
        H = torch.randn(4, 4, 10)
        scorer1 = QuantumHScorer(4, n_interference_layers=1)
        scorer2 = QuantumHScorer(4, n_interference_layers=2)
        scores1 = scorer1.score(H)
        scores2 = scorer2.score(H)
        assert not torch.allclose(scores1, scores2), (
            "Different interference depths should produce different scores"
        )

    def test_score_dtype_preserved(self):
        """Output dtype should match H dtype."""
        scorer = QuantumHScorer(num_potentials=4)
        H32 = torch.randn(4, 4, 5, dtype=torch.float32)
        scores32 = scorer.score(H32)
        assert scores32.dtype == torch.float32

    def test_score_device_preserved(self):
        """Output should be on the same device as H."""
        scorer = QuantumHScorer(num_potentials=4)
        H = torch.randn(4, 4, 5)
        scores = scorer.score(H)
        assert scores.device == H.device

    def test_batch_size_one(self):
        scorer = QuantumHScorer(num_potentials=4)
        H = torch.randn(4, 1, 10)
        scores = scorer.score(H)
        assert scores.shape == (4, 1)
        assert torch.allclose(scores.sum(dim=0), torch.ones(1), atol=1e-5)

    def test_single_potential(self):
        """num_potentials=1 edge case: single hypothesis."""
        scorer = QuantumHScorer(num_potentials=1)
        H = torch.randn(1, 4, 10)
        scores = scorer.score(H)
        assert scores.shape == (1, 4)
        # Only one potential: score must be 1.0
        assert torch.allclose(scores, torch.ones(1, 4), atol=1e-5)

    def test_simulator_scoring_is_deterministic(self):
        scorer = QuantumHScorer(num_potentials=4)
        H = torch.randn(4, 6, 8)
        scores_a = scorer.score(H)
        scores_b = scorer.score(H)
        assert torch.allclose(scores_a, scores_b)

    def test_repr(self):
        scorer = QuantumHScorer(num_potentials=4, n_interference_layers=2)
        r = repr(scorer)
        assert "QuantumHScorer" in r
        assert "num_potentials=4" in r
        assert "n_qubits=2" in r
        assert "n_interference_layers=2" in r


# ---------------------------------------------------------------------------
# QuantumPotentialFCLayer
# ---------------------------------------------------------------------------


class TestQuantumPotentialFCLayer:
    """Unit tests for QuantumPotentialFCLayer."""

    def test_output_shapes(self):
        layer = QuantumPotentialFCLayer(20, 10, num_potentials=4)
        x = torch.randn(8, 20)
        output, H = layer(x)
        assert output.shape == (8, 10)
        assert H.shape == (4, 8, 10)

    def test_default_backend_uses_simulator(self):
        layer = QuantumPotentialFCLayer(10, 5, num_potentials=4)
        assert layer.quantum_scorer._backend is None

    def test_h_metadata_present(self):
        layer = QuantumPotentialFCLayer(10, 5, num_potentials=4)
        x = torch.randn(4, 10)
        _, H = layer(x)
        assert hasattr(H, "_is_potential") and H._is_potential is True
        assert hasattr(H, "_layer") and H._layer == "QuantumFC"
        assert hasattr(H, "_scores")

    def test_scores_shape_and_sum(self):
        layer = QuantumPotentialFCLayer(10, 5, num_potentials=4)
        x = torch.randn(8, 10)
        _, H = layer(x)
        assert H._scores.shape == (4, 8)
        col_sums = H._scores.sum(dim=0)
        assert torch.allclose(col_sums, torch.ones(8), atol=1e-5)

    def test_gradient_flows_through_weights(self):
        """Gradients must reach weight_potentials despite quantum scoring."""
        layer = QuantumPotentialFCLayer(10, 5, num_potentials=4)
        x = torch.randn(4, 10, requires_grad=True)
        output, _ = layer(x)
        output.sum().backward()
        assert layer.weight_potentials.grad is not None, "weight_potentials should have gradients"
        assert x.grad is not None, "input x should have gradients"

    def test_gradient_flows_through_bias(self):
        layer = QuantumPotentialFCLayer(10, 5, num_potentials=4)
        x = torch.randn(4, 10)
        output, _ = layer(x)
        output.sum().backward()
        assert layer.bias_potentials.grad is not None

    def test_no_bias(self):
        layer = QuantumPotentialFCLayer(10, 5, num_potentials=4, bias=False)
        assert layer.bias_potentials is None
        x = torch.randn(4, 10)
        output, H = layer(x)
        assert output.shape == (4, 5)

    def test_batch_size_one(self):
        layer = QuantumPotentialFCLayer(10, 5, num_potentials=4)
        x = torch.randn(1, 10)
        output, H = layer(x)
        assert output.shape == (1, 5)

    def test_prev_h_ignored(self):
        """prev_H is accepted for API compat but doesn't change output."""
        layer = QuantumPotentialFCLayer(10, 5, num_potentials=4)
        x = torch.randn(4, 10)
        out_no_h, H_no_h = layer(x, prev_H=None)
        dummy_prev = torch.randn(4, 4, 10)
        out_with_h, H_with_h = layer(x, prev_H=dummy_prev)
        # Same x → same output (quantum scorer depends on H norms, not prev_H)
        assert torch.allclose(out_no_h, out_with_h)

    def test_quantum_scores_differ_from_classical(self):
        """Quantum scoring should produce different scores than classical norm-softmax."""
        torch.manual_seed(0)
        classical = PotentialFCLayer(10, 5, num_potentials=4)
        quantum = QuantumPotentialFCLayer(10, 5, num_potentials=4)
        # Share identical weights so only the scoring method differs
        quantum.weight_potentials.data = classical.weight_potentials.data.clone()
        quantum.bias_potentials.data = classical.bias_potentials.data.clone()

        x = torch.randn(4, 10)
        _, H_classical = classical(x)
        _, H_quantum = quantum(x)

        assert not torch.allclose(H_classical._scores, H_quantum._scores), (
            "Quantum and classical scorers should differ"
        )

    def test_extra_repr(self):
        layer = QuantumPotentialFCLayer(10, 5, num_potentials=4, n_interference_layers=3)
        r = layer.extra_repr()
        assert "n_interference_layers=3" in r
        assert "in_features=10" in r
        assert "out_features=5" in r

    def test_in_potential_sequential(self):
        """QuantumPotentialFCLayer should work inside PotentialSequential."""
        from quantum_plumbing import PotentialSequential, PotentialActivation

        net = PotentialSequential(
            QuantumPotentialFCLayer(20, 10, num_potentials=4),
            PotentialActivation("relu"),
            QuantumPotentialFCLayer(10, 5, num_potentials=4),
        )
        x = torch.randn(8, 20)
        output, H = net(x)
        assert output.shape == (8, 5)
        assert H.shape == (4, 8, 5)


# ---------------------------------------------------------------------------
# QuantumHScorer: import-error guard
# ---------------------------------------------------------------------------


class TestQiskitImportGuard:
    """Verify that a helpful ImportError is raised when qiskit is absent."""

    def test_import_error_raised_when_qiskit_unavailable(self, monkeypatch):
        import quantum_plumbing.quantum_interface as qi

        monkeypatch.setattr(qi, "_QISKIT_AVAILABLE", False)
        with pytest.raises(ImportError, match=r"(?i)qiskit"):
            QuantumHScorer(4)

    def test_import_error_message_mentions_install_command(self, monkeypatch):
        import quantum_plumbing.quantum_interface as qi

        monkeypatch.setattr(qi, "_QISKIT_AVAILABLE", False)
        with pytest.raises(ImportError, match="quantum-plumbing\\[quantum\\]"):
            QuantumHScorer(4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
