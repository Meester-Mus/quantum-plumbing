"""
Quantum hardware interface for Quantum Plumbing.

Maps classical H (hypothesis space) to/from Qiskit quantum circuits,
enabling quantum interference for hypothesis scoring without measurement
collapse.

Classical ↔ Quantum mapping (from ARCHITECTURE.md)::

    H[i]       ↔  |i⟩          (hypothesis ↔ basis state)
    scores[i]  ↔  |αᵢ|²        (probability amplitude squared)
    norm(H[i]) ↔  |αᵢ|         (amplitude magnitude)

Instead of the classical ``softmax(norm(H))`` scoring used by
:class:`~quantum_plumbing.layers.PotentialFCLayer`, the quantum scorer:

1. Encodes per-sample hypothesis norms as quantum amplitudes.
2. Applies Hadamard interference layers — hypotheses interact via
   quantum interference rather than independent softmax comparison.
3. Reads the resulting probability distribution directly from the
   statevector (no measurement collapse — full information preserved).
4. Returns these probabilities as hypothesis scores.

Requires: ``qiskit`` (install with ``pip install 'quantum-plumbing[quantum]'``)
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch

from ._potential_ops import actualize_h
from .layers import PotentialFCLayer

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    _QISKIT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _QISKIT_AVAILABLE = False


def _require_qiskit() -> None:
    """Raise a clear ImportError if Qiskit is not installed."""
    if not _QISKIT_AVAILABLE:  # pragma: no cover
        raise ImportError(
            "Qiskit is required for quantum interface features. "
            "Install it with:  pip install 'quantum-plumbing[quantum]'"
        )


class QuantumHScorer:
    """
    Score H hypotheses using quantum state simulation via Qiskit.

    Hypothesis norms are encoded as quantum amplitudes, Hadamard
    interference is applied, and the resulting statevector probabilities
    become the hypothesis scores.  Unlike classical ``softmax(norm(H))``,
    quantum scoring lets hypotheses *interact* before probability
    assignment: constructive interference boosts similar hypotheses while
    destructive interference suppresses conflicting ones.

    The statevector is read directly (no measurement), so no information
    is lost to wavefunction collapse.

    Args:
        num_potentials:        Number of hypotheses (size of H first dim).
        n_interference_layers: Number of Hadamard interference passes.
                               1 = single interference (default), >1 adds
                               deeper mixing between hypotheses.
        backend:               Optional Qiskit backend for hardware
                               execution.  ``None`` (default) uses the
                               built-in :class:`~qiskit.quantum_info.Statevector`
                               simulator — exact amplitudes, no collapse.
        shots:                 Number of measurement shots when a hardware
                               backend is provided.  Ignored for the
                               statevector simulator.
    """

    def __init__(
        self,
        num_potentials: int,
        n_interference_layers: int = 1,
        backend=None,
        shots: int = 1024,
    ) -> None:
        _require_qiskit()
        if num_potentials < 1:
            raise ValueError(f"num_potentials must be >= 1, got {num_potentials}")
        self._num_potentials = num_potentials
        self.n_interference_layers = n_interference_layers
        self._backend = backend
        self._shots = shots
        # Smallest power of 2 ≥ num_potentials
        # max(1, ...) guards against num_potentials=1 where log2(1)=0
        self._n_qubits: int = max(1, math.ceil(math.log2(num_potentials)))
        self._state_size: int = 2**self._n_qubits
        self._interference_matrix: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, H: torch.Tensor) -> torch.Tensor:
        """
        Score H hypotheses using quantum interference.

        Per-sample hypothesis norms are encoded as quantum amplitudes,
        interference is applied via Hadamard gates, and the resulting
        probabilities are returned as scores.  Each column of the output
        is a valid probability distribution summing to 1.

        Args:
            H: Hypothesis tensor ``(num_potentials, batch_size, features)``.

        Returns:
            scores: ``(num_potentials, batch_size)`` — probability
            distribution over hypotheses for each sample.
        """
        batch_size = H.shape[1]
        device = H.device
        dtype = H.dtype

        # Per-sample L2 norms as initial amplitudes → (P, B)
        norms = (
            torch.norm(H.reshape(H.shape[0], H.shape[1], -1), p=2, dim=2)
            .detach()
            .cpu()
            .numpy()
        )
        scores_np = self._score_batch(norms[:, :batch_size])
        return torch.tensor(scores_np, dtype=dtype, device=device)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_batch(self, norms: np.ndarray) -> np.ndarray:
        if self._backend is None:
            return self._score_batch_statevector(norms)
        return self._score_batch_backend(norms)

    def _score_batch_statevector(self, norms: np.ndarray) -> np.ndarray:
        totals = np.linalg.norm(norms, axis=0)
        amplitudes = np.divide(
            norms,
            totals[np.newaxis, :],
            out=np.zeros_like(norms, dtype=np.float64),
            where=totals[np.newaxis, :] > 1e-10,
        )
        zero_cols = totals < 1e-10

        state = np.zeros((self._state_size, norms.shape[1]), dtype=np.float64)
        state[: self._num_potentials] = amplitudes

        if self._interference_matrix is None:
            base = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.float64) / math.sqrt(
                2.0
            )
            matrix = base
            for _ in range(self._n_qubits - 1):
                matrix = np.kron(matrix, base)
            self._interference_matrix = np.linalg.matrix_power(
                matrix, self.n_interference_layers
            )

        mixed = self._interference_matrix @ state
        probs = mixed**2
        valid = probs[: self._num_potentials]
        valid_sums = valid.sum(axis=0)
        scores = np.divide(
            valid,
            valid_sums[np.newaxis, :],
            out=np.zeros_like(valid),
            where=valid_sums[np.newaxis, :] > 1e-10,
        )
        if np.any(zero_cols):
            scores[:, zero_cols] = 1.0 / self._num_potentials
        return scores

    def _score_batch_backend(self, norms: np.ndarray) -> np.ndarray:
        circuits = []
        zero_cols = []
        for column in range(norms.shape[1]):
            norm_vector = norms[:, column]
            total = float(np.linalg.norm(norm_vector))
            if total < 1e-10:
                zero_cols.append(column)
                circuits.append(None)
                continue
            amplitudes = norm_vector / total
            state = np.zeros(self._state_size, dtype=np.float64)
            state[: self._num_potentials] = amplitudes
            state = state / float(np.linalg.norm(state))
            circuits.append(self._build_interference_circuit(state))

        valid_circuits = [qc for qc in circuits if qc is not None]
        probabilities = []
        if valid_circuits:
            from qiskit import transpile

            measured = []
            for qc in valid_circuits:
                qc_m = qc.copy()
                qc_m.measure_all()
                measured.append(qc_m)
            transpiled = transpile(measured, self._backend)
            job = self._backend.run(transpiled, shots=self._shots)
            for counts in job.result().get_counts():
                probs = np.zeros(self._state_size, dtype=np.float64)
                total_shots = sum(counts.values())
                for bitstring, count in counts.items():
                    idx = int(bitstring.replace(" ", ""), 2)
                    if idx < self._state_size:
                        probs[idx] = count / total_shots
                probabilities.append(probs)

        scores = np.zeros((self._num_potentials, norms.shape[1]), dtype=np.float64)
        prob_index = 0
        for column, qc in enumerate(circuits):
            if qc is None:
                scores[:, column] = 1.0 / self._num_potentials
                continue
            probs = probabilities[prob_index]
            prob_index += 1
            valid = probs[: self._num_potentials]
            valid_sum = float(valid.sum())
            scores[:, column] = (
                valid / valid_sum if valid_sum > 1e-10 else 1.0 / self._num_potentials
            )
        return scores

    def _score_vector(self, norms: np.ndarray) -> np.ndarray:
        """
        Score a single hypothesis norm vector via a quantum circuit.

        Steps:

        1. Normalize norms to a unit-amplitude vector.
        2. Pad to the next power of two (``2^n_qubits``).
        3. Prepare quantum state |ψ⟩ = Σᵢ αᵢ|i⟩.
        4. Apply ``n_interference_layers`` Hadamard passes.
        5. Read probabilities from statevector (no measurement collapse).
        6. Trim to ``num_potentials`` entries and renormalize.

        Args:
            norms: ``(num_potentials,)`` array of L2 norms (≥ 0).

        Returns:
            ``(num_potentials,)`` probability array summing to 1.
        """
        total = float(np.linalg.norm(norms))
        if total < 1e-10:
            # Degenerate (all-zero) input: uniform distribution
            return (
                np.ones(self._num_potentials, dtype=np.float64) / self._num_potentials
            )

        amplitudes = norms / total

        # Pad to state_size = 2^n_qubits
        state = np.zeros(self._state_size, dtype=np.float64)
        state[: self._num_potentials] = amplitudes

        # Renormalize after padding (padding region absorbs some probability)
        state_norm = float(np.linalg.norm(state))
        if state_norm > 1e-10:
            state = state / state_norm

        qc = self._build_interference_circuit(state)
        probs = self._simulate(qc)  # (state_size,)

        # Trim to valid potentials and renormalize
        valid = probs[: self._num_potentials]
        valid_sum = float(valid.sum())
        if valid_sum > 1e-10:
            return valid / valid_sum
        return np.ones(self._num_potentials, dtype=np.float64) / self._num_potentials

    def _build_interference_circuit(self, state: np.ndarray) -> "QuantumCircuit":
        """
        Build a Qiskit circuit that amplitude-encodes *state* and applies
        Hadamard interference.

        Args:
            state: Normalised real amplitude vector of length ``2^n_qubits``.

        Returns:
            :class:`~qiskit.QuantumCircuit` ready for simulation.
        """
        qc = QuantumCircuit(self._n_qubits)
        # Amplitude encoding: |ψ⟩ = Σᵢ state[i] |i⟩
        qc.initialize(state.tolist(), qubits=list(range(self._n_qubits)))
        # Hadamard interference layers — each pass mixes all basis states
        for _ in range(self.n_interference_layers):
            qc.h(list(range(self._n_qubits)))
        return qc

    def _simulate(self, qc: "QuantumCircuit") -> np.ndarray:
        """
        Simulate *qc* and return measurement probabilities.

        * **Default (no backend)**: uses :class:`~qiskit.quantum_info.Statevector`
          for exact amplitudes — no measurement, no collapse.
        * **Hardware backend**: transpiles and runs with ``self._shots`` shots,
          converting counts to empirical probabilities.

        Args:
            qc: QuantumCircuit (no measurements appended yet).

        Returns:
            Probability array of length ``2^n_qubits``.
        """
        if self._backend is None:
            # Exact statevector simulation — full information preserved
            sv = Statevector(qc)
            return np.asarray(sv.probabilities(), dtype=np.float64)

        # Hardware / shots-based execution
        from qiskit import transpile  # local import: only needed with backend

        qc_m = qc.copy()
        qc_m.measure_all()
        transpiled = transpile(qc_m, self._backend)
        job = self._backend.run(transpiled, shots=self._shots)
        counts = job.result().get_counts()

        probs = np.zeros(self._state_size, dtype=np.float64)
        total_shots = sum(counts.values())
        for bitstring, count in counts.items():
            # Qiskit may insert spaces in bitstrings when multiple registers
            # are present (e.g. "0 1" instead of "01"); strip them for safety.
            idx = int(bitstring.replace(" ", ""), 2)
            if idx < self._state_size:
                probs[idx] = count / total_shots
        return probs

    def __repr__(self) -> str:
        return (
            f"QuantumHScorer("
            f"num_potentials={self._num_potentials}, "
            f"n_qubits={self._n_qubits}, "
            f"n_interference_layers={self.n_interference_layers}, "
            f"backend={self._backend!r})"
        )


class QuantumPotentialFCLayer(PotentialFCLayer):
    """
    Fully Connected layer with quantum-scored hypothetical space.

    Extends :class:`~quantum_plumbing.layers.PotentialFCLayer` by replacing
    the classical ``softmax(norm(H))`` scoring step with a Qiskit quantum
    circuit.  Hypothesis norms are encoded as quantum amplitudes; Hadamard
    interference layers re-weight them; the resulting statevector
    probabilities become the hypothesis scores.

    The quantum scoring is **detached** from the computational graph: only
    the score *values* (not gradients) come from the quantum circuit, so
    gradients still flow through H values (the weight matrices) exactly as
    in the classical layer.

    This implements the Classical ↔ Quantum interface from ARCHITECTURE.md::

        H[i]       ↔  |i⟩      (hypothesis ↔ basis state)
        scores[i]  ↔  |αᵢ|²    (probability amplitude squared)
        norm(H[i]) ↔  |αᵢ|     (amplitude magnitude)

    Args:
        in_features:           Input feature dimension.
        out_features:          Output feature dimension.
        num_potentials:        Number of hypotheses (size of H).
        bias:                  Whether FC layers use bias terms.
        n_interference_layers: Hadamard interference depth (default 1).
        backend:               Optional Qiskit backend.  ``None`` → exact
                               :class:`~qiskit.quantum_info.Statevector`
                               simulation (no measurement collapse).
        shots:                 Shots for hardware backends; ignored for the
                               statevector simulator.

    Example::

        layer = QuantumPotentialFCLayer(784, 128, num_potentials=8)
        output, H = layer(x)   # H scored via quantum interference
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_potentials: int = 8,
        bias: bool = True,
        n_interference_layers: int = 1,
        backend=None,
        shots: int = 1024,
    ) -> None:
        super().__init__(in_features, out_features, num_potentials, bias)
        self.quantum_scorer = QuantumHScorer(
            num_potentials,
            n_interference_layers=n_interference_layers,
            backend=backend,
            shots=shots,
        )
        self._layer_type = "QuantumFC"

    def forward(
        self,
        x: torch.Tensor,
        prev_H: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with quantum hypothesis scoring.

        H generation and actualization are identical to
        :class:`~quantum_plumbing.layers.PotentialFCLayer`, but scores are
        derived from quantum circuit interference rather than classical
        ``softmax(norm(H))``.

        Args:
            x:      Input tensor ``(batch_size, in_features)``.
            prev_H: Previous layer's H — accepted for API compatibility
                    with :class:`~quantum_plumbing.layers.PotentialFCLayer`
                    but not used in the scoring step.

        Returns:
            output: Actualized output ``(batch_size, out_features)``.
            H:      Hypothetical space ``(num_potentials, batch_size, out_features)``.
        """
        # ---- STEP 1: Generate H (identical to classical layer) ----------
        H = self._generate_h(x, prev_H=prev_H)

        # ---- STEP 2: Score via quantum circuit --------------------------
        # Scores are detached from the gradient graph: gradients flow
        # through H values (weight_potentials), not through the scores.
        scores = self.quantum_scorer.score(H)  # (num_potentials, batch)

        # ---- STEP 3: Actualize ------------------------------------------
        output = actualize_h(H, scores)

        # ---- STEP 4: Attach metadata ------------------------------------
        H._is_potential = True
        H._layer = "QuantumFC"
        H._meaning = "Possible outputs scored via quantum interference"
        H._scores = scores

        return output, H

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"num_potentials={self.num_potentials}, "
            f"n_interference_layers={self.quantum_scorer.n_interference_layers}, "
            f"bias={self.bias_potentials is not None}"
        )
