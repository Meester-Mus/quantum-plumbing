#!/usr/bin/env python
"""
Quantum hardware interface example.

Demonstrates QuantumHScorer and QuantumPotentialFCLayer — the Qiskit
integration that replaces classical norm-based hypothesis scoring with
quantum circuit interference.

Run with:
    pip install 'quantum-plumbing[quantum]'
    python examples/quantum_example.py
"""

import torch

try:
    from quantum_plumbing import (
        QuantumHScorer,
        QuantumPotentialFCLayer,
        PotentialFCLayer,
        PotentialSequential,
        PotentialActivation,
        h_utilization,
    )
except ImportError as e:
    raise SystemExit(
        f"Install requirements: pip install 'quantum-plumbing[quantum]'\n{e}"
    )


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main() -> None:
    torch.manual_seed(42)

    # ------------------------------------------------------------------
    # 1. QuantumHScorer: stand-alone quantum scoring
    # ------------------------------------------------------------------
    section("1. QuantumHScorer — stand-alone quantum scoring")

    scorer = QuantumHScorer(num_potentials=4, n_interference_layers=1)
    print(f"  {scorer}")

    H = torch.randn(4, 8, 10)  # (num_potentials, batch, features)
    scores = scorer.score(H)

    print(f"\n  H shape  : {H.shape}  (num_potentials, batch, features)")
    print(f"  scores   : {scores.shape}  (num_potentials, batch)")
    print(f"  col sums : {scores.sum(dim=0).tolist()[:4]} ...")
    print("  → Each column is a valid probability distribution")

    # ------------------------------------------------------------------
    # 2. Classical vs quantum scoring comparison
    # ------------------------------------------------------------------
    section("2. Classical vs quantum scores for the same H")

    torch.manual_seed(0)
    classical_layer = PotentialFCLayer(10, 5, num_potentials=4)
    quantum_layer = QuantumPotentialFCLayer(10, 5, num_potentials=4)

    # Share weights — only the scoring method differs
    quantum_layer.weight_potentials.data = (
        classical_layer.weight_potentials.data.clone()
    )
    quantum_layer.bias_potentials.data = classical_layer.bias_potentials.data.clone()

    x = torch.randn(4, 10)
    _, H_c = classical_layer(x)
    _, H_q = quantum_layer(x)

    print("\n  Sample 0 — hypothesis scores:")
    print(f"    Classical (softmax norm) : {H_c._scores[:, 0].detach().tolist()}")
    print(f"    Quantum (interference)   : {H_q._scores[:, 0].tolist()}")
    print("  → Quantum interference redistributes scores via basis-state mixing")

    # ------------------------------------------------------------------
    # 3. QuantumPotentialFCLayer in a full network
    # ------------------------------------------------------------------
    section("3. QuantumPotentialFCLayer inside PotentialSequential")

    net = PotentialSequential(
        QuantumPotentialFCLayer(20, 16, num_potentials=4),
        PotentialActivation("relu"),
        QuantumPotentialFCLayer(16, 8, num_potentials=4),
    )

    x = torch.randn(6, 20)
    output, H = net(x)

    print(f"\n  Input  : {x.shape}")
    print(f"  Output : {output.shape}")
    print(f"  H      : {H.shape}")
    print(f"  H util : {h_utilization(H):.4f}  (0=collapsed, 1=full thinking space)")

    # ------------------------------------------------------------------
    # 4. Gradient flow
    # ------------------------------------------------------------------
    section("4. Gradient flow through quantum-scored layer")

    layer = QuantumPotentialFCLayer(10, 5, num_potentials=4)
    x = torch.randn(4, 10, requires_grad=True)
    output, _ = layer(x)
    output.sum().backward()

    print("\n  Quantum scores are detached — gradients still flow through H values:")
    print(f"    weight_potentials.grad : {layer.weight_potentials.grad.norm():.4f}")
    print(f"    x.grad                 : {x.grad.norm():.4f}")
    print("  ✓ Training loop works correctly with quantum scoring")

    # ------------------------------------------------------------------
    # 5. Quantum interface mapping summary
    # ------------------------------------------------------------------
    section("5. Classical ↔ Quantum interface mapping")
    print("""
  Classical (Potential AI)    Quantum
  ─────────────────────────   ─────────────────────────────
  H[i]                    ↔   |i⟩         (basis state i)
  norm(H[i])              ↔   |αᵢ|        (amplitude magnitude)
  scores[i]               ↔   |αᵢ|²       (measurement probability)
  Hadamard interference   ↔   amplitude mixing before read-out
  statevector read-out    ↔   no collapse — full info preserved
    """)

    print("✓ Quantum hardware interface ready.")
    print("  Connect a real backend: QuantumPotentialFCLayer(..., backend=my_backend)")


if __name__ == "__main__":
    main()
