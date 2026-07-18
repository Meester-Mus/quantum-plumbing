#!/usr/bin/env python
"""
Network assembly example for Quantum Plumbing.

Demonstrates:
- Building a full Potential network with PotentialMLP
- Custom assembly with PotentialSequential
- Training loop with potential_loss
- Measuring h_utilization (thinking space usage) during training
"""

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
# Synthetic dataset
# ---------------------------------------------------------------------------


def make_dataset(n_samples=512, in_features=32, n_classes=4, seed=42):
    torch.manual_seed(seed)
    X = torch.randn(n_samples, in_features)
    y = torch.randint(0, n_classes, (n_samples,))
    return X, y


def make_loader(X, y, batch_size=32):
    dataset = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


# ---------------------------------------------------------------------------
# Example 1 – build with PotentialMLP (convenience builder)
# ---------------------------------------------------------------------------


def example_potential_mlp():
    print("=" * 60)
    print("Example 1: PotentialMLP builder")
    print("=" * 60)

    model = PotentialMLP(
        layer_sizes=[32, 64, 32, 4],
        num_potentials=8,
        dropout_p=0.1,
        activation="relu",
        batch_norm=True,
    )
    print(model)
    print()

    x = torch.randn(4, 32)
    output, H = model(x)
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  H shape:      {H.shape}  (num_potentials × batch × out_features)")
    print()


# ---------------------------------------------------------------------------
# Example 2 – custom assembly with PotentialSequential
# ---------------------------------------------------------------------------


def example_custom_sequential():
    print("=" * 60)
    print("Example 2: Custom PotentialSequential")
    print("=" * 60)

    model = PotentialSequential(
        PotentialFCLayer(32, 64, num_potentials=4),
        PotentialBatchNorm(64),
        PotentialActivation("relu"),
        PotentialFCLayer(64, 32, num_potentials=4),
        PotentialDropout(0.1),
        PotentialActivation("relu"),
        PotentialFCLayer(32, 4, num_potentials=4),
    )
    print(model)
    print()

    x = torch.randn(8, 32)
    model.eval()
    with torch.no_grad():
        output, H = model(x)
    print(f"  Output shape: {output.shape}")
    print(f"  H shape:      {H.shape}")
    print()


# ---------------------------------------------------------------------------
# Example 3 – full training loop
# ---------------------------------------------------------------------------


def example_training():
    print("=" * 60)
    print("Example 3: Full training loop")
    print("=" * 60)

    IN_FEATURES = 32
    N_CLASSES = 4
    EPOCHS = 10
    LR = 1e-3

    torch.manual_seed(0)
    X, y = make_dataset(n_samples=512, in_features=IN_FEATURES, n_classes=N_CLASSES)
    loader = make_loader(X, y, batch_size=64)

    model = PotentialMLP(
        layer_sizes=[IN_FEATURES, 64, 32, N_CLASSES],
        num_potentials=8,
        dropout_p=0.1,
        activation="relu",
    )
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    print(f"  {'Epoch':>5}  {'Loss':>8}  {'Acc':>7}  {'H-util':>8}")
    print("  " + "-" * 36)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        h_utils = []

        for xb, yb in loader:
            optimizer.zero_grad()

            output, H = model(xb)
            loss = potential_loss(
                output, yb, H=H, task="classification", h_diversity_weight=0.01
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            preds = output.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
            h_utils.append(h_utilization(H).item())

        avg_loss = total_loss / total
        acc = correct / total
        avg_util = sum(h_utils) / len(h_utils)

        print(f"  {epoch:>5}  {avg_loss:>8.4f}  {acc:>6.1%}  {avg_util:>8.4f}")

    print()
    print("  Training complete.")
    print()


# ---------------------------------------------------------------------------
# Example 4 – inspecting H at inference time
# ---------------------------------------------------------------------------


def example_inspect_h():
    print("=" * 60)
    print("Example 4: Inspecting H at inference time")
    print("=" * 60)

    model = PotentialMLP([32, 16, 4], num_potentials=8, dropout_p=0.0)
    model.eval()

    x = torch.randn(1, 32)
    with torch.no_grad():
        output, H = model(x)

    print(f"  Output (best choice): {output.shape}")
    print(f"  H (all possibilities): {H.shape}")
    print()

    # The output is a weighted combination of the 8 hypotheses
    scores = H._scores  # (num_potentials, batch)
    print("  Hypothesis scores (how likely each possibility is):")
    for i, s in enumerate(scores[:, 0].tolist()):
        bar = "█" * int(s * 40)
        print(f"    H[{i}]: {s:.4f}  {bar}")

    util = h_utilization(H)
    print()
    print(f"  H utilization: {util.item():.4f}  (1 = full thinking, 0 = no thinking)")
    print()

    # Show that output equals weighted sum of H
    reconstructed = torch.einsum("pb,pbo->bo", scores, H)
    print(
        f"  output ≈ weighted_sum(H):  "
        f"max_diff={torch.max(torch.abs(output - reconstructed)).item():.2e}"
    )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print()
    print("Quantum Plumbing – Network Assembly Examples")
    print()

    example_potential_mlp()
    example_custom_sequential()
    example_training()
    example_inspect_h()

    print("=" * 60)
    print("✓ Network assembly working – thinking space flows end-to-end!")
    print("=" * 60)
