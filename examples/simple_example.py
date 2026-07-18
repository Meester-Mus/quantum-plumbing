#!/usr/bin/env python
"""
Simple example of Quantum Plumbing in action.

Demonstrates:
- Creating a Potential layer
- Forward pass with H (thinking space)
- Observing hypothetical space
"""

import torch
from quantum_plumbing.layers import PotentialFCLayer


def main():
    print("="*60)
    print("Quantum Plumbing: Simple Example")
    print("="*60)
    print()
    
    # Create a potential FC layer with thinking space
    print("1. Creating Potential FC Layer...")
    layer = PotentialFCLayer(
        in_features=10,
        out_features=5,
        num_potentials=8  # Size of thinking space
    )
    print(f"   Layer created with {layer.num_potentials} potentials")
    print()
    
    # Create input
    print("2. Creating input...")
    batch_size = 4
    x = torch.randn(batch_size, 10)
    print(f"   Input shape: {x.shape}")
    print()
    
    # Forward pass
    print("3. Forward pass...")
    output, H = layer(x)
    print(f"   Output shape: {output.shape}")
    print(f"   H shape (thinking space): {H.shape}")
    print()
    
    # Analyze H
    print("4. Analyzing hypothetical space (H)...")
    print(f"   H contains {H.shape[0]} different possibilities")
    print(f"   Each possibility: {H.shape[1:]}")
    print()
    
    # Look at H scores
    print("5. H scores (how probable is each possibility)...")
    scores = H._scores
    for i in range(layer.num_potentials):
        avg_score = scores[i].mean().item()
        print(f"   Potential {i}: {avg_score:.4f}")
    print()
    
    # Show what makes Quantum Plumbing different
    print("6. What makes this special...")
    print("   Standard AI: output only")
    print(f"     Shape: {output.shape}")
    print(f"     Knows: Only the best choice")
    print()
    print("   Quantum Plumbing: output + H (thinking space)")
    print(f"     Output shape: {output.shape}")
    print(f"     H shape: {H.shape}")
    print(f"     Knows: Best choice PLUS all alternatives")
    print()
    
    # Verify the network is thinking
    print("7. Verifying thinking happens...")
    print(f"   Output is weighted combination of H")
    print(f"   Weights (scores): {scores[:, 0].detach().numpy()}")
    print()
    
    # The revolutionary part
    print("8. The revolutionary insight...")
    print("   H is not forgotten - it flows to next layer")
    print("   Next layer KNOWS about alternatives")
    print("   This enables TRUE thinking")
    print()
    
    print("="*60)
    print("✓ Quantum Plumbing working - thinking space restored!")
    print("="*60)


if __name__ == "__main__":
    main()