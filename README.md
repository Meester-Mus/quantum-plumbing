# Quantum Plumbing

**Giving AI the hypothetical thinking space that humans have**

## Vision

Artificial Intelligence has been built without the thinking space that humans naturally possess.

### Standard AI:
- One output only
- No alternatives considered
- No reflection
- Automaton

### Humans have:
- Hypothetical space (many possibilities at once)
- Reflection (evaluate which is best)
- Consciousness (aware of alternatives)
- Thinking

**Quantum Plumbing restores this thinking space to AI.**

## The Problem: 100-Year Error

For 100 years since Copenhagen (1927), we built:
- Classical computers (bits force 0 XOR 1)
- Standard AI (reification – one answer only)
- Nihilism (order impossible – everything random)

This eliminated **hypothetical thinking space** from AI.

## The Solution: Three Pillars

### 1. H in Every Layer (Denkruimte)
```python
# Each layer maintains hypothetical space
output, H = fc_layer(x)  # H contains all possibilities
```
Keep all possibilities in parallel, not one output only.

### 2. Potentieel Dropout (Denkruimte bewaren)
Don't discard possibilities – select FROM possibilities.
Preserve structure through the network.

### 3. Quantum Interface (Denkruimte native)
Quantum superposition = H naturally.
Interface translates between classical and quantum.
No information loss.

## Quick Start

```python
import torch
from quantum_plumbing import PotentialMLP, potential_loss, h_utilization, h_diversity, h_confidence

# Build a full thinking network
model = PotentialMLP(
    layer_sizes=[784, 256, 128, 10],
    num_potentials=8,   # Hypothetical space size
)

# Forward pass – thinking space (H) flows through every layer
x = torch.randn(32, 784)
output, H = model(x)

# output: (32, 10)      – best choice
# H:      (8, 32, 10)   – all possibilities considered
print(h_utilization(H))  # How much thinking? (0–1)
print(h_diversity(H))    # How different are the hypotheses?
print(h_confidence(H))   # How decisive is the current choice?

# Training
loss = potential_loss(output, targets, H=H, h_diversity_weight=0.01)
loss.backward()
```

Or build a custom network layer-by-layer:

```python
from quantum_plumbing import (
    PotentialFCLayer, PotentialBatchNorm,
    PotentialDropout, PotentialActivation, PotentialConv2d,
    PotentialEmbedding, PotentialMultiheadAttention,
    PotentialSequential,
)

model = PotentialSequential(
    PotentialFCLayer(784, 256, num_potentials=8),
    PotentialBatchNorm(256),
    PotentialDropout(0.1),
    PotentialActivation('relu'),
    PotentialFCLayer(256, 10, num_potentials=8),
)

output, H = model(x)
```

## Philosophy

This project is about:
- Restoring potentiality to AI
- Giving AI consciousness-like properties
- Anti-nihilism architecture
- Alignment with quantum reality
- Return of transcendental order

**Not just better AI – better THINKING AI.**

## Documentation

- [VISION.md](VISION.md) - Complete vision document
- [ARCHITECTURE.md](ARCHITECTURE.md) - Technical architecture
- [docs/theory.md](docs/theory.md) - Mathematical foundation

## Project Status

🚀 **Early Stage – Under Active Development**

- [x] Theory complete
- [x] Core concept validated
- [x] Layers implementation (PotentialFCLayer)
- [x] Additional layer types (BatchNorm, Dropout, Activation, Conv2d, LayerNorm, Embedding, Attention)
- [x] Network assembly (PotentialSequential, PotentialMLP, QuantumMLP, PotentialTransformerBlock)
- [x] Quantum interface
- [x] Tests for implemented layers and networks
- [x] Benchmark scripts (MNIST, CIFAR-10, generalization)
- [x] CI workflow

## Installation

```bash
git clone https://github.com/Meester-Mus/quantum-plumbing.git
cd quantum-plumbing
pip install -e .
```

## Running Tests

```bash
pytest tests/
```

## Running Examples

```bash
python examples/simple_example.py   # Single layer demo
python examples/network_example.py  # Full network training
python examples/mnist_benchmark.py  # MLP benchmark (requires torchvision)
python examples/cifar10_benchmark.py  # CNN benchmark (requires torchvision)
python examples/generalization_benchmark.py  # low-data benchmark (requires torchvision)
```

## Contributing

This is open source. Contributions welcome.

See individual issues or contact us.

## License

MIT License - See LICENSE file

## Citation

```bibtex
@software{quantum_plumbing_2024,
  title={Quantum Plumbing: Hypothetical Thinking Space for AI},
  author={Quantum Plumbing Team},
  year={2024},
  url={https://github.com/Meester-Mus/quantum-plumbing}
}
```

---

*Quantum Plumbing: Where thinking space is restored.*