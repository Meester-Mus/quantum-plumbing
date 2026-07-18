"""
Quantum Plumbing: Hypothetical thinking space for AI

This package implements thinking space (H) in neural networks,
restoring the hypothetical reasoning that humans naturally possess.
"""

__version__ = "0.0.1"
__author__ = "Quantum Plumbing Team"
__email__ = "Mariussielcken@gmail.com"

from .layers import (
    PotentialActivation,
    PotentialBatchNorm,
    PotentialConv2d,
    PotentialDropout,
    PotentialEmbedding,
    PotentialFCLayer,
    PotentialLayerNorm,
    PotentialMultiheadAttention,
)
from .network import PotentialSequential, PotentialMLP, PotentialTransformerBlock, QuantumMLP
from .loss import potential_loss, h_confidence, h_diversity, h_utilization
from .quantum_interface import QuantumHScorer, QuantumPotentialFCLayer

__all__ = [
    # Layers
    'PotentialFCLayer',
    'PotentialBatchNorm',
    'PotentialDropout',
    'PotentialActivation',
    'PotentialConv2d',
    'PotentialLayerNorm',
    'PotentialEmbedding',
    'PotentialMultiheadAttention',
    # Quantum interface (requires qiskit)
    'QuantumHScorer',
    'QuantumPotentialFCLayer',
    # Network assembly
    'PotentialSequential',
    'PotentialMLP',
    'PotentialTransformerBlock',
    'QuantumMLP',
    # Loss and metrics
    'potential_loss',
    'h_utilization',
    'h_diversity',
    'h_confidence',
]