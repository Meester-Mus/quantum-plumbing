"""
Quantum Plumbing: Hypothetical thinking space for AI

This package implements thinking space (H) in neural networks,
restoring the hypothetical reasoning that humans naturally possess.
"""

__version__ = "0.0.1"
__author__ = "Quantum Plumbing Team"
__email__ = "Mariussielcken@gmail.com"

from .layers import PotentialFCLayer, PotentialBatchNorm, PotentialDropout, PotentialActivation
from .network import PotentialSequential, PotentialMLP
from .loss import potential_loss, h_utilization

__all__ = [
    # Layers
    'PotentialFCLayer',
    'PotentialBatchNorm',
    'PotentialDropout',
    'PotentialActivation',
    # Network assembly
    'PotentialSequential',
    'PotentialMLP',
    # Loss and metrics
    'potential_loss',
    'h_utilization',
]