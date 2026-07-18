"""
Quantum Plumbing: Hypothetical thinking space for AI

This package implements thinking space (H) in neural networks,
restoring the hypothetical reasoning that humans naturally possess.
"""

__version__ = "0.0.1"
__author__ = "Quantum Plumbing Team"
__email__ = "Mariussielcken@gmail.com"

from .layers import PotentialFCLayer, PotentialBatchNorm, PotentialDropout, PotentialActivation

__all__ = [
    'PotentialFCLayer',
    'PotentialBatchNorm',
    'PotentialDropout',
    'PotentialActivation',
]