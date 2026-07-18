from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .layers import (
    PotentialActivation,
    PotentialBatchNorm,
    PotentialDropout,
    PotentialEmbedding,
    PotentialFCLayer,
    PotentialLayerNorm,
    PotentialMultiheadAttention,
)

# Layers that require (x, H) as input (not first-layer FC layers)
_H_INPUT_LAYERS = (
    PotentialBatchNorm,
    PotentialDropout,
    PotentialActivation,
    PotentialLayerNorm,
)


class PotentialSequential(nn.Module):
    """
    Sequential container for Potential layers that preserves H (thinking space).

    Works like nn.Sequential but routes H through each layer automatically.
    The first PotentialFCLayer is called with only x; every subsequent layer
    receives both x and H.

    Example::

        net = PotentialSequential(
            PotentialFCLayer(784, 256, num_potentials=8),
            PotentialBatchNorm(256),
            PotentialDropout(0.1),
            PotentialActivation('relu'),
            PotentialFCLayer(256, 10, num_potentials=8),
        )
        output, H = net(x)
    """

    def __init__(self, *layers: nn.Module):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass threading (x, H) through all layers.

        Args:
            x: Input tensor (batch_size, in_features)

        Returns:
            output: Final actualized output (batch_size, out_features)
            H: Final hypothetical space (num_potentials, batch_size, out_features)
        """
        H: Optional[torch.Tensor] = None

        for layer in self.layers:
            if isinstance(layer, PotentialMultiheadAttention):
                x, H = layer(x, H)
            elif isinstance(layer, _H_INPUT_LAYERS):
                if H is None:
                    raise RuntimeError(
                        f"{type(layer).__name__} requires H from a preceding "
                        "PotentialFCLayer but no H has been produced yet."
                    )
                x, H = layer(x, H)
            elif isinstance(layer, PotentialEmbedding):
                x, H = layer(x)
            elif isinstance(layer, PotentialFCLayer):
                x, H = layer(x, prev_H=H)
            else:
                # Plain nn.Module (e.g. a standard layer inserted for compatibility)
                x = layer(x)

        if H is None:
            raise RuntimeError(
                "No PotentialFCLayer found in the network – H was never produced."
            )

        return x, H

    def __repr__(self) -> str:
        layer_str = "\n".join(
            f"  ({i}): {layer}" for i, layer in enumerate(self.layers)
        )
        return f"PotentialSequential(\n{layer_str}\n)"


class PotentialTransformerBlock(nn.Module):
    """Transformer block with potential attention and potential feed-forward layers."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: int = 4,
        num_potentials: int = 8,
        dropout_p: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        hidden_dim = embed_dim * mlp_ratio
        self.attention = PotentialMultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_potentials=num_potentials,
            dropout=dropout_p,
            batch_first=True,
        )
        self.norm1 = PotentialLayerNorm(embed_dim)
        self.ff1 = PotentialFCLayer(
            embed_dim, hidden_dim, num_potentials=num_potentials
        )
        self.activation = PotentialActivation(activation)
        self.ff2 = PotentialFCLayer(
            hidden_dim, embed_dim, num_potentials=num_potentials
        )
        self.norm2 = PotentialLayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        H: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_H = self.attention(
            x,
            H,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )
        x = x + attn_out
        x, attn_H = self.norm1(x, attn_H)
        ff_out, ff_H = self.ff1(x, prev_H=attn_H)
        ff_out, ff_H = self.activation(ff_out, ff_H)
        ff_out, ff_H = self.ff2(ff_out, prev_H=ff_H)
        x = x + ff_out
        x, ff_H = self.norm2(x, ff_H)
        return x, ff_H


def PotentialMLP(
    layer_sizes: List[int],
    num_potentials: int = 8,
    dropout_p: float = 0.1,
    activation: str = "relu",
    batch_norm: bool = True,
    bias: bool = True,
) -> PotentialSequential:
    """
    Build a fully-connected Potential MLP from a list of layer sizes.

    Each hidden block is: FC → [BatchNorm] → [Dropout] → Activation.
    The final layer is a plain FC (no BN/Dropout/Activation).

    Args:
        layer_sizes:    List of feature dimensions, e.g. [784, 256, 128, 10].
                        Must have at least 2 elements.
        num_potentials: Hypothetical space size for every FC layer.
        dropout_p:      Dropout probability; set to 0 to disable dropout.
        activation:     Activation type ('relu', 'tanh', 'sigmoid', 'elu').
        batch_norm:     Whether to include PotentialBatchNorm in hidden blocks.
        bias:           Whether FC layers use bias terms.

    Returns:
        A PotentialSequential network ready for training.

    Example::

        # Classifier on MNIST-like data
        model = PotentialMLP([784, 256, 128, 10], num_potentials=8)
        output, H = model(x)
    """
    if len(layer_sizes) < 2:
        raise ValueError("layer_sizes must have at least 2 elements.")

    layers: List[nn.Module] = []
    for i in range(len(layer_sizes) - 1):
        in_dim = layer_sizes[i]
        out_dim = layer_sizes[i + 1]
        is_last = i == len(layer_sizes) - 2

        layers.append(
            PotentialFCLayer(in_dim, out_dim, num_potentials=num_potentials, bias=bias)
        )

        if not is_last:
            if batch_norm:
                layers.append(PotentialBatchNorm(out_dim))
            if dropout_p > 0:
                layers.append(PotentialDropout(dropout_p))
            layers.append(PotentialActivation(activation))

    return PotentialSequential(*layers)


def QuantumMLP(
    layer_sizes: List[int],
    num_potentials: int = 8,
    dropout_p: float = 0.1,
    activation: str = "relu",
    batch_norm: bool = True,
    bias: bool = True,
    n_interference_layers: int = 1,
    backend=None,
    shots: int = 1024,
) -> PotentialSequential:
    """Build a fully-connected MLP using quantum-scored potential FC layers."""
    from .quantum_interface import QuantumPotentialFCLayer

    if len(layer_sizes) < 2:
        raise ValueError("layer_sizes must have at least 2 elements.")

    layers: List[nn.Module] = []
    for i in range(len(layer_sizes) - 1):
        in_dim = layer_sizes[i]
        out_dim = layer_sizes[i + 1]
        is_last = i == len(layer_sizes) - 2
        layers.append(
            QuantumPotentialFCLayer(
                in_dim,
                out_dim,
                num_potentials=num_potentials,
                bias=bias,
                n_interference_layers=n_interference_layers,
                backend=backend,
                shots=shots,
            )
        )
        if not is_last:
            if batch_norm:
                layers.append(PotentialBatchNorm(out_dim))
            if dropout_p > 0:
                layers.append(PotentialDropout(dropout_p))
            layers.append(PotentialActivation(activation))

    return PotentialSequential(*layers)
