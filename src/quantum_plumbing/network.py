import inspect
import warnings
import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from .layers import (
    PotentialActivation,
    PotentialBatchNorm,
    PotentialDropout,
    PotentialFCLayer,
)

_H_INPUT_PARAM_NAMES = {"H"}
_H_GENERATOR_PARAM_NAMES = {"prev_H"}


def _layer_call_mode(layer: nn.Module) -> str:
    """
    Infer how a layer wants H routed through its forward signature.

    Modes:
    - ``x_only``:    forward(x)
    - ``x_with_h``:  forward(x, H)
    - ``x_with_prev_h``: forward(x, prev_H=None)

    Returns:
        One of ``"x_only"``, ``"x_with_h"``, or ``"x_with_prev_h"``.
    """
    try:
        param_names = set(inspect.signature(layer.forward).parameters)
    except (AttributeError, TypeError, ValueError):
        # Some built-in or compiled callables do not expose a Python signature.
        warnings.warn(
            f"Could not inspect forward signature for {type(layer).__name__}; "
            "treating it as an x-only layer.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "x_only"

    has_prev_h = bool(param_names & _H_GENERATOR_PARAM_NAMES)
    has_h = bool(param_names & _H_INPUT_PARAM_NAMES)

    if has_prev_h and has_h:
        raise ValueError(
            f"{type(layer).__name__} has an ambiguous forward signature: "
            "use either H or prev_H, not both."
        )
    if has_prev_h:
        return "x_with_prev_h"
    if has_h:
        return "x_with_h"
    return "x_only"


class PotentialSequential(nn.Module):
    """
    Sequential container for Potential layers that preserves H.

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

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass threading (x, H) through all layers.

        Args:
            x: Input tensor (batch_size, in_features)

        Returns:
            output: Final actualized output (batch_size, out_features)
            H: Final hypothetical space
                (num_potentials, batch_size, out_features)
        """
        H: Optional[torch.Tensor] = None

        for layer in self.layers:
            call_mode = _layer_call_mode(layer)

            if call_mode == "x_with_h":
                if H is None:
                    raise RuntimeError(
                        f"{type(layer).__name__} requires H from a preceding "
                        "PotentialFCLayer but no H has been produced yet."
                    )
                x, H = layer(x, H)
            elif call_mode == "x_with_prev_h":
                x, H = layer(x, prev_H=H)
            else:
                # Plain nn.Module inserted for compatibility.
                x = layer(x)

        if H is None:
            raise RuntimeError(
                "No layer in the network produced H. Ensure at least one "
                "layer accepts prev_H or returns H."
            )

        return x, H

    def __repr__(self) -> str:
        layer_str = "\n".join(
            f"  ({i}): {layer}" for i, layer in enumerate(self.layers)
        )
        return f"PotentialSequential(\n{layer_str}\n)"


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
            PotentialFCLayer(
                in_dim,
                out_dim,
                num_potentials=num_potentials,
                bias=bias,
            )
        )

        if not is_last:
            if batch_norm:
                layers.append(PotentialBatchNorm(out_dim))
            if dropout_p > 0:
                layers.append(PotentialDropout(dropout_p))
            layers.append(PotentialActivation(activation))

    return PotentialSequential(*layers)
