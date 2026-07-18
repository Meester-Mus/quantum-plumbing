from __future__ import annotations

from typing import Tuple, Union

import torch
import torch.nn as nn


class PotentialLayerNorm(nn.Module):
    """Layer normalization that keeps H aligned with the actualized path."""

    def __init__(
        self,
        normalized_shape: Union[int, Tuple[int, ...]],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        self.normalized_shape = (
            (normalized_shape,)
            if isinstance(normalized_shape, int)
            else tuple(normalized_shape)
        )
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(
        self, x: torch.Tensor, H: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        normalized_dims = tuple(range(H.dim() - len(self.normalized_shape), H.dim()))
        reduce_dims = (0, *normalized_dims)
        mean = H.mean(dim=reduce_dims, keepdim=True)
        var = H.var(dim=reduce_dims, unbiased=False, keepdim=True)
        x_norm = (x - mean.squeeze(0)) / torch.sqrt(var.squeeze(0) + self.eps)
        if self.elementwise_affine:
            x_norm = x_norm * self.weight + self.bias

        H_norm = (H - mean) / torch.sqrt(var + self.eps)
        if self.elementwise_affine:
            affine_shape = (1,) * (
                H.dim() - len(self.normalized_shape)
            ) + self.normalized_shape
            H_norm = H_norm * self.weight.view(affine_shape) + self.bias.view(
                affine_shape
            )

        H_norm._is_potential = True
        H_norm._layer = "LayerNorm"
        H_norm._meaning = "Hypotheses after potential layer normalization"
        return x_norm, H_norm

    def extra_repr(self) -> str:
        return (
            f"normalized_shape={self.normalized_shape}, eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}"
        )
