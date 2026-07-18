from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .._potential_ops import actualize_h, compute_h_scores, mark_hypothesis_parameter


class PotentialConv2d(nn.Module):
    """2D convolution with hypothetical thinking space."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        num_potentials: int = 8,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        use_prev_h: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_potentials = num_potentials
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_prev_h = use_prev_h

        weight_shape = (
            num_potentials,
            out_channels,
            in_channels // groups,
            *kernel_size,
        )
        scale = (in_channels * kernel_size[0] * kernel_size[1]) ** 0.5
        self.weight_potentials = mark_hypothesis_parameter(
            nn.Parameter(torch.randn(weight_shape) / scale)
        )
        self.prev_h_projections = mark_hypothesis_parameter(
            nn.Parameter(torch.randn(weight_shape) / scale)
        )
        if bias:
            self.bias_potentials = mark_hypothesis_parameter(
                nn.Parameter(torch.zeros(num_potentials, out_channels))
            )
        else:
            self.register_parameter("bias_potentials", None)

    def _conv(
        self, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        return F.conv2d(
            x,
            weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def forward(
        self, x: torch.Tensor, prev_H: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        H_list = []
        for i in range(self.num_potentials):
            out_i = self._conv(
                x,
                self.weight_potentials[i],
                self.bias_potentials[i] if self.bias_potentials is not None else None,
            )
            if prev_H is not None and self.use_prev_h:
                if prev_H.shape[0] != self.num_potentials:
                    raise ValueError("prev_H first dimension must match num_potentials")
                if prev_H.shape[2] != self.in_channels:
                    raise ValueError("prev_H channel dimension must match in_channels")
                out_i = out_i + self._conv(prev_H[i], self.prev_h_projections[i], None)
            H_list.append(out_i)

        H = torch.stack(H_list, dim=0)
        scores = compute_h_scores(H)
        output = actualize_h(H, scores)

        H._is_potential = True
        H._layer = "Conv2d"
        H._meaning = "Possible outputs from this convolution layer"
        H._scores = scores
        return output, H

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, num_potentials={self.num_potentials}, "
            f"use_prev_h={self.use_prev_h}"
        )
