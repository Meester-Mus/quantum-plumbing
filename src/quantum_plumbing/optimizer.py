from __future__ import annotations

from typing import Optional

import torch


class HAwareOptimizer:
    """
    Wrap a PyTorch optimizer and re-weight hypothesis-parameter gradients.

    Parameters whose first dimension matches the hypothesis count are scaled by
    the mean hypothesis scores before each optimizer step.
    """

    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer
        self._score_weights: Optional[torch.Tensor] = None

    def set_h_scores(self, scores: torch.Tensor) -> None:
        """Set per-hypothesis scores from the latest forward pass."""
        self._score_weights = scores.detach().mean(dim=1)

    def zero_grad(self, *args, **kwargs) -> None:
        self.optimizer.zero_grad(*args, **kwargs)

    def step(self, closure=None):
        if self._score_weights is not None:
            for group in self.optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is None or param.dim() == 0:
                        continue
                    if param.shape[0] != self._score_weights.shape[0]:
                        continue
                    scale = self._score_weights.to(param.grad.device, param.grad.dtype)
                    view_shape = (scale.shape[0], *([1] * (param.grad.dim() - 1)))
                    param.grad.mul_(scale.view(view_shape))
        return self.optimizer.step(closure=closure)
