from __future__ import annotations

from typing import Tuple

import torch


def _flatten_h_non_batch_dims(H: torch.Tensor) -> torch.Tensor:
    """Flatten all non-potential, non-batch dimensions into features."""
    if H.dim() < 3:
        raise ValueError(
            "H must have at least 3 dimensions: (potentials, batch, features...)"
        )
    return H.reshape(H.shape[0], H.shape[1], -1)


def compute_h_scores(H: torch.Tensor) -> torch.Tensor:
    """Compute per-batch hypothesis scores for any H tensor shape."""
    flat_H = _flatten_h_non_batch_dims(H)
    scores = torch.norm(flat_H, p=2, dim=2)
    return torch.softmax(scores, dim=0)


def actualize_h(H: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """Combine hypotheses into the actualized output using broadcasted scores."""
    view_shape: Tuple[int, ...] = (
        scores.shape[0],
        scores.shape[1],
        *([1] * (H.dim() - 2)),
    )
    return torch.sum(H * scores.view(view_shape), dim=0)


def mark_hypothesis_parameter(parameter: torch.nn.Parameter) -> torch.nn.Parameter:
    """Mark a parameter as belonging to the hypothesis-specific path."""
    parameter._is_hypothesis_parameter = True
    return parameter
