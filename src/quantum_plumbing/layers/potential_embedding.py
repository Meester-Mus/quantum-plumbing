from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .._potential_ops import actualize_h, compute_h_scores, mark_hypothesis_parameter


class PotentialEmbedding(nn.Module):
    """Embedding layer that emits actualized tokens plus hypothetical embeddings."""

    def __init__(
        self, num_embeddings: int, embedding_dim: int, num_potentials: int = 8
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_potentials = num_potentials
        self.weight_potentials = mark_hypothesis_parameter(
            nn.Parameter(
                torch.randn(num_potentials, num_embeddings, embedding_dim)
                / (embedding_dim**0.5)
            )
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        H = torch.stack(
            [
                F.embedding(x, self.weight_potentials[i])
                for i in range(self.num_potentials)
            ],
            dim=0,
        )
        scores = compute_h_scores(H)
        output = actualize_h(H, scores)
        H._is_potential = True
        H._layer = "Embedding"
        H._meaning = "Possible token embeddings"
        H._scores = scores
        return output, H

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}, "
            f"num_potentials={self.num_potentials}"
        )
