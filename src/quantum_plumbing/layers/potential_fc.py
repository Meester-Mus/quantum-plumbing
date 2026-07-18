from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .._potential_ops import actualize_h, compute_h_scores, mark_hypothesis_parameter


class PotentialFCLayer(nn.Module):
    """
    Fully Connected layer with hypothetical thinking space.

    This layer implements H (hypothesis space) - maintaining all possible
    outputs in parallel, rather than just the single best output.

    Key difference from standard FC:
    - Standard: One set of weights → one output
    - Potential: Multiple weight sets → multiple outputs (H) → actualize best

    This gives the network hypothetical thinking space - it knows about
    alternatives even when choosing the best.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_potentials: int = 8,
        bias: bool = True,
        use_prev_h: bool = True,
    ):
        """
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            num_potentials: Number of possible weight matrices (size of H)
            bias: Whether to use bias terms
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_potentials = num_potentials
        self.use_prev_h = use_prev_h

        # CORE: Multiple weight matrices - this IS the thinking space
        # Shape: (num_potentials, out_features, in_features)
        # Meaning: num_potentials different possible transformations
        self.weight_potentials = mark_hypothesis_parameter(
            nn.Parameter(
                torch.randn(num_potentials, out_features, in_features)
                / (in_features**0.5)
            )
        )
        self.prev_h_projections = mark_hypothesis_parameter(
            nn.Parameter(
                torch.randn(num_potentials, out_features, in_features)
                / (in_features**0.5)
            )
        )

        if bias:
            self.bias_potentials = mark_hypothesis_parameter(
                nn.Parameter(torch.randn(num_potentials, out_features) * 0.01)
            )
        else:
            self.register_parameter("bias_potentials", None)

        # Metadata
        self._is_potential_layer = True
        self._layer_type = "FC"

    def _context_from_prev_h(
        self,
        prev_H: Optional[torch.Tensor],
        index: int,
    ) -> Optional[torch.Tensor]:
        if prev_H is None or not self.use_prev_h:
            return None
        if prev_H.shape[0] != self.num_potentials:
            raise ValueError(
                "prev_H first dimension must match "
                f"num_potentials={self.num_potentials}, "
                f"got {prev_H.shape[0]}"
            )
        if prev_H.shape[-1] != self.in_features:
            raise ValueError(
                f"prev_H last dimension must match in_features={self.in_features}, "
                f"got {prev_H.shape[-1]}"
            )
        return F.linear(prev_H[index], self.prev_h_projections[index])

    def _generate_h(
        self,
        x: torch.Tensor,
        prev_H: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        H_list = []
        for i in range(self.num_potentials):
            out_i = F.linear(
                x,
                self.weight_potentials[i],
                self.bias_potentials[i] if self.bias_potentials is not None else None,
            )
            prev_context = self._context_from_prev_h(prev_H, i)
            if prev_context is not None:
                out_i = out_i + prev_context
            H_list.append(out_i)
        return torch.stack(H_list, dim=0)

    def forward(
        self, x: torch.Tensor, prev_H: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with thinking space (H) preservation.

        Args:
            x: Input tensor (batch_size, in_features)
            prev_H: Previous layer's H (num_potentials, batch_size, in_features)

        Returns:
            output: Best choice (batch_size, out_features)
            H: All possibilities (num_potentials, batch_size, out_features)
        """

        # STEP 1: GENERATE H
        # Calculate all possible outputs with all possible weights
        H = self._generate_h(x, prev_H=prev_H)

        # STEP 2: EVALUATE H
        # Which possibilities are stronger/weaker?
        # For now: use norm (later: quantum interference)
        scores = compute_h_scores(H)

        # STEP 3: ACTUALIZE
        # Choose best via weighted combination
        # This gives: "I think this is best - but I know of others"
        output = actualize_h(H, scores)

        # STEP 4: PRESERVE H FOR NEXT LAYER
        # Metadata - this IS thinking space
        H._is_potential = True
        H._layer = "FC"
        H._meaning = "Possible outputs from this FC layer"
        H._scores = scores  # How probable is each?

        return output, H

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"num_potentials={self.num_potentials}, "
            f"bias={self.bias_potentials is not None}, "
            f"use_prev_h={self.use_prev_h}"
        )
