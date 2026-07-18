import torch
import torch.nn as nn
from typing import Tuple, Optional


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
    
    def __init__(self, 
                 in_features: int, 
                 out_features: int,
                 num_potentials: int = 8,
                 bias: bool = True):
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
        
        # CORE: Multiple weight matrices - this IS the thinking space
        # Shape: (num_potentials, out_features, in_features)
        # Meaning: num_potentials different possible transformations
        self.weight_potentials = nn.Parameter(
            torch.randn(num_potentials, out_features, in_features) / (in_features ** 0.5)
        )
        
        if bias:
            self.bias_potentials = nn.Parameter(
                torch.randn(num_potentials, out_features) * 0.01
            )
        else:
            self.register_parameter('bias_potentials', None)
        
        # Metadata
        self._is_potential_layer = True
        self._layer_type = "FC"
    
    def forward(self, x: torch.Tensor, 
                prev_H: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with thinking space (H) preservation.
        
        Args:
            x: Input tensor (batch_size, in_features)
            prev_H: Previous layer's H (num_potentials, batch_size, in_features)
        
        Returns:
            output: Best choice (batch_size, out_features)
            H: All possibilities (num_potentials, batch_size, out_features)
        """
        if x.dim() != 2:
            raise ValueError(f"x must be 2D (batch, features), got shape {tuple(x.shape)}")
        if x.shape[1] != self.in_features:
            raise ValueError(
                f"Expected x to have {self.in_features} features, got {x.shape[1]}"
            )
        if prev_H is not None:
            if prev_H.dim() != 3:
                raise ValueError(
                    "prev_H must be 3D (num_potentials, batch, features), "
                    f"got shape {tuple(prev_H.shape)}"
                )
            if prev_H.shape[1] != x.shape[0]:
                raise ValueError(
                    "prev_H batch size must match x batch size: "
                    f"prev_H={tuple(prev_H.shape)}, x={tuple(x.shape)}"
                )

        # STEP 1: GENERATE H
        # Calculate all possible outputs with all possible weights in one
        # vectorized operation (no Python loop over potentials).
        #
        # weight_potentials: (num_potentials, out_features, in_features)
        # x:                 (batch_size, in_features)
        # H:                 (num_potentials, batch_size, out_features)
        #
        # einsum index legend:
        #   b = batch_size, i = in_features, p = num_potentials, o = out_features
        H = torch.einsum('bi,poi->pbo', x, self.weight_potentials)
        if self.bias_potentials is not None:
            # bias_potentials: (num_potentials, out_features)
            # unsqueeze → (num_potentials, 1, out_features) – broadcasts over batch
            H = H + self.bias_potentials.unsqueeze(1)
        
        # STEP 2: EVALUATE H
        # Which possibilities are stronger/weaker?
        # For now: use norm (later: quantum interference)
        scores = torch.norm(H, p=2, dim=2, keepdim=True)  # (num_pot, batch, 1)
        scores = torch.softmax(scores.squeeze(-1), dim=0)  # (num_pot, batch)
        
        # STEP 3: ACTUALIZE
        # Choose best via weighted combination
        # This gives: "I think this is best - but I know of others"
        output = torch.einsum('pb,pbo->bo', scores, H)
        # output: (batch, out)
        
        # STEP 4: PRESERVE H FOR NEXT LAYER
        # Metadata - this IS thinking space
        H._is_potential = True
        H._layer = "FC"
        H._meaning = "Possible outputs from this FC layer"
        H._scores = scores  # How probable is each?
        
        return output, H
    
    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, '
                f'out_features={self.out_features}, '
                f'num_potentials={self.num_potentials}, '
                f'bias={self.bias_potentials is not None}')