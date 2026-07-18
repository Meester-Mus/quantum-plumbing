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
        
        # STEP 1: GENERATE H
        # Calculate all possible outputs with all possible weights
        H_list = []
        
        for i in range(self.num_potentials):
            # Get i-th potential weight matrix
            w_i = self.weight_potentials[i]  # (out, in)
            
            # Calculate output with these weights
            out_i = torch.nn.functional.linear(
                x,  # (batch, in)
                w_i,  # (out, in)
                self.bias_potentials[i] if self.bias_potentials is not None else None
            )
            # out_i: (batch, out)
            
            H_list.append(out_i)
        
        # Stack all possibilities into H
        # H: (num_potentials, batch, out)
        H = torch.stack(H_list, dim=0)
        
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