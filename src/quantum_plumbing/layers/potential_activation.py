import torch
import torch.nn as nn
from typing import Tuple


class PotentialActivation(nn.Module):
    """
    Activation function that preserves hypothetical structure.
    
    Key difference from standard Activation:
    - Standard: Apply to x only (forget H)
    - Potential: Apply to both x and H (remember alternatives)
    
    This keeps hypothetical thinking space after activation.
    """
    
    def __init__(self, activation_type: str = 'relu'):
        """
        Args:
            activation_type: 'relu', 'tanh', 'sigmoid', etc.
        """
        super().__init__()
        self.activation_type = activation_type
        
        if activation_type == 'relu':
            self.activation = torch.relu
        elif activation_type == 'tanh':
            self.activation = torch.tanh
        elif activation_type == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation_type == 'elu':
            self.activation = torch.nn.functional.elu
        else:
            raise ValueError(f"Unknown activation: {activation_type}")
        
        self._is_potential_layer = True
        self._layer_type = f"Activation_{activation_type}"
    
    def forward(self, x: torch.Tensor, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass applying activation while preserving H.
        
        Args:
            x: Input (batch_size, features)
            H: Hypotheses (num_potentials, batch_size, features)
        
        Returns:
            x_act: Activated input
            H_act: Activated hypotheses
        """
        
        if x.dim() != 2:
            raise ValueError(f"x must be 2D (batch, features), got shape {tuple(x.shape)}")
        if H.dim() != 3:
            raise ValueError(
                f"H must be 3D (num_potentials, batch, features), got shape {tuple(H.shape)}"
            )
        if x.shape[0] != H.shape[1] or x.shape[1] != H.shape[2]:
            raise ValueError(
                "x and H feature/batch dimensions must match: "
                f"x={tuple(x.shape)}, H={tuple(H.shape)}"
            )

        # Apply activation to both
        x_act = self.activation(x)
        H_act = self.activation(H)
        
        # Preserve H metadata
        H_act._is_potential = True
        H_act._layer = f"Activation_{self.activation_type}"
        H_act._meaning = f"Hypotheses after {self.activation_type} activation"
        H_act._contains_alternatives = True
        
        return x_act, H_act