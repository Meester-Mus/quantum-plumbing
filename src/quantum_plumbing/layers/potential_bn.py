import torch
import torch.nn as nn
from typing import Tuple, Optional


class PotentialBatchNorm(nn.Module):
    """
    Batch Normalization that preserves thinking space (H).
    
    Key difference from standard BatchNorm:
    - Standard: Normalize x only
    - Potential: Normalize x FROM H perspective
    
    This preserves the hypothetical structure through normalization.
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        """
        Args:
            num_features: Number of features
            eps: Small constant for numerical stability
            momentum: Momentum for running statistics
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # Running statistics
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        
        self._is_potential_layer = True
        self._layer_type = "BatchNorm"
    
    def forward(self, x: torch.Tensor, 
                H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass normalizing FROM H perspective.
        
        Args:
            x: Input (batch_size, num_features)
            H: Hypotheses (num_potentials, batch_size, num_features)
        
        Returns:
            x_norm: Normalized input
            H_norm: Normalized hypotheses
        """
        
        # Calculate statistics FROM H (not from x alone)
        # This gives normalization from hypothetical perspective
        
        if self.training:
            # Calculate mean and variance from H
            # Mean over batch and features
            mean_H = torch.mean(H, dim=(1, 2))  # (num_potentials,)
            var_H = torch.var(H, dim=(1, 2))  # (num_potentials,)
            
            # Normalize both x and H
            x_norm = (x - mean_H[0]) / torch.sqrt(var_H[0] + self.eps)
            
            # Normalize H
            H_expanded_mean = mean_H.view(-1, 1, 1)  # (num_potentials, 1, 1)
            H_expanded_var = var_H.view(-1, 1, 1)  # (num_potentials, 1, 1)
            H_norm = (H - H_expanded_mean) / torch.sqrt(H_expanded_var + self.eps)
            
            # Update running statistics
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean_H[0]
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var_H[0]
        else:
            # Use running statistics
            x_norm = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
            H_norm = H  # In inference, just return H
        
        # Apply learnable scale and shift
        x_norm = x_norm * self.weight + self.bias
        H_norm = H_norm * self.weight.view(1, 1, -1) + self.bias.view(1, 1, -1)
        
        # Preserve H metadata
        H_norm._is_potential = True
        H_norm._layer = "BatchNorm"
        H_norm._meaning = "Normalized hypotheses"
        
        return x_norm, H_norm