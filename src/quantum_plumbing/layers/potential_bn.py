import torch
import torch.nn as nn
from typing import Tuple, Optional


class PotentialBatchNorm(nn.Module):
    """
    Batch Normalization that preserves hypothetical thinking space (H).
    
    Key insight:
    - Standard BatchNorm: Normalize x (discard H)
    - Potential BatchNorm: Normalize x FROM H perspective (preserve H)
    
    This means:
    - We calculate mean/variance from H (all possibilities)
    - We normalize x using H statistics
    - We preserve H structure through normalization
    
    Result: H flows through BatchNorm without collapse
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        """
        Args:
            num_features: Number of features to normalize
            eps: Small constant for numerical stability
            momentum: Momentum for running statistics
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable scale and shift parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # Running statistics for inference
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        
        # Metadata
        self._is_potential_layer = True
        self._layer_type = "BatchNorm"
    
    def forward(self, x: torch.Tensor, 
                H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: Normalize from H perspective.
        
        Args:
            x: Input (batch_size, num_features)
            H: Hypotheses (num_potentials, batch_size, num_features)
        
        Returns:
            x_norm: Normalized input
            H_norm: Normalized hypotheses
        """
        
        if self.training:
            # STEP 1: Calculate statistics FROM H
            # This is key - we normalize x based on what H tells us
            
            # Shape of H: (num_potentials, batch_size, num_features)
            # We want mean/var over batch and potentials
            
            # Mean across batch and potentials
            mean_H = torch.mean(H, dim=(1, 2))  # (num_potentials,)
            
            # Variance across batch and potentials
            var_H = torch.var(H, dim=(1, 2), unbiased=False)  # (num_potentials,)
            
            # Take mean of H statistics (average across potentials)
            mean = torch.mean(mean_H)
            var = torch.mean(var_H)
            
            # STEP 2: Normalize x using H statistics
            # This is crucial: x is normalized from H perspective
            x_norm = (x - mean) / torch.sqrt(var + self.eps)
            
            # STEP 3: Normalize H using same statistics
            # Reshape for broadcasting
            H_expanded_mean = mean_H.view(-1, 1, 1)  # (num_potentials, 1, 1)
            H_expanded_var = var_H.view(-1, 1, 1)  # (num_potentials, 1, 1)
            
            # Normalize H
            H_norm = (H - H_expanded_mean) / torch.sqrt(H_expanded_var + self.eps)
            
            # STEP 4: Update running statistics (for inference)
            with torch.no_grad():
                self.running_mean.copy_(
                    (1 - self.momentum) * self.running_mean + self.momentum * mean
                )
                self.running_var.copy_(
                    (1 - self.momentum) * self.running_var + self.momentum * var
                )
        
        else:
            # Inference: use running statistics
            x_norm = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
            H_norm = (
                H - self.running_mean.view(1, 1, -1)
            ) / torch.sqrt(self.running_var.view(1, 1, -1) + self.eps)
        
        # STEP 5: Apply learnable scale and shift
        # These help the network learn optimal normalization
        x_norm = x_norm * self.weight + self.bias
        H_norm = H_norm * self.weight.view(1, 1, -1) + self.bias.view(1, 1, -1)
        
        # STEP 6: Preserve H metadata
        H_norm._is_potential = True
        H_norm._layer = "BatchNorm"
        H_norm._meaning = "Hypotheses after potential batch normalization"
        H_norm._normalized_from_potentials = True
        
        return x_norm, H_norm
    
    def extra_repr(self) -> str:
        return (f'num_features={self.num_features}, '
                f'eps={self.eps}, '
                f'momentum={self.momentum}')
