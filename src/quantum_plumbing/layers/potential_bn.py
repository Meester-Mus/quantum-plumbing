import torch
import torch.nn as nn
from typing import Tuple


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
    
    def forward(self, x: torch.Tensor, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: Normalize from H perspective.
        
        Args:
            x: Input (batch_size, num_features)
            H: Hypotheses (num_potentials, batch_size, num_features)
        
        Returns:
            x_norm: Normalized input
            H_norm: Normalized hypotheses
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
        if x.shape[1] != self.num_features:
            raise ValueError(
                f"Expected x to have {self.num_features} features, got {x.shape[1]}"
            )

        if self.training:
            # Calculate per-feature statistics from H across potential and batch axes.
            mean = H.mean(dim=(0, 1))  # (features,)
            var = H.var(dim=(0, 1), unbiased=False)  # (features,)

            x_norm = (x - mean) / torch.sqrt(var + self.eps)
            H_norm = (H - mean.view(1, 1, -1)) / torch.sqrt(var.view(1, 1, -1) + self.eps)

            with torch.no_grad():
                self.running_mean.copy_(
                    (1 - self.momentum) * self.running_mean + self.momentum * mean
                )
                self.running_var.copy_(
                    (1 - self.momentum) * self.running_var + self.momentum * var
                )
        else:
            mean = self.running_mean
            var = self.running_var
            x_norm = (x - mean) / torch.sqrt(var + self.eps)
            H_norm = (H - mean.view(1, 1, -1)) / torch.sqrt(var.view(1, 1, -1) + self.eps)
        
        # STEP 5: Apply learnable scale and shift
        # These help the network learn optimal normalization
        x_norm = x_norm * self.weight + self.bias
        H_norm = H_norm * self.weight.view(1, 1, -1) + self.bias.view(1, 1, -1)
        
        # Preserve H metadata
        H_norm._is_potential = True
        H_norm._layer = "BatchNorm"
        H_norm._meaning = "Hypotheses after potential batch normalization"
        H_norm._normalized_from_potentials = True
        
        return x_norm, H_norm
    
    def extra_repr(self) -> str:
        return (f'num_features={self.num_features}, '
                f'eps={self.eps}, '
                f'momentum={self.momentum}')
