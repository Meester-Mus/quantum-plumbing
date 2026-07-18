import torch
import torch.nn as nn
from typing import Tuple


class PotentialDropout(nn.Module):
    """
    Dropout that selects FROM hypotheses rather than random discard.

    Key difference from standard Dropout:
    - Standard: Randomly discard (chaos, no structure)
    - Potential: Select from H based on potential (preserve structure)

    This keeps hypothetical thinking space structured.
    """

    def __init__(self, p: float = 0.5):
        """
        Args:
            p: Probability of dropping (0 < p < 1)
        """
        super().__init__()
        assert 0 <= p < 1, "Dropout probability must be between 0 and 1"
        self.p = p

        self._is_potential_layer = True
        self._layer_type = "Dropout"

    def forward(
        self, x: torch.Tensor, H: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass selecting from H.

        Args:
            x: Input (batch_size, features)
            H: Hypotheses (num_potentials, batch_size, features)

        Returns:
            x_out: Output with structured selection
            H_out: Hypotheses with structured selection
        """

        if not self.training or self.p == 0:
            return x, H

        # Select FROM H based on potential strength
        # Rather than random discard

        # Score feature positions across the full hypothesis space.
        feature_strength = torch.mean(torch.abs(H), dim=(0, 1))

        # Threshold - keep above threshold probability
        threshold = torch.quantile(feature_strength, self.p)
        mask = (feature_strength > threshold).float()
        scale = 1.0 / (1.0 - self.p)

        # Apply to both x and H
        # Scale to maintain mean
        x_out = x * mask.unsqueeze(0) * scale
        H_out = H * mask.unsqueeze(0).unsqueeze(0) * scale

        # Preserve H metadata
        H_out._is_potential = True
        H_out._layer = "Dropout"
        H_out._meaning = "Selected hypotheses (structured dropout)"
        H_out._mask = mask

        return x_out, H_out
