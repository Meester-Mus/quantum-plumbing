import torch
import torch.nn.functional as F
from typing import Optional


def potential_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    H: Optional[torch.Tensor] = None,
    task: str = "classification",
    h_diversity_weight: float = 0.0,
) -> torch.Tensor:
    """
    Loss function for Potential networks.

    Combines a standard task loss with an optional H-diversity regulariser that
    encourages the network to maintain a rich hypothetical space.

    Args:
        output:             Network output (batch_size, num_classes) for
                            classification or (batch_size,) / (batch_size, 1)
                            for regression.
        target:             Ground-truth labels or values.
        H:                  Hypothetical space tensor
                            (num_potentials, batch_size, features).
                            When provided and h_diversity_weight > 0, a
                            diversity regulariser is added to the loss.
        task:               'classification' (cross-entropy) or
                            'regression' (MSE).
        h_diversity_weight: Coefficient for the H-diversity regulariser
                            (encourages hypotheses to differ from each other).
                            Set to 0 to disable (default).

    Returns:
        Scalar loss tensor.
    """
    if task == "classification":
        base_loss = F.cross_entropy(output, target)
    elif task == "regression":
        base_loss = F.mse_loss(output.squeeze(-1), target.float())
    else:
        raise ValueError(f"Unknown task '{task}'. Use 'classification' or 'regression'.")

    if H is not None and h_diversity_weight > 0.0:
        # Diversity regulariser: maximise variance across potentials.
        # Higher variance → hypotheses differ → richer thinking space.
        # Shape: (num_potentials, batch, features) → mean variance over batch/features
        h_var = torch.var(H, dim=0).mean()  # scalar
        # Subtract: we want to *maximise* diversity, so reduce loss when var is high
        diversity_term = -h_diversity_weight * h_var
        return base_loss + diversity_term

    return base_loss


def h_utilization(H: torch.Tensor) -> torch.Tensor:
    """
    Measure how much the network uses its hypothetical thinking space.

    A network that assigns all weight to one hypothesis has low utilisation
    (entropy ≈ 0).  A network that spreads weight evenly has high utilisation
    (entropy ≈ log(num_potentials)).

    Uses the per-sample score entropy stored on H by PotentialFCLayer when
    available, otherwise falls back to a variance-based measure.

    Args:
        H: Hypothetical space tensor (num_potentials, batch_size, features)
           as returned by any Potential layer.

    Returns:
        Scalar tensor in [0, 1] where 1 means maximum utilisation.
    """
    scores = getattr(H, "_scores", None)

    if scores is not None:
        # scores: (num_potentials, batch_size) – already a probability distribution
        # Normalised entropy: H(p) / log(k)
        num_potentials = scores.shape[0]
        entropy = -torch.sum(scores * torch.log(scores + 1e-8), dim=0)  # (batch,)
        max_entropy = torch.log(torch.as_tensor(float(num_potentials), dtype=scores.dtype, device=scores.device))
        return (entropy / (max_entropy + 1e-8)).mean()

    # Fallback: coefficient of variation across potentials
    num_potentials = H.shape[0]
    mean_h = torch.mean(H, dim=0, keepdim=True)          # (1, batch, features)
    std_h = torch.std(H, dim=0, keepdim=True)             # (1, batch, features)
    cv = (std_h / (torch.abs(mean_h) + 1e-8)).mean()
    # Normalise loosely to [0, 1] by clamping
    return torch.clamp(cv, 0.0, 1.0)
