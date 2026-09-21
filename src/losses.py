"""Loss functions module for Deep Image Colorization.

Provides configurable loss functions for training:
- MSE Loss: Standard mean squared error regression loss (Baseline).
- Smooth L1 (Huber) Loss: Robust regression loss with quadratic behavior for small errors
  and linear behavior for large errors, reducing regression-to-the-mean penalty.
"""

from typing import Dict, Any, Tuple
import math
import torch
import torch.nn as nn

SUPPORTED_LOSSES = ("mse", "smooth_l1", "chroma_weighted_mse")
DEFAULT_SMOOTH_L1_BETA = 1.0  # Standard PyTorch default beta for SmoothL1Loss
DEFAULT_CHROMA_ALPHA = 1.0    # Default weighting factor alpha for ChromaWeightedMSELoss

class ChromaWeightedMSELoss(nn.Module):
    """Chroma-Weighted Mean Squared Error Loss.
    
    Weights pixel errors proportionally to ground-truth chroma:
        chroma = sqrt(a_true^2 + b_true^2)
        chroma_norm = chroma / sqrt(2)
        weight = 1.0 + alpha * chroma_norm
        loss = mean(weight * (prediction - target)^2)
    """
    def __init__(self, alpha: float = DEFAULT_CHROMA_ALPHA):
        super().__init__()
        if alpha < 0:
            raise ValueError(f"alpha parameter must be non-negative, got {alpha}")
        self.alpha = float(alpha)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # target: [B, 2, H, W] in [-1, 1], where channel 0 is a*, channel 1 is b*
        a_target = target[:, 0:1, :, :]
        b_target = target[:, 1:2, :, :]
        
        # Ground-truth chroma
        chroma = torch.sqrt(a_target ** 2 + b_target ** 2)
        inv_sqrt_2 = 1.0 / math.sqrt(2.0)
        chroma_norm = chroma * inv_sqrt_2  # [B, 1, H, W] in [0, 1]
        
        weight = (1.0 + self.alpha * chroma_norm).detach()
        squared_error = (prediction - target) ** 2  # [B, 2, H, W]
        
        return torch.mean(weight * squared_error)

def get_loss_function(
    name: str = "mse",
    beta: float = DEFAULT_SMOOTH_L1_BETA,
    alpha: float = DEFAULT_CHROMA_ALPHA
) -> nn.Module:
    """Factory function to retrieve a configured loss function by name.

    Args:
        name: Name of the loss function ('mse', 'smooth_l1', or 'chroma_weighted_mse').
        beta: Threshold parameter for Smooth L1 loss (default: 1.0).
        alpha: Weighting factor for Chroma-Weighted MSE loss (default: 1.0).

    Returns:
        torch.nn.Module: Instantiated PyTorch loss function.

    Raises:
        ValueError: If an unsupported loss name is provided.
    """
    normalized_name = name.strip().lower()
    if normalized_name == "mse":
        return nn.MSELoss()
    elif normalized_name in ("smooth_l1", "smoothl1", "huber"):
        if beta <= 0:
            raise ValueError(f"beta parameter must be positive, got {beta}")
        return nn.SmoothL1Loss(beta=beta)
    elif normalized_name in ("chroma_weighted_mse", "chroma_weighted", "cw_mse"):
        if alpha < 0:
            raise ValueError(f"alpha parameter must be non-negative, got {alpha}")
        return ChromaWeightedMSELoss(alpha=alpha)
    else:
        raise ValueError(
            f"Unsupported loss function '{name}'. Supported losses are: {SUPPORTED_LOSSES}"
        )

def get_loss_metadata(
    name: str = "mse",
    beta: float = DEFAULT_SMOOTH_L1_BETA,
    alpha: float = DEFAULT_CHROMA_ALPHA
) -> Dict[str, Any]:
    """Retrieve metadata describing the loss function and configuration for logging/checkpoints.

    Args:
        name: Loss function name.
        beta: Beta threshold for Smooth L1 loss.
        alpha: Alpha factor for Chroma-Weighted MSE loss.

    Returns:
        dict: Metadata dictionary containing loss_name, loss parameters, and formulation details.
    """
    normalized_name = name.strip().lower()
    if normalized_name == "mse":
        return {
            "loss_name": "mse",
            "loss_beta": None,
            "loss_alpha": None,
            "formulation": "MSELoss (mean((y_pred - y_true)^2))",
            "description": "Baseline Mean Squared Error regression loss."
        }
    elif normalized_name in ("smooth_l1", "smoothl1", "huber"):
        if beta <= 0:
            raise ValueError(f"beta parameter must be positive, got {beta}")
        return {
            "loss_name": "smooth_l1",
            "loss_beta": beta,
            "loss_alpha": None,
            "formulation": f"SmoothL1Loss(beta={beta}) (0.5*(x-y)^2/beta for |x-y|<beta, |x-y|-0.5*beta otherwise)",
            "description": f"Robust Smooth L1 / Huber regression loss with transition threshold beta={beta}."
        }
    elif normalized_name in ("chroma_weighted_mse", "chroma_weighted", "cw_mse"):
        if alpha < 0:
            raise ValueError(f"alpha parameter must be non-negative, got {alpha}")
        return {
            "loss_name": "chroma_weighted_mse",
            "loss_beta": None,
            "loss_alpha": alpha,
            "formulation": f"ChromaWeightedMSELoss(alpha={alpha}) (mean((1 + {alpha} * (sqrt(a^2+b^2)/sqrt(2))) * (y_pred - y_true)^2))",
            "description": f"Chroma-Weighted MSE regression loss with weighting factor alpha={alpha}."
        }
    else:
        raise ValueError(
            f"Unsupported loss function '{name}'. Supported losses are: {SUPPORTED_LOSSES}"
        )
