"""Loss functions module for Deep Image Colorization.

Provides configurable loss functions for training:
- MSE Loss: Standard mean squared error regression loss (Baseline).
- Smooth L1 (Huber) Loss: Robust regression loss with quadratic behavior for small errors
  and linear behavior for large errors, reducing regression-to-the-mean penalty.
- Chroma-Weighted MSE Loss: Weighted regression penalizing desaturation on chromatic pixels.
- Perceptual Loss: Multi-component loss combining pixel-level MSE on Lab chrominance with
  VGG-16 feature-space loss on reconstructed sRGB images.
"""

from typing import Dict, Any, Optional, Tuple
import math
import torch
import torch.nn as nn
import torchvision.models as models

try:
    from src.utils import lab_to_rgb_torch
except ImportError:
    from utils import lab_to_rgb_torch

SUPPORTED_LOSSES = ("mse", "smooth_l1", "chroma_weighted_mse", "perceptual")
DEFAULT_SMOOTH_L1_BETA = 1.0       # Standard PyTorch default beta for SmoothL1Loss
DEFAULT_CHROMA_ALPHA = 1.0         # Default weighting factor alpha for ChromaWeightedMSELoss
DEFAULT_PERCEPTUAL_WEIGHT = 0.01   # Default weighting lambda for perceptual loss
DEFAULT_PERCEPTUAL_LAYER = "relu2_2"

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

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        l_channel: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
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

class VGGPerceptualLoss(nn.Module):
    """VGG-16 Feature Extractor for Perceptual Loss on RGB images.
    
    Extracts intermediate representations from a pretrained, frozen VGG-16 network.
    Supported layers:
    - 'relu2_2': Slice up to layer 9 (conv2_2 + relu, 128 channels, 128x128). Captures
      low-to-mid level color distributions, edge-color alignment, and surface textures.
    - 'relu3_3': Slice up to layer 16 (conv3_3 + relu, 256 channels, 64x64). Captures
      mid-level semantic structures.
    """
    def __init__(self, layer: str = DEFAULT_PERCEPTUAL_LAYER):
        super().__init__()
        self.layer_name = layer.lower().strip()
        
        # Load pretrained VGG-16 features
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        
        if self.layer_name in ("relu2_2", "conv2_2", "layer8", "layer9"):
            self.slice = 9   # Features [0:9] ends after relu2_2 (index 8)
            self.layer_name = "relu2_2"
        elif self.layer_name in ("relu3_3", "conv3_3", "layer15", "layer16"):
            self.slice = 16  # Features [0:16] ends after relu3_3 (index 15)
            self.layer_name = "relu3_3"
        else:
            raise ValueError(f"Unsupported perceptual layer '{layer}'. Supported: ('relu2_2', 'relu3_3')")
            
        self.features = nn.Sequential(*list(vgg.features.children())[:self.slice])
        
        # Freeze all parameters
        for param in self.features.parameters():
            param.requires_grad = False
            
        # Ensure eval mode
        self.features.eval()
        
        # ImageNet standardization buffers
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    def train(self, mode: bool = True):
        # Always keep feature extractor in eval mode regardless of parent module state
        super().train(mode)
        self.features.eval()
        return self

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Standardize RGB in [0, 1] using ImageNet mean and std."""
        return (x - self.mean) / self.std

    def forward(self, pred_rgb: torch.Tensor, target_rgb: torch.Tensor) -> torch.Tensor:
        """Compute MSE in VGG feature space between predicted and target RGB images."""
        pred_norm = self.normalize(pred_rgb)
        target_norm = self.normalize(target_rgb)
        
        feat_pred = self.features(pred_norm)
        with torch.no_grad():
            feat_target = self.features(target_norm).detach()
            
        return nn.functional.mse_loss(feat_pred, feat_target)

class PerceptualColorizationLoss(nn.Module):
    """Combined Reconstruction and Perceptual Loss for CIE Lab Colorization.
    
    Formulation:
        L_total = L_recon(ab_pred, ab_target) + lambda_perc * L_perc(RGB_pred, RGB_target)
        
    where:
        - L_recon: Standard pixel-level MSE on normalized ab chrominance.
        - L_perc: Feature-space MSE from pretrained VGG-16 on reconstructed sRGB images.
        - lambda_perc: Weighting hyperparameter balancing pixel vs feature loss.
    """
    def __init__(
        self,
        perceptual_weight: float = DEFAULT_PERCEPTUAL_WEIGHT,
        perceptual_layer: str = DEFAULT_PERCEPTUAL_LAYER
    ):
        super().__init__()
        if perceptual_weight < 0:
            raise ValueError(f"perceptual_weight must be non-negative, got {perceptual_weight}")
        self.perceptual_weight = float(perceptual_weight)
        self.perceptual_layer = perceptual_layer
        
        self.recon_loss = nn.MSELoss()
        self.perceptual_loss = VGGPerceptualLoss(layer=perceptual_layer)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        l_channel: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute combined reconstruction and perceptual loss.
        
        Args:
            prediction: Predicted ab chrominance [B, 2, H, W] in [-1, 1].
            target: Ground-truth ab chrominance [B, 2, H, W] in [-1, 1].
            l_channel: Normalized luminance [B, 1, H, W] in [-1, 1]. If None, defaults to
                       neutral luminance (0.0, corresponding to L=50).
                       
        Returns:
            torch.Tensor: Total scalar loss.
        """
        # 1. Pixel-level reconstruction loss on ab channels
        l_recon = self.recon_loss(prediction, target)
        
        if self.perceptual_weight == 0.0:
            return l_recon
            
        # 2. Prepare luminance channel
        if l_channel is None:
            l_channel = torch.zeros(
                prediction.shape[0], 1, prediction.shape[2], prediction.shape[3],
                device=prediction.device, dtype=prediction.dtype
            )
            
        # 3. Differentiable reconstruction to sRGB in [0, 1]
        pred_rgb = lab_to_rgb_torch(l_channel, prediction)
        with torch.no_grad():
            target_rgb = lab_to_rgb_torch(l_channel, target)
            
        # 4. Feature-space perceptual loss
        l_perc = self.perceptual_loss(pred_rgb, target_rgb)
        
        return l_recon + self.perceptual_weight * l_perc

def get_loss_function(
    name: str = "mse",
    beta: float = DEFAULT_SMOOTH_L1_BETA,
    alpha: float = DEFAULT_CHROMA_ALPHA,
    perceptual_weight: float = DEFAULT_PERCEPTUAL_WEIGHT,
    perceptual_layer: str = DEFAULT_PERCEPTUAL_LAYER
) -> nn.Module:
    """Factory function to retrieve a configured loss function by name.

    Args:
        name: Name of the loss function ('mse', 'smooth_l1', 'chroma_weighted_mse', or 'perceptual').
        beta: Threshold parameter for Smooth L1 loss (default: 1.0).
        alpha: Weighting factor for Chroma-Weighted MSE loss (default: 1.0).
        perceptual_weight: Weight lambda for perceptual loss (default: 0.01).
        perceptual_layer: VGG-16 feature layer to extract (default: 'relu2_2').

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
    elif normalized_name in ("perceptual", "vgg_perceptual", "vgg"):
        return PerceptualColorizationLoss(
            perceptual_weight=perceptual_weight,
            perceptual_layer=perceptual_layer
        )
    else:
        raise ValueError(
            f"Unsupported loss function '{name}'. Supported losses are: {SUPPORTED_LOSSES}"
        )

def get_loss_metadata(
    name: str = "mse",
    beta: float = DEFAULT_SMOOTH_L1_BETA,
    alpha: float = DEFAULT_CHROMA_ALPHA,
    perceptual_weight: float = DEFAULT_PERCEPTUAL_WEIGHT,
    perceptual_layer: str = DEFAULT_PERCEPTUAL_LAYER
) -> Dict[str, Any]:
    """Retrieve metadata describing the loss function and configuration for logging/checkpoints.

    Args:
        name: Loss function name.
        beta: Beta threshold for Smooth L1 loss.
        alpha: Alpha factor for Chroma-Weighted MSE loss.
        perceptual_weight: Weight lambda for perceptual loss.
        perceptual_layer: VGG-16 feature layer name.

    Returns:
        dict: Metadata dictionary containing loss_name, loss parameters, and formulation details.
    """
    normalized_name = name.strip().lower()
    if normalized_name == "mse":
        return {
            "loss_name": "mse",
            "loss_beta": None,
            "loss_alpha": None,
            "perceptual_weight": None,
            "perceptual_layer": None,
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
            "perceptual_weight": None,
            "perceptual_layer": None,
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
            "perceptual_weight": None,
            "perceptual_layer": None,
            "formulation": f"ChromaWeightedMSELoss(alpha={alpha}) (mean((1 + {alpha} * (sqrt(a^2+b^2)/sqrt(2))) * (y_pred - y_true)^2))",
            "description": f"Chroma-Weighted MSE regression loss with weighting factor alpha={alpha}."
        }
    elif normalized_name in ("perceptual", "vgg_perceptual", "vgg"):
        return {
            "loss_name": "perceptual",
            "loss_beta": None,
            "loss_alpha": None,
            "perceptual_weight": perceptual_weight,
            "perceptual_layer": perceptual_layer,
            "formulation": f"PerceptualColorizationLoss(weight={perceptual_weight}, layer={perceptual_layer}) (MSE(ab) + {perceptual_weight} * MSE(VGG_{perceptual_layer}(RGB)))",
            "description": f"Combined pixel-level MSE and VGG-16 ({perceptual_layer}) perceptual feature loss with weight lambda={perceptual_weight}."
        }
    else:
        raise ValueError(
            f"Unsupported loss function '{name}'. Supported losses are: {SUPPORTED_LOSSES}"
        )
