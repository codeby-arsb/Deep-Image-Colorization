import cv2
import numpy as np
import torch
from skimage.color import rgb2lab, lab2rgb

def load_rgb_image(path: str) -> np.ndarray:
    """Load an image from the given path as an RGB numpy array."""
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Failed to load image at {path}. File may be missing or corrupted.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def resize_image(image: np.ndarray, size: tuple = (256, 256)) -> np.ndarray:
    """Resize the image to the specified size using bilinear interpolation."""
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)

def rgb_to_lab(rgb_image: np.ndarray) -> np.ndarray:
    """Convert an RGB image (uint8 or float) to CIE Lab color space."""
    if rgb_image.dtype == np.uint8:
        rgb_float = rgb_image.astype(np.float32) / 255.0
    else:
        rgb_float = rgb_image
    return rgb2lab(rgb_float)

def normalize_lab(lab_image: np.ndarray) -> tuple:
    """
    Extract L and ab channels from Lab image, normalize them, 
    and return as channel-first float32 arrays.
    """
    L = lab_image[:, :, 0]
    ab = lab_image[:, :, 1:]
    
    L_normalized = L / 50.0 - 1.0
    ab_normalized = ab / 128.0
    
    L_out = np.expand_dims(L_normalized, axis=0).astype(np.float32)
    ab_out = np.transpose(ab_normalized, (2, 0, 1)).astype(np.float32)
    
    return L_out, ab_out

def denormalize_lab(L_normalized: np.ndarray, ab_normalized: np.ndarray) -> np.ndarray:
    """
    Denormalize channel-first L and ab arrays and combine into a channel-last Lab image.
    """
    L = L_normalized[0]
    ab = np.transpose(ab_normalized, (1, 2, 0))
    
    L_denorm = (L + 1.0) * 50.0
    ab_denorm = ab * 128.0
    
    lab = np.zeros((L.shape[0], L.shape[1], 3), dtype=np.float64)
    lab[:, :, 0] = L_denorm
    lab[:, :, 1:] = ab_denorm
    return lab

def lab_to_rgb(lab_image: np.ndarray) -> np.ndarray:
    """Convert a CIE Lab image back to RGB [0, 1]."""
    rgb = lab2rgb(lab_image)
    return np.clip(rgb, 0.0, 1.0)

def preprocess_image(image: np.ndarray) -> tuple:
    """Full preprocessing pipeline from raw RGB to normalized L and ab channels."""
    resized = resize_image(image)
    lab = rgb_to_lab(resized)
    L_out, ab_out = normalize_lab(lab)
    return L_out, ab_out

def reconstruct_rgb(L_normalized: np.ndarray, ab_normalized: np.ndarray) -> np.ndarray:
    """Full reconstruction pipeline from normalized L and ab channels to RGB image."""
    lab = denormalize_lab(L_normalized, ab_normalized)
    rgb = lab_to_rgb(lab)
    return rgb

def lab_to_rgb_torch(L_norm: torch.Tensor, ab_norm: torch.Tensor) -> torch.Tensor:
    """Differentiable conversion from normalized Lab tensors to sRGB tensors in [0, 1].
    
    Reconstructs RGB images purely using PyTorch tensor operations, preserving autograd
    gradients back to normalized chrominance predictions without NumPy or PIL conversions.
    Conforms exactly to CIE 1931 D65 standard illuminant and ITU-R BT.709 sRGB curve used by skimage.
    
    Args:
        L_norm: Normalized luminance tensor [B, 1, H, W] in [-1.0, 1.0].
        ab_norm: Normalized chrominance tensor [B, 2, H, W] in [-1.0, 1.0].
        
    Returns:
        torch.Tensor: sRGB tensor [B, 3, H, W] in [0.0, 1.0].
    """
    # 1. Denormalize to CIE Lab ranges: L in [0, 100], ab in [-128, 128]
    L = (L_norm + 1.0) * 50.0
    a = ab_norm[:, 0:1, :, :] * 128.0
    b = ab_norm[:, 1:2, :, :] * 128.0
    
    # 2. Lab to XYZ intermediate representation (D65 white point: [0.95047, 1.0, 1.08883])
    y = (L + 16.0) / 116.0
    x = (a / 500.0) + y
    z = y - (b / 200.0)
    z = torch.clamp(z, min=0.0)
    
    xyz_t = torch.cat([x, y, z], dim=1)
    mask = xyz_t > 0.2068966
    xyz = torch.where(mask, torch.pow(xyz_t, 3.0), (xyz_t - 16.0 / 116.0) / 7.787)
    
    # Rescale by D65 reference white
    X = xyz[:, 0:1, :, :] * 0.95047
    Y = xyz[:, 1:2, :, :] * 1.00000
    Z = xyz[:, 2:3, :, :] * 1.08883
    
    # 3. XYZ to linear sRGB (skimage / ITU-R BT.709 D65 matrix)
    r =  3.24048134 * X - 1.53715152 * Y - 0.49853633 * Z
    g = -0.96925495 * X + 1.87599000 * Y + 0.04155593 * Z
    b_rgb = 0.05564664 * X - 0.20404134 * Y + 1.05731107 * Z
    
    rgb_arr = torch.cat([r, g, b_rgb], dim=1)
    
    # 4. Linear sRGB to standard non-linear sRGB (gamma correction)
    mask_gamma = rgb_arr > 0.0031308
    rgb_gamma = torch.where(
        mask_gamma,
        1.055 * torch.pow(torch.clamp(rgb_arr, min=1e-8), 1.0 / 2.4) - 0.055,
        rgb_arr * 12.92
    )
    return torch.clamp(rgb_gamma, 0.0, 1.0)


