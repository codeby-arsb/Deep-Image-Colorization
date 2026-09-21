"""Comprehensive unit test suite for Perceptual Loss (Step 7.4).

Tests:
- Test A: Perfect prediction produces zero perceptual loss and finite total loss.
- Test B: Perturbed prediction produces positive loss.
- Test C: Gradient flow through autograd to chrominance predictions (gradients exist and are finite).
- Test D: Feature extractor parameters are frozen (requires_grad == False).
- Test E: Feature extractor remains in eval mode (training == False) even after train() call.
- Test F: CPU forward and backward execution.
- Test G: MPS forward and backward execution (if MPS available).
- Test H: Batch compatibility ([1, 2, 256, 256] and [8, 2, 256, 256]).
- Test I: Numerical stability on extreme values (no NaN / Inf).
- Test J: skimage numerical equivalence (< 1e-4 absolute error).
- Test K: Existing loss functions non-regression check.
"""

import os
import sys
import math
import torch
import torch.nn as nn
import numpy as np
from skimage.color import lab2rgb

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import lab_to_rgb_torch
from src.losses import (
    VGGPerceptualLoss,
    PerceptualColorizationLoss,
    get_loss_function,
    get_loss_metadata,
    SUPPORTED_LOSSES
)

def run_tests():
    print("==================================================")
    print("RUNNING PERCEPTUAL LOSS UNIT TEST SUITE")
    print("==================================================")
    
    # ----------------------------------------------------
    # Test A: Perfect Prediction
    # ----------------------------------------------------
    print("Test A: Perfect Prediction...")
    loss_fn = PerceptualColorizationLoss(perceptual_weight=0.01, perceptual_layer="relu2_2")
    loss_fn.eval()
    
    target_ab = torch.zeros(2, 2, 64, 64)
    pred_ab = target_ab.clone()
    l_channel = torch.zeros(2, 1, 64, 64) # neutral luminance L=50
    
    loss = loss_fn(pred_ab, target_ab, l_channel=l_channel)
    assert torch.isfinite(loss), f"Loss is not finite: {loss}"
    assert loss.item() < 1e-5, f"Expected loss ≈ 0, got {loss.item()}"
    print(f"  [PASS] Perfect prediction loss: {loss.item():.8f} ≈ 0")
    
    # ----------------------------------------------------
    # Test B: Perturbed Prediction
    # ----------------------------------------------------
    print("Test B: Perturbed Prediction...")
    pred_perturbed = target_ab + 0.3
    loss_perturbed = loss_fn(pred_perturbed, target_ab, l_channel=l_channel)
    assert loss_perturbed.item() > 0.0, f"Expected positive loss, got {loss_perturbed.item()}"
    assert torch.isfinite(loss_perturbed), "Perturbed loss is not finite"
    print(f"  [PASS] Perturbed prediction loss: {loss_perturbed.item():.6f} > 0")
    
    # ----------------------------------------------------
    # Test C: Gradient Flow
    # ----------------------------------------------------
    print("Test C: Gradient Flow through Autograd...")
    pred_ab_grad = torch.randn(2, 2, 64, 64, requires_grad=True)
    target_ab_fixed = torch.randn(2, 2, 64, 64)
    l_fixed = torch.rand(2, 1, 64, 64) * 2.0 - 1.0
    
    loss_grad = loss_fn(pred_ab_grad, target_ab_fixed, l_channel=l_fixed)
    loss_grad.backward()
    
    assert pred_ab_grad.grad is not None, "Gradients not received by prediction tensor!"
    assert torch.isfinite(pred_ab_grad.grad).all().item(), "Gradients contain NaN or Inf!"
    assert not torch.all(pred_ab_grad.grad == 0), "Gradients are all zero!"
    print(f"  [PASS] Gradients exist, finite, non-zero. Grad range: [{pred_ab_grad.grad.min().item():.5f}, {pred_ab_grad.grad.max().item():.5f}]")
    
    # ----------------------------------------------------
    # Test D: Frozen Feature Extractor
    # ----------------------------------------------------
    print("Test D: Frozen Feature Extractor...")
    vgg_loss = loss_fn.perceptual_loss
    all_frozen = all(not p.requires_grad for p in vgg_loss.features.parameters())
    assert all_frozen, "Some VGG feature parameters require grad!"
    num_params = sum(p.numel() for p in vgg_loss.features.parameters())
    print(f"  [PASS] All {num_params} VGG feature parameters are frozen (requires_grad == False)")
    
    # ----------------------------------------------------
    # Test E: Feature Extractor Remains in Eval Mode
    # ----------------------------------------------------
    print("Test E: Feature Extractor Eval Mode Preservation...")
    loss_fn.train() # Switch parent to train mode
    assert not vgg_loss.features.training, "VGG features switched to train mode!"
    loss_fn.eval()
    assert not vgg_loss.features.training, "VGG features not in eval mode!"
    print("  [PASS] VGG features remain in eval mode (training == False)")
    
    # ----------------------------------------------------
    # Test F: CPU Compatibility
    # ----------------------------------------------------
    print("Test F: CPU Compatibility...")
    loss_cpu = PerceptualColorizationLoss(perceptual_weight=0.01).to("cpu")
    x_cpu = torch.randn(2, 2, 32, 32, device="cpu", requires_grad=True)
    y_cpu = torch.randn(2, 2, 32, 32, device="cpu")
    l_cpu = torch.randn(2, 1, 32, 32, device="cpu")
    
    out_cpu = loss_cpu(x_cpu, y_cpu, l_channel=l_cpu)
    out_cpu.backward()
    assert torch.isfinite(out_cpu).item(), "CPU loss not finite"
    assert torch.isfinite(x_cpu.grad).all().item(), "CPU grad not finite"
    print("  [PASS] CPU forward and backward execution verified finite")
    
    # ----------------------------------------------------
    # Test G: MPS Compatibility
    # ----------------------------------------------------
    print("Test G: MPS Compatibility...")
    if torch.backends.mps.is_available():
        loss_mps = PerceptualColorizationLoss(perceptual_weight=0.01).to("mps")
        x_mps = torch.randn(2, 2, 64, 64, device="mps", requires_grad=True)
        y_mps = torch.randn(2, 2, 64, 64, device="mps")
        l_mps = torch.randn(2, 1, 64, 64, device="mps")
        
        out_mps = loss_mps(x_mps, y_mps, l_channel=l_mps)
        out_mps.backward()
        assert torch.isfinite(out_mps).item(), "MPS loss not finite"
        assert torch.isfinite(x_mps.grad).all().item(), "MPS grad not finite"
        print("  [PASS] MPS forward and backward execution verified finite on Apple Silicon")
    else:
        print("  [SKIP] MPS backend not available")
        
    # ----------------------------------------------------
    # Test H: Batch Compatibility
    # ----------------------------------------------------
    print("Test H: Batch Compatibility...")
    for bs in [1, 4, 8]:
        pred_b = torch.randn(bs, 2, 64, 64)
        target_b = torch.randn(bs, 2, 64, 64)
        l_b = torch.randn(bs, 1, 64, 64)
        out_b = loss_fn(pred_b, target_b, l_channel=l_b)
        assert torch.isfinite(out_b).item(), f"Batch size {bs} loss not finite"
    print("  [PASS] Verified batch sizes 1, 4, 8")
    
    # ----------------------------------------------------
    # Test I: Numerical Stability on Extreme Values
    # ----------------------------------------------------
    print("Test I: Numerical Stability on Extreme Values...")
    extreme_pred = torch.tensor([[[[1.0, -1.0], [0.0, 1.0]], [[-1.0, 1.0], [1.0, -1.0]]]]) # [1, 2, 2, 2]
    extreme_target = torch.tensor([[[[-1.0, 1.0], [0.0, -1.0]], [[1.0, -1.0], [-1.0, 1.0]]]])
    extreme_l = torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]]]]) # [1, 1, 2, 2]
    
    rgb_test = lab_to_rgb_torch(extreme_l, extreme_pred)
    assert not torch.isnan(rgb_test).any(), "NaN in extreme RGB!"
    assert not torch.isinf(rgb_test).any(), "Inf in extreme RGB!"
    assert (rgb_test >= 0.0).all() and (rgb_test <= 1.0).all(), "RGB out of [0, 1] range!"
    print("  [PASS] Extreme normalized values produce strictly valid RGB in [0, 1] without NaN/Inf")
    
    # ----------------------------------------------------
    # Test J: skimage Numerical Equivalence
    # ----------------------------------------------------
    print("Test J: skimage Numerical Equivalence...")
    # Test on a deterministic grid of Lab values
    L_vals = np.linspace(0.0, 100.0, 8)
    a_vals = np.linspace(-100.0, 100.0, 8)
    b_vals = np.linspace(-100.0, 100.0, 8)
    
    L_grid, a_grid, b_grid = np.meshgrid(L_vals, a_vals, b_vals, indexing='ij')
    lab_dense = np.stack([L_grid, a_grid, b_grid], axis=-1).reshape(-1, 8, 3) # [H, W, 3]
    
    # skimage
    rgb_skimage = lab2rgb(lab_dense)
    
    # torch
    L_t = torch.from_numpy(lab_dense[:, :, 0:1] / 50.0 - 1.0).permute(2, 0, 1).unsqueeze(0).float()
    ab_t = torch.from_numpy(lab_dense[:, :, 1:] / 128.0).permute(2, 0, 1).unsqueeze(0).float()
    
    rgb_torch = lab_to_rgb_torch(L_t, ab_t).squeeze(0).permute(1, 2, 0).numpy()
    
    max_err = np.max(np.abs(rgb_skimage - rgb_torch))
    assert max_err < 1e-4, f"Numerical discrepancy with skimage too high: {max_err}"
    print(f"  [PASS] Max discrepancy with skimage D65 sRGB: {max_err:.6e} < 1e-4")
    
    # ----------------------------------------------------
    # Test K: Non-Regression of Existing Losses
    # ----------------------------------------------------
    print("Test K: Non-Regression of Existing Losses...")
    assert "perceptual" in SUPPORTED_LOSSES
    mse = get_loss_function("mse")
    sl1 = get_loss_function("smooth_l1")
    cw = get_loss_function("chroma_weighted_mse")
    perc = get_loss_function("perceptual", perceptual_weight=0.01)
    
    p = torch.randn(2, 2, 16, 16)
    t = torch.randn(2, 2, 16, 16)
    
    l_mse = mse(p, t)
    l_sl1 = sl1(p, t)
    l_cw = cw(p, t)
    l_perc = perc(p, t)
    
    assert torch.isfinite(l_mse) and torch.isfinite(l_sl1) and torch.isfinite(l_cw) and torch.isfinite(l_perc)
    
    meta_perc = get_loss_metadata("perceptual", perceptual_weight=0.01, perceptual_layer="relu2_2")
    assert meta_perc["loss_name"] == "perceptual"
    assert meta_perc["perceptual_weight"] == 0.01
    assert meta_perc["perceptual_layer"] == "relu2_2"
    print("  [PASS] Existing losses and factory functions intact")
    
    print("==================================================")
    print("ALL 11 TESTS PASSED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
