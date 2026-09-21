"""Unit test suite for Chroma-Weighted MSE Loss.

Verifies:
1. Perfect Prediction: loss is 0 when pred == target.
2. Neutral Pixels: weight is 1.0 when a=0, b=0.
3. High-Chroma Pixels: weight is 2.0 when |a|=1, |b|=1 (with alpha=1.0).
4. Weight Direction: identical error produces higher loss on high-chroma target than neutral.
5. Gradients: gradients exist, are finite, and backward pass succeeds.
6. MPS compatibility: forward and backward pass on MPS and CPU with [B, 2, 256, 256].
7. Existing losses: MSE and Smooth L1 remain functional.
"""

import sys
import os
import math
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.losses import (
    ChromaWeightedMSELoss,
    get_loss_function,
    get_loss_metadata,
    SUPPORTED_LOSSES
)

def test_1_perfect_prediction():
    print("--- Test 1: Perfect Prediction ---")
    loss_fn = ChromaWeightedMSELoss(alpha=1.0)
    target = torch.randn(4, 2, 64, 64)
    pred = target.clone()
    
    loss = loss_fn(pred, target)
    assert loss.item() == 0.0, f"Expected 0.0 for identical tensors, got {loss.item()}"
    print(f"  [PASS] Perfect prediction yields loss = {loss.item():.6f}")

def test_2_neutral_pixels():
    print("--- Test 2: Neutral Pixels (a=0, b=0) ---")
    loss_fn = ChromaWeightedMSELoss(alpha=1.0)
    # Pure neutral target
    target = torch.zeros(1, 2, 32, 32)
    # Error of 0.5 everywhere
    pred = torch.full((1, 2, 32, 32), 0.5)
    
    loss = loss_fn(pred, target).item()
    # Expected: weight = 1.0 everywhere. Error = 0.5 => error^2 = 0.25 => loss = 0.25
    expected = 0.25
    assert abs(loss - expected) < 1e-6, f"Expected {expected}, got {loss}"
    print(f"  [PASS] Neutral target weight = 1.0; computed loss = {loss:.6f} == expected {expected:.6f}")

def test_3_high_chroma_pixels():
    print("--- Test 3: High-Chroma Pixels (|a|=1, |b|=1) ---")
    loss_fn = ChromaWeightedMSELoss(alpha=1.0)
    # Max chroma target: a=1, b=1 => chroma = sqrt(2) => chroma_norm = 1.0 => weight = 1 + 1.0*1.0 = 2.0
    target = torch.ones(1, 2, 32, 32)
    # Error of 0.5 everywhere (pred = 1.5)
    pred = torch.full((1, 2, 32, 32), 1.5)
    
    loss = loss_fn(pred, target).item()
    # Expected: weight = 2.0 everywhere. Error = 0.5 => error^2 = 0.25 => weight * error^2 = 0.50 => loss = 0.50
    expected = 0.50
    assert abs(loss - expected) < 1e-6, f"Expected {expected}, got {loss}"
    print(f"  [PASS] Max-chroma target weight = 2.0; computed loss = {loss:.6f} == expected {expected:.6f}")

def test_4_weight_direction():
    print("--- Test 4: Weight Direction (High-Chroma vs Neutral) ---")
    loss_fn = ChromaWeightedMSELoss(alpha=1.0)
    error_val = 0.3
    
    # Neutral target
    target_neutral = torch.zeros(1, 2, 16, 16)
    pred_neutral = torch.full((1, 2, 16, 16), error_val)
    loss_neutral = loss_fn(pred_neutral, target_neutral).item()
    
    # High chroma target (a=0.8, b=0.6 => chroma = 1.0 => chroma_norm = 1/sqrt(2) ≈ 0.707 => weight ≈ 1.707)
    target_chroma = torch.zeros(1, 2, 16, 16)
    target_chroma[:, 0, :, :] = 0.8
    target_chroma[:, 1, :, :] = 0.6
    pred_chroma = target_chroma + error_val
    loss_chroma = loss_fn(pred_chroma, target_chroma).item()
    
    assert loss_chroma > loss_neutral, f"Expected loss_chroma ({loss_chroma}) > loss_neutral ({loss_neutral})"
    ratio = loss_chroma / loss_neutral
    print(f"  [PASS] High-chroma loss ({loss_chroma:.6f}) > Neutral loss ({loss_neutral:.6f}) (Ratio: {ratio:.3f})")

def test_5_gradients():
    print("--- Test 5: Gradient Flow & Finite Checks ---")
    loss_fn = ChromaWeightedMSELoss(alpha=1.0)
    target = torch.randn(2, 2, 32, 32)
    pred = torch.randn(2, 2, 32, 32, requires_grad=True)
    
    loss = loss_fn(pred, target)
    loss.backward()
    
    assert pred.grad is not None, "Gradients were not populated"
    assert not torch.isnan(pred.grad).any(), "NaN in gradients"
    assert not torch.isinf(pred.grad).any(), "Inf in gradients"
    assert pred.grad.abs().max().item() > 0, "Zero gradients"
    print(f"  [PASS] Gradients exist, max |grad| = {pred.grad.abs().max().item():.6e}, all finite.")

def test_6_device_compatibility():
    print("--- Test 6: Device (MPS/CPU) & Shape Compatibility ---")
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
        
    loss_fn = get_loss_function("chroma_weighted_mse", alpha=1.0)
    batch_sizes = [1, 4, 8]
    
    for dev in devices:
        device = torch.device(dev)
        for b in batch_sizes:
            pred = torch.randn(b, 2, 256, 256, device=device, requires_grad=True)
            target = torch.randn(b, 2, 256, 256, device=device)
            
            loss = loss_fn(pred, target)
            assert loss.dim() == 0, f"Expected 0-dim scalar, got {loss.shape}"
            assert torch.isfinite(loss).item(), "Loss is non-finite"
            
            loss.backward()
            assert torch.isfinite(pred.grad).all().item(), f"Non-finite grad on {dev}"
            pred.grad.zero_()
            
        print(f"  [PASS] chroma_weighted_mse on {dev.upper()}: batches {batch_sizes} [B, 2, 256, 256] OK")

def test_7_existing_losses():
    print("--- Test 7: Existing Losses Non-Regression ---")
    mse = get_loss_function("mse")
    sl1 = get_loss_function("smooth_l1", beta=1.0)
    
    p = torch.randn(2, 2, 16, 16)
    t = torch.randn(2, 2, 16, 16)
    
    l_mse = mse(p, t)
    l_sl1 = sl1(p, t)
    
    assert torch.isfinite(l_mse).item(), "MSE loss failed"
    assert torch.isfinite(l_sl1).item(), "Smooth L1 loss failed"
    print(f"  [PASS] MSE and Smooth L1 functional (MSE: {l_mse.item():.4f}, SL1: {l_sl1.item():.4f})")

if __name__ == "__main__":
    print("==================================================")
    print("CHROMA-WEIGHTED MSE LOSS UNIT TESTS")
    print("==================================================")
    test_1_perfect_prediction()
    test_2_neutral_pixels()
    test_3_high_chroma_pixels()
    test_4_weight_direction()
    test_5_gradients()
    test_6_device_compatibility()
    test_7_existing_losses()
    print("==================================================")
    print("ALL 7 TESTS PASSED SUCCESSFULLY!")
    print("==================================================")
