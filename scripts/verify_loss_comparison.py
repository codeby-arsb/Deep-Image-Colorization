"""Verification script for Step 7.2: MSE vs Smooth L1 Comparative Analysis.

Verifies:
1. Both experiment artifacts and checkpoints exist.
2. Checkpoints can be loaded into fresh models.
3. Expected evaluation files exist.
4. Comparison CSV is valid with all expected columns.
5. No NaN or Inf values are present in outputs.
6. Baseline files remain unchanged (verified via MD5 hashes).
"""

import os
import sys
import math
import hashlib
import csv
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import ColorizationUNet

# Known baseline MD5 checksums
BASELINE_BEST_MD5 = "4f04f028ef59d8dc11d7968a4e3d05ac"
BASELINE_LATEST_MD5 = "f81eb228b6cbe5aec9ee7dcdf2e156c3"
BASELINE_HISTORY_MD5 = "1013a6b636880c6bbe056591bc7d74af"

def compute_md5(path):
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def main():
    print("==================================================")
    print("VERIFYING LOSS COMPARISON ARTIFACTS & INTEGRITY")
    print("==================================================")
    
    # 1. Check baseline files and verify unchanged MD5
    print("\n--- 1. Baseline Integrity Check ---")
    base_best = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "best.pth")
    base_latest = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "latest.pth")
    base_hist = os.path.join(PROJECT_ROOT, "outputs", "training_history.csv")
    
    assert os.path.isfile(base_best), f"Missing {base_best}"
    assert os.path.isfile(base_latest), f"Missing {base_latest}"
    assert os.path.isfile(base_hist), f"Missing {base_hist}"
    
    best_md5 = compute_md5(base_best)
    latest_md5 = compute_md5(base_latest)
    hist_md5 = compute_md5(base_hist)
    
    assert best_md5 == BASELINE_BEST_MD5, f"Baseline best.pth altered! MD5: {best_md5}"
    assert latest_md5 == BASELINE_LATEST_MD5, f"Baseline latest.pth altered! MD5: {latest_md5}"
    assert hist_md5 == BASELINE_HISTORY_MD5, f"Baseline history altered! MD5: {hist_md5}"
    print("  [PASS] Baseline checkpoints and history are 100% UNTOUCHED.")
    
    # 2. Check Smooth L1 files
    print("\n--- 2. Smooth L1 Artifacts Check ---")
    sl1_dir = os.path.join(PROJECT_ROOT, "outputs", "experiments", "smooth_l1")
    sl1_best = os.path.join(sl1_dir, "checkpoints", "best.pth")
    sl1_latest = os.path.join(sl1_dir, "checkpoints", "latest.pth")
    sl1_hist = os.path.join(sl1_dir, "training_history.csv")
    
    assert os.path.isfile(sl1_best), f"Missing {sl1_best}"
    assert os.path.isfile(sl1_latest), f"Missing {sl1_latest}"
    assert os.path.isfile(sl1_hist), f"Missing {sl1_hist}"
    print("  [PASS] Smooth L1 checkpoints and history exist.")
    
    # 3. Check model reloading & forward passes
    print("\n--- 3. Model Reloading & Inference Check ---")
    for name, chk_path in [("MSE Baseline", base_best), ("Smooth L1", sl1_best)]:
        chk = torch.load(chk_path, map_location="cpu", weights_only=False)
        model = ColorizationUNet()
        model.load_state_dict(chk["model_state_dict"])
        model.eval()
        
        all_finite = all(torch.isfinite(p).all().item() for p in model.parameters())
        assert all_finite, f"Non-finite parameters in {name}"
        
        dummy = torch.randn(1, 1, 256, 256)
        with torch.no_grad():
            out = model(dummy)
        assert list(out.shape) == [1, 2, 256, 256], f"Invalid shape in {name}: {out.shape}"
        assert torch.isfinite(out).all().item(), f"Non-finite output in {name}"
        print(f"  [PASS] {name} model reloaded successfully; forward pass finite and shape correct.")
        
    # 4. Check analysis files
    print("\n--- 4. Analysis Artifacts Check ---")
    analysis_dir = os.path.join(PROJECT_ROOT, "outputs", "analysis")
    report_md = os.path.join(analysis_dir, "loss_comparison.md")
    summary_txt = os.path.join(analysis_dir, "loss_comparison_summary.txt")
    per_image_csv = os.path.join(analysis_dir, "loss_comparison_per_image.csv")
    plot_all = os.path.join(analysis_dir, "visual_comparison_all.png")
    plot_rep = os.path.join(analysis_dir, "visual_comparison_representative.png")
    plot_dyn = os.path.join(analysis_dir, "loss_dynamics_comparison.png")
    
    for f_path in [report_md, summary_txt, per_image_csv, plot_all, plot_rep, plot_dyn]:
        assert os.path.isfile(f_path), f"Missing analysis artifact: {f_path}"
        assert os.path.getsize(f_path) > 0, f"Empty file: {f_path}"
        print(f"  [PASS] Verified: {os.path.basename(f_path)} ({os.path.getsize(f_path)} bytes)")
        
    # 5. Check CSV validity & values
    print("\n--- 5. Comparison CSV Validation ---")
    with open(per_image_csv, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    assert len(reader) == 10, f"Expected 10 rows in per_image_csv, found {len(reader)}"
    
    expected_cols = [
        "image_id", "mse_model_mae", "sl1_model_mae", "mae_diff_sl1_minus_mse",
        "mse_model_mse", "sl1_model_mse", "mse_diff_sl1_minus_mse",
        "mse_model_psnr", "sl1_model_psnr", "psnr_diff_sl1_minus_mse",
        "better_model"
    ]
    for col in expected_cols:
        assert col in reader[0], f"Missing expected column: {col}"
        
    for row in reader:
        for k in ["mse_model_mae", "sl1_model_mae", "mse_model_mse", "sl1_model_mse", "mse_model_psnr", "sl1_model_psnr"]:
            val = float(row[k])
            assert not math.isnan(val) and not math.isinf(val), f"NaN/Inf in {k}: {val}"
            assert val > 0, f"Non-positive value in {k}: {val}"
    print("  [PASS] All 10 rows and all numerical metrics valid and finite (no NaN, no Inf).")
    
    print("\n==================================================")
    print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    main()
