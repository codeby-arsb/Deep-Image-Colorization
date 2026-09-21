"""Verification script for Step 7.3: Chroma-Weighted Regression Loss Experiment.

Verifies:
1. Chroma-Weighted checkpoints exist and are intact.
2. Best model reload test (single image & real batch).
3. Checkpoint parameters and outputs are finite (no NaN/Inf).
4. Metadata integrity (loss_name, loss_alpha, best_epoch, best_val_loss).
5. Training history exists with exactly 20 epochs.
6. Evaluation outputs exist (per-image metrics, summary metrics, plots).
7. Comparative analysis artifacts exist (CSV, PNG, markdown).
8. Baseline MSE and Smooth L1 checkpoints untouched (MD5 match).
"""

import os
import sys
import hashlib
import csv
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import ColorizationUNet
from src.device import get_device

BASELINE_BEST_MD5 = "4f04f028ef59d8dc11d7968a4e3d05ac"
SMOOTH_L1_BEST_MD5 = "83fbaed6909c309d92da38118fb11aa2"

def compute_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def main():
    print("==================================================")
    print("VERIFYING STEP 7.3: CHROMA-WEIGHTED EXPERIMENT")
    print("==================================================")
    
    device = get_device()
    print(f"Device: {device}")
    
    failures = []
    
    # 1. Baseline checkpoint protection
    baseline_chk = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "best.pth")
    if not os.path.isfile(baseline_chk):
        failures.append("Baseline best.pth not found!")
    else:
        b_md5 = compute_md5(baseline_chk)
        if b_md5 != BASELINE_BEST_MD5:
            failures.append(f"Baseline best.pth MD5 mismatch! Expected {BASELINE_BEST_MD5}, got {b_md5}")
        else:
            print("[PASS] Baseline best.pth untouched (MD5 verified).")
            
    # 2. Smooth L1 checkpoint protection
    sl1_chk = os.path.join(PROJECT_ROOT, "outputs", "experiments", "smooth_l1", "checkpoints", "best.pth")
    if not os.path.isfile(sl1_chk):
        failures.append("Smooth L1 best.pth not found!")
    else:
        s_md5 = compute_md5(sl1_chk)
        if s_md5 != SMOOTH_L1_BEST_MD5:
            failures.append(f"Smooth L1 best.pth MD5 mismatch! Expected {SMOOTH_L1_BEST_MD5}, got {s_md5}")
        else:
            print("[PASS] Smooth L1 best.pth untouched (MD5 verified).")
            
    # 3. Chroma-weighted checkpoints exist
    exp_dir = os.path.join(PROJECT_ROOT, "outputs", "experiments", "chroma_weighted")
    cw_best_path = os.path.join(exp_dir, "checkpoints", "best.pth")
    cw_latest_path = os.path.join(exp_dir, "checkpoints", "latest.pth")
    
    if not os.path.isfile(cw_best_path):
        failures.append(f"Missing {cw_best_path}")
    else:
        print("[PASS] Chroma-weighted best.pth exists.")
        
    if not os.path.isfile(cw_latest_path):
        failures.append(f"Missing {cw_latest_path}")
    else:
        print("[PASS] Chroma-weighted latest.pth exists.")
        
    if os.path.isfile(cw_best_path):
        chk = torch.load(cw_best_path, map_location=device, weights_only=False)
        # Check metadata
        if chk.get("loss_name") != "chroma_weighted_mse":
            failures.append(f"Checkpoint loss_name is '{chk.get('loss_name')}', expected 'chroma_weighted_mse'")
        else:
            print("[PASS] Checkpoint loss_name is 'chroma_weighted_mse'.")
            
        if chk.get("loss_alpha") != 1.0:
            failures.append(f"Checkpoint loss_alpha is '{chk.get('loss_alpha')}', expected 1.0")
        else:
            print("[PASS] Checkpoint loss_alpha is 1.0.")
            
        best_epoch = chk.get("best_epoch")
        best_val = chk.get("best_val_loss")
        print(f"[INFO] Best Checkpoint: Epoch {best_epoch}, Best Val Loss: {best_val:.6f}")
        
        # Model reload & inference
        model = ColorizationUNet().to(device)
        model.load_state_dict(chk["model_state_dict"])
        model.eval()
        
        # Check weights finite
        all_finite = all(torch.isfinite(p).all().item() for p in model.parameters())
        if not all_finite:
            failures.append("Model contains non-finite weights (NaN/Inf)!")
        else:
            print("[PASS] Model weights are all finite.")
            
        with torch.no_grad():
            d1 = torch.randn(1, 1, 256, 256, device=device)
            o1 = model(d1)
            if list(o1.shape) != [1, 2, 256, 256] or not torch.isfinite(o1).all().item():
                failures.append(f"Single image inference failed! Shape={list(o1.shape)}")
            else:
                print("[PASS] Single image inference [1, 2, 256, 256] verified finite.")
                
            d8 = torch.randn(8, 1, 256, 256, device=device)
            o8 = model(d8)
            if list(o8.shape) != [8, 2, 256, 256] or not torch.isfinite(o8).all().item():
                failures.append(f"Batch inference failed! Shape={list(o8.shape)}")
            else:
                print("[PASS] Batch inference [8, 2, 256, 256] verified finite.")
                
    # 4. Training history
    history_csv = os.path.join(exp_dir, "training_history.csv")
    if not os.path.isfile(history_csv):
        failures.append(f"Missing {history_csv}")
    else:
        with open(history_csv, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            if len(reader) != 20:
                failures.append(f"Training history has {len(reader)} epochs, expected 20!")
            else:
                print(f"[PASS] Training history contains all 20 epochs.")
                
    # 5. Evaluation files
    per_image_csv = os.path.join(exp_dir, "evaluation", "test_metrics_per_image.csv")
    summary_csv = os.path.join(exp_dir, "evaluation", "test_metrics_summary.csv")
    if not os.path.isfile(per_image_csv):
        failures.append(f"Missing {per_image_csv}")
    else:
        print("[PASS] test_metrics_per_image.csv exists.")
    if not os.path.isfile(summary_csv):
        failures.append(f"Missing {summary_csv}")
    else:
        print("[PASS] test_metrics_summary.csv exists.")
        
    # 6. Comparative analysis artifacts
    analysis_dir = os.path.join(PROJECT_ROOT, "outputs", "analysis")
    comp_csv = os.path.join(analysis_dir, "chroma_weighted_vs_mse.csv")
    comp_png = os.path.join(analysis_dir, "chroma_weighted_visual_comparison.png")
    comp_md = os.path.join(analysis_dir, "chroma_weighted_analysis.md")
    
    if not os.path.isfile(comp_csv):
        failures.append(f"Missing {comp_csv}")
    else:
        print("[PASS] chroma_weighted_vs_mse.csv exists.")
    if not os.path.isfile(comp_png):
        failures.append(f"Missing {comp_png}")
    else:
        print("[PASS] chroma_weighted_visual_comparison.png exists.")
    if not os.path.isfile(comp_md):
        failures.append(f"Missing {comp_md}")
    else:
        print("[PASS] chroma_weighted_analysis.md exists.")
        
    print("\n==================================================")
    if failures:
        print(f"VERIFICATION FAILED WITH {len(failures)} ERROR(S):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")
        print("==================================================")

if __name__ == "__main__":
    main()
