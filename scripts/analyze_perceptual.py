"""Comparative analysis script for Perceptual Loss Colorization (Step 7.4).

Loads the perceptual experiment checkpoint and compares against Baseline MSE
across the 10 fixed evaluation test targets, evaluating:
- MAE, MSE, and PSNR in RGB space
- Chroma retention metrics
- Foliage deep dive (ADE_train_00000028.jpg)
- Visual comparison panels
"""

import os
import sys
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import ColorizationUNet
from src.device import get_device
from src.utils import load_rgb_image, resize_image, rgb_to_lab, normalize_lab, reconstruct_rgb
from src.evaluate import load_fixed_test_images, compute_rgb_metrics

def main():
    print("==================================================")
    print("PERCEPTUAL LOSS COLORIZATION ANALYSIS")
    print("==================================================")
    
    device = get_device()
    print(f"Device: {device}")
    
    analysis_dir = os.path.join(PROJECT_ROOT, "outputs", "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Checkpoints
    mse_chk_path = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "best.pth")
    perc_chk_path = os.path.join(PROJECT_ROOT, "outputs", "experiments", "perceptual", "checkpoints", "best.pth")
    
    if not os.path.isfile(perc_chk_path):
        print(f"[INFO] Perceptual checkpoint not yet generated at: {perc_chk_path}")
        print("This script will be used for post-training evaluation once the 20-epoch experiment runs.")
        return
        
    print(f"Loading Baseline MSE: {mse_chk_path}")
    mse_chk = torch.load(mse_chk_path, map_location=device, weights_only=False)
    mse_model = ColorizationUNet().to(device)
    mse_model.load_state_dict(mse_chk["model_state_dict"])
    mse_model.eval()
    mse_epoch = mse_chk.get("best_epoch", mse_chk.get("epoch", 0) + 1)
    
    print(f"Loading Perceptual: {perc_chk_path}")
    perc_chk = torch.load(perc_chk_path, map_location=device, weights_only=False)
    perc_model = ColorizationUNet().to(device)
    perc_model.load_state_dict(perc_chk["model_state_dict"])
    perc_model.eval()
    perc_epoch = perc_chk.get("best_epoch", perc_chk.get("epoch", 0) + 1)
    
    test_paths = load_fixed_test_images()
    print(f"Loaded {len(test_paths)} test images.")
    
    records = []
    
    with torch.no_grad():
        for idx, rel_path in enumerate(test_paths):
            full_path = os.path.join(PROJECT_ROOT, rel_path)
            img_id = os.path.basename(rel_path)
            
            raw_rgb = load_rgb_image(full_path)
            resized_rgb = resize_image(raw_rgb)
            orig_rgb = resized_rgb.astype(np.float32) / 255.0
            
            lab = rgb_to_lab(orig_rgb)
            L_norm, ab_norm = normalize_lab(lab)
            gt_rgb = reconstruct_rgb(L_norm, ab_norm)
            
            gt_ab_raw = lab[:, :, 1:]
            gt_chroma = np.mean(np.sqrt(gt_ab_raw[:, :, 0]**2 + gt_ab_raw[:, :, 1]**2))
            
            L_tensor = torch.from_numpy(L_norm).unsqueeze(0).float().to(device)
            
            # Baseline MSE
            pred_mse_tensor = mse_model(L_tensor)
            pred_mse_ab = pred_mse_tensor.squeeze(0).cpu().numpy().astype(np.float32)
            pred_mse_rgb = reconstruct_rgb(L_norm, pred_mse_ab)
            mse_mae, mse_mse, mse_psnr = compute_rgb_metrics(gt_rgb, pred_mse_rgb)
            mse_ab_raw = np.transpose(pred_mse_ab * 128.0, (1, 2, 0))
            mse_chroma = np.mean(np.sqrt(mse_ab_raw[:, :, 0]**2 + mse_ab_raw[:, :, 1]**2))
            
            # Perceptual
            pred_perc_tensor = perc_model(L_tensor)
            pred_perc_ab = pred_perc_tensor.squeeze(0).cpu().numpy().astype(np.float32)
            pred_perc_rgb = reconstruct_rgb(L_norm, pred_perc_ab)
            perc_mae, perc_mse, perc_psnr = compute_rgb_metrics(gt_rgb, pred_perc_rgb)
            perc_ab_raw = np.transpose(pred_perc_ab * 128.0, (1, 2, 0))
            perc_chroma = np.mean(np.sqrt(perc_ab_raw[:, :, 0]**2 + perc_ab_raw[:, :, 1]**2))
            
            psnr_diff = perc_psnr - mse_psnr
            if abs(psnr_diff) < 0.2:
                better = "Tied"
            elif psnr_diff > 0:
                better = "Perceptual"
            else:
                better = "MSE"
                
            rec = {
                "image_id": img_id,
                "input_l": (L_norm[0] + 1.0) / 2.0,
                "gt_rgb": gt_rgb,
                "pred_mse": pred_mse_rgb,
                "pred_perc": pred_perc_rgb,
                "gt_chroma": gt_chroma,
                "mse_chroma": mse_chroma,
                "perc_chroma": perc_chroma,
                "mse_mae": mse_mae,
                "perc_mae": perc_mae,
                "mse_mse": mse_mse,
                "perc_mse": perc_mse,
                "mse_psnr": mse_psnr,
                "perc_psnr": perc_psnr,
                "psnr_diff": psnr_diff,
                "better": better
            }
            records.append(rec)
            print(f"Image {idx+1:02d} ({img_id[:20]}): MSE PSNR={mse_psnr:.2f}dB | Perc PSNR={perc_psnr:.2f}dB | Diff={psnr_diff:+.2f}dB | {better}")
            
    # Save CSV
    csv_path = os.path.join(analysis_dir, "perceptual_vs_mse.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_id", "mse_mae", "perc_mae", "mse_mse", "perc_mse",
            "mse_psnr", "perc_psnr", "psnr_diff", "gt_chroma", "mse_chroma",
            "perc_chroma", "better_model"
        ])
        for r in records:
            writer.writerow([
                r["image_id"], f"{r['mse_mae']:.6f}", f"{r['perc_mae']:.6f}",
                f"{r['mse_mse']:.6f}", f"{r['perc_mse']:.6f}",
                f"{r['mse_psnr']:.2f}", f"{r['perc_psnr']:.2f}", f"{r['psnr_diff']:+.2f}",
                f"{r['gt_chroma']:.2f}", f"{r['mse_chroma']:.2f}", f"{r['perc_chroma']:.2f}",
                r["better"]
            ])
    print(f"\nSaved CSV: {csv_path}")

if __name__ == "__main__":
    main()
