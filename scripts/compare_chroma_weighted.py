"""Comprehensive comparative analysis script: Chroma-Weighted MSE vs Baseline MSE vs Smooth L1.

Generates:
1. outputs/analysis/chroma_weighted_vs_mse.csv
2. outputs/analysis/chroma_weighted_visual_comparison.png (All 10 test images, 5 columns)
3. Detailed metrics for foliage case ADE_train_00000028.jpg
4. Chroma retention statistics across all models
5. Training dynamics comparisons
"""

import os
import sys
import math
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
    print("STARTING CHROMA-WEIGHTED MSE COMPARATIVE ANALYSIS")
    print("==================================================")
    
    device = get_device()
    print(f"Using device: {device}")
    
    analysis_dir = os.path.join(PROJECT_ROOT, "outputs", "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Checkpoint paths
    mse_chk_path = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "best.pth")
    sl1_chk_path = os.path.join(PROJECT_ROOT, "outputs", "experiments", "smooth_l1", "checkpoints", "best.pth")
    cw_chk_path = os.path.join(PROJECT_ROOT, "outputs", "experiments", "chroma_weighted", "checkpoints", "best.pth")
    
    if not os.path.isfile(cw_chk_path):
        raise FileNotFoundError(f"Chroma-weighted best checkpoint not found at: {cw_chk_path}")
        
    print(f"Loading MSE Best Checkpoint: {mse_chk_path}")
    mse_chk = torch.load(mse_chk_path, map_location=device, weights_only=False)
    mse_model = ColorizationUNet().to(device)
    mse_model.load_state_dict(mse_chk["model_state_dict"])
    mse_model.eval()
    mse_epoch = mse_chk.get("best_epoch", mse_chk.get("epoch", 0) + 1)
    
    print(f"Loading Smooth L1 Best Checkpoint: {sl1_chk_path}")
    sl1_chk = torch.load(sl1_chk_path, map_location=device, weights_only=False)
    sl1_model = ColorizationUNet().to(device)
    sl1_model.load_state_dict(sl1_chk["model_state_dict"])
    sl1_model.eval()
    sl1_epoch = sl1_chk.get("best_epoch", sl1_chk.get("epoch", 0) + 1)
    
    print(f"Loading Chroma-Weighted MSE Best Checkpoint: {cw_chk_path}")
    cw_chk = torch.load(cw_chk_path, map_location=device, weights_only=False)
    cw_model = ColorizationUNet().to(device)
    cw_model.load_state_dict(cw_chk["model_state_dict"])
    cw_model.eval()
    cw_epoch = cw_chk.get("best_epoch", cw_chk.get("epoch", 0) + 1)
    
    print(f"\nMSE Best Epoch: {mse_epoch} | Smooth L1 Best Epoch: {sl1_epoch} | Chroma-Weighted Best Epoch: {cw_epoch}\n")
    
    # Load fixed test images
    test_paths = load_fixed_test_images()
    print(f"Loaded {len(test_paths)} fixed test images.")
    
    image_records = []
    
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
            
            # Ground truth chroma: sqrt(a^2 + b^2) in Lab scale ([-128, 127])
            gt_ab_raw = lab[:, :, 1:]
            gt_chroma = np.mean(np.sqrt(gt_ab_raw[:, :, 0]**2 + gt_ab_raw[:, :, 1]**2))
            
            L_tensor = torch.from_numpy(L_norm).unsqueeze(0).float().to(device)
            
            # 1. MSE prediction
            pred_mse_tensor = mse_model(L_tensor)
            pred_mse_ab = pred_mse_tensor.squeeze(0).cpu().numpy().astype(np.float32)
            pred_mse_rgb = reconstruct_rgb(L_norm, pred_mse_ab)
            mse_mae, mse_mse, mse_psnr = compute_rgb_metrics(gt_rgb, pred_mse_rgb)
            mse_ab_raw = np.transpose(pred_mse_ab * 128.0, (1, 2, 0))
            mse_chroma = np.mean(np.sqrt(mse_ab_raw[:, :, 0]**2 + mse_ab_raw[:, :, 1]**2))
            
            # 2. Smooth L1 prediction
            pred_sl1_tensor = sl1_model(L_tensor)
            pred_sl1_ab = pred_sl1_tensor.squeeze(0).cpu().numpy().astype(np.float32)
            pred_sl1_rgb = reconstruct_rgb(L_norm, pred_sl1_ab)
            sl1_mae, sl1_mse, sl1_psnr = compute_rgb_metrics(gt_rgb, pred_sl1_rgb)
            sl1_ab_raw = np.transpose(pred_sl1_ab * 128.0, (1, 2, 0))
            sl1_chroma = np.mean(np.sqrt(sl1_ab_raw[:, :, 0]**2 + sl1_ab_raw[:, :, 1]**2))
            
            # 3. Chroma-Weighted MSE prediction
            pred_cw_tensor = cw_model(L_tensor)
            pred_cw_ab = pred_cw_tensor.squeeze(0).cpu().numpy().astype(np.float32)
            pred_cw_rgb = reconstruct_rgb(L_norm, pred_cw_ab)
            cw_mae, cw_mse, cw_psnr = compute_rgb_metrics(gt_rgb, pred_cw_rgb)
            cw_ab_raw = np.transpose(pred_cw_ab * 128.0, (1, 2, 0))
            cw_chroma = np.mean(np.sqrt(cw_ab_raw[:, :, 0]**2 + cw_ab_raw[:, :, 1]**2))
            
            psnr_diff = cw_psnr - mse_psnr
            if abs(psnr_diff) < 0.2:
                better_model = "Tied"
            elif psnr_diff > 0:
                better_model = "Chroma-Weighted"
            else:
                better_model = "MSE"
                
            L_vis = (L_norm[0] + 1.0) / 2.0
            
            rec = {
                "idx": idx,
                "image_id": img_id,
                "input_l": L_vis,
                "ground_truth": gt_rgb,
                "orig_rgb": orig_rgb,
                "pred_mse": pred_mse_rgb,
                "pred_sl1": pred_sl1_rgb,
                "pred_cw": pred_cw_rgb,
                "gt_ab_raw": gt_ab_raw,
                "mse_ab_raw": mse_ab_raw,
                "sl1_ab_raw": sl1_ab_raw,
                "cw_ab_raw": cw_ab_raw,
                "mse_mae": mse_mae,
                "sl1_mae": sl1_mae,
                "cw_mae": cw_mae,
                "mse_mse": mse_mse,
                "sl1_mse": sl1_mse,
                "cw_mse": cw_mse,
                "mse_psnr": mse_psnr,
                "sl1_psnr": sl1_psnr,
                "cw_psnr": cw_psnr,
                "psnr_diff": psnr_diff,
                "mae_diff": cw_mae - mse_mae,
                "mse_diff": cw_mse - mse_mse,
                "better_model": better_model,
                "gt_chroma": gt_chroma,
                "mse_chroma": mse_chroma,
                "sl1_chroma": sl1_chroma,
                "cw_chroma": cw_chroma,
                "mse_chroma_ratio": mse_chroma / (gt_chroma + 1e-6),
                "sl1_chroma_ratio": sl1_chroma / (gt_chroma + 1e-6),
                "cw_chroma_ratio": cw_chroma / (gt_chroma + 1e-6)
            }
            image_records.append(rec)
            print(f"Image {idx+1:02d} ({img_id[:20]}): MSE PSNR={mse_psnr:.2f}dB | CW PSNR={cw_psnr:.2f}dB | Diff={psnr_diff:+.2f}dB | {better_model}")
            
    # Save CSV: chroma_weighted_vs_mse.csv
    csv_path = os.path.join(analysis_dir, "chroma_weighted_vs_mse.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_id",
            "mse_mae", "chroma_mae", "mae_diff",
            "mse_mse", "chroma_mse", "mse_diff",
            "mse_psnr", "chroma_psnr", "psnr_diff",
            "gt_chroma", "mse_chroma", "chroma_weighted_chroma",
            "better_model"
        ])
        for r in image_records:
            writer.writerow([
                r["image_id"],
                f"{r['mse_mae']:.6f}", f"{r['cw_mae']:.6f}", f"{r['mae_diff']:+.6f}",
                f"{r['mse_mse']:.6f}", f"{r['cw_mse']:.6f}", f"{r['mse_diff']:+.6f}",
                f"{r['mse_psnr']:.2f}", f"{r['cw_psnr']:.2f}", f"{r['psnr_diff']:+.2f}",
                f"{r['gt_chroma']:.2f}", f"{r['mse_chroma']:.2f}", f"{r['cw_chroma']:.2f}",
                r["better_model"]
            ])
    print(f"\nSaved CSV: {csv_path}")
    
    # Visual comparison panel: 10 images x 5 columns
    fig, axes = plt.subplots(10, 5, figsize=(18, 30))
    fig.suptitle(
        f"Chroma-Weighted MSE vs Baseline MSE & Smooth L1\n"
        f"MSE (Ep {mse_epoch}) | Smooth L1 (Ep {sl1_epoch}) | Chroma-Weighted MSE (Ep {cw_epoch})",
        fontsize=16, fontweight="bold", y=0.995
    )
    col_names = [
        "Input Grayscale (L*)",
        "Ground Truth RGB",
        f"MSE Baseline (Ep {mse_epoch})",
        f"Smooth L1 (Ep {sl1_epoch})",
        f"Chroma-Weighted (Ep {cw_epoch})"
    ]
    
    for row_idx, r in enumerate(image_records):
        ax0 = axes[row_idx, 0]
        ax1 = axes[row_idx, 1]
        ax2 = axes[row_idx, 2]
        ax3 = axes[row_idx, 3]
        ax4 = axes[row_idx, 4]
        
        ax0.imshow(r["input_l"], cmap="gray", vmin=0.0, vmax=1.0)
        ax1.imshow(np.clip(r["ground_truth"], 0.0, 1.0))
        ax2.imshow(np.clip(r["pred_mse"], 0.0, 1.0))
        ax3.imshow(np.clip(r["pred_sl1"], 0.0, 1.0))
        ax4.imshow(np.clip(r["pred_cw"], 0.0, 1.0))
        
        if row_idx == 0:
            for c_idx, title in enumerate(col_names):
                axes[row_idx, c_idx].set_title(title, fontsize=12, fontweight="bold")
                
        ax0.set_ylabel(f"{r['image_id'][:16]}\n{r['better_model']}", fontsize=9, fontweight="bold")
        ax1.set_xlabel(f"GT Chroma: {r['gt_chroma']:.1f}", fontsize=8)
        ax2.set_xlabel(f"PSNR: {r['mse_psnr']:.2f}dB\nChroma: {r['mse_chroma']:.1f}", fontsize=8)
        ax3.set_xlabel(f"PSNR: {r['sl1_psnr']:.2f}dB\nChroma: {r['sl1_chroma']:.1f}", fontsize=8)
        ax4.set_xlabel(f"PSNR: {r['cw_psnr']:.2f}dB\nChroma: {r['cw_chroma']:.1f}", fontsize=8)
        
        for ax in (ax0, ax1, ax2, ax3, ax4):
            ax.set_xticks([])
            ax.set_yticks([])
            
    plt.tight_layout(rect=[0, 0.01, 1, 0.99])
    plot_path = os.path.join(analysis_dir, "chroma_weighted_visual_comparison.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved visual comparison panel: {plot_path}")
    
    # Foliage case deep-dive: ADE_train_00000028.jpg
    foliage_rec = None
    for r in image_records:
        if r["image_id"] == "ADE_train_00000028.jpg":
            foliage_rec = r
            break
            
    if foliage_rec:
        print("\n==================================================")
        print("FOLIAGE CASE STUDY (ADE_train_00000028.jpg)")
        print("==================================================")
        gt_ab = foliage_rec["gt_ab_raw"]
        mse_ab = foliage_rec["mse_ab_raw"]
        sl1_ab = foliage_rec["sl1_ab_raw"]
        cw_ab = foliage_rec["cw_ab_raw"]
        
        gt_a, gt_b = gt_ab[:, :, 0], gt_ab[:, :, 1]
        mse_a, mse_b = mse_ab[:, :, 0], mse_ab[:, :, 1]
        sl1_a, sl1_b = sl1_ab[:, :, 0], sl1_ab[:, :, 1]
        cw_a, cw_b = cw_ab[:, :, 0], cw_ab[:, :, 1]
        
        print(f"Target a* (mean +/- std): {np.mean(gt_a):.2f} +/- {np.std(gt_a):.2f} [min: {np.min(gt_a):.2f}, max: {np.max(gt_a):.2f}]")
        print(f"Target b* (mean +/- std): {np.mean(gt_b):.2f} +/- {np.std(gt_b):.2f} [min: {np.min(gt_b):.2f}, max: {np.max(gt_b):.2f}]")
        print(f"MSE a* (mean +/- std):    {np.mean(mse_a):.2f} +/- {np.std(mse_a):.2f}")
        print(f"MSE b* (mean +/- std):    {np.mean(mse_b):.2f} +/- {np.std(mse_b):.2f}")
        print(f"SL1 a* (mean +/- std):    {np.mean(sl1_a):.2f} +/- {np.std(sl1_a):.2f}")
        print(f"SL1 b* (mean +/- std):    {np.mean(sl1_b):.2f} +/- {np.std(sl1_b):.2f}")
        print(f"CW a* (mean +/- std):     {np.mean(cw_a):.2f} +/- {np.std(cw_a):.2f}")
        print(f"CW b* (mean +/- std):     {np.mean(cw_b):.2f} +/- {np.std(cw_b):.2f}")
        
        print(f"\nMean Chroma: GT={foliage_rec['gt_chroma']:.2f} | MSE={foliage_rec['mse_chroma']:.2f} | SL1={foliage_rec['sl1_chroma']:.2f} | CW={foliage_rec['cw_chroma']:.2f}")
        print(f"Chroma Retention Ratio: MSE={foliage_rec['mse_chroma_ratio']*100:.1f}% | SL1={foliage_rec['sl1_chroma_ratio']*100:.1f}% | CW={foliage_rec['cw_chroma_ratio']*100:.1f}%")
        print(f"PSNR: MSE={foliage_rec['mse_psnr']:.2f} dB | SL1={foliage_rec['sl1_psnr']:.2f} dB | CW={foliage_rec['cw_psnr']:.2f} dB")
        
    # Aggregate Metrics across all 10 images
    print("\n==================================================")
    print("AGGREGATE TEST SET METRICS (10 FIXED IMAGES)")
    print("==================================================")
    mean_mse_mae = np.mean([r["mse_mae"] for r in image_records])
    mean_sl1_mae = np.mean([r["sl1_mae"] for r in image_records])
    mean_cw_mae = np.mean([r["cw_mae"] for r in image_records])
    
    mean_mse_mse = np.mean([r["mse_mse"] for r in image_records])
    mean_sl1_mse = np.mean([r["sl1_mse"] for r in image_records])
    mean_cw_mse = np.mean([r["cw_mse"] for r in image_records])
    
    mean_mse_psnr = np.mean([r["mse_psnr"] for r in image_records])
    mean_sl1_psnr = np.mean([r["sl1_psnr"] for r in image_records])
    mean_cw_psnr = np.mean([r["cw_psnr"] for r in image_records])
    
    mean_gt_chroma = np.mean([r["gt_chroma"] for r in image_records])
    mean_mse_chroma = np.mean([r["mse_chroma"] for r in image_records])
    mean_sl1_chroma = np.mean([r["sl1_chroma"] for r in image_records])
    mean_cw_chroma = np.mean([r["cw_chroma"] for r in image_records])
    
    mean_mse_retention = np.mean([r["mse_chroma_ratio"] for r in image_records]) * 100
    mean_sl1_retention = np.mean([r["sl1_chroma_ratio"] for r in image_records]) * 100
    mean_cw_retention = np.mean([r["cw_chroma_ratio"] for r in image_records]) * 100
    
    cw_wins = sum(1 for r in image_records if r["better_model"] == "Chroma-Weighted")
    mse_wins = sum(1 for r in image_records if r["better_model"] == "MSE")
    ties = sum(1 for r in image_records if r["better_model"] == "Tied")
    
    print(f"MAE:  MSE={mean_mse_mae:.6f} | SL1={mean_sl1_mae:.6f} | CW={mean_cw_mae:.6f}")
    print(f"MSE:  MSE={mean_mse_mse:.6f} | SL1={mean_sl1_mse:.6f} | CW={mean_cw_mse:.6f}")
    print(f"PSNR: MSE={mean_mse_psnr:.2f} dB | SL1={mean_sl1_psnr:.2f} dB | CW={mean_cw_psnr:.2f} dB")
    print(f"Mean Chroma: GT={mean_gt_chroma:.2f} | MSE={mean_mse_chroma:.2f} | SL1={mean_sl1_chroma:.2f} | CW={mean_cw_chroma:.2f}")
    print(f"Chroma Retention: MSE={mean_mse_retention:.1f}% | SL1={mean_sl1_retention:.1f}% | CW={mean_cw_retention:.1f}%")
    print(f"Win/Tie/Loss vs MSE: CW Wins={cw_wins} | MSE Wins={mse_wins} | Ties={ties}")

if __name__ == "__main__":
    main()
