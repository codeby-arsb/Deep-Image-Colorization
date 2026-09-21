"""Comparative analysis script for Deep Image Colorization: MSE vs Smooth L1.

Generates:
1. outputs/analysis/loss_comparison_per_image.csv
2. outputs/analysis/visual_comparison_all.png (all 10 fixed test images)
3. outputs/analysis/visual_comparison_representative.png (4 key representative images)
4. outputs/analysis/loss_dynamics_comparison.png (training & validation dynamics side-by-side)
5. Comprehensive numerical and chromatic statistics.
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
    print("STARTING MSE VS SMOOTH L1 COMPARATIVE ANALYSIS")
    print("==================================================")
    
    device = get_device()
    print(f"Using device: {device}")
    
    analysis_dir = os.path.join(PROJECT_ROOT, "outputs", "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # 1. Load models
    mse_chk_path = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "best.pth")
    sl1_chk_path = os.path.join(PROJECT_ROOT, "outputs", "experiments", "smooth_l1", "checkpoints", "best.pth")
    
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
    
    print(f"MSE Best Epoch: {mse_epoch} | Smooth L1 Best Epoch: {sl1_epoch}\n")
    
    # 2. Load fixed test images
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
            
            # Ground truth chroma: sqrt(a^2 + b^2) in original Lab scale
            gt_ab_raw = lab[:, :, 1:]
            gt_chroma = np.mean(np.sqrt(gt_ab_raw[:, :, 0]**2 + gt_ab_raw[:, :, 1]**2))
            
            L_tensor = torch.from_numpy(L_norm).unsqueeze(0).float().to(device)
            
            # MSE prediction
            pred_mse_tensor = mse_model(L_tensor)
            pred_mse_ab = pred_mse_tensor.squeeze(0).cpu().numpy().astype(np.float32)
            pred_mse_rgb = reconstruct_rgb(L_norm, pred_mse_ab)
            mse_mae, mse_mse, mse_psnr = compute_rgb_metrics(gt_rgb, pred_mse_rgb)
            
            mse_ab_raw = np.transpose(pred_mse_ab * 128.0, (1, 2, 0))
            mse_chroma = np.mean(np.sqrt(mse_ab_raw[:, :, 0]**2 + mse_ab_raw[:, :, 1]**2))
            
            # Smooth L1 prediction
            pred_sl1_tensor = sl1_model(L_tensor)
            pred_sl1_ab = pred_sl1_tensor.squeeze(0).cpu().numpy().astype(np.float32)
            pred_sl1_rgb = reconstruct_rgb(L_norm, pred_sl1_ab)
            sl1_mae, sl1_mse, sl1_psnr = compute_rgb_metrics(gt_rgb, pred_sl1_rgb)
            
            sl1_ab_raw = np.transpose(pred_sl1_ab * 128.0, (1, 2, 0))
            sl1_chroma = np.mean(np.sqrt(sl1_ab_raw[:, :, 0]**2 + sl1_ab_raw[:, :, 1]**2))
            
            psnr_diff = sl1_psnr - mse_psnr
            if abs(psnr_diff) < 0.2:
                winner = "Tied"
            elif psnr_diff > 0:
                winner = "Smooth L1"
            else:
                winner = "MSE"
                
            L_vis = (L_norm[0] + 1.0) / 2.0
            
            rec = {
                "idx": idx,
                "image_id": img_id,
                "input_l": L_vis,
                "ground_truth": gt_rgb,
                "orig_rgb": orig_rgb,
                "pred_mse": pred_mse_rgb,
                "pred_sl1": pred_sl1_rgb,
                "mse_mae": mse_mae,
                "sl1_mae": sl1_mae,
                "mse_mse": mse_mse,
                "sl1_mse": sl1_mse,
                "mse_psnr": mse_psnr,
                "sl1_psnr": sl1_psnr,
                "psnr_diff": psnr_diff,
                "winner": winner,
                "gt_chroma": gt_chroma,
                "mse_chroma": mse_chroma,
                "sl1_chroma": sl1_chroma,
                "mse_chroma_ratio": mse_chroma / (gt_chroma + 1e-6),
                "sl1_chroma_ratio": sl1_chroma / (gt_chroma + 1e-6)
            }
            image_records.append(rec)
            print(f"Image {idx+1:02d} ({img_id[:20]}): MSE PSNR={mse_psnr:.2f}dB | SL1 PSNR={sl1_psnr:.2f}dB | Diff={psnr_diff:+.2f}dB | {winner}")

    # 3. Save CSV
    csv_path = os.path.join(analysis_dir, "loss_comparison_per_image.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_id",
            "mse_model_mae", "sl1_model_mae", "mae_diff_sl1_minus_mse",
            "mse_model_mse", "sl1_model_mse", "mse_diff_sl1_minus_mse",
            "mse_model_psnr", "sl1_model_psnr", "psnr_diff_sl1_minus_mse",
            "gt_mean_chroma", "mse_mean_chroma", "sl1_mean_chroma",
            "better_model"
        ])
        for r in image_records:
            writer.writerow([
                r["image_id"],
                f"{r['mse_mae']:.6f}", f"{r['sl1_mae']:.6f}", f"{r['sl1_mae'] - r['mse_mae']:+.6f}",
                f"{r['mse_mse']:.6f}", f"{r['sl1_mse']:.6f}", f"{r['sl1_mse'] - r['mse_mse']:+.6f}",
                f"{r['mse_psnr']:.2f}", f"{r['sl1_psnr']:.2f}", f"{r['psnr_diff']:+.2f}",
                f"{r['gt_chroma']:.2f}", f"{r['mse_chroma']:.2f}", f"{r['sl1_chroma']:.2f}",
                r["winner"]
            ])
    print(f"\nSaved CSV: {csv_path}")

    # 4. Generate All 10 Images Comparison Sheet
    fig_all, axes_all = plt.subplots(10, 4, figsize=(14, 28))
    fig_all.suptitle("MSE vs Smooth L1 — Complete 10-Image Side-by-Side Comparison\n"
                     f"MSE Best (Epoch {mse_epoch}) vs Smooth L1 Best (Epoch {sl1_epoch})",
                     fontsize=15, fontweight="bold", y=0.995)
    
    col_names = ["Input Grayscale (L*)", "Ground Truth RGB", f"MSE (Epoch {mse_epoch})", f"Smooth L1 (Epoch {sl1_epoch})"]
    
    for row_idx, r in enumerate(image_records):
        ax0 = axes_all[row_idx, 0]
        ax1 = axes_all[row_idx, 1]
        ax2 = axes_all[row_idx, 2]
        ax3 = axes_all[row_idx, 3]
        
        ax0.imshow(r["input_l"], cmap="gray", vmin=0.0, vmax=1.0)
        ax1.imshow(np.clip(r["ground_truth"], 0.0, 1.0))
        ax2.imshow(np.clip(r["pred_mse"], 0.0, 1.0))
        ax3.imshow(np.clip(r["pred_sl1"], 0.0, 1.0))
        
        if row_idx == 0:
            for c_idx, title in enumerate(col_names):
                axes_all[row_idx, c_idx].set_title(title, fontsize=12, fontweight="bold")
                
        ax0.set_ylabel(f"{r['image_id'][:16]}\n{r['winner']}", fontsize=9, fontweight="bold")
        ax2.set_xlabel(f"PSNR: {r['mse_psnr']:.2f}dB | MAE: {r['mse_mae']:.3f}", fontsize=8)
        ax3.set_xlabel(f"PSNR: {r['sl1_psnr']:.2f}dB | MAE: {r['sl1_mae']:.3f}", fontsize=8)
        
        for ax in (ax0, ax1, ax2, ax3):
            ax.set_xticks([])
            ax.set_yticks([])
            
    plt.tight_layout(rect=[0, 0.01, 1, 0.99])
    plot_all_path = os.path.join(analysis_dir, "visual_comparison_all.png")
    plt.savefig(plot_all_path, dpi=150)
    plt.close()
    print(f"Saved: {plot_all_path}")

    # 5. Generate Focused Representative 4-Image Sheet
    # 1. Smooth L1 better: ADE_train_00000324.jpg (+0.87 dB)
    # 2. MSE better: ADE_train_00000378.jpg (+0.76 dB for MSE)
    # 3. Similar / Tied: ADE_train_00000203.jpg (Tied at 26.22 dB)
    # 4. Difficult foliage / high chroma: ADE_train_00000028.jpg (outdoor vegetation field, PSNR ~19.1 dB)
    rep_ids = [
        ("ADE_train_00000324.jpg", "Smooth L1 Outperforms (+0.87 dB)"),
        ("ADE_train_00000378.jpg", "MSE Outperforms (+0.76 dB)"),
        ("ADE_train_00000203.jpg", "Identical Performance (Tied 26.22 dB)"),
        ("ADE_train_00000028.jpg", "Difficult Foliage / Vegetation (Tied ~19.15 dB)")
    ]
    
    rep_records = []
    for target_id, label in rep_ids:
        for r in image_records:
            if r["image_id"] == target_id:
                rep_records.append((r, label))
                break
                
    fig_rep, axes_rep = plt.subplots(4, 4, figsize=(15, 14))
    fig_rep.suptitle("MSE vs Smooth L1 — Key Representative Case Studies\n"
                     f"Comparing Colorization Quality, Chroma, and Reconstruction Fidelity",
                     fontsize=14, fontweight="bold", y=0.99)
                     
    for row_idx, (r, label) in enumerate(rep_records):
        ax0 = axes_rep[row_idx, 0]
        ax1 = axes_rep[row_idx, 1]
        ax2 = axes_rep[row_idx, 2]
        ax3 = axes_rep[row_idx, 3]
        
        ax0.imshow(r["input_l"], cmap="gray", vmin=0.0, vmax=1.0)
        ax1.imshow(np.clip(r["ground_truth"], 0.0, 1.0))
        ax2.imshow(np.clip(r["pred_mse"], 0.0, 1.0))
        ax3.imshow(np.clip(r["pred_sl1"], 0.0, 1.0))
        
        if row_idx == 0:
            for c_idx, title in enumerate(col_names):
                axes_rep[row_idx, c_idx].set_title(title, fontsize=12, fontweight="bold")
                
        ax0.set_ylabel(f"{label}\n{r['image_id'][:16]}", fontsize=9, fontweight="bold")
        ax1.set_xlabel(f"GT Chroma: {r['gt_chroma']:.1f}", fontsize=8)
        ax2.set_xlabel(f"PSNR: {r['mse_psnr']:.2f}dB | Chroma: {r['mse_chroma']:.1f}", fontsize=8)
        ax3.set_xlabel(f"PSNR: {r['sl1_psnr']:.2f}dB | Chroma: {r['sl1_chroma']:.1f}", fontsize=8)
        
        for ax in (ax0, ax1, ax2, ax3):
            ax.set_xticks([])
            ax.set_yticks([])
            
    plt.tight_layout(rect=[0, 0.01, 1, 0.98])
    plot_rep_path = os.path.join(analysis_dir, "visual_comparison_representative.png")
    plt.savefig(plot_rep_path, dpi=150)
    plt.close()
    print(f"Saved: {plot_rep_path}")

    # 6. Generate Training Dynamics & Evaluation Comparison Plot
    fig_dyn, axes_dyn = plt.subplots(1, 2, figsize=(14, 5))
    
    # Read histories
    def read_csv(path):
        with open(path) as f:
            return [row for row in csv.DictReader(f)]
            
    mse_h = read_csv(os.path.join(PROJECT_ROOT, "outputs", "training_history.csv"))
    sl1_h = read_csv(os.path.join(PROJECT_ROOT, "outputs", "experiments", "smooth_l1", "training_history.csv"))
    
    epochs_range = list(range(1, 21))
    
    # Plot 1: Validation loss progression
    ax_dyn1 = axes_dyn[0]
    ax_dyn1_twin = ax_dyn1.twinx()
    
    m_val = [float(r["val_loss"]) for r in mse_h]
    s_val = [float(r["val_loss"]) for r in sl1_h]
    
    line1 = ax_dyn1.plot(epochs_range, m_val, label="MSE Val Loss (Epoch 13 Min)", color="royalblue", marker="o", linewidth=2)
    line2 = ax_dyn1_twin.plot(epochs_range, s_val, label="Smooth L1 Val Loss (Epoch 12 Min)", color="darkorange", marker="s", linewidth=2)
    
    ax_dyn1.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax_dyn1.set_ylabel("MSE Loss Scale", color="royalblue", fontsize=11, fontweight="bold")
    ax_dyn1_twin.set_ylabel("Smooth L1 Loss Scale", color="darkorange", fontsize=11, fontweight="bold")
    ax_dyn1.set_title("Validation Loss Progression (Twin Scales)", fontsize=12, fontweight="bold")
    ax_dyn1.grid(True, linestyle="--", alpha=0.5)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax_dyn1.legend(lines, labels, loc="upper right")
    
    # Plot 2: Test PSNR progression
    mse_eval_csv = os.path.join(PROJECT_ROOT, "outputs", "evaluation", "test_metrics_summary.csv")
    sl1_eval_csv = os.path.join(PROJECT_ROOT, "outputs", "experiments", "smooth_l1", "evaluation", "test_metrics_summary.csv")
    
    mse_eval_dict = {r["checkpoint"]: r for r in read_csv(mse_eval_csv)}
    sl1_eval_dict = {r["checkpoint"]: r for r in read_csv(sl1_eval_csv)}
    
    eval_epochs = [1, 5, 10, 15, 20]
    m_psnr = [float(mse_eval_dict[f"epoch_{e:02d}"]["mean_psnr"]) for e in eval_epochs]
    s_psnr = [float(sl1_eval_dict[f"epoch_{e:02d}"]["mean_psnr"]) for e in eval_epochs]
    
    ax_dyn2 = axes_dyn[1]
    ax_dyn2.plot(eval_epochs, m_psnr, label="MSE Test PSNR", color="royalblue", marker="o", linewidth=2)
    ax_dyn2.plot(eval_epochs, s_psnr, label="Smooth L1 Test PSNR", color="darkorange", marker="s", linewidth=2)
    
    # Highlight best points
    ax_dyn2.scatter([13], [float(mse_eval_dict["best"]["mean_psnr"])], color="blue", s=100, zorder=5, label="MSE Best Checkpoint (Epoch 13)")
    ax_dyn2.scatter([12], [float(sl1_eval_dict["best"]["mean_psnr"])], color="red", s=100, zorder=5, label="Smooth L1 Best Checkpoint (Epoch 12)")
    
    ax_dyn2.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax_dyn2.set_ylabel("Test Set Mean PSNR (dB)", fontsize=11, fontweight="bold")
    ax_dyn2.set_title("Test Set PSNR Across Training", fontsize=12, fontweight="bold")
    ax_dyn2.grid(True, linestyle="--", alpha=0.5)
    ax_dyn2.legend(loc="lower left")
    
    plt.tight_layout()
    plot_dyn_path = os.path.join(analysis_dir, "loss_dynamics_comparison.png")
    plt.savefig(plot_dyn_path, dpi=150)
    plt.close()
    print(f"Saved: {plot_dyn_path}")
    
    print("\n==================================================")
    print("COMPARATIVE ANALYSIS EXECUTION COMPLETE")
    print("==================================================")

if __name__ == "__main__":
    main()
