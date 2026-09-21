import os
import csv
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_history(path):
    train_loss = []
    val_loss = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
    return train_loss, val_loss

mse_path = os.path.join(PROJECT_ROOT, "outputs", "training_history.csv")
sl1_path = os.path.join(PROJECT_ROOT, "outputs", "experiments", "smooth_l1", "training_history.csv")
cw_path = os.path.join(PROJECT_ROOT, "outputs", "experiments", "chroma_weighted", "training_history.csv")

mse_tr, mse_val = load_history(mse_path)
sl1_tr, sl1_val = load_history(sl1_path)
cw_tr, cw_val = load_history(cw_path)

epochs = list(range(1, 21))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Raw Validation Losses
axes[0].plot(epochs, mse_val, label="Baseline MSE (Best: Ep 13, 0.00978)", color="#1f77b4", marker="o", linewidth=2)
axes[0].plot(epochs, sl1_val, label="Smooth L1 (Best: Ep 12, 0.00487)", color="#ff7f0e", marker="s", linewidth=2)
axes[0].plot(epochs, cw_val, label="Chroma-Weighted MSE (Best: Ep 12, 0.01153)", color="#2ca02c", marker="^", linewidth=2)
axes[0].set_title("Validation Loss Trajectories Across 20 Epochs", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Epoch", fontsize=11)
axes[0].set_ylabel("Validation Loss", fontsize=11)
axes[0].set_xticks(range(2, 21, 2))
axes[0].grid(True, linestyle="--", alpha=0.5)
axes[0].legend(fontsize=10)

# Right: Train vs Val for Chroma-Weighted MSE specifically
axes[1].plot(epochs, cw_tr, label="Chroma-Weighted Train Loss", color="#1b9e77", marker="o", linewidth=2)
axes[1].plot(epochs, cw_val, label="Chroma-Weighted Val Loss", color="#d95f02", marker="s", linewidth=2)
axes[1].axvline(12, color="red", linestyle=":", label="Best Epoch (12)")
axes[1].set_title("Chroma-Weighted MSE Convergence Dynamics", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Epoch", fontsize=11)
axes[1].set_ylabel("Loss Value", fontsize=11)
axes[1].set_xticks(range(2, 21, 2))
axes[1].grid(True, linestyle="--", alpha=0.5)
axes[1].legend(fontsize=10)

plt.tight_layout()
out_plot = os.path.join(PROJECT_ROOT, "outputs", "analysis", "loss_dynamics_comparison_cw.png")
plt.savefig(out_plot, dpi=150)
plt.close()
print(f"Saved: {out_plot}")
