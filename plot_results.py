"""
plot_results.py — Generate all plots for the final report.

Reads logs/train_log.csv and produces:
  - Loss curve (train vs val)
  - mIoU curve (train vs val)
  - Per-class IoU bar chart (final epoch)
  - Confusion matrix heatmap (from test predictions if available)

Usage:
    python plot_results.py                        # from CSV only
    python plot_results.py --conf_matrix          # also plot confusion matrix
"""

import os
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

import config

PLOT_DIR = os.path.join(config.LOG_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

STYLE_COLORS = {
    "train": "#4F8EF7",
    "val":   "#F4774B",
    "bars":  [
        "#4CAF50", "#81C784", "#D4A017", "#8D6E63",
        "#9E9E9E", "#F48FB1", "#6D4C41", "#BDBDBD",
        "#F5DEB3", "#87CEEB"
    ]
}


# ─────────────────────────────────────────────────────────────────────────────
# Load CSV log
# ─────────────────────────────────────────────────────────────────────────────
def load_csv(path: str) -> dict:
    data = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                data.setdefault(k, []).append(float(v) if k != "epoch" else int(v))
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────
def plot_loss_and_miou(data: dict):
    epochs = data["epoch"]
    fig = plt.figure(figsize=(14, 5))
    gs  = GridSpec(1, 2, figure=fig, hspace=0.05, wspace=0.3)

    # Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, data["train_loss"], color=STYLE_COLORS["train"],
             lw=2, label="Train Loss")
    ax1.plot(epochs, data["val_loss"],   color=STYLE_COLORS["val"],
             lw=2, label="Val Loss", linestyle="--")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training vs Validation Loss", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)

    # mIoU
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, data["train_miou"], color=STYLE_COLORS["train"],
             lw=2, label="Train mIoU")
    ax2.plot(epochs, data["val_miou"],   color=STYLE_COLORS["val"],
             lw=2, label="Val mIoU", linestyle="--")
    best_epoch = epochs[int(np.argmax(data["val_miou"]))]
    best_miou  = max(data["val_miou"])
    ax2.axvline(best_epoch, color="gray", linestyle=":", alpha=0.7)
    ax2.text(best_epoch + 0.3, best_miou, f"Best {best_miou:.3f}", fontsize=9, color="gray")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("mIoU", fontsize=12)
    ax2.set_title("Training vs Validation mIoU", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    plt.suptitle("Training Curves — Duality AI Hackathon", fontsize=14, fontweight="bold", y=1.01)
    out = os.path.join(PLOT_DIR, "training_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_per_class_iou(data: dict):
    """Bar chart of per-class IoU at the best epoch (highest val_miou)."""
    best_idx = int(np.argmax(data["val_miou"]))
    iou_vals = [data[f"iou_{c}"][best_idx] for c in config.CLASS_NAMES]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(config.CLASS_NAMES, iou_vals, color=STYLE_COLORS["bars"])

    # Annotate each bar
    for bar, val in zip(bars, iou_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

    ax.axhline(np.mean(iou_vals), color="red", linestyle="--", lw=1.5,
               label=f"Mean IoU = {np.mean(iou_vals):.3f}")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("IoU", fontsize=12)
    ax.set_title(f"Per-Class IoU at Best Epoch ({data['epoch'][best_idx]})",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    plt.xticks(rotation=30, ha="right", fontsize=10)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "per_class_iou.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_confusion_matrix(conf_matrix: np.ndarray, title="Confusion Matrix"):
    """Normalised confusion matrix heatmap."""
    import matplotlib.colors as mcolors
    norm = conf_matrix.astype(float)
    row_sums = norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    norm = norm / row_sums

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    n = len(config.CLASS_NAMES)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(config.CLASS_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(config.CLASS_NAMES, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Ground Truth", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")

    for i in range(n):
        for j in range(n):
            color = "white" if norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color=color)

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "confusion_matrix.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_csv",     default=os.path.join(config.LOG_DIR, "train_log.csv"))
    parser.add_argument("--conf_matrix", action="store_true",
                        help="Also generate confusion matrix from a saved numpy file")
    parser.add_argument("--conf_npy",    default="results/confusion_matrix.npy",
                        help="Path to .npy confusion matrix (NUM_CLASSES x NUM_CLASSES)")
    args = parser.parse_args()

    print(f"\n[Plots] Reading log: {args.log_csv}")
    if not os.path.exists(args.log_csv):
        print("[ERROR] Log file not found. Run train.py first.")
        exit(1)

    data = load_csv(args.log_csv)
    plot_loss_and_miou(data)
    plot_per_class_iou(data)

    if args.conf_matrix:
        if os.path.exists(args.conf_npy):
            conf = np.load(args.conf_npy)
            plot_confusion_matrix(conf)
        else:
            print(f"[WARN] Confusion matrix file not found: {args.conf_npy}")

    print(f"\n[Done] All plots saved to: {PLOT_DIR}/")
