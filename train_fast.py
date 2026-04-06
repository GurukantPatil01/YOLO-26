"""
train_fast.py — FAST training using pre-cached DINOv2 features.

Run precompute_features.py first (one-time, ~5-10 min).
Then this script trains the segmentation head in seconds per epoch.

Usage:
    python3 train_fast.py
    python3 train_fast.py --epochs 50 --lr 5e-4
"""

import os
import argparse
import csv
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CACHE_DIR = "./feature_cache"

# ─── Model (identical to official script) ────────────────────────────────────
class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=7, padding=3), nn.GELU())
        self.block = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=7, padding=3, groups=128), nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=1), nn.GELU())
        self.classifier = nn.Conv2d(128, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.block(x)
        return self.classifier(x)


# ─── Cached Feature Dataset ──────────────────────────────────────────────────
class CachedFeatureDataset(Dataset):
    def __init__(self, split: str):
        split_dir   = os.path.join(CACHE_DIR, split)
        all_files   = os.listdir(split_dir)
        self.feat_files = sorted([
            os.path.join(split_dir, f)
            for f in all_files if f.endswith("_feat.npy")
        ])
        self.mask_files = [
            f.replace("_feat.npy", "_mask.npy") for f in self.feat_files
        ]
        assert len(self.feat_files) > 0, \
            f"No cached features found in {split_dir}. Run precompute_features.py first!"

    def __len__(self):
        return len(self.feat_files)

    def __getitem__(self, idx):
        feat = torch.from_numpy(
            np.load(self.feat_files[idx]).astype(np.float32)
        )  # (N_patches, 384)
        mask = torch.from_numpy(
            np.load(self.mask_files[idx]).astype(np.int64)
        ).squeeze(0)  # (H, W)
        return feat, mask


# ─── Metrics ─────────────────────────────────────────────────────────────────
def compute_iou(pred_logits, target, n_classes=10):
    pred = pred_logits.argmax(dim=1).view(-1)
    tgt  = target.view(-1)
    ious = []
    for c in range(n_classes):
        inter = ((pred == c) & (tgt == c)).sum().float()
        union = ((pred == c) | (tgt == c)).sum().float()
        ious.append((inter / union).item() if union > 0 else float("nan"))
    return float(np.nanmean(ious)), ious


# ─── Training ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int,   default=8)   # bigger = faster
    parser.add_argument("--n_classes",  type=int,   default=10)
    parser.add_argument("--resume",     type=str,   default=None)
    args = parser.parse_args()

    os.makedirs("train_stats", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    device = torch.device("cpu")
    print(f"Device: {device}")

    # Image patch grid dimensions (must match precompute_features.py)
    w = int(((960 / 2) // 14) * 14)
    h = int(((540 / 2) // 14) * 14)
    tokenW, tokenH = w // 14, h // 14

    # Datasets
    train_ds = CachedFeatureDataset("train")
    val_ds   = CachedFeatureDataset("val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Model
    model = SegmentationHeadConvNeXt(
        in_channels=384, out_channels=args.n_classes,
        tokenW=tokenW, tokenH=tokenH
    ).to(device)

    # Optimizer + scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Custom Class Weights to fix imbalance
    # ['Background','Trees','Lush Bushes','Dry Grass','Dry Bushes','Ground Clutter','Logs','Rocks','Landscape','Sky']
    weights = torch.tensor([1.0, 2.0, 2.0, 1.0, 5.0, 10.0, 20.0, 5.0, 1.0, 0.5], device=device)
    ce_loss_fn = nn.CrossEntropyLoss(weight=weights)
    
    def dice_loss(pred, target, smooth=1e-5):
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=args.n_classes).permute(0, 3, 1, 2).float()
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()
    
    def combo_loss_fn(logits, masks):
        return ce_loss_fn(logits, masks) + 0.5 * dice_loss(logits, masks)

    loss_fn = combo_loss_fn

    start_epoch = 1
    best_miou   = 0.0

    # Resume
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_miou   = ckpt.get("miou", 0.0)
        print(f"Resumed from epoch {ckpt['epoch']}  (best mIoU={best_miou:.4f})")

    # CSV log
    log_path = "train_stats/fast_train_log.csv"
    log_file = open(log_path, "w", newline="")
    writer   = csv.writer(log_file)
    writer.writerow(["epoch","train_loss","train_miou","val_loss","val_miou","time_s"])

    history = {"train_loss":[],"val_loss":[],"train_miou":[],"val_miou":[]}

    print(f"\n{'='*60}")
    print(f"  Fast Training — {args.epochs} epochs on cached features")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_losses, train_ious = [], []

        for feats, masks in tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs} TRAIN", leave=False):
            feats, masks = feats.to(device), masks.to(device)
            logits  = model(feats)
            # Upsample logits to match mask spatial size
            logits  = F.interpolate(logits, size=masks.shape[1:], mode="bilinear", align_corners=False)
            loss    = loss_fn(logits, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            iou, _ = compute_iou(logits.detach(), masks, args.n_classes)
            train_ious.append(iou)

        scheduler.step()

        # Validation
        model.eval()
        val_losses, val_ious = [], []
        with torch.no_grad():
            for feats, masks in tqdm(val_loader, desc=f"Epoch {epoch:02d}/{args.epochs} VAL  ", leave=False):
                feats, masks = feats.to(device), masks.to(device)
                logits = model(feats)
                logits = F.interpolate(logits, size=masks.shape[1:], mode="bilinear", align_corners=False)
                loss   = loss_fn(logits, masks)
                val_losses.append(loss.item())
                iou, _ = compute_iou(logits, masks, args.n_classes)
                val_ious.append(iou)

        tl = np.mean(train_losses); ti = float(np.nanmean(train_ious))
        vl = np.mean(val_losses);   vi = float(np.nanmean(val_ious))
        elapsed = time.time() - t0

        history["train_loss"].append(tl); history["train_miou"].append(ti)
        history["val_loss"].append(vl);   history["val_miou"].append(vi)

        print(f"Epoch {epoch:02d}  train_loss={tl:.4f}  train_mIoU={ti:.4f}  "
              f"val_loss={vl:.4f}  val_mIoU={vi:.4f}  [{elapsed:.0f}s]")
        writer.writerow([epoch, f"{tl:.6f}", f"{ti:.6f}", f"{vl:.6f}", f"{vi:.6f}", f"{elapsed:.1f}"])
        log_file.flush()

        # Save best
        if vi > best_miou:
            best_miou = vi
            torch.save({
                "epoch": epoch, "miou": vi,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }, "checkpoints/best_fast.pth")
            print(f"  ★ New best! mIoU={vi:.4f}  saved → checkpoints/best_fast.pth")

        # Also save segmentation_head.pth for test_segmentation.py compatibility
        torch.save(model.state_dict(), "segmentation_head.pth")

    log_file.close()
    print(f"\n✅ Training complete! Best val mIoU: {best_miou:.4f}")
    print(f"   Weights: segmentation_head.pth")
    print(f"   Log:     {log_path}")

    # Quick plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    eps = range(1, len(history["train_loss"]) + 1)
    ax1.plot(eps, history["train_loss"], label="Train"); ax1.plot(eps, history["val_loss"], label="Val")
    ax1.set_title("Loss"); ax1.legend(); ax1.grid(True)
    ax2.plot(eps, history["train_miou"], label="Train"); ax2.plot(eps, history["val_miou"], label="Val")
    ax2.set_title("mIoU"); ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    plt.savefig("train_stats/training_curves.png", dpi=150)
    print("   Plot:    train_stats/training_curves.png")


if __name__ == "__main__":
    main()
