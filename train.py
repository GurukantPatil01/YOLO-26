"""
train.py — Main training script.

Usage:
    python train.py
    python train.py --resume checkpoints/best.pth

Features:
  - Mixed-precision training (AMP)
  - Cosine / Step / Plateau LR scheduling
  - Best checkpoint saving (top-K by val mIoU)
  - Per-epoch CSV logging for easy plotting
  - Early stopping
"""

import os
import csv
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import segmentation_models_pytorch as smp

import config
from dataset import get_dataloaders
from losses import get_loss_fn
from metrics import SegmentationMetrics


# ─────────────────────────────────────────────────────────────────────────────
# Seeding
# ─────────────────────────────────────────────────────────────────────────────
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────
def build_model() -> nn.Module:
    arch_map = {
        "DeepLabV3Plus": smp.DeepLabV3Plus,
        "DeepLabV3":     smp.DeepLabV3,
        "Unet":          smp.Unet,
        "UnetPlusPlus":  smp.UnetPlusPlus,
        "FPN":           smp.FPN,
        "PAN":           smp.PAN,
    }
    assert config.ARCHITECTURE in arch_map, \
        f"Unknown architecture: {config.ARCHITECTURE}. Choose from {list(arch_map.keys())}"

    model = arch_map[config.ARCHITECTURE](
        encoder_name=config.ENCODER,
        encoder_weights=config.ENCODER_WEIGHTS,
        in_channels=3,
        classes=config.NUM_CLASSES,
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# LR Scheduler
# ─────────────────────────────────────────────────────────────────────────────
def build_scheduler(optimizer, num_epochs: int, steps_per_epoch: int):
    if config.LR_SCHEDULER == "cosine":
        # Warmup (linear) + cosine decay
        warmup_steps = config.LR_WARMUP_EPOCHS * steps_per_epoch
        total_steps  = num_epochs * steps_per_epoch

        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1 + np.cos(np.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif config.LR_SCHEDULER == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=15, gamma=0.5
        )
    elif config.LR_SCHEDULER == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=5, factor=0.5
        )
    else:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# One training epoch
# ─────────────────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, loss_fn, scaler, device, scheduler=None):
    model.train()
    total_loss = 0.0
    metrics    = SegmentationMetrics()
    step       = 0

    pbar = tqdm(loader, desc="  TRAIN", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        optimizer.zero_grad()

        with autocast(enabled=config.USE_AMP):
            logits = model(images)
            loss   = loss_fn(logits, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        if scheduler and config.LR_SCHEDULER in ("cosine", "step"):
            scheduler.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        metrics.update(preds, masks)

        step += 1
        if step % config.LOG_EVERY == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    results  = metrics.compute()
    return avg_loss, results["miou"]


# ─────────────────────────────────────────────────────────────────────────────
# One validation epoch
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def val_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    metrics    = SegmentationMetrics()

    for images, masks in tqdm(loader, desc="  VAL  ", leave=False):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        with autocast(enabled=config.USE_AMP):
            logits = model(images)
            loss   = loss_fn(logits, masks)

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        metrics.update(preds, masks)

    avg_loss = total_loss / len(loader)
    results  = metrics.compute()
    return avg_loss, results["miou"], results


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint management
# ─────────────────────────────────────────────────────────────────────────────
class CheckpointManager:
    def __init__(self, ckpt_dir: str, top_k: int = 3):
        self.ckpt_dir = ckpt_dir
        self.top_k    = top_k
        self.scores   = []   # list of (miou, filepath)
        os.makedirs(ckpt_dir, exist_ok=True)

    def save(self, model, optimizer, epoch, miou, extra: dict = None):
        path = os.path.join(self.ckpt_dir, f"epoch_{epoch:03d}_miou_{miou:.4f}.pth")
        payload = {
            "epoch":      epoch,
            "miou":       miou,
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        self.scores.append((miou, path))
        self.scores.sort(key=lambda x: x[0], reverse=True)

        # Save a separate "best.pth" always pointing to the best
        if self.scores[0][1] == path:
            best_path = os.path.join(self.ckpt_dir, "best.pth")
            torch.save(payload, best_path)
            print(f"  ★ New best model saved → best.pth  (mIoU={miou:.4f})")

        # Prune low-scoring checkpoints
        while len(self.scores) > self.top_k:
            _, old_path = self.scores.pop()
            if os.path.exists(old_path):
                os.remove(old_path)

        return path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    seed_everything(config.SEED)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR,        exist_ok=True)

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader = get_dataloaders()

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model().to(device)
    print(f"[Model] {config.ARCHITECTURE} + {config.ENCODER}")
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[Model] Parameters: {params_m:.1f}M")

    # ── Loss ──────────────────────────────────────────────────────────────────
    loss_fn = get_loss_fn().to(device)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # ── Scheduler ─────────────────────────────────────────────────────────────
    scheduler = build_scheduler(
        optimizer,
        num_epochs=config.NUM_EPOCHS,
        steps_per_epoch=len(train_loader),
    )
    plateau_scheduler = None
    if config.LR_SCHEDULER == "plateau":
        plateau_scheduler = scheduler
        scheduler = None

    scaler = GradScaler(enabled=config.USE_AMP)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 1
    best_miou   = 0.0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_miou   = ckpt.get("miou", 0.0)
        print(f"[Resume] From epoch {ckpt['epoch']}  (best mIoU so far: {best_miou:.4f})")

    ckpt_manager  = CheckpointManager(config.CHECKPOINT_DIR, top_k=config.SAVE_TOP_K)
    no_improve    = 0

    # ── CSV log ───────────────────────────────────────────────────────────────
    log_path = os.path.join(config.LOG_DIR, "train_log.csv")
    write_header = not os.path.exists(log_path) or start_epoch == 1
    log_file = open(log_path, "a", newline="")
    writer   = csv.writer(log_file)
    if write_header:
        writer.writerow(["epoch", "train_loss", "train_miou", "val_loss", "val_miou",
                         "lr", "time_s"] + [f"iou_{c}" for c in config.CLASS_NAMES])

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Starting training — {config.NUM_EPOCHS} epochs")
    print(f"{'='*55}\n")

    for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{config.NUM_EPOCHS}  |  LR={current_lr:.2e}")

        train_loss, train_miou = train_epoch(
            model, train_loader, optimizer, loss_fn, scaler, device,
            scheduler=(scheduler if config.LR_SCHEDULER != "plateau" else None)
        )

        val_loss, val_miou, val_results = val_epoch(model, val_loader, loss_fn, device)

        elapsed = time.time() - t0
        print(f"  train_loss={train_loss:.4f}  train_mIoU={train_miou:.4f}")
        print(f"  val_loss  ={val_loss:.4f}  val_mIoU  ={val_miou:.4f}  [{elapsed:.0f}s]")

        # Plateau scheduler step
        if plateau_scheduler:
            plateau_scheduler.step(val_miou)

        # Checkpoint
        ckpt_manager.save(model, optimizer, epoch, val_miou)

        # Early stopping
        if val_miou > best_miou:
            best_miou  = val_miou
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= config.EARLY_STOP_PATIENCE:
                print(f"\n[Early Stop] No improvement for {config.EARLY_STOP_PATIENCE} epochs.")
                break

        # CSV row
        row = [epoch, f"{train_loss:.6f}", f"{train_miou:.6f}",
               f"{val_loss:.6f}", f"{val_miou:.6f}", f"{current_lr:.2e}",
               f"{elapsed:.1f}"]
        row += [f"{v:.6f}" for v in val_results["iou_per_class"]]
        writer.writerow(row)
        log_file.flush()

        print()

    log_file.close()
    print(f"\n[Done] Best val mIoU: {best_miou:.4f}")
    print(f"[Done] Logs saved to  : {log_path}")
    print(f"[Done] Best model at  : {config.CHECKPOINT_DIR}/best.pth")


if __name__ == "__main__":
    main()
