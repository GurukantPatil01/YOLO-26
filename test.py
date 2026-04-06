"""
test.py — Inference on the test set.

Usage:
    python test.py --checkpoint checkpoints/best.pth

Outputs:
  - results/  colour-overlaid prediction images
  - results/metrics.json  (if test masks are available)
"""

import os
import json
import argparse
import numpy as np
from PIL import Image
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
import segmentation_models_pytorch as smp

import config
from dataset import TestDataset, DesertSegDataset, get_val_transforms
from metrics import SegmentationMetrics


# ─────────────────────────────────────────────────────────────────────────────
# Colour-overlay helper
# ─────────────────────────────────────────────────────────────────────────────
def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Map class indices to RGB using config.CLASS_PALETTE."""
    h, w  = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, rgb in enumerate(config.CLASS_PALETTE):
        color[mask == cls_idx] = rgb
    return color


def overlay(image_np: np.ndarray, color_mask: np.ndarray, alpha: float = 0.5):
    """Blend original image with colour mask."""
    return (alpha * image_np + (1 - alpha) * color_mask).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────────────
def load_model(ckpt_path: str, device: torch.device):
    arch_map = {
        "DeepLabV3Plus": smp.DeepLabV3Plus,
        "DeepLabV3":     smp.DeepLabV3,
        "Unet":          smp.Unet,
        "UnetPlusPlus":  smp.UnetPlusPlus,
        "FPN":           smp.FPN,
        "PAN":           smp.PAN,
    }
    model = arch_map[config.ARCHITECTURE](
        encoder_name=config.ENCODER,
        encoder_weights=None,       # weights come from checkpoint
        in_channels=3,
        classes=config.NUM_CLASSES,
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    epoch = ckpt.get("epoch", "?")
    miou  = ckpt.get("miou",  "?")
    print(f"[Model] Loaded from epoch {epoch}  (val mIoU={miou})")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# TTA (Test-Time Augmentation) — horizontal flip
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_tta(model, image_tensor: torch.Tensor, device):
    """
    Returns averaged softmax probability map with horizontal-flip TTA.
    image_tensor: (1, C, H, W)
    """
    img = image_tensor.to(device)
    with autocast(enabled=config.USE_AMP):
        logits_orig = model(img)
        logits_flip = model(torch.flip(img, dims=[-1]))
    probs = (
        torch.softmax(logits_orig, dim=1)
        + torch.flip(torch.softmax(logits_flip, dim=1), dims=[-1])
    ) / 2
    return probs


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(config.CHECKPOINT_DIR, "best.pth"),
                        help="Path to model checkpoint")
    parser.add_argument("--test_img_dir", type=str, default=config.TEST_IMG_DIR,
                        help="Directory with test images")
    parser.add_argument("--test_mask_dir", type=str, default=None,
                        help="(Optional) directory with test masks for evaluation")
    parser.add_argument("--use_tta", action="store_true",
                        help="Enable test-time augmentation (horizontal flip)")
    parser.add_argument("--out_dir", type=str, default=config.RESULTS_DIR,
                        help="Where to save prediction overlays")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    model = load_model(args.checkpoint, device)

    transform = get_val_transforms()

    # ── Check if test masks are available for evaluation ──────────────────────
    has_masks = args.test_mask_dir and os.path.isdir(args.test_mask_dir)
    if has_masks:
        print("[Eval] Test masks found — will compute IoU")
        metrics = SegmentationMetrics()

    dataset = TestDataset(args.test_img_dir, transform=transform)
    print(f"[Dataset] {len(dataset)} test images")

    all_preds = {}

    for idx in tqdm(range(len(dataset)), desc="Inference"):
        image_tensor, img_name = dataset[idx]
        image_tensor = image_tensor.unsqueeze(0)   # add batch dim

        # Predict
        if args.use_tta:
            probs = predict_tta(model, image_tensor, device)
        else:
            with torch.no_grad(), autocast(enabled=config.USE_AMP):
                logits = model(image_tensor.to(device))
            probs = torch.softmax(logits, dim=1)

        pred_mask = probs.argmax(dim=1).squeeze(0).cpu().numpy()   # (H, W)
        all_preds[img_name] = pred_mask

        # Compute metrics if masks available
        if has_masks:
            from dataset import remap_mask
            mask_path = os.path.join(
                args.test_mask_dir, img_name
            )
            if os.path.exists(mask_path):
                gt_mask = np.array(Image.open(mask_path))
                gt_mask = remap_mask(gt_mask)
                metrics.update(
                    torch.from_numpy(pred_mask).unsqueeze(0),
                    torch.from_numpy(gt_mask).unsqueeze(0),
                )

        # Save coloured overlay
        stem      = os.path.splitext(img_name)[0]
        orig_img  = np.array(
            Image.open(os.path.join(args.test_img_dir, img_name)).convert("RGB")
        )
        color_mask     = colorize_mask(pred_mask)
        # Resize color_mask to original image size for overlay
        cm_pil = Image.fromarray(color_mask).resize(
            (orig_img.shape[1], orig_img.shape[0]), Image.NEAREST
        )
        blended = overlay(orig_img, np.array(cm_pil))

        # Save side-by-side: original | mask | overlay
        side_by_side = np.concatenate([orig_img, np.array(cm_pil), blended], axis=1)
        Image.fromarray(side_by_side).save(
            os.path.join(args.out_dir, f"{stem}_pred.png")
        )

    print(f"\n[Done] Predictions saved to: {args.out_dir}")

    if has_masks:
        results = metrics.print_report()
        out_json = os.path.join(args.out_dir, "metrics.json")
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[Done] Metrics saved to: {out_json}")


if __name__ == "__main__":
    main()
