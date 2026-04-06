"""
infer_fast.py — Run inference using cached DINOv2 + trained head.

Works with the test dataset. Generates:
  - predictions/masks/           raw class-ID masks
  - predictions/masks_color/     RGB coloured masks
  - predictions/comparisons/     side-by-side (image | GT | pred) for N samples
  - predictions/evaluation_metrics.txt  (if Segmentation/ folder exists)

Usage:
    python3 infer_fast.py
    python3 infer_fast.py --test_dir /path/to/test --model checkpoints/best_fast.pth
"""

import os, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Same model as training ──────────────────────────────────────────────────
class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        self.stem  = nn.Sequential(nn.Conv2d(in_channels, 128, 7, padding=3), nn.GELU())
        self.block = nn.Sequential(
            nn.Conv2d(128, 128, 7, padding=3, groups=128), nn.GELU(),
            nn.Conv2d(128, 128, 1), nn.GELU())
        self.classifier = nn.Conv2d(128, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        return self.classifier(self.block(self.stem(x)))


# ─── Class info ──────────────────────────────────────────────────────────────
CLASS_NAMES = ['Background','Trees','Lush Bushes','Dry Grass','Dry Bushes',
               'Ground Clutter','Logs','Rocks','Landscape','Sky']
VALUE_MAP = {0:0, 100:1, 200:2, 300:3, 500:4, 550:5, 700:6, 800:7, 7100:8, 10000:9}
COLOR_PALETTE = np.array([
    [0,0,0],[34,139,34],[0,255,0],[210,180,140],[139,90,43],
    [128,128,0],[139,69,19],[128,128,128],[160,82,45],[135,206,235]
], dtype=np.uint8)

def convert_mask(mask_img):
    arr = np.array(mask_img)
    out = np.zeros_like(arr, dtype=np.uint8)
    for raw, idx in VALUE_MAP.items():
        out[arr == raw] = idx
    return out

def mask_to_color(mask):
    h, w = mask.shape
    out  = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(len(COLOR_PALETTE)):
        out[mask == c] = COLOR_PALETTE[c]
    return out

def compute_iou(pred, target, n=10):
    ious = []
    for c in range(n):
        inter = ((pred == c) & (target == c)).sum()
        union = ((pred == c) | (target == c)).sum()
        ious.append(inter / union if union > 0 else float("nan"))
    return float(np.nanmean(ious)), ious


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", default="/Users/gurukantpatil/Downloads/Offroad_Segmentation_Training_Dataset/val",
                        help="Dir with Color_Images/ (and optionally Segmentation/)")
    parser.add_argument("--model",    default="checkpoints/best_fast.pth")
    parser.add_argument("--out_dir",  default="predictions")
    parser.add_argument("--n_compare",type=int, default=10,
                        help="How many side-by-side comparison images to save")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for sub in ["masks","masks_color","comparisons"]:
        os.makedirs(os.path.join(args.out_dir, sub), exist_ok=True)

    device = torch.device("cpu")

    # Image dimensions
    w = int(((960/2)//14)*14)
    h = int(((540/2)//14)*14)
    tokenW, tokenH = w//14, h//14

    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # Load backbone
    print("Loading DINOv2 from cache...")
    backbone = torch.hub.load(
        repo_or_dir="/Users/gurukantpatil/.cache/torch/hub/facebookresearch_dinov2_main",
        model="dinov2_vits14", source="local"
    )
    backbone.eval().to(device)

    # Load model
    print(f"Loading segmentation head from {args.model}...")
    model = SegmentationHeadConvNeXt(384, 10, tokenW, tokenH).to(device)
    ckpt  = torch.load(args.model, map_location=device)
    # Handle both raw state_dict and wrapped checkpoint
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    img_dir  = os.path.join(args.test_dir, "Color_Images")
    mask_dir = os.path.join(args.test_dir, "Segmentation")
    has_gt   = os.path.isdir(mask_dir)
    if has_gt:
        print(f"GT masks found → will compute IoU")
    else:
        print(f"No GT masks → inference only")

    files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
    print(f"\nRunning inference on {len(files)} images...\n")

    all_miou, all_class_iou = [], []
    sample_count = 0

    for fname in tqdm(files, desc="Inference"):
        stem = os.path.splitext(fname)[0]

        # Load image
        img_pil = Image.open(os.path.join(img_dir, fname)).convert("RGB")
        img_t   = transform(img_pil).unsqueeze(0).to(device)

        # Extract features + predict using TTA (Test-Time Augmentation)
        with torch.no_grad():
            # Pass 1: Normal
            feats1  = backbone.forward_features(img_t)["x_norm_patchtokens"]
            logits1 = model(feats1)
            logits1 = F.interpolate(logits1, size=(h, w), mode="bilinear", align_corners=False)
            
            # Pass 2: Horizontally Flipped
            img_t_flipped = torch.flip(img_t, dims=[3])
            feats2  = backbone.forward_features(img_t_flipped)["x_norm_patchtokens"]
            logits2 = model(feats2)
            logits2 = F.interpolate(logits2, size=(h, w), mode="bilinear", align_corners=False)
            logits2 = torch.flip(logits2, dims=[3])  # Un-flip the semantic map
            
            # Average the logits to ensemble the predictions
            logits_avg = (logits1 + logits2) / 2.0
            pred   = logits_avg.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Save raw mask
        Image.fromarray(pred).save(os.path.join(args.out_dir, "masks", f"{stem}_pred.png"))

        # Save colour mask
        color_pred = mask_to_color(pred)
        Image.fromarray(color_pred).save(os.path.join(args.out_dir, "masks_color", f"{stem}_pred_color.png"))

        # IoU if GT available
        if has_gt:
            mask_path = os.path.join(mask_dir, fname)
            if os.path.exists(mask_path):
                gt = convert_mask(Image.open(mask_path))
                # Resize GT to match pred
                gt_pil = Image.fromarray(gt).resize((w, h), Image.NEAREST)
                gt     = np.array(gt_pil)
                miou, class_iou = compute_iou(pred, gt)
                all_miou.append(miou)
                all_class_iou.append(class_iou)

        # Save comparison image (first N)
        if sample_count < args.n_compare:
            orig_np = np.array(img_pil.resize((w, h)))
            panels  = [orig_np, color_pred]
            titles  = ["Input Image", "Prediction"]

            if has_gt and os.path.exists(os.path.join(mask_dir, fname)):
                gt_color = mask_to_color(np.array(gt_pil.convert("L")))
                panels  = [orig_np, mask_to_color(np.array(Image.fromarray(convert_mask(Image.open(os.path.join(mask_dir, fname)))).resize((w,h),Image.NEAREST))), color_pred]
                titles  = ["Input Image", "Ground Truth", "Prediction"]

            fig, axes = plt.subplots(1, len(panels), figsize=(5*len(panels), 5))
            for ax, panel, title in zip(axes, panels, titles):
                ax.imshow(panel); ax.set_title(title, fontsize=12); ax.axis("off")
            plt.suptitle(fname, fontsize=10)
            plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir, "comparisons", f"sample_{sample_count:03d}.png"), dpi=100)
            plt.close()
            sample_count += 1

    # Summary
    print(f"\n{'='*50}")
    if all_miou:
        mean_miou    = float(np.nanmean(all_miou))
        mean_per_cls = np.nanmean(all_class_iou, axis=0)
        print(f"  Mean IoU: {mean_miou:.4f}")
        print(f"\n  Per-class IoU:")
        for name, iou in zip(CLASS_NAMES, mean_per_cls):
            bar = "█" * int(iou * 20)
            print(f"  {name:<20} {iou:.4f}  {bar}")

        # Save metrics
        metrics_path = os.path.join(args.out_dir, "evaluation_metrics.txt")
        with open(metrics_path, "w") as f:
            f.write(f"Mean IoU: {mean_miou:.4f}\n\nPer-Class IoU:\n")
            for name, iou in zip(CLASS_NAMES, mean_per_cls):
                f.write(f"  {name:<20}: {iou:.4f}\n")
        print(f"\n  Saved: {metrics_path}")

        # Bar chart
        fig, ax = plt.subplots(figsize=(11, 5))
        colors  = [COLOR_PALETTE[i]/255 for i in range(len(CLASS_NAMES))]
        bars    = ax.bar(CLASS_NAMES, mean_per_cls, color=colors, edgecolor="black")
        ax.axhline(mean_miou, color="red", linestyle="--", lw=1.5, label=f"Mean={mean_miou:.3f}")
        for bar, val in zip(bars, mean_per_cls):
            if not np.isnan(val):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                        f"{val:.3f}", ha="center", fontsize=8, fontweight="bold")
        ax.set_ylim(0, 1.1); ax.set_ylabel("IoU"); ax.legend()
        ax.set_title(f"Per-Class IoU — Mean={mean_miou:.4f}", fontsize=13, fontweight="bold")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "per_class_iou.png"), dpi=150)
        plt.close()
        print(f"  Saved: {args.out_dir}/per_class_iou.png")
    print(f"{'='*50}")
    print(f"\n✅ Done! Outputs in {args.out_dir}/")


if __name__ == "__main__":
    main()
