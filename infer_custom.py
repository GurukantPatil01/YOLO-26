"""
infer_custom.py — Run inference on a folder of raw images (no dataset structure needed).

Takes any folder of jpg/png images and outputs coloured segmentation overlays.
"""

import os, sys
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

CLASS_NAMES = ['Background','Trees','Lush Bushes','Dry Grass','Dry Bushes',
               'Ground Clutter','Logs','Rocks','Landscape','Sky']
COLOR_PALETTE = np.array([
    [0,0,0],[34,139,34],[0,255,0],[210,180,140],[139,90,43],
    [128,128,0],[139,69,19],[128,128,128],[160,82,45],[135,206,235]
], dtype=np.uint8)

def mask_to_color(mask):
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(len(COLOR_PALETTE)):
        out[mask == c] = COLOR_PALETTE[c]
    return out

# ── Config ────────────────────────────────────────────────────────────────────
IMG_DIR   = sys.argv[1] if len(sys.argv) > 1 else "/Users/gurukantpatil/Desktop/Test"
MODEL     = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/best_fast.pth"
OUT_DIR   = sys.argv[3] if len(sys.argv) > 3 else "test_predictions"

os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cpu")
w = int(((960/2)//14)*14)
h = int(((540/2)//14)*14)
tokenW, tokenH = w//14, h//14

transform = transforms.Compose([
    transforms.Resize((h, w)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# Load backbone
print("Loading DINOv2...")
backbone = torch.hub.load(
    repo_or_dir="/Users/gurukantpatil/.cache/torch/hub/facebookresearch_dinov2_main",
    model="dinov2_vits14", source="local", verbose=False
)
backbone.eval().to(device)

# Load model
print(f"Loading model from {MODEL}...")
model = SegmentationHeadConvNeXt(384, 10, tokenW, tokenH).to(device)
ckpt  = torch.load(MODEL, map_location=device)
state = ckpt["model"] if "model" in ckpt else ckpt
model.load_state_dict(state)
model.eval()

# Find all images
exts  = {".jpg", ".jpeg", ".png"}
files = sorted([f for f in os.listdir(IMG_DIR) if os.path.splitext(f)[1].lower() in exts])
print(f"\nFound {len(files)} images in {IMG_DIR}\n")

for fname in tqdm(files, desc="Inference"):
    stem    = os.path.splitext(fname)[0]
    img_pil = Image.open(os.path.join(IMG_DIR, fname)).convert("RGB")
    orig_w, orig_h = img_pil.size

    img_t = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        feats  = backbone.forward_features(img_t)["x_norm_patchtokens"]
        logits = model(feats)
        logits = F.interpolate(logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        pred   = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # Colour mask
    color_mask = mask_to_color(pred)

    # Overlay (50% blend)
    orig_np  = np.array(img_pil)
    overlay  = (0.5 * orig_np + 0.5 * color_mask).astype(np.uint8)

    # Legend patches
    from matplotlib.patches import Patch
    legend = [Patch(color=COLOR_PALETTE[i]/255, label=CLASS_NAMES[i]) for i in range(10)]

    # Save side-by-side figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(orig_np);    axes[0].set_title("Input",       fontsize=13); axes[0].axis("off")
    axes[1].imshow(color_mask); axes[1].set_title("Segmentation",fontsize=13); axes[1].axis("off")
    axes[2].imshow(overlay);    axes[2].set_title("Overlay",     fontsize=13); axes[2].axis("off")
    fig.legend(handles=legend, loc="lower center", ncol=5, fontsize=9, framealpha=0.9)
    plt.suptitle(f"DINOv2 Convolutional Head — {fname}\nOverall Validation Metrics: mIoU = 36.15% | Pixel Accuracy = ~81.40%", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.08, 1, 0.9])
    out_path = os.path.join(OUT_DIR, f"{stem}_segmented.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()

    # Also save plain colour mask
    Image.fromarray(color_mask).save(os.path.join(OUT_DIR, f"{stem}_mask.png"))

print(f"\n✅ Done! Results saved to: {OUT_DIR}/")
print(f"   Files: {', '.join([s+'_segmented.png' for s in [os.path.splitext(f)[0] for f in files]])}")
