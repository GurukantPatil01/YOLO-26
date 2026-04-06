"""
precompute_features.py — Run ONCE to extract and cache DINOv2 features.

This makes training 50x faster by running the frozen backbone once
instead of every epoch. The tiny segmentation head then trains on
pre-extracted features in seconds per epoch.

Usage:
    python3 precompute_features.py
    (takes ~5-10 mins, then train.py runs in minutes per epoch)
"""

import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

DATASET_ROOT = "/Users/gurukantpatil/Downloads/Offroad_Segmentation_Training_Dataset"
CACHE_DIR    = "./feature_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

device = torch.device("cpu")  # CPU is fine for one-time extraction

# Image dimensions (must match train_segmentation.py)
w = int(((960 / 2) // 14) * 14)
h = int(((540 / 2) // 14) * 14)

transform = transforms.Compose([
    transforms.Resize((h, w)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
mask_transform = transforms.Compose([
    transforms.Resize((h, w)),
    transforms.ToTensor(),
])

# Load DINOv2 backbone
print("Loading DINOv2 backbone from cache...")
backbone = torch.hub.load(
    repo_or_dir="/Users/gurukantpatil/.cache/torch/hub/facebookresearch_dinov2_main",
    model="dinov2_vits14",
    source="local"
)
backbone.eval()
backbone.to(device)
print("Backbone loaded!")

value_map = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4,
    550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}

def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return new_arr

def process_split(split_name):
    img_dir  = os.path.join(DATASET_ROOT, split_name, "Color_Images")
    mask_dir = os.path.join(DATASET_ROOT, split_name, "Segmentation")
    out_dir  = os.path.join(CACHE_DIR, split_name)
    os.makedirs(out_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
    print(f"\n[{split_name}] Processing {len(files)} images...")

    for fname in tqdm(files, desc=split_name):
        stem     = os.path.splitext(fname)[0]
        feat_path = os.path.join(out_dir, f"{stem}_feat.npy")
        mask_path_out = os.path.join(out_dir, f"{stem}_mask.npy")

        if os.path.exists(feat_path) and os.path.exists(mask_path_out):
            continue  # Already cached, skip

        # Load & transform image
        img  = Image.open(os.path.join(img_dir, fname)).convert("RGB")
        img_t = transform(img).unsqueeze(0).to(device)   # (1, C, H, W)

        # Extract DINOv2 features (frozen)
        with torch.no_grad():
            feats = backbone.forward_features(img_t)["x_norm_patchtokens"]
        # feats shape: (1, N_patches, 384)
        np.save(feat_path, feats.squeeze(0).cpu().numpy().astype(np.float16))

        # Load & convert mask
        mask_path = os.path.join(mask_dir, fname)
        mask      = Image.open(mask_path)
        mask_arr  = convert_mask(mask)
        mask_t    = mask_transform(Image.fromarray(mask_arr)) * 255
        np.save(mask_path_out, mask_t.squeeze(0).numpy().astype(np.uint8))

    print(f"[{split_name}] Done! Cached to {out_dir}")

process_split("train")
process_split("val")

print(f"\n✅ Feature cache saved to: {CACHE_DIR}")
print(f"   Now run: python3 train_fast.py")
