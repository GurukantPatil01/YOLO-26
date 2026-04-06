"""
explore_data.py — Run first to understand the dataset before training.

Prints:
  - Directory structure
  - Image and mask count
  - Unique mask pixel values
  - Class frequency / imbalance statistics
  - Saves sample visualizations to explore_output/
"""

import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter

import config


OUTPUT_DIR = "./explore_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Directory Walk
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  DATASET EXPLORER")
print("=" * 60)

for root, dirs, files in os.walk(config.DATA_ROOT):
    level = root.replace(config.DATA_ROOT, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    if level < 3:
        subindent = ' ' * 2 * (level + 1)
        for f in files[:5]:
            print(f"{subindent}{f}")
        if len(files) > 5:
            print(f"{subindent}... ({len(files)} files total)")

# ─────────────────────────────────────────────────────────────────────────────
# 2. File counts
# ─────────────────────────────────────────────────────────────────────────────
IMG_EXTS = ["*.png", "*.jpg", "*.jpeg"]

def count_files(directory):
    total = 0
    for ext in IMG_EXTS:
        total += len(glob.glob(os.path.join(directory, ext)))
    return total

n_train = count_files(config.TRAIN_IMG_DIR)
n_masks = count_files(config.TRAIN_MASK_DIR)
n_test  = count_files(config.TEST_IMG_DIR)
print(f"\n  Train images : {n_train}")
print(f"  Train masks  : {n_masks}")
print(f"  Test images  : {n_test}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Inspect masks — find unique pixel values
# ─────────────────────────────────────────────────────────────────────────────
mask_paths = []
for ext in IMG_EXTS:
    mask_paths += glob.glob(os.path.join(config.TRAIN_MASK_DIR, ext))

N_SAMPLE = min(50, len(mask_paths))
print(f"\n  Scanning {N_SAMPLE} masks for unique values…")

pixel_counter = Counter()
unique_vals   = set()

for mp in mask_paths[:N_SAMPLE]:
    arr = np.array(Image.open(mp))
    vals = np.unique(arr)
    unique_vals.update(vals.tolist())
    for v in vals:
        pixel_counter[v] += int((arr == v).sum())

print(f"\n  Unique pixel values found: {sorted(unique_vals)}")
print(f"\n  Pixel value → class name:")
for v in sorted(unique_vals):
    name = "UNKNOWN"
    for raw, idx in config.CLASS_MAP.items():
        if raw == v:
            name = config.CLASS_NAMES[idx]
            break
    print(f"    {v:>8}  →  {name}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Class frequency (imbalance check)
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Class pixel counts (from sampled masks):")
total_pixels = sum(pixel_counter.values())
class_pixels = {}
for raw_val, count in pixel_counter.items():
    if raw_val in config.CLASS_MAP:
        name = config.CLASS_NAMES[config.CLASS_MAP[raw_val]]
        class_pixels[name] = class_pixels.get(name, 0) + count

for name, cnt in sorted(class_pixels.items(), key=lambda x: -x[1]):
    pct = 100 * cnt / total_pixels
    print(f"    {name:<20}  {cnt:>12,d}  ({pct:.1f}%)")

# Bar chart
if class_pixels:
    fig, ax = plt.subplots(figsize=(10, 4))
    names  = list(class_pixels.keys())
    counts = [class_pixels[n] for n in names]
    colors = [
        "#{:02x}{:02x}{:02x}".format(*config.CLASS_PALETTE[config.CLASS_NAMES.index(n)])
        for n in names
    ]
    ax.bar(names, counts, color=colors)
    ax.set_title("Pixel count per class (imbalance check)", fontsize=13)
    ax.set_ylabel("Pixel count")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"), dpi=120)
    plt.close()
    print(f"\n  Saved class distribution chart → {OUTPUT_DIR}/class_distribution.png")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Visualise a few samples
# ─────────────────────────────────────────────────────────────────────────────
img_paths = []
for ext in IMG_EXTS:
    img_paths += glob.glob(os.path.join(config.TRAIN_IMG_DIR, ext))

N_VIZ = min(4, len(img_paths))
fig, axes = plt.subplots(N_VIZ, 3, figsize=(14, N_VIZ * 4))
if N_VIZ == 1:
    axes = [axes]

legend_patches = [
    mpatches.Patch(
        color=[c/255 for c in config.CLASS_PALETTE[i]],
        label=config.CLASS_NAMES[i]
    )
    for i in range(config.NUM_CLASSES)
]

for row, ip in enumerate(img_paths[:N_VIZ]):
    stem     = os.path.splitext(os.path.basename(ip))[0]
    mp       = os.path.join(config.TRAIN_MASK_DIR, os.path.basename(ip))
    if not os.path.exists(mp):
        mp = os.path.join(config.TRAIN_MASK_DIR, stem + "_mask.png")

    img = np.array(Image.open(ip).convert("RGB"))
    axes[row][0].imshow(img)
    axes[row][0].set_title("Image")
    axes[row][0].axis("off")

    if os.path.exists(mp):
        mask_raw = np.array(Image.open(mp))
        # Build colour mask
        h, w = mask_raw.shape[:2]
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for raw_val, cls_idx in config.CLASS_MAP.items():
            color_mask[mask_raw == raw_val] = config.CLASS_PALETTE[cls_idx]

        axes[row][1].imshow(color_mask)
        axes[row][1].set_title("Ground Truth Mask")
        axes[row][1].axis("off")

        blended = (0.5 * img + 0.5 * color_mask).astype(np.uint8)
        axes[row][2].imshow(blended)
        axes[row][2].set_title("Overlay")
        axes[row][2].axis("off")
    else:
        axes[row][1].text(0.5, 0.5, "mask not found", ha="center", va="center")
        axes[row][2].text(0.5, 0.5, "mask not found", ha="center", va="center")

fig.legend(handles=legend_patches, loc="lower center", ncol=5, fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "sample_visualizations.png"), dpi=120)
plt.close()
print(f"  Saved sample visualizations → {OUTPUT_DIR}/sample_visualizations.png")
print("\n[Done] Data exploration complete. Check explore_output/")
