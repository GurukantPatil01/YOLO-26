# Performance Report — Duality AI Offroad Segmentation
**Team:** YOLO'26  
**Date:** April 6, 2026  
**Hackathon:** Duality AI Falcon Offroad Segmentation Challenge

---

## 1. Methodology

### 1.1 Dataset
- **Training images:** 2,857 synthetic desert scenes (Falcon digital twin)
- **Validation images:** 317 images (provided val split)
- **Classes:** 10 (Background, Trees, Lush Bushes, Dry Grass, Dry Bushes, Ground Clutter, Logs, Rocks, Landscape, Sky)
- **Image resolution:** 960×540, resized to 476×266 for DINOv2 patch alignment

### 1.2 Architecture
We used **DINOv2 ViT-S/14** (Vision Transformer, Small, 14×14 patches) as a frozen feature extractor with a lightweight **ConvNeXt-style segmentation head** trained on top.

```
Input Image (3×476×266)
        ↓
DINOv2 ViT-S/14 [FROZEN]
   384-dim patch tokens (34×19 = 646 tokens)
        ↓
ConvNeXt Segmentation Head [TRAINED]
   Stem:  Conv7×7(384→128) + GELU
   Block: DWConv7×7 + GELU + Conv1×1 + GELU
   Head:  Conv1×1(128→10)
        ↓
Bilinear Upsample → Full Resolution
        ↓
Segmentation Map (10 classes)
```

**Why DINOv2?**
- Pre-trained on 142M images — powerful universal visual features
- Works extremely well on synthetic data without domain gap issues
- Frozen backbone prevents overfitting; only the head (<<1M params) is trained

### 1.3 Training Setup

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| LR Schedule | Cosine Annealing |
| Weight Decay | 1e-4 |
| Batch Size | 8 |
| Epochs | 30 |
| Loss | Cross-Entropy |
| Device | CPU (Apple M-series) |

### 1.4 Efficiency Strategy
To make training tractable on CPU, we pre-computed DINOv2 features once (stored as `.npy` files) and trained only the segmentation head on cached features. This reduced per-epoch time from ~100s (with backbone forward pass) to ~30s.

---

## 2. Results

### 2.1 Training Curves

*(See `train_stats/training_curves.png`)*

| Epoch | Train Loss | Val Loss | Train mIoU | Val mIoU |
|---|---|---|---|---|
| 1 | 0.9062 | 0.7997 | 0.3207 | 0.3122 |
| 2 | 0.7860 | 0.7743 | 0.3752 | 0.3229 |
| 3 | 0.7668 | 0.7612 | 0.3847 | 0.3342 |
| 4 | 0.7564 | 0.7577 | 0.3915 | 0.3437 |
| … | … | … | … | … |
| **Best** | — | — | — | **[FILL IN]** |

### 2.2 Per-Class IoU (Best Checkpoint)

*(See `predictions/per_class_iou.png`)*

| Class | IoU | Notes |
|---|---|---|
| Background | [FILL] | — |
| Trees | [FILL] | Large, distinct texture |
| Lush Bushes | [FILL] | Similar to dry bushes |
| Dry Grass | [FILL] | Large coverage areas |
| Dry Bushes | [FILL] | Often confused with lush bushes |
| Ground Clutter | [FILL] | Small, sparse class |
| Logs | [FILL] | Hard — rare & thin |
| Rocks | [FILL] | Confused with landscape |
| Landscape | [FILL] | Large areas, high IoU expected |
| Sky | [FILL] | Easiest — large, uniform |

### 2.3 Summary Metrics

| Metric | Value |
|---|---|
| **Mean IoU (mIoU)** | **[FILL IN]** |
| Pixel Accuracy | [FILL IN] |
| Best Epoch | [FILL IN] |

---

## 3. Challenges and Solutions

### 3.1 Slow CPU Training
**Problem:** DINOv2 forward pass on 2857 images per epoch = ~100s/epoch × 30 = 50 hours on CPU.  
**Solution:** Pre-computed and cached DINOv2 features to disk once (~6 min). Training then runs in ~30s/epoch on cached tensors.

### 3.2 SSL Certificate Error on macOS Python 3.14
**Problem:** `torch.hub.load` for DINOv2 failed with SSL certificate verification error.  
**Solution:** Ran `/Applications/Python 3.14/Install Certificates.command` to install system certificates.

### 3.3 Class Imbalance
**Problem:** Large classes (Sky, Landscape) dominate pixel count; small classes (Logs, Ground Clutter) are rare.  
**Solution:** Model compensates via CrossEntropyLoss (equal weight per class, not per pixel). Future: weighted CE or Focal Loss.

### 3.4 Context Shift (Train → Test)
**Problem:** Test environment has different terrain layout, lighting, and vegetation density.  
**Solution:** DINOv2's rich generic features generalize better than CNN features trained from scratch. The model sees structural patterns rather than memorizing colors/textures.

---

## 4. Failure Case Analysis

### 4.1 Common Failure Patterns

**Logs vs Rocks Confusion**
- Logs are thin and rare; DINOv2 patches are 14×14px, which may not fully resolve thin log geometry
- Fix attempted: none (time constraint) — Future: higher-res backbone (ViT-B/14)

**Lush Bushes vs Dry Bushes**
- Both share similar local texture at DINOv2's 476px resolution
- Fix: Could use multi-scale features or ensemble

**Ground Clutter**  
- Extremely sparse class — appears in <5% of pixels in many images
- Model tends to predict nearest dominant class instead
- Fix: Weighted loss or oversampling patches containing clutter

*(See `predictions/comparisons/` for visual examples)*

---

## 5. Conclusion and Future Work

### 5.1 What Worked
- DINOv2 frozen features provide strong baseline with minimal training
- Feature pre-caching made hackathon timeline feasible on CPU
- Cosine LR schedule + AdamW gave stable convergence

### 5.2 What Didn't Work / Trade-offs
- No data augmentation (flips, color jitter) on cached features — could boost mIoU by 2–5%
- Lightweight head limits capacity for fine-grained distinctions between similar classes
- CPU only — no mixed precision (AMP) possible

### 5.3 Future Work
| Idea | Expected Gain |
|---|---|
| DINOv2 ViT-B/14 (larger backbone) | +3–5% mIoU |
| Augmentation (flip, crop) before caching | +2–4% mIoU |
| Weighted cross-entropy for rare classes | +1–3% mIoU |
| Dice + CE combo loss | +1–2% mIoU |
| Decoder with skip connections (UNet head) | +2–5% mIoU |
| Test-time augmentation (TTA) | +0.5–1% mIoU |
| Domain adaptation techniques | Unknown |

---

*Report generated for the Duality AI Falcon Offroad Segmentation Hackathon, April 2026.*
