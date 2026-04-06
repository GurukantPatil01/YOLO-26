₹# Duality AI Offroad Segmentation — YOLO'26 Team

Semantic segmentation of synthetic desert scenes for the Duality AI Falcon Hackathon.

## Project Structure

```
├── config.py          # All hyperparameters and paths
├── dataset.py         # Dataset, mask remapping, augmentations
├── losses.py          # CE / Dice / Combo loss
├── metrics.py         # IoU / confusion-matrix tracker
├── train.py           # Training loop
├── test.py            # Inference + overlay generation
├── explore_data.py    # Run this FIRST to understand the data
├── plot_results.py    # Generate report plots from train log CSV
├── requirements.txt
└── data/
    ├── train/
    │   ├── images/
    │   └── masks/
    └── test/
        └── images/    # ← NEVER use these for training
```

## Setup

```bash
conda create -n hackathon python=3.10 -y
conda activate hackathon

# Install PyTorch (CUDA 11.8 — adjust if needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

## Quick Start

### 1. Explore the data first (always do this before training)
```bash
python explore_data.py
# → check explore_output/ for class distribution chart & sample visualizations
```

### 2. Configure your run
Edit `config.py` to set:
- `DATA_ROOT`, `TRAIN_IMG_DIR`, `TRAIN_MASK_DIR`, `TEST_IMG_DIR`
- `ARCHITECTURE`, `ENCODER` (default: DeepLabV3Plus + efficientnet-b4)
- `NUM_EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`

### 3. Train
```bash
python train.py

# Resume from a checkpoint:
python train.py --resume checkpoints/best.pth
```

Training logs every epoch to `logs/train_log.csv`.
Best model saved automatically to `checkpoints/best.pth`.

### 4. Inference / evaluation
```bash
# Without test masks (generates colour overlays only):
python test.py --checkpoint checkpoints/best.pth

# With test masks (also computes IoU):
python test.py --checkpoint checkpoints/best.pth \
               --test_mask_dir data/test/masks

# With test-time augmentation (usually +0.5–1.0 mIoU):
python test.py --checkpoint checkpoints/best.pth --use_tta
```

Results saved to `results/` as side-by-side `original | mask | overlay` images.

### 5. Generate report plots
```bash
python plot_results.py
# → logs/plots/training_curves.png
# → logs/plots/per_class_iou.png
```

## Class Labels

| Class | Pixel Value | Index |
|---|---|---|
| Trees | 100 | 0 |
| Lush Bushes | 200 | 1 |
| Dry Grass | 300 | 2 |
| Dry Bushes | 500 | 3 |
| Ground Clutter | 550 | 4 |
| Flowers | 600 | 5 |
| Logs | 700 | 6 |
| Rocks | 800 | 7 |
| Landscape | 7100 | 8 |
| Sky | 10000 | 9 |

## Model

- **Architecture**: DeepLabV3+ (configurable)
- **Encoder**: EfficientNet-B4 pretrained on ImageNet
- **Loss**: Combo CE + Dice (configurable)
- **Augmentation**: HFlip, RandomCrop, ColorJitter, GaussNoise, Rotation

## Important Rules

> ⚠️ **Never train on test images.** The `data/test/` split is for evaluation only.
> Using test data for training results in disqualification.

## Reproducing Results

```bash
# Set seed is fixed in config.py (SEED=42)
python train.py
python test.py --checkpoint checkpoints/best.pth
python plot_results.py
```
