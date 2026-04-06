"""
config_fast.py — SPEED MODE for hackathon crunch time.
Imports everything from config but overrides for fastest possible first run.
Use this when you want a baseline result in < 30 minutes.

Usage: copy these overrides into config.py or:
  python train.py  (after copying values here into config.py)
"""

import os

DATA_ROOT       = "./data"
TRAIN_IMG_DIR   = os.path.join(DATA_ROOT, "train", "images")
TRAIN_MASK_DIR  = os.path.join(DATA_ROOT, "train", "masks")
TEST_IMG_DIR    = os.path.join(DATA_ROOT, "test",  "images")
CHECKPOINT_DIR  = "./checkpoints"
LOG_DIR         = "./logs"
RESULTS_DIR     = "./results"

CLASS_MAP = {
    100:   0,   # Trees
    200:   1,   # Lush Bushes
    300:   2,   # Dry Grass
    500:   3,   # Dry Bushes
    550:   4,   # Ground Clutter
    600:   5,   # Flowers
    700:   6,   # Logs
    800:   7,   # Rocks
    7100:  8,   # Landscape
    10000: 9,   # Sky
}
CLASS_NAMES = [
    "Trees","Lush Bushes","Dry Grass","Dry Bushes",
    "Ground Clutter","Flowers","Logs","Rocks","Landscape","Sky"
]
NUM_CLASSES  = 10
IGNORE_INDEX = 255
CLASS_PALETTE = [
    (34,139,34),(0,200,100),(210,180,140),(139,115,85),
    (105,105,105),(255,182,193),(101,67,33),(130,130,130),
    (194,178,128),(135,206,235),
]

# ── SPEED OVERRIDES ──────────────────────────────────────────────────────────
ARCHITECTURE    = "Unet"            # Faster than DeepLabV3+
ENCODER         = "resnet34"        # Tiny encoder — very fast
ENCODER_WEIGHTS = "imagenet"

IMAGE_HEIGHT    = 320               # Lower res = 4x faster than 512
IMAGE_WIDTH     = 320
BATCH_SIZE      = 16               # Bigger batch = faster
NUM_WORKERS     = 2
NUM_EPOCHS      = 20               # Quick baseline
LEARNING_RATE   = 3e-4             # Slightly higher for fast convergence
WEIGHT_DECAY    = 1e-4
VAL_SPLIT       = 0.15

LOSS_TYPE       = "combo"
DICE_WEIGHT     = 0.5
CE_WEIGHT       = 0.5

LR_SCHEDULER    = "cosine"
LR_WARMUP_EPOCHS = 2

EARLY_STOP_PATIENCE = 6
SAVE_TOP_K          = 2

USE_HFLIP         = True
USE_RANDOM_CROP   = True
USE_COLOR_JITTER  = True
USE_GAUSS_NOISE   = False          # Skip for speed
USE_RANDOM_ROTATE = False          # Skip for speed
ROTATE_LIMIT      = 10

SEED      = 42
DEVICE    = "cuda"
USE_AMP   = True
LOG_EVERY = 5
