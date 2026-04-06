"""
config.py — Central configuration for all hyperparameters and paths.
Change settings here instead of hunting through train.py.
"""

import os

# ─────────────────────────────────────────────
# PATHS  (adjust to match your downloaded dataset layout)
# ─────────────────────────────────────────────
DATA_ROOT       = "./data"                    # root of the dataset
TRAIN_IMG_DIR   = os.path.join(DATA_ROOT, "train", "images")
TRAIN_MASK_DIR  = os.path.join(DATA_ROOT, "train", "masks")
TEST_IMG_DIR    = os.path.join(DATA_ROOT, "test",  "images")   # NO labels here
CHECKPOINT_DIR  = "./checkpoints"
LOG_DIR         = "./logs"
RESULTS_DIR     = "./results"                 # saved prediction overlays

# ─────────────────────────────────────────────
# CLASS MAPPING  (raw mask pixel value → class index 0..N-1)
# ─────────────────────────────────────────────
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
    "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes",
    "Ground Clutter", "Flowers", "Logs", "Rocks", "Landscape", "Sky"
]

NUM_CLASSES = len(CLASS_NAMES)
IGNORE_INDEX = 255   # pixels with this index are ignored in loss

# Visual palette for overlays (RGB)
CLASS_PALETTE = [
    (34,  139, 34),    # Trees          — forest green
    (0,   200, 100),   # Lush Bushes    — bright green
    (210, 180, 140),   # Dry Grass      — tan
    (139, 115, 85),    # Dry Bushes     — brownish
    (105, 105, 105),   # Ground Clutter — dim gray
    (255, 182, 193),   # Flowers        — pink
    (101, 67,  33),    # Logs           — dark brown
    (130, 130, 130),   # Rocks          — gray
    (194, 178, 128),   # Landscape      — sandy
    (135, 206, 235),   # Sky            — sky blue
]

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
ARCHITECTURE    = "DeepLabV3Plus"   # or "Unet", "FPN", "PAN"
ENCODER         = "efficientnet-b4" # or "resnet50", "resnet101"
ENCODER_WEIGHTS = "imagenet"        # None to train from scratch

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
IMAGE_HEIGHT    = 512
IMAGE_WIDTH     = 512
BATCH_SIZE      = 8
NUM_WORKERS     = 4
NUM_EPOCHS      = 50
LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 1e-4
VAL_SPLIT       = 0.15              # fraction of train data used for validation

# Loss — "ce" | "dice" | "combo" (CE + Dice)
LOSS_TYPE       = "combo"
DICE_WEIGHT     = 0.5               # weight for dice loss in combo
CE_WEIGHT       = 0.5               # weight for CE   loss in combo

# LR Scheduler
LR_SCHEDULER    = "cosine"          # "cosine" | "step" | "plateau"
LR_WARMUP_EPOCHS = 3

# Early stopping
EARLY_STOP_PATIENCE = 10
SAVE_TOP_K          = 3             # keep top-K checkpoints by val IoU

# ─────────────────────────────────────────────
# AUGMENTATION FLAGS
# ─────────────────────────────────────────────
USE_HFLIP          = True
USE_RANDOM_CROP    = True
USE_COLOR_JITTER   = True
USE_GAUSS_NOISE    = True
USE_RANDOM_ROTATE  = True
ROTATE_LIMIT       = 15             # degrees

# ─────────────────────────────────────────────
# MISC
# ─────────────────────────────────────────────
SEED        = 42
DEVICE      = "cuda"    # "cpu" for CPU-only
USE_AMP     = True      # mixed precision (faster on modern GPUs)
LOG_EVERY   = 10        # log loss every N batches
