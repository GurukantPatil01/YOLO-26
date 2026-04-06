#!/usr/bin/env python3
"""
quickstart.py — ONE COMMAND to verify setup and start training.
Run this first to catch any issues before the full train.py

Usage:
    python quickstart.py                         # checks env, then trains
    python quickstart.py --fast                  # uses fast config (UNet ResNet34 320px)
    python quickstart.py --check_only            # just verify env, no training
"""

import sys, os, argparse, subprocess

def check(name, fn):
    try:
        result = fn()
        print(f"  ✅  {name}: {result}")
        return True
    except Exception as e:
        print(f"  ❌  {name}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast",       action="store_true", help="Use fast config")
    parser.add_argument("--check_only", action="store_true", help="Only run checks")
    args = parser.parse_args()

    print("\n" + "="*55)
    print("  HACKATHON QUICKSTART CHECKER")
    print("="*55 + "\n")

    ok = True

    # Python
    ok &= check("Python version", lambda: sys.version.split()[0])

    # Torch
    ok &= check("PyTorch", lambda: __import__("torch").__version__)

    # CUDA
    def cuda_check():
        import torch
        if torch.cuda.is_available():
            return f"CUDA {torch.version.cuda} — GPU: {torch.cuda.get_device_name(0)}"
        return "No GPU — will use CPU (training will be slow!)"
    check("GPU", cuda_check)

    # SMP
    ok &= check("segmentation-models-pytorch", lambda: __import__("segmentation_models_pytorch").__version__)

    # Albumentations
    ok &= check("albumentations", lambda: __import__("albumentations").__version__)

    # Data directories
    import config as cfg
    def check_data():
        if not os.path.isdir(cfg.TRAIN_IMG_DIR):
            raise FileNotFoundError(f"{cfg.TRAIN_IMG_DIR} not found")
        imgs = [f for f in os.listdir(cfg.TRAIN_IMG_DIR) if f.endswith((".png",".jpg"))]
        if len(imgs) == 0:
            raise ValueError("No images found in train/images/")
        masks = [f for f in os.listdir(cfg.TRAIN_MASK_DIR) if f.endswith((".png",".jpg"))]
        return f"{len(imgs)} images, {len(masks)} masks"
    ok &= check("Training data", check_data)

    def check_test():
        if not os.path.isdir(cfg.TEST_IMG_DIR):
            raise FileNotFoundError(f"{cfg.TEST_IMG_DIR} not found")
        imgs = [f for f in os.listdir(cfg.TEST_IMG_DIR) if f.endswith((".png",".jpg"))]
        return f"{len(imgs)} test images"
    check("Test data", check_test)

    print()
    if not ok:
        print("  ⚠️  Some checks failed. Fix issues above before training.")
        print("  Most common fix: pip3 install -r requirements.txt")
        print("  Data fix: make sure data/train/images/ and data/train/masks/ exist")
        sys.exit(1)

    if args.check_only:
        print("  All checks passed! Ready to train.")
        return

    # Switch to fast config if requested
    if args.fast:
        print("  Using FAST config (UNet + ResNet34 + 320px)")
        import shutil
        shutil.copy("config_fast.py", "config.py")
        print("  config.py replaced with config_fast.py")

    print("\n  Starting training...\n")
    os.execv(sys.executable, [sys.executable, "train.py"])


if __name__ == "__main__":
    main()
