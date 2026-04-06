#!/usr/bin/env python3
"""
setup_dataset.py — Run this after downloading the dataset to set up paths correctly.

Usage:
    # If you have the zip on Desktop:
    python3 setup_dataset.py --zip ~/Desktop/dataset.zip

    # If already extracted somewhere:
    python3 setup_dataset.py --extracted_path /path/to/Offroad_Segmentation_Training_Dataset

    # If dataset folders are in Downloads:
    python3 setup_dataset.py --search
"""

import os
import sys
import shutil
import zipfile
import argparse
import subprocess

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR  = os.path.dirname(PROJECT_DIR)

def find_dataset():
    """Search common locations for dataset files."""
    search_dirs = [
        os.path.expanduser("~/Downloads"),
        os.path.expanduser("~/Desktop"),
        os.path.expanduser("~/Documents"),
        PARENT_DIR,
    ]
    found = []
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for item in os.listdir(d):
            full = os.path.join(d, item)
            if "offroad" in item.lower() or "segmentation" in item.lower() or "falcon" in item.lower():
                found.append(full)
                print(f"  Found: {full}")
    return found

def extract_zip(zip_path, dest):
    print(f"\nExtracting {zip_path} → {dest}")
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(dest)
    print("Extraction complete!")

def verify_structure(base_path):
    """Check if the dataset has the expected structure."""
    expected = [
        os.path.join(base_path, "train", "Color_Images"),
        os.path.join(base_path, "train", "Segmentation"),
        os.path.join(base_path, "val",   "Color_Images"),
        os.path.join(base_path, "val",   "Segmentation"),
    ]
    ok = True
    for path in expected:
        exists = os.path.isdir(path)
        count = len(os.listdir(path)) if exists else 0
        status = f"✅ {count} files" if exists else "❌ MISSING"
        print(f"  {status}  {path}")
        if not exists:
            ok = False
    return ok

def print_start_command(train_dir, val_dir):
    print("\n" + "="*60)
    print("  READY TO TRAIN! Run this command:")
    print("="*60)
    print(f"\n  TRAIN_DIR='{train_dir}' VAL_DIR='{val_dir}' python3 train_segmentation.py\n")
    print("  Or set paths directly in train_segmentation.py around line 425")
    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip",            type=str, help="Path to dataset zip file")
    parser.add_argument("--extracted_path", type=str, help="Path to already-extracted dataset root")
    parser.add_argument("--search",         action="store_true", help="Search common locations")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  DATASET SETUP")
    print("="*60 + "\n")

    if args.search:
        print("Searching for dataset files...")
        found = find_dataset()
        if not found:
            print("  Nothing found. Please provide --zip or --extracted_path")
        return

    if args.zip:
        if not os.path.exists(args.zip):
            print(f"ERROR: File not found: {args.zip}")
            sys.exit(1)
        dest = os.path.join(PARENT_DIR, "dataset_extracted")
        extract_zip(args.zip, dest)
        # Try to find the training dataset folder
        for root, dirs, _ in os.walk(dest):
            if "Color_Images" in dirs:
                print(f"\nFound Color_Images at: {root}")
                base = os.path.dirname(root)
                break
        else:
            base = dest
        args.extracted_path = base

    if args.extracted_path:
        path = os.path.abspath(args.extracted_path)
        print(f"Verifying dataset at: {path}\n")
        ok = verify_structure(path)

        if ok:
            train_dir = os.path.join(path, "train")
            val_dir   = os.path.join(path, "val")
            print_start_command(train_dir, val_dir)
        else:
            print("\n⚠️  Structure doesn't match expected layout.")
            print("   Expected:")
            print("   <root>/train/Color_Images/")
            print("   <root>/train/Segmentation/")
            print("   <root>/val/Color_Images/")
            print("   <root>/val/Segmentation/")
            print("\n   Run: python3 setup_dataset.py --search  to find dataset files")
    else:
        print("Usage:")
        print("  python3 setup_dataset.py --zip ~/Desktop/dataset.zip")
        print("  python3 setup_dataset.py --extracted_path /path/to/dataset")
        print("  python3 setup_dataset.py --search")

if __name__ == "__main__":
    main()
