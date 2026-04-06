import os
import shutil
from pathlib import Path
import subprocess

TARGET_DIR = "/Users/gurukantpatil/Desktop/Hackathon/YOLO'26/Final_Submission_Images"
BEFORE_DIR = "/Users/gurukantpatil/Desktop/Test/Before "
os.makedirs(TARGET_DIR, exist_ok=True)

# Define images
images = sorted([f for f in os.listdir(BEFORE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

print(f"Found {len(images)} images in {BEFORE_DIR}")

# 1. Run inference on those 5 images
cmd = [
    "python3", "infer_custom.py", 
    BEFORE_DIR, 
    "checkpoints/best_fast.pth", 
    f"{TARGET_DIR}/temp_predictions"
]
subprocess.run(cmd)

# 2. Rename and organize into final structure
for i, img_name in enumerate(images, 1):
    stem = os.path.splitext(img_name)[0]
    
    # Paths
    orig_path = os.path.join(BEFORE_DIR, img_name)
    pred_path = os.path.join(TARGET_DIR, "temp_predictions", f"{stem}_segmented.png")
    
    # Target paths
    before_target = os.path.join(TARGET_DIR, f"Image {i}_Before.jpg")
    after_target = os.path.join(TARGET_DIR, f"Image {i}_After.jpg")
    
    # Copy
    shutil.copy(orig_path, before_target)
    if os.path.exists(pred_path):
        shutil.copy(pred_path, after_target)

# Clean up temp
shutil.rmtree(os.path.join(TARGET_DIR, "temp_predictions"), ignore_errors=True)

print(f"\nAll images processed and saved perfectly in: {TARGET_DIR}")
