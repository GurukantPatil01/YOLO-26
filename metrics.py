"""
metrics.py — IoU and related segmentation metrics.
"""

import torch
import numpy as np
import config


class SegmentationMetrics:
    """
    Accumulates predictions over batches and computes:
      - Per-class IoU
      - Mean IoU (mIoU)
      - Pixel accuracy
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES, ignore_index: int = config.IGNORE_INDEX):
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        # Confusion matrix: rows=GT, cols=Pred
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        preds   : (B, H, W) — argmax of logits
        targets : (B, H, W) — ground truth class indices
        """
        preds   = preds.cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()

        # Mask out ignore pixels
        valid = targets != self.ignore_index
        preds   = preds[valid]
        targets = targets[valid]

        # Fast confusion matrix accumulation
        combined = self.num_classes * targets.astype(np.int64) + preds.astype(np.int64)
        counts   = np.bincount(combined, minlength=self.num_classes ** 2)
        self.confusion += counts.reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict:
        """Returns dict with per-class IoU, mIoU, pixel-accuracy."""
        tp = np.diag(self.confusion)
        fp = self.confusion.sum(axis=0) - tp
        fn = self.confusion.sum(axis=1) - tp

        iou_per_class = tp / (tp + fp + fn + 1e-6)

        # Only include classes that actually appeared in GT
        present_mask = self.confusion.sum(axis=1) > 0
        miou = iou_per_class[present_mask].mean()

        pixel_acc = tp.sum() / (self.confusion.sum() + 1e-6)

        return {
            "miou":          float(miou),
            "iou_per_class": iou_per_class.tolist(),
            "pixel_acc":     float(pixel_acc),
        }

    def print_report(self):
        results = self.compute()
        print("\n" + "=" * 55)
        print(f"  mIoU        : {results['miou']:.4f}")
        print(f"  Pixel Acc   : {results['pixel_acc']:.4f}")
        print("-" * 55)
        print(f"  {'Class':<20} {'IoU':>10}")
        print("-" * 55)
        for i, (name, iou) in enumerate(
            zip(config.CLASS_NAMES, results["iou_per_class"])
        ):
            present = self.confusion[i].sum() > 0
            suffix  = "" if present else "  (absent)"
            print(f"  {name:<20} {iou:>10.4f}{suffix}")
        print("=" * 55 + "\n")
        return results
