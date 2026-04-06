"""
losses.py — Loss functions for semantic segmentation.
Supports:
  - Cross-Entropy (with class weights for imbalance)
  - Dice Loss
  - Combo Loss (CE + Dice)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import config


# ─────────────────────────────────────────────────────────────────────────────
# Dice Loss
# ─────────────────────────────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, ignore_index: int = 255):
        super().__init__()
        self.smooth       = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B, C, H, W) raw scores
        targets : (B, H, W)    class indices
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)   # (B, C, H, W)

        # One-hot encode targets, masking ignore pixels
        valid_mask = (targets != self.ignore_index)          # (B, H, W)
        safe_targets = targets.clone()
        safe_targets[~valid_mask] = 0                        # avoid indexing error

        one_hot = F.one_hot(safe_targets, num_classes)       # (B, H, W, C)
        one_hot = one_hot.permute(0, 3, 1, 2).float()       # (B, C, H, W)

        # Zero out ignored pixels in both one_hot and probs
        mask = valid_mask.unsqueeze(1).float()               # (B, 1, H, W)
        one_hot = one_hot * mask
        probs   = probs   * mask

        # Compute per-class Dice
        intersection = (probs * one_hot).sum(dim=(0, 2, 3))
        union        = probs.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
        dice_per_class = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - dice_per_class.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Combo Loss  (weighted CE + Dice)
# ─────────────────────────────────────────────────────────────────────────────
class ComboLoss(nn.Module):
    def __init__(
        self,
        ce_weight: float  = 0.5,
        dice_weight: float = 0.5,
        class_weights: torch.Tensor = None,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.ce_weight   = ce_weight
        self.dice_weight = dice_weight
        self.ce   = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index,
            reduction="mean",
        )
        self.dice = DiceLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.ce_weight   * self.ce(logits, targets)
            + self.dice_weight * self.dice(logits, targets)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────
def get_loss_fn(class_weights: torch.Tensor = None) -> nn.Module:
    """
    Returns the configured loss function.
    class_weights: optional Tensor of shape (NUM_CLASSES,) for CE.
    """
    if config.LOSS_TYPE == "ce":
        return nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=config.IGNORE_INDEX,
        )
    elif config.LOSS_TYPE == "dice":
        return DiceLoss(ignore_index=config.IGNORE_INDEX)
    elif config.LOSS_TYPE == "combo":
        return ComboLoss(
            ce_weight=config.CE_WEIGHT,
            dice_weight=config.DICE_WEIGHT,
            class_weights=class_weights,
            ignore_index=config.IGNORE_INDEX,
        )
    else:
        raise ValueError(f"Unknown LOSS_TYPE: {config.LOSS_TYPE}")
