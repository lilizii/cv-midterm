from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int = 3, smooth: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        target_oh = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        inter = torch.sum(probs * target_oh, dims)
        cardinality = torch.sum(probs + target_oh, dims)
        dice = (2.0 * inter + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


def build_loss(loss_mode: str, ce_weight: float = 0.25, dice_weight: float = 0.75) -> Callable:
    ce = nn.CrossEntropyLoss()
    dice = DiceLoss(num_classes=3)

    if loss_mode == 'ce':
        return lambda out, y: ce(out, y)
    if loss_mode == 'dice':
        return lambda out, y: dice(out, y)
    if loss_mode == 'combo':
        return lambda out, y: ce_weight * ce(out, y) + dice_weight * dice(out, y)

    raise ValueError(f'Unknown loss_mode={loss_mode}')
