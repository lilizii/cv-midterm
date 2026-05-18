from typing import List

import torch


def batch_miou(logits: torch.Tensor, target: torch.Tensor, num_classes: int = 3) -> float:
    pred = torch.argmax(logits, dim=1)
    ious: List[float] = []
    for c in range(num_classes):
        p = pred == c
        t = target == c
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        if union == 0:
            continue
        ious.append(inter / union)
    if not ious:
        return 0.0
    return float(sum(ious) / len(ious))
