import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import build_loss
from .metrics import batch_miou
from .model import UNet


@dataclass
class RunResult:
    mode: str
    best_miou: float
    best_epoch: int
    history: List[Dict[str, float]]
    final_test_loss: float
    final_test_miou: float


def evaluate(model: nn.Module, loader: DataLoader, criterion, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_miou = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item()
            total_miou += batch_miou(out, y, num_classes=3)
    avg_loss = total_loss / max(1, len(loader))
    avg_miou = total_miou / max(1, len(loader))
    return avg_loss, avg_miou


def train_one_mode(
    mode: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    out_dir: str,
    amp: bool = True,
    base_ch: int = 32,
    patience: int = 10,
    ce_weight: float = 0.25,
    dice_weight: float = 0.75,
) -> RunResult:
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f'metrics_{mode}.csv')
    history: List[Dict[str, float]] = []
    model = UNet(in_ch=3, num_classes=3, base_ch=base_ch).to(device)
    criterion = build_loss(mode, ce_weight=ce_weight, dice_weight=dice_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=(amp and device.type == 'cuda'))

    best_miou = -1.0
    best_epoch = -1
    best_state = None
    epochs_without_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f'[{mode}] Epoch {epoch}/{epochs}', leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=(amp and device.type == 'cuda')):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.4f}')
        scheduler.step()

        train_loss /= max(1, len(train_loader))
        val_loss, val_miou = evaluate(model, val_loader, criterion, device)
        print(
            f'[{mode}] epoch={epoch:02d} train_loss={train_loss:.4f} '
            f'val_loss={val_loss:.4f} val_mIoU={val_miou:.4f}'
        )

        if val_miou > best_miou:
            best_miou = val_miou
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        epoch_lr = optimizer.param_groups[0]['lr']
        history.append(
            {
                'epoch': float(epoch),
                'train_loss': float(train_loss),
                'val_loss': float(val_loss),
                'val_miou': float(val_miou),
                'lr': float(epoch_lr),
            }
        )

        if patience > 0 and epochs_without_improve >= patience:
            print(f'[{mode}] Early stopping at epoch {epoch} (patience={patience})')
            break

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_loss', 'val_miou', 'lr'])
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    'epoch': int(row['epoch']),
                    'train_loss': f"{row['train_loss']:.6f}",
                    'val_loss': f"{row['val_loss']:.6f}",
                    'val_miou': f"{row['val_miou']:.6f}",
                    'lr': f"{row['lr']:.8f}",
                }
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)
    final_test_loss, final_test_miou = evaluate(model, test_loader, criterion, device)

    ckpt_path = os.path.join(out_dir, f'unet_{mode}_best.pth')
    torch.save(
        {
            'model': best_state,
            'mode': mode,
            'best_miou': best_miou,
            'best_epoch': best_epoch,
            'final_test_loss': final_test_loss,
            'final_test_miou': final_test_miou,
            'loss_cfg': {'ce_weight': ce_weight, 'dice_weight': dice_weight},
        },
        ckpt_path,
    )
    print(f'[{mode}] final_test_loss={final_test_loss:.4f} final_test_mIoU={final_test_miou:.4f}')

    return RunResult(
        mode=mode,
        best_miou=best_miou,
        best_epoch=best_epoch,
        history=history,
        final_test_loss=final_test_loss,
        final_test_miou=final_test_miou,
    )
