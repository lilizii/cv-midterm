import argparse
import csv
import json
import os
import random
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from models_attention import StageAttentionResNet


@dataclass
class Config:
    data_root: str
    output_dir: str
    model_name: str
    num_classes: int
    image_size: int
    batch_size: int
    epochs: int
    lr: float
    backbone_lr_mult: float
    weight_decay: float
    momentum: float
    pretrained: bool
    freeze_backbone_epochs: int
    num_workers: int
    val_split: float
    seed: int
    device: str


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=1)
    return (pred == targets).float().mean().item()


def build_dataloaders(cfg: Config):
    train_tf = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    trainval_ds_train_tf = datasets.OxfordIIITPet(
        root=cfg.data_root, split="trainval", target_types="category", transform=train_tf, download=True
    )
    trainval_ds_eval_tf = datasets.OxfordIIITPet(
        root=cfg.data_root, split="trainval", target_types="category", transform=test_tf, download=True
    )
    test_ds = datasets.OxfordIIITPet(root=cfg.data_root, split="test", target_types="category", transform=test_tf, download=True)

    total = len(trainval_ds_train_tf)
    val_size = int(total * cfg.val_split)
    if val_size <= 0 or val_size >= total:
        raise ValueError(f"Invalid val_split={cfg.val_split}. It must make validation set size in (0, {total}).")

    indices = torch.randperm(total, generator=torch.Generator().manual_seed(cfg.seed)).tolist()
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_ds = Subset(trainval_ds_train_tf, train_idx)
    val_ds = Subset(trainval_ds_eval_tf, val_idx)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


def build_model(cfg: Config):
    m = cfg.model_name.lower()
    if m in {"resnet18", "resnet34"}:
        if m == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if cfg.pretrained else None
            backbone = models.resnet18(weights=weights)
        else:
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if cfg.pretrained else None
            backbone = models.resnet34(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, cfg.num_classes)
        return backbone

    if m in {"resnet18_se", "resnet34_se", "resnet18_cbam", "resnet34_cbam"}:
        use34 = "34" in m
        attn = "cbam" if "cbam" in m else "se"
        if use34:
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if cfg.pretrained else None
            backbone = models.resnet34(weights=weights)
        else:
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if cfg.pretrained else None
            backbone = models.resnet18(weights=weights)
        return StageAttentionResNet(backbone=backbone, num_classes=cfg.num_classes, attention=attn)

    if m == "swin_t":
        weights = models.Swin_T_Weights.IMAGENET1K_V1 if cfg.pretrained else None
        model = models.swin_t(weights=weights)
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, cfg.num_classes)
        return model

    raise ValueError(f"Unsupported model_name: {cfg.model_name}")


def split_params(model: nn.Module):
    head_params = []
    backbone_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("fc") or n.startswith("head"):
            head_params.append(p)
        else:
            backbone_params.append(p)
    return head_params, backbone_params


def set_backbone_trainable(model: nn.Module, trainable: bool):
    for n, p in model.named_parameters():
        if n.startswith("fc") or n.startswith("head"):
            p.requires_grad = True
        else:
            p.requires_grad = trainable


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    acc_sum = 0.0
    count = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            bs = x.shape[0]
            loss_sum += loss.item() * bs
            acc_sum += accuracy(logits, y) * bs
            count += bs
    return loss_sum / max(1, count), acc_sum / max(1, count)


def train(cfg: Config):
    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)

    if cfg.freeze_backbone_epochs > 0:
        set_backbone_trainable(model, trainable=False)

    head_params, backbone_params = split_params(model)
    optim_groups = [{"params": head_params, "lr": cfg.lr}]
    if backbone_params:
        optim_groups.append({"params": backbone_params, "lr": cfg.lr * cfg.backbone_lr_mult})

    optimizer = torch.optim.SGD(optim_groups, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.CrossEntropyLoss()

    log_path = os.path.join(cfg.output_dir, "train_log.csv")
    ckpt_path = os.path.join(cfg.output_dir, "best.pt")

    best_acc = 0.0
    best_epoch = 0

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr_head", "lr_backbone"])

        for epoch in range(1, cfg.epochs + 1):
            if epoch == cfg.freeze_backbone_epochs + 1 and cfg.freeze_backbone_epochs > 0:
                set_backbone_trainable(model, trainable=True)
                head_params, backbone_params = split_params(model)
                optim_groups = [{"params": head_params, "lr": cfg.lr}]
                if backbone_params:
                    optim_groups.append({"params": backbone_params, "lr": cfg.lr * cfg.backbone_lr_mult})
                optimizer = torch.optim.SGD(optim_groups, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs - epoch + 1)

            model.train()
            total_loss = 0.0
            total_acc = 0.0
            total_count = 0

            train_iter = train_loader
            if tqdm is not None:
                train_iter = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}", leave=False)

            for x, y in train_iter:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                bs = x.shape[0]
                total_loss += loss.item() * bs
                total_acc += accuracy(logits, y) * bs
                total_count += bs
                if tqdm is not None:
                    train_iter.set_postfix(
                        loss=f"{(total_loss / max(1, total_count)):.4f}",
                        acc=f"{(total_acc / max(1, total_count)):.4f}",
                    )

            scheduler.step()

            train_loss = total_loss / max(1, total_count)
            train_acc = total_acc / max(1, total_count)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            lrs = [g["lr"] for g in optimizer.param_groups]
            lr_head = lrs[0] if lrs else 0.0
            lr_backbone = lrs[1] if len(lrs) > 1 else 0.0
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, lr_head, lr_backbone])
            f.flush()

            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                torch.save({"model": model.state_dict(), "config": asdict(cfg), "best_acc": best_acc, "best_epoch": best_epoch}, ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    with open(os.path.join(cfg.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_val_acc": best_acc,
                "best_val_epoch": best_epoch,
                "final_test_loss": test_loss,
                "final_test_acc": test_acc,
                "config": asdict(cfg),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Best val acc: {best_acc:.4f} at epoch {best_epoch} | Final test acc: {test_acc:.4f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--output_dir", type=str, default="./runs/exp")
    p.add_argument("--model_name", type=str, default="resnet18")
    p.add_argument("--num_classes", type=int, default=37)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--backbone_lr_mult", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--freeze_backbone_epochs", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = Config(**vars(args))
    train(cfg)
