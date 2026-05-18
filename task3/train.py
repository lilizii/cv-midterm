import argparse
from typing import Dict

import torch
from torch.utils.data import DataLoader, Subset

from unet_pet.data import PetSegmentationDataset, set_seed, split_train_val
from unet_pet.engine import RunResult, train_one_mode
from unet_pet.visualize import save_plots, save_summary


def main():
    parser = argparse.ArgumentParser(description='Train U-Net on Oxford-IIIT Pet with CE/Dice/Combo losses.')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--out_dir', type=str, default='./outputs')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--base_ch', type=int, default=32)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--modes', nargs='+', default=['ce', 'dice', 'combo'], choices=['ce', 'dice', 'combo'])
    parser.add_argument('--combo_ce_weight', type=float, default=0.25)
    parser.add_argument('--combo_dice_weight', type=float, default=0.75)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    full_trainval_aug = PetSegmentationDataset(args.data_root, split='trainval', image_size=args.image_size, augment=True)
    full_trainval_plain = PetSegmentationDataset(args.data_root, split='trainval', image_size=args.image_size, augment=False)

    train_subset, val_subset = split_train_val(full_trainval_aug, val_ratio=args.val_ratio, seed=args.seed)
    train_ds = Subset(full_trainval_aug, train_subset.indices)
    val_ds = Subset(full_trainval_plain, val_subset.indices)
    test_ds = PetSegmentationDataset(args.data_root, split='test', image_size=args.image_size, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )

    print(f'Dataset split: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} (val_ratio={args.val_ratio})')

    results: Dict[str, RunResult] = {}
    for mode in args.modes:
        print(f'\n=== Training loss mode: {mode} ===')
        results[mode] = train_one_mode(
            mode=mode,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            out_dir=args.out_dir,
            amp=args.amp,
            base_ch=args.base_ch,
            patience=args.patience,
            ce_weight=args.combo_ce_weight,
            dice_weight=args.combo_dice_weight,
        )

    save_plots(results, args.out_dir)
    save_summary(results, args.out_dir)

    print('\n===== Final Comparison =====')
    for mode in args.modes:
        r = results[mode]
        print(
            f'{r.mode:>5s}: best_val_mIoU={r.best_miou:.4f} @ epoch {r.best_epoch}, '
            f'final_test_mIoU={r.final_test_miou:.4f}, final_test_loss={r.final_test_loss:.4f}'
        )
    print(f'\nSaved CSV and plots to: {args.out_dir}')


if __name__ == '__main__':
    main()
