import argparse

import torch
from torch.utils.data import DataLoader

from unet_pet.data import PetSegmentationDataset
from unet_pet.engine import evaluate
from unet_pet.losses import build_loss
from unet_pet.model import UNet


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained U-Net checkpoint on Oxford-IIIT Pet test set.')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint file, e.g. outputs/unet_combo_best.pth')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--base_ch', type=int, default=32)
    parser.add_argument('--mode', type=str, default='combo', choices=['ce', 'dice', 'combo'])
    parser.add_argument('--combo_ce_weight', type=float, default=0.25)
    parser.add_argument('--combo_dice_weight', type=float, default=0.75)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_ch=3, num_classes=3, base_ch=args.base_ch).to(device)

    ckpt = torch.load(args.ckpt, map_location='cpu')
    state = ckpt.get('model', ckpt)
    model.load_state_dict(state, strict=True)

    criterion = build_loss(args.mode, ce_weight=args.combo_ce_weight, dice_weight=args.combo_dice_weight)
    test_ds = PetSegmentationDataset(args.data_root, split='test', image_size=args.image_size, augment=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )

    test_loss, test_miou = evaluate(model, test_loader, criterion, device)
    print(f'[Test] mode={args.mode} checkpoint={args.ckpt}')
    print(f'[Test] loss={test_loss:.6f} mIoU={test_miou:.6f}')


if __name__ == '__main__':
    main()
