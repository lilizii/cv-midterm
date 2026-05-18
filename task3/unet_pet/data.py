import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import InterpolationMode


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PetSegmentationDataset(Dataset):
    def __init__(self, root: str, split: str, image_size: int, augment: bool = False):
        self.ds = OxfordIIITPet(
            root=root,
            split=split,
            target_types='segmentation',
            download=True,
        )
        self.augment = augment
        self.img_tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.mask_resize = transforms.Resize((image_size, image_size), interpolation=InterpolationMode.NEAREST)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        image, mask = self.ds[idx]
        if self.augment and random.random() < 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)
        image = self.img_tf(image)
        mask = self.mask_resize(mask)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64)) - 1
        mask = mask.clamp(0, 2)
        return image, mask


def split_train_val(dataset: Dataset, val_ratio: float, seed: int) -> Tuple[Subset, Subset]:
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_size = max(1, int(len(indices) * val_ratio))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)
