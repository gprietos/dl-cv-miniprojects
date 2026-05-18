import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2 as transforms

from .transforms import get_classification_transforms
from .datasets import SubsetWrapper 

def get_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
    img_size: int = 224,
    split_ratio: tuple = (0.8, 0.1, 0.1),
    backbone: str = "resnet",
    batch_level_transforms: bool = False
):

    train_transform = get_classification_transforms(split="train", img_size=img_size, backbone=backbone)
    val_transform = get_classification_transforms(split="val", img_size=img_size, backbone=backbone)

    
    train_path = os.path.join(data_dir, "train")    # Check if dataset is already split or monolithic
    
    if os.path.exists(train_path):
        # --- Dataset is already split ---
        print("=> Found pre-split datasets (train/validation/test).")
        train_dataset = ImageFolder(root=train_path, transform=train_transform)
        val_dataset = ImageFolder(root=os.path.join(data_dir, "validation"), transform=val_transform)
        test_dataset = ImageFolder(root=os.path.join(data_dir, "test"), transform=val_transform)
        
        num_classes = len(train_dataset.classes)
        
    else:
        # --- Dataset is a single directory ---
        print("=> No splits found. Dynamically splitting the root dataset...")
        full_dataset = ImageFolder(root=data_dir, transform=None)
        num_classes = len(full_dataset.classes)
        
        total_len = len(full_dataset)
        train_len = int(split_ratio[0] * total_len)
        val_len = int(split_ratio[1] * total_len)
        test_len = total_len - train_len - val_len
        
        generator = torch.Generator().manual_seed(42)
        base_train, base_val, base_test = random_split(
            full_dataset, [train_len, val_len, test_len], generator=generator
        )

        train_dataset = SubsetWrapper(base_train, transform=train_transform)
        val_dataset = SubsetWrapper(base_val, transform=val_transform)
        test_dataset = SubsetWrapper(base_test, transform=val_transform)

    # Batch-Level Transforms
    collate_fn = None
    if batch_level_transforms:
        mixup = transforms.Mixup(alpha=0.8, num_classes=num_classes)
        cutmix = transforms.Cutmix(alpha=1.0, num_classes=num_classes)
        cutmix_or_mixup = transforms.RandomChoice([mixup, cutmix])
        
        def vi_collate_fn(batch):
            return cutmix_or_mixup(*torch.utils.data.dataloader.default_collate(batch))
        collate_fn = vi_collate_fn

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader