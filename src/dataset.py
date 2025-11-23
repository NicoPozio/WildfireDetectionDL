import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset

def get_dataloaders(cfg):
    # Define Transforms (Standard ImageNet stats)
    transform = transforms.Compose([
        transforms.Resize(tuple(cfg.dataset.input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load Real Data
    try:
        real_train_ds = datasets.ImageFolder(cfg.dataset.real_train_path, transform=transform)
        val_ds = datasets.ImageFolder(cfg.dataset.val_path, transform=transform)
        test_ds = datasets.ImageFolder(cfg.dataset.test_path, transform=transform)
    except FileNotFoundError as e:
        print(f" Error loading data: {e}")
        return None, None

    # Augmentation Logic
    if cfg.dataset.use_synthetic:
        print(f" AUGMENTATION ON: Mixing Real data with Synthetic data from {cfg.dataset.synthetic_train_path}")
        if os.path.exists(cfg.dataset.synthetic_train_path):
            # Synthetic folder might not have class subfolders. 
            # If it's just images, we might need a custom wrapper or assume 'wildfire' structure.
            # Assuming standard ImageFolder structure: root/class/image.jpg
            synthetic_ds = datasets.ImageFolder(cfg.dataset.synthetic_train_path, transform=transform)
            train_ds = ConcatDataset([real_train_ds, synthetic_ds])
            print(f"   -> Added {len(synthetic_ds)} synthetic images.")
        else:
            print(f"WARNING: Synthetic path not found. Using only Real data.")
            train_ds = real_train_ds
    else:
        print("BASELINE MODE: Using only Real data.")
        train_ds = real_train_ds

    # Dataloaders (numworkers should be defined in conf)
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader
