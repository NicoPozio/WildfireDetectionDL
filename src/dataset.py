import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder

class EnforcedMapDataset(ImageFolder):
    """
    An ImageFolder that forces a specific class-to-index mapping.
    This allows loading a dataset where some classes might be empty (e.g., nowildfire)
    without shifting the indices of the existing classes (wildfire).
    """
    def __init__(self, root, transform=None, forced_class_to_idx=None):
        # We save the map BEFORE calling super().__init__
        self.forced_class_to_idx = forced_class_to_idx
        super().__init__(root, transform=transform)

    def find_classes(self, directory):
        """
        Override the default auto-detection.
        Instead of scanning the disk for folder names, we return the forced map.
        """
        if self.forced_class_to_idx is None:
            return super().find_classes(directory)
        
        # Reconstruct the classes list from the dict keys
        classes = list(self.forced_class_to_idx.keys())
        return classes, self.forced_class_to_idx



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
