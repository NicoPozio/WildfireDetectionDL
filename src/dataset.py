import os
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
from typing import Tuple, Optional

# ---------------------------------------------------------
# Custom Dataset Class
# ---------------------------------------------------------
class EnforcedMapDataset(datasets.ImageFolder):
    """
    An extension of ImageFolder that forces a specific class-to-index mapping.
    This prevents 'Index Shifting' when a folder (like 'nowildfire') is empty
    in the synthetic dataset.
    """
    def __init__(self, root, transform=None, forced_class_to_idx=None):
        self.forced_class_to_idx = forced_class_to_idx
        super().__init__(root, transform=transform)

    def find_classes(self, directory: str) -> Tuple[list, dict]:
        if self.forced_class_to_idx is None:
            return super().find_classes(directory)
        
        # Return the classes explicitly defined by the Real Dataset
        classes = list(self.forced_class_to_idx.keys())
        return classes, self.forced_class_to_idx

# ---------------------------------------------------------
# Main Loader Logic
# ---------------------------------------------------------
def get_dataloaders(cfg):
    """
    Constructs DataLoaders for Train, Validation, and Test sets.
    Handles dynamic augmentation and synthetic data injection.
    """
    
    input_size = tuple(cfg.dataset.params.input_size)
    mean = cfg.dataset.params.mean
    std = cfg.dataset.params.std
    
    # 1. Training Transform (Includes Augmentation)
    # Uses rotation defined in config/sweep
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomRotation(degrees=cfg.dataset.augmentation.rotation_degrees),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # 2. Validation/Test Transform (No Augmentation)
    # Only Resize and Normalize ensures fair evaluation
    eval_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # 3. Load Real Data (The Master Source)
    try:
        real_train_ds = datasets.ImageFolder(cfg.dataset.paths.real_train_path, transform=train_transform)
        val_ds = datasets.ImageFolder(cfg.dataset.paths.val_path, transform=eval_transform)
        test_ds = datasets.ImageFolder(cfg.dataset.paths.test_path, transform=eval_transform)
        
        # Capture the True Mapping (e.g., {'nowildfire': 0, 'wildfire': 1})
        master_mapping = real_train_ds.class_to_idx
        print(f"Real Data Loaded. Class Mapping: {master_mapping}")
        
    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: Could not load real data: {e}")
        return None, None, None

    # 4. Augmentation Logic (Synthetic Injection)
    if cfg.dataset.params.use_synthetic:
        print(f"AUGMENTATION ON: Injecting synthetic data from {cfg.dataset.paths.synthetic_train_path}")
        
        if os.path.exists(cfg.dataset.paths.synthetic_train_path):
            try:
                # Use EnforcedMapDataset to ensure labels match the real data
                synthetic_ds = EnforcedMapDataset(
                    root=cfg.dataset.paths.synthetic_train_path,
                    transform=train_transform,
                    forced_class_to_idx=master_mapping 
                )
                
                # Merge the datasets
                train_ds = ConcatDataset([real_train_ds, synthetic_ds])
                
                print(f"   -> Added {len(synthetic_ds)} synthetic images.")
                print(f"   -> Combined Training Set: {len(train_ds)} images.")
                
            except Exception as e:
                print(f"WARNING: Failed to load synthetic data ({e}). Fallback to Real Only.")
                train_ds = real_train_ds
        else:
            print(f"WARNING: Synthetic path not found. Using only Real data.")
            train_ds = real_train_ds
    else:
        print("BASELINE MODE: Using only Real data.")
        train_ds = real_train_ds

    # 5. Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.dataset.params.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.dataset.params.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.dataset.params.num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
