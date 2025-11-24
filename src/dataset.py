import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder
from typing import Tuple, Optional

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
    # Using specific stats for ResNet is crucial for convergence
    transform = transforms.Compose([
        transforms.Resize(tuple(cfg.dataset.input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #the normalize value are good for the resnet but they are not good for the customized CNN, we need to compute them
    ])

    try:
        real_train_ds = datasets.ImageFolder(cfg.dataset.real_train_path, transform=transform)
        val_ds = datasets.ImageFolder(cfg.dataset.val_path, transform=transform)
        test_ds = datasets.ImageFolder(cfg.dataset.test_path, transform=transform)
        
        #Capture the mapping ({'nowildfire': 0, 'wildfire': 1})
        master_mapping = real_train_ds.class_to_idx
        print(f"Real Data Loaded. Class Mapping: {master_mapping}")
        
    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: Could not load real data: {e}")
        return None, None, None # Return 3 Nones (Train, Val, Test)

    #Synthetic data
    if cfg.dataset.use_synthetic:
        print(f"AUGMENTATION ON: Injecting synthetic data from {cfg.dataset.synthetic_train_path}")
        
        if os.path.exists(cfg.dataset.synthetic_train_path):
            try:
                # We use the custom class and pass the 'master_mapping'
                synthetic_ds = EnforcedMapDataset(
                    root=cfg.dataset.synthetic_train_path,
                    transform=transform,
                    forced_class_to_idx=master_mapping 
                )
                
                # Merge the datasets
                train_ds = ConcatDataset([real_train_ds, synthetic_ds])
                
                print(f"   -> Added {len(synthetic_ds)} synthetic images.")
                print(f"   -> Combined Training Set: {len(train_ds)} images.")
                
                # Sanity Check: Ensure synthetic_ds actually respects the label 1
                if len(synthetic_ds) > 0:
                     # Check the label of the first item
                    _, label = synthetic_ds[0]
                    expected_label = master_mapping['wildfire'] # Should be 1
                    if label != expected_label:
                        print(f"WARNING: Label mismatch! Synthetic label is {label}, expected {expected_label}")

            except Exception as e:
                print(f"WARNING: Failed to load synthetic data ({e}). Fallback to Real Only.")
                train_ds = real_train_ds
        else:
            print(f"WARNING: Synthetic path not found. Using only Real data.")
            train_ds = real_train_ds
    else:
        print("BASELINE MODE: Using only Real data.")
        train_ds = real_train_ds

    # Create DataLoaders
    # Note: num_workers should ideally come from cfg, but hardcoded 2 is safe for Colab
    batch_size = cfg.training.batch_size
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader
