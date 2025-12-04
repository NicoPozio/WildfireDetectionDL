%%writefile /content/WildfireDetectionDL/src/dataset.py
import os
import torch
import glob
from torch.utils.data import DataLoader, ConcatDataset, Dataset, Subset
from torchvision import datasets, transforms
from PIL import Image
from typing import Tuple, Optional, Dict, List

#SAFETY: Enforced Mapping Loader 
class EnforcedFolder(datasets.ImageFolder):
    def __init__(self, root: str, transform=None, forced_map: Dict[str, int] = None):
        self.forced_map = forced_map
        super().__init__(root, transform=transform)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        if self.forced_map:
            classes = list(self.forced_map.keys())
            return classes, self.forced_map
        return super().find_classes(directory)

# SYNTHETIC DATASET HANDLER 
class SyntheticWildfireDataset(Dataset):
    def __init__(self, root_dir: str, transform=None, target_label: int = 1):
        self.root_dir = root_dir
        self.transform = transform
        self.target_label = target_label # Dynamic Label
        
        target_folder = os.path.join(root_dir, "wildfire")
        self.image_paths = []
        if os.path.exists(target_folder):
            for ext in ["*.png", "*.jpg", "*.jpeg"]:
                self.image_paths.extend(glob.glob(os.path.join(target_folder, ext)))
        
        print(f"   -> Found {len(self.image_paths)} synthetic wildfire images (Label: {self.target_label}).")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, self.target_label #Uses the passed index

#MAIN DATA FACTORY
def get_dataloaders(cfg):
    input_size = tuple(cfg.dataset.params.input_size)
    mean = cfg.dataset.params.mean
    std = cfg.dataset.params.std
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)), 
        transforms.RandomRotation(degrees=cfg.dataset.augmentation.rotation_degrees),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    try:
        full_real_train_ds = datasets.ImageFolder(cfg.dataset.paths.real_train_path, transform=train_transform)
        master_mapping = full_real_train_ds.class_to_idx
        print(f"Class Mapping Locked: {master_mapping}")
        
        # Identify the correct index for 'wildfire'
        # We try lowercase lookups to be robust
        fire_idx = 1 # Default
        for class_name, idx in master_mapping.items():
            if "no" not in class_name.lower() and ("fire" in class_name.lower() or "wild" in class_name.lower()):
                fire_idx = idx
                break
        print(f"   -> Synthetic data will use Class Index: {fire_idx}")
        
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load real training data: {e}")
        return None, None, None

    #Load Validation & Test
    try:
        val_ds = EnforcedFolder(cfg.dataset.paths.val_path, transform=eval_transform, forced_map=master_mapping)
        test_ds = EnforcedFolder(cfg.dataset.paths.test_path, transform=eval_transform, forced_map=master_mapping)
    except Exception as e:
        print(f"Error loading Val/Test sets: {e}")
        return None, None, None

    #Data Scarcity Logic
    fraction = 1.0
    if hasattr(cfg.dataset.params, 'real_data_fraction'):
        fraction = cfg.dataset.params.real_data_fraction

    if fraction < 1.0:
        total_count = len(full_real_train_ds)
        subset_count = int(total_count * fraction)
        print(f"SCARCITY MODE: Using {fraction*100}% of Real Data.")
        
        indices = torch.randperm(total_count, generator=torch.Generator().manual_seed(42))[:subset_count]
        real_train_ds = Subset(full_real_train_ds, indices)
        real_train_ds.class_to_idx = master_mapping
    else:
        real_train_ds = full_real_train_ds

    print(f"   Real Training Samples: {len(real_train_ds)}")

    # --- PHASE 4: Synthetic Injection ---
    if cfg.dataset.params.use_synthetic:
        print(f"SYNTHETIC INJECTION: Enabled")
        # Pass the dynamically found fire_idx
        synthetic_ds = SyntheticWildfireDataset(
            root_dir=cfg.dataset.paths.synthetic_train_path,
            transform=train_transform,
            target_label=fire_idx 
        )
        
        if len(synthetic_ds) > 0:
            train_ds = ConcatDataset([real_train_ds, synthetic_ds])
            print(f"Total Training Size: {len(train_ds)} (Real + Synthetic)")
        else:
            print(" Warning: No synthetic files found. Using Real only.")
            train_ds = real_train_ds
    else:
        print("   Baseline Mode: Real Data Only")
        train_ds = real_train_ds

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.dataset.params.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.dataset.params.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.dataset.params.num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
