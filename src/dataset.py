import os
import torch
import glob
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torchvision import datasets, transforms
from PIL import Image
from typing import Tuple, Optional

# ---------------------------------------------------------
# Robust Synthetic Dataset Class
# ---------------------------------------------------------
class SyntheticWildfireDataset(Dataset):
    """
    Specifically designed to load ONLY wildfire images from the synthetic folder
    and ignore the empty 'nowildfire' folder.
    It forces the label to be 1 (Wildfire).
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # 1. We look specifically inside the 'wildfire' subfolder
        # This bypasses the empty 'nowildfire' folder entirely.
        target_folder = os.path.join(root_dir, "wildfire")
        
        # 2. Find all images (png, jpg, jpeg)
        self.image_paths = []
        if os.path.exists(target_folder):
            for ext in ["*.png", "*.jpg", "*.jpeg"]:
                self.image_paths.extend(glob.glob(os.path.join(target_folder, ext)))
        
        print(f"   -> Found {len(self.image_paths)} synthetic wildfire images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Open image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        # FORCE LABEL 1 (Wildfire)
        # We know these are all fires, so we hardcode the 1.
        return image, 1

# ---------------------------------------------------------
# Main Loader Logic
# ---------------------------------------------------------
def get_dataloaders(cfg):
    input_size = tuple(cfg.dataset.params.input_size)
    mean = cfg.dataset.params.mean
    std = cfg.dataset.params.std
    
    # 1. Transforms
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)
        transforms.RandomRotation(degrees=cfg.dataset.augmentation.rotation_degrees),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # 2. Load Real Data
    try:
        real_train_ds = datasets.ImageFolder(cfg.dataset.paths.real_train_path, transform=train_transform)
        val_ds = datasets.ImageFolder(cfg.dataset.paths.val_path, transform=eval_transform)
        test_ds = datasets.ImageFolder(cfg.dataset.paths.test_path, transform=eval_transform)
        print(f"Real Data Loaded. Train Size: {len(real_train_ds)}")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load real data: {e}")
        return None, None, None

    # 3. Synthetic Injection
    if cfg.dataset.params.use_synthetic:
        print(f"AUGMENTATION ON: Injecting synthetic data...")
        
        # Use the new Robust Class
        synthetic_ds = SyntheticWildfireDataset(
            root_dir=cfg.dataset.paths.synthetic_train_path,
            transform=train_transform
        )
        
        if len(synthetic_ds) > 0:
            train_ds = ConcatDataset([real_train_ds, synthetic_ds])
            print(f"   -> Merged Dataset Size: {len(train_ds)}")
        else:
            print("   WARNING: No synthetic images found. Using Real Only.")
            train_ds = real_train_ds
    else:
        print("BASELINE MODE: Using only Real data.")
        train_ds = real_train_ds

    # 4. DataLoaders
    # Pin_memory=True is faster on GPU, but crashes on CPU if CUDA is missing.
    #                                             
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.dataset.params.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.dataset.params.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.dataset.params.num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
