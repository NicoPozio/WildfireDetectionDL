import os
import torch
import glob
from torch.utils.data import DataLoader, ConcatDataset, Dataset, Subset
from torchvision import datasets, transforms
from PIL import Image
from typing import Tuple, Optional

# Robust Synthetic Dataset Class
class SyntheticWildfireDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        target_folder = os.path.join(root_dir, "wildfire")
        self.image_paths = []
        if os.path.exists(target_folder):
            for ext in ["*.png", "*.jpg", "*.jpeg"]:
                self.image_paths.extend(glob.glob(os.path.join(target_folder, ext)))
        print(f"   -> Found {len(self.image_paths)} synthetic wildfire images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 1 # Force Label 1

# Main Loader Logic
def get_dataloaders(cfg):
    input_size = tuple(cfg.dataset.params.input_size)
    mean = cfg.dataset.params.mean
    std = cfg.dataset.params.std
    
    #Transforms
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomRotation(degrees=cfg.dataset.augmentation.rotation_degrees),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    #Load Full Real Data
    try:
        full_real_train_ds = datasets.ImageFolder(cfg.dataset.paths.real_train_path, transform=train_transform)
        val_ds = datasets.ImageFolder(cfg.dataset.paths.val_path, transform=eval_transform)
        test_ds = datasets.ImageFolder(cfg.dataset.paths.test_path, transform=eval_transform)
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load real data: {e}")
        return None, None, None

    # DATA SCARCITY LOGIC (Reads from dataset.yaml)
    # Check if 'real_data_fraction' exists in config, default to 1.0 (100%)
    fraction = 1.0
    if hasattr(cfg.dataset.params, 'real_data_fraction'):
        fraction = cfg.dataset.params.real_data_fraction

    if fraction < 1.0:
        total_count = len(full_real_train_ds)
        subset_count = int(total_count * fraction)
        
        print(f"SCARCITY MODE ENABLED: Using {fraction*100}% of Real Data.")
        print(f"   Original Size: {total_count} -> Reduced Size: {subset_count}")
        
        # We use a FIXED seed (42) to ensure we pick the exact same images 
        # every time we run this, ensuring a fair A/B test.
        indices = torch.randperm(total_count, generator=torch.Generator().manual_seed(42))[:subset_count]
        
        real_train_ds = Subset(full_real_train_ds, indices)
        
        real_train_ds.class_to_idx = full_real_train_ds.class_to_idx
    else:
        real_train_ds = full_real_train_ds

    print(f"Real Data Ready. Size: {len(real_train_ds)}")

    #Synthetic Injection
    if cfg.dataset.params.use_synthetic:
        print(f"AUGMENTATION ON: Injecting synthetic data...")
        synthetic_ds = SyntheticWildfireDataset(
            root_dir=cfg.dataset.paths.synthetic_train_path,
            transform=train_transform
        )
        
        if len(synthetic_ds) > 0:
            train_ds = ConcatDataset([real_train_ds, synthetic_ds])
            print(f"   -> Merged Dataset Size: {len(train_ds)}")
            
            #Calculate influence ratio
            ratio = len(synthetic_ds) / len(train_ds)
            print(f"   -> Synthetic Influence: {ratio*100:.2f}% of training data")
        else:
            train_ds = real_train_ds
    else:
        print("BASELINE MODE: Using only Real data.")
        train_ds = real_train_ds

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.dataset.params.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.dataset.params.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.dataset.params.num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
