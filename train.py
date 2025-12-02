%%writefile /content/WildfireDetectionDL/train.py
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
import shutil
from tqdm import tqdm
from src.models import WildfireResNet, SimpleCNN, WildfireEfficientNet
from src.dataset import get_dataloaders
from src.utils import seed_everything
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.training.seed)
    
    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.init(project=cfg.wandb.project, config=wandb_config, mode=cfg.wandb.mode)
    
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    print(f"Training on {device} using model: {cfg.model.name}")

    train_loader, val_loader, _ = get_dataloaders(cfg) 
    if not train_loader: return

    # Model Factory
    if cfg.model.name == "resnet50":
        model = WildfireResNet(
            num_classes=cfg.model.num_classes, 
            pretrained=cfg.model.pretrained, 
            dropout=cfg.model.dropout
        ).to(device)
    elif cfg.model.name == "simple_cnn":
        model = SimpleCNN(
            num_classes=cfg.model.num_classes, 
            dropout=cfg.model.dropout
        ).to(device)
    elif cfg.model.name == "efficientnet":
        model = WildfireEfficientNet(
            num_classes=cfg.model.num_classes,
            pretrained=cfg.model.pretrained,
            dropout=cfg.model.dropout
        ).to(device)
    else:
        raise ValueError(f"Unknown model: {cfg.model.name}")

    # Optimizer Factory
    if cfg.training.optimizer == "adam":
        optimizer = optim.Adam(
            model.parameters(), 
            lr=cfg.training.learning_rate, 
            weight_decay=cfg.training.weight_decay
        )
    elif cfg.training.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(), 
            lr=cfg.training.learning_rate, 
            momentum=0.9, 
            weight_decay=cfg.training.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer type: {cfg.training.optimizer}")

    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    patience = cfg.training.early_stopping_patience
    trigger_times = 0 
    
    # Use unique filename locally to prevent collision during parallel/sequential runs
    unique_filename = f"model_{wandb.run.id}.pth"

    print("Starting Training Loop...")
    for epoch in range(cfg.training.epochs):
        # Train
        model.train()
        train_loss, correct, total = 0, 0, 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.training.epochs}", mininterval=10.0):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        # Metrics
        metrics = {
            "train_loss": train_loss / len(train_loader),
            "train_acc": 100. * correct / total,
            "val_loss": val_loss / len(val_loader),
            "val_acc": 100. * val_correct / val_total,
            "epoch": epoch + 1
        }
        
        wandb.log(metrics)
        print(f"   Train Loss: {metrics['train_loss']:.4f} | Train Acc: {metrics['train_acc']:.2f}%")
        print(f"   Val Loss:   {metrics['val_loss']:.4f} | Val Acc:   {metrics['val_acc']:.2f}%")

        # Save & Early Stopping
        if metrics['val_acc'] > best_val_acc:
            print(f"   Validation Accuracy improved ({best_val_acc:.2f}% -> {metrics['val_acc']:.2f}%). Saving model...")
            best_val_acc = metrics['val_acc']
            trigger_times = 0 
            
            # 1. Save Locally (Unique Name)
            torch.save(model.state_dict(), unique_filename)
            
            # 2. Upload to WandB (Standard Name for API consistency)
            # We copy it to 'best_model.pth' just for the upload, then delete
            shutil.copy(unique_filename, "best_model.pth")
            wandb.save("best_model.pth")
            
            # 3. Permanent Drive Backup
            try:
                drive_root = "/content/drive/MyDrive/Wildfire_Project/saved_models"
                run_name = f"{cfg.model.name}_{wandb.run.id}"
                drive_run_dir = os.path.join(drive_root, run_name)
                os.makedirs(drive_run_dir, exist_ok=True)
                
                # Save weights
                shutil.copyfile(unique_filename, os.path.join(drive_run_dir, "best_weights.pth"))
                # Save config
                OmegaConf.save(cfg, os.path.join(drive_run_dir, "config.yaml"))
                print(f"   Backed up to Drive: {drive_run_dir}")
            except Exception as e:
                print(f"   Drive Backup Failed: {e}")
                
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("   Early Stopping Triggered.")
                break

    wandb.finish()

if __name__ == "__main__":
    main()
