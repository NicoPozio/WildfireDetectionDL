import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
import shutil
from tqdm import tqdm
from models import WildfireResNet, SimpleCNN
from src.dataset import get_dataloaders
from utils import seed_everything

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.training.seed)
    
    # 1. WandB Setup
    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.init(
        project=cfg.wandb.project, 
        config=wandb_config, 
        mode=cfg.wandb.mode
    )
    
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    print(f"Training on {device} using model: {cfg.model.name}")

    # 2. Data
    # Note: test_loader is unused in training, but get_dataloaders returns 3 items
    train_loader, val_loader, _ = get_dataloaders(cfg) 
    if not train_loader: return

    # 3. Model Architecture
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
    else:
        raise ValueError(f"Unknown model: {cfg.model.name}")

    # 4. Optimizer Selection (Dynamic)
    # We support switching between Adam and SGD via config/sweep
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

    # 5. Tracking Variables
    best_val_acc = 0.0
    patience = cfg.training.early_stopping_patience
    trigger_times = 0 
    filename = "best_model.pth"

    # 6. Training Loop
    print("Starting Training Loop...")
    for epoch in range(cfg.training.epochs):
        # --- TRAIN ---
        model.train()
        train_loss, correct, total = 0, 0, 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.training.epochs}"):
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

        # --- VALIDATE ---
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

        # --- METRICS & LOGGING ---
        metrics = {
            "train_loss": train_loss / len(train_loader),
            "train_acc": 100. * correct / total,
            "val_loss": val_loss / len(val_loader),
            "val_acc": 100. * val_correct / val_total,
            "epoch": epoch + 1
        }
        
        wandb.log(metrics)
        print(f"   Train Acc: {metrics['train_acc']:.2f}% | Val Acc: {metrics['val_acc']:.2f}%")

        # --- EARLY STOPPING & SAVING ---
        if metrics['val_acc'] > best_val_acc:
            print(f"   Validation Accuracy improved ({best_val_acc:.2f}% -> {metrics['val_acc']:.2f}%). Saving model...")
            best_val_acc = metrics['val_acc']
            trigger_times = 0 
            
            # Save inside the Hydra output folder (current working directory)
            torch.save(model.state_dict(), filename)
            wandb.save(filename)
            
            # Save a copy to the Original Working Directory so test.py can always find it
            orig_cwd = hydra.utils.get_original_cwd()
            shutil.copyfile(filename, os.path.join(orig_cwd, filename))
            
        else:
            trigger_times += 1
            print(f"   No improvement. Early Stopping Patience: {trigger_times}/{patience}")
            
            if trigger_times >= patience:
                print("   Early Stopping Triggered. Training stopped.")
                break

    print("Training Complete.")
    wandb.finish()

if __name__ == "__main__":
    main()
