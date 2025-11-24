import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
import shutil # Added for file copying
from tqdm import tqdm
from models import WildfireResNet, SimpleCNN
from dataset import get_dataloaders
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
    train_loader, val_loader, _ = get_dataloaders(cfg) # We ignore test_loader here
    if not train_loader: return

    # 3. Model
    if cfg.model.name == "resnet50":
        model = WildfireResNet(cfg.model.num_classes, cfg.model.pretrained, cfg.model.dropout).to(device)
    elif cfg.model.name == "simple_cnn":
        model = SimpleCNN(cfg.model.num_classes, cfg.model.dropout).to(device)
    else:
        raise ValueError(f"Unknown model: {cfg.model.name}")

    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # 4. Variables for Tracking
    best_val_acc = 0.0
    patience = cfg.training.early_stopping_patience
    trigger_times = 0 # Counter for early stopping

    # 5. Training Loop
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

        # --- METRICS ---
        metrics = {
            "train_loss": train_loss / len(train_loader),
            "train_acc": 100. * correct / total,
            "val_loss": val_loss / len(val_loader),
            "val_acc": 100. * val_correct / val_total,
            "epoch": epoch + 1
        }
        
        wandb.log(metrics)
        print(f"   Train Acc: {metrics['train_acc']:.2f}% | Val Acc: {metrics['val_acc']:.2f}%")

        # --- SAVE & EARLY STOPPING ---
        # We use a fixed name so test.py can find it easily
        filename = "best_model.pth" 
        
        if metrics['val_acc'] > best_val_acc:
            print(f"Validation Accuracy improved ({best_val_acc:.2f}% -> {metrics['val_acc']:.2f}%). Saving model...")
            best_val_acc = metrics['val_acc']
            trigger_times = 0 # Reset early stopping counter
            
            # Save locally (inside Hydra folder)
            torch.save(model.state_dict(), filename)
            
            # Update WandB
            wandb.save(filename)
            
            # CRITICAL: Save a copy to the Original Working Directory (Colab Root)
            # This ensures test.py can find it even if it runs in a different Hydra folder
            orig_cwd = hydra.utils.get_original_cwd()
            shutil.copyfile(filename, os.path.join(orig_cwd, filename))
            
        else:
            trigger_times += 1
            print(f"No improvement. Early Stopping Patience: {trigger_times}/{patience}")
            
            if trigger_times >= patience:
                print("Early Stopping Triggered. Training stopped.")
                break

    print("Training Complete.")

if __name__ == "__main__":
    main()
