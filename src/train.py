import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from tqdm import tqdm
from models import WildfireResNet, SimpleCNN
from dataset import get_dataloaders
from utils import seed_everything



@hydra.main(version_base=None, config_path="conf", config_name="config") # Ensure path is correct relative to script
def main(cfg: DictConfig):
    seed_everything(cfg.training.seed)
    
    # --- FIX 1: Proper Configuration Serialization ---
    # resolve=True ensures calculations (like ${training.batch_size}) are fixed values
    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    wandb.init(
        project=cfg.wandb.project, 
        config=wandb_config,  # Pass the clean dict, not the DictConfig object
        mode=cfg.wandb.mode
    )
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    print(f"Training on {device} using model: {cfg.model.name}")

    train_loader, val_loader = get_dataloaders(cfg)
    if not train_loader: return

    # Model Factory
    if cfg.model.name == "resnet50":
        model = WildfireResNet(cfg.model.num_classes, cfg.model.pretrained, cfg.model.dropout).to(device)
    elif cfg.model.name == "simple_cnn":
        model = SimpleCNN(cfg.model.num_classes, cfg.model.dropout).to(device)
    else:
        raise ValueError(f"Unknown model: {cfg.model.name}")

    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(cfg.training.epochs):
        # Train
        model.train()
        train_loss, correct, total = 0, 0, 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
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
        print(f"   Acc: {metrics['train_acc']:.2f}% | Val Acc: {metrics['val_acc']:.2f}%")

        # --- RECOMMENDATION: Save model with a unique name ---
    # Using "best_model.pth" can get overwritten if you run multiple experiments in the same folder.
    # Hydra changes directories automatically, so you are likely safe, but explicit naming is better.
        model_name = f"{cfg.model.name}_best.pth"
    
        if metrics['val_acc'] > best_val_acc:
            best_val_acc = metrics['val_acc']
            torch.save(model.state_dict(), model_name)
        
            # Save to WandB cloud so you don't lose it if Colab disconnects
            wandb.save(model_name) 

    print("Training Complete.")

if __name__ == "__main__":
    main()
