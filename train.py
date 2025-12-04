import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from tqdm import tqdm
from src.models import WildfireResNet, SimpleCNN, WildfireEfficientNet
from src.dataset import get_dataloaders
from src.utils import seed_everything
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.training.seed)
    
    # Save config as dict for Artifact Metadata
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    
    run = wandb.init(project=cfg.wandb.project, config=config_dict, mode=cfg.wandb.mode)
    
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    print(f"Training on {device} using model: {cfg.model.name}")

    train_loader, val_loader, _ = get_dataloaders(cfg) 
    if not train_loader: return

    # Initialize Model
    if cfg.model.name == "resnet50":
        model = WildfireResNet(num_classes=2, pretrained=cfg.model.pretrained, dropout=cfg.model.dropout).to(device)
    elif cfg.model.name == "simple_cnn":
        model = SimpleCNN(num_classes=2, dropout=cfg.model.dropout).to(device)
    elif cfg.model.name == "efficientnet":
        model = WildfireEfficientNet(num_classes=2, pretrained=cfg.model.pretrained, dropout=cfg.model.dropout).to(device)
    else:
        raise ValueError(f"Unknown model: {cfg.model.name}")

    if cfg.training.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    elif cfg.training.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=cfg.training.learning_rate, momentum=0.9, weight_decay=cfg.training.weight_decay)
    
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    patience = cfg.training.early_stopping_patience
    trigger_times = 0 
    
    # Local temp file
    temp_file = "model_weights.pth"

    print("Starting Training Loop...")
    for epoch in range(cfg.training.epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", mininterval=10.0):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(outputs := model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                _, predicted = model(images).max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_acc = 100. * correct / total
        print(f"   Val Acc: {val_acc:.2f}%")
        wandb.log({"val_acc": val_acc, "epoch": epoch+1})

        if val_acc > best_val_acc:
            print(f"   New Best! Saving Artifact...")
            best_val_acc = val_acc
            trigger_times = 0 
            
            # 1. Save locally
            torch.save(model.state_dict(), temp_file)
            
            # 2. Log as Artifact
            # We create a new artifact version for every improvement or just overwrite
            # Here we create a new one to be safe
            artifact = wandb.Artifact(
                name=f"model-{cfg.model.name}-{wandb.run.id}", 
                type="model",
                metadata=config_dict
            )
            artifact.add_file(temp_file)
            run.log_artifact(artifact)
                
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("   Early Stopping.")
                break

    wandb.finish()

if __name__ == "__main__":
    main()
