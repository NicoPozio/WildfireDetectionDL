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
    
    # Convert config to dictionary for metadata
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    
    # Initialize Run
    run = wandb.init(
        project=cfg.wandb.project, 
        config=config_dict, 
        mode=cfg.wandb.mode,
        job_type="train"
    )
    
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

    # Initialize Optimizer
    if cfg.training.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    elif cfg.training.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=cfg.training.learning_rate, momentum=0.9, weight_decay=cfg.training.weight_decay)
    
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    patience = cfg.training.early_stopping_patience
    trigger_times = 0 
    
    # Temporary local filename (will be deleted after upload)
    temp_model_name = "model.pth"

    print("Starting Training Loop...")
    for epoch in range(cfg.training.epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", mininterval=10.0):
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

        metrics = {
            "train_loss": train_loss / len(train_loader),
            "train_acc": 100. * correct / total,
            "val_loss": val_loss / len(val_loader),
            "val_acc": 100. * val_correct / val_total,
            "epoch": epoch + 1
        }
        
        wandb.log(metrics)
        print(f"   Val Acc: {metrics['val_acc']:.2f}%")

        if metrics['val_acc'] > best_val_acc:
            print(f"   New Best Model ({metrics['val_acc']:.2f}%)! Saving Artifact...")
            best_val_acc = metrics['val_acc']
            trigger_times = 0 
            
            # 1. Save weights locally
            torch.save(model.state_dict(), temp_model_name)
            
            # 2. Create Artifact
            # We name the artifact based on the architecture so they are grouped cleanly
            artifact = wandb.Artifact(
                name=f"model-{cfg.model.name}", 
                type="model",
                description=f"Accuracy: {best_val_acc:.2f}%",
                metadata=config_dict # Critical: Embeds the config into the artifact
            )
            
            # 3. Add file and Log
            artifact.add_file(temp_model_name)
            run.log_artifact(artifact)
                
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("   Early Stopping Triggered.")
                break

    wandb.finish()

if __name__ == "__main__":
    main()
