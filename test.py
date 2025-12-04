import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import wandb
import os
from src.models import WildfireResNet, SimpleCNN, WildfireEfficientNet
from src.dataset import get_dataloaders
from src.utils import seed_everything
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, f1_score, precision_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def load_state_dict_robust(model, path, device):
    state_dict = torch.load(path, map_location=device)
    new_state = {}
    for k, v in state_dict.items():
        k = k.replace("backbone.", "base_model.").replace("module.", "")
        new_state[k] = v
    
    try:
        model.load_state_dict(new_state, strict=True)
    except:
        print("Warning: Strict loading failed. Retrying...")
        model.load_state_dict(new_state, strict=False)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    wandb.init(project=cfg.wandb.project, mode="disabled")

    print(f"Testing Model: {cfg.model.name}")
    
    if cfg.model.name == "resnet50":
        model = WildfireResNet(num_classes=2, pretrained=False, dropout=cfg.model.dropout)
    elif cfg.model.name == "simple_cnn":
        model = SimpleCNN(num_classes=2, dropout=cfg.model.dropout)
    elif cfg.model.name == "efficientnet":
        model = WildfireEfficientNet(num_classes=2, pretrained=False, dropout=cfg.model.dropout)
    
    model.to(device)

    load_path = cfg.get("model_path", "best_model.pth")
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Weight file not found: {load_path}")

    load_state_dict_robust(model, load_path, device)
    model.eval()

    _, _, test_loader = get_dataloaders(cfg)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\nRESULTS:")
    print(f"Accuracy:  {accuracy_score(all_labels, all_preds):.4f}")
    print(f"Precision: {precision_score(all_labels, all_preds):.4f}")
    print(f"Recall:    {recall_score(all_labels, all_preds):.4f}")
    print(f"F1-Score:  {f1_score(all_labels, all_preds):.4f}")
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.savefig("confusion_matrix.png")

if __name__ == "__main__":
    main()
