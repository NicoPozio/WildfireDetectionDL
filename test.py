import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import os
from src.models import WildfireResNet, SimpleCNN, WildfireEfficientNet
from src.dataset import get_dataloaders
from src.utils import seed_everything
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def fix_key(key):
    if key.startswith("backbone."):
        return key.replace("backbone.", "base_model.")
    if key.startswith("module."):
        return key.replace("module.", "")
    return key

def load_state_dict_robust(model, path, device):
    state_dict = torch.load(path, map_location=device)
    new_state_dict = {}
    
    for k, v in state_dict.items():
        new_state_dict[fix_key(k)] = v
        
    try:
        model.load_state_dict(new_state_dict, strict=True)
    except RuntimeError:
        print("Warning: Strict loading failed. Retrying with strict=False.")
        model.load_state_dict(new_state_dict, strict=False)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.training.seed)
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    
    print(f"Initializing Model: {cfg.model.name}")
    _, _, test_loader = get_dataloaders(cfg)
    
    if cfg.model.name == "resnet50":
        model = WildfireResNet(num_classes=2, pretrained=cfg.model.pretrained, dropout=cfg.model.dropout)
    elif cfg.model.name == "simple_cnn":
        model = SimpleCNN(num_classes=2, dropout=cfg.model.dropout)
    elif cfg.model.name == "efficientnet":
        model = WildfireEfficientNet(num_classes=2, pretrained=cfg.model.pretrained, dropout=cfg.model.dropout)
    else:
        raise ValueError(f"Unknown model: {cfg.model.name}")

    load_path = cfg.get("model_path", "best_model.pth")
    
    if not os.path.exists(load_path):
        try:
            load_path = os.path.join(hydra.utils.get_original_cwd(), "best_model.pth")
        except:
            pass

    if not os.path.exists(load_path):
        raise FileNotFoundError(f"CRITICAL ERROR: Weight file not found at: {load_path}")

    print(f"Loading weights from: {load_path}")
    model.to(device)
    load_state_dict_robust(model, load_path, device)
    model.eval()

    all_preds = []
    all_labels = []
    
    print("Starting Inference...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary', pos_label=1)
    recall = recall_score(all_labels, all_preds, average='binary', pos_label=1)
    f1 = f1_score(all_labels, all_preds, average='binary', pos_label=1)

    print("-" * 30)
    print("FINAL TEST RESULTS")
    print("-" * 30)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("-" * 30)

    print("\nDetailed Classification Report:")
    report = classification_report(all_labels, all_preds, target_names=["No Wildfire", "Wildfire"])
    print(report)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Pred: No Fire", "Pred: Fire"], 
                yticklabels=["Actual: No Fire", "Actual: Fire"])
    plt.title(f'Confusion Matrix: {cfg.model.name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("Confusion Matrix saved to confusion_matrix.png")

if __name__ == "__main__":
    main()
