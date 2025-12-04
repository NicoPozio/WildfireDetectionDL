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
    
    # Get current model structure for reference
    model_str = str(model)
    
    print(f"DEBUG: Processing {len(state_dict)} keys from artifact...")
    
    for k, v in state_dict.items():
        # 1. Standard Cleanup
        k = k.replace("module.", "")
        
        # 2. FIX: The Artifact has 'base_model', but Model wants 'backbone'
        # We replace the file's name with the model's expected name.
        if "base_model." in k:
            k = k.replace("base_model.", "backbone.")
            
        # 3. Handle EfficientNet specific mismatch (classifier vs _fc)
        # If the key has 'classifier' but the model uses '_fc'
        if "classifier" in k and "classifier" not in model_str:
             k = k.replace("classifier", "_fc")
        # If the key has '_fc' but the model uses 'classifier'
        elif "_fc" in k and "_fc" not in model_str:
             k = k.replace("_fc", "classifier")

        new_state[k] = v
    
    try:
        model.load_state_dict(new_state, strict=True)
        print(">> SUCCESS: Weights loaded with strict=True")
    except RuntimeError as e:
        print(f">> WARNING: Strict loading failed. Attempting loose load...")
        missing, unexpected = model.load_state_dict(new_state, strict=False)
        
        # Filter out noise (sometimes unexpected keys are just metadata)
        real_missing = [k for k in missing if "num_batches_tracked" not in k]
        
        if real_missing:
            print(f"   [MISSING] Model still needs: {real_missing[:3]} ...")
        else:
            print("   [INFO] Loose load successful (only harmless keys missing).")

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    wandb.init(project=cfg.wandb.project, mode="disabled")

    print(f"Testing Model: {cfg.model.name}")
    
    # Initialize Model
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

if __name__ == "__main__":
    main()
