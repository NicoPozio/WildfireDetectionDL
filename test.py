%%writefile /content/WildfireDetectionDL/test.py
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
    
    # Get the keys from the current model to compare
    model_keys = list(model.state_dict().keys())
    model_str = str(model)
    
    print(f"DEBUG: Loaded {len(state_dict)} keys from artifact.")
    
    for k, v in state_dict.items():
        # 1. Remove standard DistributedDataParallel wrappers
        k = k.replace("module.", "")
        
        # 2. Handle 'backbone' prefixes often used in custom classes
        # If the model has 'backbone' but the weight doesn't, we might need to add it? 
        # Or usually the inverse: Weight has 'backbone.' but model is just 'resnet'.
        k = k.replace("backbone.", "base_model.") 
        
        # 3. ResNet Specific Fixes (fc vs classifier)
        if "fc." in k and "fc" not in model_str and "classifier" in model_str:
            # ResNet saved as 'fc' but new model uses 'classifier'
            k = k.replace("fc.", "classifier.")
            
        # 4. EfficientNet Specific Fixes (_fc vs classifier)
        if "classifier" in k and "classifier" not in model_str:
             k = k.replace("classifier", "_fc")
        elif "_fc" in k and "_fc" not in model_str:
             k = k.replace("_fc", "classifier")
             
        new_state[k] = v
    
    # Try Strict Load First
    try:
        model.load_state_dict(new_state, strict=True)
        print(">> SUCCESS: Weights loaded with strict=True")
    except RuntimeError as e:
        print(f">> WARNING: Strict loading failed. Attempting loose load...")
        
        # Perform loose load
        missing, unexpected = model.load_state_dict(new_state, strict=False)
        
        # --- CRITICAL DEBUGGING INFO ---
        # This will tell us EXACTLY why it is failing
        print(f"   Mismatch Analysis:")
        if len(missing) > 0:
            print(f"   [MISSING] Model expects these, but didn't find them: {missing[:3]} ... ({len(missing)} total)")
        if len(unexpected) > 0:
            print(f"   [UNEXPECTED] Artifact has these, but model doesn't want them: {unexpected[:3]} ... ({len(unexpected)} total)")
            
        # If we are missing the HEAD (fc/classifier), the model is useless.
        critical_missing = [k for k in missing if 'fc' in k or 'classifier' in k or 'head' in k]
        if critical_missing:
             print(f"   !!! CRITICAL ERROR: The Classification Layer is missing weights: {critical_missing}")

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
