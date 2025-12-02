import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import wandb
import os
from src.models import WildfireResNet, SimpleCNN, WildfireEfficientNet
from src.dataset import get_dataloaders
from src.utils import seed_everything
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def fix_key(key):
    # Standardize prefixes to match the current model definition
    if key.startswith("backbone."):
        return key.replace("backbone.", "base_model.")
    return key

def load_state_dict_robust(model, path, device):
    state_dict = torch.load(path, map_location=device)
    new_state_dict = {}
    
    # Check if the file is SimpleCNN but model is ResNet/Efficient (Architecture mismatch)
    keys = list(state_dict.keys())
    is_file_simple = any("features.0" in k for k in keys)
    is_model_resnet = isinstance(model, WildfireResNet)
    
    if is_file_simple and is_model_resnet:
        raise RuntimeError("Architecture Mismatch: File contains SimpleCNN weights, but Model is ResNet.")

    for k, v in state_dict.items():
        new_state_dict[fix_key(k)] = v
        
    try:
        model.load_state_dict(new_state_dict, strict=True)
    except RuntimeError:
        print("Warning: Strict loading failed. Retrying with strict=False to ignore prefix mismatches.")
        model.load_state_dict(new_state_dict, strict=False)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Setup
    seed_everything(cfg.training.seed)
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    
    # Init WandB for the Test Run
    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(project=cfg.wandb.project, config=wandb_config, job_type="test", name="final_evaluation")

    # Data Loading
    print("Loading Test Data...")
    _, _, test_loader = get_dataloaders(cfg)
    
    # Model Initialization
    print(f"Initializing Model Architecture: {cfg.model.name}")
    
    if cfg.model.name == "resnet50":
        model = WildfireResNet(
            num_classes=cfg.model.num_classes, 
            pretrained=cfg.model.pretrained, 
            dropout=cfg.model.dropout
        )
    elif cfg.model.name == "simple_cnn":
        model = SimpleCNN(
            num_classes=cfg.model.num_classes, 
            dropout=cfg.model.dropout
        )
    elif cfg.model.name == "efficientnet":
        model = WildfireEfficientNet(
            num_classes=cfg.model.num_classes,
            pretrained=cfg.model.pretrained,
            dropout=cfg.model.dropout
        )
    else:
        raise ValueError(f"Unknown model: {cfg.model.name}")

    # Load Weights safely
    load_path = "best_model.pth"
    if not os.path.exists(load_path):
        try:
            orig_cwd = hydra.utils.get_original_cwd()
            alt_path = os.path.join(orig_cwd, "best_model.pth")
            if os.path.exists(alt_path):
                load_path = alt_path
        except Exception:
            pass

    if not os.path.exists(load_path):
        raise FileNotFoundError(f"CRITICAL ERROR: Could not find 'best_model.pth' at {load_path}. Did you run training first?")

    print(f"Loading weights from: {load_path}")
    model.to(device)
    load_state_dict_robust(model, load_path, device)
    model.eval()

    # Inference Loop
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

    # Metrics Calculation
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary', pos_label=1)
    recall = recall_score(all_labels, all_preds, average='binary', pos_label=1)
    f1 = f1_score(all_labels, all_preds, average='binary', pos_label=1)

    print("\n" + "="*30)
    print("FINAL TEST RESULTS")
    print("="*30)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("="*30)

    # Classification Report
    print("\nDetailed Classification Report:")
    report = classification_report(all_labels, all_preds, target_names=["No Wildfire", "Wildfire"])
    print(report)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                xticklabels=["Pred: No Fire", "Pred: Fire"], 
                yticklabels=["Actual: No Fire", "Actual: Fire"])
    plt.title('Test Set Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.savefig("confusion_matrix.png")
    
    # Logging to WandB
    wandb.log({
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
        "confusion_matrix": wandb.Image("confusion_matrix.png"),
    })
    
    wandb.finish()

if __name__ == "__main__":
    main()
