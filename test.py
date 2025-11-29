import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import wandb
import os
from src.models import WildfireResNet, SimpleCNN, WildfireEfficientNet
from src.dataset import get_dataloaders
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # 1. Setup
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    
    # Init WandB for the Test Run
    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(project=cfg.wandb.project, config=wandb_config, job_type="test", name="final_evaluation")

    # 2. Data Loading
    print("Loading Test Data...")
    # We call get_dataloaders to ensure the config context is loaded correctly,
    # but we only extract the test_loader (3rd return value).
    _, _, test_loader = get_dataloaders(cfg)
    
    # 3. Model Initialization
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
        raise ValueError(f"Unknown model name: {cfg.model.name}")

    # 4. Load Weights safely
    # Hydra changes the current working directory to a timestamped folder.
    # We need to find 'best_model.pth' in the ORIGINAL working directory (the root of the project).
    try:
        orig_cwd = hydra.utils.get_original_cwd()
        load_path = os.path.join(orig_cwd, "best_model.pth")
    except Exception:
        # Fallback if not running under Hydra (e.g., direct execution)
        load_path = "best_model.pth"
    
    if not os.path.exists(load_path):
        # Fallback check in current directory
        if os.path.exists("best_model.pth"):
            load_path = "best_model.pth"
        else:
            raise FileNotFoundError(f"CRITICAL ERROR: Could not find 'best_model.pth' at {load_path}. Did you run training first?")

    print(f"Loading weights from: {load_path}")
    model.load_state_dict(torch.load(load_path, map_location=device))
    model.to(device)
    model.eval()

    # 5. Inference Loop
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

    # 6. Metrics Calculation
    # We assume: 0 = No Wildfire, 1 = Wildfire
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

    # 7. Classification Report
    print("\nDetailed Classification Report:")
    report = classification_report(all_labels, all_preds, target_names=["No Wildfire", "Wildfire"])
    print(report)

    # 8. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                xticklabels=["Pred: No Fire", "Pred: Fire"], 
                yticklabels=["Actual: No Fire", "Actual: Fire"])
    plt.title('Test Set Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.savefig("confusion_matrix.png")
    print("Confusion matrix saved to confusion_matrix.png")
    
    # 9. Logging to WandB
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
