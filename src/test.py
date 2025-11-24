import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import wandb
import os
from models import WildfireResNet, SimpleCNN
from dataset import get_dataloaders
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # 1. Setup
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    
    # Init WandB
    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(project=cfg.wandb.project, config=wandb_config, job_type="test", name="final_evaluation")

    # 2. Data & Model
    print("Loading Test Data...")
    # Note: We need the cfg here to know if we are using synthetic data or not
    # even if test set is always real.
    _, _, test_loader = get_dataloaders(cfg)
    
    print(f"Initializing Model Architecture: {cfg.model.name}...")
    if cfg.model.name == "resnet50":
        model = WildfireResNet(cfg.model.num_classes, cfg.model.pretrained, cfg.model.dropout)
    else:
        model = SimpleCNN(cfg.model.num_classes, cfg.model.dropout)
    
    # 3. Load Weights safely
    # We look for the file in the Original Working Directory (Colab root)
    # because Hydra changes the current directory to a new timestamp folder.
    load_path = os.path.join(hydra.utils.get_original_cwd(), "best_model.pth")
    
    if not os.path.exists(load_path):
        # Fallback: Check current directory just in case
        if os.path.exists("best_model.pth"):
            load_path = "best_model.pth"
        else:
            raise FileNotFoundError(f"CRITICAL ERROR: Could not find 'best_model.pth' at {load_path}. Did you run training first?")

    print(f"Loading weights from: {load_path}")
    model.load_state_dict(torch.load(load_path, map_location=device))
    model.to(device)
    model.eval()

    # 4. Inference
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

    # 5. Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary', pos_label=1)
    recall = recall_score(all_labels, all_preds, average='binary', pos_label=1)
    f1 = f1_score(all_labels, all_preds, average='binary', pos_label=1)

    print("FINAL TEST RESULTS")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    # 6. Report & Confusion Matrix
    report = classification_report(all_labels, all_preds, target_names=["No Wildfire", "Wildfire"])
    print(report)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                xticklabels=["Pred: No Fire", "Pred: Fire"], 
                yticklabels=["Actual: No Fire", "Actual: Fire"])
    plt.title('Test Set Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.savefig("confusion_matrix.png")
    
    # 7. Log to WandB
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
