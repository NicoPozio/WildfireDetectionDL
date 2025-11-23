import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import wandb
from models import WildfireResNet, SimpleCNN
from dataset import get_dataloaders
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Init WandB for the Test Run (distinct from training)
    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(project=cfg.wandb.project, config=wandb_config, job_type="test", name="final_evaluation")

    # 2. Data & Model
    print("Loading Test Data...")
    _, _, test_loader = get_dataloaders(cfg) # We only need the 3rd return value
    
    print(f"Loading Model: {cfg.model.name}...")
    if cfg.model.name == "resnet50":
        model = WildfireResNet(cfg.model.num_classes, cfg.model.pretrained, cfg.model.dropout)
    else:
        model = SimpleCNN(cfg.model.num_classes, cfg.model.dropout)
    
    # Load the best weights saved during training
    # Ensure this path matches where train.py saved it
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device)
    model.eval()

    # 3. Inference Loop
    all_preds = []
    all_labels = []
    
    print("Starting Inference...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            # Move to CPU and convert to numpy for sklearn
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 4. Calculation of Metrics
    # 'average="binary"' calculates metrics for the Positive class (index 1)
    # We assume: 0 = No Wildfire, 1 = Wildfire
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary', pos_label=1)
    recall = recall_score(all_labels, all_preds, average='binary', pos_label=1)
    f1 = f1_score(all_labels, all_preds, average='binary', pos_label=1)

    print("\n" + "="*30)
    print("FINAL TEST RESULTS")
    print("="*30)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f} (Low FP)")
    print(f"Recall:    {recall:.4f}    (Low FN - Critical for Wildfire)")
    print(f"F1-Score:  {f1:.4f}")
    print("="*30)

    # 5. Generate Comprehensive Text Report
    # target_names must match your dataset class order (usually alphabetical or folder order)
    report = classification_report(all_labels, all_preds, target_names=["No Wildfire", "Wildfire"])
    print("\nDetailed Classification Report:\n")
    print(report)

    # 6. Visualization: Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                xticklabels=["Pred: No Fire", "Pred: Fire"], 
                yticklabels=["Actual: No Fire", "Actual: Fire"])
    plt.title('Test Set Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save plot locally
    plt.savefig("confusion_matrix.png")
    
    # 7. Log Everything to WandB
    wandb.log({
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
        "confusion_matrix": wandb.Image("confusion_matrix.png"),
        # Log the text report as a table or text is tricky, easier to rely on the scalar metrics above
    })
    
    print("Metrics and Confusion Matrix logged to WandB.")
    wandb.finish()

if __name__ == "__main__":
    main()
