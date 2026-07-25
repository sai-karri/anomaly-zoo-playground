import torch
import argparse
import torch.nn as nn
from pathlib import Path
from src.datasets.mvtec import MVTecDataset
from src.models.image.ae import AutoEncoder
from src.models.image.patchcore import PatchCore
from src.evaluation import metrics
from torch.utils.data import DataLoader
import mlflow
import modal

app = modal.App("anomaly-detection")

image = modal.Image.debian_slim().pip_install(
    "torch", "torchvision", "tqdm", "scikit-learn"
)


parser = argparse.ArgumentParser()
parser.add_argument('--category', type=str, default='bottle')
parser.add_argument('--model', type=str, default='AE')
args = parser.parse_args()

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
root = Path(__file__).parent.parent / 'data' / 'mvtec'
checkpoint_path = Path(__file__).parent.parent / 'checkpoints' / f"{args.model}_{args.category}.pt"
memory_bank_path = Path(__file__).parent.parent / 'checkpoints' / f"{args.model}_memory_bank_{args.category}.pt"

mlflow.set_experiment("anomaly-detection")

with mlflow.start_run():

    # ── params ────────────────────────────────────────────────────────────────
    mlflow.log_param("model",    args.model)
    mlflow.log_param("category", args.category)
    mlflow.log_param("img_size", 256)
    mlflow.log_param("batch_size", 8)
    mlflow.set_tag("dataset", "mvtec")

    dataloader = DataLoader(
        MVTecDataset(root=root, category=args.category, split='train'),
        batch_size=8,
        shuffle=True
    )

    if args.model == 'AE':
        mlflow.log_param("epochs", 50)
        mlflow.log_param("lr", 1e-3)
        mlflow.set_tag("backbone", "none")

        model = AutoEncoder().to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        epochs = 50
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0

            for batch in dataloader:
                images = batch['image'].to(device)
                optimizer.zero_grad()
                out = model(images)
                loss = criterion(out, images)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            average_loss = epoch_loss / len(dataloader)
            mlflow.log_metric("train_loss", average_loss, step=epoch)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.6f}")

        torch.save(model.state_dict(), checkpoint_path)
        mlflow.log_artifact(str(checkpoint_path))

    elif args.model == 'PatchCore':
        mlflow.log_param("coreset_ratio", 0.1)
        mlflow.log_param("k", 9)
        mlflow.set_tag("backbone", "resnet50")

        model = PatchCore().to(device)
        model.fit(dataloader)
        print(f"Memory bank shape after fit: {model.memory_bank.shape}")

        torch.save(model.memory_bank, memory_bank_path)
        mlflow.log_artifact(str(memory_bank_path))

    # ── evaluation ────────────────────────────────────────────────────────────
    test_dataloader = DataLoader(
        MVTecDataset(root=root, category=args.category, split='test')
    )
    auroc = metrics.evaluation(model, test_dataloader, device)
    print(f"AUROC: {auroc:.4f}")
    mlflow.log_metric("auroc", auroc)