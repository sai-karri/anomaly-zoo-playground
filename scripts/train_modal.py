import modal
from pathlib import Path

app = modal.App("anomaly-detection")

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "torchvision", "tqdm", "scikit-learn", "mlflow")
    .add_local_dir(
        Path(__file__).parent.parent / "src",
        remote_path="/root/src"
    )
)

data_volume        = modal.Volume.from_name("mvtec-dataset")
checkpoints_volume = modal.Volume.from_name("anomaly-checkpoints", create_if_missing=True)


@app.function(
    gpu="A10G",
    image=image,
    volumes={
        "/data/mvtec":    data_volume,
        "/checkpoints":   checkpoints_volume,
    },
    timeout=3600,
)
def train(category: str, model_name: str):

    import sys
    sys.path.insert(0, "/root")

    import torch
    import torch.nn as nn
    import mlflow
    from torch.utils.data import DataLoader
    from src.datasets.mvtec import MVTecDataset
    from src.models.image.ae import AutoEncoder
    from src.models.image.patchcore import PatchCore
    from src.evaluation import metrics

    device             = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root               = Path("/data/mvtec")
    checkpoint_path    = Path(f"/checkpoints/{model_name}_{category}.pt")
    memory_bank_path   = Path(f"/checkpoints/{model_name}_memory_bank_{category}.pt")

    print(f"Device: {device}")
    print(f"Training {model_name} on {category}")

    mlflow.set_experiment("anomaly-detection")

    with mlflow.start_run():

        mlflow.log_param("model",    model_name)
        mlflow.log_param("category", category)
        mlflow.log_param("device",   str(device))

        dataloader = DataLoader(
            MVTecDataset(root=root, category=category, split="train"),
            batch_size=8,
            shuffle=True,
            num_workers=4,
        )

        if model_name == "AE":
            mlflow.log_param("epochs", 50)
            mlflow.log_param("lr", 1e-3)
            mlflow.set_tag("backbone", "none")

            model     = AutoEncoder().to(device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            for epoch in range(50):
                model.train()
                epoch_loss = 0.0
                for batch in dataloader:
                    images = batch["image"].to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(images), images)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                avg = epoch_loss / len(dataloader)
                mlflow.log_metric("train_loss", avg, step=epoch)
                print(f"Epoch {epoch+1}/50  loss={avg:.6f}")

            torch.save(model.state_dict(), checkpoint_path)
            mlflow.log_artifact(str(checkpoint_path))

        elif model_name == "PatchCore":
            mlflow.log_param("coreset_ratio", 0.1)
            mlflow.log_param("k", 9)
            mlflow.set_tag("backbone", "resnet50")

            model = PatchCore().to(device)
            model.fit(dataloader)
            print(f"Memory bank shape: {model.memory_bank.shape}")

            torch.save(model.memory_bank, memory_bank_path)
            mlflow.log_artifact(str(memory_bank_path))

        # ── evaluation ────────────────────────────────────────────────────────
        test_dataloader = DataLoader(
            MVTecDataset(root=root, category=category, split="test"),
            num_workers=4,
        )
        auroc = metrics.evaluation(model, test_dataloader, device)
        print(f"AUROC: {auroc:.4f}")
        mlflow.log_metric("auroc", auroc)


@app.local_entrypoint()
def main():
    train.remote("bottle", "PatchCore")