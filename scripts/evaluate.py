import torch
import argparse
from pathlib import Path
from src.datasets.mvtec import MVTecDataset
from src.models.image.ae import AutoEncoder
from torch.utils.data import DataLoader
from src.evaluation import metrics
from viz import show_heatmap

parser = argparse.ArgumentParser()
parser.add_argument('--category', type=str, default='bottle')
args = parser.parse_args()

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
root = Path(__file__).parent.parent / 'data' / 'mvtec'

parser = argparse.ArgumentParser()
parser.add_argument('--category', type=str, default='bottle')
args = parser.parse_args()

checkpoint = Path(__file__).parent.parent / 'checkpoints' / f"ae_{args.category}.pt"

model = AutoEncoder().to(device)

model.load_state_dict(torch.load(checkpoint, map_location=device))

test_dataloader = DataLoader(
    MVTecDataset(root=root, category=args.category, split='test')
)

auroc = metrics.evaluation(model, test_dataloader, device)
print(f"AUROC: {auroc:.4f}")

for img in test_dataloader:
     if img['defect_type']:
        image = img['image'].to(device)
        reconstructed = model(img['image'].to(device))
        show_heatmap(image=image, reconstruction=reconstructed, label=img['label'])
        break

