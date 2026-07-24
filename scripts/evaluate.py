import torch
import argparse
from pathlib import Path
from src.datasets.mvtec import MVTecDataset
from src.models.image.ae import AutoEncoder
from src.models.image.patchcore import PatchCore
from torch.utils.data import DataLoader
from src.evaluation import metrics
from viz import show_heatmap, show_heatmap_patchcore

parser = argparse.ArgumentParser()
parser.add_argument('--category', type=str, default='bottle')
parser.add_argument('--model', type=str, default='AE')
args = parser.parse_args()

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
root = Path(__file__).parent.parent / 'data' / 'mvtec'

checkpoint = Path(__file__).parent.parent / 'checkpoints' / f"{args.model}_{args.category}.pt"
memory_bank_path = Path(__file__).parent.parent / 'checkpoints' / f"{args.model}_memory_bank.pt"

if args.model == 'AE':

    model = AutoEncoder().to(device)

    model.load_state_dict(torch.load(checkpoint, map_location=device))

    test_dataloader = DataLoader(
        MVTecDataset(root=root, category=args.category, split='test')
    )

    auroc = metrics.evaluation(model, test_dataloader, device)
    print(f"AUROC: {auroc:.4f}")

elif args.model == 'PatchCore':

    model = PatchCore().to(device)

    model.memory_bank = torch.load(memory_bank_path, map_location=device)

    model.load_state_dict(torch.load(checkpoint, map_location=device))

    test_dataloader = DataLoader(
        MVTecDataset(root=root, category=args.category, split='test')
    )

    auroc = metrics.evaluation(model, test_dataloader, device)
    print(f"AUROC: {auroc:.4f}")



for img in test_dataloader:
     if img['defect_type']:
        image = img['image'].to(device)
        img_score, amap = model.predict(img['image'].to(device))
        show_heatmap_patchcore(image=image, amap= amap, label=img['label'])
        break

