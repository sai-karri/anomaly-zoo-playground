import torch
import argparse
from pathlib import Path
from src.datasets.mvtec import MVTecDataset
from src.models.image.ae import AutoEncoder
from src.models.image.patchcore import PatchCore
from src.evaluation import metrics
from src.visualization.viz import show_heatmap, show_heatmap_patchcore
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--category', type=str, default='bottle')
parser.add_argument('--model',    type=str, default='AE')
args = parser.parse_args()

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
root   = Path(__file__).parent.parent / 'data' / 'mvtec'

checkpoint       = Path(__file__).parent.parent / 'checkpoints' / f"{args.model}_{args.category}.pt"
memory_bank_path = Path(__file__).parent.parent / 'checkpoints' / f"{args.model}_memory_bank_{args.category}.pt"

test_dataloader = DataLoader(
    MVTecDataset(root=root, category=args.category, split='test')
)

if args.model == 'AE':
    model = AutoEncoder().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))

elif args.model == 'PatchCore':
    model = PatchCore().to(device)
    model.memory_bank = torch.load(memory_bank_path, map_location=device)
    model.memory_bank = model.memory_bank.to(device)  # ← add this

auroc = metrics.evaluation(model, test_dataloader, device)
print(f"AUROC: {auroc:.4f}")

# visualise best defective image
for img in test_dataloader:
    if img['defect_type'][0] != 'good':
        image = img['image'].to(device)
        score, amap = model.predict(image)
        if args.model == 'AE':
            show_heatmap(image=image, reconstruction=model(image), label=img['label'][0])
        else:
            show_heatmap_patchcore(image=image, amap=amap, label=img['label'][0])
        break


