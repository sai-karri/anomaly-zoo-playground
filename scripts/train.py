import torch
import torch.nn as nn
from pathlib import Path
from src.datasets.mvtec import MVTecDataset
from src.models.ae import AutoEncoder
from torch.utils.data import DataLoader


device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
root = Path(__file__).parent.parent / 'data' / 'mvtec'

dataloader = DataLoader(
    MVTecDataset(root=root, category='bottle', split='train'),
    batch_size=8,
    shuffle=True
    )

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

    print(
        f"Epoch {epoch + 1}/{epochs}, "
        f"Loss: {average_loss:.6f}"
    )