import torch
import matplotlib.pyplot as plt

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)


def denormalize(tensor):
    return tensor * std + mean


def show_heatmap(image, reconstruction, label, save_path=None):

    image = denormalize(image).detach().cpu().squeeze(0)
    reconstruction = denormalize(reconstruction).detach().cpu().squeeze(0)

    # Squared reconstruction error for each channel
    anomaly_map = (image - reconstruction) ** 2

    # Combine the three RGB channel errors into one heatmap
    anomaly_map = anomaly_map.sum(dim=0)

    # Convert from (C, H, W) to (H, W, C)
    image = image.permute(1, 2, 0).clamp(0, 1)
    reconstruction = reconstruction.permute(1, 2, 0).clamp(0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image)
    axes[0].set_title("Original")

    axes[1].imshow(anomaly_map, cmap="hot")
    axes[1].set_title(f"Anomaly Map — Label: {label.item()}")

    axes[2].imshow(image)
    axes[2].imshow(anomaly_map, cmap='hot', alpha=0.5)
    axes[2].set_title("Overlay")

    for axis in axes:
        axis.axis("off")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()
    plt.close(fig)