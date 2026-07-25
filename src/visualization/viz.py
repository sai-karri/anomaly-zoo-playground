import torch
import matplotlib.pyplot as plt

mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denormalize(tensor):
    return tensor.cpu() * std + mean


def show_heatmap(image, reconstruction, label, save_path=None):
    image         = denormalize(image).detach().squeeze(0)
    reconstruction = denormalize(reconstruction).detach().squeeze(0)

    anomaly_map = ((image - reconstruction) ** 2).sum(dim=0)

    image         = image.permute(1, 2, 0).clamp(0, 1)
    reconstruction = reconstruction.permute(1, 2, 0).clamp(0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image);          axes[0].set_title("Original")
    axes[1].imshow(anomaly_map, cmap="hot"); axes[1].set_title(f"Anomaly map — label: {label.item()}")
    axes[2].imshow(image);          axes[2].imshow(anomaly_map, cmap='hot', alpha=0.5); axes[2].set_title("Overlay")
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    if save_path: plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def show_heatmap_patchcore(image, amap, label, save_path=None):
    image = denormalize(image).detach().squeeze(0).permute(1, 2, 0).clamp(0, 1)
    amap  = amap.detach().cpu()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image);   axes[0].set_title("Original")
    axes[1].imshow(image);   axes[1].imshow(amap, cmap='hot', alpha=0.5)
    axes[1].set_title(f"Anomaly overlay — label: {label.item()}")
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    if save_path: plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close(fig)