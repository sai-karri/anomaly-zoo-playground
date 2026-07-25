import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from tqdm import tqdm


class PatchCore(nn.Module):
    def __init__(self, backbone_name='dinov2_vits14', k=9, coreset_ratio=0.05):
        super().__init__()
        self.backbone_name = backbone_name
        self.k = k
        self.coreset_ratio = coreset_ratio

        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        for params in self.backbone.parameters():
            params.requires_grad = False

        self.features = {}

        def hook_fn(name):
            def fn(module, input, output):
                self.features[name] = output

            return fn

        self.backbone.layer2.register_forward_hook(hook_fn('layer2'))
        self.backbone.layer3.register_forward_hook(hook_fn('layer3'))

        self.memory_bank = None

    # def coreset_sample(self, patches):
    #
    #     # n_coreset = int(len(patches) * self.coreset_ratio)
    #     # selected = [torch.randint(0, len(patches), (1,)).item()]
    #     #
    #     # pbar = tqdm(total=n_coreset, desc="Coreset sampling")
    #     # # step 2-4 — greedy loop
    #     # while len(selected) < n_coreset:
    #     #     distances = torch.cdist(patches, patches[selected])
    #     #     min_distances = distances.min(dim = 1).values
    #     #     next_idx = min_distances.argmax().item()
    #     #     selected.append(next_idx)
    #     #     pbar.update(1)
    #     # pbar.close()
    #     # return patches[selected]
    #     n_coreset = int(len(patches) * self.coreset_ratio)
    #     indices = torch.randperm(len(patches))[:n_coreset]
    #     return patches[indices]

    def coreset_sample(self, patches):
        n_coreset = int(len(patches) * self.coreset_ratio)
        selected = [torch.randint(0, len(patches), (1,)).item()]

        # track minimum distances for each patch
        min_distances = torch.full((len(patches),), float('inf'), device=patches.device)

        for _ in tqdm(range(n_coreset - 1), desc="Coreset sampling"):
            # only update distances to the LAST selected patch
            # not ALL selected patches — this is the key optimisation
            last = patches[selected[-1]].unsqueeze(0)  # (1, D)
            new_distances = torch.cdist(patches, last).squeeze(1)  # (N,)

            # update minimum distances
            min_distances = torch.minimum(min_distances, new_distances)

            # pick patch furthest from any selected patch
            next_idx = min_distances.argmax().item()
            selected.append(next_idx)

        return patches[selected]


    def fit(self, trainloader):

        self.backbone.eval()
        device = next(self.backbone.parameters()).device
        all_patches = []

        for batch in tqdm(trainloader, desc = "Extracting Features"):
            images = batch['image'].to(device)

            self.backbone(images)

            layer2_features = self.features['layer2']
            layer3_features = self.features['layer3']

            upsampled_layer3 = F.interpolate(
                layer3_features,
                size=layer2_features.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

            combined_layers = torch.cat(
                [layer2_features, upsampled_layer3],
                dim=1,
            )

            combined_layers = combined_layers.permute(0,2,3,1)
            C = combined_layers.shape[-1]
            combined_patches = combined_layers.reshape(-1,C)

            all_patches.append(combined_patches)

        all_patches = torch.cat(all_patches, dim=0)

        self.memory_bank = self.coreset_sample(all_patches)

    def predict(self, image):

        self.backbone.eval()
        device =  next(self.backbone.parameters()).device

        self.backbone(image.to(device))

        layer2_features = self.features['layer2']
        layer3_features = self.features['layer3']

        upsampled_layer3 = F.interpolate(
            layer3_features,
            size=layer2_features.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        combined_layers = torch.cat(
            [layer2_features, upsampled_layer3],
            dim=1,
        )

        combined_layers = combined_layers.permute(0, 2, 3, 1)
        C = combined_layers.shape[-1]
        combined_patches = combined_layers.reshape(-1, C)

        distances = torch.cdist(combined_patches, self.memory_bank)
        topk_distances = distances.topk(self.k, largest=False, dim=1).values

        patch_scores = topk_distances.mean(dim=1)

        H, W = layer2_features.shape[-2:]
        patch_scores = patch_scores.reshape(H, W)

        amap = F.interpolate(
            patch_scores.unsqueeze(0).unsqueeze(0),
            size=(256, 256),
            mode='bilinear',
            align_corners=False
        ).squeeze()

        image_score = patch_scores.max()

        return image_score, amap


    def save(self, path):
        torch.save(self.memory_bank, path)


    def load(self, path, device):
        self.memory_bank = torch.load(path, map_location=device)

