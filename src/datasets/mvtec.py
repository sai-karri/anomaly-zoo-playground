from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset


class MVTecDataset(Dataset):

    def __init__(self, root, category, split, img_size=256, verbose=False):
        self.root = root
        self.category = category
        self.split = split
        self.img_size = img_size

        split_dir = Path(self.root) / self.category / self.split

        if not split_dir.exists():
            raise FileNotFoundError(f"Directory does not exist: {split_dir}")

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        self.img_paths = []
        self.labels = []
        self.defect_types = []

        valid_extensions = {".png", ".jpg", ".jpeg", ".bmp"}

        for defect_folder in sorted(split_dir.iterdir()):

            if not defect_folder:
                continue

            if verbose:
                print(f"\nDefect folder: {defect_folder.name}")

            image_paths = sorted(
                path
                for path in defect_folder.iterdir()
                if path.is_file() and path.suffix.lower() in valid_extensions
            )

            if not image_paths:
                print("  No images found")
                continue

            for image_path in image_paths:

                if verbose:
                    print(f"{image_path.name}")

                self.img_paths.append(image_path)
                self.defect_types.append(defect_folder.name)

                label = 0 if defect_folder.name == 'good' else 1
                self.labels.append(label)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image_path = self.img_paths[idx]

        parent_folder = image_path.parent.name
        file_name = image_path.stem
        suffix = image_path.suffix

        mask_path = Path(self.root) / self.category / 'ground_truth' / parent_folder / f"{file_name}_mask{suffix}"

        with Image.open(image_path) as image:
            image = image.convert('RGB')
            image = self.transform(image)

        if mask_path.is_file():
            with Image.open(mask_path) as mask:
                mask = mask.convert('L')
                mask = self.mask_transform(mask)
                mask = (mask > 0.5).float()
        else:
            mask = torch.zeros((1,self.img_size, self.img_size))

        return {
            "image": image,
            "label": self.labels[idx],
            "defect_type": self.defect_types[idx],
            "path": str(image_path),
            "mask_image": mask
        }


# if __name__ == '__main__':
#
#     root = Path.cwd().parent.parent / 'data' / 'mvtec'
#
#     # dataset = MVTecDataset(
#     #     root=root,
#     #     category="bottle",
#     #     split="train"
#     # )
#     # print(f"Total images: {len(dataset)}")
#     #
#     # sample = dataset[0]
#     # print(f"Image shape: {sample['image'].shape}")
#     # print(f"Mask shape: {sample['mask_image'].shape}")
#     # print(f"Label: {sample['label']}")
#
#     test_dataset = MVTecDataset(
#         root=root,
#         category="bottle",
#         split="test"
#     )
#     print(f"Total images: {len(test_dataset)}")
#
#     # grab a defective sample
#     for i in range(len(test_dataset)):
#         sample = test_dataset[i]
#         if sample['label'] == 1:
#             print(f"Defective sample found")
#             print(f"Defect type: {sample['defect_type']}")
#             print(f"Mask sum: {sample['mask_image'].sum()}")
#             break