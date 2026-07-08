import os
from PIL import Image
from torch.utils.data import Dataset

class RVLCDIPDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        self.class_to_idx = {
            cls: idx
            for idx, cls in enumerate(self.classes)
        }

        self.samples = []

        for cls in self.classes:

            folder = os.path.join(root_dir, cls)

            for file in os.listdir(folder):

                if file.lower().endswith((".tif", ".tiff")):

                    self.samples.append(
                        (
                            os.path.join(folder, file),
                            self.class_to_idx[cls]
                        )
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label, img_path