import random
import matplotlib.pyplot as plt
from PIL import Image
from src.dataset_loader import RVLCDIPDataset

dataset = RVLCDIPDataset("data")

print("Dataset Size:", len(dataset))

for _ in range(5):
    idx = random.randint(0, len(dataset)-1)

    image_path, label = dataset.samples[idx]

    img = Image.open(image_path)

    plt.figure(figsize=(6,6))
    plt.imshow(img, cmap="gray")
    plt.title(dataset.classes[label])
    plt.axis("off")
    plt.show()