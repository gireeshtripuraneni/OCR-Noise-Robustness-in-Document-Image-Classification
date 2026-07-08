import torch
from PIL import Image

from src.model import DocumentClassifier
from src.dataset_loader import RVLCDIPDataset
from src.transforms import test_transform

# =====================================
# Configuration
# =====================================

MODEL_PATH = "models/document_classifier.pth"

IMAGE_PATH = "data/form/00040534.tif"
###

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================
# Load Dataset (only to get class names)
# =====================================

dataset = RVLCDIPDataset(
    root_dir="data",
    transform=test_transform
)

classes = dataset.classes

# =====================================
# Load Model
# =====================================

model = DocumentClassifier(
    num_classes=len(classes)
).to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

print("Model loaded successfully.")

# =====================================
# Load Image
# =====================================

image = Image.open(IMAGE_PATH).convert("RGB")

image = test_transform(image)

image = image.unsqueeze(0).to(device)

# =====================================
# Prediction
# =====================================

with torch.no_grad():

    outputs = model(image)

    probabilities = torch.softmax(outputs, dim=1)

    confidence, predicted = torch.max(probabilities, 1)

predicted_class = classes[predicted.item()]

print("\nPrediction")
print("-" * 40)
print(f"Document Class : {predicted_class}")
print(f"Confidence     : {confidence.item() * 100:.2f}%")