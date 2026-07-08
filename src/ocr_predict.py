import torch
import easyocr
from PIL import Image

from src.model import DocumentClassifier
from src.dataset_loader import RVLCDIPDataset
from src.transforms import test_transform

# =====================================
# Configuration
# =====================================

MODEL_PATH = "models/document_classifier.pth"

# Change this to any image you want to test
IMAGE_PATH = "data/form/00040534.tif"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================
# Load Dataset (Only to get class names)
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

print("Model loaded successfully.\n")

# =====================================
# Predict Document Type
# =====================================

image = Image.open(IMAGE_PATH).convert("RGB")

input_tensor = test_transform(image).unsqueeze(0).to(device)

with torch.no_grad():

    output = model(input_tensor)

    probabilities = torch.softmax(output, dim=1)

    confidence, prediction = torch.max(probabilities, 1)

predicted_class = classes[prediction.item()]

print("========== DOCUMENT CLASSIFICATION ==========")
print(f"Document Class : {predicted_class}")
print(f"Confidence     : {confidence.item()*100:.2f}%")

# =====================================
# OCR
# =====================================

reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

results = reader.readtext(IMAGE_PATH, detail=0)

print("\n========== OCR OUTPUT ==========\n")

if len(results) == 0:
    print("No text detected.")
else:
    for line in results:
        print(line)