import os
import torch
import pandas as pd
from PIL import Image

from src.model import DocumentClassifier
from src.dataset_loader import RVLCDIPDataset
from src.transforms import test_transform

# ==========================================
# Configuration
# ==========================================

MODEL_PATH = "models/document_classifier.pth"
NOISY_FOLDER = "results/noisy"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# Load Dataset (to get class names)
# ==========================================

dataset = RVLCDIPDataset(
    root_dir="data",
    transform=test_transform
)

classes = dataset.classes

# ==========================================
# Load Model
# ==========================================

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

# ==========================================
# Evaluate Images
# ==========================================

results = []

image_extensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

for filename in sorted(os.listdir(NOISY_FOLDER)):

    if not filename.lower().endswith(image_extensions):
        continue

    image_path = os.path.join(NOISY_FOLDER, filename)

    image = Image.open(image_path).convert("RGB")

    image = test_transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, prediction = torch.max(probabilities, 1)

    predicted_class = classes[prediction.item()]
    confidence = confidence.item() * 100

    print("----------------------------------------")
    print(f"Image      : {filename}")
    print(f"Prediction : {predicted_class}")
    print(f"Confidence : {confidence:.2f}%")

    results.append({
        "Image": filename,
        "Predicted Class": predicted_class,
        "Confidence (%)": round(confidence, 2)
    })

# ==========================================
# Save Results
# ==========================================

results_df = pd.DataFrame(results)

output_csv = os.path.join(NOISY_FOLDER, "evaluation_results.csv")

results_df.to_csv(output_csv, index=False)

print("\n========================================")
print("Evaluation completed successfully!")
print(f"Results saved to: {output_csv}")