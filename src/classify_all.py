import os
import cv2
import torch
import pandas as pd
import numpy as np

from PIL import Image
from tqdm import tqdm

from src.dataset_loader import RVLCDIPDataset
from src.transforms import test_transform
from src.model import DocumentClassifier
from src.noise import NOISE_FUNCTIONS

# =====================================================
# Configuration
# =====================================================

DATASET_PATH = "data"

MODEL_PATH = "models/document_classifier.pth"

OUTPUT_DIR = "results"

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "classification_results.csv"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# -----------------------------------------------------
# None = Full Dataset
#
# Example:
# MAX_IMAGES_PER_CLASS = 50
#
# evaluates only 50 images/class
# -----------------------------------------------------

MAX_IMAGES_PER_CLASS = None

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else
    "cpu"
)

print(f"\nUsing Device : {device}")

# =====================================================
# Load Dataset
# =====================================================

dataset = RVLCDIPDataset(
    root_dir=DATASET_PATH,
    transform=None
)

classes = dataset.classes

print(f"Total Images : {len(dataset.samples)}")
print(f"Classes      : {len(classes)}")

# =====================================================
# Load Model
# =====================================================

model = DocumentClassifier(
    num_classes=len(classes)
).to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("\nModel Loaded Successfully!")

# =====================================================
# Results
# =====================================================

results = []

overall_correct = 0
overall_total = 0

noise_statistics = {}

class_statistics = {}

for noise in NOISE_FUNCTIONS.keys():

    noise_statistics[noise] = {

        "correct": 0,
        "total": 0

    }

for cls in classes:

    class_statistics[cls] = {

        "correct": 0,
        "total": 0

    }

# =====================================================
# Image Counter
# =====================================================

image_counter = {}

for cls in classes:

    image_counter[cls] = 0

# =====================================================
# Start Evaluation
# =====================================================

print("\nStarting Classification...\n")

for img_path, label in tqdm(
    dataset.samples,
    total=len(dataset.samples),
    desc="Evaluating Dataset"
):

    class_name = classes[label]

    if MAX_IMAGES_PER_CLASS is not None:

        if image_counter[class_name] >= MAX_IMAGES_PER_CLASS:

            continue

    image_counter[class_name] += 1

    image = cv2.imread(img_path)

    if image is None:

        continue
    
    # ==========================================
    # Process Every Noise Type
    # ==========================================

    for noise_name, noise_function in NOISE_FUNCTIONS.items():

        # Apply Noise
        noisy_image = noise_function(image.copy())

        # Convert OpenCV -> PIL
        rgb = cv2.cvtColor(
            noisy_image,
            cv2.COLOR_BGR2RGB
        )

        pil_image = Image.fromarray(rgb)

        # Apply Transform
        input_tensor = test_transform(
            pil_image
        )

        input_tensor = input_tensor.unsqueeze(0).to(device)

        # -------------------------------
        # Prediction
        # -------------------------------

        with torch.no_grad():

            outputs = model(input_tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, prediction = torch.max(
                probabilities,
                1
            )

        predicted_label = prediction.item()

        predicted_class = classes[predicted_label]

        confidence = confidence.item() * 100

        # -------------------------------
        # Correct Prediction?
        # -------------------------------

        is_correct = (
            predicted_label == label
        )

        overall_total += 1

        if is_correct:

            overall_correct += 1

        # -------------------------------
        # Noise Statistics
        # -------------------------------

        noise_statistics[noise_name]["total"] += 1

        if is_correct:

            noise_statistics[noise_name]["correct"] += 1

        # -------------------------------
        # Class Statistics
        # -------------------------------

        class_statistics[class_name]["total"] += 1

        if is_correct:

            class_statistics[class_name]["correct"] += 1

        # -------------------------------
        # Store Results
        # -------------------------------

        results.append({

            "Image":

                os.path.basename(img_path),

            "True Class":

                class_name,

            "Noise":

                noise_name,

            "Predicted Class":

                predicted_class,

            "Confidence (%)":

                round(confidence, 2),

            "Correct":

                is_correct

        })

        # -------------------------------
        # Console Output
        # -------------------------------

        print(

            f"{os.path.basename(img_path):30}"

            f" | "

            f"{noise_name:18}"

            f" | "

            f"{predicted_class:25}"

            f" | "

            f"{confidence:6.2f}%"

        )
# =====================================================
# Save CSV
# =====================================================

df = pd.DataFrame(results)

df.to_csv(
    OUTPUT_CSV,
    index=False
)

# =====================================================
# Overall Accuracy
# =====================================================

overall_accuracy = 0.0

if overall_total > 0:
    overall_accuracy = (
        overall_correct / overall_total
    ) * 100

print("\n")
print("=" * 70)
print("OVERALL RESULTS")
print("=" * 70)

print(f"Total Evaluations : {overall_total}")
print(f"Correct Predictions : {overall_correct}")
print(f"Overall Accuracy : {overall_accuracy:.2f}%")

# =====================================================
# Accuracy per Noise
# =====================================================

print("\n")
print("=" * 70)
print("ACCURACY PER NOISE TYPE")
print("=" * 70)

noise_summary = []

for noise_name, stats in noise_statistics.items():

    total = stats["total"]

    correct = stats["correct"]

    accuracy = 0.0

    if total > 0:

        accuracy = (correct / total) * 100

    noise_summary.append({

        "Noise": noise_name,
        "Correct": correct,
        "Total": total,
        "Accuracy (%)": round(accuracy, 2)

    })

    if overall_total % 100 == 0:
        print(f"{overall_total} images processed...")

# =====================================================
# Accuracy per Class
# =====================================================

print("\n")
print("=" * 70)
print("ACCURACY PER DOCUMENT CLASS")
print("=" * 70)

class_summary = []

for class_name, stats in class_statistics.items():

    total = stats["total"]

    correct = stats["correct"]

    accuracy = 0.0

    if total > 0:

        accuracy = (correct / total) * 100

    class_summary.append({

        "Class": class_name,
        "Correct": correct,
        "Total": total,
        "Accuracy (%)": round(accuracy, 2)

    })

    print(
        f"{class_name:25}"
        f"{correct:6}/{total:<6}"
        f"{accuracy:8.2f}%"
    )

# =====================================================
# Save Summary CSVs
# =====================================================

noise_df = pd.DataFrame(noise_summary)

noise_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "noise_summary.csv"
    ),

    index=False

)

class_df = pd.DataFrame(class_summary)

class_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "class_summary.csv"
    ),

    index=False

)

# =====================================================
# Finish
# =====================================================

print("\n")
print("=" * 70)
print("FILES GENERATED")
print("=" * 70)

print(f"Classification Results : {OUTPUT_CSV}")

print(
    "Noise Summary          : "
    + os.path.join(
        OUTPUT_DIR,
        "noise_summary.csv"
    )
)

print(
    "Class Summary          : "
    + os.path.join(
        OUTPUT_DIR,
        "class_summary.csv"
    )
)

print("\nClassification evaluation completed successfully!")