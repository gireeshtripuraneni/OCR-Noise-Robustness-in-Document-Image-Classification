# OCR evaluation for the complete dataset

import os
import cv2
import easyocr
import pandas as pd
from tqdm import tqdm

from src.dataset_loader import RVLCDIPDataset
from src.noise import NOISE_FUNCTIONS

# =====================================================
# Configuration
# =====================================================

DATASET_PATH = "data"

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "ocr_results.csv"
)

SUMMARY_CSV = os.path.join(
    OUTPUT_DIR,
    "ocr_summary.csv"
)

# None = Entire Dataset
MAX_IMAGES_PER_CLASS = 20

# =====================================================
# Load Dataset
# =====================================================

dataset = RVLCDIPDataset(
    root_dir=DATASET_PATH,
    transform=None
)

classes = dataset.classes

reader = easyocr.Reader(
    ["en"],
    gpu=False
)

# =====================================================
# Select Images
# =====================================================

selected_samples = []

image_counter = {
    c: 0
    for c in classes
}

for img_path, label in dataset.samples:

    cls = classes[label]

    if (
        MAX_IMAGES_PER_CLASS is None
        or image_counter[cls] < MAX_IMAGES_PER_CLASS
    ):

        selected_samples.append(
            (img_path, label)
        )

        image_counter[cls] += 1

    if (
        MAX_IMAGES_PER_CLASS is not None
        and all(
            count >= MAX_IMAGES_PER_CLASS
            for count in image_counter.values()
        )
    ):
        break

print(f"\nSelected Images : {len(selected_samples)}")

# =====================================================
# Statistics
# =====================================================

noise_stats = {

    noise: {

        "images": 0,
        "characters": 0,
        "words": 0

    }

    for noise in NOISE_FUNCTIONS

}

rows = []

# =====================================================
# OCR Evaluation
# =====================================================

for img_path, label in tqdm(

    selected_samples,

    total=len(selected_samples),

    desc="OCR Evaluation"

):

    cls = classes[label]

    img = cv2.imread(img_path)

    if img is None:
        continue

    for noise_name, noise_fn in NOISE_FUNCTIONS.items():

        noisy = noise_fn(img.copy())

        rgb = cv2.cvtColor(
            noisy,
            cv2.COLOR_BGR2RGB
        )

        text_list = reader.readtext(
            rgb,
            detail=0
        )

        text = " ".join(text_list)

        chars = len(text)

        words = len(text.split())

        noise_stats[noise_name]["images"] += 1
        noise_stats[noise_name]["characters"] += chars
        noise_stats[noise_name]["words"] += words

        rows.append({

            "Image": os.path.basename(img_path),

            "True Class": cls,

            "Noise": noise_name,

            "OCR Text": text,

            "Characters": chars,

            "Words": words

        })

# =====================================================
# Save OCR Results
# =====================================================

ocr_df = pd.DataFrame(rows)

ocr_df.to_csv(

    OUTPUT_CSV,

    index=False

)

# =====================================================
# Build Summary
# =====================================================

summary = []

for noise, stats in noise_stats.items():

    count = max(

        stats["images"],

        1

    )

    summary.append({

        "Noise": noise,

        "Images": stats["images"],

        "Average Characters": round(

            stats["characters"] / count,

            2

        ),

        "Average Words": round(

            stats["words"] / count,

            2

        )

    })

summary_df = pd.DataFrame(summary)

summary_df.to_csv(

    SUMMARY_CSV,

    index=False

)

# =====================================================
# Finish
# =====================================================

print("\n===================================")
print("OCR Evaluation Completed")
print("===================================")

print(f"Images Evaluated : {len(selected_samples)}")

print(f"OCR Results      : {OUTPUT_CSV}")

print(f"OCR Summary      : {SUMMARY_CSV}")