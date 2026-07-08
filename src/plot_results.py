import os
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# Configuration
# ==========================================================

RESULTS_DIR = "results"

CLASSIFICATION_RESULTS = os.path.join(
    RESULTS_DIR,
    "classification_results.csv"
)

OCR_RESULTS = os.path.join(
    RESULTS_DIR,
    "ocr_results.csv"
)

OUTPUT_DIR = os.path.join(
    RESULTS_DIR,
    "plots"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ==========================================================
# Load CSV Files
# ==========================================================

if not os.path.exists(CLASSIFICATION_RESULTS):
    raise FileNotFoundError(
        f"Missing:\n{CLASSIFICATION_RESULTS}"
    )

classification_df = pd.read_csv(
    CLASSIFICATION_RESULTS
)

print(
    f"Loaded classification results ({len(classification_df)} rows)"
)

ocr_df = None

if os.path.exists(OCR_RESULTS):

    ocr_df = pd.read_csv(
        OCR_RESULTS
    )

    print(
        f"Loaded OCR results ({len(ocr_df)} rows)"
    )

else:

    print(
        "OCR results not found. OCR plots will be skipped."
    )

# ==========================================================
# Helper
# ==========================================================

def save_plot(filename):

    plt.tight_layout()

    plt.savefig(

        os.path.join(
            OUTPUT_DIR,
            filename
        ),

        dpi=300

    )

    plt.close()

    print(f"Saved -> {filename}")

# ==========================================================
# Overall Accuracy
# ==========================================================

overall_accuracy = (

    classification_df["Correct"]

    .mean()

    * 100

)

plt.figure(figsize=(5,5))

plt.bar(

    ["Accuracy"],

    [overall_accuracy]

)

plt.ylim(0,100)

plt.ylabel("Accuracy (%)")

plt.title("Overall Classification Accuracy")

save_plot(
    "overall_accuracy.png"
)

# ==========================================================
# Accuracy by Noise Type
# ==========================================================

noise_accuracy = (

    classification_df

    .groupby("Noise")["Correct"]

    .mean()

    * 100

).sort_values(ascending=False)

plt.figure(figsize=(10,6))

plt.bar(

    noise_accuracy.index,

    noise_accuracy.values

)

plt.xticks(rotation=45)

plt.ylabel("Accuracy (%)")

plt.xlabel("Noise Type")

plt.title("Classification Accuracy by Noise Type")

save_plot(
    "accuracy_per_noise.png"
)

# ==========================================================
# Average Confidence by Noise
# ==========================================================

confidence_noise = (

    classification_df

    .groupby("Noise")["Confidence (%)"]

    .mean()

    .sort_values(ascending=False)

)

plt.figure(figsize=(10,6))

plt.bar(

    confidence_noise.index,

    confidence_noise.values

)

plt.xticks(rotation=45)

plt.ylabel("Confidence (%)")

plt.xlabel("Noise Type")

plt.title("Average Confidence by Noise")

save_plot(
    "confidence_per_noise.png"
)

# ==========================================================
# Prediction Distribution
# ==========================================================

prediction_counts = (

    classification_df

    ["Predicted Class"]

    .value_counts()

)

plt.figure(figsize=(12,6))

plt.bar(

    prediction_counts.index,

    prediction_counts.values

)

plt.xticks(rotation=90)

plt.ylabel("Number of Predictions")

plt.xlabel("Predicted Class")

plt.title("Prediction Distribution")

save_plot(
    "prediction_distribution.png"
)

# ==========================================================
# Confidence Histogram
# ==========================================================

plt.figure(figsize=(8,6))

plt.hist(

    classification_df["Confidence (%)"],

    bins=20

)

plt.xlabel("Confidence (%)")

plt.ylabel("Number of Images")

plt.title("Confidence Distribution")

save_plot(
    "confidence_histogram.png"
)

# ==========================================================
# Accuracy by Document Class
# ==========================================================

class_accuracy = (

    classification_df

    .groupby("True Class")["Correct"]

    .mean()

    * 100

).sort_values(ascending=False)

plt.figure(figsize=(12,6))

plt.bar(

    class_accuracy.index,

    class_accuracy.values

)

plt.xticks(rotation=90)

plt.ylabel("Accuracy (%)")

plt.xlabel("Document Class")

plt.title("Classification Accuracy by Document Class")

save_plot(
    "accuracy_per_class.png"
)

# ==========================================================
# OCR Graphs
# ==========================================================

if ocr_df is not None:

    # ------------------------------------------------------

    if "Average Words" in ocr_df.columns:

        plt.figure(figsize=(10,6))

        plt.bar(

            ocr_df["Noise"],

            ocr_df["Average Words"]

        )

        plt.xticks(rotation=45)

        plt.ylabel("Average Words")

        plt.xlabel("Noise Type")

        plt.title("Average OCR Word Count")

        save_plot(
            "ocr_words.png"
        )

    # ------------------------------------------------------

    if "Average Characters" in ocr_df.columns:

        plt.figure(figsize=(10,6))

        plt.bar(

            ocr_df["Noise"],

            ocr_df["Average Characters"]

        )

        plt.xticks(rotation=45)

        plt.ylabel("Average Characters")

        plt.xlabel("Noise Type")

        plt.title("Average OCR Character Count")

        save_plot(
            "ocr_characters.png"
        )

# ==========================================================
# Summary
# ==========================================================

print("\n")
print("=" * 60)
print("PLOT GENERATION COMPLETE")
print("=" * 60)

print(f"Total Classification Samples : {len(classification_df)}")

if ocr_df is not None:

    print(f"Total OCR Samples            : {len(ocr_df)}")

print("\nGenerated Plots:\n")

plots = [

    "overall_accuracy.png",
    "accuracy_per_noise.png",
    "confidence_per_noise.png",
    "prediction_distribution.png",
    "confidence_histogram.png",
    "accuracy_per_class.png",
    "ocr_words.png",
    "ocr_characters.png"

]

for plot in plots:

    path = os.path.join(

        OUTPUT_DIR,

        plot

    )

    if os.path.exists(path):

        print(path)

print("\nAll graphs generated successfully!")