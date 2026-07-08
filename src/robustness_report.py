import os
import pandas as pd

# ==========================================================
# Configuration
# ==========================================================

RESULTS_DIR = "results"

CLASSIFICATION_CSV = os.path.join(
    RESULTS_DIR,
    "classification_results.csv"
)

OCR_SUMMARY_CSV = os.path.join(
    RESULTS_DIR,
    "ocr_summary.csv"
)

OUTPUT_REPORT = os.path.join(
    RESULTS_DIR,
    "robustness_report.txt"
)

OUTPUT_SUMMARY = os.path.join(
    RESULTS_DIR,
    "robustness_summary.csv"
)

# ==========================================================
# Check Files
# ==========================================================

if not os.path.exists(CLASSIFICATION_CSV):
    raise FileNotFoundError(CLASSIFICATION_CSV)

if not os.path.exists(OCR_SUMMARY_CSV):
    raise FileNotFoundError(OCR_SUMMARY_CSV)

classification_df = pd.read_csv(CLASSIFICATION_CSV)
ocr_df = pd.read_csv(OCR_SUMMARY_CSV)

print("CSV files loaded successfully.")

# ==========================================================
# Overall Statistics
# ==========================================================

overall_accuracy = classification_df["Correct"].mean() * 100

overall_confidence = classification_df["Confidence (%)"].mean()

# ==========================================================
# Classification Accuracy by Noise
# ==========================================================

noise_accuracy = (

    classification_df
    .groupby("Noise")["Correct"]
    .mean()
    .mul(100)
    .round(2)
    .reset_index()

)

noise_accuracy.columns = [

    "Noise",

    "Classification Accuracy (%)"

]

# ==========================================================
# Merge with OCR Statistics
# ==========================================================

summary = pd.merge(

    noise_accuracy,

    ocr_df,

    on="Noise",

    how="left"

)

summary.to_csv(

    OUTPUT_SUMMARY,

    index=False

)

# ==========================================================
# Best and Worst Noise
# ==========================================================

best_noise = summary.loc[
    summary["Classification Accuracy (%)"].idxmax()
]

worst_noise = summary.loc[
    summary["Classification Accuracy (%)"].idxmin()
]

# ==========================================================
# Write Report
# ==========================================================

with open(

    OUTPUT_REPORT,

    "w",

    encoding="utf-8"

) as f:

    f.write("=" * 60 + "\n")
    f.write("OCR ROBUSTNESS EVALUATION REPORT\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Total Evaluations : {len(classification_df)}\n")
    f.write(f"Overall Accuracy : {overall_accuracy:.2f}%\n")
    f.write(f"Average Confidence : {overall_confidence:.2f}%\n\n")

    f.write("=" * 60 + "\n")
    f.write("NOISE PERFORMANCE\n")
    f.write("=" * 60 + "\n\n")

    for _, row in summary.iterrows():

        f.write(f"Noise : {row['Noise']}\n")
        f.write(
            f"Classification Accuracy : "
            f"{row['Classification Accuracy (%)']:.2f}%\n"
        )

        if "Average Words" in row.index:
            f.write(
                f"Average OCR Words : "
                f"{row['Average Words']:.2f}\n"
            )

        if "Average Characters" in row.index:
            f.write(
                f"Average OCR Characters : "
                f"{row['Average Characters']:.2f}\n"
            )

        f.write("\n")

    f.write("=" * 60 + "\n")
    f.write("BEST PERFORMING NOISE\n")
    f.write("=" * 60 + "\n")

    f.write(
        f"{best_noise['Noise']} "
        f"({best_noise['Classification Accuracy (%)']:.2f}%)\n\n"
    )

    f.write("=" * 60 + "\n")
    f.write("WORST PERFORMING NOISE\n")
    f.write("=" * 60 + "\n")

    f.write(
        f"{worst_noise['Noise']} "
        f"({worst_noise['Classification Accuracy (%)']:.2f}%)\n"
    )

print("\n")
print("=" * 60)
print("ROBUSTNESS REPORT GENERATED")
print("=" * 60)

print(f"Summary CSV : {OUTPUT_SUMMARY}")
print(f"Report      : {OUTPUT_REPORT}")