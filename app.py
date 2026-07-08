import os
import time
from src.noise import NOISE_FUNCTIONS

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import easyocr
import numpy as np

from torchvision import models
from torchvision import transforms

from src.model import DocumentClassifier
from src.dataset_loader import RVLCDIPDataset
from src.transforms import test_transform

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Intelligent Document Analysis System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# PATHS
# ==========================================================

DATASET_PATH = "data"
MODEL_PATH = "models/document_classifier.pth"
RESULTS_PATH = "results"

# ==========================================================
# CHECK PATHS
# ==========================================================

if not os.path.exists(DATASET_PATH):
    st.error(f"❌ Dataset not found: {DATASET_PATH}")
    st.stop()

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model not found: {MODEL_PATH}")
    st.stop()

# ==========================================================
# DEVICE
# ==========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ==========================================================
# LOAD DATASET
# ==========================================================

@st.cache_resource
def load_dataset():

    dataset = RVLCDIPDataset(
        root_dir=DATASET_PATH,
        transform=test_transform
    )

    return dataset

dataset = load_dataset()

classes = dataset.classes


# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():

    model = DocumentClassifier(
        num_classes=len(classes)
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)

    model.eval()

    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model:\n{e}")
    st.stop()

    
# ==========================================================
# LOAD OCR
# ==========================================================

@st.cache_resource
def load_ocr():

    reader = easyocr.Reader(
        ['en'],
        gpu=torch.cuda.is_available()
    )

    return reader

#reader = load_ocr()

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("📄 OCR Robustness")

page = st.sidebar.radio(

    "Navigation",

    [

        "Dashboard",

        "Classification",

        "OCR",

        "Noise Analysis",

        "Performance",

        "Reports"

    ]

)

st.sidebar.divider()

st.sidebar.success("✅ Model Loaded")

st.sidebar.success(f"✅ {len(classes)} Classes")

st.sidebar.success("🔤 OCR Module Ready")

st.sidebar.info(f"💻 Device : {DEVICE}")


# ----------------------------------------------------
# Dashboard
# ----------------------------------------------------

if page == "Dashboard":

    st.title("📄 Intelligent Document Analysis System")

    st.caption(
        "Deep Learning-based Document Classification & OCR Robustness Evaluation"
    )

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("🧠 CNN Model", "ResNet18")

    with c2:
        st.metric("📂 Classes", len(classes))

    with c3:
        st.metric("🔤 OCR Engine", "EasyOCR")

    with c4:
        st.metric("🎯 Best Accuracy", "94.47%")

    st.divider()

    left, right = st.columns([2, 1])

    with left:

        uploaded = st.file_uploader(
            "Upload a Document",
            type=["png", "jpg", "jpeg", "tif", "tiff"]
        )

        if uploaded:

            image = Image.open(uploaded).convert("RGB")

            st.image(
                image,
                caption="Uploaded Document",
                width=500
            )

        else:

            st.info("Upload a document to begin.")

    with right:

        st.success("✅ Model Loaded")

        st.success("✅ Dataset Ready")

        st.success("✅ OCR Ready")

        st.success("✅ Noise Module Ready")

        st.info(f"Running on: **{DEVICE.type.upper()}**")
    

# ----------------------------------------------------
# Classification
# ----------------------------------------------------

elif page == "Classification":

    st.title("📄 Document Classification")

    uploaded = st.file_uploader(
        "Upload a document",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        key="classification"
    )

    if uploaded is not None:

        image = Image.open(uploaded).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("🔍 Predict Document"):

            start_time = time.time()

            # Transform image
            input_tensor = test_transform(image).unsqueeze(0).to(DEVICE)

            # Prediction
            with torch.no_grad():

                outputs = model(input_tensor)

                probabilities = torch.softmax(outputs, dim=1)

                confidence, predicted = torch.max(probabilities, 1)

            prediction = classes[predicted.item()]

            elapsed = (time.time() - start_time) * 1000

            with col2:

                st.success("Prediction Complete")

                st.metric(
                    "Predicted Class",
                    prediction
                )

                st.metric(
                    "Confidence",
                    f"{confidence.item()*100:.2f}%"
                )

                st.metric(
                    "Inference Time",
                    f"{elapsed:.2f} ms"
                )

    else:

        st.info("Upload an image to classify.")
    
# ----------------------------------------------------
# OCR
# ----------------------------------------------------

elif page == "OCR":

    st.title("🔤 Optical Character Recognition")

    uploaded = st.file_uploader(
        "Upload Document",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        key="ocr"
    )

    if uploaded is not None:

        image = Image.open(uploaded).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                image,
                caption="Uploaded Document",
                use_container_width=True
            )

        if st.button("Extract Text"):
            
            reader = load_ocr()   # ← ADD THIS LINE

            with st.spinner("Running EasyOCR..."):

                start = time.time()

                img_np = np.array(image)

                results = reader.readtext(img_np)

                elapsed = (time.time() - start) * 1000

                extracted_text = ""

                for item in results:

                    extracted_text += item[1] + "\n"

            with col2:

                st.success("OCR Complete")

                st.metric(
                    "Words Detected",
                    len(extracted_text.split())
                )

                st.metric(
                    "Characters",
                    len(extracted_text)
                )

                st.metric(
                    "Processing Time",
                    f"{elapsed:.2f} ms"
                )

            st.subheader("Extracted Text")

            st.text_area(
                "",
                extracted_text,
                height=350
            )

    else:

        st.info("Upload a document to perform OCR.")
    

# ----------------------------------------------------
# Noise
# ----------------------------------------------------

elif page == "Noise Analysis":

    st.title("🌫️ Noise Robustness Analysis")

    uploaded = st.file_uploader(
        "Upload Document",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        key="noise"
    )

    if uploaded is not None:

        image = Image.open(uploaded).convert("RGB")
        original = np.array(image)

        noise_name = st.selectbox(
            "Select Noise Type",
            list(NOISE_FUNCTIONS.keys())
        )

        noisy = NOISE_FUNCTIONS[noise_name](original.copy())

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                original,
                caption="Original Image",
                use_container_width=True
            )

        with col2:
            st.image(
                noisy,
                caption=noise_name.replace("_", " ").title(),
                use_container_width=True
            )

        if st.button("🚀 Compare Predictions"):

            def predict(img):

                pil = Image.fromarray(img)

                tensor = test_transform(pil).unsqueeze(0).to(DEVICE)

                with torch.no_grad():

                    outputs = model(tensor)

                    probabilities = torch.softmax(outputs, dim=1)

                    confidence, predicted = torch.max(probabilities, 1)

                return (
                    classes[predicted.item()],
                    confidence.item() * 100
                )

            original_class, original_conf = predict(original)

            noisy_class, noisy_conf = predict(noisy)

            c1, c2 = st.columns(2)

            with c1:

                st.success("Original Prediction")

                st.metric(
                    "Class",
                    original_class
                )

                st.metric(
                    "Confidence",
                    f"{original_conf:.2f}%"
                )

            with c2:

                st.warning("Noisy Prediction")

                st.metric(
                    "Class",
                    noisy_class
                )

                st.metric(
                    "Confidence",
                    f"{noisy_conf:.2f}%"
                )

            st.metric(
                "Confidence Drop",
                f"{original_conf - noisy_conf:.2f}%"
            )
            if st.button("🔤 Compare OCR"):
                reader = load_ocr()

                original_text = reader.readtext(original)

                noisy_text = reader.readtext(noisy)
                
                original_text = "\n".join([item[1] for item in original_text])
                noisy_text = "\n".join([item[1] for item in noisy_text])
                
                original_words = len(original_text.split())

                noisy_words = len(noisy_text.split())

                original_chars = len(original_text)

                noisy_chars = len(noisy_text)
                
                st.subheader("OCR Statistics")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric(
                        "Original Words",
                        original_words
                    )
                with c2: 
                    st.metric(
                        "Noisy Words",
                        noisy_words
                    )
                
                with c3:
                    st.metric(
                        "Original Characters",
                        original_chars
                    )
                with c4:
                    st.metric(
                        "Noisy Characters",
                        noisy_chars
                    )
                
                
                
                st.subheader("OCR Comparison")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### Original OCR")

                    st.text_area(
                        "Original Text",
                        original_text,
                        height=300
                    )

                with col2:
                    st.markdown("### Noisy OCR")
                    st.text_area(
                        "Noisy Text",
                        noisy_text,
                        height=300
       )

    else:

        st.info("Upload a document to analyse noise robustness.")

   

# ----------------------------------------------------
# Performance
# ----------------------------------------------------

elif page == "Performance":

    st.title("📊 Performance Dashboard")

    st.markdown("### Generated Performance Charts")

    plot_dir = os.path.join(RESULTS_PATH, "plots")

    if os.path.exists(plot_dir):

        images = sorted(

            [

                f for f in os.listdir(plot_dir)

                if f.endswith(".png")

            ]

        )

        for img in images:
            st.subheader(
                img.replace("_", " ").replace(".png", "").title()
            )

            st.image(
                os.path.join(plot_dir, img),
                use_container_width=True
            )

            st.divider()

    else:

        st.warning("No plots found.")

    
# ----------------------------------------------------
# Reports
# ----------------------------------------------------

elif page == "Reports":

    st.title("📄 Robustness Report")

    report_path = os.path.join(

        RESULTS_PATH,

        "robustness_report.txt"

    )

    if os.path.exists(report_path):

        with open(

            report_path,

            "r",

            encoding="utf-8"

        ) as f:

            report = f.read()

        st.text_area(
            "Robustness Report",
            report,
            height=500
        )

        st.download_button(

            "⬇ Download Report",

            report,

            file_name="robustness_report.txt"

        )

    else:

        st.warning("Report not found.")
    