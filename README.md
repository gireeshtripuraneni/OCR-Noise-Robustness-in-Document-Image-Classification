# 📄 OCR Noise Robustness in Document Image Classification

A deep learning-based document analysis system that classifies scanned document images, extracts text using OCR, and evaluates model robustness under different image degradations through an interactive Streamlit web application.
LINK to Colab notebook: https://colab.research.google.com/drive/1RCJAquO-z9nRWithkbo61uP8PsCvUXO0?usp=sharing

---

## 🚀 Project Overview

This project investigates the robustness of document image classification when document images are affected by different types of noise.

The system combines:

- 📑 Document Classification using ResNet18
- 🔤 Optical Character Recognition (EasyOCR)
- 🌫️ Noise Robustness Analysis
- 📊 Performance Visualization
- 🌐 Interactive Streamlit Dashboard

The goal is to understand how image degradations affect both document classification accuracy and OCR performance.

---

## ✨ Features

- Document image classification (16 document classes)
- OCR text extraction using EasyOCR
- Noise simulation:
  - Gaussian Noise
  - Gaussian Blur
  - Motion Blur
  - Salt & Pepper Noise
  - Rotation (5° and 10°)
- Confidence comparison before and after noise
- Performance visualization
- Robustness report generation
- Interactive Streamlit interface

---

## 🧠 Model

### CNN Backbone
- ResNet18 (PyTorch)

### OCR Engine
- EasyOCR

### Frameworks
- PyTorch
- Torchvision
- Streamlit
- OpenCV
- NumPy
- Pillow

---

## 📂 Dataset

This project uses the **RVL-CDIP (Ryerson Vision Lab Complex Document Information Processing)** dataset.

Dataset Statistics

- 400,000 document images
- 16 document classes
- 320,000 Training Images
- 40,000 Validation Images
- 40,000 Test Images

Document Classes

- Letter
- Form
- Email
- Handwritten
- Advertisement
- Scientific Report
- Scientific Publication
- Specification
- File Folder
- News Article
- Budget
- Invoice
- Presentation
- Questionnaire
- Resume
- Memo

### Dataset Links

Official RVL-CDIP Dataset

https://adamharley.com/rvl-cdip/ :contentReference[oaicite:0]{index=0}

Hugging Face

:contentReference[oaicite:1]{index=1}

Kaggle (Small Version)

:contentReference[oaicite:2]{index=2}

---

## 📁 Project Structure

```
OCR_Robustness_Project/
│
├── app.py
├── src/
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   ├── noise.py
│   ├── dataset_loader.py
│   ├── evaluate_noise.py
│   ├── ocr_predict.py
│   ├── plot_results.py
│   └── robustness_report.py
│
├── results/
│
├── models/
│
└── .gitignore
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/gireeshtripuraneni/OCR-Noise-Robustness-in-Document-Image-Classification.git

cd OCR-Noise-Robustness-in-Document-Image-Classification
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 📊 Workflow

1. Load RVL-CDIP dataset
2. Preprocess document images
3. Train ResNet18 classifier
4. Predict document category
5. Extract text using EasyOCR
6. Apply synthetic noise
7. Compare classification confidence
8. Generate robustness report
9. Visualize performance

---

## 📈 Results

- CNN Model: ResNet18
- Number of Classes: 16
- OCR Engine: EasyOCR
- Best Classification Accuracy: **94.47%**

---

## 📸 Application Modules

- Dashboard
- Document Classification
- OCR
- Noise Analysis
- Performance Dashboard
- Robustness Report

---

## ⚠️ Note

The trained model (`document_classifier.pth`) is **not included** in this repository because it exceeds GitHub's maximum file size limit (100 MB).

Users can train the model using:

```bash
python src/train.py
```

---

## 👨‍💻 Author

**Gireesh Chowdary Tripuraneni**

GitHub:

https://github.com/gireeshtripuraneni

---

## 📚 References

Harley, A. W., Ufkes, A., & Derpanis, K. G.

**Evaluation of Deep Convolutional Nets for Document Image Classification and Retrieval**

ICDAR 2015. :contentReference[oaicite:3]{index=3}

---
