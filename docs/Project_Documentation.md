# Polyp Segmentation: CNN vs. Transformer

## Project Overview — Deep Learning

**Group:** PolypVision
**Members:** Nour Ounissi, Taha Mersni, Baha Amar
**Course:** Deep Learning
**Instructor:** Dr. Mohammed Salah Al-Radhi

---

# Project Problem

Automatic polyp segmentation aims to identify and delineate polyps in colonoscopy images by predicting a binary segmentation mask at the pixel level.

In this project, we compare two representative approaches:

* **PraNet** — a CNN-based polyp segmentation model.
* **Polyp-PVT** — a Transformer-based polyp segmentation model.

The project first focuses on reproducing published results and then extends the comparison through controlled experiments.

PraNet and Polyp-PVT are related works in the same research line and are evaluated within the established polyp-segmentation benchmark setting, using Kvasir-SEG and CVC-ClinicDB for training and multiple public datasets for testing.

---

# Project Idea

This project compares two deep learning approaches for automatic polyp segmentation:

* **PraNet**, based on convolutional neural networks (CNNs).
* **Polyp-PVT**, based on Transformer architecture.

The goal is to understand how these approaches differ in practice, not only in segmentation performance, but also in their behavior under different data conditions and evaluation settings.

---

# Research Question

**How do a CNN-based model (PraNet) and a Transformer-based model (Polyp-PVT) compare in segmentation accuracy, generalization to data from different sources, robustness when limited training data is available, and sensitivity to evaluation protocols?**

---

# Models and Data

* **PraNet:** A CNN-based polyp segmentation model that uses reverse attention mechanisms to refine segmentation predictions and boundaries.

* **Polyp-PVT:** A Transformer-based polyp segmentation model built using Pyramid Vision Transformer architecture.

* **Training Data:** Kvasir-SEG and CVC-ClinicDB.

* **Test Data:** Five public benchmark datasets:

  * CVC-300
  * CVC-ClinicDB
  * CVC-ColonDB
  * ETIS-LaribPolypDB
  * Kvasir

* **Evaluation Metrics:**

  * Mean Dice coefficient
  * Mean Intersection over Union (IoU)

---

# Approach

The project is organized into three main stages:

### 1. Reproduce

Run the official implementations and pretrained models of PraNet and Polyp-PVT to reproduce their published results.

### 2. Improve

Retrain Polyp-PVT in Google Colab using a proper training, validation, and test protocol.

Selected experiments will investigate:

* Different loss functions
* Data augmentation strategies
* Input image sizes
* Training and validation behavior

### 3. Contribute

Extend the analysis beyond reproduction by investigating:

* Generalization to other datasets
* Training with fewer labels
* Failure cases
* Sensitivity to evaluation protocols and metrics

---

# Expected Outcome

The project aims to produce a **fair and well-documented comparative study** of PraNet and Polyp-PVT.

The final project will include:

* A reproducible GitHub repository
* Google Colab notebooks
* Experimental results and visual comparisons
* A final report
* Presentation slides

