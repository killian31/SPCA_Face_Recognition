
# Sparse Principal Component Analysis (SPCA)

This repository contains a comprehensive implementation of **Sparse Principal Component Analysis (SPCA)**, applied to dimensionality reduction for the **Labeled Faces in the Wild (LFW)** dataset. The project explores the trade-offs between interpretability, accuracy, and computational efficiency using different PCA methods. Additionally, it evaluates these techniques' performance in a facial recognition task using a Support Vector Machine (SVM) classifier.

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Methods](#methods)
4. [Key Findings](#key-findings)
5. [Repository Structure](#repository-structure)
6. [Getting Started](#getting-started)
7. [Results](#results)
8. [Future Directions](#future-directions)
9. [Acknowledgments](#acknowledgments)
10. [Bibliography](#bibliography)

---

## Overview

Dimensionality reduction is essential in machine learning and data analysis, especially for high-dimensional datasets. In this project, we compare the performance of **Principal Component Analysis (PCA)**, **Sparse PCA (SPCA)**, and **MiniBatch Sparse PCA** on the LFW dataset, emphasizing feature interpretability and classification accuracy. Additionally, we propose the application of **Structured Sparse PCA (SSPCA)** for future exploration.

---

## Dataset

We use the **Labeled Faces in the Wild (LFW)** dataset, a benchmark for facial image analysis. Below are the dataset details:

- **Total Samples:** 1,288 grayscale images.
- **Classes:** 7 distinct individuals (Ariel Sharon, Colin Powell, Donald Rumsfeld, George W. Bush, Gerhard Schroeder, Hugo Chavez, Tony Blair).
- **Image Dimensions:** Each image is resized to \(50 	imes 37\) pixels and flattened into 1,850-dimensional feature vectors.
- **Objective:** Reduce dimensionality while retaining meaningful features for classification.

---

## Methods

The project includes the following dimensionality reduction methods:

1. **Principal Component Analysis (PCA):**
   - Captures maximum variance but lacks feature interpretability.

2. **Sparse Principal Component Analysis (SPCA):**
   - Introduces sparsity to improve interpretability, isolating localized facial features.

3. **MiniBatch Sparse PCA:**
   - Optimized for large datasets with improved computational efficiency.

4. **Support Vector Machine (SVM):**
   - Evaluates the reduced feature sets in a facial recognition task.

5. **Structured Sparse PCA (SSPCA):**
   - Proposed for future exploration, incorporating domain-specific structures to improve interpretability.

---

## Key Findings

1. **Dimensionality Reduction:**
   - PCA reduced the dataset from 1,850 to 171 components, retaining 95% of the variance.
   - SPCA and MiniBatch SPCA achieved similar reductions with sparse components.

2. **Interpretability:**
   - SPCA outperformed PCA in isolating localized facial features (e.g., eyes, nose).

3. **Classification Accuracy:**
   - PCA and SPCA achieved accuracies of 0.845 and 0.841, respectively.
   - MiniBatch SPCA achieved the highest accuracy (0.853).

---

## Repository Structure

```
Sparse_PCA/
├── assets/
│   ├── figures/              # Visualization assets
│   ├── scripts/              # Python scripts for analysis
├── data/                     # Dataset
├── results/                  # Output and results (e.g., confusion matrices)
├── main.tex                  # LaTeX report
├── README.md                 # Project documentation
```

---

## Getting Started

### Prerequisites

- Python 3.8 or later
- Required libraries: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn`, `jupyter`.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/theodruilhe/Sparse_PCA.git
   cd Sparse_PCA
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Code

1. Perform Exploratory Data Analysis (EDA):
   ```bash
   python assets/scripts/eda.py
   ```

2. Apply Sparse PCA and evaluate results:
   ```bash
   python assets/scripts/spca.py
   ```

3. Train and evaluate the SVM model:
   ```bash
   python assets/scripts/svm.py
   ```

4. View the final report:
   - Compile `main.tex` using any LaTeX editor (e.g., Overleaf, TeXShop).

---

## Results

1. **Classification Accuracy:**
   - PCA: 0.845
   - SPCA: 0.841
   - MiniBatch SPCA: 0.853

2. **Insights:**
   - PCA components (eigenfaces) captured global patterns but lacked sparsity.
   - SPCA components highlighted localized facial features (e.g., eyes, mouth).

3. **Confusion Matrices:**
   - Detailed confusion matrices for each method are available in the `results/` folder.

---

## Future Directions

1. **Apply Structured Sparse PCA (SSPCA):**
   - Incorporate domain-specific structures to enhance feature interpretability and classification accuracy.

2. **Experiment with PCA Variance Thresholds:**
   - Retain 90\%, 85\%, or 99\% variance to study the impact on dimensionality reduction and classification.

3. **Integrate Sparse Features with Deep Learning:**
   - Combine sparse representations with Convolutional Neural Networks (CNNs) for hybrid approaches.

---

## Acknowledgments

This project is part of the **Master 2 Data Science for Social Sciences (D3S)** program at **Toulouse School of Economics (TSE)**. Theoretical foundations are based on the works of:

1. **Zou et al. (2006):** Sparse Principal Component Analysis.
2. **Jenatton et al. (2010):** Structured Sparse Principal Component Analysis.

---

## Bibliography

- Zou, H., Hastie, T., & Tibshirani, R. (2006). Sparse principal component analysis. *Journal of Computational and Graphical Statistics*, *15*(2), 265-286. [Link](https://doi.org/10.1198/106186006X113430)
- Jenatton, R., Obozinski, G., & Bach, F. (2010). Structured sparse principal component analysis. *Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics*. [Link](http://proceedings.mlr.press/v9/jenatton10a/jenatton10a.pdf)

---

Enjoy working with SPCA on high-dimensional datasets!
