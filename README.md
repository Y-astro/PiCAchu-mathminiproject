# Principal Component Analysis for Feature Reduction in Machine Learning: A Linear Algebra Approach

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: Pytest Passing](https://img.shields.io/badge/tests-25%20passed-brightgreen.svg)](pca_project/test_pca.py)

A mathematically rigorous, from-scratch implementation and empirical study of **Principal Component Analysis (PCA)** using linear algebra first principles (sample covariance, orthogonal diagonalization, quadratic forms, and SVD). Validated against `scikit-learn` across 25 automated unit tests to machine precision, with experiments evaluated across **3 carefully chosen datasets**.

---

## 📖 Quick Links & Documentation
- 📘 **[Comprehensive Project & Viva Voce Guide](PROJECT_GUIDE_AND_EXPLANATION.md)** — Master reference explaining all the linear algebra in plain English, bridging theory to code, the 4-point manual calculation, and an exhaustive 15-question viva playbook.
- 📓 **[Jupyter Notebook (`pca_project.ipynb`)](pca_project/pca_project.ipynb)** — Self-contained, runnable notebook containing 16 clean sections and core empirical experiments.
- 🐍 **[From-Scratch Python Engine (`pca_scratch.py`)](pca_project/pca_scratch.py)** — Pure NumPy implementation featuring both covariance eigendecomposition (`np.linalg.eigh`) and direct SVD (`np.linalg.svd`).
- 🧪 **[Validation Test Suite (`test_pca.py`)](pca_project/test_pca.py)** — 25 automated pytest tests validating numerical equivalence against Scikit-Learn.
- ☁️ **[Kaggle Cloud Running Instructions](pca_project/KAGGLE_INSTRUCTIONS.md)** — Step-by-step guide to run the notebook in the cloud on Kaggle without local hardware limitations.

---

## 📁 Repository Structure
```
├── README.md                         # Main repository landing page
├── PROJECT_GUIDE_AND_EXPLANATION.md # Comprehensive math theory & viva defense guide
├── PCA_mini_project_plan.md         # Original master specification & rubric
├── .gitignore                       # Clean Git tracking filters
└── pca_project/
    ├── pca_scratch.py               # Pure NumPy PCA implementation (eigh + SVD)
    ├── test_pca.py                  # 25 automated mathematical unit tests
    ├── pca_project.ipynb            # Self-contained Jupyter notebook (16 sections)
    ├── generate_notebook.py         # Programmatic notebook builder script (nbformat)
    ├── KAGGLE_INSTRUCTIONS.md       # Kaggle cloud setup & execution instructions
    ├── figures/                     # High-resolution (300 DPI) generated figures
    └── results/                     # Raw experimental benchmark CSV tables
```

---

## 🎯 The 3 Datasets Used

To keep the project concise, fast, and easy to explain during evaluations, we focus on **3 datasets**:

| # | Dataset | Dimensions | Purpose in Project |
|---|---|---|---|
| **1** | **Toy 2D Data (4 points)** | $2\text{D} \to 1\text{D}$ | **Pencil-and-paper verification:** Traces the math step-by-step on paper to prove formulas match code. |
| **2** | **MNIST Handwritten Digits** | $784\text{D} \to 154\text{D}$ | **Primary Machine Learning Benchmark:** Large-scale image compression, eigendigits, denoising, and classifier speedup. |
| **3** | **Wine Dataset** | $13\text{D}$ (Tabular) | **Standardization Study:** Proves why features with different scales (Proline vs. Alcohol) require scaling before PCA. |

---

## ⚡ Core Concept: The Steps of PCA

```
1. Mean Centering:        B = A - x̄ 1ᵀ
2. Covariance Matrix:     S = (1 / (N - 1)) * B * Bᵀ
3. Eigendecomposition:    S * q = λ * q  (or SVD: B = U * Σ * Vᵀ)
4. Sort Eigenvalues:      λ₁ ≥ λ₂ ≥ ... ≥ λ_P
5. Select Top-k:          Q_k = [q₁, q₂, ..., q_k]
6. Project & Compress:    Z = Q_kᵀ * B
```

---

## 🧪 Automated Testing
To run the automated validation test suite:
```bash
cd pca_project
pytest test_pca.py -v
```
All **25 tests pass**, confirming:
- Eigenvalues and explained variance ratios match Scikit-Learn to $10^{-8}$ precision.
- Directional invariance ($|\mathbf{q}_{\text{scratch}} \cdot \mathbf{q}_{\text{sklearn}}| = 1.0$).
- Strict orthonormality ($Q^T Q = I$).
- Reconstruction error matches the sum of discarded eigenvalues ($\sum_{i > k} \lambda_i$).
- Transformed latent space covariance is strictly diagonal ($S_Z = D$).

---

## 📊 Summary of Experiments

| Experiment | Focus Area | Dataset | Key Finding / Artifact |
|---|---|---|---|
| **§1** | Hand-Worked 2D Example | Toy 2D (4 pts) | Pencil-and-paper math verified; exact MSE $= \lambda_2 = 0.5114$. |
| **§2** | Sklearn Validation | Digits | 8 mathematical invariant checks against `sklearn.decomposition.PCA`. |
| **§4** | Scree & Cumulative Variance | MNIST | 90% variance at $k \approx 87$, 95% at $k \approx 154$, 99% at $k \approx 331$. |
| **§5** | Classifier Accuracy vs. $k$ | MNIST | Accuracy plateaus at $k \approx 50\text{--}100$; full 784D is unnecessary. |
| **§6** | Timing & Memory Benchmark | MNIST | $5.2\times$ memory compression and massive $k\text{NN}$ distance calculation speedup. |
| **§7** | Digit Reconstruction Gallery | MNIST | Visual recovery from blurry smudges ($k=5$) to crisp digits ($k=150$). |
| **§8** | Theoretical Error Identity | MNIST | Confirms empirical MSE $= \sum_{i > k} \lambda_i$ to machine precision. |
| **§9** | 2D Latent Projection | MNIST | Natural digit cluster separation in 2D without using class labels. |
| **§10** | Eigendigits (Top 16 Components)| MNIST | Reshapes top eigenvectors to reveal handwriting stroke primitives. |
| **§11** | Standardization Effect | Wine (13D) | Proves unscaled Proline dominates 99.8% of PC1 until standardized. |
| **§12** | eigh vs. SVD Timing | Digits | Verifies numerical equivalence between covariance and SVD engines. |
| **§13** | Noise Robustness / Denoising | MNIST | Discarding small eigenvalues filters out uncorrelated Gaussian noise. |
| **§14** | Best Classifier Confusion Matrix| MNIST | Confusion matrix showing per-digit classification accuracy. |
