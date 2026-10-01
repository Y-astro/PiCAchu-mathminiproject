# Principal Component Analysis for Feature Reduction in Machine Learning: A Linear Algebra Approach

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: Pytest Passing](https://img.shields.io/badge/tests-25%20passed-brightgreen.svg)](pca_project/test_pca.py)

A mathematically rigorous, from-scratch implementation and empirical study of **Principal Component Analysis (PCA)** using linear algebra first principles (sample covariance, orthogonal diagonalization, and SVD). Validated against `scikit-learn` across 25 unit tests to machine precision, with 14 empirical benchmarks evaluated across MNIST, Olivetti Faces, Digits, and the Wine dataset.

---

## 📖 Quick Links & Documentation
- 📘 **[Comprehensive Project & Viva Voce Guide](PROJECT_GUIDE_AND_EXPLANATION.md)** — Master reference explaining all the linear algebra in plain English, bridging theory to code, the 4-point manual calculation, and an exhaustive 15-question viva playbook.
- 📓 **[Jupyter Notebook (`pca_project.ipynb`)](pca_project/pca_project.ipynb)** — Self-contained, runnable notebook containing all 18 sections and 14 empirical experiments.
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
    ├── pca_project.ipynb            # Self-contained Jupyter notebook (14 experiments)
    ├── generate_notebook.py         # Programmatic notebook builder script (nbformat)
    ├── KAGGLE_INSTRUCTIONS.md       # Kaggle cloud setup & execution instructions
    ├── figures/                     # High-resolution (300 DPI) generated figures
    └── results/                     # Raw experimental benchmark CSV tables
```

---

## ⚡ Core Concept: The 6 Steps of PCA

```
1. Mean Centering:        X̃ = X - μ
2. Covariance Matrix:     C = (1 / (m - 1)) * X̃ᵀ * X̃
3. Eigendecomposition:    C * v = λ * v  (or SVD: X̃ = U * Σ * Vᵀ)
4. Sort Eigenvalues:      λ₁ ≥ λ₂ ≥ ... ≥ λₙ
5. Select Top-k:          Q_k = [v₁, v₂, ..., v_k]
6. Project & Compress:    Z = X̃ * Q_k
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
- Directional invariance ($|\mathbf{v}_{\text{scratch}} \cdot \mathbf{v}_{\text{sklearn}}| = 1.0$).
- Strict orthonormality ($Q^T Q = I$).
- Reconstruction error matches the sum of discarded eigenvalues ($\sum_{i > k} \lambda_i$).
- Transformed latent space covariance is strictly diagonal ($\operatorname{Cov}(Z) = \Lambda$).

---

## 📊 Summary of Experiments

| Experiment | Focus Area | Dataset | Key Finding / Artifact |
|---|---|---|---|
| **E1** | Scree Plot | MNIST | Exponential spectral decay with clear elbow at $k \approx 30\text{--}50$. |
| **E2** | Cumulative Variance | MNIST | 90% variance at $k \approx 87$, 95% at $k \approx 154$, 99% at $k \approx 331$. |
| **E3** | Classifier Accuracy | MNIST | Accuracy plateaus at $k \approx 50\text{--}100$; full 784D is unnecessary. |
| **E4** | Timing & Memory | MNIST | $5.2\times$ memory compression and massive $k\text{NN}$ distance speedup. |
| **E5** | Image Reconstruction | MNIST | Visual recovery from blurry smudges ($k=5$) to crisp digits ($k=150$). |
| **E6** | Theoretical Error | MNIST | Confirms empirical MSE $= \sum_{i > k} \lambda_i$ to machine precision. |
| **E7** | 2D Latent Projection | MNIST | Natural digit cluster separation in 2D without using class labels. |
| **E8** | Eigendigits | MNIST | Reshapes top eigenvectors to reveal handwriting stroke primitives. |
| **E9** | Eigenfaces | Olivetti | Reconstructs human faces using SVD where features outnumber samples ($n \gg m$). |
| **E10** | Standardization | Wine | Proves unscaled Proline dominates 99.8% of PC1 until standardized. |
| **E11** | Sklearn Validation | Digits | 8 mathematical invariant checks against `sklearn.decomposition.PCA`. |
| **E12** | eigh vs. SVD | Digits | Verifies numerical equivalence between covariance and SVD engines. |
| **E13** | Noise Denoising | MNIST | Discarding small eigenvalues filters out uncorrelated Gaussian noise. |
| **E14** | Unsupervised Limit | Iris | Demonstrates that maximum variance does not always equal class separation. |
