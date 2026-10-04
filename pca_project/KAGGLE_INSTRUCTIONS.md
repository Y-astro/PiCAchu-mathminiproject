# Running the PCA Mini Project on Kaggle

## Quick Start (3 Steps to Run)

### Step 1: Create a New Kaggle Notebook
1. Go to [kaggle.com](https://www.kaggle.com/) and log in (or create a free account).
2. Click **"+ Create"** → **"New Notebook"** (or visit: https://www.kaggle.com/code).

### Step 2: Upload the Notebook
1. In the notebook editor, click **File** → **"Import Notebook"**.
2. Upload `pca_project.ipynb` from your computer.

### Step 3: Configure Settings & Run
In the **⚙️ Settings** panel on the right sidebar:
- **Internet**: **ON** ✅ *(Required to automatically download MNIST)*.
- **Accelerator**: None (CPU is plenty fast; no GPU needed).
- Click **"Run All"** (▶▶ button at the top).

---

## The 3 Datasets Used

To keep the project clean, fast, and mathematically transparent, the notebook uses exactly **3 datasets**:

| # | Dataset | Dimensions | Role in Project |
|---|---|---|---|
| **1** | **Toy 2D Data (4 points)** | $2\text{D} \to 1\text{D}$ | **Pencil-and-paper verification:** Traces the math step-by-step to prove formulas match code. |
| **2** | **MNIST Handwritten Digits** | $784\text{D} \to 154\text{D}$ | **Primary Machine Learning Benchmark:** Large-scale image compression, eigendigits, denoising, and classifier speedup. |
| **3** | **Wine Dataset** | $13\text{D}$ (Tabular) | **Standardization Study:** Proves why features with different scales (Proline vs. Alcohol) require scaling before PCA. |

---

## Expected Runtime per Section

| Section | Topic | Dataset | Estimated Time |
|---|---|---|---|
| §0 | Imports & From-Scratch PCA Classes | — | ~3 seconds |
| §1 | Hand-Worked 2D Example | Toy 2D (4 pts) | ~1 second |
| §2 | Validation Against Scikit-Learn | Digits | ~3 seconds |
| §3 | MNIST Data Loading | MNIST | ~15–30 seconds |
| §4 | Scree Plot & Cumulative Variance | MNIST | ~5 seconds |
| §5 | Accuracy vs. $k$ (LogReg, kNN, SVM) | MNIST (10k subset) | ~2–4 minutes ⏳ |
| §6 | Timing & Memory Benchmarks | MNIST | ~30 seconds |
| §7 | Digit Reconstruction Gallery | MNIST | ~5 seconds |
| §8 | Reconstruction Error Identity | MNIST | ~10 seconds |
| §9 | 2D Scatter Projection | MNIST | ~3 seconds |
| §10 | Eigendigits (Top 16 Components) | MNIST | ~3 seconds |
| §11 | Standardization Effect | Wine (13D) | ~2 seconds |
| §12 | Eigendecomposition vs. SVD Timing | Digits | ~2 seconds |
| §13 | PCA Noise Denoising | MNIST | ~5 seconds |
| §14 | Best Classifier Confusion Matrix | MNIST | ~10 seconds |
| §15 | Summary & Conclusion | — | ~1 second |

**Total Estimated Runtime:** **~4 to 6 minutes** on Kaggle's free CPU.

---

## Generated Output Files

### Figures (`figures/`)
- `toy_example.png` (§1): 2D points with PC1/PC2 eigenvector axes and projection lines.
- `scree_plot.png` (§4): Eigenvalue spectrum showing exponential decay and elbow.
- `cumulative_variance.png` (§4): Cumulative EVR curve with 90%, 95%, and 99% threshold markers.
- `accuracy_vs_k.png` (§5): Test accuracy vs. number of components for 3 classifiers.
- `timing_comparison.png` (§6): Fit time, predict time, and memory savings bar charts.
- `reconstruction_gallery.png` (§7): Digits reconstructed at $k = 5, 20, 50, 150$.
- `reconstruction_error_vs_k.png` (§8): Overlay confirming empirical MSE $= \sum_{i > k} \lambda_i$.
- `2d_projection.png` (§9): 2D cluster scatter plot of digits in latent space.
- `eigendigits.png` (§10): Top-16 eigenvectors visualized as stroke patterns.
- `standardization_effect.png` (§11): Comparison of raw vs. standardized EVR on the Wine dataset.
- `denoising.png` (§13): Clean vs. noisy vs. PCA-denoised digits.
- `confusion_matrix.png` (§14): Best classifier confusion matrix.

### Tables (`results/`)
- `validation_tests.csv`: 8 mathematical invariant checks against Scikit-Learn.
- `variance_thresholds.csv`: Exact $k$ values for 90%, 95%, and 99% variance.
- `accuracy_vs_k.csv`: Benchmark results per classifier per $k$.
- `timing_memory.csv`: Speedup and compression metrics.
- `standardization_comparison.csv`: Wine dataset raw vs. standardized EVR.
- `eigh_vs_svd.csv`: Eigendecomposition vs. SVD execution timing.

---

## How to Download Your Results from Kaggle
1. After running the notebook, click on the **Output** tab in the right-hand panel.
2. Select the `figures/` and `results/` folders (or click **"Download All"**).
3. All plots are saved at high-resolution **300 DPI**, ready to be embedded directly into slides or reports.
