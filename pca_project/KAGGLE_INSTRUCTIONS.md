# Running the PCA Mini Project on Kaggle

## Quick Start (5 minutes to get running)

### Step 1: Create a Kaggle Account
- Go to [kaggle.com](https://www.kaggle.com/) and sign up (free)
- Verify your phone number (required for internet access in notebooks)

### Step 2: Create a New Notebook
1. Click **"+ Create"** → **"New Notebook"**
2. Or go directly to: https://www.kaggle.com/code

### Step 3: Upload the Notebook
1. In the new notebook page, click **File** → **"Import Notebook"**
2. Select `pca_project.ipynb` from your computer
3. The notebook will load with all 17 sections

### Step 4: Configure Notebook Settings
Click the **⚙️ Settings** panel on the right side:

| Setting | Value | Why |
|---|---|---|
| **Accelerator** | None (CPU is fine) | PCA is CPU-bound, no GPU needed |
| **Language** | Python | Default |
| **Persistence** | Files only | Saves output files |
| **Internet** | On ✅ | Needed to download MNIST and Olivetti faces |
| **Environment** | Pin to latest | Uses latest packages |

> ⚠️ **IMPORTANT**: Make sure **Internet is ON** — the notebook downloads MNIST from OpenML and Olivetti faces from sklearn. If you prefer offline, add the MNIST dataset (see Alternative Method below).

### Step 5: Run the Notebook
- Click **"Run All"** (▶▶ button at the top), OR
- Run cells one by one with **Shift+Enter**

### Step 6: Download Results
After execution completes:
1. Look at the **Output** tab in the right panel
2. All figures are in `figures/` and tables in `results/`
3. Click **"Download All"** to get a zip of all output files

---

## Alternative: Add MNIST as a Kaggle Dataset (Offline, Faster)

If internet is unreliable or you want faster loading:

1. In your notebook, click **"+ Add Data"** on the right panel
2. Search for **"Digit Recognizer"** (the official Kaggle MNIST competition)
3. Click **"Add"** — this mounts the data at `/kaggle/input/digit-recognizer/train.csv`
4. The notebook auto-detects this path and loads from there instead of downloading

> The notebook checks for Kaggle dataset paths first, then falls back to OpenML download, then to sklearn's `digits` dataset.

---

## Expected Runtime

| Section | Description | Estimated Time |
|---|---|---|
| §0 Setup | Imports & PCA class | ~5 seconds |
| §1 Hand Example | 4-point toy dataset | ~2 seconds |
| §2 Validation | sklearn comparison tests | ~5 seconds |
| §3 MNIST Loading | Download or load from disk | 10–60 seconds |
| §4 Scree & Variance | Fit PCA on all components | ~10 seconds |
| §5 Accuracy vs k | 3 classifiers × 9 k-values | **3–8 minutes** ⏳ |
| §6 Timing & Memory | Benchmark comparisons | ~2 minutes |
| §7 Reconstruction Gallery | Image grid at various k | ~15 seconds |
| §8 Recon Error vs k | MSE verification plot | ~30 seconds |
| §9 2D Projection | Scatter plot | ~5 seconds |
| §10 Eigendigits | Top 16 components as images | ~5 seconds |
| §11 Eigenfaces | Olivetti faces (downloads ~2MB) | ~30 seconds |
| §12 Standardization | Breast cancer dataset | ~5 seconds |
| §13 eigh vs SVD | Timing comparison | ~5 seconds |
| §14 Denoising (Bonus) | Noise robustness demo | ~10 seconds |
| §15 Unsupervised (Bonus) | Class separation analysis | ~5 seconds |
| §16 Confusion Matrix | Best config visualization | ~30 seconds |
| §17 Summary | Print conclusion | ~1 second |

**Total estimated: ~8–15 minutes** on Kaggle's free CPU tier.

> The slowest section is §5 (Accuracy vs k) because it trains SVM(RBF) on 10k samples at multiple dimensionalities. This is already optimized — full-dimension SVM is skipped.

---

## Output Files

After running, you'll have:

### Figures (`figures/`)
| File | Experiment | Description |
|---|---|---|
| `toy_example.png` | §1 | 2D PCA with PC axes as arrows |
| `scree_plot.png` | E1 | Eigenvalue spectrum (linear + log) |
| `cumulative_variance.png` | E2 | Cumulative EVR with 90/95/99% markers |
| `accuracy_vs_k.png` | E3 | Accuracy vs components for 3 classifiers |
| `timing_comparison.png` | E4 | Fit/predict time bar charts |
| `reconstruction_gallery.png` | E5 | Digits at k = 5, 20, 50, 150 |
| `reconstruction_error_vs_k.png` | E6 | Empirical vs theoretical MSE |
| `2d_projection.png` | E7 | PC1 vs PC2 scatter colored by digit |
| `eigendigits.png` | E8 | Top 16 eigenvectors as images |
| `eigenfaces.png` | E9 | Mean face + top eigenfaces |
| `face_reconstruction.png` | E9 | Face at various k values |
| `standardization_effect.png` | E10 | Raw vs z-scored PCA comparison |
| `denoising.png` | E13 | Before/after PCA denoising |
| `unsupervised_limitation.png` | E14 | Variance vs class separation |
| `confusion_matrix.png` | E16 | Best classifier confusion matrix |

### Tables (`results/`)
| File | Content |
|---|---|
| `validation_tests.csv` | 8 validation checks (PASS/FAIL) |
| `variance_thresholds.csv` | k values for 90%, 95%, 99% variance |
| `accuracy_vs_k.csv` | Full accuracy/F1 results per classifier per k |
| `timing_memory.csv` | Speedup ratios and compression |
| `standardization_comparison.csv` | Raw vs standardized EVR |
| `eigh_vs_svd.csv` | Timing comparison |

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'sklearn'"
This shouldn't happen on Kaggle (sklearn is pre-installed). If it does:
```python
!pip install scikit-learn
```

### MNIST download fails
- Make sure **Internet is ON** in notebook settings
- Or add the Digit Recognizer dataset (see Alternative Method above)

### Olivetti faces download fails
- Not critical — this section is skipped gracefully with a warning
- The eigenfaces experiment uses a smaller dataset that sklearn downloads

### Notebook times out
- Kaggle gives **9 hours** of CPU time per session — more than enough
- If a single cell seems stuck, restart the kernel and run again

### "MemoryError" on MNIST
- Unlikely on Kaggle (13GB RAM available)
- If it happens, the notebook falls back to the `digits` dataset (1797×64)

---

## Project File Overview

```
pca_project/
├── pca_project.ipynb      ← Upload this to Kaggle (self-contained)
├── pca_scratch.py          ← Standalone PCA module (for local use / report appendix)
├── test_pca.py             ← Validation tests (already passed locally)
├── generate_notebook.py    ← Script that generated the notebook (reference)
├── figures/                ← Created by notebook (empty until you run it)
└── results/                ← Created by notebook (empty until you run it)
```

**Only `pca_project.ipynb` needs to be uploaded to Kaggle.** It's fully self-contained — the PCA classes are inlined, no external `.py` files needed.
