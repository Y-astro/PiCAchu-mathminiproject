# PCA Math Mini Project: Complete Plan

**Working title (recommended):** *Principal Component Analysis for Feature Reduction in Machine Learning: A Linear Algebra Approach*
**Alt title (results-focused):** *Dimensionality Reduction of High-Dimensional Datasets using PCA and its Effect on Classifier Accuracy*

**Terminology rule:** PCA does **not** "fit data". It is dimensionality reduction / feature extraction (orthogonal projection onto top-variance eigen-directions). "Fitting" = model training. Use the correct term in the title, abstract, and viva.

---

## 0. Project Definition

| Item | Specification |
|---|---|
| Core claim | Eigendecomposition of the covariance matrix gives an orthogonal basis ordered by variance; truncating it compresses data with provably minimal squared reconstruction error |
| Math focus | Symmetric matrices, spectral theorem, orthogonal diagonalization, eigenvalues/eigenvectors, orthogonal projection, quadratic forms / Rayleigh quotient, Lagrange multipliers, SVD (optional) |
| Engineering focus | From-scratch NumPy implementation validated against `sklearn.decomposition.PCA`; effect on classifier accuracy, time, memory |
| Differentiator | (1) Full variance-maximization derivation, (2) from-scratch code matching sklearn to numerical precision, (3) accuracy-vs-k analysis with a quantified conclusion, (4) eigenfaces visual demo |
| Deliverables | Report (PDF), Jupyter notebook, slides (optional, depends on evaluation format), viva prep |

---

## 1. Requirements

### 1.1 Software
| Tool | Purpose |
|---|---|
| Python 3.10+ | Language |
| NumPy | Core PCA implementation (`eigh`, `svd`, `cov`, matmul) |
| Matplotlib | All plots |
| scikit-learn | Datasets, classifiers, train/test split, metrics, **validation only** for PCA |
| pandas (optional) | Results tables |
| Jupyter / JupyterLab | Notebook |
| LaTeX (Overleaf) or Word | Report; LaTeX strongly preferred for equations |

Install: `pip install numpy matplotlib scikit-learn pandas jupyter`

### 1.2 Data
| Dataset | Loader | Shape | Role |
|---|---|---|---|
| Toy 2D (4 points) | manual | 4×2 | Hand-worked example (Section 4) |
| sklearn `digits` | `load_digits()` (offline) | 1797×64 | Dev/debug, fast iteration |
| MNIST | `fetch_openml('mnist_784', as_frame=False)` (needs internet) | 70000×784 | **Primary experiment** |
| Olivetti faces | `fetch_olivetti_faces()` (downloads once) | 400×4096 | **Eigenfaces demo** |
| Breast Cancer Wisconsin | `load_breast_cancer()` | 569×30 | Secondary; features have different units → demonstrates need for standardization |
| Iris | `load_iris()` | 150×4 | Optional sanity check only |

### 1.3 Prerequisite knowledge to revise
- Mean, variance, covariance, covariance matrix
- Eigenvalue equation `Av = λv`, characteristic polynomial `det(A − λI) = 0`
- Spectral theorem: real symmetric A ⇒ real eigenvalues, orthogonal eigenvectors, `A = QΛQᵀ`
- Positive semi-definiteness: `xᵀAx ≥ 0`
- Orthogonal projection onto a subspace, orthonormal basis
- Lagrange multipliers for constrained optimization
- SVD: `X = UΣVᵀ`

---

## 2. Mathematical Background (goes into report Section 3)

### 2.1 Setup and notation
- Data matrix `X ∈ ℝ^(m×n)`: m samples (rows), n features (columns)
- Mean vector `μ = (1/m) Σᵢ xᵢ ∈ ℝⁿ`
- Centered data `X̃ = X − 1μᵀ` (1 = ones column vector)
- Sample covariance `C = (1/(m−1)) X̃ᵀX̃ ∈ ℝ^(n×n)`
- Properties: `C = Cᵀ` (symmetric); `vᵀCv = (1/(m−1))‖X̃v‖² ≥ 0` (PSD) ⇒ all λᵢ ≥ 0
- `C_ij = cov(feature i, feature j)`; diagonal = variances; `tr(C)` = total variance

### 2.2 Variance along a direction
For unit vector `u` (‖u‖=1), projected scalar coordinates are `X̃u`. Their variance:
`Var(u) = (1/(m−1)) ‖X̃u‖² = uᵀCu`
This is a **quadratic form** / Rayleigh quotient.

### 2.3 Variance-maximization derivation (KEY PROOF, write fully)
**Problem:** maximize `uᵀCu` subject to `uᵀu = 1`.
1. Lagrangian: `L(u, λ) = uᵀCu − λ(uᵀu − 1)`
2. Stationarity: `∂L/∂u = 2Cu − 2λu = 0 ⇒ Cu = λu`
3. So critical points are eigenvectors of C; at such a point `uᵀCu = λuᵀu = λ`
4. Maximum variance = largest eigenvalue λ₁; direction = its eigenvector v₁ = **PC1**
5. **PC2:** maximize `uᵀCu` s.t. `uᵀu = 1` and `uᵀv₁ = 0`. Adding a second multiplier and using `Cv₁ = λ₁v₁` shows the extra multiplier is 0, again giving `Cu = λu` ⇒ second largest eigenvalue, eigenvector v₂ ⟂ v₁
6. Induct for k components
7. Orthogonality of distinct-eigenvalue eigenvectors (proof for report): if `Cv₁=λ₁v₁`, `Cv₂=λ₂v₂`, `λ₁≠λ₂`, then `λ₁v₁ᵀv₂ = (Cv₁)ᵀv₂ = v₁ᵀCv₂ = λ₂v₁ᵀv₂ ⇒ (λ₁−λ₂)v₁ᵀv₂ = 0 ⇒ v₁ᵀv₂=0` (uses symmetry of C)

### 2.4 Diagonalization (decorrelation)
- `C = QΛQᵀ`, `Q = [v₁ … vₙ]` orthogonal (`QᵀQ = I`), `Λ = diag(λ₁ ≥ … ≥ λₙ ≥ 0)`
- Transformed data `Z = X̃Q` has covariance `(1/(m−1))ZᵀZ = QᵀCQ = Λ` ⇒ **principal components are uncorrelated**, with variances λᵢ

### 2.5 Dimensionality reduction and reconstruction
- Keep first k columns `Q_k ∈ ℝ^(n×k)`
- Project: `Z_k = X̃Q_k ∈ ℝ^(m×k)`
- Reconstruct: `X̂ = Z_kQ_kᵀ + 1μᵀ`
- `Q_kQ_kᵀ` is the orthogonal projector onto span(v₁..v_k): symmetric, idempotent (`P²=P`)
- **Reconstruction error identity:** `(1/(m−1))‖X̃ − X̃Q_kQ_kᵀ‖_F² = Σᵢ₌ₖ₊₁ⁿ λᵢ` (sum of discarded eigenvalues). Verify numerically in the notebook; this is a strong result to show
- **Optimality (Eckart–Young–Mirsky):** among all rank-k approximations, truncated SVD/PCA minimizes Frobenius (and spectral) error. State the theorem, cite it; full proof not required

### 2.6 Explained variance
- Explained variance ratio of PCi: `λᵢ / Σⱼλⱼ`, and `Σⱼλⱼ = tr(C)`
- Cumulative: `Σᵢ₌₁ᵏ λᵢ / Σⱼλⱼ`
- Choose smallest k with cumulative ≥ threshold (0.90 / 0.95 / 0.99). Report all three

### 2.7 SVD connection (optional but high-value)
- `X̃ = UΣVᵀ` ⇒ `X̃ᵀX̃ = VΣ²Vᵀ` ⇒ `C = V(Σ²/(m−1))Vᵀ`
- Therefore: principal directions = columns of V; `λᵢ = σᵢ²/(m−1)`; scores `Z = X̃V = UΣ`
- Why SVD preferred numerically: avoids forming `X̃ᵀX̃` (squares the condition number), works when n > m
- Sign ambiguity: eigenvectors defined up to ±1; compare via `|v_scratch · v_sklearn| ≈ 1`

### 2.8 Standardization vs centering
- **Always** mean-center
- Scale to unit variance (z-score) when features have different units/scales (breast cancer dataset). Then PCA is on the **correlation matrix**
- For pixel data (all same units), skip z-scoring; just divide by 255. Z-scoring MNIST amplifies noise in near-constant border pixels (std≈0 ⇒ division blow-up)

### 2.9 Limitations (goes into discussion)
| Limitation | Explanation |
|---|---|
| Linear only | Cannot capture curved manifolds (mention kernel PCA, t-SNE, UMAP, autoencoders as extensions; no need to implement) |
| Variance ≠ discriminability | PCA is unsupervised; top-variance directions may not separate classes (contrast LDA) |
| Scale sensitivity | Features with large variance dominate unless standardized |
| Interpretability loss | PCs are linear mixtures of all original features |
| Outlier sensitivity | Covariance is not robust |
| Gaussian assumption | Optimal decorrelation is only "complete" independence for Gaussian data |
| Eigenvalue degeneracy | Repeated eigenvalues ⇒ eigenvectors not unique (any orthonormal basis of eigenspace) |

---

## 3. Algorithm (pseudocode for report)

```
Input: X_train (m×n), target k or variance threshold τ
1. μ ← mean(X_train, axis=0)
2. X̃ ← X_train − μ
3. C ← X̃ᵀX̃ / (m−1)
4. (λ, Q) ← eigh(C)                 # ascending order
5. reorder λ, Q descending
6. if τ given: k ← min{k : cumsum(λ)[k]/sum(λ) ≥ τ}
7. Q_k ← Q[:, :k]
8. return μ, Q_k, λ
Transform(X):    (X − μ) Q_k
Inverse(Z):      Z Q_kᵀ + μ
```
Complexity: covariance `O(mn²)`, `eigh` `O(n³)`. For n=784 trivial; for n=4096 (faces) still fine (~seconds) but see Gram trick below.

**Gram-matrix trick (faces, m ≪ n):** eigendecompose `G = X̃X̃ᵀ ∈ ℝ^(m×m)` (400×400). If `Gu = λu` then `v = X̃ᵀu/‖X̃ᵀu‖` is an eigenvector of `X̃ᵀX̃` with same nonzero eigenvalue. Mention in report; optionally implement.

---

## 4. Hand-Worked Example (verified numbers; put in report Section 4)

**Data (m=4, n=2):** points `(2,1), (3,5), (4,3), (5,7)`

**Step 1 mean:** μ = (3.5, 4)

**Step 2 centered X̃:** `(−1.5,−3), (−0.5,1), (0.5,−1), (1.5,3)`

**Step 3 covariance (m−1=3):**
- Σx̃² = 2.25+0.25+0.25+2.25 = 5
- Σỹ² = 9+1+1+9 = 20
- Σx̃ỹ = 4.5−0.5−0.5+4.5 = 8
- `C = (1/3)[[5, 8],[8, 20]] = [[1.6667, 2.6667],[2.6667, 6.6667]]`

**Step 4 eigenvalues:** for `M=[[5,8],[8,20]]`: `det(M−λI)=λ²−25λ+36=0 ⇒ λ = (25 ± √481)/2 = 23.4659, 1.5341`
Divide by 3 for C: **λ₁ = 7.8220, λ₂ = 0.5114**; total variance = 25/3 = 8.3333 (= tr C ✓)

**Step 5 eigenvectors:** for λ₁(M)=23.4659: `(5−23.4659)x + 8y = 0 ⇒ y = 2.3082x`; normalize ⇒
**v₁ = (0.3975, 0.9176)ᵀ**, **v₂ = (−0.9176, 0.3975)ᵀ** (check: v₁·v₂ = 0 ✓, ‖v‖=1 ✓)

**Step 6 explained variance:** PC1 = 23.4659/25 = **93.86%**, PC2 = 6.14%

**Step 7 project to k=1:** `z = 0.3975·x̃ + 0.9176·ỹ`
| Point | z₁ |
|---|---|
| (−1.5,−3) | −3.3491 |
| (−0.5, 1) | 0.7189 |
| (0.5,−1) | −0.7189 |
| (1.5, 3) | 3.3491 |

Check: Σz² / 3 = 23.4666/3 = 7.822 = λ₁ ✓

**Step 8 reconstruction error (k=1):** discarded λ₂ = 0.5114; `Σ‖x̃ − x̂‖² / 3 = 0.5114` ✓ (matches identity in 2.5)

Include a plot: centered points, PC1/PC2 axes drawn as arrows, projections as dashed lines. Verify with `np.linalg.eigh` in the notebook.

---

## 5. Implementation

### 5.1 File / notebook structure
```
pca_project/
├── pca_project.ipynb        # main notebook (sections mirror report)
├── pca_scratch.py           # PCA class
├── figures/                 # saved PNG/PDF plots (300 dpi)
├── results/                 # CSV tables
└── report/                  # LaTeX / docx
```

### 5.2 From-scratch class (reference implementation)
```python
import numpy as np

class PCAScratch:
    def __init__(self, n_components=None, var_threshold=None):
        self.n_components = n_components
        self.var_threshold = var_threshold

    def fit(self, X):
        m = X.shape[0]
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        C = (Xc.T @ Xc) / (m - 1)
        eigvals, eigvecs = np.linalg.eigh(C)        # ascending
        idx = np.argsort(eigvals)[::-1]
        eigvals = np.clip(eigvals[idx], 0, None)     # kill tiny negative numerical noise
        eigvecs = eigvecs[:, idx]
        self.eigvals_all_, self.eigvecs_all_ = eigvals, eigvecs
        self.evr_all_ = eigvals / eigvals.sum()
        self.cum_evr_ = np.cumsum(self.evr_all_)
        if self.var_threshold is not None:
            k = int(np.searchsorted(self.cum_evr_, self.var_threshold) + 1)
        else:
            k = self.n_components or X.shape[1]
        self.k_ = k
        self.components_ = eigvecs[:, :k]            # n×k
        self.explained_variance_ = eigvals[:k]
        return self

    def transform(self, X):
        return (X - self.mean_) @ self.components_

    def inverse_transform(self, Z):
        return Z @ self.components_.T + self.mean_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

class PCAScratchSVD:
    """Same interface, SVD-based; for equivalence demo."""
    def fit(self, X, n_components=None):
        m = X.shape[0]
        self.mean_ = X.mean(axis=0)
        U, S, Vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        self.eigvals_all_ = S**2 / (m - 1)
        self.components_ = Vt[:n_components].T if n_components else Vt.T
        return self
```

### 5.3 Validation tests (all must pass; put in a table in the report)
```python
from sklearn.decomposition import PCA
sk = PCA(n_components=k).fit(X_train)
mine = PCAScratch(n_components=k).fit(X_train)

np.allclose(mine.explained_variance_, sk.explained_variance_)                  # eigenvalues
np.allclose(mine.evr_all_[:k], sk.explained_variance_ratio_)                   # ratios
# eigenvectors up to sign:
dots = np.abs(np.sum(mine.components_.T * sk.components_, axis=1))
np.allclose(dots, 1)
# transform up to sign per column:
np.allclose(np.abs(mine.transform(X_test)), np.abs(sk.transform(X_test)))
# orthonormality:
np.allclose(mine.components_.T @ mine.components_, np.eye(k))
# reconstruction error identity:
Xc = X_train - mine.mean_
err = np.sum((Xc - Xc @ mine.components_ @ mine.components_.T)**2) / (len(X_train) - 1)
np.isclose(err, mine.eigvals_all_[k:].sum())
# SVD equivalence:
np.allclose(svd_model.eigvals_all_[:k], mine.eigvals_all_[:k])
# diagonalization / decorrelation:
np.allclose(np.cov(mine.transform(X_train).T), np.diag(mine.explained_variance_), atol=1e-8)
```
Note: `np.linalg.eigh` is for symmetric matrices only (returns real eigenvalues, orthonormal vectors, ascending order). Do not use `np.linalg.eig` (general, may return complex/unsorted).

---

## 6. Experimental Protocol

### 6.1 General rules
- **No data leakage:** split first; fit `mean`, `Q` on **train only**; apply to test with the same train mean/Q. Same for any scaler.
- Fixed `random_state=42` everywhere; stratified split (`stratify=y`)
- Split: 80/20 (digits) or 60000/10000 (MNIST's standard split)
- Pixel scaling: `X = X/255.0`
- kNN on full MNIST is slow: use a 10k–20k training subset for the *original vs reduced* timing comparison, and state it explicitly
- Repeat stochastic parts (if any) with multiple seeds or use k-fold CV; report mean ± std

### 6.2 Experiment table

| ID | Experiment | Dataset | Method | Output artifact |
|---|---|---|---|---|
| E1 | Eigenvalue spectrum | MNIST | plot λᵢ (log-y also) | Scree plot |
| E2 | Cumulative explained variance | MNIST | cumsum(evr); mark k at 90/95/99% | Line plot + table of k values |
| E3 | Accuracy vs k | MNIST | k ∈ {2,5,10,20,30,50,100,200,full}; 3 classifiers | Line plot (x=k, y=test acc) + table |
| E4 | Time & memory | MNIST | fit time, predict time, feature-matrix bytes: original vs k=95%-variance | Table with speedup and compression ratio |
| E5 | Reconstruction gallery | digits/MNIST | same 10 images at k = 5, 20, 50, 150, full | Image grid |
| E6 | Reconstruction error vs k | MNIST | MSE(k) plot; overlay theoretical `Σ_{i>k}λᵢ` | Confirms identity in 2.5 |
| E7 | 2D projection | MNIST | PC1 vs PC2, colored by digit; optional 3D | Scatter |
| E8 | "Eigendigits" | MNIST | top 16 eigenvectors reshaped 28×28 | Image grid |
| E9 | Eigenfaces | Olivetti | mean face + top 16 eigenfaces; reconstruct a face at multiple k; optional face classification | Image grids |
| E10 | Standardization effect | Breast cancer | PCA raw vs z-scored; compare PC1 loadings/evr | Table + biplot; shows scale dominance |
| E11 | From-scratch vs sklearn | digits/MNIST | Section 5.3 tests | Validation table |
| E12 | Eigendecomp vs SVD | digits | timing + equivalence of λ | Small table |
| E13 | Noise robustness (optional bonus) | MNIST | add Gaussian noise, project to k, reconstruct = denoising | Before/after image grid |
| E14 | PCA is unsupervised (optional bonus) | any | show a case where a lower-variance PC separates classes better than PC1 | Plot; supports discussion |

### 6.3 Classifiers
| Model | Notes |
|---|---|
| Logistic Regression | `max_iter=1000`; scale inputs; fast |
| kNN (k=5) | Biggest beneficiary of PCA (distance cost ∝ dimension); best for timing discussion |
| SVM (RBF) | Strong baseline on MNIST; slow on the full set ⇒ use subset |

Metrics: accuracy, macro-F1; confusion matrix for best config; timing via `time.perf_counter()`; memory via `X.nbytes`.

### 6.4 Expected results (sanity ranges, not to be quoted before you measure)
- MNIST: ~95% variance retained at k≈150; 90% at k≈87; 99% at k≈331 (roughly)
- kNN accuracy on MNIST stays near ~97% with k≈50–100; drops sharply below k≈10
- sklearn `digits`: 95% variance at k≈29 of 64
- If results deviate wildly, check for leakage, missing centering, or wrong axis in `mean`

### 6.5 Conclusion statement template (fill with measured values)
> "Retaining X% of variance required k = ___ of n = ___ dimensions (___× compression), with test accuracy changing from ___% to ___% and kNN prediction time reduced by ___×."

---

## 7. Report Structure (target 15–25 pages incl. figures)

| # | Section | Contents | Length |
|---|---|---|---|
| — | Title page | Title, names, USNs, course code, guide, date | 1 p |
| — | Abstract | Problem, method, dataset, key numeric result | ≤200 words |
| 1 | Introduction | Curse of dimensionality, redundancy/correlation, motivation, objectives, contributions | 1–1.5 p |
| 2 | Literature/Background | Pearson (1901), Hotelling (1933), applications (eigenfaces, genomics, finance, compression, denoising) | 0.5–1 p |
| 3 | Mathematical foundations | Sections 2.1–2.8 of this plan, theorems with proofs | 4–6 p |
| 4 | Worked example | Section 4 of this plan + plot | 1–2 p |
| 5 | Algorithm & implementation | Pseudocode, complexity, code excerpts, design choices, validation table | 2–3 p |
| 6 | Experiments & results | Datasets, setup, E1–E14 with figures/tables and 2–3 lines of interpretation each | 5–8 p |
| 7 | Discussion | Trade-offs, limitations (2.9), when PCA fails, extensions | 1–2 p |
| 8 | Conclusion & future work | Numeric conclusion, kernel PCA, LDA, t-SNE/UMAP, autoencoders | 0.5 p |
| — | References | IEEE style | 0.5 p |
| A | Appendix | Full code, extra plots, proofs | as needed |

Formatting: number all equations; every figure needs caption + axis labels + units + legend; refer to each figure in the text; tables for numeric results; consistent notation (bold lowercase vectors, bold uppercase matrices).

### 7.1 References (verify each before citing)
- K. Pearson, "On lines and planes of closest fit to systems of points in space," *Phil. Mag.*, 1901
- H. Hotelling, "Analysis of a complex of statistical variables into principal components," *J. Educ. Psychol.*, 1933
- I. T. Jolliffe, *Principal Component Analysis*, 2nd ed., Springer, 2002
- I. T. Jolliffe & J. Cadima, "Principal component analysis: a review and recent developments," *Phil. Trans. R. Soc. A*, 2016
- J. Shlens, "A Tutorial on Principal Component Analysis," arXiv:1404.1100
- M. Turk & A. Pentland, "Eigenfaces for recognition," *J. Cognitive Neuroscience*, 1991
- G. Strang, *Linear Algebra and Its Applications* / *Introduction to Linear Algebra*
- Course textbook (engineering mathematics) for eigenvalue/diagonalization chapters
- Y. LeCun et al., MNIST database
- F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *JMLR*, 2011
- C. Eckart & G. Young, "The approximation of one matrix by another of lower rank," *Psychometrika*, 1936

---

## 8. Step-by-Step Execution Plan

| Phase | Tasks | Done when |
|---|---|---|
| **P0 Setup (0.5 day)** | Create environment/repo; install libs; download MNIST + Olivetti once and cache; confirm title with guide | `fetch_openml` works, imports run |
| **P1 Math (1–2 days)** | Write derivations 2.1–2.7 in LaTeX; do the hand example by hand and confirm numerically | Section 3 & 4 drafted |
| **P2 Core code (1 day)** | Implement `PCAScratch`; run on toy example; run all Section 5.3 tests on `digits` | All asserts pass |
| **P3 Experiments (1–2 days)** | Run E1–E14 on MNIST/faces/breast-cancer; save every figure to `figures/` and every table to `results/*.csv` | All artifacts saved |
| **P4 Analysis (0.5 day)** | Write interpretation for each result; extract headline numbers | Conclusion template filled |
| **P5 Report (1–2 days)** | Assemble report per Section 7; proofread; check equation numbering and citations | PDF exported |
| **P6 Slides & viva (0.5–1 day)** | 10–12 slides (Section 9); rehearse Section 10 questions | Timed run-through ≤ allotted time |
| **P7 Submission (0.5 day)** | Zip code + notebook + report; test that the notebook runs top-to-bottom on a clean kernel | Clean-run confirmed |

**Team split (if applicable):** A = math/derivations + hand example + report Sections 1–4; B = code + experiments + Sections 5–6; both = discussion, conclusion, slides.

**Title submission:** submit Section 0 title now. If PCA isn't explicitly in the syllabus, frame under the covered topic, e.g. "Application of Eigenvalues, Eigenvectors and Orthogonal Diagonalization in …". The rest of the plan is unchanged.

---

## 9. Presentation Outline (10–12 slides)
1. Title
2. Problem: high-dimensional data (784-D digit → why reduce)
3. Idea: rotate axes to align with variance
4. Math: covariance, eigen-equation, Lagrange derivation
5. Spectral theorem ⇒ orthogonal diagonalization
6. Algorithm (pseudocode)
7. Hand-worked 2D example (plot)
8. Scree + cumulative variance (choose k)
9. Accuracy vs k + time/memory table
10. Reconstruction + eigendigits/eigenfaces (visual)
11. Validation vs sklearn + limitations
12. Conclusion + future work

---

## 10. Viva / Q&A Preparation

| Question | Answer core |
|---|---|
| Why eigenvectors of the covariance matrix? | Maximizing `uᵀCu` s.t. `‖u‖=1` via Lagrange gives `Cu=λu` |
| Why are PCs orthogonal? | C symmetric ⇒ spectral theorem ⇒ orthogonal eigenvectors |
| Why are eigenvalues non-negative? | C is PSD: `vᵀCv = ‖X̃v‖²/(m−1) ≥ 0` |
| What does λᵢ mean? | Variance of data along vᵢ |
| Why center the data? | Covariance is defined about the mean; otherwise PC1 tracks the mean offset |
| When standardize? | Features on different scales/units |
| How to choose k? | Cumulative variance threshold, scree elbow, or validation accuracy |
| PCA vs SVD? | Same result; SVD avoids forming `XᵀX`, more stable; `λ=σ²/(m−1)` |
| PCA vs LDA? | PCA unsupervised (max variance); LDA supervised (max class separation) |
| Why fit on train only? | Otherwise test information leaks into the projection ⇒ optimistic bias |
| Is PCA lossy? | Yes when k<n; loss = sum of discarded eigenvalues |
| Why might accuracy not drop after reduction? | Discarded directions are mostly noise/redundancy; can even regularize |
| Why sign differences vs sklearn? | Eigenvectors defined up to ±; sklearn applies a sign convention (`svd_flip`) |
| Limitations? | Linear, scale-sensitive, unsupervised, outlier-sensitive, hard to interpret |
| Complexity? | `O(mn² + n³)` via covariance; SVD `O(min(mn², m²n))` |
| What if n > m? | Covariance rank ≤ m−1 ⇒ at most m−1 nonzero eigenvalues; use SVD or Gram trick |
| Is PCA the same as regression? | No: PCA minimizes perpendicular distances (total least squares) vs OLS vertical residuals |
| What is the projection matrix? | `P = Q_kQ_kᵀ`, symmetric and idempotent |

---

## 11. Common Pitfalls Checklist
- [ ] Fitting PCA (or scaler) on full data before split ⇒ leakage
- [ ] Forgetting to mean-center
- [ ] `np.linalg.eig` instead of `eigh` ⇒ complex/unsorted output
- [ ] Not sorting eigenpairs descending
- [ ] Rows vs columns confusion: `eigh` returns eigenvectors as **columns**; sklearn `components_` stores them as **rows**
- [ ] Dividing by `m` vs `m−1` inconsistently (sklearn uses `m−1`)
- [ ] Z-scoring MNIST (near-zero-std pixels)
- [ ] Comparing eigenvector signs directly
- [ ] Tiny negative eigenvalues from float error ⇒ clip at 0 before ratios/sqrt
- [ ] Reporting accuracy on training set
- [ ] Timing including data loading; time `fit` and `predict` separately
- [ ] Unlabeled axes / uncaptioned figures
- [ ] Quoting expected numbers from this plan instead of measured ones
- [ ] Notebook not re-runnable top-to-bottom

---

## 12. Grading-Oriented Rubric Self-Check
| Criterion | Evidence in project |
|---|---|
| Correct application of course topic | Sections 2, 4 |
| Mathematical rigor | Lagrange derivation, orthogonality proof, error identity, SVD link |
| Originality/effort | From-scratch code, eigenfaces, denoising, standardization study |
| Results & analysis | E1–E14 with interpretation |
| Presentation & clarity | Structured report, captioned figures, clean slides |
| Understanding (viva) | Section 10 |

---

## 13. Optional Extensions (only if time remains)
- Kernel PCA (RBF) on a nonlinear dataset (concentric circles / swiss roll) vs linear PCA
- Incremental/randomized PCA for large data (`sklearn` `IncrementalPCA`, randomized SVD)
- Power iteration to compute PC1 from scratch (ties to eigenvalue algorithms)
- Whitening: `Z_white = Z Λ^(−1/2)`
- Probabilistic PCA interpretation
- Biplot with loadings for breast-cancer features
- Compare against LDA and t-SNE/UMAP visualizations
