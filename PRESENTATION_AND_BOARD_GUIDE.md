# Complete Presentation, Whiteboard, and Viva Voce Guide
**PCA for Feature Reduction in Machine Learning: A Linear Algebra Approach**
*A 4-Person Team Presentation & Live Board Derivation Playbook*

---

## 📋 Table of Contents
1. [Team Role Distribution & Timing (4 Members)](#1-team-role-distribution--timing-4-members)
2. [Master Presentation Script (Slide-by-Slide)](#2-master-presentation-script-slide-by-slide)
   - [Speaker 1: The Big Picture & Theoretical Setup](#speaker-1-the-big-picture--theoretical-setup-slides-1-3)
   - [Speaker 2: The Live Whiteboard Derivation](#speaker-2-the-live-whiteboard-derivation-slide-4--board)
   - [Speaker 3: Implementation, SVD Duality & MNIST Results](#speaker-3-implementation-svd-duality--mnist-results-slides-5-8)
   - [Speaker 4: The Wine Scaling Experiment, Limitations & Conclusion](#speaker-4-the-wine-scaling-experiment-limitations--conclusion-slides-9-11)
3. [The Complete Step-by-Step Whiteboard Script](#3-the-complete-step-by-step-whiteboard-script)
   - 3.1 [Board Layout Strategy](#31-board-layout-strategy)
   - 3.2 [Exact Equations to Write (Line-by-Line)](#32-exact-equations-to-write-line-by-line)
   - 3.3 [What to Say While Writing](#33-what-to-say-while-writing)
4. [Team Q&A and Defense Delegation](#4-team-qa-and-defense-delegation)
5. [Presentation Day Checklist & Rehearsal Tips](#5-presentation-day-checklist--rehearsal-tips)

---

## 1. Team Role Distribution & Timing (4 Members)

**Total Recommended Time:** 10 to 12 Minutes (+ 3–5 Minutes Viva Q&A)

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                               TEAM WORKLOAD DISTRIBUTION                                 │
├───────────────┬──────────────────────────────────┬────────────────────────┬──────────────┤
│ Member        │ Focus Area                       │ Primary Artifact       │ Time         │
├───────────────┼──────────────────────────────────┼────────────────────────┼──────────────┤
│ **Speaker 1** │ Motivation, Curse of Dim, Theory │ Slides 1–3             │ ~2.5 Minutes │
│ **Speaker 2** │ Live Board Derivation (Toy 2D)   │ Whiteboard / Marker    │ ~3.5 Minutes │
│ **Speaker 3** │ Code, Tests, SVD & MNIST Results │ Slides 5–8 (Figures)   │ ~3.0 Minutes │
│ **Speaker 4** │ Wine Experiment, Limits, Summary │ Slides 9–11 & Viva lead│ ~2.5 Minutes │
└───────────────┴──────────────────────────────────┴────────────────────────┴──────────────┘
```

---

## 2. Master Presentation Script (Slide-by-Slide)

### Speaker 1: The Big Picture & Theoretical Setup (Slides 1–3)
**Time:** ~2.5 Minutes  
**Goal:** Hook the audience, define the curse of dimensionality, and lay down the Unit 2 math foundation.

#### Slide 1: Title & Team Introduction
- **Visual:** Project Title, Course Code, Team Member Names & USNs.
- **Spoken Script:**
  > *"Good morning respected evaluators and classmates. Today, our team is presenting our mathematics mini-project: **Principal Component Analysis for Feature Reduction in Machine Learning: A Linear Algebra Approach**.*
  >
  > *Rather than treating PCA as a black-box machine learning tool, our goal was to build it entirely from first principles using the core linear algebra concepts from our Unit 2 syllabus: **sample covariance matrices, orthogonal projections, matrix diagonalization, and Singular Value Decomposition (SVD)**.*
  >
  > *We will demonstrate the complete mathematics on the whiteboard, verify our implementation with 25 automated tests against Scikit-Learn, and analyze its performance across three distinct datasets."*

#### Slide 2: The Problem: The Curse of Dimensionality
- **Visual:** A $28 \times 28$ handwritten digit image flattened into a 784-dimensional vector ($\mathbb{R}^{784}$).
- **Spoken Script:**
  > *"In modern data science, datasets live in very high-dimensional spaces. For instance, a single grayscale handwritten digit from MNIST is $28 \times 28$ pixels—meaning every single image is a point in a **784-dimensional space**.*
  >
  > *Working directly in 784 dimensions creates three major bottlenecks:*
  > 1. ***Distance Slowness:*** *Distance-based algorithms like $k$-Nearest Neighbors must calculate 784 squared differences for every single comparison, consuming massive CPU time.*
  > 2. ***Memory Overhead:*** *Storing thousands of high-dimensional vectors exhausts RAM.*
  > 3. ***Heavy Redundancy:*** *Adjacent pixels are strongly correlated, and border pixels are almost always black background. Most of these 784 axes contain zero informative variance.*
  >
  > *The core question PCA answers is: **Can we find a new, compact orthogonal coordinate system that captures the true spread of the data while eliminating redundancy?**"*

#### Slide 3: The Mathematical Pipeline (Unit 2 Formulation)
- **Visual:** Flowchart: Mean $\bar{\mathbf{x}} \to$ Deviation Matrix $B \to$ Covariance $S = \frac{1}{N-1}BB^T \to$ Eigenvalues $\det(S - \lambda I) = 0 \to$ Diagonalization $S = QDQ^T$.
- **Spoken Script:**
  > *"To do this, we follow the exact linear algebra framework taught in Unit 2:*
  > - *First, we compute the sample mean vector $\bar{\mathbf{x}}$ and subtract it to get the deviation matrix $B$. Mean-centering is mandatory because covariance measures spread about the center of mass.*
  > - *Second, we compute the sample covariance matrix $S = \frac{1}{N - 1} B B^T$. Because $S$ measures variance along any direction as a quadratic form $Q(\mathbf{u}) = \mathbf{u}^T S \mathbf{u} \ge 0$, $S$ is **Positive Semi-Definite**, guaranteeing all eigenvalues $\lambda_i \ge 0$.*
  > - *Third, by the **Spectral Theorem**, because $S$ is real and symmetric, its eigenvectors form an **orthonormal basis** that orthogonally diagonalizes $S$ into $S = Q D Q^T$. In this new coordinate system, all off-diagonal covariances vanish—features are completely decorrelated.*
  >
  > *To prove that our formulas produce exact numerical results, I will now hand over to **[Speaker 2]**, who will walk you through a complete hand-worked proof on the board."*

---

### Speaker 2: The Live Whiteboard Derivation (Slide 4 + Board)
**Time:** ~3.5 Minutes  
**Goal:** Step up to the board, write out the 2D toy problem with marker, solve the characteristic polynomial, and prove the reconstruction error identity.

*(See [Section 3](#3-the-complete-step-by-step-whiteboard-script) below for the complete word-for-word whiteboard guide).*

#### Hand-off to Speaker 3:
> *"As you can see on the board, our hand calculations prove that the variance along PC1 is exactly equal to the largest eigenvalue $\lambda_1 = 7.8220$, and the mean squared error from dropping PC2 equals $\lambda_2 = 0.5114$. Now, **[Speaker 3]** will demonstrate how we scaled this exact math into pure NumPy code and evaluated it on real-world images."*

---

### Speaker 3: Implementation, SVD Duality & MNIST Results (Slides 5–8)
**Time:** ~3.0 Minutes  
**Goal:** Showcase the pure NumPy engine, 25 unit tests, SVD duality, and MNIST compression/accuracy benchmarks.

#### Slide 5: Code Architecture & Scikit-Learn Validation
- **Visual:** Code snippet of `PCAScratch` and table showing **25/25 Pytest Tests Passed**.
- **Spoken Script:**
  > *"We implemented this linear algebra pipeline from scratch using pure NumPy in `pca_scratch.py`. We wrote two complementary classes:*
  > 1. `PCAScratch`: *Eigendecomposition via `np.linalg.eigh`.*
  > 2. `PCAScratchSVD`: *Direct Singular Value Decomposition via `np.linalg.svd`.*
  >
  > *To ensure strict mathematical correctness, we built a test suite in `test_pca.py` with **25 automated tests**. Our code matches Scikit-Learn to **$10^{-8}$ precision** on eigenvalues, variance ratios, strict orthonormality ($Q^T Q = I$), and latent decorrelation."*

#### Slide 6: The SVD Connection ($A = U \Sigma V^T$)
- **Visual:** Equation $B = U \Sigma V^T \implies S = \frac{1}{N-1} B B^T = U \left(\frac{\Sigma \Sigma^T}{N-1}\right) U^T \implies \lambda_i = \frac{\sigma_i^2}{N-1}$.
- **Spoken Script:**
  > *"In Unit 2, we learned Singular Value Decomposition: $B = U \Sigma V^T$. In our project, we proved that PCA and SVD are two sides of the same coin:*
  > - *The left singular vectors $U$ are identical to the covariance eigenvectors $Q$.*
  > - *The singular values relate directly to eigenvalues by $\lambda_i = \frac{\sigma_i^2}{N - 1}$.*
  >
  > *Why does industry use SVD instead of covariance? Because computing $B B^T$ directly squares the matrix condition number, risking numerical precision loss. SVD computes the singular values directly on $B$ without squaring."*

#### Slide 7: MNIST Spectral Decay & Cumulative Variance (E1, E2)
- **Visual:** Side-by-side plots: `scree_plot.png` and `cumulative_variance.png`.
- **Spoken Script:**
  > *"Now let's examine our primary machine learning benchmark: **MNIST (70,000 images, 784 dimensions)**.*
  >
  > *In the Scree Plot on the left, we observe an immediate exponential decay in variance with a sharp 'elbow' around $k \approx 30\text{--}50$.*
  >
  > *In the Cumulative Variance plot on the right, we identify the exact dimensions needed for standard thresholds:*
  > - **90% variance** *is retained at just $k = 87$ dimensions.*
  > - **95% variance** *is retained at $k = 154$ dimensions.*
  > - **99% variance** *requires $k = 331$ dimensions.*
  >
  > *This demonstrates a **$5.2\times$ compression ratio** ($784 \to 154$) while preserving 95% of the information!"*

#### Slide 8: Classification Accuracy, Speedup & Denoising (E3, E4, E5, E13)
- **Visual:** Grid showing `accuracy_vs_k.png`, `timing_comparison.png`, `reconstruction_gallery.png`, and `denoising.png`.
- **Spoken Script:**
  > *"What happens to classifier performance in this compressed 154-dimensional space?*
  > 1. ***Accuracy Plateau:*** *As shown in our benchmark, classifier accuracy collapses below $k = 10$, but completely saturates by $k \approx 50\text{--}100$. At $k = 154$, SVM and $k\text{NN}$ achieve over **97% accuracy**, matching the full 784-dimensional baseline!*
  > 2. ***kNN Inference Speedup:*** *Because Euclidean distance calculations scale directly with dimension, reducing from 784 to 154 features cuts operations by $5.2\times$, resulting in an immediate inference speedup and $5.2\times$ memory reduction.*
  > 3. ***Visual Reconstruction:*** *Reconstructed digits at $k=5$ look like blurry outlines, but by $k=150$, they are crisp and indistinguishable from originals.*
  > 4. ***Noise Filtering:*** *Because random Gaussian static is uncorrelated across pixels, it gets pushed into the tiny trailing eigenvalues. Truncating to top-$k$ components automatically filters out the noise!*
  >
  > *Now, **[Speaker 4]** will explain a critical engineering question: When must you standardize features before PCA?"*

---

### Speaker 4: The Wine Scaling Experiment, Limitations & Conclusion (Slides 9–11)
**Time:** ~2.5 Minutes  
**Goal:** Present the Wine dataset experiment, explain Covariance vs. Correlation, state PCA limitations, and deliver the final punchy conclusion.

#### Slide 9: The Scaling Nuance: The Wine Dataset Experiment (E10)
- **Visual:** `standardization_effect.png` comparing Raw vs. Standardized EVR.
- **Spoken Script:**
  > *"A critical question examiners frequently ask is: **'Should you always standardize features to unit variance before running PCA?'***
  >
  > *The answer depends on the data domain:*
  > - *For **images (like MNIST)**, all pixels share the identical physical unit (brightness 0–255). Standardizing images would divide near-zero variance border pixels by tiny numbers, blowing up background noise. So images must **not** be scaled.*
  > - *For **tabular data with mixed units**, you **must standardize**.*
  >
  > *To prove this, we evaluated the **Wine dataset (13 chemical attributes)**:*
  > - *Attribute **Proline** has huge values up to 1,680 ($\text{Variance} \approx 98,610$).*
  > - *Attribute **Alcohol** has small percentages ($\text{Variance} \approx 0.65$).*
  >
  > *On unscaled data, **Proline alone accounted for 99.81% of the first principal component!** PCA was completely blinded to the other 12 chemical traits.*
  >
  > *When we standardized the features ($z = \frac{x - \bar{x}}{\sigma}$), PC1 dropped to **36.20%**, allowing all 13 chemical properties to contribute fairly. This proves that for multi-unit data, PCA must be performed on the **Correlation matrix**, not the raw Covariance matrix."*

#### Slide 10: Limitations of PCA
- **Visual:** Three clear bullet points with concise visual icons/diagrams.
- **Spoken Script:**
  > *"As engineers, we must also recognize the theoretical boundaries of PCA:*
  > 1. ***Linearity:*** *PCA finds flat hyperplanes. It cannot unroll complex curved manifolds like a Swiss roll. (Nonlinear methods like Kernel PCA or autoencoders are required for that).*
  > 2. ***Sensitivity to Outliers:*** *Because variance squares distance deviations, extreme outliers can distort the principal axes.*
  > 3. ***Unsupervised Nature:*** *PCA maximizes total variance, not class separability. If between-class variance is smaller than within-class variance, PCA will not separate categories well, which is why supervised methods like Linear Discriminant Analysis (LDA) exist."*

#### Slide 11: Summary & Conclusion
- **Visual:** Clean summary table: Toy 2D $\to$ MNIST $\to$ Wine, with key quantified takeaways.
- **Spoken Script:**
  > *"To summarize our findings:*
  > 1. ***Mathematical Equivalence:*** *We verified that the sample covariance eigendecomposition, SVD duality, and discarded eigenvalue error identities hold to machine precision.*
  > 2. ***Efficiency Gains:*** *On MNIST, retaining 95% variance at $k=154$ yields a **$5.2\times$ compression ratio**, saves substantial memory, and significantly accelerates $k\text{NN}$ distance inference with zero loss in classification accuracy.*
  > 3. ***Standardization Rule:*** *Feature scaling is mandatory for multi-unit tabular data (Wine), but must be avoided on homogeneous image pixels (MNIST).*
  >
  > *All code, test suites, and documentation are open-sourced on our GitHub repository. Thank you, and we are now ready for your questions!"*

---

## 3. The Complete Step-by-Step Whiteboard Script

This section is dedicated to **Speaker 2**. Practice this on a whiteboard with markers until you can write it smoothly in 3 to 4 minutes.

### 3.1 Board Layout Strategy
Divide the whiteboard into three vertical sections:

```
┌─────────────────────────┬─────────────────────────┬─────────────────────────┐
│       LEFT PANEL        │      MIDDLE PANEL       │       RIGHT PANEL       │
│  1. Data Points         │  3. Covariance S        │  5. EVR & Projection    │
│  2. Mean & Centering    │  4. Eigenvalues &       │  6. Error Verification  │
│                         │     Eigenvectors        │     (MSE = λ₂)          │
└─────────────────────────┴─────────────────────────┴─────────────────────────┘
```

---

### 3.2 Exact Equations to Write (Line-by-Line)

#### [LEFT PANEL: Setup & Centering]
Write this first:
```
1. Data (N = 4, P = 2):
   x₁ = (2, 1)    x₂ = (3, 5)
   x₃ = (4, 3)    x₄ = (5, 7)

2. Mean Vector x̄:
   x̄₁ = (2 + 3 + 4 + 5) / 4 = 3.5
   x̄₂ = (1 + 5 + 3 + 7) / 4 = 4.0
   ==> x̄ = [3.5, 4.0]ᵀ

3. Deviation Matrix B (P × N):
   B = [ -1.5  -0.5  +0.5  +1.5 ]
       [ -3.0  +1.0  -1.0  +3.0 ]
```

#### [MIDDLE PANEL: Covariance & Eigendecomposition]
Write this in the center:
```
4. Covariance Matrix S = (1 / (N - 1)) * B * Bᵀ  (N - 1 = 3)
   B*Bᵀ = [ 5.0   8.0  ]
          [ 8.0  20.0  ]
   
   S = (1/3) * [ 5.0   8.0  ] = [ 1.67  2.67 ]
               [ 8.0  20.0  ]   [ 2.67  6.67 ]

5. Characteristic Eq: det(S - λ I) = 0
   (1.67 - λ)(6.67 - λ) - (2.67)² = 0
   λ² - 8.33 λ + 4.00 = 0

   Using Quadratic Formula:
   λ₁ = 7.8220   (Max Variance)
   λ₂ = 0.5114   (Min Variance)
   Trace Check: 1.67 + 6.67 = 8.33 = λ₁ + λ₂  ✓

6. Eigenvector for λ₁ = 7.82:
   (S - λ₁ I) q₁ = 0 ==> -6.16 q₁ + 2.67 q₂ = 0
   q₂ ≈ 2.31 q₁
   Normalize (length = 1):
   q₁ = [ 0.398, 0.918 ]ᵀ
   q₂ = [ -0.918, 0.398 ]ᵀ  (q₁ ⟂ q₂)
```

#### [RIGHT PANEL: Projection & Error Proof]
Write this on the right:
```
7. Explained Variance Ratio (EVR):
   EVR₁ = 7.8220 / 8.3333 = 93.86%
   EVR₂ = 0.5114 / 8.3333 =  6.14%

8. Projection to k = 1 (z = q₁ᵀ * x_centered):
   z₁ = 0.398(-1.5) + 0.918(-3.0) = -3.349
   z₂ = 0.398(-0.5) + 0.918(+1.0) = +0.719
   z₃ = 0.398(+0.5) + 0.918(-1.0) = -0.719
   z₄ = 0.398(+1.5) + 0.918(+3.0) = +3.349

9. FUNDAMENTAL RECONSTRUCTION ERROR IDENTITY:
   Theoretical Error = Sum of Discarded Eigenvalues
                     = λ₂ = 0.5114
   Empirical MSE     = (1 / (N - 1)) * ||B - q₁ zᵀ||²
                     = 0.5114
   ==> EXACT MATCH!  Q.E.D.
```

---

### 3.3 What to Say While Writing

Here is the exact narrative for **Speaker 2**:

1. **While writing Left Panel:**
   > *"Let's take a simple 2D toy dataset with $N = 4$ observations and $P = 2$ features: $(2,1), (3,5), (4,3), (5,7)$.*
   > *First, we compute the sample mean for each feature: $\bar{x}_1 = 3.5$ and $\bar{x}_2 = 4.0$.*
   > *Subtracting the mean from each column gives our centered deviation matrix $B$ of size $2 \times 4$."*

2. **While writing Middle Panel:**
   > *"Next, as defined in our Unit 2 notes, we form the $2 \times 2$ covariance matrix $S = \frac{1}{N - 1} B B^T$, dividing by Bessel's correction $N - 1 = 3$. This gives diagonal variances of $1.67$ and $6.67$, with cross-covariance $2.67$.*
   > *To find the principal axes, we solve the characteristic equation $\det(S - \lambda I) = 0$.*
   > *The quadratic equation $\lambda^2 - 8.33\lambda + 4.00 = 0$ yields two eigenvalues: $\lambda_1 = 7.8220$ and $\lambda_2 = 0.5114$.*
   > *We verify this instantly: the sum of the eigenvalues $\lambda_1 + \lambda_2 = 8.33$, which exactly matches the trace of $S$!*
   > *Solving $(S - \lambda_1 I)\mathbf{q}_1 = \mathbf{0}$ and normalizing yields our first principal component vector $\mathbf{q}_1 = [0.398, 0.918]^T$, and its orthogonal partner $\mathbf{q}_2$."*

3. **While writing Right Panel:**
   > *"Now observe the power of PCA:*
   > *The explained variance ratio for PC1 is $\frac{7.8220}{8.3333} = \mathbf{93.86\%}$. A single line captures nearly 94% of all information in this dataset!*
   > *When we project our points onto $\mathbf{q}_1$, we get our compressed 1D coordinates $z$.*
   > *Finally, notice our error verification: if we compress from 2D to 1D, the theoretical loss is the discarded eigenvalue $\lambda_2 = \mathbf{0.5114}$. When we calculate the empirical mean squared error between the original points and their 1D reconstruction, it equals exactly $0.5114$!*
   > *This proves mathematically that PCA discards the exact directions whose combined variance equals the reconstruction error."*

---

## 4. Team Q&A and Defense Delegation

During the viva voce, evaluators ask different types of questions. Here is how your team should divide and conquer questions so no one talks over each other:

```
┌───────────────────────┬────────────────────────────────────────────────────────┐
│ Question Category     │ Primary Defender                                       │
├───────────────────────┼────────────────────────────────────────────────────────┤
│ **Theory & Proofs**   │ **Speaker 1 & Speaker 2**                              │
│                       │ (Lagrange, Spectral Theorem, PSD, Orthogonality)       │
├───────────────────────┼────────────────────────────────────────────────────────┤
│ **Code & Algorithms** │ **Speaker 3**                                          │
│                       │ (NumPy implementation, SVD, eigh vs eig, Scikit tests) │
├───────────────────────┼────────────────────────────────────────────────────────┤
│ **Data & Nuances**    │ **Speaker 4**                                          │
│                       │ (Wine scaling, MNIST pixels, Limitations vs LDA)       │
└───────────────────────┴────────────────────────────────────────────────────────┘
```

### Top 6 Viva Questions & Quick Model Answers:

#### Q1: "Why is the covariance matrix guaranteed to have non-negative eigenvalues?"
- **Answer (Speaker 1 or 2):**  
  *"Because the variance along any direction $\mathbf{u}$ is the quadratic form $\mathbf{u}^T S \mathbf{u} = \frac{1}{N-1} \|B^T \mathbf{u}\|^2$. Since Euclidean norms are always non-negative, $\mathbf{u}^T S \mathbf{u} \ge 0$, which is the exact definition of a Positive Semi-Definite matrix. By the Spectral Theorem, all eigenvalues of a PSD matrix must be real and $\ge 0$."*

#### Q2: "Why did you use `np.linalg.eigh` instead of `np.linalg.eig` in your code?"
- **Answer (Speaker 3):**  
  *"The `h` in `eigh` stands for Hermitian (real symmetric). `eig` is for general matrices and can return imaginary complex numbers in arbitrary order. `eigh` exploits matrix symmetry, guarantees real eigenvalues, strictly enforces orthogonal eigenvectors, and runs roughly twice as fast."*

#### Q3: "What is the physical connection between SVD and PCA?"
- **Answer (Speaker 3):**  
  *"SVD factors the centered matrix as $B = U \Sigma V^T$. Substituting this into $S = \frac{1}{N-1} B B^T$ reveals that the left singular vectors $U$ are identical to the PCA eigenvectors $Q$, and $\lambda_i = \frac{\sigma_i^2}{N-1}$. SVD is preferred numerically because it avoids computing $B B^T$, preventing condition-number squaring."*

#### Q4: "Why did you NOT standardize MNIST, but DID standardize the Wine dataset?"
- **Answer (Speaker 4):**  
  *"MNIST pixels all share the identical physical scale (brightness 0–255). Standardizing would divide near-zero variance border pixels by tiny numbers, blowing up sensor noise. The Wine dataset, however, has mixed units—Proline has variance of 98,610 while Alcohol has 0.65. Without standardization, Proline monopolizes 99.8% of PC1. Standardizing puts all features on an equal footing."*

#### Q5: "Why did your eigenvectors sometimes have negative signs compared to Scikit-Learn?"
- **Answer (Speaker 2 or 3):**  
  *"Eigenvectors are defined up to a sign flip: if $S\mathbf{q} = \lambda\mathbf{q}$, then $S(-\mathbf{q}) = \lambda(-\mathbf{q})$. Both $+\mathbf{q}$ and $-\mathbf{q}$ define the exact same 1D line in space. Scikit-Learn applies an arbitrary deterministic convention called `svd_flip`. In our unit tests, we validated that $|\mathbf{q}_{\text{scratch}} \cdot \mathbf{q}_{\text{sklearn}}| = 1.0$, which proves directional equivalence."*

#### Q6: "Why is PCA called unsupervised, and when does it fail?"
- **Answer (Speaker 4):**  
  *"PCA only looks at the feature matrix $X$ and never looks at class labels $y$. It maximizes total variance. If the direction of maximum variance happens to be orthogonal to the direction that separates classes, PCA's top components will fail to separate them. In such cases, supervised methods like Linear Discriminant Analysis (LDA) are required."*

---

## 5. Presentation Day Checklist & Rehearsal Tips

### What to Bring:
- [ ] 2 Whiteboard markers (Black and Blue/Red) + 1 Whiteboard eraser (don't rely on the exam hall having working markers).
- [ ] Laptop with `pca_project.ipynb` open (with outputs already executed and visible).
- [ ] 1 Printed copy of this guide (for quick team reference before stepping into the room).
- [ ] GitHub repository link ready on a browser tab: `https://github.com/Y-astro/PiCAchu-mathminiproject`.

### Rehearsal Tips:
1. **Time Speaker 2's Board Work:** Do a practice run where Speaker 2 writes the board equations while speaking. Make sure it stays under 4 minutes.
2. **Smooth Transitions:** Practice the hand-offs:
   - Speaker 1 $\to$ Speaker 2: *"I will now hand over to [Name] for the whiteboard proof..."*
   - Speaker 2 $\to$ Speaker 3: *"Now [Name] will show how we scaled this into pure NumPy code on MNIST..."*
   - Speaker 3 $\to$ Speaker 4: *"Now [Name] will address the critical question of feature scaling..."*
3. **Never Say "PCA fits data":** Always say *"PCA performs an orthogonal change of basis / dimensionality reduction"*.
4. **Speak with Confidence on SVD:** Remind the evaluators that SVD is simply an efficient computational shortcut to find the exact same eigenpairs without forming $B B^T$.
