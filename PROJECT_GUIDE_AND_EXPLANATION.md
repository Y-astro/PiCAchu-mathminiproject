# Principal Component Analysis (PCA): From Linear Algebra Theory to Practical Engineering
**The Comprehensive Student, Teammate, and Viva Reference Guide**
*(Aligned with Unit 2 Mathematics: Orthogonal Projections, Gram-Schmidt, Eigenvalues, Diagonalization, SVD & PCA)*

---

## Welcome to the Guide

If you and your teammates have been attending the Unit 2 Mathematics lectures—studying **Least Squares Projections, Gram-Schmidt Orthonormalization, Eigenvalues & Eigenvectors, Diagonalization, Quadratic Forms, and Singular Value Decomposition (SVD)**—this guide is written specifically for you.

In your class notebook, you solved these problems on paper:
- You computed characteristic equations like $\det(A - \lambda I) = 0$.
- You built modal matrices $S = [x_1, x_2, \dots]$ to diagonalize matrices into $D = S^{-1} A S$.
- You calculated covariance matrices using $S = \frac{1}{N - 1} B B^T$.
- You decomposed matrices into $A = U \Sigma V^T$ with singular values $\sigma_i = \sqrt{\lambda_i}$.

This guide takes the **exact variables, definitions, and theorems from your Unit 2 notes** and shows how they directly power **Principal Component Analysis (PCA)** in real code across **3 carefully chosen datasets**.

Anyone reading this guide—even with zero machine learning background—will understand how the linear algebra fits together, how the experiments work, and how to defend every line of it in a viva voce exam with complete confidence.

---

## Table of Contents
1. [The Big Picture: What is PCA Trying to Solve?](#1-the-big-picture-what-is-pca-trying-to-solve)
   - 1.1 [Representing Real Data as Vectors and Matrices](#11-representing-real-data-as-vectors-and-matrices)
   - 1.2 [The Curse of Dimensionality & Redundancy](#12-the-curse-of-dimensionality--redundancy)
   - 1.3 [The Geometric Intuition: The Camera Angle Metaphor](#13-the-geometric-intuition-the-camera-angle-metaphor)
2. [Unit 2 Linear Algebra Mapped to PCA (Step-by-Step)](#2-unit-2-linear-algebra-mapped-to-pca-step-by-step)
   - 2.1 [Notation & Matrix Setup ($N$ Samples, $P$ Attributes)](#21-notation--matrix-setup-n-samples-p-attributes)
   - 2.2 [Step 1: Mean Vector & Deviation Matrix $B$](#22-step-1-mean-vector--deviation-matrix-b)
   - 2.3 [Step 2: The Covariance Matrix $S = \frac{1}{N - 1} B B^T$](#23-step-2-the-covariance-matrix-s--frac1n---1-b-bt)
   - 2.4 [Step 3: Quadratic Forms & Positive Semi-Definiteness](#24-step-3-quadratic-forms--positive-semi-definiteness)
   - 2.5 [Step 4: Eigenvalues & Eigenvectors of $S$](#25-step-4-eigenvalues--eigenvectors-of-s)
   - 2.6 [Step 5: Orthogonal Diagonalization (Spectral Theorem)](#26-step-5-orthogonal-diagonalization-spectral-theorem)
   - 2.7 [Step 6: Decorrelation (Why the Off-Diagonals Vanish)](#27-step-6-decorrelation-why-the-off-diagonals-vanish)
   - 2.8 [Step 7: Projection & Dimension Reduction ($P \to k$)](#28-step-7-projection--dimension-reduction-p-to-k)
   - 2.9 [Step 8: Reconstruction & The Discarded Eigenvalue Identity](#29-step-8-reconstruction--the-discarded-eigenvalue-identity)
   - 2.10 [Step 9: SVD Duality ($A = U \Sigma V^T$)](#210-step-9-svd-duality-a--u-sigma-vt)
   - 2.11 [Step 10: Covariance vs. Correlation (The Wine Dataset)](#211-step-10-covariance-vs-correlation-the-wine-dataset)
   - 2.12 [Concepts Beyond the Classroom: 8 Practical Tools Demystified](#212-concepts-beyond-the-classroom-8-practical-tools-demystified)
3. [The Complete Hand-Worked 4-Point Numerical Example](#3-the-complete-hand-worked-4-point-numerical-example)
4. [The 3 Datasets and Why We Chose Them](#4-the-3-datasets-and-why-we-chose-them)
5. [The Core Empirical Experiments Explained in Plain English](#5-the-core-empirical-experiments-explained-in-plain-english)
6. [Code Architecture: Bridging Notes to NumPy](#6-code-architecture-bridging-notes-to-numpy)
7. [Comprehensive Presentation & Viva Voce Playbook](#7-comprehensive-presentation--viva-voce-playbook)
   - 7.1 [30-Second Elevator Pitch](#71-30-second-elevator-pitch)
   - 7.2 [2-Minute Complete Walkthrough](#72-2-minute-complete-walkthrough)
   - 7.3 [Slide-by-Slide Presentation Structure (Telling the 3-Dataset Story)](#73-slide-by-slide-presentation-structure-telling-the-3-dataset-story)
   - 7.4 [Top 15 Viva Questions with Direct Answers](#74-top-15-viva-questions-with-direct-answers)
   - 7.5 [Examiner Traps to Avoid](#75-examiner-traps-to-avoid)

---

## 1. The Big Picture: What is PCA Trying to Solve?

### 1.1 Representing Real Data as Vectors and Matrices
In classroom geometry, you work with 2D points $(x, y)$ or 3D points $(x, y, z)$. But real-world data has dozens, hundreds, or thousands of attributes.

Consider an image of a handwritten digit from the **MNIST** dataset:
- The image is $28$ pixels wide and $28$ pixels tall.
- That makes a grid of $28 \times 28 = 784$ total pixels.
- Each pixel contains a number from $0$ (pure black background) to $255$ (bright white stroke).

```
   28 pixels wide
   ┌───────────────┐
   │    .....      │
   │   ..   ..     │
   │     ...       │ 28 pixels tall   ===> Flatten into a single list of P = 784 numbers:
   │    ..  ..     │                       [0, 0, ..., 128, 254, 180, ..., 0]
   │    .....      │
   └───────────────┘
```

In mathematics:
- **One digit image is a single vector in a $784$-dimensional space ($\mathbb{R}^{784}$)**.
- If we collect $N = 60,000$ digit images, we have $N = 60,000$ points scattered throughout $\mathbb{R}^{784}$.

---

### 1.2 The Curse of Dimensionality & Redundancy
Working with 784 dimensions creates three major problems:
1. **Computational Slowness**: Classifiers like $k$-Nearest Neighbors ($k\text{NN}$) must calculate the Euclidean distance $\sqrt{\sum_{j=1}^{784} (x_j - y_j)^2}$ between test images and thousands of training images. Comparing 784 numbers for every distance check requires millions of CPU cycles.
2. **Memory Overhead**: Storing 70,000 vectors with 784 floating-point entries requires substantial memory.
3. **Severe Redundancy**: Hand-drawn digits share enormous amounts of redundant information. Corner pixels are almost always black ($0$) across every single image. Neighboring pixels are strongly correlated.

We do not need 784 separate axes to distinguish between digits. Most of the information lives in a much smaller, compact subspace.

---

### 1.3 The Geometric Intuition: The Camera Angle Metaphor
Imagine a running person in 3D space ($x, y, z$). You want to take a single 2D photograph of them that captures the most information:
- If you photograph them from directly above their head looking straight down, their body collapses into a small circular blob. You lose arms, legs, and facial shape.
- If you step to the side and photograph them at eye level, you see their outstretched arms, stride, and face clearly.

The runner did not change; **you simply rotated your viewing angle to align with the direction of maximal spread (variance)**.

```
       Poor Angle (Low Variance):              Optimal Angle (High Variance - PC1):
               ● (head)                            \      /
             /   \  (arms collapse)                 \ ●  /   (arms and legs
               |                                      |       clearly visible)
                                                     / \
```

**Principal Component Analysis (PCA) automatically finds the optimal coordinate axes (camera angles) for any high-dimensional dataset:**
- **PC1 (1st Principal Component)**: The single direction in space along which the data points spread out the most.
- **PC2 (2nd Principal Component)**: The direction perpendicular to PC1 that captures the next greatest spread.
- **PC$k$**: Continues picking mutually perpendicular directions until all variance is accounted for.

---

## 2. Unit 2 Linear Algebra Mapped to PCA (Step-by-Step)

Here is how the concepts from your Unit 2 notes build the entire PCA algorithm:

```
Step 1: Compute Mean Vector x̄ & Subtract to form Deviation Matrix B
                             │
                             ▼
Step 2: Compute Covariance Matrix: S = (1 / (N - 1)) * B * Bᵀ
                             │
                             ▼
Step 3: Connect to Quadratic Forms: Q(u) = uᵀ S u ≥ 0 (Positive Semi-Definite)
                             │
                             ▼
Step 4: Solve Characteristic Equation: det(S - λ I) = 0
        ==> Eigenvalues λ₁ ≥ λ₂ ≥ ... ≥ λ_P  (variances)
        ==> Eigenvectors q₁, q₂, ..., q_P    (principal directions)
                             │
                             ▼
Step 5: Orthogonal Diagonalization (Spectral Theorem): S = Q D Qᵀ
                             │
                             ▼
Step 6: Transform Data: Z = Qᵀ B  ==> Cov(Z) = D  (Features Decorrelated!)
                             │
                             ▼
Step 7: Truncate to Top-k: Q_k = [q₁, ..., q_k]  ==> Compressed Z_k
                             │
                             ▼
Step 8: Reconstruction & Error Check: MSE = Σ_{i=k+1}^P λᵢ
                             │
                             ▼
Step 9: SVD Connection: B = U Σ Vᵀ  ==> λᵢ = σᵢ² / (N - 1)
```

---

### 2.1 Notation & Matrix Setup ($N$ Samples, $P$ Attributes)
Following the exact notation from your notes (Page 45):
- Let $N$ = total number of observations / samples (e.g., $N = 60,000$ images).
- Let $P$ = total number of attributes / features (e.g., $P = 784$ pixels).

In classroom mathematics (notes page 45), the data is organized so each row is an attribute and each column is a sample:
$$
\text{Data Matrix } A = \begin{bmatrix}
a_{1,1} & a_{1,2} & \cdots & a_{1,N} \\
a_{2,1} & a_{2,2} & \cdots & a_{2,N} \\
\vdots & \vdots & \ddots & \vdots \\
a_{P,1} & a_{P,2} & \cdots & a_{P,N}
\end{bmatrix} \in \mathbb{R}^{P \times N}
$$

*(Note on NumPy / Scikit-Learn convention: In Python, data matrices are transposed so rows are samples and columns are features, $X \in \mathbb{R}^{N \times P}$. Both formulations are identical mathematically—transposing swaps rows and columns).*

---

### 2.2 Step 1: Mean Vector & Deviation Matrix $B$

#### The Mathematics (Notes Page 44):
For each attribute $a_j$, calculate its sample average $\bar{x}_j$:
$$
\bar{x}_j = \frac{1}{N} \sum_{i=1}^N a_{j,i}
$$
Form the mean vector:
$$
\bar{\mathbf{x}} = \begin{bmatrix} \bar{x}_1 \\ \bar{x}_2 \\ \vdots \\ \bar{x}_P \end{bmatrix} \in \mathbb{R}^P
$$

Now subtract the mean from every attribute. This produces the **deviation matrix $B$** (notes page 45):
$$
B = A - \bar{\mathbf{x}} \mathbf{1}_N^T \in \mathbb{R}^{P \times N}
$$
Each column of $B$ represents a centered sample: $(x_i - \bar{x})$.

#### Python / NumPy Translation:
```python
# In standard Python (where X has shape N samples, P features):
mean_ = np.mean(X, axis=0)       # shape: (P,)
Xc = X - mean_                   # deviation matrix, shape: (N, P)
```

#### Why Centering is Mandatory:
Covariance measures spread **around the average value**: $(x - \bar{x})(y - \bar{y})$.
If you do not subtract the mean, the origin remains at $(0, 0, \dots, 0)$. The first eigenvector will simply point from $(0, 0, \dots, 0)$ directly to the center of the data cloud. It will describe the offset of the dataset rather than its internal variance!

---

### 2.3 Step 2: The Covariance Matrix $S = \frac{1}{N - 1} B B^T$

#### The Mathematics (Notes Pages 44–46):
From your notes:
- **Sample Variance** of an attribute:
  $$
  \sigma^2 = \frac{\sum (x_i - \bar{x})^2}{N - 1}
  $$
- **Sample Covariance** between two attributes $x$ and $y$:
  $$
  \operatorname{Cov}(x, y) = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{N - 1}
  $$

To compute all variances and covariances simultaneously, we multiply the deviation matrix $B$ by its transpose $B^T$:
$$
S = \frac{1}{N - 1} B B^T \in \mathbb{R}^{P \times P}
$$

Let's check the matrix dimensions:
$$
(P \times N) \times (N \times P) = (P \times P)
$$

> **Direct quote from Notes Page 45:**
> *"We take $B B^T$ because we want a $P \times P$ matrix so that we can compare all attributes."*

#### Structure of the Matrix $S$ (Notes Page 46):
$$
S = \begin{bmatrix}
\operatorname{Var}(a_1) & \operatorname{Cov}(a_1, a_2) & \cdots & \operatorname{Cov}(a_1, a_P) \\
\operatorname{Cov}(a_2, a_1) & \operatorname{Var}(a_2) & \cdots & \operatorname{Cov}(a_2, a_P) \\
\vdots & \vdots & \ddots & \vdots \\
\operatorname{Cov}(a_P, a_1) & \operatorname{Cov}(a_P, a_2) & \cdots & \operatorname{Var}(a_P)
\end{bmatrix}
$$
1. **Principal Diagonal Elements**: The sample variances of the individual attributes ($\sigma_{a_1}^2, \sigma_{a_2}^2, \dots, \sigma_{a_P}^2$).
2. **Off-Diagonal Elements**: The covariances between different attributes.
3. **Symmetry**: Because $\operatorname{Cov}(a_i, a_j) = \operatorname{Cov}(a_j, a_i)$, the matrix is symmetric:
   $$
   S = S^T
   $$

#### Python / NumPy Translation:
```python
# Using Python's (N, P) convention:
S = (Xc.T @ Xc) / (N - 1)         # shape: (P, P)
```

---

### 2.4 Step 3: Quadratic Forms & Positive Semi-Definiteness

In your notes (Pages 47–49), you studied **Quadratic Forms** $Q(X) = X^T A X$ and tests for definiteness.
Here is where that theory applies directly to PCA!

Suppose we want to measure the data variance along an arbitrary unit direction vector $\mathbf{u} \in \mathbb{R}^P$ ($\|\mathbf{u}\| = 1$).
The variance along $\mathbf{u}$ is:
$$
\operatorname{Var}_{\mathbf{u}} = \mathbf{u}^T S \mathbf{u}
$$
Notice this expression: $\mathbf{u}^T S \mathbf{u}$ is a **quadratic form** associated with the symmetric matrix $S$!

From your notes (Page 48):
- Because variance can never be negative ($\operatorname{Var}_{\mathbf{u}} \ge 0$ for every direction $\mathbf{u}$), we have:
  $$
  \mathbf{u}^T S \mathbf{u} \ge 0 \quad \forall \mathbf{u}
  $$
- Therefore, the covariance matrix $S$ is **Positive Semi-Definite (PSD)**!
- **Consequence (Notes Page 48)**: All eigenvalues of a positive semi-definite matrix are non-negative:
  $$
  \lambda_i \ge 0 \quad \text{for all } i = 1, 2, \dots, P
  $$
  *(There are no negative eigenvalues in PCA!)*

---

### 2.5 Step 4: Eigenvalues & Eigenvectors of $S$

To find the directions of maximum spread, we solve the characteristic equation (Notes Page 49):
$$
\det(S - \lambda I) = 0
$$
This yields $P$ real eigenvalues:
$$
\lambda_1 \ge \lambda_2 \ge \cdots \ge \lambda_P \ge 0
$$

For each eigenvalue $\lambda_i$, solve the linear system:
$$
(S - \lambda_i I) \mathbf{q}_i = \mathbf{0}
$$
to obtain the corresponding unit eigenvector $\mathbf{q}_i$ ($\|\mathbf{q}_i\| = 1$).

#### What Do These Mean Physically?
1. **The Eigenvector $\mathbf{q}_i$** is the direction of the $i$-th Principal Component axis.
2. **The Eigenvalue $\lambda_i$** is the exact amount of variance along that axis:
   $$
   \operatorname{Var}(\mathbf{q}_i) = \mathbf{q}_i^T S \mathbf{q}_i = \mathbf{q}_i^T (\lambda_i \mathbf{q}_i) = \lambda_i (\mathbf{q}_i^T \mathbf{q}_i) = \lambda_i
   $$
- **PC1 ($\mathbf{q}_1$)**: The axis with the largest variance $\lambda_1$.
- **PC2 ($\mathbf{q}_2$)**: The perpendicular axis with the second largest variance $\lambda_2$.

#### Python / NumPy Translation:
```python
eigvals, eigvecs = np.linalg.eigh(S)

# eigh returns ascending order; sort descending:
idx = np.argsort(eigvals)[::-1]
eigvals = np.clip(eigvals[idx], 0, None)   # clamp tiny numerical noise
eigvecs = eigvecs[:, idx]                 # columns are the eigenvectors q₁, q₂, ...
```

---

### 2.6 Step 5: Orthogonal Diagonalization (Spectral Theorem)

In your notes (Pages 10–15, 30), you studied the **Gram-Schmidt process** and **Diagonalization**.
Because $S$ is real and symmetric, the **Spectral Theorem** guarantees that its eigenvectors form an **orthonormal basis** ($\mathbf{q}_i \cdot \mathbf{q}_j = 0$ for $i \ne j$, and $\|\mathbf{q}_i\| = 1$).

Assemble the eigenvectors into an orthogonal matrix $Q$:
$$
Q = [\mathbf{q}_1 \mid \mathbf{q}_2 \mid \cdots \mid \mathbf{q}_P] \in \mathbb{R}^{P \times P}
$$
Because the columns are orthonormal:
$$
Q^T Q = Q Q^T = I_P
$$
We can **orthogonally diagonalize** the covariance matrix:
$$
S = Q D Q^T \quad \iff \quad Q^T S Q = D
$$
where $D = \operatorname{diag}(\lambda_1, \lambda_2, \dots, \lambda_P)$ is the diagonal matrix of eigenvalues.

---

### 2.7 Step 6: Decorrelation (Why the Off-Diagonals Vanish)

What happens when we transform our centered data into the coordinate system of the eigenvectors?
$$
Z = Q^T B \in \mathbb{R}^{P \times N}
$$
Now compute the covariance matrix of the new variables $Z$:
$$
S_Z = \frac{1}{N - 1} Z Z^T = \frac{1}{N - 1} (Q^T B) (Q^T B)^T = Q^T \left( \frac{1}{N - 1} B B^T \right) Q = Q^T S Q = \mathbf{D}
$$
Look at the resulting matrix $D$:
$$
S_Z = \begin{bmatrix}
\lambda_1 & 0 & 0 & \cdots & 0 \\
0 & \lambda_2 & 0 & \cdots & 0 \\
0 & 0 & \lambda_3 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & \lambda_P
\end{bmatrix}
$$
- All off-diagonal elements are **ZERO**.
- **Every principal component is completely uncorrelated with every other component!**
- All overlapping redundancy between the original pixels has been eliminated.

---

### 2.8 Step 7: Projection & Dimension Reduction ($P \to k$)

To compress the data from $P$ dimensions down to $k$ dimensions ($k \ll P$):
Keep only the first $k$ columns of $Q$:
$$
Q_k = [\mathbf{q}_1 \mid \mathbf{q}_2 \mid \cdots \mid \mathbf{q}_k] \in \mathbb{R}^{P \times k}
$$

#### The Projection Formula:
$$
Z_k = Q_k^T B \in \mathbb{R}^{k \times N}
$$
*(In NumPy convention: $Z = \tilde{X} Q_k \in \mathbb{R}^{N \times k}$)*.
Each sample is now described by just $k$ numbers instead of $P = 784$ numbers!

#### Explained Variance Ratio (EVR) Demystified:
In class, you were taught eigenvalues $\lambda_1, \dots, \lambda_P$ and the trace property:
$$
\operatorname{tr}(S) = \sum_{j=1}^P \operatorname{Var}(a_j) = \sum_{i=1}^P \lambda_i
$$
The sum of the diagonal elements of $S$ is the sum of variances of all $P$ features. **This trace represents 100% of the total variance (the total spread of information) in your entire dataset.**

Now, think of the total variance as a **whole pie of size $\operatorname{tr}(S)$**:
- Each eigenvalue $\lambda_i$ is the slice of that pie captured along the $i$-th principal component.
- The **Explained Variance Ratio (EVR)** is simply the percentage size of that slice:
$$
\operatorname{EVR}_i = \frac{\lambda_i}{\operatorname{tr}(S)} = \frac{\lambda_i}{\sum_{j=1}^P \lambda_j} \times 100\%
$$
- The **Cumulative Explained Variance Ratio** is the running sum of the slices:
$$
\text{Cumulative EVR}(k) = \frac{\sum_{i=1}^k \lambda_i}{\operatorname{tr}(S)} \times 100\%
$$

**Why does this matter?**
In high-dimensional datasets like MNIST ($P = 784$), we cannot manually inspect 784 eigenvalues. Instead, we use Cumulative EVR to choose $k$ objectively: we find the smallest $k$ that retains a target threshold, such as 90% ($k = 87$) or 95% ($k = 154$).

---

### 2.9 Step 8: Reconstruction & The Discarded Eigenvalue Identity

To decompress the low-dimensional coordinates $Z_k$ back into the original $P$-dimensional space:
$$
\hat{B} = Q_k Z_k = Q_k Q_k^T B
$$
Adding back the mean gives the reconstructed data (e.g., reconstructed image):
$$
\hat{A} = \hat{B} + \bar{\mathbf{x}} \mathbf{1}_N^T
$$

#### Why Does the Reconstruction Error Equal the Discarded Eigenvalues?
In class, you learned two key theorems:
1. **The Spectral Theorem**: The eigenvectors $\mathbf{q}_1, \dots, \mathbf{q}_P$ form an **orthonormal basis** ($Q^T Q = I$). They are mutually perpendicular.
2. **Orthogonal Projections**: In least squares (Notes Pages 1–10), projecting onto a subspace creates an error vector $e$ that is strictly perpendicular to the projection subspace.

By the **generalized Pythagorean theorem in $P$ dimensions** (also known as Parseval's identity), the total variance of the data splits cleanly into two perpendicular components:
$$
\text{Total Variance} = \text{Retained Variance} + \text{Lost Variance}
$$
$$
\sum_{i=1}^P \lambda_i = \sum_{i=1}^k \lambda_i + \sum_{j=k+1}^P \lambda_j
$$
When we compress from $P$ dimensions down to $k$ dimensions, we keep the first $k$ eigenvectors and drop the remaining $(P - k)$ eigenvectors $\mathbf{q}_{k+1}, \dots, \mathbf{q}_P$.

The variance along the dropped axes is literally thrown away! Because the axes are mutually perpendicular, no error leaks into the retained subspace.
Therefore, the **Mean Squared Reconstruction Error (MSE)** is **identically equal to the sum of the discarded eigenvalues**:
$$
\text{Mean Squared Error} = \frac{1}{N - 1} \|B - \hat{B}\|_F^2 = \sum_{j = k+1}^P \lambda_j
$$
*(In Experiment E6 and our 2D hand derivation, we proved this identity: the empirical pixel error matches the theoretical sum of dropped eigenvalues to machine precision!)*

---

### 2.10 Step 9: SVD Duality ($A = U \Sigma V^T$)

In your notes (Pages 35–43), you studied **Singular Value Decomposition (SVD)**:
Any matrix $B \in \mathbb{R}^{P \times N}$ can be factored as:
$$
B = U \Sigma V^T
$$
where:
- $U \in \mathbb{R}^{P \times P}$ is an orthogonal matrix of left singular vectors (eigenvectors of $B B^T$).
- $\Sigma \in \mathbb{R}^{P \times N}$ is a diagonal matrix of non-negative singular values $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$.
- $V \in \mathbb{R}^{N \times N}$ is an orthogonal matrix of right singular vectors (eigenvectors of $B^T B$).

#### How SVD Relates to PCA (Notes Pages 35, 45):
Compute the covariance matrix using the SVD of $B$:
$$
S = \frac{1}{N - 1} B B^T = \frac{1}{N - 1} (U \Sigma V^T)(U \Sigma V^T)^T = U \left( \frac{\Sigma \Sigma^T}{N - 1} \right) U^T
$$
Comparing this directly to $S = Q D Q^T$:
1. **The left singular vectors $U$ are identical to the PCA eigenvectors $Q$ ($U = Q$)!**
2. **The eigenvalues $\lambda_i$ relate directly to the singular values $\sigma_i$ by:**
   $$
   \lambda_i = \frac{\sigma_i^2}{N - 1} \quad \iff \quad \sigma_i = \sqrt{(N - 1)\lambda_i}
   $$

#### Why SVD is Preferred in Numerical Libraries:
- Forming $B B^T$ directly squares the matrix condition number ($\kappa(S) = \kappa(B)^2$), which can lead to numerical rounding errors. SVD decomposes $B$ directly without squaring, guaranteeing higher precision.

---

### 2.11 Step 10: Covariance vs. Correlation (The Wine Dataset)

Should you always scale features to unit variance before PCA?
- **For Images (MNIST)**: **DO NOT scale.**
  - All pixels already share the exact same physical unit (brightness from $0$ to $255$).
  - In MNIST, border pixels are black across almost every image ($\sigma \approx 0$). Dividing by near-zero variance blows up sensor noise.
- **For Tabular / Multi-Unit Data (The Wine Dataset)**: **YOU MUST SCALE.**

#### The Wine Dataset Demonstration:
The **Wine dataset** contains 13 chemical features measured across 178 wine samples:
- **Proline**: Measured in mg/L with values ranging from $278$ to $1,680$ ($\text{Variance} \approx \mathbf{98,610}$).
- **Alcohol**: Measured in percentage, around $11\%\text{--}14\%$ ($\text{Variance} \approx \mathbf{0.65}$).
- **Nonflavanoid Phenols**: Measured in small decimals ($\text{Variance} \approx \mathbf{0.015}$).

```
   Raw Data Variance:
   Proline:     [==================================================] 98,610
   Alcohol:     [=]                                                  0.65
   Phenols:     []                                                   0.015
```

**What happens on raw unscaled data?**
Because Proline has such huge numbers, its variance ($98,610$) completely dominates the covariance matrix.
- **In raw data, Proline alone accounts for 99.81% of the first principal component!**
- PCA becomes blind to the other 12 chemical features.

**When standardized ($z = \frac{x - \bar{x}}{\sigma}$)**:
Every feature is rescaled to variance $= 1.0$.
- In standardized data, PC1 drops from $99.81\%$ to **$36.20\%$**, capturing a balanced combination of all 13 chemical properties.

---

### 2.12 Concepts Beyond the Classroom: 8 Practical Tools Demystified

If you review your Unit 2 handwritten lecture notes, you will notice that class lectures focus primarily on foundational linear algebra: solving $\det(A - \lambda I) = 0$, Gram-Schmidt orthogonalization, and finding $S = \frac{1}{N-1} B B^T$.

However, applying PCA to real-world datasets requires a few additional practical tools. Below is a simple, intuitive breakdown of every concept used in this project that wasn't explicitly covered in class:

#### 1. Explained Variance Ratio (EVR) & Cumulative EVR
- **Classroom Foundation**: In class, you proved the trace theorem: $\operatorname{tr}(S) = \sum_{j=1}^P \operatorname{Var}(a_j) = \sum_{i=1}^P \lambda_i$. The trace is the sum of variances of all features—the total spread of the dataset.
- **The Concept**: Total variance is a whole pie. The Explained Variance Ratio is simply what percentage of that pie belongs to one principal component:
  $$\operatorname{EVR}_i = \frac{\lambda_i}{\operatorname{tr}(S)} \times 100\%$$
- **The Analogy**: In our 2D toy problem, total variance is $\operatorname{tr}(S) = 8.3333$. The first eigenvalue is $\lambda_1 = 7.8220$. Therefore, PC1 captures $\frac{7.8220}{8.3333} = 93.86\%$ of the entire dataset's spread!

#### 2. The Reconstruction Error Identity (Discarded Eigenvalues)
- **Classroom Foundation**: Orthogonal projections onto subspaces and the Spectral Theorem ($Q^T Q = I$).
- **The Concept**: If you keep $k$ principal components and drop the remaining $(P - k)$ components, the Mean Squared Error (MSE) of reconstructing the original data equals the exact sum of the discarded eigenvalues:
  $$\text{MSE} = \sum_{j=k+1}^P \lambda_j$$
- **The Intuition**: Because the eigenvectors are mutually perpendicular, by the multi-dimensional Pythagorean theorem, total variance decomposes into:
  $$\text{Total Variance} = \text{Retained Variance } (\sum_{i=1}^k \lambda_i) + \text{Dropped Variance } (\sum_{j=k+1}^P \lambda_j)$$
  Squashing the data onto the top-$k$ hyperplane throws away the perpendicular components. The error is literally the discarded eigenvalues!

#### 3. Feature Standardization ($z$-score) vs. Just Centering
- **Classroom Foundation**: Centering $(x - \bar{x})$ to form $S = \frac{1}{N-1} B B^T$.
- **The Concept**: Variance squares deviations! If one attribute is measured in milligrams ($0\text{--}1,600$) and another in percentages ($0\text{--}14\%$), the milligram feature's variance will be hundreds of thousands of times larger. Raw PCA will pick the milligram axis as PC1 purely because of the measurement unit, completely ignoring other features!
- **The Fix**: Dividing by the standard deviation ($z = \frac{x - \bar{x}}{\sigma}$) normalizes every feature's variance to $1.0$. This transforms the Covariance Matrix into the **Correlation Matrix**, giving all features an equal voice.
- **Rule of Thumb**:
  - **Images (MNIST)**: Do NOT standardize (all pixels share the same 0–255 brightness scale; scaling blows up dark background noise).
  - **Tabular Data (Wine)**: MUST standardize (features have different physical units).

#### 4. SVD as a Computational Engine for PCA
- **Classroom Foundation**: Singular Value Decomposition $A = U \Sigma V^T$ with singular values $\sigma_i = \sqrt{\lambda_i}$.
- **The Concept**: SVD is actually a numerical shortcut to do PCA! If $B$ is the centered data matrix, then $S = \frac{1}{N-1} B B^T = U \left(\frac{\Sigma^2}{N-1}\right) U^T$. The left singular vectors $U$ ARE the eigenvectors of $S$, and $\lambda_i = \frac{\sigma_i^2}{N - 1}$.
- **Why Scikit-Learn Uses SVD**: Computing $B B^T$ directly on a computer squares the matrix condition number ($\kappa(S) = \kappa(B)^2$), which magnifies floating-point rounding errors. SVD calculates the exact same directions directly on $B$ without ever forming $B B^T$.

#### 5. The Scree Plot & The "Elbow Method"
- **Classroom Foundation**: Eigenvalues sorted in descending order $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_P \ge 0$.
- **The Concept**: A Scree plot is a graph of eigenvalues plotted in descending order. In geology, "scree" refers to the accumulation of loose rock rubble at the foot of a cliff.
- **The Intuition**: The steep cliff at the beginning represents the few dominant components holding real signals (e.g. digit shapes). The flat rubble represents background noise. The "elbow" where the cliff flattens tells engineers the exact cutoff point $k$ where adding more dimensions yields diminishing returns.

#### 6. Images as High-Dimensional Vectors & Eigendigits
- **Classroom Foundation**: Column vectors $\mathbf{x} \in \mathbb{R}^P$.
- **The Concept**: A digital image is a 2D grid of pixel intensities. A $28 \times 28$ MNIST image has 784 pixels. By reading the pixels row by row, we flatten the grid into a single column vector with $P = 784$ coordinates in $\mathbb{R}^{784}$.
- **What is an Eigendigit?** The eigenvectors of the $784 \times 784$ covariance matrix are also vectors of length 784. If we reshape an eigenvector back into a $28 \times 28$ grid and display it as an image, we get an **eigendigit**—a visual blueprint of which pixels vary together across all handwritten digits.

#### 7. Downstream ML Acceleration ($k\text{NN}$ Distance Calculation)
- **Classroom Foundation**: Euclidean distance $\sqrt{\sum_{j=1}^P (x_j - y_j)^2}$.
- **The Concept**: $k$-Nearest Neighbors ($k\text{NN}$) classifies a new test image by measuring Euclidean distance to thousands of training images. In 784 dimensions, each comparison requires 784 subtractions and squares. In PCA space ($k = 154$), it requires only 154 operations.
- **The Result**: A **$5.2\times$ computational speedup** and $5.2\times$ memory reduction, while retaining $>97\%$ classification accuracy!

#### 8. PCA Denoising via Subspace Filtering
- **Classroom Foundation**: Projecting data onto a subspace and reconstructing: $\hat{B} = Q_k Q_k^T B$.
- **The Concept**: Real signal patterns (digit strokes) are strongly correlated across neighboring pixels, meaning their energy concentrates into the top eigenvalues ($\lambda_1 \dots \lambda_k$). Random Gaussian static has no correlation and spreads uniformly in all directions, so its energy gets pushed into the tiny trailing eigenvalues.
- **The Filtering Effect**: Setting trailing components to zero during reconstruction acts as an optimal low-rank spatial filter, eliminating sensor noise!

---

## 3. The Complete Hand-Worked 4-Point Numerical Example

Here is a 2D toy problem worked out with pencil-and-paper numbers ($N = 4$ points, $P = 2$ attributes):
$$
\mathbf{x}_1 = (2, 1), \quad \mathbf{x}_2 = (3, 5), \quad \mathbf{x}_3 = (4, 3), \quad \mathbf{x}_4 = (5, 7)
$$

### 1. Compute Mean Vector $\bar{\mathbf{x}}$:
$$
\bar{x}_1 = \frac{2 + 3 + 4 + 5}{4} = 3.5, \quad \bar{x}_2 = \frac{1 + 5 + 3 + 7}{4} = 4.0 \implies \bar{\mathbf{x}} = \begin{bmatrix} 3.5 \\ 4.0 \end{bmatrix}
$$

### 2. Subtract Mean to get Deviation Matrix $B$:
$$
B = \begin{bmatrix}
2 - 3.5 & 3 - 3.5 & 4 - 3.5 & 5 - 3.5 \\
1 - 4.0 & 5 - 4.0 & 3 - 4.0 & 7 - 4.0
\end{bmatrix} = \begin{bmatrix}
-1.5 & -0.5 & +0.5 & +1.5 \\
-3.0 & +1.0 & -1.0 & +3.0
\end{bmatrix}
$$

### 3. Compute Covariance Matrix $S$ ($N - 1 = 3$):
$$
B B^T = \begin{bmatrix} 5.0 & 8.0 \\ 8.0 & 20.0 \end{bmatrix} \implies S = \frac{1}{3} B B^T = \begin{bmatrix} 1.6667 & 2.6667 \\ 2.6667 & 6.6667 \end{bmatrix}
$$

### 4. Find Eigenvalues (Characteristic Equation):
$$
\det(S - \lambda I) = \lambda^2 - 8.3333\lambda + 4.000 = 0 \implies \lambda_1 = \mathbf{7.8220}, \quad \lambda_2 = \mathbf{0.5114}
$$
- Total variance $= \operatorname{tr}(S) = 8.3333 = 7.8220 + 0.5114$ ✅

### 5. Find Eigenvectors:
$$
\mathbf{q}_1 = \begin{bmatrix} \mathbf{0.3975} \\ \mathbf{0.9176} \end{bmatrix}, \quad \mathbf{q}_2 = \begin{bmatrix} -\mathbf{0.9176} \\ \mathbf{0.3975} \end{bmatrix} \quad (\mathbf{q}_1 \cdot \mathbf{q}_2 = 0)
$$

### 6. Explained Variance Ratio:
$$
\text{EVR}_1 = \frac{7.8220}{8.3333} = \mathbf{93.86\%}, \quad \text{EVR}_2 = \frac{0.5114}{8.3333} = \mathbf{6.14\%}
$$
A single component (PC1) preserves **93.86%** of all information!

### 7. Projection & Error Check ($k = 1$):
Projecting points onto $\mathbf{q}_1$ yields coordinates $z = [-3.3491, +0.7189, -0.7189, +3.3491]$.
- Variance of $z = \mathbf{7.8220} = \lambda_1$! ✅
- Reconstruction MSE from dropping PC2 $= \mathbf{0.5114} = \lambda_2$! ✅

---

## 4. The 3 Datasets and Why We Chose Them

To keep the project clean, focused, and fast to present, we use exactly **3 datasets**:

```
                       OUR 3-DATASET STORY
                       
   1. Toy 2D Data (4 pts)       2. MNIST Digits (784D)       3. Wine Dataset (13D)
  ┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
  │  Pencil & Paper Math │     │ Real-World ML & CV   │     │ The Scaling Nuance   │
  │  • 2D -> 1D          │     │ • 784D -> 154D (95%) │     │ • 13 chemical traits │
  │  • Manual det(S-λI)=0│     │ • Visual digits      │     │ • Proline: 98,610 var│
  │  • Exact MSE = λ₂    │     │ • kNN speedup        │     │ • Cov vs Correlation │
  └──────────────────────┘     └──────────────────────┘     └──────────────────────┘
```

| # | Dataset | Dimensions | Purpose in Presentation | What You Tell the Evaluator |
|---|---|---|---|---|
| **1** | **Toy 2D Data (4 points)** | $2\text{D} \to 1\text{D}$ | **The Mathematical Proof** | *"We do this by hand on paper/whiteboard in 60 seconds to prove our formulas ($\det(S - \lambda I)=0$, $Q^T S Q = D$) match code exactly."* |
| **2** | **MNIST Handwritten Digits** | $784\text{D} \to 154\text{D}$ | **The Star Engineering Benchmark** | *"This proves PCA on real high-dimensional images: $5.2\times$ compression, visual reconstruction, and $k\text{NN}$ speedup while keeping $>97\%$ accuracy."* |
| **3** | **Wine Dataset** | $13\text{D}$ (Tabular) | **The Practical Rule (Feature Scaling)** | *"This answers the classic viva question: 'When must you standardize?' Without scaling, Proline's huge scale dominates 99.8% of PC1. Scaling fixes it."* |

---

## 5. The Core Empirical Experiments Explained in Plain English

Here is a summary of the experiments in the streamlined notebook:

1. **§1: Hand-Worked 2D Example (`figures/toy_example.png`)**: Visualizes the 4 points, the mean $\bar{\mathbf{x}}$, the PC1/PC2 eigenvector arrows, and orthogonal projections.
2. **§2: Scikit-Learn Validation (`results/validation_tests.csv`)**: Evaluates 8 mathematical invariant checks against Scikit-Learn (all pass 100%).
3. **§4: Scree Plot & Cumulative Variance (`figures/scree_plot.png`, `figures/cumulative_variance.png`)**: Shows the sharp eigenvalue elbow and verifies that 95% variance on MNIST requires only $k \approx 154$ dimensions.
4. **§5: Classifier Accuracy vs. $k$ (`figures/accuracy_vs_k.png`)**: Benchmarks Logistic Regression, $k\text{NN}$, and SVM across $k$. Shows accuracy saturates around $k \approx 50\text{--}100$.
5. **§6: Computational Speed & Memory Benchmark (`figures/timing_comparison.png`)**: Measures the $5.2\times$ memory reduction and $k\text{NN}$ distance calculation speedup.
6. **§7: Digit Reconstruction Gallery (`figures/reconstruction_gallery.png`)**: Shows digits reconstructed at $k = 5, 20, 50, 150$, showing blurry smudges sharpen into crisp digits.
7. **§8: Reconstruction Error Identity (`figures/reconstruction_error_vs_k.png`)**: Proves that empirical MSE matches the sum of discarded eigenvalues $\sum_{i > k} \lambda_i$ to machine precision.
8. **§9: 2D Latent Projection (`figures/2d_projection.png`)**: Projects 784D digits down to 2D (PC1 vs. PC2), showing how different digit clusters naturally separate without labels.
9. **§10: Eigendigits (`figures/eigendigits.png`)**: Reshapes top-16 eigenvectors into $28 \times 28$ images to reveal handwriting stroke primitives (loops, diagonals, vertical stems).
10. **§11: Standardization Study on Wine (`figures/standardization_effect.png`)**: Demonstrates that Proline dominates 99.81% of unscaled PC1, whereas standardization balances all 13 features.
11. **§12: Eigendecomposition vs. SVD Timing (`results/eigh_vs_svd.csv`)**: Verifies that covariance `eigh` and economy `svd` produce identical eigenvalues.
12. **§13: PCA Denoising (`figures/denoising.png`)**: Demonstrates that discarding small trailing eigenvalues automatically filters out random Gaussian static noise.
13. **§14: Best Classifier Confusion Matrix (`figures/confusion_matrix.png`)**: Displays per-digit recognition accuracy for the best configuration.

---

## 6. Code Architecture: Bridging Notes to NumPy

### `pca_scratch.py`:
Contains two clean classes written with pure NumPy:
- `PCAScratch`: Implements the covariance eigendecomposition pipeline (`np.linalg.eigh`).
- `PCAScratchSVD`: Implements the Singular Value Decomposition pipeline (`np.linalg.svd`).
- Methods:
  - `.fit(X)`: Computes mean vector $\bar{\mathbf{x}}$, centers data into deviation matrix, computes covariance $S = \frac{1}{N-1} B B^T$, and extracts eigenpairs.
  - `.transform(X)`: Projects data into latent coordinates: $Z = (X - \bar{\mathbf{x}}) Q_k$.
  - `.inverse_transform(Z)`: Reconstructs data: $\hat{X} = Z Q_k^T + \bar{\mathbf{x}}$.

### `test_pca.py`:
Contains **25 unit tests** run with pytest:
- Validates eigenvalues, explained variance ratios, eigenvector dot products, transform coordinates, orthonormality ($Q^T Q = I$), error identities, and the 4-point toy calculations.

---

## 7. Comprehensive Presentation & Viva Voce Playbook

### 7.1 30-Second Elevator Pitch
> *"In this project, we implemented Principal Component Analysis from mathematical first principles using linear algebra. By computing the sample covariance matrix $S = \frac{1}{N-1} B B^T$ and solving its eigenvalue problem, we proved that the directions of maximum data variance are precisely the eigenvectors of the covariance matrix. We verified our from-scratch NumPy implementation against scikit-learn across 25 unit tests to machine precision. On the 784-dimensional MNIST benchmark, we demonstrated that retaining 95% of the variance requires only ~154 dimensions—a 5.2× compression—preserving classification accuracy above 97% while speeding up kNN distance calculations. Finally, using the Wine dataset, we proved why feature standardization is essential when working with mixed physical units."*

---

### 7.2 2-Minute Complete Walkthrough
1. **The Core Problem**: High-dimensional datasets (like handwritten digit images) contain heavy multicollinearity, waste memory, and slow down distance-dependent algorithms like $k$-Nearest Neighbors.
2. **The Mathematical Solution**: Rather than using black-box libraries, we explored how PCA finds a new orthogonal coordinate system where features are completely decorrelated. We showed that subtracting the mean, computing the covariance matrix $S = \frac{1}{N-1} B B^T$, and solving $S\mathbf{q} = \lambda\mathbf{q}$ yields eigenvectors that point along maximal variance axes.
3. **The Implementation**: We built both covariance-based and SVD-based PCA engines in pure NumPy. We proved our code's accuracy with 25 automated pytest tests matching Scikit-Learn to $10^{-8}$ precision.
4. **The 3-Dataset Story**:
   - **Toy 2D Data**: Proved our manual pencil-and-paper math matches code to 8 decimals.
   - **MNIST Images (784D)**: Showed 784 features can be compressed to 154 dimensions ($5.2\times$ reduction) with $>97\%$ accuracy and faster $k\text{NN}$ inference.
   - **Wine Dataset (13D)**: Proved why feature scaling is critical—without standardization, Proline alone takes 99.8% of PC1, blinding PCA to all other attributes.

---

### 7.3 Slide-by-Slide Presentation Structure (Telling the 3-Dataset Story)

| Slide # | Slide Title | What to Show | Spoken Talking Point |
|---|---|---|---|
| **1** | Title & Overview | Title, team members, course | *"Our project implements PCA from linear algebra first principles, evaluating its effect on machine learning classifiers."* |
| **2** | Motivation: Curse of Dimensionality | $28 \times 28$ image $\to$ 784D vector | *"A digit is a point in 784D space. But neighboring pixels are redundant. We want an optimal orthogonal basis that removes redundancy."* |
| **3** | The Math Pipeline | Mean $\to$ Covariance $\to$ Eigenpairs | *"We center the data into deviation matrix $B$, compute covariance $S = \frac{1}{N-1} B B^T$, and solve $S\mathbf{q} = \lambda\mathbf{q}$. The Spectral Theorem guarantees orthogonal axes."* |
| **4** | Dataset 1: Hand-Worked 2D Example | Figure: `toy_example.png` | *"Here are 4 points calculated by hand. PC1 captures 93.86% of the variance, and the reconstruction error exactly matches eigenvalue $\lambda_2$."* |
| **5** | Code & 25 Unit Tests | Table of passed tests | *"We built PCAScratch in pure NumPy. All 25 unit tests pass, matching Scikit-Learn to machine precision."* |
| **6** | Dataset 2: Scree Plot & Choosing $k$ | Figures: `scree_plot.png`, `cumulative_variance.png` | *"Eigenvalues decay exponentially. On MNIST, 95% variance is reached at $k \approx 154$—a $5.2\times$ reduction."* |
| **7** | Dataset 2: Accuracy vs. $k$ | Figure: `accuracy_vs_k.png` | *"Classifiers reach full-dimension accuracy at $k \approx 50\text{--}100$. Below $k=10$, accuracy drops sharply because essential geometry is lost."* |
| **8** | Dataset 2: Speedup & Memory | Figure: `timing_comparison.png` | *"kNN inference is significantly faster because distance checks require $5.2\times$ fewer operations, with $5.2\times$ less RAM."* |
| **9** | Dataset 2: Visual Reconstructions & Denoising | Figures: `reconstruction_gallery.png`, `denoising.png` | *"Visual proof: $k=150$ reconstructs crisp digits. Dropping small eigenvalues automatically filters out random static noise."* |
| **10** | Dataset 3: Standardization (Wine) | Figure: `standardization_effect.png` | *"In the Wine dataset, Proline variance is 98,610 while Alcohol is 0.65. Without scaling, Proline takes 99.8% of PC1. Standardizing fixes this."* |
| **11** | Conclusion & Summary | Quantified metrics table | *"PCA is powerful, linear, and unsupervised. We verified its theory, proved its speedups, and verified its exact error identity."* |

---

### 7.4 Top 15 Viva Questions with Direct Answers

#### Q1: "What does PCA actually do in simple terms?"
> **Answer:** PCA performs a linear change of basis. It rotates the coordinate axes of the dataset so that the first axis aligns with the direction of greatest variance, the second axis aligns with the next greatest variance perpendicular to the first, and so on. We can then drop the low-variance axes to compress the data with minimal information loss.

#### Q2: "Why do we subtract the mean before computing PCA?"
> **Answer:** Covariance is defined around the center of mass of the data: $\mathbb{E}[(x - \bar{x})(y - \bar{y})]$. If data is not mean-centered, the first principal component will point from the coordinate origin $(0, 0, \dots, 0)$ to the mean center of the data points, capturing the offset location rather than the internal shape or variance of the data.

#### Q3: "What do the eigenvalues and eigenvectors represent?"
> **Answer:** The **eigenvectors** are the new coordinate directions (the principal components), and the **eigenvalues** represent the exact sample variance of the data along each respective eigenvector.

#### Q4: "Why are the principal components always perpendicular (orthogonal)?"
> **Answer:** Because the sample covariance matrix $S = \frac{1}{N-1} B B^T$ is a real symmetric matrix ($S = S^T$). By the Spectral Theorem in linear algebra, any real symmetric matrix has an orthonormal basis of eigenvectors.

#### Q5: "What does it mean that PCA decorrelates the features?"
> **Answer:** When we project data into the principal component space ($Z = Q^T B$), the covariance matrix of $Z$ is diagonal ($S_Z = D$). All off-diagonal entries are zero, meaning every principal component has zero correlation with every other component.

#### Q6: "Why is the covariance matrix Positive Semi-Definite?"
> **Answer:** For any direction vector $\mathbf{u}$, the variance along $\mathbf{u}$ is the quadratic form $\mathbf{u}^T S \mathbf{u} = \frac{1}{N-1} \|B^T \mathbf{u}\|^2 \ge 0$. Because squared norms can never be negative, $\mathbf{u}^T S \mathbf{u} \ge 0$, which is the exact definition of a Positive Semi-Definite matrix. This guarantees all eigenvalues $\lambda_i \ge 0$.

#### Q7: "How do you choose how many components ($k$) to keep?"
> **Answer:** We use the cumulative explained variance ratio ($\sum_{i=1}^k \lambda_i / \sum_{j=1}^P \lambda_j$) and choose the smallest $k$ that retains a target percentage of information, such as 90% or 95%. Alternatively, we can locate the "elbow" on a scree plot or evaluate cross-validation accuracy.

#### Q8: "What is the mathematical connection between PCA and SVD?"
> **Answer:** When we compute the SVD of the deviation matrix $B = U \Sigma V^T$, the left singular vectors $U$ are identical to the PCA eigenvectors $Q$, and the singular values $\sigma_i$ relate to eigenvalues by $\lambda_i = \frac{\sigma_i^2}{N - 1}$. SVD is numerically cleaner because it avoids computing $B B^T$ directly, preventing the condition number from squaring.

#### Q9: "Why does PCA speed up k-Nearest Neighbors (kNN)?"
> **Answer:** $k\text{NN}$ computes Euclidean distance between query points and training points, which takes $O(P)$ operations per comparison. By reducing $P = 784$ to $k = 154$, the distance calculation requires $5.2\times$ fewer operations, leading to an immediate inference speedup.

#### Q10: "What is the reconstruction error equal to?"
> **Answer:** The average squared reconstruction error from keeping $k$ components is mathematically equal to the sum of the discarded eigenvalues: $\text{MSE} = \sum_{i = k+1}^P \lambda_i$.

#### Q11: "Why should you NOT standardize MNIST images?"
> **Answer:** In MNIST, all features are already in the identical unit (pixel intensities from $0$ to $255$). Border pixels are almost always black, meaning their standard deviation is near zero. If you standardize, you divide by near-zero variance, which blows up sensor noise.

#### Q12: "Why DID you standardize the Wine dataset?"
> **Answer:** The Wine dataset contains 13 features with wildly different units and scales. Proline has values over 1,000 ($\text{variance} \approx 98,610$), while Alcohol is around 13 ($\text{variance} \approx 0.65$). Without standardization, Proline accounts for 99.81% of PC1 on its own. Standardizing puts all features on an equal footing ($\text{variance} = 1$).

#### Q13: "Why might your eigenvector signs differ from scikit-learn?"
> **Answer:** Eigenvectors are unique only up to a sign flip: if $S\mathbf{q} = \lambda\mathbf{q}$, then $S(-\mathbf{q}) = \lambda(-\mathbf{q})$. Both $+\mathbf{q}$ and $-\mathbf{q}$ define the exact same line in space. Scikit-Learn applies an arbitrary sign convention (`svd_flip`), so signs may differ, but the subspace and projections are identical.

#### Q14: "How does PCA perform noise reduction (denoising)?"
> **Answer:** True signal tends to be correlated across many pixels, producing large eigenvalues. Random noise (like Gaussian static) is uncorrelated, so its variance is spread thinly across all dimensions into small trailing eigenvalues. Truncating to the top-$k$ components drops the small eigenvalues, discarding the noise while keeping the digit structure.

#### Q15: "What are the core limitations of PCA?"
> **Answer:** 
> 1. It is strictly **linear** (it cannot unroll curved manifolds like a Swiss roll; Kernel PCA or autoencoders are required).
> 2. It is **sensitive to outliers** because variance squares large deviations.
> 3. It is **unsupervised** (it maximizes total variance, which does not always align with class separability).

---

### 7.5 Common Traps & Pitfalls to Avoid

| Examiner Trap | Correct Response |
|---|---|
| Saying *"PCA fits a model to data"* | **Incorrect.** PCA does not "fit" or train a predictive model. It is an **unsupervised orthogonal coordinate transformation / projection**. |
| Fitting PCA before train/test split | **Data Leakage!** Never compute the mean $\bar{\mathbf{x}}$ or covariance $S$ on the whole dataset before splitting. Always fit PCA on `X_train` only, then transform both `X_train` and `X_test`. |
| Using `np.linalg.eig` instead of `eigh` | `eig` is for general matrices and can return imaginary/complex numbers in arbitrary order. `eigh` is specialized for symmetric matrices, guaranteeing real eigenvalues and orthogonal eigenvectors. |
| Mixing up row vs. column eigenvectors | In NumPy `eigh`, eigenvectors are stored as **columns** ($Q[:, i]$ is $\mathbf{q}_i$). In Scikit-Learn, `pca.components_` stores them as **rows** (`components_[i, :]` is $\mathbf{q}_i$). |
| Dividing by $N$ instead of $N - 1$ | Dividing by $N$ is the biased MLE. Dividing by $N - 1$ is Bessel's correction, providing an unbiased sample covariance estimate that matches Scikit-Learn. |

---

*This document is the complete master reference for the Semester 3 Mathematics Mini Project on Principal Component Analysis.*
