# Principal Component Analysis (PCA): From Linear Algebra Theory to Practical Engineering
**The Comprehensive Student, Teammate, and Viva Reference Guide**

---

## Welcome to the Project Guide

If you and your teammates have taken college linear algebra—learning about **eigenvalues, eigenvectors, matrix diagonalization, and SVD**—this guide is written specifically for you.

In classroom mathematics, you solve characteristic polynomials like $\det(A - \lambda I) = 0$ on paper with $2 \times 2$ or $3 \times 3$ matrices. But what happens when you have **70,000 images, each with 784 dimensions**? How do abstract concepts like *orthogonal diagonalization* and *positive semi-definiteness* transform into real Python code that compresses images, speeds up machine learning algorithms, and filters out noise?

This guide bridges that exact gap. It is written so that **anyone—even someone with zero machine learning background—can read it from top to bottom and fully understand what we built, why the math works, and how to defend every line of it in a viva voce examination.**

---

## Table of Contents
1. [The Big Picture: What is Dimensionality Reduction?](#1-the-big-picture-what-is-dimensionality-reduction)
   - 1.1 [Understanding High-Dimensional Data (Images as Vectors)](#11-understanding-high-dimensional-data-images-as-vectors)
   - 1.2 [The Curse of Dimensionality & Feature Redundancy](#12-the-curse-of-dimensionality--feature-redundancy)
   - 1.3 [The Intuition of PCA: The Camera Angle Metaphor](#13-the-intuition-of-pca-the-camera-angle-metaphor)
2. [Linear Algebra Theory to NumPy Code (Step-by-Step)](#2-linear-algebra-theory-to-numpy-code-step-by-step)
   - 2.1 [The Data Matrix $X$](#21-the-data-matrix-x)
   - 2.2 [Step 1: Mean Vector & Mean Centering](#22-step-1-mean-vector--mean-centering)
   - 2.3 [Step 2: The Sample Covariance Matrix $C$](#23-step-2-the-sample-covariance-matrix-c)
   - 2.4 [Step 3: Eigenvalues & Eigenvectors ($C\mathbf{v} = \lambda\mathbf{v}$)](#24-step-3-eigenvalues--eigenvectors-cv--lambdav)
   - 2.5 [Step 4: Spectral Theorem & Orthogonal Diagonalization](#25-step-4-spectral-theorem--orthogonal-diagonalization)
   - 2.6 [Step 5: Decorrelation in Latent Space](#26-step-5-decorrelation-in-latent-space)
   - 2.7 [Step 6: Projection (Compression) onto Top-$k$ Components](#27-step-6-projection-compression-onto-top-k-components)
   - 2.8 [Step 7: Reconstruction (Decompression) & Error Identity](#28-step-7-reconstruction-decompression--error-identity)
   - 2.9 [Step 8: Singular Value Decomposition (SVD) Duality](#29-step-8-singular-value-decomposition-svd-duality)
   - 2.10 [Step 9: Covariance vs. Correlation (When to Standardize: The Wine Example)](#210-step-9-covariance-vs-correlation-when-to-standardize-the-wine-example)
3. [The Complete Hand-Worked 4-Point Numerical Example](#3-the-complete-hand-worked-4-point-numerical-example)
4. [Datasets Used and Why We Chose Them](#4-datasets-used-and-why-we-chose-them)
5. [The 14 Empirical Experiments Explained in Plain English](#5-the-14-empirical-experiments-explained-in-plain-english)
6. [Code Architecture: How the Repository Works](#6-code-architecture-how-the-repository-works)
7. [Presentation Script & Viva Voce Defense Playbook](#7-presentation-script--viva-voce-defense-playbook)
   - 7.1 [30-Second Elevator Pitch](#71-30-second-elevator-pitch)
   - 7.2 [2-Minute Project Walkthrough](#72-2-minute-project-walkthrough)
   - 7.3 [Slide-by-Slide Presentation Structure](#73-slide-by-slide-presentation-structure)
   - 7.4 [Top 15 Viva Questions with Model Answers](#74-top-15-viva-questions-with-model-answers)
   - 7.5 [Common Traps & Pitfalls to Avoid](#75-common-traps--pitfalls-to-avoid)

---

## 1. The Big Picture: What is Dimensionality Reduction?

### 1.1 Understanding High-Dimensional Data (Images as Vectors)
In school, geometry is usually 2D (a graph with $x$ and $y$) or 3D ($x, y, z$). But in modern computing, data has hundreds or thousands of dimensions.

Consider a small grayscale image of a handwritten digit from the **MNIST** dataset:
- The image has a width of $28$ pixels and a height of $28$ pixels.
- That is a grid of $28 \times 28 = 784$ total pixels.
- Each pixel contains a single number representing its brightness (from $0$ for black to $255$ for white).

```
   28 pixels wide
   ┌───────────────┐
   │    .....      │
   │   ..   ..     │
   │     ...       │ 28 pixels high   ===> Flatten into a single row of 784 numbers:
   │    ..  ..     │                       [0, 0, ..., 128, 254, 180, ..., 0]
   │    .....      │                       Length = 784 features!
   └───────────────┘
```

In mathematics, **one digit image is a single point living in a 784-dimensional space ($\mathbb{R}^{784}$)**. If we have $60,000$ digit images, we have $60,000$ points scattered in $\mathbb{R}^{784}$.

---

### 1.2 The Curse of Dimensionality & Feature Redundancy
Working directly with 784 dimensions creates serious practical problems:
1. **Memory & Storage:** Storing 70,000 images with 784 float values takes tens of megabytes of RAM.
2. **Computational Slowness:** Algorithms that calculate Euclidean distance (like $k$-Nearest Neighbors) must compute $\sqrt{\sum_{j=1}^{784} (p_j - q_j)^2}$ for every single pair of points. That requires millions of calculations.
3. **Severe Redundancy:** Think about how people write digits. Border pixels (the corners of the image) are almost always pure black (value $0$). Pixels right next to each other are almost always similar shades. The features are **highly correlated** with one another.

We don't need 784 distinct axes to describe a handwritten digit. Most of those dimensions contain either **redundant information** or **meaningless background noise**.

---

### 1.3 The Intuition of PCA: The Camera Angle Metaphor
Imagine a running person in real 3D space ($x, y, z$). If you want to take a single 2D photograph of them that captures the most information:
- If you take the photo from directly above their head looking down, their body collapses into a round blob. You lose almost all detail.
- If you step to the side and take the photo from eye level, you see their outstretched arms, legs, and face.

The person didn't change; **you just rotated your viewing angle to align with the direction where their body has the most spread (variance)**.

```
       Poor Angle (Low Variance):              Optimal Angle (High Variance - PC1):
               ● (head)                            \      /
             /   \  (arms collapse)                 \ ●  /   (arms and legs
               |                                      |       clearly visible)
                                                     / \
```

**Principal Component Analysis (PCA) is an algorithm that automatically finds the best "camera angles" (coordinate axes) for any high-dimensional dataset.**
- **PC1 (1st Principal Component)**: The single direction in space along which the data varies the most.
- **PC2 (2nd Principal Component)**: The direction perpendicular to PC1 that captures the next greatest amount of variance.
- ...and so on.

By keeping only the top-$k$ directions and throwing away the rest, we compress the data while preserving the vast majority of its useful information.

---

## 2. Linear Algebra Theory to NumPy Code (Step-by-Step)

Here is how theoretical linear algebra translates directly into working Python code:

---

### 2.1 The Data Matrix $X$
Suppose we have $m$ data samples (e.g., $m = 60,000$ images) and each sample has $n$ features (e.g., $n = 784$ pixels). We organize this into a matrix $X$:
$$
X = \begin{bmatrix}
x_{1,1} & x_{1,2} & \cdots & x_{1,n} \\
x_{2,1} & x_{2,2} & \cdots & x_{2,n} \\
\vdots & \vdots & \ddots & \vdots \\
x_{m,1} & x_{m,2} & \cdots & x_{m,n}
\end{bmatrix} \in \mathbb{R}^{m \times n}
$$
- Rows represent **observations / samples** ($i = 1, \dots, m$).
- Columns represent **variables / features** ($j = 1, \dots, n$).

---

### 2.2 Step 1: Mean Vector & Mean Centering

#### The Mathematics:
Before doing anything else, compute the average value for every feature across all $m$ samples:
$$
\mu_j = \frac{1}{m} \sum_{i=1}^m x_{i,j}, \quad \boldsymbol{\mu} = \begin{bmatrix} \mu_1 \\ \mu_2 \\ \vdots \\ \mu_n \end{bmatrix} \in \mathbb{R}^n
$$
Subtract the mean vector from every row of $X$ to obtain the **mean-centered matrix** $\tilde{X}$:
$$
\tilde{X} = X - \mathbf{1}_m \boldsymbol{\mu}^T \in \mathbb{R}^{m \times n}
$$
where $\mathbf{1}_m$ is a column vector of ones of length $m$.

#### The Python / NumPy Equivalent:
```python
# X has shape (m, n)
mean_ = np.mean(X, axis=0)       # shape (n,)
Xc = X - mean_                   # NumPy broadcasts the subtraction across all m rows!
```

#### Why Centering is Mandatory:
Covariance is defined as the spread **around the mean**: $\mathbb{E}[(X - \mu_X)(Y - \mu_Y)]$.
If you do not subtract the mean, the origin of your coordinate system remains at $(0, 0, \dots, 0)$. The first eigenvector will simply point from $(0, 0, \dots, 0)$ straight to the center of the data cloud. It will describe where the data is located in space rather than the internal shape or variance of the data!

---

### 2.3 Step 2: The Sample Covariance Matrix $C$

#### The Mathematics:
The sample covariance matrix $C \in \mathbb{R}^{n \times n}$ is computed using matrix multiplication:
$$
C = \frac{1}{m - 1} \tilde{X}^T \tilde{X}
$$

Let's check the matrix dimensions:
$$
(n \times m) \times (m \times n) = (n \times n)
$$

Every entry $C_{j,k}$ in this matrix tells you something specific:
$$
C_{j,k} = \frac{1}{m - 1} \sum_{i=1}^m \tilde{x}_{i,j} \tilde{x}_{i,k}
$$
- **Diagonal entries ($j = k$)**: $C_{j,j}$ is the **variance** of feature $j$ (how wide feature $j$ spreads on its own).
- **Off-diagonal entries ($j \ne k$)**: $C_{j,k}$ is the **covariance** between feature $j$ and feature $k$:
  - If $C_{j,k} > 0$: When feature $j$ increases, feature $k$ tends to increase.
  - If $C_{j,k} < 0$: When feature $j$ increases, feature $k$ tends to decrease.
  - If $C_{j,k} \approx 0$: Features $j$ and $k$ have no linear relationship.

#### Why Divide by $m - 1$ instead of $m$?
In statistics, dividing by $m$ produces a slightly biased underestimate of variance for a sample. Dividing by $m - 1$ (**Bessel's correction**) makes the sample covariance an **unbiased estimator** of the true population covariance. This is what `np.cov()` and `scikit-learn` use.

#### The Python / NumPy Equivalent:
```python
m = X.shape[0]
C = (Xc.T @ Xc) / (m - 1)         # shape (n, n)
```

---

### 2.4 Step 3: Eigenvalues & Eigenvectors ($C\mathbf{v} = \lambda\mathbf{v}$)

This is where classroom linear algebra connects directly to machine learning.

#### What is an Eigenvector Geometrically?
When you multiply a matrix $C$ by a vector $\mathbf{x}$, the matrix typically does two things: it **rotates** the vector and **stretches/shrinks** it.
An **eigenvector** $\mathbf{v}$ is a special, magical direction where the matrix $C$ **does NOT rotate the vector at all**—it only scales it:
$$
C \mathbf{v} = \lambda \mathbf{v}
$$
The scaling factor $\lambda$ is the **eigenvalue**.

#### The Golden Rule of PCA:
> **In PCA, the eigenvectors of the covariance matrix represent the directions of the new coordinate axes, and the corresponding eigenvalues represent the exact amount of variance along those axes!**

- Direction with the most spread? That's the eigenvector $\mathbf{v}_1$ with the largest eigenvalue $\lambda_1$.
- Direction with the 2nd most spread? That's the eigenvector $\mathbf{v}_2$ with the 2nd largest eigenvalue $\lambda_2$.

#### The Python / NumPy Equivalent:
```python
# Compute eigenvalues and eigenvectors
eigvals, eigvecs = np.linalg.eigh(C)

# eigh returns them in ASCENDING order, so we reverse them to DESCENDING:
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]           # λ₁ ≥ λ₂ ≥ ... ≥ λₙ
eigvecs = eigvecs[:, idx]        # columns are the eigenvectors v₁, v₂, ...
```

> **Why `np.linalg.eigh` instead of `np.linalg.eig`?**
> The letter `h` stands for *Hermitian* (symmetric). Because $C = C^T$, its eigenvalues are mathematically guaranteed to be purely real numbers, and its eigenvectors are guaranteed to be mutually orthogonal. `np.linalg.eigh` is faster, mathematically guaranteed not to produce complex numbers with imaginary parts, and enforces strict orthogonality.

---

### 2.5 Step 4: Spectral Theorem & Orthogonal Diagonalization

In college linear algebra, you learned the **Spectral Theorem**:
> *Every real symmetric matrix $C = C^T$ can be orthogonally diagonalized:*
$$
C = Q \Lambda Q^T
$$
where:
- $Q = [\mathbf{v}_1 \mid \mathbf{v}_2 \mid \cdots \mid \mathbf{v}_n] \in \mathbb{R}^{n \times n}$ is an **orthogonal matrix** whose columns are the orthonormal eigenvectors ($Q^T Q = Q Q^T = I_n$).
- $\Lambda = \operatorname{diag}(\lambda_1, \lambda_2, \dots, \lambda_n)$ is a **diagonal matrix** containing the eigenvalues in descending order.

#### What does $Q^T Q = I$ mean?
It means every eigenvector has length $1$ ($\|\mathbf{v}_i\| = 1$) and any two distinct eigenvectors are strictly perpendicular ($\mathbf{v}_i \cdot \mathbf{v}_j = 0$ when $i \ne j$). The new principal component axes are completely independent 90-degree coordinates!

---

### 2.6 Step 5: Decorrelation in Latent Space

What happens if we project our centered data $\tilde{X}$ onto this new eigenvector coordinate system?
$$
Z = \tilde{X} Q \in \mathbb{R}^{m \times n}
$$
Now let's compute the covariance matrix of this new data $Z$:
$$
\operatorname{Cov}(Z) = \frac{1}{m - 1} Z^T Z = \frac{1}{m - 1} (Q^T \tilde{X}^T) (\tilde{X} Q) = Q^T \left( \frac{\tilde{X}^T \tilde{X}}{m - 1} \right) Q = Q^T C Q
$$
Since $C = Q \Lambda Q^T$, substituting this in gives:
$$
\operatorname{Cov}(Z) = Q^T (Q \Lambda Q^T) Q = (Q^T Q) \Lambda (Q^T Q) = I \Lambda I = \mathbf{\Lambda}
$$

Look at that result:
$$
\operatorname{Cov}(Z) = \begin{bmatrix}
\lambda_1 & 0 & 0 & \cdots & 0 \\
0 & \lambda_2 & 0 & \cdots & 0 \\
0 & 0 & \lambda_3 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & \lambda_n
\end{bmatrix}
$$
- All the off-diagonal covariances are **identically ZERO**.
- **Every single principal component is completely uncorrelated with every other component!**
- All the redundant, overlapping information in the original 784 pixels has been untangled and separated into independent features.

---

### 2.7 Step 6: Projection (Compression) onto Top-$k$ Components

Now we can perform **dimensionality reduction**.
Instead of keeping all $n$ eigenvectors, we keep only the first $k$ eigenvectors ($k \ll n$, e.g., $k = 150$ instead of $n = 784$):
$$
Q_k = [\mathbf{v}_1 \mid \mathbf{v}_2 \mid \cdots \mid \mathbf{v}_k] \in \mathbb{R}^{n \times k}
$$

#### The Projection Formula:
$$
Z_k = \tilde{X} Q_k \in \mathbb{R}^{m \times k}
$$
- Dimensions: $(m \times n) \times (n \times k) = \mathbf{m \times k}$.
- Every 784-pixel image is now compressed into just $k$ numbers!

#### The Python / NumPy Equivalent:
```python
Z = (X - mean_) @ components_     # shape (m, k)
```

---

### 2.8 Step 7: Reconstruction (Decompression) & Error Identity

Can we get our original image back from the $k$ numbers in $Z_k$?
Yes! We multiply by $Q_k^T$ and add back the mean:
$$
\hat{X} = Z_k Q_k^T + \boldsymbol{\mu} = \tilde{X} Q_k Q_k^T + \boldsymbol{\mu}
$$

#### The Fundamental Reconstruction Error Identity:
Because we dropped components $k+1, k+2, \dots, n$, the reconstructed images will not be 100% identical to the originals—there is a small reconstruction error.

How big is that error? **It is mathematically identical to the sum of the eigenvalues we threw away!**
$$
\text{Mean Squared Error} = \frac{1}{m - 1} \|\tilde{X} - \hat{X}_{\text{centered}}\|_F^2 = \sum_{i = k+1}^n \lambda_i
$$

> **Why this matters for your viva:**
> In our project, we verified this numerically in Experiment E6. We calculated the empirical pixel-by-pixel squared error between original images and reconstructed images, and compared it to $\sum_{i > k} \lambda_i$. They match to 8 decimal places!

---

### 2.9 Step 8: Singular Value Decomposition (SVD) Duality

In your math class, you learned about SVD: any matrix $\tilde{X} \in \mathbb{R}^{m \times n}$ can be decomposed into:
$$
\tilde{X} = U \Sigma V^T
$$
where:
- $U \in \mathbb{R}^{m \times m}$ is an orthogonal matrix of left singular vectors.
- $\Sigma \in \mathbb{R}^{m \times n}$ is a diagonal matrix containing singular values $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$.
- $V \in \mathbb{R}^{n \times n}$ is an orthogonal matrix of right singular vectors.

#### The Bridge: How SVD relates to PCA
Let's compute $\tilde{X}^T \tilde{X}$ using the SVD formula:
$$
\tilde{X}^T \tilde{X} = (U \Sigma V^T)^T (U \Sigma V^T) = V \Sigma^T U^T U \Sigma V^T = V \Sigma^T (I) \Sigma V^T = V \Sigma^2 V^T
$$
Now divide both sides by $m - 1$:
$$
C = \frac{\tilde{X}^T \tilde{X}}{m - 1} = V \left( \frac{\Sigma^2}{m - 1} \right) V^T
$$

Look at that equation closely! It is identical to $C = Q \Lambda Q^T$:
1. **The right singular vectors $V$ are the exact same as the PCA eigenvectors $Q$ ($V = Q$)!**
2. **The eigenvalues $\lambda_i$ are related to the singular values $\sigma_i$ by:**
   $$
   \lambda_i = \frac{\sigma_i^2}{m - 1}
   $$

#### Why does Scikit-Learn use SVD instead of Covariance?
1. **Numerical Stability:** In floating-point computing, computing $\tilde{X}^T \tilde{X}$ squares the matrix condition number ($\kappa(C) = \kappa(\tilde{X})^2$). If your data has very small numbers, squaring them can cause roundoff errors and loss of precision. SVD operates directly on $\tilde{X}$ without squaring anything.
2. **When Features Exceed Samples ($n > m$):** In face recognition, each image is $64 \times 64 = 4,096$ pixels ($n = 4,096$), but we only have 400 photos ($m = 400$).
   - The covariance matrix $C$ would be $4,096 \times 4,096$ ($16.7$ million numbers).
   - Economy SVD only computes matrices sized $400 \times 400$, running hundreds of times faster and using a fraction of the memory!

---

### 2.10 Step 9: Covariance vs. Correlation (When to Standardize: The Wine Example)

Should you always scale your data to zero mean and unit variance ($z$-score standardization) before PCA?
- **Rule for Images (MNIST, Faces, Digits)**: **DO NOT standardize.**
  - All pixels already share the exact same physical scale (grayscale brightness from $0$ to $255$).
  - If you divide by standard deviation, border pixels that are black across almost every image have near-zero variance ($\sigma \approx 0$). Dividing by near-zero variance blows up random sensor noise!
- **Rule for Tabular / Physical Data (Wine Dataset)**: **YOU MUST STANDARDIZE.**

#### The Wine Dataset Experiment (Experiment E10):
The **Wine dataset** contains 13 chemical properties measured across 178 wine samples:
- **Proline**: Measured in milligrams per liter, with values ranging from $278$ to $1,680$ ($\text{Variance} \approx \mathbf{98,610}$).
- **Alcohol**: Measured in percentage, around $11\%\text{--}14\%$ ($\text{Variance} \approx \mathbf{0.65}$).
- **Nonflavanoid Phenols**: Measured in tiny decimals ($\text{Variance} \approx \mathbf{0.015}$).

```
   Raw Data:
   Proline:     [==================================================] Variance = 98,610
   Alcohol:     [=]                                                  Variance = 0.65
   Phenols:     []                                                   Variance = 0.015
```

**What happens if you run PCA on raw, unscaled data?**
Because Proline has such huge numbers, its variance ($98,610$) completely dwarfs everything else.
In our experiment:
- **In raw data, Proline alone accounts for 99.81% of the first principal component!**
- PCA becomes blind to the other 12 chemical features. It essentially just measures Proline and ignores everything else.

**What happens when you standardize ($z = \frac{x - \mu}{\sigma}$)?**
Every feature is rescaled to have $\text{mean} = 0$ and $\text{variance} = 1.0$.
- In standardized data, PC1 drops from $99.81\%$ down to **$36.20\%$**.
- Now PC1 represents a genuine, balanced multivariate chemical fingerprint of the wine, rather than just measuring a single feature with large units.

---

## 3. The Complete Hand-Worked 4-Point Numerical Example

To understand the calculations completely, trace this exact 2D toy problem ($m = 4$ points, $n = 2$ dimensions):
$$
\mathbf{x}_1 = (2, 1), \quad \mathbf{x}_2 = (3, 5), \quad \mathbf{x}_3 = (4, 3), \quad \mathbf{x}_4 = (5, 7)
$$

```
   y ^
   7 │                 • (5,7)
   6 │
   5 │        • (3,5)
   4 │            x Mean (3.5, 4.0)
   3 │            • (4,3)
   2 │
   1 │   • (2,1)
   0 └──────────────────────> x
       0  1  2  3  4  5  6
```

### 1. Compute Mean Vector $\boldsymbol{\mu}$:
$$
\mu_x = \frac{2 + 3 + 4 + 5}{4} = 3.5, \quad \mu_y = \frac{1 + 5 + 3 + 7}{4} = 4.0 \implies \boldsymbol{\mu} = \begin{bmatrix} 3.5 \\ 4.0 \end{bmatrix}
$$

### 2. Subtract Mean to get $\tilde{X}$:
$$
\tilde{\mathbf{x}}_1 = (2 - 3.5, 1 - 4.0) = (-1.5, -3.0)
$$
$$
\tilde{\mathbf{x}}_2 = (3 - 3.5, 5 - 4.0) = (-0.5, +1.0)
$$
$$
\tilde{\mathbf{x}}_3 = (4 - 3.5, 3 - 4.0) = (+0.5, -1.0)
$$
$$
\tilde{\mathbf{x}}_4 = (5 - 3.5, 7 - 4.0) = (+1.5, +3.0)
$$

### 3. Compute Covariance Matrix $C$ ($m - 1 = 3$):
$$
\sum \tilde{x}^2 = (-1.5)^2 + (-0.5)^2 + 0.5^2 + 1.5^2 = 2.25 + 0.25 + 0.25 + 2.25 = 5.0
$$
$$
\sum \tilde{y}^2 = (-3.0)^2 + 1.0^2 + (-1.0)^2 + 3.0^2 = 9.0 + 1.0 + 1.0 + 9.0 = 20.0
$$
$$
\sum \tilde{x}\tilde{y} = (-1.5)(-3.0) + (-0.5)(1.0) + (0.5)(-1.0) + (1.5)(3.0) = 4.5 - 0.5 - 0.5 + 4.5 = 8.0
$$
$$
C = \frac{1}{3} \begin{bmatrix} 5.0 & 8.0 \\ 8.0 & 20.0 \end{bmatrix} = \begin{bmatrix} 1.6667 & 2.6667 \\ 2.6667 & 6.6667 \end{bmatrix}
$$

### 4. Find Eigenvalues:
Solve $\det(C - \lambda I) = 0$:
$$
(1.6667 - \lambda)(6.6667 - \lambda) - (2.6667)^2 = 0
$$
$$
\lambda^2 - 8.3333\lambda + 4.000 = 0
$$
Using the quadratic formula:
$$
\lambda_1 = \mathbf{7.8220}, \quad \lambda_2 = \mathbf{0.5114}
$$
- Total variance $= \operatorname{tr}(C) = 1.6667 + 6.6667 = 8.3333 = 7.8220 + 0.5114$ ✅

### 5. Find Eigenvectors:
Substitute $\lambda_1 = 7.8220$ into $(C - \lambda I)\mathbf{v} = 0$:
$$
(1.6667 - 7.8220)v_x + 2.6667v_y = 0 \implies -6.1553v_x + 2.6667v_y = 0 \implies v_y \approx 2.3082 v_x
$$
Normalize to unit length ($\sqrt{1^2 + 2.3082^2} \approx 2.5155$):
$$
\mathbf{v}_1 = \begin{bmatrix} 1 / 2.5155 \\ 2.3082 / 2.5155 \end{bmatrix} = \begin{bmatrix} \mathbf{0.3975} \\ \mathbf{0.9176} \end{bmatrix}
$$
Since $\mathbf{v}_2 \perp \mathbf{v}_1$:
$$
\mathbf{v}_2 = \begin{bmatrix} -\mathbf{0.9176} \\ \mathbf{0.3975} \end{bmatrix}
$$

### 6. Explained Variance Ratio:
$$
\text{EVR}_1 = \frac{7.8220}{8.3333} = \mathbf{93.86\%}, \quad \text{EVR}_2 = \frac{0.5114}{8.3333} = \mathbf{6.14\%}
$$
A single line (PC1) captures **93.86%** of all the variance in this data!

### 7. Project onto PC1 ($k = 1$):
Multiply centered points by $\mathbf{v}_1$ ($z = 0.3975 \tilde{x} + 0.9176 \tilde{y}$):
- Point 1: $0.3975(-1.5) + 0.9176(-3.0) = \mathbf{-3.3491}$
- Point 2: $0.3975(-0.5) + 0.9176(+1.0) = \mathbf{+0.7189}$
- Point 3: $0.3975(+0.5) + 0.9176(-1.0) = \mathbf{-0.7189}$
- Point 4: $0.3975(+1.5) + 0.9176(+3.0) = \mathbf{+3.3491}$

- Check variance of $z$: $\frac{(-3.3491)^2 + 0.7189^2 + (-0.7189)^2 + 3.3491^2}{3} = \mathbf{7.8220} = \lambda_1$! ✅
- Reconstruction MSE from dropping PC2: $\mathbf{0.5114} = \lambda_2$! ✅

*(All these numbers match Experiment E11 in our test suite!)*

---

## 4. Datasets Used and Why We Chose Them

Every dataset in our project was chosen with a distinct educational and mathematical purpose:

| Dataset | Dimensions ($n$) | Samples ($m$) | Why We Chose It |
|---|---|---|---|
| **Toy 2D Data** | 2 | 4 | **Pencil-and-paper verification:** Allows anyone to calculate every number by hand to prove the code is mathematically sound. |
| **Scikit-Learn Digits** | 64 ($8 \times 8$) | 1,797 | **Fast offline testing:** Small enough that our 25 automated pytest tests finish in $< 3.5$ seconds with zero internet connection required. |
| **MNIST 784** | 784 ($28 \times 28$) | 70,000 | **Primary Machine Learning benchmark:** Real-world image data showing that 784 dimensions can be compressed to ~154 dimensions ($5.2\times$ compression) while keeping $>97\%$ accuracy. |
| **Olivetti Faces** | 4,096 ($64 \times 64$) | 400 | **High-dimensional SVD demo ($n \gg m$):** Classic computer vision benchmark (Turk & Pentland, 1991). Features (4,096 pixels) vastly outnumber samples (400 photos). Produces human-interpretable "ghostly" eigenface components. |
| **Wine Dataset** | 13 | 178 | **Standardization study:** 13 chemical features with wildly different units. Proline variance ($\approx 98,610$) dwarfs alcohol ($\approx 0.65$), proving why scaling is required for tabular data. |
| **Iris Dataset** | 4 | 150 | **Limitation demonstration:** Shows that PCA is unsupervised. The direction with the largest variance does not always separate classes best. |

---

## 5. The 14 Empirical Experiments Explained in Plain English

Here is what happens in each experiment in the Jupyter notebook:

- **E1: Scree Plot (`scree_plot.png`)**
  - *What we do:* Plot the eigenvalues $\lambda_1, \lambda_2, \dots, \lambda_{784}$ from largest to smallest.
  - *What it shows:* An exponential drop-off with a sharp "elbow" around $k \approx 30\text{--}50$. The first few components carry immense information, while trailing components contain negligible variance.
- **E2: Cumulative Explained Variance (`cumulative_variance.png`)**
  - *What we do:* Plot the running sum $\sum_{i=1}^k \lambda_i / \sum \lambda_j$ and draw horizontal threshold lines at 90%, 95%, and 99%.
  - *What it shows:* On MNIST, 90% variance is achieved at $k \approx 87$, 95% at $k \approx 154$, and 99% at $k \approx 331$.
- **E3: Classifier Accuracy vs. $k$ (`accuracy_vs_k.png`)**
  - *What we do:* Train three different classifiers (Logistic Regression, $k\text{NN}$, and SVM) across different values of $k$ ($2, 5, 10, 20, 30, 50, 100, 200, 784$).
  - *What it shows:* Below $k=10$, accuracy collapses. But by $k \approx 50\text{--}100$, accuracy reaches the full 784-dimensional baseline! You do not need all 784 pixels to recognize digits accurately.
- **E4: Computational Speed & Memory Benchmark (`timing_comparison.png`)**
  - *What we do:* Time how fast classifiers train and predict on raw 784D images vs. 154D PCA-reduced images.
  - *What it shows:* $k\text{NN}$ inference speeds up dramatically because Euclidean distance checks require $5.2\times$ fewer arithmetic operations, while memory is cut by $5.2\times$.
- **E5: Reconstruction Gallery (`reconstruction_gallery.png`)**
  - *What we do:* Compress digits to $k = 5, 20, 50, 150$ and reconstruct them back into $28 \times 28$ images.
  - *What it shows:* At $k=5$, digits look like blurry smudges. At $k=20$, the digit identity is clear. At $k=150$, the reconstruction is virtually indistinguishable from the original.
- **E6: Reconstruction Error Identity (`reconstruction_error_vs_k.png`)**
  - *What we do:* Calculate empirical pixel error and overlay it on the theoretical sum of discarded eigenvalues.
  - *What it shows:* The curves lie directly on top of each other, confirming the mathematical proof.
- **E7: 2D Latent Projection (`2d_projection.png`)**
  - *What we do:* Compress 784D digits down to just 2 dimensions (PC1 and PC2) and create a scatter plot colored by digit class ($0\text{--}9$).
  - *What it shows:* Different digits naturally separate into distinct spatial clusters, proving PCA extracts meaningful patterns without ever looking at the labels!
- **E8: Eigendigits (`eigendigits.png`)**
  - *What we do:* Take the top-16 eigenvectors (each length 784) and reshape them into $28 \times 28$ images.
  - *What it shows:* They look like basic handwriting strokes: vertical bars (like digit 1), central loops (like 0 and 8), and diagonal slashes.
- **E9: Eigenfaces (`eigenfaces.png`, `face_reconstruction.png`)**
  - *What we do:* Compute PCA on the Olivetti face dataset.
  - *What it shows:* The "mean face" looks like a generic human template. Higher eigenfaces capture specific lighting directions, eyebrow thickness, and jaw shapes.
- **E10: Standardization Study on Wine (`standardization_effect.png`)**
  - *What we do:* Run PCA on raw vs. standardized Wine data.
  - *What it shows:* Proline dominates 99.81% of raw PC1. When standardized, PC1 drops to 36.20%, giving all 13 chemical features a fair voice.
- **E11: Validation Suite (`results/validation_tests.csv`)**
  - *What we do:* Run 8 automated numerical checks comparing our from-scratch code against Scikit-Learn.
  - *What it shows:* All 8 tests pass with 100% True values.
- **E12: Eigendecomposition vs. SVD Benchmark (`results/eigh_vs_svd.csv`)**
  - *What we do:* Compare execution time and eigenvalue precision between `eigh` and `svd`.
  - *What it shows:* Both produce identical eigenvalues up to machine precision ($10^{-14}$).
- **E13: PCA Denoising (`figures/denoising.png`)**
  - *What we do:* Add random Gaussian static noise to clean digits, project them to $k=50$, and reconstruct them.
  - *What it shows:* Because random noise is uncorrelated across pixels, it gets pushed into the tiny trailing eigenvalues. Truncating to top-$k$ removes the noise automatically!
- **E14: Unsupervised Nature Limitation (`figures/unsupervised_limitation.png`)**
  - *What we do:* Using the Iris dataset, plot a lower-variance component (like PC2) that separates biological classes better than PC1.
  - *What it shows:* Proves that PCA maximizes overall variance, not class separability (motivating supervised techniques like Linear Discriminant Analysis).

---

## 6. Code Architecture: How the Repository Works

### 1. `pca_scratch.py`:
Contains two clean classes written with pure NumPy:
- `PCAScratch`: Implements the covariance eigendecomposition pipeline (`eigh`).
- `PCAScratchSVD`: Implements the Singular Value Decomposition pipeline (`svd`).
- Methods implemented:
  - `.fit(X)`: Computes mean $\boldsymbol{\mu}$, centers data, calculates covariance, extracts and sorts eigenpairs.
  - `.transform(X)`: Projects data via $(X - \boldsymbol{\mu}) Q_k$.
  - `.inverse_transform(Z)`: Decompresses via $Z Q_k^T + \boldsymbol{\mu}$.
  - `.fit_transform(X)`: Performs fit and transform in one call.

### 2. `test_pca.py`:
Contains **25 unit tests** run with pytest:
- Validates eigenvalues, explained variance ratios, eigenvector dot products, transform coordinates, orthonormality ($Q^T Q = I$), error identities, and the 4-point toy calculations.

### 3. `pca_project.ipynb`:
A self-contained notebook containing all 18 sections and 14 experiments. Designed to run seamlessly in the cloud on Kaggle or locally in Jupyter.

---

## 7. Presentation Script & Viva Voce Defense Playbook

### 7.1 30-Second Elevator Pitch
> *"In this project, we implemented Principal Component Analysis from mathematical first principles using linear algebra. By computing the sample covariance matrix and solving its eigenvalue problem, we proved that the directions of maximum data variance are precisely the eigenvectors of the covariance matrix. We verified our from-scratch NumPy implementation against scikit-learn across 25 unit tests to machine precision. On the 784-dimensional MNIST benchmark, we demonstrated that retaining 95% of the variance requires only ~154 dimensions—a 5.2× compression—preserving classification accuracy above 97% while speeding up kNN distance calculations. Finally, using the Wine dataset, we proved why feature standardization is essential when working with mixed physical units."*

---

### 7.2 2-Minute Project Walkthrough
1. **The Core Problem**: High-dimensional datasets, like handwritten digit images, contain heavy multicollinearity, waste memory, and slow down distance-dependent algorithms like $k$-Nearest Neighbors.
2. **The Mathematical Solution**: Rather than using black-box libraries, we explored how PCA finds a new orthogonal coordinate system where features are completely decorrelated. We showed that subtracting the mean, computing the covariance matrix $C = \frac{1}{m-1} \tilde{X}^T \tilde{X}$, and solving $C\mathbf{v} = \lambda\mathbf{v}$ yields eigenvectors that point along maximal variance axes.
3. **The Engineering**: We built both covariance-based and SVD-based PCA engines in pure NumPy. We proved our code's accuracy with 25 automated pytest tests matching Scikit-Learn to $10^{-8}$ precision.
4. **Key Experimental Results**:
   - **Compression**: MNIST can be reduced from 784 to 154 dimensions ($5.2\times$ compression) with negligible loss of accuracy.
   - **Theoretical Validation**: The empirical reconstruction error matches the sum of discarded eigenvalues ($\sum_{i > k} \lambda_i$) exactly.
   - **Feature Scaling**: On the Wine dataset, we demonstrated that unstandardized data allows high-variance features like Proline to monopolize 99.8% of PC1, proving why $z$-scoring is mandatory for multi-unit data.
   - **Denoising**: We showed how PCA acts as a natural noise filter by discarding small trailing eigenvalues.

---

### 7.3 Slide-by-Slide Presentation Structure

| Slide # | Slide Title | What to Show | Spoken Talking Point |
|---|---|---|---|
| **1** | Title & Overview | Title, team members, course | *"Our project implements PCA from linear algebra first principles, evaluating its effect on machine learning classifiers."* |
| **2** | Motivation | $28 \times 28$ image $\to$ 784D vector | *"A digit is a single point in 784D space. But neighboring pixels are correlated. We want an optimal orthogonal basis that removes redundancy."* |
| **3** | The Math Pipeline | Mean $\to$ Covariance $\to$ Eigenpairs | *"We center the data, compute the covariance matrix C, and solve the eigenvalue problem. The Spectral Theorem guarantees orthogonal axes."* |
| **4** | Hand-Worked 2D Example | Figure: `toy_example.png` | *"Here are 4 points calculated by hand. PC1 captures 93.86% of the variance, and the reconstruction error exactly matches eigenvalue $\lambda_2$."* |
| **5** | Code & 25 Unit Tests | Table of passed tests | *"We built PCAScratch in NumPy. All 25 unit tests pass, matching Scikit-Learn to machine precision."* |
| **6** | Scree Plot & Choosing $k$ | Figures: `scree_plot.png`, `cumulative_variance.png` | *"Eigenvalues decay exponentially. On MNIST, 95% variance is reached at $k \approx 154$—a $5.2\times$ reduction."* |
| **7** | Accuracy vs. $k$ | Figure: `accuracy_vs_k.png` | *"Classifiers reach full-dimension accuracy at $k \approx 50\text{--}100$. Below $k=10$, accuracy drops sharply because essential geometry is lost."* |
| **8** | Speedup & Memory | Figure: `timing_comparison.png` | *"kNN inference is significantly faster because distance checks require $5.2\times$ fewer operations, with $5.2\times$ less RAM."* |
| **9** | Reconstructions & Denoising | Figures: `reconstruction_gallery.png`, `denoising.png` | *"Visual proof: $k=150$ reconstructs crisp digits. Dropping small eigenvalues automatically filters out random static noise."* |
| **10** | Eigenfaces (Olivetti) | Figures: `eigenfaces.png`, `face_reconstruction.png` | *"In faces, $n=4096 \gg m=400$. We use SVD to compute facial components like lighting, eyes, and jawlines."* |
| **11** | Standardization (Wine) | Figure: `standardization_effect.png` | *"In the Wine dataset, Proline variance is 98,610 while Alcohol is 0.65. Without scaling, Proline takes 99.8% of PC1. Standardizing fixes this."* |
| **12** | Conclusion & Limitations | Bulleted summary | *"PCA is powerful, linear, and unsupervised. We verified its theory, proved its speedups, and highlighted its boundaries."* |

---

### 7.4 Top 15 Viva Questions with Model Answers

#### Q1: "What does PCA actually do in simple terms?"
> **Answer:** PCA performs a linear change of basis. It rotates the coordinate axes of the dataset so that the first axis aligns with the direction of greatest variance, the second axis aligns with the next greatest variance perpendicular to the first, and so on. We can then drop the low-variance axes to compress the data with minimal information loss.

#### Q2: "Why do we subtract the mean before computing PCA?"
> **Answer:** Covariance is defined around the center of mass of the data: $\mathbb{E}[(X - \mu_X)(Y - \mu_Y)]$. If data is not mean-centered, the first principal component will point from the coordinate origin $(0, 0, \dots, 0)$ to the mean center of the data points, capturing the offset location rather than the internal shape or variance of the data.

#### Q3: "What do the eigenvalues and eigenvectors represent?"
> **Answer:** The **eigenvectors** are the new coordinate directions (the principal components), and the **eigenvalues** represent the exact sample variance of the data along each respective eigenvector.

#### Q4: "Why are the principal components always perpendicular (orthogonal)?"
> **Answer:** Because the sample covariance matrix $C = \frac{1}{m-1} \tilde{X}^T \tilde{X}$ is a real symmetric matrix ($C = C^T$). By the Spectral Theorem in linear algebra, any real symmetric matrix has an orthonormal basis of eigenvectors.

#### Q5: "What does it mean that PCA decorrelates the features?"
> **Answer:** When we project data into the principal component space ($Z = \tilde{X} Q$), the covariance matrix of $Z$ is diagonal ($\operatorname{Cov}(Z) = \Lambda$). All off-diagonal entries are zero, meaning every principal component has zero correlation with every other component.

#### Q6: "How do you choose how many components ($k$) to keep?"
> **Answer:** We use the cumulative explained variance ratio ($\sum_{i=1}^k \lambda_i / \sum_{j=1}^n \lambda_j$) and choose the smallest $k$ that retains a target percentage of information, such as 90% or 95%. Alternatively, we can locate the "elbow" on a scree plot or evaluate cross-validation accuracy.

#### Q7: "What is the mathematical connection between PCA and SVD?"
> **Answer:** When we compute the SVD of centered data $\tilde{X} = U \Sigma V^T$, the right singular vectors $V$ are identical to the PCA eigenvectors $Q$, and the singular values $\sigma_i$ relate to eigenvalues by $\lambda_i = \frac{\sigma_i^2}{m - 1}$. SVD is numerically cleaner because it avoids computing $\tilde{X}^T \tilde{X}$ directly, preventing the condition number from squaring.

#### Q8: "Why does PCA speed up k-Nearest Neighbors (kNN)?"
> **Answer:** $k\text{NN}$ computes Euclidean distance between query points and training points, which takes $O(n)$ operations per comparison. By reducing $n = 784$ to $k = 154$, the distance calculation requires $5.2\times$ fewer operations, leading to an immediate inference speedup.

#### Q9: "What is the reconstruction error equal to?"
> **Answer:** The average squared reconstruction error from keeping $k$ components is mathematically equal to the sum of the discarded eigenvalues: $\text{MSE} = \sum_{i = k+1}^n \lambda_i$.

#### Q10: "Why should you NOT standardize MNIST images?"
> **Answer:** In MNIST, all features are already in the identical unit (pixel intensities from $0$ to $255$). Border pixels are almost always black, meaning their standard deviation is near zero. If you standardize, you divide by near-zero variance, which blows up sensor noise.

#### Q11: "Why DID you standardize the Wine dataset?"
> **Answer:** The Wine dataset contains 13 features with wildly different units and scales. Proline has values over 1,000 ($\text{variance} \approx 98,610$), while Alcohol is around 13 ($\text{variance} \approx 0.65$). Without standardization, Proline accounts for 99.81% of PC1 on its own. Standardizing puts all features on an equal footing ($\text{variance} = 1$).

#### Q12: "Why might your eigenvector signs differ from scikit-learn?"
> **Answer:** Eigenvectors are unique only up to a sign flip: if $C\mathbf{v} = \lambda\mathbf{v}$, then $C(-\mathbf{v}) = \lambda(-\mathbf{v})$. Both $+\mathbf{v}$ and $-\mathbf{v}$ define the exact same line in space. Scikit-Learn applies an arbitrary sign convention (`svd_flip`), so signs may differ, but the subspace and projections are identical.

#### Q13: "How does PCA perform noise reduction (denoising)?"
> **Answer:** True signal tends to be correlated across many pixels, producing large eigenvalues. Random noise (like Gaussian static) is uncorrelated, so its variance is spread thinly across all dimensions into small trailing eigenvalues. Truncating to the top-$k$ components drops the small eigenvalues, discarding the noise while keeping the digit structure.

#### Q14: "Why is PCA called an unsupervised technique?"
> **Answer:** Because it only looks at the feature matrix $X$ and never looks at class labels $y$. It finds directions that maximize overall variance, regardless of whether those directions separate different classes.

#### Q15: "What are the core limitations of PCA?"
> **Answer:** 
> 1. It is strictly **linear** (it cannot unroll curved manifolds like a Swiss roll; Kernel PCA or autoencoders are required).
> 2. It is **sensitive to outliers** because variance squares large deviations.
> 3. Its components are linear mixtures of all original features, making them harder to interpret than original measurements.

---

### 7.5 Common Traps & Pitfalls to Avoid

| Examiner Trap | Correct Response |
|---|---|
| Saying *"PCA fits a model to data"* | **Incorrect.** PCA does not "fit" or train a predictive model. It is an **unsupervised orthogonal coordinate transformation / projection**. |
| Fitting PCA before train/test split | **Data Leakage!** Never compute the mean $\boldsymbol{\mu}$ or covariance $C$ on the whole dataset before splitting. Always fit PCA on `X_train` only, then transform both `X_train` and `X_test`. |
| Using `np.linalg.eig` instead of `eigh` | `eig` is for general matrices and can return imaginary/complex numbers in arbitrary order. `eigh` is specialized for symmetric matrices, guaranteeing real eigenvalues and orthogonal eigenvectors. |
| Mixing up row vs. column eigenvectors | In NumPy `eigh`, eigenvectors are stored as **columns** ($Q[:, i]$ is $\mathbf{v}_i$). In Scikit-Learn, `pca.components_` stores them as **rows** (`components_[i, :]` is $\mathbf{v}_i$). |
| Dividing by $m$ instead of $m - 1$ | Dividing by $m$ is the biased MLE. Dividing by $m - 1$ is Bessel's correction, providing an unbiased sample covariance estimate that matches Scikit-Learn. |

---

*This document is the complete master reference for the Semester 3 Mathematics Mini Project on Principal Component Analysis.*
