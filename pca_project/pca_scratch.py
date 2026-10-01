"""
From-scratch PCA implementation using eigendecomposition and SVD.

Two classes with unified API:
  - PCAScratch:    eigendecomposition of the covariance matrix (via np.linalg.eigh)
  - PCAScratchSVD: SVD of the centered data matrix (via np.linalg.svd)

Both produce identical results (up to eigenvector sign) and are validated
against sklearn.decomposition.PCA in test_pca.py.

Author: PCA Mini Project
"""

import numpy as np


class PCAScratch:
    """
    Principal Component Analysis via eigendecomposition of the covariance matrix.

    Parameters
    ----------
    n_components : int or None
        Number of principal components to keep. If None and var_threshold
        is also None, all components are kept.
    var_threshold : float or None
        Minimum cumulative explained variance ratio (e.g. 0.95).
        Overrides n_components if both are set.
    """

    def __init__(self, n_components=None, var_threshold=None):
        self.n_components = n_components
        self.var_threshold = var_threshold

    def fit(self, X):
        """
        Fit PCA on training data X.

        1. Compute mean vector μ
        2. Center data: X̃ = X − μ
        3. Covariance matrix: C = X̃ᵀX̃ / (m−1)
        4. Eigendecompose C (symmetric → eigh)
        5. Sort eigenvalues descending; select top-k components

        Parameters
        ----------
        X : ndarray of shape (m, n)
            Training data matrix (m samples, n features).

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        m, n = X.shape

        # Step 1–2: Mean and centering
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_

        # Step 3: Sample covariance matrix  C ∈ ℝ^(n×n)
        C = (Xc.T @ Xc) / (m - 1)

        # Step 4: Eigendecomposition (eigh returns ascending order, real values)
        eigvals, eigvecs = np.linalg.eigh(C)

        # Step 5: Sort descending
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Clip tiny negative eigenvalues (numerical noise) to zero
        eigvals = np.clip(eigvals, 0, None)

        # Store full spectrum
        self.eigvals_all_ = eigvals
        self.eigvecs_all_ = eigvecs  # columns are eigenvectors

        # Explained variance ratios
        total_var = eigvals.sum()
        self.evr_all_ = eigvals / total_var if total_var > 0 else np.zeros_like(eigvals)
        self.cum_evr_ = np.cumsum(self.evr_all_)

        # Determine number of components k
        if self.var_threshold is not None:
            k = int(np.searchsorted(self.cum_evr_, self.var_threshold) + 1)
            k = min(k, n)
        elif self.n_components is not None:
            k = self.n_components
        else:
            k = n

        self.k_ = k
        self.components_ = eigvecs[:, :k]          # n × k (columns = principal directions)
        self.explained_variance_ = eigvals[:k]
        self.explained_variance_ratio_ = self.evr_all_[:k]

        return self

    def transform(self, X):
        """
        Project X onto the top-k principal components.

        Z_k = (X − μ) Q_k

        Parameters
        ----------
        X : ndarray of shape (m, n)

        Returns
        -------
        Z : ndarray of shape (m, k)
        """
        return (np.asarray(X, dtype=np.float64) - self.mean_) @ self.components_

    def inverse_transform(self, Z):
        """
        Reconstruct from reduced representation.

        X̂ = Z Q_kᵀ + μ

        Parameters
        ----------
        Z : ndarray of shape (m, k)

        Returns
        -------
        X_hat : ndarray of shape (m, n)
        """
        return Z @ self.components_.T + self.mean_

    def fit_transform(self, X):
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


class PCAScratchSVD:
    """
    Principal Component Analysis via SVD of the centered data matrix.

    Avoids forming XᵀX explicitly (better numerical stability).
    Relation: X̃ = UΣVᵀ  ⇒  C = V(Σ²/(m−1))Vᵀ  ⇒  λᵢ = σᵢ²/(m−1)

    Parameters
    ----------
    n_components : int or None
        Number of principal components to keep.
    var_threshold : float or None
        Minimum cumulative explained variance ratio.
    """

    def __init__(self, n_components=None, var_threshold=None):
        self.n_components = n_components
        self.var_threshold = var_threshold

    def fit(self, X):
        """
        Fit PCA using SVD.

        Parameters
        ----------
        X : ndarray of shape (m, n)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        m, n = X.shape

        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_

        # SVD: X̃ = U Σ Vᵀ  (full_matrices=False → economy SVD)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

        # Eigenvalues of covariance: λᵢ = σᵢ² / (m−1)
        eigvals = S ** 2 / (m - 1)
        # Principal directions = rows of Vᵀ = columns of V
        eigvecs = Vt.T

        # Store full spectrum
        self.eigvals_all_ = eigvals
        self.eigvecs_all_ = eigvecs

        total_var = eigvals.sum()
        self.evr_all_ = eigvals / total_var if total_var > 0 else np.zeros_like(eigvals)
        self.cum_evr_ = np.cumsum(self.evr_all_)

        # Determine k
        if self.var_threshold is not None:
            k = int(np.searchsorted(self.cum_evr_, self.var_threshold) + 1)
            k = min(k, min(m, n))
        elif self.n_components is not None:
            k = self.n_components
        else:
            k = min(m, n)

        self.k_ = k
        self.components_ = eigvecs[:, :k]
        self.explained_variance_ = eigvals[:k]
        self.explained_variance_ratio_ = self.evr_all_[:k]

        return self

    def transform(self, X):
        """Project X onto the top-k principal components."""
        return (np.asarray(X, dtype=np.float64) - self.mean_) @ self.components_

    def inverse_transform(self, Z):
        """Reconstruct from reduced representation."""
        return Z @ self.components_.T + self.mean_

    def fit_transform(self, X):
        """Fit and transform in one step."""
        return self.fit(X).transform(X)
