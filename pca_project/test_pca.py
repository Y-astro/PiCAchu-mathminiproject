"""
Validation tests for PCAScratch and PCAScratchSVD against sklearn.decomposition.PCA.

Tests correspond to Section 5.3 of the project plan.
Run with:  python -m pytest test_pca.py -v
"""

import numpy as np
import pytest
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA as SklearnPCA

from pca_scratch import PCAScratch, PCAScratchSVD


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def digits_data():
    """Load sklearn digits dataset and split train/test."""
    data = load_digits()
    X, y = data.data, data.target
    # Simple deterministic split: first 80% train, last 20% test
    n_train = int(0.8 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    return X_train, X_test


@pytest.fixture(scope="module", params=[10, 30, 50])
def k(request):
    """Number of components to test."""
    return request.param


@pytest.fixture(scope="module")
def fitted_models(digits_data):
    """Fit both scratch and sklearn PCA for multiple k values."""
    X_train, _ = digits_data
    models = {}
    for k in [10, 30, 50]:
        sk = SklearnPCA(n_components=k).fit(X_train)
        mine = PCAScratch(n_components=k).fit(X_train)
        svd = PCAScratchSVD(n_components=k).fit(X_train)
        models[k] = {"sklearn": sk, "scratch": mine, "svd": svd}
    return models


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEigenvalueMatch:
    """Test 1: Eigenvalues match sklearn's explained_variance_."""

    def test_eigenvalues_k10(self, fitted_models):
        m = fitted_models[10]
        assert np.allclose(
            m["scratch"].explained_variance_, m["sklearn"].explained_variance_
        ), "Eigenvalues do not match sklearn for k=10"

    def test_eigenvalues_k30(self, fitted_models):
        m = fitted_models[30]
        assert np.allclose(
            m["scratch"].explained_variance_, m["sklearn"].explained_variance_
        ), "Eigenvalues do not match sklearn for k=30"

    def test_eigenvalues_k50(self, fitted_models):
        m = fitted_models[50]
        assert np.allclose(
            m["scratch"].explained_variance_, m["sklearn"].explained_variance_
        ), "Eigenvalues do not match sklearn for k=50"


class TestEVRMatch:
    """Test 2: Explained variance ratios match."""

    def test_evr_k10(self, fitted_models):
        m = fitted_models[10]
        assert np.allclose(
            m["scratch"].explained_variance_ratio_,
            m["sklearn"].explained_variance_ratio_,
        )

    def test_evr_k30(self, fitted_models):
        m = fitted_models[30]
        assert np.allclose(
            m["scratch"].explained_variance_ratio_,
            m["sklearn"].explained_variance_ratio_,
        )


class TestEigenvectorMatch:
    """Test 3: Eigenvectors match up to sign (|dot| ≈ 1)."""

    def test_eigenvectors_k10(self, fitted_models):
        m = fitted_models[10]
        # sklearn stores components as rows; scratch stores as columns
        dots = np.abs(np.sum(m["scratch"].components_.T * m["sklearn"].components_, axis=1))
        assert np.allclose(dots, 1.0, atol=1e-6), (
            f"Eigenvector dot products deviate from 1: {dots}"
        )

    def test_eigenvectors_k30(self, fitted_models):
        m = fitted_models[30]
        dots = np.abs(np.sum(m["scratch"].components_.T * m["sklearn"].components_, axis=1))
        assert np.allclose(dots, 1.0, atol=1e-6)


class TestTransformMatch:
    """Test 4: Transform results match up to per-column sign."""

    def test_transform_k10(self, fitted_models, digits_data):
        _, X_test = digits_data
        m = fitted_models[10]
        Z_mine = m["scratch"].transform(X_test)
        Z_sk = m["sklearn"].transform(X_test)
        assert np.allclose(np.abs(Z_mine), np.abs(Z_sk), atol=1e-8)

    def test_transform_k30(self, fitted_models, digits_data):
        _, X_test = digits_data
        m = fitted_models[30]
        Z_mine = m["scratch"].transform(X_test)
        Z_sk = m["sklearn"].transform(X_test)
        assert np.allclose(np.abs(Z_mine), np.abs(Z_sk), atol=1e-8)


class TestOrthonormality:
    """Test 5: Principal directions are orthonormal (Q_kᵀ Q_k = I_k)."""

    def test_orthonormal_k10(self, fitted_models):
        Q = fitted_models[10]["scratch"].components_
        assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=1e-10)

    def test_orthonormal_k30(self, fitted_models):
        Q = fitted_models[30]["scratch"].components_
        assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=1e-10)

    def test_orthonormal_k50(self, fitted_models):
        Q = fitted_models[50]["scratch"].components_
        assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=1e-10)


class TestReconstructionErrorIdentity:
    """Test 6: Reconstruction error = sum of discarded eigenvalues."""

    def test_recon_error_k10(self, fitted_models, digits_data):
        X_train, _ = digits_data
        m = fitted_models[10]["scratch"]
        Xc = X_train - m.mean_
        P = m.components_ @ m.components_.T  # projection matrix
        err = np.sum((Xc - Xc @ P) ** 2) / (len(X_train) - 1)
        expected = m.eigvals_all_[10:].sum()
        assert np.isclose(err, expected, rtol=1e-6), (
            f"Reconstruction error {err:.6f} ≠ discarded eigenvalue sum {expected:.6f}"
        )

    def test_recon_error_k30(self, fitted_models, digits_data):
        X_train, _ = digits_data
        m = fitted_models[30]["scratch"]
        Xc = X_train - m.mean_
        P = m.components_ @ m.components_.T
        err = np.sum((Xc - Xc @ P) ** 2) / (len(X_train) - 1)
        expected = m.eigvals_all_[30:].sum()
        assert np.isclose(err, expected, rtol=1e-6)


class TestSVDEquivalence:
    """Test 7: Eigenvalues from eigh ≈ eigenvalues from SVD."""

    def test_svd_eigvals_k10(self, fitted_models):
        eigh_vals = fitted_models[10]["scratch"].eigvals_all_
        svd_vals = fitted_models[10]["svd"].eigvals_all_
        # SVD may have fewer values (min(m,n)), compare the overlap
        n = min(len(eigh_vals), len(svd_vals))
        assert np.allclose(eigh_vals[:n], svd_vals[:n], atol=1e-8)

    def test_svd_eigvals_k50(self, fitted_models):
        eigh_vals = fitted_models[50]["scratch"].eigvals_all_
        svd_vals = fitted_models[50]["svd"].eigvals_all_
        n = min(len(eigh_vals), len(svd_vals))
        assert np.allclose(eigh_vals[:n], svd_vals[:n], atol=1e-8)


class TestDecorrelation:
    """Test 8: Transformed data has diagonal covariance (PCs uncorrelated)."""

    def test_decorrelation_k10(self, fitted_models, digits_data):
        X_train, _ = digits_data
        m = fitted_models[10]["scratch"]
        Z = m.transform(X_train)
        cov_Z = np.cov(Z.T)
        expected = np.diag(m.explained_variance_)
        assert np.allclose(cov_Z, expected, atol=1e-8), (
            "Covariance of transformed data is not diagonal"
        )

    def test_decorrelation_k30(self, fitted_models, digits_data):
        X_train, _ = digits_data
        m = fitted_models[30]["scratch"]
        Z = m.transform(X_train)
        cov_Z = np.cov(Z.T)
        expected = np.diag(m.explained_variance_)
        assert np.allclose(cov_Z, expected, atol=1e-8)


class TestVarThreshold:
    """Test variance threshold selection."""

    def test_threshold_95(self, digits_data):
        X_train, _ = digits_data
        m = PCAScratch(var_threshold=0.95).fit(X_train)
        assert m.cum_evr_[m.k_ - 1] >= 0.95
        if m.k_ > 1:
            assert m.cum_evr_[m.k_ - 2] < 0.95

    def test_threshold_99(self, digits_data):
        X_train, _ = digits_data
        m = PCAScratch(var_threshold=0.99).fit(X_train)
        assert m.cum_evr_[m.k_ - 1] >= 0.99


class TestInverseTransform:
    """Test that inverse_transform reverses transform (for full rank)."""

    def test_full_rank_roundtrip(self, digits_data):
        X_train, _ = digits_data
        m = PCAScratch().fit(X_train)  # all components
        Z = m.transform(X_train)
        X_recon = m.inverse_transform(Z)
        assert np.allclose(X_train, X_recon, atol=1e-8)


class TestToyExample:
    """Verify the hand-worked 4-point example from Section 4 of the plan."""

    def test_toy_eigenvalues(self):
        X = np.array([[2, 1], [3, 5], [4, 3], [5, 7]], dtype=float)
        m = PCAScratch().fit(X)

        # Expected eigenvalues (Section 4): λ₁ ≈ 7.8220, λ₂ ≈ 0.5114
        assert np.isclose(m.explained_variance_[0], 7.8220, atol=0.001)
        assert np.isclose(m.explained_variance_[1], 0.5114, atol=0.001)

    def test_toy_explained_variance_ratio(self):
        X = np.array([[2, 1], [3, 5], [4, 3], [5, 7]], dtype=float)
        m = PCAScratch().fit(X)

        # PC1 should explain ~93.86%
        assert np.isclose(m.evr_all_[0], 0.9386, atol=0.001)

    def test_toy_total_variance(self):
        X = np.array([[2, 1], [3, 5], [4, 3], [5, 7]], dtype=float)
        m = PCAScratch().fit(X)

        # Total variance = 25/3 ≈ 8.3333
        assert np.isclose(m.eigvals_all_.sum(), 25.0 / 3, atol=0.001)

    def test_toy_eigenvector_direction(self):
        X = np.array([[2, 1], [3, 5], [4, 3], [5, 7]], dtype=float)
        m = PCAScratch().fit(X)

        # v₁ ≈ (0.3975, 0.9176) up to sign
        v1 = m.components_[:, 0]
        assert np.isclose(abs(v1[0]), 0.3975, atol=0.001)
        assert np.isclose(abs(v1[1]), 0.9176, atol=0.001)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
