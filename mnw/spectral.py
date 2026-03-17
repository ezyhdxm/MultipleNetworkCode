"""
Spectral methods for low-rank matrix estimation.

Provides rank-r approximation of symmetric and asymmetric matrices
via eigendecomposition and SVD.
"""

import warnings

import numpy as np
from scipy.sparse.linalg import eigsh, eigs


def rank_r_approximation(A: np.ndarray, r: int):
    """
    Compute the best rank-r approximation of a symmetric matrix.

    Uses the top-r eigenvalues (by magnitude) and their eigenvectors.

    Args:
        A: Symmetric n x n matrix.
        r: Target rank (1 <= r <= n).

    Returns:
        A_r: Rank-r approximation (n x n).
        eigvecs: Corresponding eigenvectors (n x r).
        eigvals: Top-r eigenvalues (r,).
    """
    n = A.shape[0]
    assert A.shape == (n, n), "Matrix must be square."
    assert 1 <= r <= n, f"Rank r={r} must be between 1 and n={n}."

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        eigvals, eigvecs = eigsh(A, k=r, which="LM")

    idx = np.argsort(-np.abs(eigvals))
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    A_r = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return A_r, eigvecs, eigvals


def rank_r_entrywise(A: np.ndarray, r: int) -> np.ndarray:
    """
    Compute the rank-r approximation using full eigendecomposition.

    Suitable for dense matrices where scipy.sparse.linalg.eigsh may be
    less accurate.

    Args:
        A: Symmetric n x n matrix.
        r: Target rank.

    Returns:
        A_r: Rank-r approximation (n x n).
    """
    eigvals, eigvecs = np.linalg.eigh(A)

    idx = np.argsort(-np.abs(eigvals))
    eigvals_r = eigvals[idx[:r]]
    eigvecs_r = eigvecs[:, idx[:r]]

    return eigvecs_r @ np.diag(eigvals_r) @ eigvecs_r.T


def rank_r_asymmetric(A: np.ndarray, r: int):
    """
    Compute the top-r right eigenvectors/eigenvalues of a general matrix.

    Used by the debiased refinement stage when working with asymmetrically
    arranged observation pairs.

    Args:
        A: Square n x n matrix (not necessarily symmetric).
        r: Number of leading eigencomponents.

    Returns:
        eigvecs: Right eigenvectors (n x r), real part.
        eigvals: Eigenvalues (r,), real part.
    """
    n = A.shape[0]
    assert A.shape == (n, n), "Matrix must be square."
    assert 1 <= r <= n, f"Rank r={r} must be between 1 and n={n}."

    eigvals, eigvecs = eigs(A, k=r, which="LM")
    return np.real(eigvecs), np.real(eigvals)
