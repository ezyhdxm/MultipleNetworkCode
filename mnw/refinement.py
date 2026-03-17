"""
Debiased refinement estimators for the shared low-rank structure.

After the node-level support of perturbations has been recovered, these
estimators refine the initial spectral estimate of the shared structure M*
by correcting for coherence-induced bias. The key idea is to use multiple
observations arranged asymmetrically to construct a bias-corrected eigenspace.

Reference: Sections 5 and Appendix of Yan & Levin (2025), arXiv:2506.15915.
"""

import numpy as np
from scipy.linalg import sqrtm

from mnw.spectral import rank_r_asymmetric


def asymmetric_arrange(M1: np.ndarray, M2: np.ndarray) -> np.ndarray:
    """
    Construct an asymmetric matrix by taking the upper triangle from M2
    and the lower triangle from M1.

    This breaks the symmetry of two independent observations of the same
    signal, enabling bias-corrected eigenspace estimation.

    Args:
        M1: First symmetric observation (n x n).
        M2: Second symmetric observation (n x n).

    Returns:
        Asymmetric matrix (n x n) with lower triangle from M1,
        upper triangle (including diagonal) from M2.
    """
    M = M2.copy()
    i_lower = np.tril_indices_from(M1, k=-1)
    M[i_lower] = M1[i_lower]
    return M


def debiased_eigenvectors(
    eigvals: np.ndarray,
    U: np.ndarray,
    M1: np.ndarray,
    M2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute bias-corrected eigenvectors and the refined low-rank estimate.

    Given the eigenvalues and right eigenvectors of an asymmetrically
    arranged matrix, uses two additional independent observations M1 and M2
    to compute a correction factor Psi_hat that removes the coherence bias.

    Args:
        eigvals: Leading eigenvalues from the asymmetric arrangement (r,).
        U: Corresponding right eigenvectors (n x r).
        M1: Independent observation for bias correction (n x n).
        M2: Independent observation for bias correction (n x n).

    Returns:
        U_hat: Bias-corrected eigenvectors (n x r).
        M_hat: Refined low-rank estimate (n x n).
    """
    evs_inv = np.diag(1.0 / eigvals)
    evs = np.diag(eigvals)

    G = np.linalg.inv(evs_inv @ U.T @ M1 @ M2 @ U @ evs_inv)
    G_sym = (G + G.T) / 2
    Psi_hat = sqrtm(G_sym)
    Psi_hat = np.real(Psi_hat)

    U_hat = U @ Psi_hat
    M_hat = U_hat @ evs @ U_hat.T
    return U_hat, M_hat


def debiased_estimate(
    observations: list[np.ndarray], rank: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the debiased low-rank estimate from four observations.

    Pairs observations into two asymmetric matrices, computes eigenvectors
    from the first, and applies bias correction using the second pair.

    The four observations are indexed as (M11, M12, M21, M22):
    - (M11, M12) are arranged asymmetrically to compute eigenvectors.
    - (M21, M22) provide the independent data for bias correction.

    Args:
        observations: List of exactly 4 symmetric n x n matrices.
        rank: Target rank r.

    Returns:
        U_hat: Bias-corrected eigenvectors (n x r).
        M_hat: Refined low-rank estimate (n x n).

    Raises:
        ValueError: If not exactly 4 observations are provided.
    """
    if len(observations) != 4:
        raise ValueError(
            f"Debiased estimation requires exactly 4 observations, got {len(observations)}."
        )

    M11, M12, M21, M22 = observations
    M_asym = asymmetric_arrange(M11, M12)
    U, eigvals = rank_r_asymmetric(M_asym, rank)
    return debiased_eigenvectors(eigvals, U, M21, M22)
