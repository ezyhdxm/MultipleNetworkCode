"""
Utility functions for matrix norms, coherence, and input validation.
"""

import numpy as np
from math import log


def max_row_norm(A: np.ndarray) -> float:
    """Maximum row L2 norm of a matrix."""
    return np.max(np.linalg.norm(A, axis=1, ord=2))


def max_abs(A: np.ndarray) -> float:
    """Maximum absolute entry of a matrix."""
    return np.max(np.abs(A))


def coherence(U: np.ndarray) -> float:
    """
    Compute the coherence parameter mu of an orthonormal basis U.

    mu = (n / r) * ||U||_{2,inf}^2, where ||U||_{2,inf} is the max row norm.
    Coherence ranges from 1 (incoherent) to n/r (maximally coherent).

    Args:
        U: n x r matrix with orthonormal columns.

    Returns:
        Coherence parameter mu.
    """
    n, r = U.shape
    return (n / r) * max_row_norm(U) ** 2


def validate_matrices(matrices: list[np.ndarray], label: str = "input") -> int:
    """
    Validate a list of adjacency matrices: check square, symmetric, same size.

    Args:
        matrices: List of 2D numpy arrays.
        label: Label for error messages (e.g. 'control', 'treatment').

    Returns:
        n: Common matrix dimension.

    Raises:
        ValueError: If matrices are invalid.
    """
    if not matrices:
        raise ValueError(f"{label}: must provide at least one matrix.")

    n = matrices[0].shape[0]
    for i, M in enumerate(matrices):
        if M.ndim != 2:
            raise ValueError(f"{label}[{i}]: expected 2D array, got {M.ndim}D.")
        if M.shape[0] != M.shape[1]:
            raise ValueError(f"{label}[{i}]: expected square matrix, got shape {M.shape}.")
        if M.shape[0] != n:
            raise ValueError(
                f"{label}[{i}]: dimension mismatch. Expected {n}x{n}, got {M.shape[0]}x{M.shape[0]}."
            )
        if not np.allclose(M, M.T, atol=1e-8):
            raise ValueError(f"{label}[{i}]: matrix is not symmetric.")

    return n


def validate_inputs(
    Y_control: list[np.ndarray],
    Y_treatment: dict[str, list[np.ndarray]],
) -> int:
    """
    Validate all input matrices for the pipeline.

    Returns:
        n: Common matrix dimension across all inputs.

    Raises:
        ValueError: If inputs are inconsistent.
    """
    n = validate_matrices(Y_control, label="Y_control")

    if not Y_treatment:
        raise ValueError("Y_treatment must contain at least one group.")

    for group_id, matrices in Y_treatment.items():
        n_group = validate_matrices(matrices, label=f"Y_treatment['{group_id}']")
        if n_group != n:
            raise ValueError(
                f"Y_treatment['{group_id}']: dimension {n_group} does not match "
                f"control group dimension {n}."
            )

    return n
