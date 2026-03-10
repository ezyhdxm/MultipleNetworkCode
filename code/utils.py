import numpy as np
from math import sqrt, log
import matplotlib.pyplot as plt
import seaborn as sns




def max_row_norm(A):
    """
    Compute the maximum row norm of a matrix A.

    Parameters:
    A (ndarray): Input matrix (m x n).

    Returns:
    float: Maximum row ℓ₂-norm.
    """
    row_norms = np.linalg.norm(A, axis=1, ord=2)  # Compute ℓ₂ norm for each row
    return np.max(row_norms)  # Return the maximum row norm

def max_abs(A):
    """
    Compute the maximum absolute value in a matrix A.

    Parameters:
    A (ndarray): Input matrix (m x n).

    Returns:
    float: Maximum absolute value in the matrix.
    """
    return np.max(np.abs(A))  # Return the maximum absolute value in the matrix

def coh_rate(A):
    n = A.shape[0]  # Number of rows
    return log(n * max_row_norm(A)**2) / log(n)  # Compute coherence



def tti_approx(Uhat, Ustar):
    H = Uhat.T @ Ustar
    R, _, Vt = np.linalg.svd(H, full_matrices=False)
    P = R @ Vt
    return max_row_norm(Uhat @ P - Ustar)