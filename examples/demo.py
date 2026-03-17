#!/usr/bin/env python3
"""
Demo: Multiple Weighted Network Analysis on Synthetic Data.

Generates synthetic networks following the model from Yan & Levin (2026),
runs the full analysis pipeline, and reports the results.

This script validates the pipeline and serves as a usage example.

Usage:
    python examples/demo.py
    python examples/demo.py --n 500 --rank 3 --support-size 8
"""

import argparse
import sys
import os
import warnings

import numpy as np
from math import sqrt, log

warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mnw import MultipleNetworkPipeline


# ---------------------------------------------------------------------------
# Synthetic data generation (inline, not part of the mnw package)
# ---------------------------------------------------------------------------

def generate_orthonormal_basis(n: int, r: int) -> np.ndarray:
    """Generate r orthonormal vectors in R^n via QR decomposition."""
    A = np.random.randn(n, r)
    Q, _ = np.linalg.qr(A)
    return Q[:, :r]


def generate_low_rank_signal(n: int, r: int, eigenvalues: np.ndarray) -> tuple:
    """Generate a symmetric rank-r signal matrix M* = U Lambda U^T."""
    U = generate_orthonormal_basis(n, r)
    M = U @ np.diag(eigenvalues) @ U.T
    return M, U


def generate_node_sparse_perturbation(
    n: int, m: int, scale: float = None
) -> tuple:
    """
    Generate a symmetric node-sparse perturbation matrix B*.

    Exactly m rows/columns are nonzero (the perturbed nodes).
    """
    if scale is None:
        scale = 2 * n ** (-0.25) * log(n) ** 0.25

    I_star = np.sort(np.random.choice(n, m, replace=False))
    B = np.zeros((n, n))
    B[I_star] = scale * np.random.randn(m, n)
    B = B + B.T
    return B, I_star


def generate_symmetric_noise(n: int, variance: float = 1.0) -> np.ndarray:
    """Generate a symmetric Gaussian noise matrix."""
    std = np.sqrt(variance)
    A = np.zeros((n, n))
    np.fill_diagonal(A, np.random.normal(0, std, size=n))
    upper = np.random.normal(0, std, size=(n, n))
    upper = np.triu(upper, k=1)
    return A + upper + upper.T


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def run_demo(n: int = 200, r: int = 3, m: int = 6, n_control: int = 2, n_treatment: int = 2):
    """Run the full pipeline on synthetic data and evaluate accuracy."""

    print("=" * 60)
    print("  Multiple Weighted Network Analysis -- Demo")
    print("=" * 60)
    print(f"\n  n={n}, rank={r}, perturbed nodes={m}")
    print(f"  {n_control} control matrices, {n_treatment} treatment matrices\n")

    np.random.seed(42)

    eigenvalues = 3 * sqrt(n) + 2 * log(n) * np.arange(r)
    M_star, U_star = generate_low_rank_signal(n, r, eigenvalues)
    B_star, I_star = generate_node_sparse_perturbation(n, m)

    print(f"  Ground truth perturbed nodes: {I_star.tolist()}")
    print(f"  Ground truth eigenvalues:     {np.array2string(eigenvalues, precision=1)}")

    Y_control = [M_star + generate_symmetric_noise(n) for _ in range(n_control)]
    Y_treatment_matrices = [
        M_star + B_star + generate_symmetric_noise(n)
        for _ in range(n_treatment)
    ]

    print("\n  Running pipeline (method=glasso)...")
    pipeline = MultipleNetworkPipeline(
        rank=r,
        support_size=m,
        support_method="glasso",
        verbose=False,
    )

    result = pipeline.fit(
        Y_control=Y_control,
        Y_treatment={"treatment": Y_treatment_matrices},
    )

    print("\n" + result.summary())

    I_hat = result.perturbed_nodes["treatment"]
    n_correct = np.sum(np.isin(I_hat, I_star))
    fnr = 1 - n_correct / m
    print(f"\n  --- Evaluation ---")
    print(f"  Detected nodes:  {np.sort(I_hat).tolist()}")
    print(f"  True nodes:      {I_star.tolist()}")
    print(f"  Correct:         {n_correct}/{m}")
    print(f"  False neg rate:  {fnr:.2f}")

    M_err = np.max(np.abs(result.M_hat - M_star))
    print(f"  M* L_inf error:  {M_err:.4f}")

    print("\n  Demo complete.\n")
    return result, fnr, M_err


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo: synthetic network analysis.")
    parser.add_argument("--n", type=int, default=200, help="Network size (default: 200).")
    parser.add_argument("--rank", type=int, default=3, help="Rank of shared structure.")
    parser.add_argument("--support-size", type=int, default=6, help="Number of perturbed nodes.")
    args = parser.parse_args()

    run_demo(n=args.n, r=args.rank, m=args.support_size)
