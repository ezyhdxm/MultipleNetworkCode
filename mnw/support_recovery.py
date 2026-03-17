"""
Node-level support recovery via SDP and Group Lasso.

Given a residual matrix Y_tilde (treatment observation minus estimated
shared structure), identifies the set of nodes whose connectivity
patterns differ from the baseline.
"""

import numpy as np
from math import sqrt, log


def _import_mosek():
    """Lazily import MOSEK Fusion. Raises ImportError with guidance if unavailable."""
    try:
        from mosek.fusion import Model, Domain, Expr, Matrix, ObjectiveSense
        return Model, Domain, Expr, Matrix, ObjectiveSense
    except ImportError:
        raise ImportError(
            "MOSEK is required for SDP-based support recovery. "
            "Install it with 'pip install Mosek' and obtain a license from "
            "https://www.mosek.com/ (free for academic use). "
            "Alternatively, use support_method='glasso' which has no "
            "commercial dependencies."
        )


# ---------------------------------------------------------------------------
# SDP-based support recovery
# ---------------------------------------------------------------------------

def recover_support_sdp(Y: np.ndarray, support_size: int) -> np.ndarray:
    """
    Recover node-level support via semidefinite programming (SDP).

    Solves the SDP relaxation on the elementwise-squared residual matrix
    and selects the `support_size` nodes with the largest row sums in the
    solution matrix.

    Requires MOSEK (install with ``pip install Mosek``).

    Args:
        Y: Residual matrix (n x n), typically Y_tilde from Algorithm 1.
        support_size: Expected number of perturbed nodes (m).

    Returns:
        Array of node indices identified as perturbed (length = support_size).
    """
    C = Y * Y
    Z = _solve_sdp(C, support_size)
    return np.argpartition(np.sum(Z, axis=0), support_size)[:support_size]


def _solve_sdp(C: np.ndarray, m: int) -> np.ndarray:
    """
    Solve the SDP:  min <C, Z>  s.t.  Z >= 0, tr(Z) = K, sum(Z) = K^2.

    where K = n - m.

    Args:
        C: n x n cost matrix (elementwise-squared residuals).
        m: Number of perturbed nodes.

    Returns:
        Z: Optimal n x n solution matrix.
    """
    Model, Domain, Expr, Matrix, ObjectiveSense = _import_mosek()

    n = C.shape[0]
    K = n - m

    with Model("node_support_sdp") as model:
        Z = model.variable("Z", [n, n], Domain.inPSDCone())
        model.constraint("trace", Expr.sum(Z.diag()), Domain.equalsTo(float(K)))
        model.constraint("sum", Expr.sum(Z), Domain.equalsTo(float(K ** 2)))
        model.objective(
            ObjectiveSense.Minimize,
            Expr.sum(Expr.mulElm(Z, Matrix.dense(C))),
        )
        model.solve()
        return np.array(Z.level()).reshape(n, n)


# ---------------------------------------------------------------------------
# Group Lasso-based support recovery
# ---------------------------------------------------------------------------

def recover_support_glasso(Y: np.ndarray, support_size: int) -> np.ndarray:
    """
    Recover node-level support via Group Lasso (ADMM).

    Uses a heuristic binary search over the regularization parameter to
    ensure exactly `support_size` nodes are selected.

    Args:
        Y: Residual matrix (n x n).
        support_size: Expected number of perturbed nodes (m).

    Returns:
        Array of node indices identified as perturbed (length = support_size).
    """
    n = Y.shape[0]
    m = support_size
    c0 = (
        np.partition(np.linalg.norm(Y, axis=1), -(m - 1))[-(m - 1)] - sqrt(n)
    ) / (n ** 0.25 * log(n) ** 0.25)

    for _ in range(30):
        lam = sqrt(n) + (c0 - 1) * n ** 0.25 * log(n) ** 0.25
        I_hat, V = _admm_group_lasso(Y, lam)
        if len(I_hat) < m:
            c0 -= 0.5
        else:
            return np.argpartition(-np.sum(V ** 2, axis=0), m)[:m]

    return I_hat


def _admm_group_lasso(
    Y: np.ndarray, lam: float, rho: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    ADMM solver for the Group Lasso node-sparse recovery problem.

    Args:
        Y: Observed residual matrix (n x n).
        lam: Regularization parameter.
        rho: ADMM penalty parameter.

    Returns:
        I_hat: Indices of nodes with nonzero group norms.
        V: Auxiliary variable encoding group-sparse structure.
    """
    homotopy_factor = 5
    rho_max = 5
    n_stages = int(np.floor(np.log(rho_max) / np.log(homotopy_factor) + 1))
    n = Y.shape[0]

    Theta = Y.copy()
    V = np.zeros((n, n))
    W = np.zeros((n, n))
    Delta1 = np.zeros((n, n))
    Delta2 = np.zeros((n, n))

    for _ in range(n_stages):
        for _ in range(60):
            Theta = 2 / (1 + 2 * rho) * (Y / 2 + Delta1 + rho * (V + W))

            A = (Delta1 + Delta2) / (2 * rho) + (W - W.T - Theta) / 2
            col_norms = np.sqrt(np.sum(A ** 2, axis=0))
            shrink = np.maximum(col_norms - lam / (2 * rho), 0)
            safe_norms = np.where(col_norms > 0, col_norms, 1.0)
            V = -A * (shrink / safe_norms)[np.newaxis, :]

            W = -0.5 * (V - V.T - Theta) - 0.5 * (Delta1 - Delta2.T) / rho
            Delta1 = Delta1 + rho * (V + W - Theta)
            Delta2 = Delta2 + rho * (V - W.T)

        rho *= homotopy_factor

    alphas = np.sqrt(np.sum(V ** 2, axis=0))
    I_hat = np.nonzero(alphas)[0]
    return I_hat, V
