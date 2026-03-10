import cvxpy as cp
import numpy as np
import mosek

from mosek.fusion import Model, Domain, Expr, Matrix, ObjectiveSense

#import torchgit 
#import torch.nn as nn
#import torch.optim as optim
from scipy.sparse.linalg import eigsh

#from pymanopt.optimizers import TrustRegions, AugmentedLagrangian

# Warning: interior point method is very slow!
'''
def solve_sdp_naive(C, m):
    """
    Solves the SDP:
        minimize    <C, Z>
        subject to  Z >= 0 (elementwise)
                    Z >> 0 (semidefinite)
                    diag(Z) <= 1
                    <I, Z> = K
                    <J, Z> = K^2

    Parameters:
        C (ndarray): n x n cost matrix
        K (int): trace constraint and sum constraint, K = n - m

    Returns:
        Z_opt (ndarray): the optimal solution matrix
    """
    n = C.shape[0]
    K = n - m
    Z = cp.Variable((n, n), PSD=True)

    constraints = [
        cp.diag(Z) <= 1,
        #Z >= 0,
        cp.trace(Z) == K,
        cp.sum(Z) == K**2
    ]

    objective = cp.Minimize(cp.trace(C @ Z))
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, eps=0.1)  # or try 'MOSEK'/'CVXOPT' if installed

    return Z.value

'''


def solve_sdp_mosek(C, m) -> np.ndarray:
    """
    Solve the SDP
        minimize   <C, Z>
        subject to Z >= 0, trace(Z) = K, sum(Z) = K^2
    using MOSEK Fusion.

    Args:
        C: (n x n) cost matrix
        K: scalar

    Returns:
        Z_opt: the optimal (n x n) numpy array
    """
    n = C.shape[0]
    K = n - m
    with Model("SDP") as M:
        # decision variable Z ∈ S^n, PSD
        Z = M.variable("Z", [n, n], Domain.inPSDCone())

        # trace(Z) = K
        M.constraint("trace_eq",
                     Expr.sum(Z.diag()),
                     Domain.equalsTo(K))    

        # sum of all entries = K^2
        M.constraint("sum_eq", Expr.sum(Z), Domain.equalsTo(K**2))

        # objective: minimize <C, Z> = sum_{i,j} C_{ij} Z_{ij}
        M.objective(
            ObjectiveSense.Minimize,
            Expr.sum(Expr.mulElm(Z, Matrix.dense(C)))
        )

        M.solve()
        # Z.level() returns a flat list; reshape to (n,n)
        return np.array(Z.level()).reshape(n, n)

'''
def burer_monteiro_sdp_naive(C, m, r=10, lr=1e-2, max_iter=50000, verbose=False, tol_grad=1e-5):
    """
    Approximate SDP solver via Burer–Monteiro factorization.

    Args:
        C (np.ndarray): Cost matrix of shape (n, n)
        K (int): Target trace and sum constraint
        r (int): Rank of the factorization (r << n for speed)
        lr (float): Learning rate
        max_iter (int): Maximum number of optimization steps
        verbose (bool): Print loss and constraint values

    Returns:
        Z (np.ndarray): Approximate solution matrix Y Y^T
    """
    n = C.shape[0]
    K = n - m
    C_torch = torch.tensor(C, dtype=torch.float32)
    Y = nn.Parameter(torch.randn(n, r))  # optimization variable

    optimizer = optim.Adam([Y], lr=lr)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.5, patience=1000,
                                                           verbose=verbose, min_lr=1e-4)

    for it in range(max_iter):
        optimizer.zero_grad()

        # Objective: tr(Y^T C Y)
        obj = torch.trace(Y.T @ C_torch @ Y)

        # Constraint penalties
        # diag_penalty = torch.relu(Y.pow(2).sum(dim=1) - 1).sum()

        trace = Y.pow(2).sum()
        trace_penalty = (trace - K)**2

        sum_inner = (Y @ Y.T).sum()
        sum_penalty = (sum_inner - K**2)**2

        total_loss = obj + K * trace_penalty + 1 * sum_penalty

        total_loss.backward()
        
        grad_norm = Y.grad.norm().item()
        if grad_norm < tol_grad:
            if verbose:
                print(f"Terminated at iter {it} due to small gradient norm: {grad_norm:.2e}")
            break
        optimizer.step()
        scheduler.step(total_loss)

        if verbose and it % 500 == 0:
            print(f"Iter {it}: Loss = {total_loss.item():.4f}, Obj = {obj.item():.4f}")

    # Return the outer product
    with torch.no_grad():
        Z_approx = Y @ Y.T
        return Z_approx.numpy()
    '''



