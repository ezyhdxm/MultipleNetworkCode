import os
import numpy as np
from math import sqrt, log
import generator as g
import utils as u
import lowRank as lr
from scipy.linalg import sqrtm
import sys


def exp8(
        mu:float,
    ):
    
    results = {}
    n = 30000
    
    delta = 2*log(n)
    
    lambdamin = 3*sqrt(n)
    r = 3
    
    evs = delta * np.arange(r) + lambdamin
    Mstar, Ustar = g.generate_low_rank_coherent_signal(n, r, eigenvalues=evs, mu=mu)
    results["n"] = n
    
    M = Mstar + g.symmetric_gaussian_noise_homo(n)
    Mhat, Uhat, _ = lr.top_r_low_rank_symmetric(M, r)
    results["spec_error"] = u.max_abs(Mhat - Mstar)
    U, eigvals_r = lr.top_r_low_rank_asymmetric_right(Mstar + g.asymmetric_gaussian_noise_homo(n), r)
    U1 = (Mstar + g.symmetric_gaussian_noise_homo(n)) @ U
    U2 = (Mstar + g.symmetric_gaussian_noise_homo(n)) @ U
    evs_rinv = np.diag(1/eigvals_r)
    evs_r = np.diag(eigvals_r)
    G = np.linalg.inv(evs_rinv @ (U1.T @ U2) @ evs_rinv)
    Gsymm = (G + G.T) / 2
    Psihat = sqrtm(Gsymm)
    Uhat = U @ Psihat
    Mhat = Uhat @ evs_r @ Uhat.T
    results["hat_error"] = u.max_abs(Mhat - Mstar)
    results['mu'] = mu
    
    # Gamma, _, Rt = np.linalg.svd(U.T @ Ustar, full_matrices=False)
    # O = Gamma @ Rt
    
    return results

if __name__ == "__main__":
    mu = float(sys.argv[1])
    result = exp8(mu)
    print(result)
