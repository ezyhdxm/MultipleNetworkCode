import os
import numpy as np
from math import sqrt, log
import generator as g
import utils as u
import lowRank as lr
from scipy.linalg import sqrtm
import sys


def improved_eigvec(U, W):
    n, r = U.shape
    Uhat = np.zeros((n,r))
    U = np.real(U)
    W = np.real(W)
    for l in range(r):
        UWl = U[:,l].dot(W[:,l])
        Uhat[:,l] = np.sign(U[:,l]) * np.minimum(
            np.sqrt(
                np.abs(U[:,l]*W[:,l]/UWl)
                ), 
            1)
    
    return Uhat
    
def asymm_arrange(M1, M2):
    M = M2
    i_lower = np.tril_indices_from(M1, k=-1)
    M[i_lower] = M1[i_lower]
    return M

def get_Uhat_Mhat_util(eigvals_r, U, M1, M2):
    evs_rinv = np.diag(1/eigvals_r)
    evs_r = np.diag(eigvals_r)
    G = np.linalg.inv(evs_rinv @ U.T @ M1 @ M2 @ U @ evs_rinv)
    Gsymm = (G + G.T) / 2
    Psihat = sqrtm(Gsymm)
    Uhat = U @ Psihat
    Mhat = Uhat @ evs_r @ Uhat.T
    return Uhat, Mhat

def get_Uhat_Mhat(M11, M12, M21, M22, r):
    M = asymm_arrange(M11, M12)
    _, U, eigvals_r = lr.top_r_low_rank_asymmetric(M, r)
    Uhat1, Mhat1 = get_Uhat_Mhat_util(eigvals_r, U, M21, M22)
    M = asymm_arrange(M11, M12)
    _, U, eigvals_r = lr.top_r_low_rank_asymmetric(M, r)
    Uhat2, Mhat2 = get_Uhat_Mhat_util(eigvals_r, U, M21, M22)
    
    Uhat = (Uhat1 + Uhat2) / 2
    Mhat = (Mhat1 + Mhat2) / 2
    
    return Uhat, Mhat

def get_Uhat_Mhat_base(M11, M12, M21, M22, r):
    M1 = asymm_arrange(M11, M12)
    M2 = asymm_arrange(M21, M22)
    M = (M1 + M2) / 2
    del M1, M2
    
    W, U, eigvals_r = lr.top_r_low_rank_asymmetric(M, r)
    evs_r = np.diag(eigvals_r)
    Uhat = improved_eigvec(U, W)
    Mhat = Uhat @ evs_r @ Uhat.T
    return Uhat, Mhat

def exp6(
        n:int, delta_ratio: float, rate: str,
    ):
    
    results = {}
    
    r = 3
    mu = np.sqrt(n) * log(n)
    delta = log(n)
    lambdamin = 3*sqrt(n)
    evs = np.array([0, delta, delta*delta_ratio]) + lambdamin
        
    Mstar, Ustar = g.generate_low_rank_coherent_signal(n, r, eigenvalues=evs, mu=mu)
    M11 = Mstar + g.symmetric_gaussian_noise_homo(n)
    M12 = Mstar + g.symmetric_gaussian_noise_homo(n)
    M21 = Mstar + g.symmetric_gaussian_noise_homo(n)
    M22 = Mstar + g.symmetric_gaussian_noise_homo(n)
    
    Uhat, Mhat = get_Uhat_Mhat(M11, M12, M21, M22, r)
    
    # Gamma, _, Rt = np.linalg.svd(U.T @ Ustar, full_matrices=False)
    # O = Gamma @ Rt
    
    results["n"] = n
    results["mu"] = mu
    results["lambdamin"] = lambdamin
    results["delta"] = delta
    results["delta_ratio"] = delta_ratio
    results["rate"] = rate
    results["r"] = r
    
    results["Uhat_error"] = u.tti_approx(Uhat, Ustar)
    results["Mhat_error"] = u.max_abs(Mhat - Mstar)
    
    
    
    return results

if __name__ == "__main__":
    n = int(sys.argv[1])
    delta_ratio = float(sys.argv[2])
    rate = str(sys.argv[3])
    result = exp6(n, delta_ratio, rate)
    print(result)