import os
import numpy as np
from math import sqrt, log
import generator as g
import utils as u
import lowRank as lr
from scipy.linalg import sqrtm
import sys


    
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
    Uhat, Mhat = get_Uhat_Mhat_util(eigvals_r, U, M21, M22)
    
    return Uhat, Mhat

def exp10(
        n:int, mu:float, mu_rate:str
    ):
    
    results = {}
    
    r = 3
    delta = 2*log(n)
    lambdamin = 2.1 * sqrt(n)
    evs = delta * np.arange(r) + lambdamin
        
    Mstar, Ustar = g.generate_low_rank_coherent_signal(n, r, eigenvalues=evs, mu=mu)
    M11 = Mstar + g.symmetric_gaussian_noise_homo(n)
    M12 = Mstar + g.symmetric_gaussian_noise_homo(n)
    M21 = Mstar + g.symmetric_gaussian_noise_homo(n)
    M22 = Mstar + g.symmetric_gaussian_noise_homo(n)
    
    Uhat, Mhat = get_Uhat_Mhat(M11, M12, M21, M22, r)
    
    # Gamma, _, Rt = np.linalg.svd(U.T @ Ustar, full_matrices=False)
    # O = Gamma @ Rt
    
    results["Uhat_error"] = u.tti_approx(Uhat, Ustar)
    results["Mhat_error"] = u.max_abs(Mhat - Mstar)

    Mhat, Uhat, _ = lr.top_r_low_rank_symmetric((M11+M12+M21+M22)/4, r)
    results["Uspec_error"] = u.tti_approx(Uhat, Ustar)
    results["Mspec_error"] = u.max_abs(Mhat - Mstar)
    
    results["n"] = n
    results["mu"] = mu
    results["mu_rate"] = mu_rate
    
    return results

if __name__ == "__main__":
    n = int(sys.argv[1])
    mu = float(sys.argv[2])
    mu_rate = str(sys.argv[3])
    result = exp10(n, mu, mu_rate)
    print(result)