import numpy as np
from math import sqrt, log
import generator as g
import utils as u
import lowRank as lr
import sdp
import glasso as gl
from scipy.linalg import sqrtm
import sys

def spectral_initializer(n: int, mu: float, r: int = 3):
    evs = 2*log(n)*np.arange(r)  
    evs += 3*sqrt(n)

    Mstar, _ = g.generate_low_rank_coherent_signal(n, r, eigenvalues=evs, mu=mu)
    W0 = g.symmetric_gaussian_noise_heter(n)
    M = Mstar + W0
    del W0
    
    return (Mstar + g.symmetric_gaussian_noise_heter(n) - lr.low_rank_entrywise(M, r))
    

def exp1(n: int, mu:float, rate:str, m:int=10):
    scale = 2*n**(-1/4)*log(n)**(1/4)
    
    Bstar, Istar = g.generate_node_sparse_signal_general(n, m, scale)
    Ytil = Bstar + spectral_initializer(n, mu=mu, r=3)
    
    Zhat = sdp.solve_sdp_mosek(Ytil*Ytil, m)
    Ihat_sdp = np.argpartition(np.sum(Zhat, axis=0), m)[:m]
    
    Ihat_gl = gl.glasso(Ytil, m)
    
    results = {
        "n": n,
        "rate": rate,
        "mu": mu,
        "method": method,
        "sdp_error":  np.sum(~np.isin(Ihat_sdp, Istar)),
        "gl_error":  np.sum(~np.isin(Ihat_gl, Istar))
    }
    
    return results



if __name__ == "__main__":
    n = int(sys.argv[1])
    mu = float(sys.argv[2])
    rate = str(sys.argv[3])
    result = exp1(n, mu, rate)
    print(result)