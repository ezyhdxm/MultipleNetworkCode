import os
os.environ["MOSEKLM_LICENSE_FILE"] = 'mosek.lic'

import numpy as np
from math import sqrt, log
import generator as g
import utils as u
import lowRank as lr
import sdp
import glasso as gl
from scipy.linalg import sqrtm
import sys

def exp4(n: int, df: int = 4):
    
    results = {}

    m = 2*log(n)
    
    Bstar, Istar = g.generate_node_sparse_signal_general(n,m)
    Ytil = Bstar + g.symmetric_t_noise_homo(n, df)
    
    tau = 1
    
    Zhat = sdp.solve_sdp_mosek(np.minimum(Ytil*Ytil, tau**2), m)
    Ihat = np.argpartition(np.sum(Zhat, axis=0), m)[:m]
    
    results["trunc_error"] = np.sum(~np.isin(Ihat, Istar))
    
    Zhat = sdp.solve_sdp_mosek(Ytil*Ytil, m)
    Ihat = np.argpartition(np.sum(Zhat, axis=0), m)[:m]
    
    results["sdp_error"] = np.sum(~np.isin(Ihat, Istar))
    
    results["n"] = n
    results["m"] = m
    
    return results

if __name__ == "__main__":
    n = int(sys.argv[1])
    result = exp4(n)
    print(result)