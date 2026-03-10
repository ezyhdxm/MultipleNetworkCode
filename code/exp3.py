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

def exp3(n: int):
    
    results = {}
    
    m = int(2*log(n))
    
    Bstar, Istar = g.generate_node_sparse_signal_general(n,m)
    Y0 = Bstar + g.symmetric_gaussian_noise_row_heter(n)
    Y1 = Bstar + g.symmetric_gaussian_noise_row_heter(n)
    
    Zhat = sdp.solve_sdp_mosek(Y0*Y1, m)
    Ihat = np.argpartition(np.sum(Zhat, axis=0), m)[:m]

    results["multiple_error"] = np.sum(~np.isin(Ihat, Istar))
    
    Y = (Y0 + Y1) / 2
    Zhat = sdp.solve_sdp_mosek(Y*Y, m)
    Ihat = np.argpartition(np.sum(Zhat, axis=0), m)[:m]

    results["single_error"] = np.sum(~np.isin(Ihat, Istar))
    
    results["n"] = n
    results["m"] = m
    
    
    return results


if __name__ == "__main__":
    n = int(sys.argv[1])
    result = exp3(n)
    print(result)