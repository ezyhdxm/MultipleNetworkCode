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

def exp2(n:int):
    m = int(2*sqrt(n))
    k = max(5, int(0.2*m))
    Bstar, Istar = g.generate_Bstar_failure(n,m,k)
    Ytil = Bstar + g.symmetric_gaussian_noise_homo(n)
    
    Zhat = sdp.solve_sdp_mosek(Ytil*Ytil, m)
    Ihat_sdp = np.argpartition(np.sum(Zhat, axis=0), m)[:m]
    
    Ihat_gl = gl.glasso(Ytil, m)
    
    results = {
        "n": n,
        "m": m,
        "method": method,
        "sdp_error":  np.sum(~np.isin(Ihat_sdp, Istar)),
        "gl_error":  np.sum(~np.isin(Ihat_gl, Istar)),
    }
    
    return results


if __name__ == "__main__":
    n = int(sys.argv[1])
    result = exp2(n)
    print(result)