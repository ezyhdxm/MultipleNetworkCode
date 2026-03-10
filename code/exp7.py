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
    

def exp7(n: int, const:float):
    
    scale = const * (log(n)/n)**(1/4)
    m = int(2*log(n))
    
    Bstar, Istar = g.generate_node_sparse_signal_general(n, m, scale)
    Ytil = Bstar + g.symmetric_gaussian_noise_heter(n)
    
    Zhat = sdp.solve_sdp_mosek(Ytil*Ytil, m)
    Ihat_sdp = np.argpartition(np.sum(Zhat, axis=0), m)[:m]
    
    Ihat_gl = gl.glasso(Ytil, m)
    
    results = {
        "n": n,
        "m": m,
        "scale": scale,
        "const": const,
        "sdp_error":  np.sum(~np.isin(Ihat_sdp, Istar)),
        "gl_error":  np.sum(~np.isin(Ihat_gl, Istar))
    }
    
    return results



if __name__ == "__main__":
    n = int(sys.argv[1])
    const = float(sys.argv[2])
    result = exp7(n, const)
    print(result)