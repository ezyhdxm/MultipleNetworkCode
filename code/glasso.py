import numpy as np
from math import sqrt, log
import utils as u

#############################
# Group Lasso and Utilities #
#############################

def T2(Y, lamda):
    w = np.sqrt(np.sum(Y**2, axis=0)) - lamda
    w[w<0] = 0
    W = np.diag(w / np.sqrt(np.sum(Y**2, axis=0)))
    return Y@W

def theta_step(Y, V, W, Delta1, rho):
    return 2 / (1 + 2 * rho) * (Y / 2 + Delta1 + rho * (V + W))

def delta1_step(Delta1, V, W, Theta, rho):
    return Delta1 + rho * (V + W - Theta)

def delta2_step(Delta2, V, W, rho):
    return Delta2 + rho * (V - W.T)

def v_step(Theta, Delta1, Delta2, W, rho, lambda_val):
    A = (Delta1 + Delta2) / (2 * rho) + (W - W.T - Theta) / 2
    return T2(-A, lambda_val / (2 * rho))

def w_step(Theta, Delta1, Delta2, V, rho):
    return -0.5 * (V - V.T - Theta) - 0.5 * (Delta1 - Delta2.T) / rho

def ADMM(Y, lamda, rho=1):
    homotopy = 5
    rho_max = 5
    homotopy_size = int(np.floor(np.log(rho_max) / np.log(homotopy) + 1))
    n = Y.shape[0]
    Theta = Y
    V = np.zeros((n, n))
    W = np.zeros((n, n))
    Delta1 = np.zeros((n, n))
    Delta2 = np.zeros((n, n))

    for r in range(homotopy_size):
        for i in range(60):
            Theta = theta_step(Y, V, W, Delta1, rho)
            V = v_step(Theta, Delta1, Delta2, W, rho, lamda)
            W = w_step(Theta, Delta1, Delta2, V, rho)
            Delta1 = delta1_step(Delta1, V, W, Theta, rho)
            Delta2 = delta2_step(Delta2, V, W, rho)
        rho *= homotopy

    alphas = np.sqrt(np.sum(V**2, axis=0))
    Ihat = np.nonzero(alphas)[0]

    return Ihat, V
    # return {"Theta": Theta, "V": V, "W": W, "Delta1": Delta1, "Delta2": Delta2, "alphas":alphas, "Ihat":Ihat}


def glasso(Y, m, rho=1, max_iter=30):
    n = Y.shape[0]
    c0 = (np.partition(np.linalg.norm(Y, axis=1), 
                       -(m-1))[-(m-1)] - sqrt(n))/((n**(0.25))*(log(n)**(0.25)))
    for i in range(max_iter):
        lamda = sqrt(n) + (c0-1)*((n**(0.25))*(log(n)**(0.25)))
        Ihat, V = ADMM(Y, lamda)
        if len(Ihat) < m:
            c0 -= 0.5
        else:
            Ihat = np.argpartition(-np.sum(V**2, axis=0), m)[:m]
            break 
    
    return Ihat